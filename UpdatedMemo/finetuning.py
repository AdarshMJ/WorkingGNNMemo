import torch
import numpy as np
import os
import logging
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Tuple
from datetime import datetime
import copy
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier
from scipy import stats
import pickle
import pandas as pd

# Import necessary modules from existing code
from model import NodeGCN, NodeGAT, NodeGraphConv
from dataloader import load_npz_dataset, get_heterophilic_datasets
from main import get_model, load_dataset, get_node_splits, setup_logging
from generalization_analysis import mask_nodes_and_edges, plot_generalization_results


def load_trained_model(model_type, data, log_dir, device):
    """
    Load a saved model from the specified directory.
    
    Args:
        model_type: Type of GNN model (gcn, gat, graphconv)
        data: PyG data object with graph information
        log_dir: Directory containing the saved model
        device: torch device
    """
    # Create a new model instance
    num_features = data.x.size(1)
    num_classes = data.y.max().item() + 1
    model = get_model(model_type, num_features, num_classes, 
                     hidden_dim=128, num_layers=3)
    
    # Check if log_dir is an absolute path or relative path
    if os.path.isabs(log_dir):
        model_path = os.path.join(log_dir, 'f_model.pt')
    else:
        model_path = log_dir + '/f_model.pt'  # Simple concatenation for relative paths
    
    # Load saved weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model


def load_memorization_scores(log_dir):
    """
    Load memorization scores from saved results.
    
    Args:
        log_dir: Directory containing the saved results
    
    Returns:
        Dictionary containing node memorization scores
    """
    # Try to load from pickle file first (if it exists)
    pickle_path = os.path.join(log_dir, 'node_scores.pkl')
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)
    
    # If pickle doesn't exist, try to reconstruct from CSV files
    node_scores = {}
    node_types = ['candidate', 'shared', 'independent', 'extra']
    
    for node_type in node_types:
        csv_path = os.path.join(log_dir, f'{node_type}_scores.csv')
        if os.path.exists(csv_path):
            # Read the CSV file
            df = pd.read_csv(csv_path)
            
            # Calculate statistics from the dataframe
            mem_scores = df['mem_score'].values
            f_confidences = df['conf_f'].values
            g_confidences = df['conf_g'].values
            
            avg_score = np.mean(mem_scores)
            nodes_above_threshold = sum(1 for score in mem_scores if score > 0.5)
            percentage_above_threshold = (nodes_above_threshold / len(mem_scores)) * 100 if mem_scores else 0
            
            # Store in the node_scores dictionary
            node_scores[node_type] = {
                'mem_scores': mem_scores,
                'f_confidences': f_confidences,
                'g_confidences': g_confidences,
                'avg_score': avg_score,
                'nodes_above_threshold': nodes_above_threshold,
                'percentage_above_threshold': percentage_above_threshold,
                'raw_data': df
            }
    
    return node_scores


def extract_node_embeddings(model, data, device):
    """
    Extract node embeddings from the trained model.
    
    Args:
        model: Trained GNN model
        data: PyG data object
        device: torch device
    
    Returns:
        Node embeddings tensor
    """
    model.eval()
    with torch.no_grad():
        _, node_embeddings = model(data.x.to(device), data.edge_index.to(device), return_node_emb=True)
    
    return node_embeddings


def train_classifier(embeddings, train_mask, labels, classifier_type='logistic', **kwargs):
    """
    Train a classifier on the node embeddings.
    
    Args:
        embeddings: Node embeddings tensor
        train_mask: Boolean mask for training nodes
        labels: Node labels
        classifier_type: Type of classifier to use ('logistic', 'ridge', 'mlp')
        **kwargs: Additional parameters for specific classifiers
    
    Returns:
        Trained classifier
    """
    # Get training data
    X_train = embeddings[train_mask].cpu().numpy()
    y_train = labels[train_mask].cpu().numpy()
    
    # Train classifier based on specified type
    if classifier_type == 'logistic':
        C = kwargs.get('C', 1.0)
        clf = LogisticRegression(max_iter=1000, C=C, multi_class='multinomial', solver='lbfgs')
        clf.fit(X_train, y_train)
    
    elif classifier_type == 'ridge':
        alpha = kwargs.get('alpha', 1e-7)  # Similar to lbd in lineargnn.py
        clf = Ridge(alpha=alpha, solver='auto')
        
        # One-hot encode labels for regression
        num_classes = int(labels.max().item()) + 1
        y_train_onehot = np.zeros((len(y_train), num_classes))
        for i, label in enumerate(y_train):
            y_train_onehot[i, int(label)] = 1
        
        clf.fit(X_train, y_train_onehot)
    
    elif classifier_type == 'mlp':
        hidden_layer_sizes = kwargs.get('hidden_layer_sizes', (64, 32))
        max_iter = kwargs.get('max_iter', 500)
        alpha = kwargs.get('alpha', 0.0001)  # L2 regularization parameter
        
        clf = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes, 
            activation='relu', 
            solver='adam',
            alpha=alpha,
            max_iter=max_iter,
            random_state=42
        )
        clf.fit(X_train, y_train)
    
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    return clf


def evaluate_classifier(clf, embeddings, mask, labels, classifier_type='logistic'):
    """
    Evaluate the classifier on the given mask.
    
    Args:
        clf: Trained classifier
        embeddings: Node embeddings tensor
        mask: Boolean mask for evaluation nodes
        labels: Node labels
        classifier_type: Type of classifier used ('logistic', 'ridge', 'mlp')
    
    Returns:
        Classification accuracy
    """
    X = embeddings[mask].cpu().numpy()
    y_true = labels[mask].cpu().numpy()
    
    # Calculate accuracy based on classifier type
    if classifier_type == 'logistic' or classifier_type == 'mlp':
        y_pred = clf.predict(X)
        accuracy = np.mean(y_pred == y_true)
    
    elif classifier_type == 'ridge':
        y_pred = clf.predict(X)
        # For ridge regression, take the argmax of predictions
        y_pred_classes = np.argmax(y_pred, axis=1)
        accuracy = np.mean(y_pred_classes == y_true)
    
    return accuracy


def analyze_linear_classifier_generalization(
    model,
    data, 
    node_scores,
    drop_percentages=[0.1, 0.2, 0.5, 1.0],
    seeds=[42, 123, 456],
    classifier_type='logistic',
    classifier_kwargs={},
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    logger=None
):
    """
    Analyze how the test accuracy of a classifier changes when dropping 
    different types of nodes (memorized vs non-memorized).
    
    Args:
        model: Trained GNN model to use as feature encoder
        data: PyG data object
        node_scores: Dictionary containing memorization scores
        drop_percentages: Percentages of nodes to drop
        seeds: Random seeds for multiple runs
        classifier_type: Type of classifier to use ('logistic', 'ridge', 'mlp')
        classifier_kwargs: Additional parameters for the classifier
        device: torch device
        logger: Logger for printing status updates
    
    Returns:
        Results dictionary
    """
    # Extract embeddings from the model
    node_embeddings = extract_node_embeddings(model, data, device)
    
    # Get memorized and non-memorized nodes
    candidate_data = node_scores['candidate']['raw_data']
    shared_data = node_scores['shared']['raw_data']
    
    # Sort memorized nodes by score in decreasing order
    memorized_candidates = candidate_data[candidate_data['mem_score'] > 0.5].sort_values('mem_score', ascending=False)['node_idx'].values
    # Get non-memorized shared nodes (no ranking needed)
    non_memorized_shared = shared_data[shared_data['mem_score'] < 0.5]['node_idx'].values
    
    # Calculate number of memorized nodes - our maximum drop size
    num_memorized = len(memorized_candidates)
    if logger:
        logger.info(f"Found {num_memorized} memorized nodes")
        logger.info(f"Using classifier type: {classifier_type}")
    
    # Store results
    results = {
        'drop_memorized_ranked': {p: {'test_accs': [], 'masking_stats': []} for p in drop_percentages},
        'drop_non_memorized_random': {p: {'test_accs': [], 'masking_stats': []} for p in drop_percentages}
    }
    
    # Run experiments with different seeds
    for seed in seeds:
        if logger:
            logger.info(f"\nRunning classifier analysis with seed {seed}")
        
        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Train baseline classifier on all embeddings
        baseline_clf = train_classifier(
            node_embeddings, data.train_mask, data.y,
            classifier_type=classifier_type, **classifier_kwargs
        )
        baseline_test_acc = evaluate_classifier(
            baseline_clf, node_embeddings, data.test_mask, data.y,
            classifier_type=classifier_type
        )
        
        if logger:
            logger.info(f"Baseline test accuracy: {baseline_test_acc:.4f}")
        
        # For each drop percentage
        for drop_pct in drop_percentages:
            num_nodes_to_drop = int(num_memorized * drop_pct)
            
            if logger:
                logger.info(f"\nDropping {num_nodes_to_drop} nodes ({drop_pct*100:.0f}% of memorized nodes)")
            
            # 1. Drop top-k memorized nodes
            nodes_to_drop = memorized_candidates[:num_nodes_to_drop]
            masked_data, mem_masking_stats = mask_nodes_and_edges(data, nodes_to_drop, device)
            drop_memorized_train_mask = data.train_mask.clone()
            drop_memorized_train_mask[nodes_to_drop] = False
            
            # Extract embeddings with masked nodes
            masked_embeddings = extract_node_embeddings(model, masked_data, device)
            
            # Train and evaluate classifier
            memorized_clf = train_classifier(
                masked_embeddings, drop_memorized_train_mask, data.y,
                classifier_type=classifier_type, **classifier_kwargs
            )
            memorized_test_acc = evaluate_classifier(
                memorized_clf, masked_embeddings, data.test_mask, data.y,
                classifier_type=classifier_type
            )
            
            # Store results
            results['drop_memorized_ranked'][drop_pct]['test_accs'].append(memorized_test_acc)
            results['drop_memorized_ranked'][drop_pct]['masking_stats'].append(mem_masking_stats)
            
            if logger:
                logger.info(f"Memorized drop test accuracy: {memorized_test_acc:.4f}")
            
            # 2. Drop random non-memorized shared nodes
            random_nodes_to_drop = np.random.choice(non_memorized_shared, num_nodes_to_drop, replace=False)
            masked_data, non_mem_masking_stats = mask_nodes_and_edges(data, random_nodes_to_drop, device)
            drop_non_memorized_train_mask = data.train_mask.clone()
            drop_non_memorized_train_mask[random_nodes_to_drop] = False
            
            # Extract embeddings with masked nodes
            masked_embeddings = extract_node_embeddings(model, masked_data, device)
            
            # Train and evaluate classifier
            non_memorized_clf = train_classifier(
                masked_embeddings, drop_non_memorized_train_mask, data.y,
                classifier_type=classifier_type, **classifier_kwargs
            )
            non_memorized_test_acc = evaluate_classifier(
                non_memorized_clf, masked_embeddings, data.test_mask, data.y,
                classifier_type=classifier_type
            )
            
            # Store results
            results['drop_non_memorized_random'][drop_pct]['test_accs'].append(non_memorized_test_acc)
            results['drop_non_memorized_random'][drop_pct]['masking_stats'].append(non_mem_masking_stats)
            
            if logger:
                logger.info(f"Non-memorized drop test accuracy: {non_memorized_test_acc:.4f}")
    
    # Add baseline results
    results['baseline_test_acc'] = baseline_test_acc
    
    return results, num_memorized


def plot_classifier_results(results, save_path, num_memorized_nodes, classifier_type, title_suffix=""):
    """
    Create line plot comparing test accuracies under different node dropping scenarios.
    
    Args:
        results: Results dictionary from analyze_linear_classifier_generalization
        save_path: Path to save the plot
        num_memorized_nodes: Number of memorized nodes (for x-axis scaling)
        classifier_type: Type of classifier used ('logistic', 'ridge', 'mlp')
        title_suffix: Optional text to add to plot title
    """
    plt.figure(figsize=(10, 6))
    
    # Convert percentages to actual number of nodes
    drop_percentages = list(results['drop_memorized_ranked'].keys())
    nodes_dropped = [0] + [int(pct * num_memorized_nodes) for pct in drop_percentages]  # Add 0 nodes dropped
    
    # Calculate means and confidence intervals
    def get_stats(data):
        mean = np.mean(data)
        sem = np.std(data) / np.sqrt(len(data))
        ci = 1.96 * sem  # 95% confidence interval
        return mean, ci

    # Plot results for each dropping strategy
    for strategy, color, label in [
        ('drop_memorized_ranked', 'red', 'Top-k Memorized'),
        ('drop_non_memorized_random', 'blue', 'Random Non-memorized')
    ]:
        means = [results['baseline_test_acc']]  # Start with baseline accuracy
        cis = [0]  # No confidence interval for baseline
        
        # Add results for each dropping percentage
        for pct in drop_percentages:
            test_accs = results[strategy][pct]['test_accs']
            mean, ci = get_stats(test_accs)
            means.append(mean)
            cis.append(ci)
        
        means = np.array(means)
        cis = np.array(cis)
        
        # Plot line with error bars
        plt.errorbar(nodes_dropped, means, yerr=cis, color=color, label=label, 
                    marker='o', capsize=5, capthick=1, markersize=6, 
                    linewidth=2, elinewidth=1)
    
    # Set y-axis limits to focus on the relevant range
    all_means = []
    all_cis = []
    for strategy in ['drop_memorized_ranked', 'drop_non_memorized_random']:
        # Include baseline accuracy
        all_means.append(results['baseline_test_acc'])
        all_cis.append(0)
        # Add results for each dropping percentage
        for pct in drop_percentages:
            test_accs = results[strategy][pct]['test_accs']
            mean, ci = get_stats(test_accs)
            all_means.append(mean)
            all_cis.append(ci)
    
    y_min = min(np.array(all_means) - np.array(all_cis)) - 0.01
    y_max = max(np.array(all_means) + np.array(all_cis)) + 0.01
    plt.ylim(y_min, y_max)
    
    plt.xlabel('Number of Nodes Dropped')
    plt.ylabel(f'{classifier_type.capitalize()} Test Accuracy')
    plt.title(f'Impact of Node Dropping on {classifier_type.capitalize()} Classifier Performance')
    
    if title_suffix:
        plt.suptitle(title_suffix, fontsize=12)
        plt.subplots_adjust(top=0.85)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def analyze_classifier_vs_full_model(
    data, 
    node_scores,
    model_f,
    log_dir,
    timestamp,
    classifier_type='logistic',
    classifier_kwargs={},
    seeds=[42, 123, 456],
    device=None,
    logger=None
):
    """
    Compare the impact of memorization on a classifier vs. the full GNN model.
    
    Args:
        data: PyG data object
        node_scores: Dictionary containing memorization scores
        model_f: Trained GNN model to use as feature encoder
        log_dir: Directory to save results
        timestamp: Timestamp for file names
        classifier_type: Type of classifier to use ('logistic', 'ridge', 'mlp')
        classifier_kwargs: Additional parameters for the classifier
        seeds: Random seeds for multiple runs
        device: torch device
        logger: Logger for printing status updates
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Run classifier analysis
    results, num_memorized = analyze_linear_classifier_generalization(
        model=model_f,
        data=data,
        node_scores=node_scores,
        seeds=seeds,
        classifier_type=classifier_type,
        classifier_kwargs=classifier_kwargs,
        device=device,
        logger=logger
    )
    
    # Create plot with classifier type in filename
    plot_path = os.path.join(log_dir, f'{classifier_type}_classifier_analysis_{timestamp}.png')
    plot_classifier_results(
        results,
        save_path=plot_path,
        num_memorized_nodes=num_memorized,
        classifier_type=classifier_type
    )
    
    if logger:
        logger.info(f"\n{classifier_type.capitalize()} classifier analysis plot saved to: {plot_path}")
    
    # Calculate and log average impact
    baseline_mean = results['baseline_test_acc']
    
    # Get results for different node dropping strategies
    drop_pcts = list(results['drop_memorized_ranked'].keys())
    drop_mem_ranked_mean = np.mean([np.mean(results['drop_memorized_ranked'][p]['test_accs']) for p in drop_pcts])
    drop_mem_ranked_std = np.mean([np.std(results['drop_memorized_ranked'][p]['test_accs']) for p in drop_pcts])
    
    drop_non_mem_mean = np.mean([np.mean(results['drop_non_memorized_random'][p]['test_accs']) for p in drop_pcts])
    drop_non_mem_std = np.mean([np.std(results['drop_non_memorized_random'][p]['test_accs']) for p in drop_pcts])
    
    if logger:
        logger.info(f"\n{classifier_type.capitalize()} Classifier Analysis Summary:")
        logger.info(f"Baseline Test Accuracy: {baseline_mean:.4f}")
        logger.info(f"Test Acc. after dropping ranked memorized nodes: {drop_mem_ranked_mean:.4f} ± {drop_mem_ranked_std:.4f}")
        logger.info(f"Test Acc. after dropping random non-memorized nodes: {drop_non_mem_mean:.4f} ± {drop_non_mem_std:.4f}")
        logger.info(f"Impact of dropping ranked memorized nodes: {(drop_mem_ranked_mean - baseline_mean):.4f}")
        logger.info(f"Impact of dropping random non-memorized nodes: {(drop_non_mem_mean - baseline_mean):.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    
    # Get all available heterophilic datasets
    heterophilic_datasets = get_heterophilic_datasets()
    # Combine all dataset choices
    all_datasets = ['Cora', 'Citeseer', 'Pubmed', 'Computers', 'Photo', 'Actor', 
                   'Chameleon', 'Squirrel', 'Cornell', 'Wisconsin', 'Texas',
                   'Roman-empire', 'Amazon-ratings'] + heterophilic_datasets
                   
    parser.add_argument('--dataset', type=str, required=True,
                       choices=all_datasets,
                       help='Dataset to use for analysis')
    parser.add_argument('--model_type', type=str, default='gcn',
                       choices=['gcn', 'gat', 'graphconv'],
                       help='Type of GNN model to use')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory with pretrained model and memorization scores')
    parser.add_argument('--classifier_type', type=str, default='logistic',
                        choices=['logistic', 'ridge', 'mlp'],
                        help='Type of classifier to use')
    parser.add_argument('--alpha', type=float, default=1e-7,
                        help='Regularization parameter for ridge regression (similar to lbd in lineargnn.py)')
    parser.add_argument('--hidden_layers', type=str, default='64,32',
                        help='Hidden layer sizes for MLP, comma-separated')
    
    args = parser.parse_args()
    
    # Setup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger = logging.getLogger('finetuning')
    logger.setLevel(logging.INFO)
    
    # Setup console handler if not already set
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(console_handler)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load dataset
    dataset = load_dataset(args)
    data = dataset[0].to(device)
    
    # Get node splits (needed to recreate the same setup as in the main analysis)
    shared_idx, candidate_idx, independent_idx = get_node_splits(
        data, data.train_mask, swap_candidate_independent=False
    )
    
    # Get extra indices from test set
    test_indices = torch.where(data.test_mask)[0]
    extra_size = len(candidate_idx)
    extra_indices = test_indices[:extra_size].tolist()  # Take first extra_size test indices

    # Create nodes_dict
    nodes_dict = {
        'shared': shared_idx,
        'candidate': candidate_idx,
        'independent': independent_idx,
        'extra': extra_indices,
        'val': torch.where(data.val_mask)[0].tolist(),
        'test': torch.where(data.test_mask)[0].tolist()
    }
    
    # Load pretrained model
    logger.info(f"Loading model from {args.results_dir}")
    model_f = load_trained_model(args.model_type, data, args.results_dir, device)
    
    # Load saved memorization scores instead of recalculating them
    logger.info("Loading memorization scores...")
    node_scores = load_memorization_scores(args.results_dir)
    
    if not node_scores:
        logger.error("Failed to load memorization scores. Please make sure they were saved during the initial run.")
        logger.error("Falling back to recalculating them (this might give inconsistent results).")
        # Import only if needed as a fallback
        from memorization import calculate_node_memorization_score
        
        # Check if there's also a g_model.pt in the results directory
        g_model_path = os.path.join(args.results_dir, 'g_model.pt')
        if not os.path.exists(g_model_path):
            logger.error(f"g_model.pt not found in {args.results_dir}. Both f_model.pt and g_model.pt are required.")
            return
        
        # Load model_g
        model_g = get_model(args.model_type, data.x.size(1), data.y.max().item() + 1, 
                          hidden_dim=32, num_layers=3)
        model_g.load_state_dict(torch.load(g_model_path, map_location=device))
        model_g = model_g.to(device)
        model_g.eval()
        
        logger.info("Calculating memorization scores...")
        node_scores = calculate_node_memorization_score(
            model_f=model_f,
            model_g=model_g,
            data=data,
            nodes_dict=nodes_dict,
            device=device,
            logger=logger
        )
    
    # Prepare classifier-specific parameters
    classifier_kwargs = {}
    if args.classifier_type == 'ridge':
        classifier_kwargs = {'alpha': args.alpha}
    elif args.classifier_type == 'mlp':
        hidden_sizes = tuple(int(size) for size in args.hidden_layers.split(','))
        classifier_kwargs = {'hidden_layer_sizes': hidden_sizes, 'alpha': args.alpha}
    
    # Run classifier analysis
    logger.info(f"\nPerforming {args.classifier_type} classifier analysis...")
    analyze_classifier_vs_full_model(
        data=data,
        node_scores=node_scores,
        model_f=model_f,
        log_dir=args.results_dir,
        timestamp=timestamp,
        classifier_type=args.classifier_type,
        classifier_kwargs=classifier_kwargs,
        device=device,
        logger=logger
    )


if __name__ == '__main__':
    main()