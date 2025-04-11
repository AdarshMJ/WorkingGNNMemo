import argparse
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, Amazon, Actor, WikipediaNetwork,WebKB,HeterophilousGraphDataset
from torch_geometric.transforms import RandomNodeSplit, LargestConnectedComponents
from torch_geometric.transforms import Compose
import os
import logging
from model import NodeGCN, NodeGAT, NodeGraphConv
import numpy as np
import random
from datetime import datetime
import sys
import pandas as pd
import pickle
from memorization import calculate_node_memorization_score, plot_node_memorization_analysis
from scipy import stats
from nli_analysis import *
from reliability_analysis import *
from nodeli import get_graph_and_labels_from_pyg_dataset, li_node, h_adj
import csv
from dimensionality_analysis import plot_memorization_dimensionality
from analysis import analyze_memorization_vs_misclassification
from feature_divergence_analysis import run_feature_divergence_analysis
from aggregation_alignment import run_aggregation_alignment_analysis
from weighted_label_heterophily import run_weighted_heterophily_analysis
#from feature_label_coherence import analyze_memorization_vs_flc, plot_memorization_flc_comparison, perform_statistical_tests as perform_flc_tests, analyze_flc_components, plot_component_contributions
from dataloader import load_npz_dataset, get_heterophilic_datasets
from chf_analysis import analyze_memorization_vs_cfh, perform_statistical_tests as perform_cfh_tests, analyze_correlation

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train(model, x, edge_index, train_mask, y, optimizer, device):
    model.train()
    optimizer.zero_grad()
    out = model(x.to(device), edge_index.to(device))
    loss = F.cross_entropy(out[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, x, edge_index, mask, y, device):
    model.eval()
    with torch.no_grad():
        out = model(x.to(device), edge_index.to(device))
        pred = out[mask].max(1)[1]
        correct = pred.eq(y[mask]).sum().item()
        total = mask.sum().item()
    return correct / total

def setup_logging(args):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create directory name with the structure ModelType_Datasetname_NumLayers_timestamp
    dir_name = f"{args.model_type}_{args.dataset}_{args.num_layers}_{timestamp}"
    
    # Create base results directory if it doesn't exist
    base_dir = 'results'
    os.makedirs(base_dir, exist_ok=True)
    
    # Create full directory path
    log_dir = os.path.join(base_dir, dir_name)
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup main logger
    logger = logging.getLogger('main')
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers
    
    # File handler
    log_file = os.path.join(log_dir, f'{args.model_type}_{args.dataset}_{timestamp}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)

    return logger, log_dir, timestamp

def load_dataset(args):
    transforms = Compose([
        LargestConnectedComponents(),
        RandomNodeSplit(split='train_rest', num_val=0.2, num_test=0.2)
    ])
    
    # Handle standard PyTorch Geometric datasets
    if args.dataset.lower() in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root='data', name=args.dataset, transform=transforms)
    elif args.dataset.lower() in ['computers', 'photo']:
        dataset = Amazon(root='data', name=args.dataset, transform=transforms)
    elif args.dataset.lower() == 'actor':
        dataset = Actor(root='data/Actor', transform=transforms)
    elif args.dataset.lower() in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(root=f'data/{args.dataset}', name=args.dataset, transform=transforms)
    elif args.dataset.lower() in ['cornell', 'wisconsin','texas']:
        dataset = WebKB(root=f'data/{args.dataset}', name=args.dataset, transform=transforms)
    elif args.dataset.lower() in ['roman-empire', 'amazon-ratings']:
        dataset = HeterophilousGraphDataset(root=f'data/{args.dataset}', name=args.dataset, transform=transforms)
    # Handle NPZ heterophilic datasets
    elif args.dataset.lower() in map(str.lower, get_heterophilic_datasets()):
        # Load the NPZ dataset and convert to a PyG dataset
        pyg_data = load_npz_dataset(args.dataset)
        
        # Create a dummy dataset-like object to maintain compatibility with the rest of the code
        class DummyDataset:
            def __init__(self, data):
                self.data = data
                self.num_classes = data.num_classes
                
            def __getitem__(self, idx):
                return self.data
                
            def __len__(self):
                return 1
                
        dataset = DummyDataset(pyg_data)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    dataset._data_list = None
    return dataset

def get_model(model_type, num_features, num_classes, hidden_dim, num_layers, gat_heads=4):
    """Create a new model instance based on specified type"""
    if model_type.lower() == 'gcn':
        return NodeGCN(num_features, num_classes, hidden_dim, num_layers)
    elif model_type.lower() == 'gat':
        return NodeGAT(num_features, num_classes, hidden_dim, num_layers, heads=gat_heads)
    elif model_type.lower() == 'graphconv':
        return NodeGraphConv(num_features, num_classes, hidden_dim, num_layers)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_node_splits(data, train_mask, swap_candidate_independent=False):
    """
    Create node splits without shuffling to preserve natural ordering.
    
    Args:
        data: PyG data object
        train_mask: Mask for train nodes
        swap_candidate_independent: If True, swap the roles of candidate and independent nodes
    """
    # Get train indices in their original order
    train_indices = torch.where(train_mask)[0]
    
    # Calculate sizes
    num_nodes = len(train_indices)
    shared_size = int(0.50 * num_nodes)
    remaining = num_nodes - shared_size
    split_size = remaining // 2
    
    # Split indices sequentially without shuffling
    shared_idx = train_indices[:shared_size].tolist()
    original_candidate_idx = train_indices[shared_size:shared_size + split_size].tolist()
    original_independent_idx = train_indices[shared_size + split_size:shared_size + split_size * 2].tolist()
    
    # Return swapped indices if requested
    if swap_candidate_independent:
        return shared_idx, original_independent_idx, original_candidate_idx
    else:
        return shared_idx, original_candidate_idx, original_independent_idx

def verify_no_data_leakage(shared_idx, candidate_idx, independent_idx, logger):
    """Verify there is no direct overlap between candidate and independent sets"""
    # Convert to sets for easy comparison
    candidate_set = set(candidate_idx)
    independent_set = set(independent_idx)
    
    # Check: No overlap between candidate and independent sets
    overlap = candidate_set.intersection(independent_set)
    if overlap:
        raise ValueError(f"Data leakage detected! Found {len(overlap)} nodes in both candidate and independent sets")
    
    logger.info("\nData Leakage Check:")
    logger.info(f"✓ No overlap between candidate and independent sets")

def train_models(args, data, shared_idx, candidate_idx, independent_idx, device, logger, output_dir, seeds):
    """Train model f and g on their respective node sets"""
    
    # Create train masks for model f and g
    train_mask_f = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    train_mask_f[shared_idx + candidate_idx] = True
    
    train_mask_g = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    train_mask_g[shared_idx + independent_idx] = True
    
    # Print training set information
    logger.info("\nTraining Set Information:")
    logger.info(f"Model f training nodes: {train_mask_f.sum().item()}")
    logger.info(f"- Shared nodes: {len(shared_idx)}")
    logger.info(f"- Candidate nodes: {len(candidate_idx)}")
    
    logger.info(f"\nModel g training nodes: {train_mask_g.sum().item()}")
    logger.info(f"- Shared nodes: {len(shared_idx)}")
    logger.info(f"- Independent nodes: {len(independent_idx)}")
    
    # Get number of classes
    num_classes = data.y.max().item() + 1
    
    # Lists to store models and their accuracies
    f_models = []
    g_models = []
    f_val_accs = []
    g_val_accs = []
    f_test_accs = []  # Add list for test accuracies
    g_test_accs = []  # Add list for test accuracies
    
    # Seeds for multiple training runs
    training_seeds = seeds
    
    logger.info("\nModel Architecture Details:")
    logger.info(f"Model Type: {args.model_type.upper()}")
    logger.info(f"Input Features: {data.x.size(1)}")
    logger.info(f"Hidden Dimensions: {args.hidden_dim}")
    logger.info(f"Number of Layers: {args.num_layers}")
    logger.info(f"LR: {args.lr}")
    logger.info(f"Weight Decay: {args.weight_decay}")

    if args.model_type == 'gat':
        logger.info(f"Number of Attention Heads: {args.gat_heads}")
    logger.info(f"Output Classes: {num_classes}")
    logger.info(f"Training with seeds: {training_seeds}")
    
    # Train multiple models with different seeds
    for seed in training_seeds:
        set_seed(seed)
        
        logger.info(f"\nTraining with seed {seed}")
        
        # Initialize models
        model_f = get_model(args.model_type, data.x.size(1), num_classes, 
                           args.hidden_dim, args.num_layers, args.gat_heads).to(device)
        model_g = get_model(args.model_type, data.x.size(1), num_classes, 
                           args.hidden_dim, args.num_layers, args.gat_heads).to(device)
        
        opt_f = torch.optim.Adam(model_f.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        opt_g = torch.optim.Adam(model_g.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        best_f_val_acc = 0
        best_g_val_acc = 0
        best_f_state = None
        best_g_state = None
        best_f_test_acc = 0
        best_g_test_acc = 0

        for epoch in range(args.epochs):
            # Train model f on shared+candidate nodes
            f_loss = train(model_f, data.x, data.edge_index, 
                         train_mask_f, data.y, opt_f, device)
            f_val_acc = test(model_f, data.x, data.edge_index, 
                           data.val_mask, data.y, device)
            current_f_test_acc = test(model_f, data.x, data.edge_index, 
                                    data.test_mask, data.y, device)
            
            # Train model g on shared+independent nodes
            g_loss = train(model_g, data.x, data.edge_index, 
                         train_mask_g, data.y, opt_g, device)
            g_val_acc = test(model_g, data.x, data.edge_index, 
                           data.val_mask, data.y, device)
            current_g_test_acc = test(model_g, data.x, data.edge_index, 
                                    data.test_mask, data.y, device)
            
            # Save best models based on validation accuracy
            if f_val_acc > best_f_val_acc:
                best_f_val_acc = f_val_acc
                best_f_test_acc = current_f_test_acc
                best_f_state = model_f.state_dict()
            
            if g_val_acc > best_g_val_acc:
                best_g_val_acc = g_val_acc
                best_g_test_acc = current_g_test_acc
                best_g_state = model_g.state_dict()
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                logger.info(f'Seed {seed}, Epoch {epoch+1}/{args.epochs}:')
                logger.info(f'Model f - Loss: {f_loss:.4f}, Val Acc: {f_val_acc:.4f}, Test Acc: {current_f_test_acc:.4f}')
                logger.info(f'Model g - Loss: {g_loss:.4f}, Val Acc: {g_val_acc:.4f}, Test Acc: {current_g_test_acc:.4f}')
        
        # Load best states
        model_f.load_state_dict(best_f_state)
        model_g.load_state_dict(best_g_state)
        
        # Store models and accuracies
        f_models.append(model_f.state_dict())
        g_models.append(model_g.state_dict())
        f_val_accs.append(best_f_val_acc)
        g_val_accs.append(best_g_val_acc)
        f_test_accs.append(best_f_test_acc)  # Store test accuracy
        g_test_accs.append(best_g_test_acc)  # Store test accuracy
        
        logger.info(f"\nSeed {seed} Results:")
        logger.info(f"Best Model f - Val Acc: {best_f_val_acc:.4f}, Test Acc: {best_f_test_acc:.4f}")
        logger.info(f"Best Model g - Val Acc: {best_g_val_acc:.4f}, Test Acc: {best_g_test_acc:.4f}")
    
    # Select models with best validation accuracy
    f_best_idx = np.argmax(f_val_accs)
    g_best_idx = np.argmax(g_val_accs)
    
    # Save best models
    save_dir = output_dir  # Use the provided output directory
    
    torch.save(f_models[f_best_idx], os.path.join(save_dir, 'f_model.pt'))
    torch.save(g_models[g_best_idx], os.path.join(save_dir, 'g_model.pt'))
    
    logger.info("\nSaved models with best validation accuracy:")
    logger.info(f"Model f - Val Acc: {f_val_accs[f_best_idx]:.4f}")
    logger.info(f"Model g - Val Acc: {g_val_accs[g_best_idx]:.4f}")
    
    # Load best states into models
    model_f.load_state_dict(f_models[f_best_idx])
    model_g.load_state_dict(g_models[g_best_idx])
    
    # Return best models along with validation accuracies and test accuracy lists
    return model_f, model_g, f_val_accs[f_best_idx], g_val_accs[g_best_idx], f_test_accs, g_test_accs

def perform_memorization_statistical_tests(node_scores, logger):
    """
    Perform statistical tests to check if memorization scores are statistically significant.
    
    Args:
        node_scores: Dictionary of memorization scores by node type
        logger: Logger to output results
    """
    logger.info("\n===== Statistical Significance Tests =====")
    
    # Check if all required node types exist
    required_types = ['candidate', 'shared', 'independent', 'extra']
    for node_type in required_types:
        if node_type not in node_scores:
            logger.info(f"Skipping some statistical tests: '{node_type}' nodes not found")
    
    # 1. Candidate vs other node types
    if 'candidate' in node_scores:
        candidate_scores = node_scores['candidate']['mem_scores']
        
        # Test against each other node type
        for other_type in ['shared', 'independent', 'extra']:
            if other_type not in node_scores:
                continue
                
            other_scores = node_scores[other_type]['mem_scores']
            
            # Run t-test
            t_stat, p_val = stats.ttest_ind(candidate_scores, other_scores, equal_var=False)
            
            # Calculate effect size (Cohen's d)
            mean_diff = np.mean(candidate_scores) - np.mean(other_scores)
            pooled_std = np.sqrt((np.std(candidate_scores)**2 + np.std(other_scores)**2) / 2)
            effect_size = abs(mean_diff) / pooled_std
            
            # Interpret effect size
            if effect_size < 0.2:
                effect_size_interp = "negligible"
            elif effect_size < 0.5:
                effect_size_interp = "small"
            elif effect_size < 0.8:
                effect_size_interp = "medium"
            else:
                effect_size_interp = "large"
                
            # Interpret p-value
            significant = p_val < 0.01
            
            # Log results
            logger.info(f"\nCandidate vs {other_type} nodes:")
            logger.info(f"  T-statistic: {t_stat:.4f}")
            logger.info(f"  P-value: {p_val:.6f} ({'significant' if significant else 'not significant'} at p<0.01)")
            logger.info(f"  Effect size (Cohen's d): {effect_size:.4f} ({effect_size_interp})")
            logger.info(f"  Mean difference: {mean_diff:.4f}")
    
    # 2. Shared vs Independent nodes
    if 'shared' in node_scores and 'independent' in node_scores:
        shared_scores = node_scores['shared']['mem_scores']
        independent_scores = node_scores['independent']['mem_scores']
        
        # Run t-test
        t_stat, p_val = stats.ttest_ind(shared_scores, independent_scores, equal_var=False)
        
        # Calculate effect size (Cohen's d)
        mean_diff = np.mean(shared_scores) - np.mean(independent_scores)
        pooled_std = np.sqrt((np.std(shared_scores)**2 + np.std(independent_scores)**2) / 2)
        effect_size = abs(mean_diff) / pooled_std
        
        # Interpret effect size
        if effect_size < 0.2:
            effect_size_interp = "negligible"
        elif effect_size < 0.5:
            effect_size_interp = "small"
        elif effect_size < 0.8:
            effect_size_interp = "medium"
        else:
            effect_size_interp = "large"
            
        # Interpret p-value
        significant = p_val < 0.01
        
        # Log results
        logger.info(f"\nShared vs Independent nodes:")
        logger.info(f"  T-statistic: {t_stat:.4f}")
        logger.info(f"  P-value: {p_val:.6f} ({'significant' if significant else 'not significant'} at p<0.01)")
        logger.info(f"  Effect size (Cohen's d): {effect_size:.4f} ({effect_size_interp})")
        logger.info(f"  Mean difference: {mean_diff:.4f}")
    
    # 3. Extra vs Independent nodes
    if 'extra' in node_scores and 'independent' in node_scores:
        extra_scores = node_scores['extra']['mem_scores']
        independent_scores = node_scores['independent']['mem_scores']
        
        # Run t-test
        t_stat, p_val = stats.ttest_ind(extra_scores, independent_scores, equal_var=False)
        
        # Calculate effect size (Cohen's d)
        mean_diff = np.mean(extra_scores) - np.mean(independent_scores)
        pooled_std = np.sqrt((np.std(extra_scores)**2 + np.std(independent_scores)**2) / 2)
        effect_size = abs(mean_diff) / pooled_std
        
        # Interpret effect size
        if effect_size < 0.2:
            effect_size_interp = "negligible"
        elif effect_size < 0.5:
            effect_size_interp = "small"
        elif effect_size < 0.8:
            effect_size_interp = "medium"
        else:
            effect_size_interp = "large"
            
        # Interpret p-value
        significant = p_val < 0.01
        
        # Log results
        logger.info(f"\nExtra vs Independent nodes:")
        logger.info(f"  T-statistic: {t_stat:.4f}")
        logger.info(f"  P-value: {p_val:.6f} ({'significant' if significant else 'not significant'} at p<0.01)")
        logger.info(f"  Effect size (Cohen's d): {effect_size:.4f} ({effect_size_interp})")
        logger.info(f"  Mean difference: {mean_diff:.4f}")

def calculate_graph_metrics(data):
    """Calculate adjusted homophily and node label informativeness for the graph"""
    # Get graph and labels in correct format
    graph, labels = get_graph_and_labels_from_pyg_dataset(data)
    
    # Calculate adjusted homophily
    adj_homophily = h_adj(graph, labels)
    
    # Calculate node label informativeness
    nli = li_node(graph, labels)
    
    return adj_homophily, nli

def log_results_to_csv(args, adj_homophily, nli, test_acc_mean, test_acc_std):
    """Log results to a CSV file"""
    csv_file = 'experiment_results.csv'
    file_exists = os.path.exists(csv_file)
    
    # Prepare the results row
    results = {
        'Dataset': args.dataset,
        'Model': args.model_type,
        'Test_Accuracy_Mean': f"{test_acc_mean:.4f}",
        'Test_Accuracy_Std': f"{test_acc_std:.4f}",
        'Learning_Rate': args.lr,
        'Weight_Decay': args.weight_decay,
        'Adjusted_Homophily': f"{adj_homophily:.4f}",
        'Node_Label_Informativeness': f"{nli:.4f}",
        'Num_Layers': args.num_layers
    }
    
    # Write to CSV
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)
    
    return csv_file

def save_memorization_scores(node_scores, log_dir, logger=None):
    """
    Save memorization scores to disk for later use.
    
    Args:
        node_scores: Dictionary containing node memorization scores
        log_dir: Directory to save the scores
        logger: Logger for status updates
    """
    # Save the entire node_scores dictionary as a pickle file
    pickle_path = os.path.join(log_dir, 'node_scores.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(node_scores, f)
    
    # Also save individual node types as CSV files for easier inspection
    for node_type, data in node_scores.items():
        if 'raw_data' in data:
            csv_path = os.path.join(log_dir, f'{node_type}_scores.csv')
            data['raw_data'].to_csv(csv_path, index=False)
    
    if logger:
        logger.info(f"Memorization scores saved to {log_dir}")
        logger.info(f"- Complete data: {os.path.basename(pickle_path)}")
        for node_type in node_scores:
            if 'raw_data' in node_scores[node_type]:
                logger.info(f"- {node_type} scores: {node_type}_scores.csv")

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
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--gat_heads', type=int, default=4,
                       help='Number of attention heads for GAT')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--swap_nodes', action='store_true', 
                       help='Swap candidate and independent nodes')
    parser.add_argument('--num_passes', type=int, default=1,
                       help='Number of forward passes to average for confidence scores')
    parser.add_argument('--k_hops', type=int, default=2,
                       help='Number of hops for local neighborhood in NLI calculation')
    parser.add_argument('--noise_level', type=float, default=1.0,
                      help='Standard deviation of Gaussian noise for reliability analysis (default: 0.1)')
    parser.add_argument('--perturb_ratio', type=float, default=0.05,
                      help='Ratio of labels to perturb for label-based reliability analysis (default: 0.05)')
    parser.add_argument('--label_perturb_epochs', type=int, default=1,
                      help='Number of epochs to train on perturbed labels (default: 1)')
    parser.add_argument('--drop_ratio', type=float, default=1.0,
                      help='Ratio of nodes to exclude in generalization analysis (default: 0.05)')
    
    args = parser.parse_args()
    
    # Setup
    logger, log_dir, timestamp = setup_logging(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Define training seeds here so they can be used throughout
    training_seeds = [42, 123, 456]
    
    # Load dataset
    dataset = load_dataset(args)
    data = dataset[0].to(device)
    
    # Calculate graph metrics
    adj_homophily, nli = calculate_graph_metrics(data)
    logger.info("\nGraph Metrics:")
    logger.info(f"Adjusted Homophily: {adj_homophily:.4f}")
    logger.info(f"Node Label Informativeness: {nli:.4f}")
    
    # Log dataset information
    logger.info(f"\nDataset Information:")
    logger.info(f"Dataset Name: {args.dataset}")
    logger.info(f"Number of Nodes: {data.num_nodes}")
    logger.info(f"Number of Edges: {data.edge_index.size(1)}")
    logger.info(f"Number of Features: {data.num_features}")
    logger.info(f"Number of Classes: {dataset.num_classes}")
    
    # Get node splits
    shared_idx, candidate_idx, independent_idx = get_node_splits(
        data, data.train_mask, swap_candidate_independent=args.swap_nodes
    )
    
    # Get extra indices from test set
    test_indices = torch.where(data.test_mask)[0]
    extra_size = len(candidate_idx)
    extra_indices = test_indices[:extra_size].tolist()  # Take first extra_size test indices

    logger.info("\nPartition Statistics:")
    if args.swap_nodes:
        logger.info("Note: Candidate and Independent nodes have been swapped!")
        logger.info("Original independent nodes are now being used as candidate nodes")
        logger.info("Original candidate nodes are now being used as independent nodes")
    logger.info(f"Total train nodes: {data.train_mask.sum().item()}")
    logger.info(f"Shared: {len(shared_idx)} nodes")
    logger.info(f"Candidate: {len(candidate_idx)} nodes")
    logger.info(f"Independent: {len(independent_idx)} nodes")
    logger.info(f"Extra test nodes: {len(extra_indices)} nodes")
    logger.info(f"Val set: {data.val_mask.sum().item()} nodes")
    logger.info(f"Test set: {data.test_mask.sum().item()} nodes")
    
    # Create nodes_dict
    nodes_dict = {
        'shared': shared_idx,
        'candidate': candidate_idx,
        'independent': independent_idx,
        'extra': extra_indices,
        'val': torch.where(data.val_mask)[0].tolist(),
        'test': torch.where(data.test_mask)[0].tolist()
    }
    
    # Verify no data leakage
    verify_no_data_leakage(shared_idx, candidate_idx, independent_idx, logger)
    
    # Initialize test accuracy tracking
    f_test_accs = []
    g_test_accs = []
    
    # Train models
    logger.info("\nTraining models...")
    model_f, model_g, f_val_acc, g_val_acc, f_test_accs, g_test_accs = train_models(
        args=args,
        data=data,
        shared_idx=shared_idx,
        candidate_idx=candidate_idx,
        independent_idx=independent_idx,
        device=device,
        logger=logger,
        output_dir=log_dir,
        seeds=training_seeds  # Pass seeds to train_models
    )
    
    # Get test accuracy statistics
    avg_f_test_acc = np.mean(f_test_accs)
    std_f_test_acc = np.std(f_test_accs)
    
    # Calculate memorization scores
    logger.info("\nCalculating memorization scores...")
    node_scores = calculate_node_memorization_score(
        model_f=model_f,
        model_g=model_g,
        data=data,
        nodes_dict=nodes_dict,
        device=device,
        logger=logger,
        num_passes=args.num_passes
    )
    
    # Save memorization scores to disk
    save_memorization_scores(node_scores, log_dir, logger)
    
    # Calculate and log average scores for each node type
    for node_type, scores_dict in node_scores.items():
        logger.info(f"Average memorization score for {node_type} nodes: {scores_dict['avg_score']:.4f}")
        # Also log the number of nodes above threshold
        logger.info(f"Nodes with score > 0.5: {scores_dict['nodes_above_threshold']}/{len(scores_dict['mem_scores'])} ({scores_dict['percentage_above_threshold']:.1f}%)")
    
    # Perform statistical tests on memorization scores
    perform_memorization_statistical_tests(node_scores, logger)
    
    # Create visualization
    plot_filename = f'{args.model_type}_{args.dataset}_{args.num_layers}_{timestamp}.png'
    plot_path = os.path.join(log_dir, plot_filename)
    
    plot_node_memorization_analysis(
        node_scores=node_scores,
        save_path=plot_path,
        title_suffix=f"Dataset: {args.dataset}, Model: {args.model_type}\nf_acc={f_val_acc:.3f}, g_acc={g_val_acc:.3f}",
        node_types_to_plot=['shared', 'candidate', 'independent', 'extra']
    )
    logger.info(f"Memorization score plot saved to: {plot_path}")

    ###Perform reliability analysis
    if logger:
        logger.info("\nPerforming reliability analysis...")
    reliability_results = analyze_reliability_vs_memorization(
        model_f, model_g, data, 
        node_scores,
        noise_level=args.noise_level,
        perturb_ratio=args.perturb_ratio,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.label_perturb_epochs,
        device=device
    )
    reliability_summary = plot_reliability_analysis(
        reliability_results,
        os.path.join(log_dir, f'reliability_analysis_{timestamp}.png')
    )

    # Log reliability analysis results
    if logger:
        logger.info("\nReliability Analysis Summary:")
        logger.info("\nFeature Perturbation Results:")
        for node_type in reliability_results:
            feat_data = reliability_results[node_type]['feature_perturbation']
            mem_scores = feat_data['memorized_reliability']
            non_mem_scores = feat_data['non_memorized_reliability']
            logger.info(f"\n{node_type.capitalize()} Nodes:")
            logger.info(f"  Memorized (n={len(mem_scores)}): mean={np.mean(mem_scores):.3f}")
            logger.info(f"  Non-memorized (n={len(non_mem_scores)}): mean={np.mean(non_mem_scores):.3f}")
            if feat_data['stat_test']['pvalue'] is not None:
                logger.info(f"  P-value: {feat_data['stat_test']['pvalue']:.2e}")

        logger.info("\nLabel Perturbation Results:")
        for node_type in reliability_results:
            label_data = reliability_results[node_type]['label_perturbation']
            mem_scores = label_data['memorized_reliability']
            non_mem_scores = label_data['non_memorized_reliability']
            logger.info(f"\n{node_type.capitalize()} Nodes:")
            logger.info(f"  Memorized (n={len(mem_scores)}): mean={np.mean(mem_scores):.3f}")
            logger.info(f"  Non-memorized (n={len(non_mem_scores)}): mean={np.mean(non_mem_scores):.3f}")
            if label_data['stat_test']['pvalue'] is not None:
                logger.info(f"  P-value: {label_data['stat_test']['pvalue']:.2e}")
    
    # Perform generalization analysis
    if logger:
        logger.info("\nPerforming generalization analysis...")
    from generalization_analysis import analyze_generalization_impact, plot_generalization_results
    
    gen_results = analyze_generalization_impact(
        data=data,
        node_scores=node_scores,
        model_type=args.model_type,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        gat_heads=args.gat_heads,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        drop_ratio=args.drop_ratio,
        seeds=training_seeds,  # Use same seeds as main training
        device=device,
        logger=logger
    )
    
    # Create visualization
    gen_plot_filename = f'generalization_analysis_{args.model_type}_{args.dataset}_{args.num_layers}_{timestamp}.png'
    gen_plot_path = os.path.join(log_dir, gen_plot_filename)
    
    # Get number of memorized nodes
    num_memorized = len(node_scores['candidate']['raw_data'][node_scores['candidate']['raw_data']['mem_score'] > 0.5])
    
    plot_generalization_results(
        gen_results,
        save_path=gen_plot_path,
        num_memorized_nodes=num_memorized,
    )
    logger.info(f"Generalization analysis plot saved to: {gen_plot_path}")
    
    # Calculate and log average impact
    baseline_mean = gen_results['baseline_test_acc']
    
    # Get results for ranked memorized nodes
    drop_pcts = list(gen_results['drop_memorized_ranked'].keys())
    drop_mem_ranked_mean = np.mean([np.mean(gen_results['drop_memorized_ranked'][p]['test_accs']) for p in drop_pcts])
    drop_mem_ranked_std = np.mean([np.std(gen_results['drop_memorized_ranked'][p]['test_accs']) for p in drop_pcts])
    
    # Get results for non-memorized nodes
    drop_non_mem_mean = np.mean([np.mean(gen_results['drop_non_memorized_random'][p]['test_accs']) for p in drop_pcts])
    drop_non_mem_std = np.mean([np.std(gen_results['drop_non_memorized_random'][p]['test_accs']) for p in drop_pcts])
    
    logger.info("\nGeneralization Analysis Summary:")
    logger.info(f"Baseline Test Accuracy: {baseline_mean:.4f}")
    logger.info(f"Test Acc. after dropping ranked memorized nodes: {drop_mem_ranked_mean:.4f} ± {drop_mem_ranked_std:.4f}")
    logger.info(f"Test Acc. after dropping random non-memorized nodes: {drop_non_mem_mean:.4f} ± {drop_non_mem_std:.4f}")
    logger.info(f"Impact of dropping ranked memorized nodes: {(drop_mem_ranked_mean - baseline_mean):.4f}")
    logger.info(f"Impact of dropping random non-memorized nodes: {(drop_non_mem_mean - baseline_mean):.4f}")
    
    # After training and getting test accuracies, log to CSV
    csv_file = log_results_to_csv(
        args=args,
        adj_homophily=adj_homophily,
        nli=nli,
        test_acc_mean=avg_f_test_acc,  # Using model f's test accuracy
        test_acc_std=std_f_test_acc
    )
    logger.info(f"\nResults have been logged to {csv_file}")

    # After dimensionality analysis, add feature divergence analysis
    logger.info("\nPerforming feature divergence analysis...")
    feature_div_results = run_feature_divergence_analysis(
        model_f=model_f,
        data=data,
        nodes_dict=nodes_dict,
        node_scores=node_scores,
        save_dir=log_dir,
        timestamp=timestamp,
        model_type=args.model_type,
        dataset_name=args.dataset,
        threshold=0.5,
        device=device,
        logger=logger
    )
    
    # Perform weighted label heterophily analysis
    logger.info("\nPerforming Weighted Label Heterophily analysis...")
    weighted_heterophily_results = run_weighted_heterophily_analysis(
        data=data,
        node_scores=node_scores,
        nodes_dict=nodes_dict,
        save_dir=log_dir,
        timestamp=timestamp,
        model_type=args.model_type,
        dataset_name=args.dataset,
        threshold=0.5,
        similarity_measure='cosine',  # Can be 'cosine' or 'gaussian'
        gaussian_sigma=1.0,  # Only used if similarity_measure is 'gaussian'
        epsilon=1e-6,
        device=device,
        logger=logger
    )

if __name__ == '__main__':
    main()