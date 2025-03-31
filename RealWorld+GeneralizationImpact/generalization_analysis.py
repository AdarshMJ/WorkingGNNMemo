import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import copy
from model import NodeGCN, NodeGAT, NodeGraphConv
import torch.nn.functional as F
from tqdm import tqdm
import scipy.stats as stats
from torch_geometric.utils import dropout_node

def train_and_evaluate(model, data, train_mask, val_mask, test_mask, 
                      lr: float, weight_decay: float, epochs: int, device,
                      training_edge_index=None) -> Tuple[float, float]:
    """Train model and return validation and test accuracies
    
    Args:
        model: The GNN model to train
        data: PyG data object
        train_mask: Mask for training nodes
        val_mask: Mask for validation nodes
        test_mask: Mask for test nodes
        lr: Learning rate
        weight_decay: Weight decay factor
        epochs: Number of epochs to train
        device: Device to train on
        training_edge_index: Optional modified edge index to use during training (for node dropping)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_acc = 0
    final_test_acc = 0
    
    # Use provided edge_index for training if given, otherwise use original
    train_edge_index = training_edge_index if training_edge_index is not None else data.edge_index
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Use modified edge_index for training
        out = model(data.x.to(device), train_edge_index.to(device))
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            # Use full edge_index for validation and testing
            out = model(data.x.to(device), data.edge_index.to(device))
            val_acc = (out[val_mask].argmax(dim=1) == data.y[val_mask]).float().mean()
            test_acc = (out[test_mask].argmax(dim=1) == data.y[test_mask]).float().mean()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                final_test_acc = test_acc
        
        model.train()
    
    return best_val_acc.item(), final_test_acc.item()

def analyze_generalization_impact(
    data,
    node_scores: Dict,
    model_type: str,
    hidden_dim: int,
    num_layers: int,
    gat_heads: int,
    lr: float,
    weight_decay: float,
    epochs: int,
    drop_ratio: float = 0.05,
    seeds: List[int] = [42, 123, 456],
    device=None,
    logger=None
) -> Dict:
    """Analyze impact of excluding memorized vs non-memorized nodes on model generalization"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get candidate nodes with high memorization and shared nodes with low memorization
    candidate_data = node_scores['candidate']['raw_data']
    shared_data = node_scores['shared']['raw_data']
    
    # Sort memorized nodes by score in decreasing order
    memorized_candidates = candidate_data[candidate_data['mem_score'] > 0.5].sort_values('mem_score', ascending=False)['node_idx'].values
    # Get non-memorized shared nodes (no ranking needed)
    non_memorized_shared = shared_data[shared_data['mem_score'] < 0.5]['node_idx'].values
    
    # Calculate number of memorized nodes - this will be our maximum drop size
    num_memorized = len(memorized_candidates)
    
    # Calculate percentages to drop - now including 0.0
    drop_percentages = [0.0, 0.1, 0.2, 0.5, 1.0]
    
    results = {
        'drop_memorized_ranked': {p: {'val_accs': [], 'test_accs': []} for p in drop_percentages},
        'drop_non_memorized_random': {p: {'val_accs': [], 'test_accs': []} for p in drop_percentages}
    }
    
    # Run original model once to get baseline accuracy
    model = get_fresh_model(model_type, data, hidden_dim, num_layers, gat_heads, device)
    baseline_val_acc, baseline_test_acc = train_and_evaluate(
        model, data, data.train_mask, data.val_mask, data.test_mask,
        lr, weight_decay, epochs, device
    )
    
    if logger:
        logger.info("\nGeneralization Analysis Setup:")
        logger.info(f"Total memorized nodes: {num_memorized}")
        logger.info(f"Total non-memorized shared nodes available: {len(non_memorized_shared)}")
        logger.info(f"Original training mask size: {data.train_mask.sum().item()} nodes")
        logger.info(f"Baseline test accuracy: {baseline_test_acc:.4f}")
        logger.info("\nNode dropping schedule:")
        for pct in drop_percentages[1:]:  # Skip 0% in the log
            nodes_to_drop = int(num_memorized * pct)
            logger.info(f"  {pct*100}% = {nodes_to_drop} nodes")
    
    # Run experiments for each seed
    for seed in seeds:
        if logger:
            logger.info(f"\nRunning generalization analysis with seed {seed}")
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # For each percentage
        for drop_pct in drop_percentages:
            # For 0%, just use baseline accuracy
            if drop_pct == 0.0:
                results['drop_memorized_ranked'][drop_pct]['val_accs'].append(baseline_val_acc)
                results['drop_memorized_ranked'][drop_pct]['test_accs'].append(baseline_test_acc)
                results['drop_non_memorized_random'][drop_pct]['val_accs'].append(baseline_val_acc)
                results['drop_non_memorized_random'][drop_pct]['test_accs'].append(baseline_test_acc)
                continue
                
            # Calculate number of nodes to drop based on number of memorized nodes
            num_nodes_to_drop = int(num_memorized * drop_pct)
            
            if logger:
                logger.info(f"\nDropping {num_nodes_to_drop} nodes ({drop_pct*100}% of memorized nodes)")
            
            # 1. Drop top-k memorized nodes
            # Create training mask excluding memorized nodes
            drop_memorized_ranked_mask = data.train_mask.clone()
            nodes_to_drop = memorized_candidates[:num_nodes_to_drop]
            drop_memorized_ranked_mask[nodes_to_drop] = False
            
            # Create node mask for edge dropping
            node_mask = torch.ones(data.num_nodes, dtype=torch.bool)
            node_mask[nodes_to_drop] = False
            
            # Remove edges connected to dropped nodes
            edge_mask = node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]]
            dropped_edge_index = data.edge_index[:, edge_mask]
            
            if logger:
                remaining_nodes = drop_memorized_ranked_mask.sum().item()
                remaining_edges = edge_mask.sum().item()
                logger.info(f"  Top-k memorized - Remaining training nodes: {remaining_nodes} (dropped {data.train_mask.sum().item() - remaining_nodes})")
                logger.info(f"  Top-k memorized - Remaining edges: {remaining_edges} (dropped {data.edge_index.size(1) - remaining_edges})")
            
            model = get_fresh_model(model_type, data, hidden_dim, num_layers, gat_heads, device)
            val_acc, test_acc = train_and_evaluate(
                model, data, drop_memorized_ranked_mask, data.val_mask, data.test_mask,
                lr, weight_decay, epochs, device,
                training_edge_index=dropped_edge_index
            )
            results['drop_memorized_ranked'][drop_pct]['val_accs'].append(val_acc)
            results['drop_memorized_ranked'][drop_pct]['test_accs'].append(test_acc)
            
            # 2. Drop random non-memorized shared nodes
            drop_non_memorized_mask = data.train_mask.clone()
            random_nodes_to_drop = np.random.choice(non_memorized_shared, num_nodes_to_drop, replace=False)
            drop_non_memorized_mask[random_nodes_to_drop] = False
            
            # Create node mask for edge dropping
            node_mask = torch.ones(data.num_nodes, dtype=torch.bool)
            node_mask[random_nodes_to_drop] = False
            
            # Remove edges connected to dropped nodes
            edge_mask = node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]]
            dropped_edge_index = data.edge_index[:, edge_mask]
            
            if logger:
                remaining_nodes = drop_non_memorized_mask.sum().item()
                remaining_edges = edge_mask.sum().item()
                logger.info(f"  Random non-memorized - Remaining training nodes: {remaining_nodes} (dropped {data.train_mask.sum().item() - remaining_nodes})")
                logger.info(f"  Random non-memorized - Remaining edges: {remaining_edges} (dropped {data.edge_index.size(1) - remaining_edges})")
            
            model = get_fresh_model(model_type, data, hidden_dim, num_layers, gat_heads, device)
            val_acc, test_acc = train_and_evaluate(
                model, data, drop_non_memorized_mask, data.val_mask, data.test_mask,
                lr, weight_decay, epochs, device,
                training_edge_index=dropped_edge_index
            )
            results['drop_non_memorized_random'][drop_pct]['val_accs'].append(val_acc)
            results['drop_non_memorized_random'][drop_pct]['test_accs'].append(test_acc)
            
            if logger:
                logger.info(f"\nResults for dropping {drop_pct*100}% nodes ({num_nodes_to_drop} nodes):")
                logger.info(f"  Top-k memorized - Test Acc: {results['drop_memorized_ranked'][drop_pct]['test_accs'][-1]:.4f}")
                logger.info(f"  Random non-memorized - Test Acc: {test_acc:.4f}")
    
    # No need to store baseline_test_acc separately since it's included in results
    return results

def get_fresh_model(model_type, data, hidden_dim, num_layers, gat_heads, device):
    """Helper function to create a new model instance"""
    num_features = data.x.size(1)
    num_classes = data.y.max().item() + 1
    
    if model_type.lower() == 'gcn':
        return NodeGCN(num_features, num_classes, hidden_dim, num_layers).to(device)
    elif model_type.lower() == 'gat':
        return NodeGAT(num_features, num_classes, hidden_dim, num_layers, heads=gat_heads).to(device)
    else:  # graphconv
        return NodeGraphConv(num_features, num_classes, hidden_dim, num_layers).to(device)

def plot_generalization_results(results: Dict, save_path: str, num_memorized_nodes: int, title_suffix: str = ""):
    """Create line plot comparing test accuracies under different node dropping scenarios"""
    plt.figure(figsize=(10, 6))
    
    # Convert percentages to actual number of nodes
    drop_percentages = list(results['drop_memorized_ranked'].keys())
    nodes_dropped = [int(pct * num_memorized_nodes) for pct in drop_percentages]
    
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
        means = []
        cis = []
        
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
        
        # Make the baseline point (0% dropped) more prominent
        plt.plot(0, means[0], marker='o', markersize=8, color=color)
    
    # Set y-axis limits to focus on the relevant range
    all_means = []
    all_cis = []
    for strategy in ['drop_memorized_ranked', 'drop_non_memorized_random']:
        for pct in drop_percentages:
            test_accs = results[strategy][pct]['test_accs']
            mean, ci = get_stats(test_accs)
            all_means.append(mean)
            all_cis.append(ci)
    
    y_min = min(np.array(all_means) - np.array(all_cis)) - 0.01
    y_max = max(np.array(all_means) + np.array(all_cis)) + 0.01
    plt.ylim(y_min, y_max)
    
    plt.xlabel('Number of Nodes Dropped')
    plt.ylabel('Test Accuracy')
    plt.title('Impact of Node Dropping Strategies on Model Generalization')
    
    if title_suffix:
        plt.suptitle(title_suffix, fontsize=12)
        plt.subplots_adjust(top=0.85)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_memorization_vs_misclassification(
    model_f: torch.nn.Module,
    data,
    node_scores: Dict,
    save_path: str,
    device=None
) -> Dict:
    """
    Analyze relationship between memorization and misclassification.
    Specifically for candidate nodes, check if memorized nodes (score > 0.5)
    are more likely to be misclassified.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get predictions from model_f
    model_f.eval()
    with torch.no_grad():
        logits = model_f(data.x.to(device), data.edge_index.to(device))
        preds = logits.argmax(dim=1)
    
    # Get candidate nodes data
    candidate_data = node_scores['candidate']['raw_data']
    candidate_indices = candidate_data['node_idx'].values
    candidate_scores = candidate_data['mem_score'].values
    
    # Get actual and predicted labels for candidate nodes
    true_labels = data.y[candidate_indices].cpu()
    pred_labels = preds[candidate_indices].cpu()
    
    # Calculate misclassification
    is_misclassified = (true_labels != pred_labels).numpy()
    is_memorized = candidate_scores > 0.5
    
    # Calculate statistics
    total_memorized = is_memorized.sum()
    total_non_memorized = len(is_memorized) - total_memorized
    
    memorized_misclassified = (is_memorized & is_misclassified).sum()
    non_memorized_misclassified = (~is_memorized & is_misclassified).sum()
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    
    # Create 2x2 grid for memorized/non-memorized vs correct/incorrect
    categories = ['Memorized', 'Non-memorized']
    correct_counts = [
        total_memorized - memorized_misclassified,
        total_non_memorized - non_memorized_misclassified
    ]
    incorrect_counts = [memorized_misclassified, non_memorized_misclassified]
    
    x = np.arange(len(categories))
    width = 0.35
    
    # Plot bars
    plt.bar(x - width/2, correct_counts, width, label='Correctly Classified', color='green', alpha=0.6)
    plt.bar(x + width/2, incorrect_counts, width, label='Misclassified', color='red', alpha=0.6)
    
    # Add counts and percentages on top of bars
    def add_label(count, total, x_pos, y_pos):
        pct = (count/total) * 100
        plt.text(x_pos, y_pos, f'{count}\n({pct:.1f}%)', 
                ha='center', va='bottom')
    
    # Add labels for correct predictions
    add_label(correct_counts[0], total_memorized, x[0]-width/2, correct_counts[0])
    add_label(correct_counts[1], total_non_memorized, x[1]-width/2, correct_counts[1])
    
    # Add labels for incorrect predictions
    add_label(incorrect_counts[0], total_memorized, x[0]+width/2, incorrect_counts[0])
    add_label(incorrect_counts[1], total_non_memorized, x[1]+width/2, incorrect_counts[1])
    
    plt.xlabel('Node Type')
    plt.ylabel('Number of Nodes')
    plt.title('Classification Performance vs Memorization\nfor Candidate Nodes')
    plt.xticks(x, categories)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Compute chi-square test for independence
    contingency_table = np.array([
        [total_memorized - memorized_misclassified, memorized_misclassified],
        [total_non_memorized - non_memorized_misclassified, non_memorized_misclassified]
    ])
    chi2, p_value = stats.chi2_contingency(contingency_table)[:2]
    
    return {
        'total_memorized': total_memorized,
        'total_non_memorized': total_non_memorized,
        'memorized_misclassified': memorized_misclassified,
        'non_memorized_misclassified': non_memorized_misclassified,
        'memorized_misclassification_rate': memorized_misclassified / total_memorized,
        'non_memorized_misclassification_rate': non_memorized_misclassified / total_non_memorized,
        'chi2_statistic': chi2,
        'p_value': p_value
    }