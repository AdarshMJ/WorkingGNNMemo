import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import copy
from model import NodeGCN, NodeGAT, NodeGraphConv
import torch.nn.functional as F
from tqdm import tqdm
import types

def mask_nodes_and_edges(data, nodes_to_mask, device):
    """
    Mask nodes by zeroing their features and masking their edges in both directions.
    Returns modified data object and masking statistics.
    
    Args:
        data: PyG Data object
        nodes_to_mask: List of node indices to mask
        device: torch device
    Returns:
        masked_data: Modified PyG Data object
        masking_stats: Dictionary of masking statistics
    """
    # Create a deep copy to avoid modifying the original data
    masked_data = copy.deepcopy(data)
    
    # 1. Zero out node features for masked nodes
    masked_data.x[nodes_to_mask] = torch.zeros_like(masked_data.x[nodes_to_mask])
    
    # 2. Create edge weights (1 for unmasked edges, 0 for masked edges)
    edge_weights = torch.ones(masked_data.edge_index.size(1), device=device)
    
    # Get source and target nodes for each edge
    src_nodes = masked_data.edge_index[0]
    dst_nodes = masked_data.edge_index[1]
    
    # Create mask for edges where either source or target is in nodes_to_mask
    masked_edges = torch.isin(src_nodes, torch.tensor(nodes_to_mask, device=device)) | \
                  torch.isin(dst_nodes, torch.tensor(nodes_to_mask, device=device))
    
    # Set weights to 0 for masked edges
    edge_weights[masked_edges] = 0
    
    # Calculate statistics
    total_edges = masked_data.edge_index.size(1)
    masked_edges_count = masked_edges.sum().item()
    total_nodes = masked_data.x.size(0)
    
    # Calculate degree statistics
    degrees = torch.bincount(masked_data.edge_index.flatten())
    avg_degree_before = degrees.float().mean().item()
    
    # Calculate degrees after masking
    # Create a boolean mask for valid edges (weight > 0)
    valid_edges = edge_weights > 0
    masked_adj = masked_data.edge_index[:, valid_edges]
    masked_degrees = torch.bincount(masked_adj.flatten())
    avg_degree_after = masked_degrees.float().mean().item()
    
    # Calculate average degree of masked nodes before masking
    masked_nodes_degrees = degrees[nodes_to_mask].float().mean().item()
    
    # Add edge weights to data object
    masked_data.edge_weights = edge_weights
    
    masking_stats = {
        'masked_nodes': len(nodes_to_mask),
        'node_masking_percentage': (len(nodes_to_mask) / total_nodes) * 100,
        'masked_edges': masked_edges_count,
        'edge_masking_percentage': (masked_edges_count / total_edges) * 100,
        'avg_degree_before': avg_degree_before,
        'avg_degree_after': avg_degree_after,
        'masked_nodes_avg_degree': masked_nodes_degrees
    }
    
    return masked_data, masking_stats

def apply_edge_weights_to_model(model):
    """
    Modify the GNN model to use edge weights in message passing.
    This modifies the model's conv layers to use edge_weights during aggregation.
    """
    for module in model.modules():
        if hasattr(module, 'propagate'):
            # Store the original message and aggregate functions
            if not hasattr(module, '_original_aggregate'):
                module._original_aggregate = module.aggregate
            if not hasattr(module, '_original_message'):
                module._original_message = module.message
            
            # Override the message function to apply edge weights
            def weighted_message(self, x_j, edge_weight=None):
                if edge_weight is not None:
                    x_j = x_j * edge_weight.view(-1, 1)
                return x_j
            
            # Override the aggregate function to handle all possible arguments
            def weighted_aggregate(self, inputs, index, ptr=None, dim_size=None):
                return self._original_aggregate(inputs, index, ptr=ptr, dim_size=dim_size)
            
            module.message = types.MethodType(weighted_message, module)
            module.aggregate = types.MethodType(weighted_aggregate, module)

def forward_with_edge_weights(self, x, edge_index, edge_weights=None, return_node_emb=False):
    """
    Modified forward pass that handles edge weights and optionally returns node embeddings.
    """
    # First convolution layer
    x = self.convs[0](x, edge_index, edge_weight=edge_weights)
    x = F.relu(x)
    x = F.dropout(x, p=0.5, training=self.training)
    
    # Hidden layers
    for conv in self.convs[1:-1]:
        x = conv(x, edge_index, edge_weight=edge_weights)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
    
    # Get node embeddings before final layer
    node_embeddings = x.clone()
    
    # Final layer
    x = self.convs[-1](x, edge_index, edge_weight=edge_weights)
    
    if return_node_emb:
        return x, node_embeddings
    return x

# Update model classes to use the new forward pass
NodeGCN.forward = forward_with_edge_weights
NodeGAT.forward = forward_with_edge_weights
NodeGraphConv.forward = forward_with_edge_weights

def train_and_evaluate(model, data, train_mask, val_mask, test_mask, 
                      lr: float, weight_decay: float, epochs: int, device) -> Tuple[float, float]:
    """Train model and return validation and test accuracies"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_acc = 0
    final_test_acc = 0
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Pass edge_weights if they exist in the data object
        edge_weights = data.edge_weights if hasattr(data, 'edge_weights') else None
        out = model(data.x.to(device), data.edge_index.to(device), edge_weights=edge_weights)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            out = model(data.x.to(device), data.edge_index.to(device), edge_weights=edge_weights)
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
    
    # Calculate percentages to drop
    drop_percentages = [0.1, 0.2, 0.5, 1.0]
    
    results = {
        'drop_memorized_ranked': {p: {'val_accs': [], 'test_accs': [], 'masking_stats': []} for p in drop_percentages},
        'drop_non_memorized_random': {p: {'val_accs': [], 'test_accs': [], 'masking_stats': []} for p in drop_percentages}
    }

    # Run experiments for each seed
    for seed in seeds:
        if logger:
            logger.info(f"\nRunning generalization analysis with seed {seed}")
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create baseline model and get baseline accuracy
        model = get_fresh_model(model_type, data, hidden_dim, num_layers, gat_heads, device)
        apply_edge_weights_to_model(model)
        baseline_val_acc, baseline_test_acc = train_and_evaluate(
            model, data, data.train_mask, data.val_mask, data.test_mask,
            lr, weight_decay, epochs, device
        )
        
        for drop_pct in drop_percentages:
            num_nodes_to_drop = int(num_memorized * drop_pct)
            
            if logger:
                logger.info(f"\nDropping {num_nodes_to_drop} nodes ({drop_pct*100}% of memorized nodes)")
            
            # 1. Drop top-k memorized nodes
            nodes_to_drop = memorized_candidates[:num_nodes_to_drop]
            masked_data, mem_masking_stats = mask_nodes_and_edges(data, nodes_to_drop, device)
            drop_memorized_ranked_mask = data.train_mask.clone()
            drop_memorized_ranked_mask[nodes_to_drop] = False
            
            if logger:
                logger.info("\nMemorized nodes masking statistics:")
                logger.info(f"  Nodes masked: {mem_masking_stats['masked_nodes']} ({mem_masking_stats['node_masking_percentage']:.2f}% of total)")
                logger.info(f"  Edges masked: {mem_masking_stats['masked_edges']} ({mem_masking_stats['edge_masking_percentage']:.2f}% of total)")
                logger.info(f"  Average degree before masking: {mem_masking_stats['avg_degree_before']:.2f}")
                logger.info(f"  Average degree after masking: {mem_masking_stats['avg_degree_after']:.2f}")
                logger.info(f"  Average degree of masked nodes: {mem_masking_stats['masked_nodes_avg_degree']:.2f}")
            
            model = get_fresh_model(model_type, masked_data, hidden_dim, num_layers, gat_heads, device)
            apply_edge_weights_to_model(model)
            val_acc, test_acc = train_and_evaluate(
                model, masked_data, drop_memorized_ranked_mask, 
                masked_data.val_mask, masked_data.test_mask,
                lr, weight_decay, epochs, device
            )
            results['drop_memorized_ranked'][drop_pct]['val_accs'].append(val_acc)
            results['drop_memorized_ranked'][drop_pct]['test_accs'].append(test_acc)
            results['drop_memorized_ranked'][drop_pct]['masking_stats'].append(mem_masking_stats)
            
            # 2. Drop random non-memorized shared nodes
            random_nodes_to_drop = np.random.choice(non_memorized_shared, num_nodes_to_drop, replace=False)
            masked_data, non_mem_masking_stats = mask_nodes_and_edges(data, random_nodes_to_drop, device)
            drop_non_memorized_mask = data.train_mask.clone()
            drop_non_memorized_mask[random_nodes_to_drop] = False
            
            if logger:
                logger.info("\nNon-memorized nodes masking statistics:")
                logger.info(f"  Nodes masked: {non_mem_masking_stats['masked_nodes']} ({non_mem_masking_stats['node_masking_percentage']:.2f}% of total)")
                logger.info(f"  Edges masked: {non_mem_masking_stats['masked_edges']} ({non_mem_masking_stats['edge_masking_percentage']:.2f}% of total)")
                logger.info(f"  Average degree before masking: {non_mem_masking_stats['avg_degree_before']:.2f}")
                logger.info(f"  Average degree after masking: {non_mem_masking_stats['avg_degree_after']:.2f}")
                logger.info(f"  Average degree of masked nodes: {non_mem_masking_stats['masked_nodes_avg_degree']:.2f}")
            
            model = get_fresh_model(model_type, masked_data, hidden_dim, num_layers, gat_heads, device)
            apply_edge_weights_to_model(model)
            val_acc, test_acc = train_and_evaluate(
                model, masked_data, drop_non_memorized_mask, 
                masked_data.val_mask, masked_data.test_mask,
                lr, weight_decay, epochs, device
            )
            results['drop_non_memorized_random'][drop_pct]['val_accs'].append(val_acc)
            results['drop_non_memorized_random'][drop_pct]['test_accs'].append(test_acc)
            results['drop_non_memorized_random'][drop_pct]['masking_stats'].append(non_mem_masking_stats)
            
            if logger:
                logger.info(f"\nResults for {drop_pct*100}% drop:")
                logger.info(f"  Memorized - Test Acc: {test_acc:.4f}")
                logger.info(f"  Non-memorized - Test Acc: {test_acc:.4f}")
    
    # Add baseline results
    results['baseline_test_acc'] = baseline_test_acc
    
    # Calculate and log average masking statistics across seeds
    if logger:
        logger.info("\nAverage masking statistics across all seeds:")
        for drop_type in ['drop_memorized_ranked', 'drop_non_memorized_random']:
            logger.info(f"\n{drop_type.replace('_', ' ').title()}:")
            for drop_pct in drop_percentages:
                stats_list = results[drop_type][drop_pct]['masking_stats']
                avg_stats = {
                    k: np.mean([s[k] for s in stats_list]) 
                    for k in stats_list[0].keys()
                }
                logger.info(f"\n  {drop_pct*100}% drop:")
                logger.info(f"    Average edges masked: {avg_stats['masked_edges']:.1f} ({avg_stats['edge_masking_percentage']:.2f}%)")
                logger.info(f"    Average degree reduction: {avg_stats['avg_degree_before']:.2f} → {avg_stats['avg_degree_after']:.2f}")
                logger.info(f"    Average degree of masked nodes: {avg_stats['masked_nodes_avg_degree']:.2f}")
    
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
    nodes_dropped = [0] + [int(pct * num_memorized_nodes) for pct in drop_percentages]  # Add 0 nodes dropped
    
    # Calculate means and confidence intervals for all points
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
        cis = [0]  # No confidence interval for baseline point
        
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
    plt.ylabel('Test Accuracy')
    #plt.title('Impact of Node Dropping Strategies on Model Generalization')
    
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