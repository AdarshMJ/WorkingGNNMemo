import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import copy
from model import NodeGCN, NodeGAT, NodeGraphConv
import torch.nn.functional as F
from tqdm import tqdm

def train_and_evaluate(model, data, train_mask, val_mask, test_mask, 
                      lr: float, weight_decay: float, epochs: int, device) -> Tuple[float, float]:
    """Train model and return validation and test accuracies"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_acc = 0
    final_test_acc = 0
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x.to(device), data.edge_index.to(device))
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
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
    """
    Analyze impact of excluding memorized vs non-memorized nodes on model generalization
    
    Args:
        data: PyG Data object
        node_scores: Dictionary containing memorization scores
        model_type: Type of GNN model ('gcn', 'gat', 'graphconv')
        hidden_dim: Hidden dimension size
        num_layers: Number of layers
        gat_heads: Number of attention heads (for GAT)
        lr: Learning rate
        weight_decay: Weight decay
        epochs: Number of epochs
        drop_ratio: Ratio of nodes to exclude (default: 0.05)
        seeds: List of random seeds
        device: Device to run on
        logger: Logger object
    
    Returns:
        Dictionary containing results
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get candidate nodes with high memorization and shared nodes with low memorization
    candidate_data = node_scores['candidate']['raw_data']
    shared_data = node_scores['shared']['raw_data']
    
    memorized_candidates = candidate_data[candidate_data['mem_score'] > 0.5]['node_idx'].values
    non_memorized_shared = shared_data[shared_data['mem_score'] < 0.5]['node_idx'].values
    
    # Calculate number of nodes to drop
    num_memorized = len(memorized_candidates)
    num_non_memorized = len(non_memorized_shared)
    num_nodes_to_drop = min(
        int(min(num_memorized, num_non_memorized) * drop_ratio),
        min(num_memorized, num_non_memorized)
    )
    
    if logger:
        logger.info("\nGeneralization Analysis Setup:")
        logger.info(f"Total memorized candidate nodes: {num_memorized}")
        logger.info(f"Total non-memorized shared nodes: {num_non_memorized}")
        logger.info(f"Number of nodes to exclude: {num_nodes_to_drop} ({drop_ratio*100:.1f}%)")
    
    results = {
        'baseline': {'val_accs': [], 'test_accs': []},
        'drop_memorized': {'val_accs': [], 'test_accs': []},
        'drop_non_memorized': {'val_accs': [], 'test_accs': []}
    }
    
    # Run experiments for each seed
    for seed in seeds:
        if logger:
            logger.info(f"\nRunning generalization analysis with seed {seed}")
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Randomly sample nodes to exclude
        memorized_to_drop = np.random.choice(memorized_candidates, num_nodes_to_drop, replace=False)
        non_memorized_to_drop = np.random.choice(non_memorized_shared, num_nodes_to_drop, replace=False)
        
        # Create masks
        baseline_mask = data.train_mask.clone()
        drop_memorized_mask = data.train_mask.clone()
        drop_non_memorized_mask = data.train_mask.clone()
        
        # Update masks
        drop_memorized_mask[memorized_to_drop] = False
        drop_non_memorized_mask[non_memorized_to_drop] = False
        
        # Initialize models
        def get_fresh_model():
            num_features = data.x.size(1)
            num_classes = data.y.max().item() + 1
            if model_type.lower() == 'gcn':
                return NodeGCN(num_features, num_classes, hidden_dim, num_layers).to(device)
            elif model_type.lower() == 'gat':
                return NodeGAT(num_features, num_classes, hidden_dim, num_layers, heads=gat_heads).to(device)
            else:  # graphconv
                return NodeGraphConv(num_features, num_classes, hidden_dim, num_layers).to(device)
        
        # Baseline
        model = get_fresh_model()
        val_acc, test_acc = train_and_evaluate(
            model, data, baseline_mask, data.val_mask, data.test_mask,
            lr, weight_decay, epochs, device
        )
        results['baseline']['val_accs'].append(val_acc)
        results['baseline']['test_accs'].append(test_acc)
        
        # Drop memorized nodes
        model = get_fresh_model()
        val_acc, test_acc = train_and_evaluate(
            model, data, drop_memorized_mask, data.val_mask, data.test_mask,
            lr, weight_decay, epochs, device
        )
        results['drop_memorized']['val_accs'].append(val_acc)
        results['drop_memorized']['test_accs'].append(test_acc)
        
        # Drop non-memorized nodes
        model = get_fresh_model()
        val_acc, test_acc = train_and_evaluate(
            model, data, drop_non_memorized_mask, data.val_mask, data.test_mask,
            lr, weight_decay, epochs, device
        )
        results['drop_non_memorized']['val_accs'].append(val_acc)
        results['drop_non_memorized']['test_accs'].append(test_acc)
        
        if logger:
            logger.info(f"\nSeed {seed} Results:")
            logger.info(f"Baseline - Test Acc: {results['baseline']['test_accs'][-1]:.4f}")
            logger.info(f"Drop Memorized - Test Acc: {results['drop_memorized']['test_accs'][-1]:.4f}")
            logger.info(f"Drop Non-memorized - Test Acc: {results['drop_non_memorized']['test_accs'][-1]:.4f}")
    
    return results

def plot_generalization_results(results: Dict, save_path: str, title_suffix: str = ""):
    """Create visualization comparing test accuracies under different node dropping scenarios"""
    
    # Calculate means and standard deviations
    baseline_mean = np.mean(results['baseline']['test_accs'])
    baseline_std = np.std(results['baseline']['test_accs'])
    
    drop_mem_mean = np.mean(results['drop_memorized']['test_accs'])
    drop_mem_std = np.std(results['drop_memorized']['test_accs'])
    
    drop_non_mem_mean = np.mean(results['drop_non_memorized']['test_accs'])
    drop_non_mem_std = np.std(results['drop_non_memorized']['test_accs'])
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    
    x = np.arange(3)
    width = 0.35
    
    means = [baseline_mean, drop_mem_mean, drop_non_mem_mean]
    stds = [baseline_std, drop_mem_std, drop_non_mem_std]
    
    plt.bar(x, means, width, yerr=stds, capsize=5)
    
    plt.xticks(x, ['Baseline', 'Drop\nMemorized', 'Drop\nNon-memorized'])
    plt.ylabel('Test Accuracy')
    plt.title('Impact of Node Dropping on Model Generalization')
    
    if title_suffix:
        plt.suptitle(title_suffix, fontsize=12)
        plt.subplots_adjust(top=0.85)
    
    # Add value labels on top of bars
    for i, v in enumerate(means):
        plt.text(i, v + stds[i] + 0.01, f'{v:.4f}±{stds[i]:.4f}', 
                ha='center', va='bottom', fontsize=10)
    
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()