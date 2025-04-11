#!/usr/bin/env python
# analyze_hidden_dimension_node.py
"""
This script analyzes how hidden dimension size affects GNN memorization in node classification.
It runs node classification experiments with different hidden dimension sizes and seeds
and tracks the number of nodes with memorization scores > 0.5.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import logging
from datetime import datetime
import pandas as pd
from torch_geometric.datasets import Planetoid, Amazon, Actor, WikipediaNetwork, WebKB
from torch_geometric.transforms import RandomNodeSplit, LargestConnectedComponents
from torch_geometric.transforms import Compose

from model import NodeGCN, NodeGAT, NodeGraphConv
from memorization import calculate_node_memorization_score, plot_node_memorization_analysis
from dataloader import load_npz_dataset, get_heterophilic_datasets

# Set up device
if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

def set_seed(seed):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup_logging(output_dir="results/hidden_dimension_analysis_node"):
    """Set up logging directory and file."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create directory structure
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logger
    logger = logging.getLogger('hidden_dimension_analysis_node')
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers
    
    # Create file handler
    log_file = os.path.join(output_dir, f'hidden_dim_analysis_node_{timestamp}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    
    return logger, output_dir, timestamp

def load_dataset(dataset_name):
    """Load the specified dataset."""
    transforms = Compose([
        LargestConnectedComponents(),
        RandomNodeSplit(split='train_rest', num_val=0.2, num_test=0.2)
    ])
    
    # Handle standard PyTorch Geometric datasets
    if dataset_name.lower() in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root='data', name=dataset_name, transform=transforms)
    elif dataset_name.lower() in ['computers', 'photo']:
        dataset = Amazon(root='data', name=dataset_name, transform=transforms)
    elif dataset_name.lower() == 'actor':
        dataset = Actor(root='data/Actor', transform=transforms)
    elif dataset_name.lower() in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(root=f'data/{dataset_name}', name=dataset_name, transform=transforms)
    elif dataset_name.lower() in ['cornell', 'wisconsin','texas']:
        dataset = WebKB(root=f'data/{dataset_name}', name=dataset_name, transform=transforms)
    # Handle NPZ heterophilic datasets
    elif dataset_name.lower() in map(str.lower, get_heterophilic_datasets()):
        # Load the NPZ dataset and convert to a PyG dataset
        pyg_data = load_npz_dataset(dataset_name)
        
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
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
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

def get_node_splits(data, train_mask):
    """
    Create node splits without shuffling to preserve natural ordering.
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
    candidate_idx = train_indices[shared_size:shared_size + split_size].tolist()
    independent_idx = train_indices[shared_size + split_size:shared_size + split_size * 2].tolist()
    
    return shared_idx, candidate_idx, independent_idx

def train(model, x, edge_index, train_mask, y, optimizer, device):
    """Train a GNN model for one epoch."""
    model.train()
    optimizer.zero_grad()
    out = model(x.to(device), edge_index.to(device))
    loss = torch.nn.functional.cross_entropy(out[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, x, edge_index, mask, y, device):
    """Test a GNN model on the given data."""
    model.eval()
    with torch.no_grad():
        out = model(x.to(device), edge_index.to(device))
        pred = out[mask].max(1)[1]
        correct = pred.eq(y[mask]).sum().item()
        total = mask.sum().item()
    return correct / total

def train_models_with_hidden_dim(data, hidden_dim, model_type, num_layers, gat_heads, lr, weight_decay, epochs, device, logger, seed=42):
    """Train model f and g with the specified hidden dimension."""
    set_seed(seed)
    
    # Get node splits
    shared_idx, candidate_idx, independent_idx = get_node_splits(data, data.train_mask)
    
    # Create train masks for model f and g
    train_mask_f = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    train_mask_f[shared_idx + candidate_idx] = True
    
    train_mask_g = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    train_mask_g[shared_idx + independent_idx] = True
    
    # Get number of classes
    num_classes = data.y.max().item() + 1
    
    # Initialize models
    model_f = get_model(model_type, data.x.size(1), num_classes, 
                       hidden_dim, num_layers, gat_heads).to(device)
    model_g = get_model(model_type, data.x.size(1), num_classes, 
                       hidden_dim, num_layers, gat_heads).to(device)
    
    # Initialize optimizers
    opt_f = torch.optim.Adam(model_f.parameters(), lr=lr, weight_decay=weight_decay)
    opt_g = torch.optim.Adam(model_g.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Track best models and validation accuracies
    best_f_val_acc = 0
    best_g_val_acc = 0
    best_f_state = None
    best_g_state = None
    
    # Train the models
    for epoch in range(epochs):
        # Train model f on shared+candidate nodes
        f_loss = train(model_f, data.x, data.edge_index, 
                     train_mask_f, data.y, opt_f, device)
        f_val_acc = test(model_f, data.x, data.edge_index, 
                       data.val_mask, data.y, device)
        
        # Train model g on shared+independent nodes
        g_loss = train(model_g, data.x, data.edge_index, 
                     train_mask_g, data.y, opt_g, device)
        g_val_acc = test(model_g, data.x, data.edge_index, 
                       data.val_mask, data.y, device)
        
        # Save best models based on validation accuracy
        if f_val_acc > best_f_val_acc:
            best_f_val_acc = f_val_acc
            best_f_state = model_f.state_dict()
        
        if g_val_acc > best_g_val_acc:
            best_g_val_acc = g_val_acc
            best_g_state = model_g.state_dict()
        
        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0:
            logger.info(f'[Seed {seed}] Hidden dim {hidden_dim}, Epoch {epoch+1}/{epochs}:')
            logger.info(f'Model f - Loss: {f_loss:.4f}, Val Acc: {f_val_acc:.4f}')
            logger.info(f'Model g - Loss: {g_loss:.4f}, Val Acc: {g_val_acc:.4f}')
    
    # Load best states
    model_f.load_state_dict(best_f_state)
    model_g.load_state_dict(best_g_state)
    
    logger.info(f"\n[Seed {seed}] Hidden dim {hidden_dim} Results:")
    logger.info(f"Best Model f - Val Acc: {best_f_val_acc:.4f}")
    logger.info(f"Best Model g - Val Acc: {best_g_val_acc:.4f}")
    
    # Create nodes dictionary for memorization score calculation
    test_indices = torch.where(data.test_mask)[0]
    extra_size = len(candidate_idx)
    extra_indices = test_indices[:extra_size].tolist()
    
    nodes_dict = {
        'shared': shared_idx,
        'candidate': candidate_idx,
        'independent': independent_idx,
        'extra': extra_indices,
        'val': torch.where(data.val_mask)[0].tolist(),
        'test': torch.where(data.test_mask)[0].tolist()
    }
    
    return model_f, model_g, nodes_dict, best_f_val_acc, best_g_val_acc

def analyze_hidden_dimensions(hidden_dims, dataset_name='Cora', model_type='gcn', num_layers=3, 
                             gat_heads=4, lr=0.01, weight_decay=5e-4, epochs=100, seeds=[42, 123, 456]):
    """
    Analyze how hidden dimension affects the number of nodes with high memorization scores.
    Runs the experiment with multiple seeds and reports average results with error bars.
    
    Args:
        hidden_dims: List of hidden dimension sizes to evaluate
        dataset_name: Name of dataset to use
        model_type: GNN model type ('gcn', 'gat', or 'graphconv')
        num_layers: Number of GNN layers
        gat_heads: Number of attention heads (for GAT only)
        lr: Learning rate
        weight_decay: Weight decay parameter
        epochs: Number of training epochs
        seeds: List of random seeds for multiple runs
    """
    # Set up logging
    logger, output_dir, timestamp = setup_logging()
    logger.info(f"Hidden Dimension Analysis for Node Memorization with Multiple Seeds")
    logger.info(f"Dataset: {dataset_name}, Model: {model_type}")
    logger.info(f"Hidden dimensions to evaluate: {hidden_dims}")
    logger.info(f"Using device: {device}")
    logger.info(f"Number of layers: {num_layers}")
    if model_type.lower() == 'gat':
        logger.info(f"Number of attention heads: {gat_heads}")
    logger.info(f"Training parameters: lr={lr}, weight_decay={weight_decay}, epochs={epochs}")
    logger.info(f"Running with {len(seeds)} seeds: {seeds}")
    
    # Load dataset
    logger.info(f"Loading {dataset_name} dataset...")
    dataset = load_dataset(dataset_name)
    data = dataset[0].to(device)
    
    # Log dataset statistics
    logger.info(f"Dataset loaded:")
    logger.info(f"  Number of nodes: {data.num_nodes}")
    logger.info(f"  Number of edges: {data.edge_index.size(1)}")
    logger.info(f"  Number of node features: {data.num_features}")
    logger.info(f"  Number of classes: {dataset.num_classes}")
    
    # Create multi-seed results storage
    multi_seed_results = {
        'hidden_dims': hidden_dims,
        'seeds': seeds,
        'shared_memorized': {seed: [] for seed in seeds},
        'candidate_memorized': {seed: [] for seed in seeds},
        'independent_memorized': {seed: [] for seed in seeds},
        'extra_memorized': {seed: [] for seed in seeds},
        'shared_percentage': {seed: [] for seed in seeds},
        'candidate_percentage': {seed: [] for seed in seeds},
        'independent_percentage': {seed: [] for seed in seeds},
        'extra_percentage': {seed: [] for seed in seeds},
        'f_val_accs': {seed: [] for seed in seeds},
        'g_val_accs': {seed: [] for seed in seeds},
    }
    
    # For each seed, run the complete analysis over all hidden dimensions
    for seed_idx, seed in enumerate(seeds):
        logger.info(f"\n{'#'*80}")
        logger.info(f"Running with seed {seed} ({seed_idx+1}/{len(seeds)})")
        logger.info(f"{'#'*80}")
        
        # For each hidden dimension, train models and calculate memorization scores
        for hidden_dim in hidden_dims:
            logger.info(f"\n{'='*80}")
            logger.info(f"Analyzing hidden dimension: {hidden_dim} with seed {seed}")
            logger.info(f"{'='*80}")
            
            # Train models with the current hidden dimension
            model_f, model_g, nodes_dict, f_val_acc, g_val_acc = train_models_with_hidden_dim(
                data=data,
                hidden_dim=hidden_dim,
                model_type=model_type,
                num_layers=num_layers,
                gat_heads=gat_heads,
                lr=lr,
                weight_decay=weight_decay,
                epochs=epochs,
                device=device,
                logger=logger,
                seed=seed
            )
            
            # Store validation accuracies
            multi_seed_results['f_val_accs'][seed].append(f_val_acc)
            multi_seed_results['g_val_accs'][seed].append(g_val_acc)
            
            # Calculate memorization scores
            logger.info(f"Calculating node memorization scores with seed {seed}...")
            node_scores = calculate_node_memorization_score(
                model_f=model_f,
                model_g=model_g,
                data=data,
                nodes_dict=nodes_dict,
                device=device,
                logger=None,  # Disable detailed logging to avoid clutter
                num_passes=1
            )
            
            # Extract and store metrics for each node type
            for node_type in ['shared', 'candidate', 'independent', 'extra']:
                if node_type in node_scores:
                    above_threshold = node_scores[node_type]['nodes_above_threshold']
                    total = len(node_scores[node_type]['mem_scores'])
                    percentage = node_scores[node_type]['percentage_above_threshold']
                    
                    multi_seed_results[f'{node_type}_memorized'][seed].append(above_threshold)
                    multi_seed_results[f'{node_type}_percentage'][seed].append(percentage)
                    
                    logger.info(f"[Seed {seed}] Hidden dim {hidden_dim}, {node_type.capitalize()} nodes:")
                    logger.info(f"  Nodes with score > 0.5: {above_threshold}/{total} ({percentage:.1f}%)")
                    logger.info(f"  Average memorization score: {node_scores[node_type]['avg_score']:.4f}")
            
            # Create memorization plot for this hidden dimension and seed
            plot_filename = f'mem_score_{model_type}_{dataset_name}_hidden{hidden_dim}_seed{seed}_{timestamp}.png'
            plot_path = os.path.join(output_dir, plot_filename)
            
            plot_node_memorization_analysis(
                node_scores=node_scores,
                save_path=plot_path,
                title_suffix=f"Dataset: {dataset_name}, Model: {model_type}, Hidden Dim: {hidden_dim}, Seed: {seed}",
                node_types_to_plot=['shared', 'candidate', 'independent', 'extra']
            )
            logger.info(f"Memorization score plot saved to: {plot_path}")
    
    # Calculate averages and standard deviations across seeds
    # These will be used for plotting with error bars
    aggregated_results = {
        'hidden_dims': hidden_dims,
        'shared_memorized_avg': [],
        'candidate_memorized_avg': [],
        'independent_memorized_avg': [],
        'extra_memorized_avg': [],
        'shared_percentage_avg': [],
        'candidate_percentage_avg': [],
        'independent_percentage_avg': [],
        'extra_percentage_avg': [],
        'shared_memorized_std': [],
        'candidate_memorized_std': [],
        'independent_memorized_std': [],
        'extra_memorized_std': [],
        'shared_percentage_std': [],
        'candidate_percentage_std': [],
        'independent_percentage_std': [],
        'extra_percentage_std': [],
        'f_val_accs_avg': [],
        'g_val_accs_avg': [],
        'f_val_accs_std': [],
        'g_val_accs_std': [],
    }
    
    # Calculate averages and standard deviations for each metric and hidden dimension
    for i, _ in enumerate(hidden_dims):
        for node_type in ['shared', 'candidate', 'independent', 'extra']:
            # Extract values for this hidden dimension across all seeds
            memorized_values = [multi_seed_results[f'{node_type}_memorized'][seed][i] for seed in seeds]
            percentage_values = [multi_seed_results[f'{node_type}_percentage'][seed][i] for seed in seeds]
            
            # Calculate and store averages
            aggregated_results[f'{node_type}_memorized_avg'].append(np.mean(memorized_values))
            aggregated_results[f'{node_type}_percentage_avg'].append(np.mean(percentage_values))
            
            # Calculate and store standard deviations
            aggregated_results[f'{node_type}_memorized_std'].append(np.std(memorized_values))
            aggregated_results[f'{node_type}_percentage_std'].append(np.std(percentage_values))
        
        # Also aggregate validation accuracies
        f_val_values = [multi_seed_results['f_val_accs'][seed][i] for seed in seeds]
        g_val_values = [multi_seed_results['g_val_accs'][seed][i] for seed in seeds]
        
        aggregated_results['f_val_accs_avg'].append(np.mean(f_val_values))
        aggregated_results['g_val_accs_avg'].append(np.mean(g_val_values))
        aggregated_results['f_val_accs_std'].append(np.std(f_val_values))
        aggregated_results['g_val_accs_std'].append(np.std(g_val_values))
    
    # Create visualization with error bars
    create_hidden_dim_plots_with_error_bars(aggregated_results, output_dir, timestamp, model_type, dataset_name)
    
    # Save the multi-seed raw data as CSV for future reference
    multi_seed_df = pd.DataFrame({
        'hidden_dim': list(hidden_dims) * len(seeds),
        'seed': sorted([seed for seed in seeds for _ in hidden_dims]),
        'shared_memorized': [multi_seed_results['shared_memorized'][seed][i] 
                            for seed in seeds for i in range(len(hidden_dims))],
        'candidate_memorized': [multi_seed_results['candidate_memorized'][seed][i] 
                              for seed in seeds for i in range(len(hidden_dims))],
        'independent_memorized': [multi_seed_results['independent_memorized'][seed][i] 
                                for seed in seeds for i in range(len(hidden_dims))],
        'extra_memorized': [multi_seed_results['extra_memorized'][seed][i] 
                          for seed in seeds for i in range(len(hidden_dims))],
        'shared_percentage': [multi_seed_results['shared_percentage'][seed][i] 
                            for seed in seeds for i in range(len(hidden_dims))],
        'candidate_percentage': [multi_seed_results['candidate_percentage'][seed][i] 
                              for seed in seeds for i in range(len(hidden_dims))],
        'independent_percentage': [multi_seed_results['independent_percentage'][seed][i] 
                                 for seed in seeds for i in range(len(hidden_dims))],
        'extra_percentage': [multi_seed_results['extra_percentage'][seed][i] 
                           for seed in seeds for i in range(len(hidden_dims))],
        'f_val_acc': [multi_seed_results['f_val_accs'][seed][i] 
                    for seed in seeds for i in range(len(hidden_dims))],
        'g_val_acc': [multi_seed_results['g_val_accs'][seed][i] 
                     for seed in seeds for i in range(len(hidden_dims))]
    })
    
    multi_seed_df_path = os.path.join(output_dir, f'hidden_dim_analysis_node_multi_seed_{model_type}_{dataset_name}_{timestamp}.csv')
    multi_seed_df.to_csv(multi_seed_df_path, index=False)
    logger.info(f"Raw data with multiple seeds saved as CSV: {multi_seed_df_path}")
    
    # Also save the aggregated results
    aggregated_df = pd.DataFrame({
        'hidden_dim': hidden_dims,
        'shared_memorized_avg': aggregated_results['shared_memorized_avg'],
        'candidate_memorized_avg': aggregated_results['candidate_memorized_avg'],
        'independent_memorized_avg': aggregated_results['independent_memorized_avg'],
        'extra_memorized_avg': aggregated_results['extra_memorized_avg'],
        'shared_percentage_avg': aggregated_results['shared_percentage_avg'],
        'candidate_percentage_avg': aggregated_results['candidate_percentage_avg'],
        'independent_percentage_avg': aggregated_results['independent_percentage_avg'],
        'extra_percentage_avg': aggregated_results['extra_percentage_avg'],
        'shared_memorized_std': aggregated_results['shared_memorized_std'],
        'candidate_memorized_std': aggregated_results['candidate_memorized_std'],
        'independent_memorized_std': aggregated_results['independent_memorized_std'],
        'extra_memorized_std': aggregated_results['extra_memorized_std'],
        'shared_percentage_std': aggregated_results['shared_percentage_std'],
        'candidate_percentage_std': aggregated_results['candidate_percentage_std'],
        'independent_percentage_std': aggregated_results['independent_percentage_std'],
        'extra_percentage_std': aggregated_results['extra_percentage_std'],
        'f_val_acc_avg': aggregated_results['f_val_accs_avg'],
        'g_val_acc_avg': aggregated_results['g_val_accs_avg'],
        'f_val_acc_std': aggregated_results['f_val_accs_std'],
        'g_val_acc_std': aggregated_results['g_val_accs_std']
    })
    
    aggregated_df_path = os.path.join(output_dir, f'hidden_dim_analysis_node_aggregated_{model_type}_{dataset_name}_{timestamp}.csv')
    aggregated_df.to_csv(aggregated_df_path, index=False)
    logger.info(f"Aggregated data saved as CSV: {aggregated_df_path}")
    
    return aggregated_results, multi_seed_results

def create_hidden_dim_plots_with_error_bars(results, output_dir, timestamp, model_type, dataset_name):
    """Create plots showing the relationship between hidden dimension and memorization with error bars."""
    # Plot absolute numbers with error bars
    plt.figure(figsize=(12, 8))
    
    # Create plot for the number of memorized nodes
    plt.subplot(211)
    plt.errorbar(results['hidden_dims'], results['shared_memorized_avg'], yerr=results['shared_memorized_std'], 
                fmt='o-', color='red', capsize=5, label='Shared nodes')
    plt.errorbar(results['hidden_dims'], results['candidate_memorized_avg'], yerr=results['candidate_memorized_std'], 
                fmt='o-', color='blue', capsize=5, label='Candidate nodes')
    plt.errorbar(results['hidden_dims'], results['independent_memorized_avg'], yerr=results['independent_memorized_std'], 
                fmt='o-', color='orange', capsize=5, label='Independent nodes')
    plt.errorbar(results['hidden_dims'], results['extra_memorized_avg'], yerr=results['extra_memorized_std'], 
                fmt='o-', color='green', capsize=5, label='Extra nodes')
    
    plt.xlabel('Hidden Dimension Size')
    plt.ylabel('Number of Nodes with Score > 0.5')
    plt.title(f'Number of Memorized Nodes vs. Hidden Dimension ({dataset_name}, {model_type.upper()})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xscale('log', base=2)  # Log scale for x-axis as dimensions increase exponentially
    
    # Create plot for the percentage of memorized nodes
    plt.subplot(212)
    plt.errorbar(results['hidden_dims'], results['shared_percentage_avg'], yerr=results['shared_percentage_std'], 
                fmt='o-', color='red', capsize=5, label='Shared nodes')
    plt.errorbar(results['hidden_dims'], results['candidate_percentage_avg'], yerr=results['candidate_percentage_std'], 
                fmt='o-', color='blue', capsize=5, label='Candidate nodes')
    plt.errorbar(results['hidden_dims'], results['independent_percentage_avg'], yerr=results['independent_percentage_std'], 
                fmt='o-', color='orange', capsize=5, label='Independent nodes')
    plt.errorbar(results['hidden_dims'], results['extra_percentage_avg'], yerr=results['extra_percentage_std'], 
                fmt='o-', color='green', capsize=5, label='Extra nodes')
    
    plt.xlabel('Hidden Dimension Size')
    plt.ylabel('Percentage of Nodes with Score > 0.5 (%)')
    plt.title(f'Percentage of Memorized Nodes vs. Hidden Dimension ({dataset_name}, {model_type.upper()})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xscale('log', base=2)  # Log scale for x-axis as dimensions increase exponentially
    
    # Add tight layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'hidden_dim_analysis_node_with_error_bars_{model_type}_{dataset_name}_{timestamp}.png'), 
                dpi=300, bbox_inches='tight')
    
    # Create a separate plot for candidate nodes with validation accuracy
    plt.figure(figsize=(10, 8))
    
    # Create plot for candidate nodes memorization vs hidden dim with val accuracy
    ax1 = plt.gca()
    
    # Plot percentage of memorized nodes with error bars (left y-axis)
    line1 = ax1.errorbar(results['hidden_dims'], results['candidate_percentage_avg'], 
                        yerr=results['candidate_percentage_std'],
                        fmt='o-', color='blue', linewidth=2, capsize=5,
                        label='Candidate nodes memorized (%)')
    ax1.set_xlabel('Hidden Dimension Size')
    ax1.set_ylabel('Percentage of Nodes with Score > 0.5 (%)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xscale('log', base=2)
    
    # Create second y-axis for validation accuracy
    ax2 = ax1.twinx()
    line2 = ax2.errorbar(results['hidden_dims'], results['f_val_accs_avg'], 
                        yerr=results['f_val_accs_std'],
                        fmt='s--', color='red', linewidth=2, capsize=5,
                        label='Model f validation accuracy')
    line3 = ax2.errorbar(results['hidden_dims'], results['g_val_accs_avg'], 
                        yerr=results['g_val_accs_std'],
                        fmt='^--', color='green', linewidth=2, capsize=5,
                        label='Model g validation accuracy')
    ax2.set_ylabel('Validation Accuracy', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    
    # Add legend
    lines = [line1, line2, line3]
    labels = ['Candidate nodes memorized (%)', 'Model f validation accuracy', 'Model g validation accuracy']
    plt.legend(lines, labels, loc='best')
    
    plt.title(f'Candidate Node Memorization vs. Hidden Dimension ({dataset_name}, {model_type.upper()})')
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'candidate_memorization_vs_accuracy_with_error_bars_{model_type}_{dataset_name}_{timestamp}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a bar plot with error bars for all node types
    plt.figure(figsize=(14, 8))
    
    # Set up bar positions
    x = np.arange(len(results['hidden_dims']))
    width = 0.2  # Width of bars
    
    # Create bar chart with error bars
    plt.bar(x - width*1.5, results['shared_percentage_avg'], width, yerr=results['shared_percentage_std'],
            capsize=5, label='Shared', color='red', alpha=0.7)
    plt.bar(x - width/2, results['candidate_percentage_avg'], width, yerr=results['candidate_percentage_std'],
            capsize=5, label='Candidate', color='blue', alpha=0.7)
    plt.bar(x + width/2, results['independent_percentage_avg'], width, yerr=results['independent_percentage_std'], 
            capsize=5, label='Independent', color='orange', alpha=0.7)
    plt.bar(x + width*1.5, results['extra_percentage_avg'], width, yerr=results['extra_percentage_std'],
            capsize=5, label='Extra', color='green', alpha=0.7)
    
    plt.xlabel('Hidden Dimension Size')
    plt.ylabel('Percentage of Nodes with Score > 0.5 (%)')
    plt.title(f'Memorization Rate by Node Type vs. Hidden Dimension ({dataset_name}, {model_type.upper()})')
    plt.xticks(x, [str(dim) for dim in results['hidden_dims']])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'node_type_comparison_with_error_bars_{model_type}_{dataset_name}_{timestamp}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Hidden Dimension Analysis for Node Memorization')
    
    # Dataset and model parameters
    parser.add_argument('--dataset', type=str, default='Cora',
                        help='Dataset name')
    parser.add_argument('--model_type', type=str, default='gcn',
                        choices=['gcn', 'gat', 'graphconv'],
                        help='GNN model type')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of GNN layers')
    parser.add_argument('--gat_heads', type=int, default=4,
                        help='Number of attention heads (for GAT only)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay parameter')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456],
                        help='Random seeds for multiple runs')
    
    args = parser.parse_args()
    
    # Define hidden dimensions to analyze (powers of 2 from 8 to 256)
    hidden_dims = [8, 16, 32, 64, 128, 256]
    
    # Run analysis with multiple seeds
    analyze_hidden_dimensions(
        hidden_dims=hidden_dims,
        dataset_name=args.dataset,
        model_type=args.model_type,
        num_layers=args.num_layers,
        gat_heads=args.gat_heads,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        seeds=args.seeds
    )

if __name__ == "__main__":
    main()