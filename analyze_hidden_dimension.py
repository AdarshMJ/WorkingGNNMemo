#!/usr/bin/env python
# analyze_hidden_dimension.py
"""
This script analyzes how hidden dimension size affects GNN memorization.
It runs link prediction experiments with different hidden dimension sizes
and tracks the number of candidate edges with memorization scores > 0.5.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import logging
from datetime import datetime
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
import pandas as pd
from model import LinkGNN
from train import train_models, create_training_data
from memorizationscore import get_edge_splits, calculate_edge_memorization_score

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

def setup_logging(output_dir="results/hidden_dimension_analysis"):
    """Set up logging directory and file."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create directory structure
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logger
    logger = logging.getLogger('hidden_dimension_analysis')
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers
    
    # Create file handler
    log_file = os.path.join(output_dir, f'hidden_dim_analysis_{timestamp}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    
    return logger, output_dir, timestamp

def analyze_hidden_dimensions(hidden_dims, dataset_name='Cora', model_type='gcn', num_layers=3, out_dim=32, seed=42):
    """
    Analyze how hidden dimension affects the number of candidate edges with high memorization scores.
    
    Args:
        hidden_dims: List of hidden dimension sizes to evaluate
        dataset_name: Name of dataset to use
        model_type: GNN model type ('gcn', 'gat', or 'graphconv')
        num_layers: Number of GNN layers
        out_dim: Output dimension size
        seed: Random seed for reproducibility
    """
    # Set up logging
    logger, output_dir, timestamp = setup_logging()
    logger.info(f"Hidden Dimension Analysis for Link Memorization")
    logger.info(f"Dataset: {dataset_name}, Model: {model_type}")
    logger.info(f"Hidden dimensions to evaluate: {hidden_dims}")
    logger.info(f"Using device: {device}")
    logger.info(f"Number of layers: {num_layers}, Output dimension: {out_dim}")
    
    # Set random seed
    set_seed(seed)
    
    # Create data transformation
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                          add_negative_train_samples=False),
    ])
    
    # Load dataset
    logger.info(f"Loading {dataset_name} dataset...")
    dataset = Planetoid(root='data', name=dataset_name, transform=transform)
    data_full = dataset
    train_data, val_data, test_data = data_full[0]
    
    # Log dataset statistics
    logger.info(f"Dataset loaded:")
    logger.info(f"  Number of nodes: {train_data.num_nodes}")
    logger.info(f"  Number of edges: {train_data.edge_index.size(1)}")
    logger.info(f"  Number of node features: {train_data.num_features}")
    
    # Get edge splits
    logger.info("Creating edge splits...")
    split_ratios = (0.50, 0.25, 0.25)  # [shared, candidate, independent]
    edges_dict = get_edge_splits(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        split_ratios=split_ratios
    )
    
    # Log edge split statistics
    for edge_type in ['shared', 'candidate', 'independent']:
        pos_edges = edges_dict[edge_type]['pos']
        neg_edges = edges_dict[edge_type]['neg']
        logger.info(f"  {edge_type.capitalize()} edges: {len(pos_edges)} positive, {len(neg_edges)} negative")
    
    # Analysis results storage
    results = {
        'hidden_dims': hidden_dims,
        'pos_candidate_memorized': [],
        'neg_candidate_memorized': [],
        'pos_candidate_percentage': [],
        'neg_candidate_percentage': [],
    }
    
    # Configure common arguments for all runs
    common_args = argparse.Namespace(
        model_type=model_type,
        out_dim=out_dim,
        num_layers=num_layers,
        heads=4 if model_type == 'gat' else None,
        lr=0.01,
        weight_decay=0.0,
        epochs=100,
        seed=seed,
        split_ratios=list(split_ratios),
        dataset=dataset_name
    )
    
    # Run analysis for each hidden dimension
    for hidden_dim in hidden_dims:
        logger.info(f"\n{'='*80}")
        logger.info(f"Analyzing hidden dimension: {hidden_dim}")
        logger.info(f"{'='*80}")
        
        # Set hidden dimension for this run
        args = argparse.Namespace(**vars(common_args))
        args.hidden_dim = hidden_dim
        
        # Train models
        logger.info("Training models with hidden dimension = {hidden_dim}...")
        model_f, model_g = train_models(data_full, edges_dict, args, logger)
        
        # Calculate memorization scores
        logger.info("Calculating edge memorization scores...")
        edge_scores = calculate_edge_memorization_score(
            model_f=model_f,
            model_g=model_g,
            data=train_data,
            edges_dict=edges_dict,
            device=device,
            logger=logger
        )
        
        # Extract metrics for candidate edges
        if 'candidate' in edge_scores:
            # Positive candidate edges
            pos_above_threshold = edge_scores['candidate']['positive_edges']['above_threshold']
            pos_total = edge_scores['candidate']['positive_edges']['count']
            pos_percentage = edge_scores['candidate']['positive_edges']['percentage_above']
            
            # Negative candidate edges
            neg_above_threshold = edge_scores['candidate']['negative_edges']['above_threshold']
            neg_total = edge_scores['candidate']['negative_edges']['count']
            neg_percentage = edge_scores['candidate']['negative_edges']['percentage_above']
            
            # Store results
            results['pos_candidate_memorized'].append(pos_above_threshold)
            results['neg_candidate_memorized'].append(neg_above_threshold)
            results['pos_candidate_percentage'].append(pos_percentage)
            results['neg_candidate_percentage'].append(neg_percentage)
            
            logger.info(f"Hidden dimension {hidden_dim}:")
            logger.info(f"  Positive candidate edges with score > 0.5: {pos_above_threshold}/{pos_total} ({pos_percentage:.1f}%)")
            logger.info(f"  Negative candidate edges with score > 0.5: {neg_above_threshold}/{neg_total} ({neg_percentage:.1f}%)")
        else:
            logger.warning(f"No 'candidate' edges found in edge_scores")
    
    # Create visualization of results
    create_hidden_dim_plots(results, output_dir, timestamp, model_type, dataset_name)
    
    return results

def create_hidden_dim_plots(results, output_dir, timestamp, model_type, dataset_name):
    """Create plots showing the relationship between hidden dimension and memorization."""
    # Plot absolute numbers
    plt.figure(figsize=(12, 8))
    
    # Create plot for the number of memorized edges
    plt.subplot(211)
    plt.plot(results['hidden_dims'], results['pos_candidate_memorized'], 'o-', color='blue', 
             label='Positive candidate edges')
    plt.plot(results['hidden_dims'], results['neg_candidate_memorized'], 's-', color='red',
             label='Negative candidate edges')
    
    plt.xlabel('Hidden Dimension Size')
    plt.ylabel('Number of Edges with Score > 0.5')
    plt.title(f'Number of Memorized Candidate Edges vs. Hidden Dimension ({dataset_name}, {model_type.upper()})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xscale('log', base=2)  # Log scale for x-axis as dimensions increase exponentially
    
    # Create plot for the percentage of memorized edges
    plt.subplot(212)
    plt.plot(results['hidden_dims'], results['pos_candidate_percentage'], 'o-', color='blue',
             label='Positive candidate edges')
    plt.plot(results['hidden_dims'], results['neg_candidate_percentage'], 's-', color='red',
             label='Negative candidate edges')
    
    plt.xlabel('Hidden Dimension Size')
    plt.ylabel('Percentage of Edges with Score > 0.5 (%)')
    plt.title(f'Percentage of Memorized Candidate Edges vs. Hidden Dimension ({dataset_name}, {model_type.upper()})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xscale('log', base=2)  # Log scale for x-axis as dimensions increase exponentially
    
    # Add tight layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'hidden_dim_analysis_{model_type}_{dataset_name}_{timestamp}.png'), 
                dpi=300, bbox_inches='tight')
    
    # Create a separate plot for a more detailed visualization
    plt.figure(figsize=(14, 10))
    
    # Create a bar chart for positive edges
    plt.subplot(211)
    x = np.arange(len(results['hidden_dims']))
    bar_width = 0.35
    
    plt.bar(x, results['pos_candidate_memorized'], bar_width, color='skyblue', 
            label='Count')
    
    # Add a second y-axis for percentage
    ax2 = plt.twinx()
    ax2.plot(x, results['pos_candidate_percentage'], 'o-', color='darkblue', linewidth=2,
             label='Percentage')
    
    # Add labels and title
    plt.xlabel('Hidden Dimension Size')
    plt.ylabel('Number of Edges')
    ax2.set_ylabel('Percentage (%)')
    plt.title(f'Positive Candidate Edges with Score > 0.5 ({dataset_name}, {model_type.upper()})')
    
    # Add x-tick labels
    plt.xticks(x, [str(dim) for dim in results['hidden_dims']])
    
    # Add legends
    plt.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    # Create a bar chart for negative edges
    plt.subplot(212)
    plt.bar(x, results['neg_candidate_memorized'], bar_width, color='salmon', 
            label='Count')
    
    # Add a second y-axis for percentage
    ax2 = plt.twinx()
    ax2.plot(x, results['neg_candidate_percentage'], 'o-', color='darkred', linewidth=2,
             label='Percentage')
    
    # Add labels and title
    plt.xlabel('Hidden Dimension Size')
    plt.ylabel('Number of Edges')
    ax2.set_ylabel('Percentage (%)')
    plt.title(f'Negative Candidate Edges with Score > 0.5 ({dataset_name}, {model_type.upper()})')
    
    # Add x-tick labels
    plt.xticks(x, [str(dim) for dim in results['hidden_dims']])
    
    # Add legends
    plt.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'hidden_dim_analysis_detail_{model_type}_{dataset_name}_{timestamp}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save the raw data as CSV for future reference
 
    df = pd.DataFrame({
        'hidden_dim': results['hidden_dims'],
        'pos_candidate_memorized': results['pos_candidate_memorized'],
        'neg_candidate_memorized': results['neg_candidate_memorized'],
        'pos_candidate_percentage': results['pos_candidate_percentage'],
        'neg_candidate_percentage': results['neg_candidate_percentage'],
    })
    df.to_csv(os.path.join(output_dir, f'hidden_dim_analysis_{model_type}_{dataset_name}_{timestamp}.csv'), index=False)
    
    print(f"Plots saved in {output_dir}")
    print(f"Raw data saved as CSV in {output_dir}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Hidden Dimension Analysis for Link Memorization')
    
    # Dataset and model parameters
    parser.add_argument('--dataset', type=str, default='Cora',
                        help='Dataset name')
    parser.add_argument('--model_type', type=str, default='gcn',
                        choices=['gcn', 'gat', 'graphconv'],
                        help='GNN model type')
    parser.add_argument('--out_dim', type=int, default=32,
                        help='Output dimension size')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of GNN layers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Define hidden dimensions to analyze
    hidden_dims = [16, 32, 64, 128, 256, 512]
    
    # Run analysis
    analyze_hidden_dimensions(
        hidden_dims=hidden_dims,
        dataset_name=args.dataset,
        model_type=args.model_type,
        num_layers=args.num_layers,
        out_dim=args.out_dim,
        seed=args.seed
    )

if __name__ == "__main__":
    main()