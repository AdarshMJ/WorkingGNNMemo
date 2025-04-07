#!/usr/bin/env python
# analyze_hidden_dimension.py
"""
This script analyzes how hidden dimension size affects GNN memorization.
It runs link prediction experiments with different hidden dimension sizes
and tracks the number of candidate edges with memorization scores > 0.5.
The analysis is repeated with multiple seeds to ensure statistical reliability.
"""

import torch
import torch_geometric.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import logging
import pandas as pd
from torch_geometric.datasets import Planetoid
from tqdm import tqdm
from datetime import datetime
import random

from model import LinkGNN, LinkGNN_MLP 
from train import train_models_fixed  # Use the fixed training function
from memorizationscore import get_edge_splits, calculate_edge_memorization_score
from dataloader import load_npz_dataset, process_heterophilic_dataset_for_link_prediction, get_heterophilic_datasets
from torch_geometric.datasets import Planetoid, Actor, WikipediaNetwork, WebKB

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

def load_dataset(dataset_name, transform):
    """Load PyG dataset with transformation."""
    if dataset_name.lower() in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root='data', name=dataset_name, transform=transform)
        return dataset
    elif dataset_name.lower() == 'actor':
        dataset = Actor(root='data', transform=transform)
        return dataset
    elif dataset_name.lower() in ['cornell', 'texas', 'wisconsin']:
        dataset = WebKB(root='data', name=dataset_name, transform=transform)
        return dataset
    elif dataset_name.lower() in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(root='data', name=dataset_name, transform=transform)
        return dataset
    elif dataset_name.lower() in get_heterophilic_datasets():
        # Load heterophilic dataset
        data = load_npz_dataset(dataset_name)
        
        # Apply normalization and device transformation first
        normalize_transform = T.Compose([
            T.NormalizeFeatures(),
            T.ToDevice(device)
        ])
        data = normalize_transform(data)
        
        # Now apply the RandomLinkSplit
        link_split = T.RandomLinkSplit(
            num_val=0.05, 
            num_test=0.1, 
            is_undirected=True,
            add_negative_train_samples=False
        )
        
        # Apply the split
        train_data, val_data, test_data = link_split(data)
        
        # Return the data in the same format as Planetoid datasets
        return [(train_data, val_data, test_data)]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def analyze_hidden_dimensions_with_multiple_seeds(
    hidden_dims, 
    dataset_name='Cora', 
    model_type='gcn',
    num_layers=3,
    out_dim=32,
    seeds=[42, 123, 456],  # Multiple seeds for statistical reliability
    split_ratio=(0.50, 0.25, 0.25)  # [shared, candidate, independent]
):
    """
    Analyze how hidden dimension affects the number of candidate edges with high memorization scores.
    Run the analysis multiple times with different seeds for statistical reliability.
    
    Args:
        hidden_dims: List of hidden dimension sizes to evaluate
        dataset_name: Name of dataset to use
        model_type: GNN model type ('gcn', 'gat', or 'graphconv')
        num_layers: Number of GNN layers
        out_dim: Output dimension size
        seeds: List of random seeds to use for multiple runs
        split_ratio: Ratio of shared/candidate/independent edges
    """
    # Set up logging
    logger, output_dir, timestamp = setup_logging()
    logger.info(f"Hidden Dimension Analysis for Link Memorization with Multiple Seeds")
    logger.info(f"Dataset: {dataset_name}, Model: {model_type}")
    logger.info(f"Hidden dimensions to evaluate: {hidden_dims}")
    logger.info(f"Using seeds: {seeds}")
    logger.info(f"Using device: {device}")
    logger.info(f"Number of layers: {num_layers}, Output dimension: {out_dim}")
    
    # Results for all runs with different seeds
    all_results = {dim: [] for dim in hidden_dims}
    
    # Analysis results storage for aggregated statistics
    aggregated_results = {
        'hidden_dims': hidden_dims,
        'pos_candidate_memorized_mean': [],
        'pos_candidate_memorized_std': [],
        'neg_candidate_memorized_mean': [],
        'neg_candidate_memorized_std': [],
        'pos_candidate_percentage_mean': [],
        'pos_candidate_percentage_std': [],
        'neg_candidate_percentage_mean': [],
        'neg_candidate_percentage_std': [],
        'all_runs': []  # Will store data from all runs
    }
    
    # Create data transformation
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                        add_negative_train_samples=False),
    ])
    
    # Load dataset
    logger.info(f"Loading {dataset_name} dataset...")
    dataset = load_dataset(dataset_name, transform)
    
    # Configure common arguments for all runs
    common_args = argparse.Namespace(
        model_type=model_type,
        out_dim=out_dim,
        num_layers=num_layers,
        heads=4 if model_type == 'gat' else None,
        lr=0.01,
        weight_decay=0.0,
        epochs=100,
        dataset=dataset_name,
        split_ratios=list(split_ratio)
    )
    
    # Run analysis for each seed
    for seed_idx, seed in enumerate(seeds):
        logger.info(f"\n\n{'#'*80}")
        logger.info(f"Running analysis with seed {seed} ({seed_idx+1}/{len(seeds)})")
        logger.info(f"{'#'*80}")
        
        # Set random seed
        set_seed(seed)
        
        # Get data with current seed
        data_full = dataset
        train_data, val_data, test_data = data_full[0]
        
        # Get edge splits
        logger.info("Creating edge splits...")
        edges_dict = get_edge_splits(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            split_ratios=split_ratio
        )
        
        # Log edge split statistics
        for edge_type in ['shared', 'candidate', 'independent']:
            pos_edges = edges_dict[edge_type]['pos']
            neg_edges = edges_dict[edge_type]['neg']
            logger.info(f"  {edge_type.capitalize()} edges: {len(pos_edges)} positive, {len(neg_edges)} negative")
        
        # Run analysis for each hidden dimension
        for hidden_dim in tqdm(hidden_dims, desc=f"Seed {seed}"):
            logger.info(f"\n{'='*80}")
            logger.info(f"Analyzing hidden dimension: {hidden_dim} with seed {seed}")
            logger.info(f"{'='*80}")
            
            # Set hidden dimension for this run
            args = argparse.Namespace(**vars(common_args))
            args.hidden_dim = hidden_dim
            args.seed = seed
            
            # Train models using the fixed function
            logger.info(f"Training models with hidden dimension = {hidden_dim}...")
            model_f, model_g = train_models_fixed(data_full, edges_dict, args, logger)
            
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
                
                # Store results for this run
                run_results = {
                    'seed': seed,
                    'hidden_dim': hidden_dim,
                    'pos_candidate_memorized': pos_above_threshold,
                    'neg_candidate_memorized': neg_above_threshold,
                    'pos_candidate_percentage': pos_percentage,
                    'neg_candidate_percentage': neg_percentage,
                    'pos_total': pos_total,
                    'neg_total': neg_total
                }
                
                all_results[hidden_dim].append(run_results)
                aggregated_results['all_runs'].append(run_results)
                
                logger.info(f"Hidden dimension {hidden_dim} with seed {seed}:")
                logger.info(f"  Positive candidate edges with score > 0.5: {pos_above_threshold}/{pos_total} ({pos_percentage:.1f}%)")
                logger.info(f"  Negative candidate edges with score > 0.5: {neg_above_threshold}/{neg_total} ({neg_percentage:.1f}%)")
            else:
                logger.warning(f"No 'candidate' edges found in edge_scores")
    
    # Calculate aggregated statistics for each hidden dimension
    for hidden_dim in hidden_dims:
        runs = all_results[hidden_dim]
        
        if runs:
            # Extract metrics
            pos_memorized = [run['pos_candidate_memorized'] for run in runs]
            neg_memorized = [run['neg_candidate_memorized'] for run in runs]
            pos_percentages = [run['pos_candidate_percentage'] for run in runs]
            neg_percentages = [run['neg_candidate_percentage'] for run in runs]
            
            # Calculate mean and std
            aggregated_results['pos_candidate_memorized_mean'].append(np.mean(pos_memorized))
            aggregated_results['pos_candidate_memorized_std'].append(np.std(pos_memorized))
            aggregated_results['neg_candidate_memorized_mean'].append(np.mean(neg_memorized))
            aggregated_results['neg_candidate_memorized_std'].append(np.std(neg_memorized))
            aggregated_results['pos_candidate_percentage_mean'].append(np.mean(pos_percentages))
            aggregated_results['pos_candidate_percentage_std'].append(np.std(pos_percentages))
            aggregated_results['neg_candidate_percentage_mean'].append(np.mean(neg_percentages))
            aggregated_results['neg_candidate_percentage_std'].append(np.std(neg_percentages))
        else:
            # No results for this dimension
            for metric in ['pos_candidate_memorized_mean', 'pos_candidate_memorized_std', 
                          'neg_candidate_memorized_mean', 'neg_candidate_memorized_std',
                          'pos_candidate_percentage_mean', 'pos_candidate_percentage_std',
                          'neg_candidate_percentage_mean', 'neg_candidate_percentage_std']:
                aggregated_results[metric].append(0)
    
    # Create visualization of results
    create_hidden_dim_plots_with_error_bars(aggregated_results, output_dir, timestamp, model_type, dataset_name, seeds)
    
    # Save all raw data as CSV
    df_all_runs = pd.DataFrame(aggregated_results['all_runs'])
    df_all_runs.to_csv(os.path.join(output_dir, f'hidden_dim_all_runs_{model_type}_{dataset_name}_{timestamp}.csv'), index=False)
    
    return aggregated_results

def create_hidden_dim_plots_with_error_bars(results, output_dir, timestamp, model_type, dataset_name, seeds):
    """Create plots showing the relationship between hidden dimension and memorization with error bars."""
    # Plot absolute numbers with error bars
    plt.figure(figsize=(12, 8))
    
    # Create plot for the number of memorized edges
    plt.subplot(211)
    plt.errorbar(results['hidden_dims'], results['pos_candidate_memorized_mean'], 
                yerr=results['pos_candidate_memorized_std'], 
                fmt='o-', color='blue', capsize=5, 
                label='Positive candidate edges')
    plt.errorbar(results['hidden_dims'], results['neg_candidate_memorized_mean'], 
                yerr=results['neg_candidate_memorized_std'], 
                fmt='s-', color='red', capsize=5,
                label='Negative candidate edges')
    
    plt.xlabel('Hidden Dimension Size')
    plt.ylabel('Number of Edges with Score > 0.5')
    plt.title(f'Number of Memorized Candidate Edges vs. Hidden Dimension\n({dataset_name}, {model_type.upper()}, {len(seeds)} seeds)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xscale('log', base=2)  # Log scale for x-axis as dimensions increase exponentially
    
    # Create plot for the percentage of memorized edges
    plt.subplot(212)
    plt.errorbar(results['hidden_dims'], results['pos_candidate_percentage_mean'], 
                yerr=results['pos_candidate_percentage_std'], 
                fmt='o-', color='blue', capsize=5,
                label='Positive candidate edges')
    plt.errorbar(results['hidden_dims'], results['neg_candidate_percentage_mean'], 
                yerr=results['neg_candidate_percentage_std'], 
                fmt='s-', color='red', capsize=5,
                label='Negative candidate edges')
    
    plt.xlabel('Hidden Dimension Size')
    plt.ylabel('Percentage of Edges with Score > 0.5 (%)')
    plt.title(f'Percentage of Memorized Candidate Edges vs. Hidden Dimension\n({dataset_name}, {model_type.upper()}, {len(seeds)} seeds)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xscale('log', base=2)  # Log scale for x-axis as dimensions increase exponentially
    
    # Add tight layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'hidden_dim_analysis_{model_type}_{dataset_name}_{timestamp}.png'), 
                dpi=300, bbox_inches='tight')
    
    # Create a separate plot for a more detailed visualization with bar plots
    plt.figure(figsize=(14, 10))
    
    # Create a bar chart for positive edges
    plt.subplot(211)
    x = np.arange(len(results['hidden_dims']))
    bar_width = 0.35
    
    plt.bar(x, results['pos_candidate_memorized_mean'], bar_width, color='skyblue', 
            yerr=results['pos_candidate_memorized_std'], capsize=5,
            label='Count (mean)')
    
    # Add a second y-axis for percentage
    ax2 = plt.twinx()
    ax2.errorbar(x, results['pos_candidate_percentage_mean'], 
                yerr=results['pos_candidate_percentage_std'],
                fmt='o-', color='darkblue', linewidth=2, capsize=5,
                label='Percentage (mean)')
    
    # Add labels and title
    plt.xlabel('Hidden Dimension Size')
    plt.ylabel('Number of Edges')
    ax2.set_ylabel('Percentage (%)')
    plt.title(f'Positive Candidate Edges with Score > 0.5\n({dataset_name}, {model_type.upper()}, {len(seeds)} seeds)')
    
    # Add x-tick labels
    plt.xticks(x, [str(dim) for dim in results['hidden_dims']])
    
    # Add legends
    plt.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    # Create a bar chart for negative edges
    plt.subplot(212)
    plt.bar(x, results['neg_candidate_memorized_mean'], bar_width, color='salmon', 
            yerr=results['neg_candidate_memorized_std'], capsize=5,
            label='Count (mean)')
    
    # Add a second y-axis for percentage
    ax2 = plt.twinx()
    ax2.errorbar(x, results['neg_candidate_percentage_mean'], 
                yerr=results['neg_candidate_percentage_std'],
                fmt='o-', color='darkred', linewidth=2, capsize=5,
                label='Percentage (mean)')
    
    # Add labels and title
    plt.xlabel('Hidden Dimension Size')
    plt.ylabel('Number of Edges')
    ax2.set_ylabel('Percentage (%)')
    plt.title(f'Negative Candidate Edges with Score > 0.5\n({dataset_name}, {model_type.upper()}, {len(seeds)} seeds)')
    
    # Add x-tick labels
    plt.xticks(x, [str(dim) for dim in results['hidden_dims']])
    
    # Add legends
    plt.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'hidden_dim_analysis_detail_{model_type}_{dataset_name}_{timestamp}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a table of results with statistics
    plt.figure(figsize=(15, 10))
    
    # Define table data
    table_data = []
    for i, dim in enumerate(results['hidden_dims']):
        pos_mean = results['pos_candidate_memorized_mean'][i]
        pos_std = results['pos_candidate_memorized_std'][i]
        pos_pct_mean = results['pos_candidate_percentage_mean'][i]
        pos_pct_std = results['pos_candidate_percentage_std'][i]
        
        neg_mean = results['neg_candidate_memorized_mean'][i]
        neg_std = results['neg_candidate_memorized_std'][i]
        neg_pct_mean = results['neg_candidate_percentage_mean'][i]
        neg_pct_std = results['neg_candidate_percentage_std'][i]
        
        table_data.append([
            dim,
            f"{pos_mean:.1f} ± {pos_std:.1f}",
            f"{pos_pct_mean:.1f} ± {pos_pct_std:.1f}",
            f"{neg_mean:.1f} ± {neg_std:.1f}",
            f"{neg_pct_mean:.1f} ± {neg_pct_std:.1f}"
        ])
    
    # Create the table
    column_labels = ['Hidden Dim', 'Pos Edge Count', 'Pos Edge %', 'Neg Edge Count', 'Neg Edge %']
    
    plt.axis('tight')
    plt.axis('off')
    table = plt.table(cellText=table_data, colLabels=column_labels, loc='center',
                     cellLoc='center', colLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    plt.title(f"Hidden Dimension Analysis Results\n{dataset_name}, {model_type.upper()}, {len(seeds)} seeds", fontsize=16, pad=20)
    
    plt.savefig(os.path.join(output_dir, f'hidden_dim_analysis_table_{model_type}_{dataset_name}_{timestamp}.png'),
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save the aggregated statistics as CSV for future reference
    agg_df = pd.DataFrame({
        'hidden_dim': results['hidden_dims'],
        'pos_memorized_mean': results['pos_candidate_memorized_mean'],
        'pos_memorized_std': results['pos_candidate_memorized_std'],
        'neg_memorized_mean': results['neg_candidate_memorized_mean'],
        'neg_memorized_std': results['neg_candidate_memorized_std'],
        'pos_percentage_mean': results['pos_candidate_percentage_mean'],
        'pos_percentage_std': results['pos_candidate_percentage_std'],
        'neg_percentage_mean': results['neg_candidate_percentage_mean'],
        'neg_percentage_std': results['neg_candidate_percentage_std'],
    })
    agg_df.to_csv(os.path.join(output_dir, f'hidden_dim_stats_{model_type}_{dataset_name}_{timestamp}.csv'), index=False)
    
    print(f"Plots and statistics saved in {output_dir}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Hidden Dimension Analysis for Link Memorization')
    
    # Dataset and model parameters
    parser.add_argument('--dataset', type=str, default='Cora',
                        choices=['Cora', 'Citeseer', 'Pubmed', 'Cornell', 'Texas', 'Wisconsin', 'Chameleon', 'Squirrel', 'Actor'] + get_heterophilic_datasets(),
                        help='Dataset name')
    parser.add_argument('--model_type', type=str, default='gcn',
                        choices=['gcn', 'gat', 'graphconv'],
                        help='GNN model type')
    parser.add_argument('--out_dim', type=int, default=32,
                        help='Output dimension size')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of GNN layers')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456],
                        help='Random seeds for multiple runs')
    parser.add_argument('--decoder', type=str, default='dot',
                        choices=['dot', 'mlp'],
                        help='Link decoder type: dot product or MLP')
    parser.add_argument('--split_ratios', nargs='+', type=float, default=[0.50, 0.25, 0.25],
                        help='Split ratios for [shared, candidate, independent] edges')
    
    args = parser.parse_args()
    
    # Ensure split ratios sum to 1
    split_sum = sum(args.split_ratios)
    if abs(split_sum - 1.0) > 1e-6:
        args.split_ratios = [ratio / split_sum for ratio in args.split_ratios]
        print(f"Normalized split ratios to sum to 1.0: {args.split_ratios}")
    
    # Define hidden dimensions to analyze
    hidden_dims = [16, 32, 64, 128, 256, 512]
    
    # Run analysis with multiple seeds
    analyze_hidden_dimensions_with_multiple_seeds(
        hidden_dims=hidden_dims,
        dataset_name=args.dataset,
        model_type=args.model_type,
        num_layers=args.num_layers,
        out_dim=args.out_dim,
        seeds=args.seeds,
        split_ratio=tuple(args.split_ratios)
    )

if __name__ == "__main__":
    main()