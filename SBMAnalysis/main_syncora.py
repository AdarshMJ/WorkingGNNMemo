import argparse
import torch
import os
import logging
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import networkx as nx
from scipy import sparse
import pickle
from dataset import CustomDataset
from model import NodeGCN, NodeGAT, NodeGraphConv
from memorization import calculate_node_memorization_score
from main import (set_seed, train_models, verify_no_data_leakage, 
                 setup_logging, get_model, test)
from nodeli import li_node
from neuron_analysis import (analyze_neuron_flipping, analyze_neuron_flipping_with_memorization,
                           plot_neuron_flipping_analysis, plot_neuron_flipping_by_memorization)


def load_and_process_dataset(args, dataset_name, logger):
    """Load synthetic Cora dataset and convert to PyG format"""
    # Construct full path to dataset
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "syn-cora")
    dataset = CustomDataset(root=root_dir, name=dataset_name, setting="gcn")
    
    # Convert to PyG format
    edge_index = torch.from_numpy(np.vstack(dataset.adj.nonzero())).long()
    
    # Convert sparse features to dense numpy array
    if sparse.issparse(dataset.features):
        x = torch.from_numpy(dataset.features.todense()).float()
    else:
        x = torch.from_numpy(dataset.features).float()
    
    y = torch.from_numpy(dataset.labels).long()
    
    # Create train/val/test masks
    train_mask = torch.zeros(len(y), dtype=torch.bool)
    val_mask = torch.zeros(len(y), dtype=torch.bool)
    test_mask = torch.zeros(len(y), dtype=torch.bool)
    
    train_mask[dataset.idx_train] = True
    val_mask[dataset.idx_val] = True
    test_mask[dataset.idx_test] = True
    
    # Convert to networkx for informativeness calculation
    G = nx.Graph()
    G.add_nodes_from(range(len(y)))
    G.add_edges_from(edge_index.t().numpy())
    
    # Calculate label informativeness using existing function
    informativeness = li_node(G, dataset.labels)
    
    # Calculate homophily (edge homophily)
    edges = edge_index.t().numpy()
    same_label = dataset.labels[edges[:, 0]] == dataset.labels[edges[:, 1]]
    homophily = same_label.mean()
    
    # Create a data object
    data = type('Data', (), {
        'x': x,
        'y': y,
        'edge_index': edge_index,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
        'num_nodes': len(y),
        'informativeness': informativeness,
        'homophily': homophily
    })()
    
    logger.info(f"\nDataset Statistics:")
    logger.info(f"Number of nodes: {data.num_nodes}")
    logger.info(f"Number of edges: {len(edges)}")
    logger.info(f"Number of features: {x.shape[1]}")
    logger.info(f"Number of classes: {len(torch.unique(y))}")
    logger.info(f"Homophily: {homophily:.4f}")
    logger.info(f"Label Informativeness: {informativeness:.4f}")
    
    return data

def create_visualization(results_df, save_path, args):
    """Create three plots:
    1. Original scatter plot of homophily vs informativeness colored by memorization rate
    2. Label informativeness vs memorization rate for candidate nodes
    3. Homophily vs memorization rate for candidate nodes
    """
    # Create a figure with three subplots side by side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))
    
    # Plot 1: Original Homophily vs Informativeness with memorization rate as color
    scatter = ax1.scatter(
        results_df['homophily'],
        results_df['informativeness'],
        c=results_df['percent_memorized'],
        cmap='viridis',
        s=100,
        alpha=0.7
    )
    
    ax1.set_xlabel('Homophily', fontsize=12)
    ax1.set_ylabel('Label Informativeness', fontsize=12)
    ax1.set_title('Homophily vs Label Informativeness\n(color: memorization rate)', fontsize=12)
    
    # Add colorbar for the first plot
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Memorization Rate (%)', fontsize=10)
    
    # Plot 2: Label Informativeness vs Memorization Rate
    ax2.scatter(results_df['informativeness'], results_df['percent_memorized'], 
                color='blue', s=100, alpha=0.7)
    
    # Add trend line using numpy's polyfit
    z = np.polyfit(results_df['informativeness'], results_df['percent_memorized'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(results_df['informativeness'].min(), results_df['informativeness'].max(), 100)
    ax2.plot(x_trend, p(x_trend), "r--", alpha=0.8)
    
    # Calculate correlation coefficient
    corr_inf = np.corrcoef(results_df['informativeness'], results_df['percent_memorized'])[0,1]
    
    ax2.set_xlabel('Label Informativeness', fontsize=12)
    ax2.set_ylabel('Memorization Rate (%)', fontsize=12)
    ax2.set_title(f'Label Informativeness vs Memorization\n(r = {corr_inf:.2f})', fontsize=12)
    
    # Plot 3: Homophily vs Memorization Rate
    ax3.scatter(results_df['homophily'], results_df['percent_memorized'], 
                color='green', s=100, alpha=0.7)
    
    # Add trend line
    z = np.polyfit(results_df['homophily'], results_df['percent_memorized'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(results_df['homophily'].min(), results_df['homophily'].max(), 100)
    ax3.plot(x_trend, p(x_trend), "r--", alpha=0.8)
    
    # Calculate correlation coefficient
    corr_hom = np.corrcoef(results_df['homophily'], results_df['percent_memorized'])[0,1]
    
    ax3.set_xlabel('Homophily', fontsize=12)
    ax3.set_ylabel('Memorization Rate (%)', fontsize=12)
    ax3.set_title(f'Homophily vs Memorization\n(r = {corr_hom:.2f})', fontsize=12)
    
    # Add grid to all plots
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax3.grid(True, linestyle='--', alpha=0.3)
    
    # Add main title
    plt.suptitle(f'Memorization Analysis in {args.model_type.upper()}', fontsize=14, y=1.05)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def get_node_splits(data, train_mask, swap_candidate_independent=False):
    """
    Create node splits using all available training nodes.
    
    Args:
        data: PyG data object
        train_mask: Mask for train nodes
        swap_candidate_independent: If True, swap the roles of candidate and independent nodes
    """
    # Get train indices in their original order
    train_indices = torch.where(train_mask)[0]
    num_train_nodes = len(train_indices)
    
    # Calculate split sizes: 50% shared, 25% candidate, 25% independent
    shared_size = int(0.50 * num_train_nodes)
    remaining = num_train_nodes - shared_size
    split_size = remaining // 2
    
    # Split indices sequentially without shuffling
    shared_idx = train_indices[:shared_size].tolist()
    candidate_idx = train_indices[shared_size:shared_size + split_size].tolist()
    independent_idx = train_indices[shared_size + split_size:].tolist()
    
    # Return swapped indices if requested
    if swap_candidate_independent:
        return shared_idx, independent_idx, candidate_idx
    else:
        return shared_idx, candidate_idx, independent_idx

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='gcn',
                       choices=['gcn', 'gat', 'graphconv'])
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--gat_heads', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_passes', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='results/syncora_analysis')
    #parser.add_argument('--analyze_superposition', action='store_true',
     #                  help='Perform superposition analysis to study channel sharing')
    
    args = parser.parse_args()
    
    # Setup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f'syncora_analysis_{args.model_type}_{timestamp}'
    log_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup logging with both file and console output
    logger = logging.getLogger('syncora_analysis')
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear any existing handlers
    
    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, 'analysis.log'))
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Select homophily levels to analyze
    homophily_levels = [0.0, 0.3, 0.5, 0.7, 1.0]  # Changed to match available files
    dataset_files = [f'h{h:.2f}-r1' for h in homophily_levels]
    
    # Initialize results containers
    results = []  # Initialize as list for storing main results
    neuron_results = {}  # Initialize as dict for storing neuron analysis results
    
    for dataset_name in tqdm(dataset_files, desc="Processing datasets"):
        logger.info(f"\nProcessing dataset: {dataset_name}")
        
        # Load and process dataset
        data = load_and_process_dataset(args, dataset_name, logger)
        
        # Move individual tensors to device instead of entire Data object
        data.x = data.x.to(device) if hasattr(data, 'x') else None
        data.edge_index = data.edge_index.to(device) if hasattr(data, 'edge_index') else None
        data.y = data.y.to(device) if hasattr(data, 'y') else None
        if hasattr(data, 'train_mask'):
            data.train_mask = data.train_mask.to(device)
        if hasattr(data, 'val_mask'):
            data.val_mask = data.val_mask.to(device)
        if hasattr(data, 'test_mask'):
            data.test_mask = data.test_mask.to(device)
        
        # Get node splits
        shared_idx, candidate_idx, independent_idx = get_node_splits(
            data, data.train_mask, swap_candidate_independent=False
        )
        
        # Get extra indices from test set
        test_indices = torch.where(data.test_mask)[0]
        extra_size = len(candidate_idx)
        extra_indices = test_indices[:extra_size].tolist()
        
        # Create nodes_dict
        nodes_dict = {
            'shared': shared_idx,
            'candidate': candidate_idx,
            'independent': independent_idx,
            'extra': extra_indices,
            'val': torch.where(data.val_mask)[0].tolist(),
            'test': torch.where(data.test_mask)[0].tolist()
        }
        
        # Train models
        model_f, model_g, f_val_acc, g_val_acc = train_models(
            args=args,
            data=data,
            shared_idx=shared_idx,
            candidate_idx=candidate_idx,
            independent_idx=independent_idx,
            device=device,
            logger=logger,
            output_dir=None
        )
        
        # Calculate memorization scores
        node_scores = calculate_node_memorization_score(
            model_f=model_f,
            model_g=model_g,
            data=data,
            nodes_dict=nodes_dict,
            device=device,
            logger=logger,
            num_passes=args.num_passes
        )
        
        # Perform neuron flipping analysis on model_f - original analysis
        # logger.info("\nPerforming neuron flipping analysis...")
        # flip_results = analyze_neuron_flipping(
        #     model_f=model_f,
        #     model_g=model_g,
        #     data=data,
        #     nodes_dict=nodes_dict,
        #     device=device,
        #     logger=logger
        # )
        
        # # Create neuron flipping plot for this homophily level
        # plot_path = os.path.join(log_dir, f'neuron_flipping_{dataset_name}.png')
        # plot_neuron_flipping_analysis(
        #     results=flip_results,
        #     save_path=plot_path,
        #     title=f'Neuron Flipping Analysis (Homophily={data.homophily:.2f})'
        # )
        
        # # NEW: Perform neuron flipping analysis with memorization categorization
        # logger.info("\nPerforming neuron flipping analysis with memorization categorization...")
        # mem_flip_results = analyze_neuron_flipping_with_memorization(
        #     model_f=model_f,
        #     model_g=model_g,
        #     data=data,
        #     nodes_dict=nodes_dict,
        #     memorization_scores=node_scores,
        #     device=device,
        #     threshold=0.5,
        #     logger=logger
        # )
        
        # # Create directory for memorization-based plots
        # mem_plot_dir = os.path.join(log_dir, f'mem_neuron_flipping_{dataset_name}')
        # os.makedirs(mem_plot_dir, exist_ok=True)
        
        # # Create plots separated by memorization threshold
        # plot_neuron_flipping_by_memorization(
        #     results=mem_flip_results,
        #     save_dir=mem_plot_dir,
        #     threshold=0.5,
        #     bins=30
        # )
        
        # # Log paths to the new plots
        # logger.info(f"\nMemorization-categorized neuron flipping plots saved to: {mem_plot_dir}")
        
        # # Store neuron flipping results
        # neuron_results[dataset_name] = {
        #     'standard': flip_results,
        #     'by_memorization': mem_flip_results
        # }
        
        # Store results
        results.append({
            'dataset': dataset_name,
            'homophily': float(data.homophily),
            'informativeness': float(data.informativeness),
            'percent_memorized': node_scores['candidate']['percentage_above_threshold'],
            'avg_memorization': node_scores['candidate']['avg_score'],
            'num_memorized': node_scores['candidate']['nodes_above_threshold'],
            'total_nodes': len(node_scores['candidate']['mem_scores']),
            'f_val_acc': float(f_val_acc),
            'g_val_acc': float(g_val_acc)
        })
        
        # Log summary statistics for this homophily level
        # logger.info("\nNeuron Flipping Summary:")
        # for node_type in ['shared', 'candidate', 'independent', 'extra']:
        #     if node_type in flip_results and flip_results[node_type]['channels']:
        #         avg_channels = np.mean(flip_results[node_type]['channels'])
        #         std_channels = np.std(flip_results[node_type]['channels'])
        #         logger.info(f"{node_type.capitalize()} nodes: {avg_channels:.2f} ± {std_channels:.2f} channels")
                
        #         # Log layer-wise statistics
        #         logger.info(f"\nLayer distribution for {node_type} nodes:")
        #         for layer_name, orders in flip_results[node_type]['layer_stats'].items():
        #             logger.info(f"  {layer_name}: {len(orders)} channels zeroed")
        
        # Perform superposition analysis if requested
        # if args.analyze_superposition:
        #     logger.info("\nPerforming superposition analysis...")
            
        #     # Create superposition analysis directory inside log_dir
        #     superposition_dir = os.path.join(log_dir, 'superposition_analysis')
        #     os.makedirs(superposition_dir, exist_ok=True)
            
        #     # Create dataset-specific directory
        #     dataset_dir = os.path.join(superposition_dir, dataset_name)
        #     os.makedirs(dataset_dir, exist_ok=True)
            
        #     # Create model-specific directories with dataset name included
        #     f_dir = os.path.join(dataset_dir, 'model_f')
        #     g_dir = os.path.join(dataset_dir, 'model_g')
        #     os.makedirs(f_dir, exist_ok=True)
        #     os.makedirs(g_dir, exist_ok=True)
            
        #     # Analyze both model f and g
        #     superposition_results = {
        #         'model_f': analyze_superposition(
        #             model=model_f,
        #             data=data,
        #             nodes_dict=nodes_dict,
        #             output_dir=f_dir,
        #             device=device,
        #             logger=logger
        #         ),
        #         'model_g': analyze_superposition(
        #             model=model_g,
        #             data=data,
        #             nodes_dict=nodes_dict,
        #             output_dir=g_dir,
        #             device=device,
        #             logger=logger
        #         )
        #     }
            
        #     # Log some summary statistics
        #     logger.info("\nSuperposition Analysis Summary:")
        #     for model_name, sup_results in superposition_results.items():
        #         logger.info(f"\n{model_name.upper()} Analysis:")
                
        #         # Analyze channel sharing
        #         for layer_name, impacts in sup_results['channel_impacts'].items():
        #             logger.info(f"\nLayer: {layer_name}")
                    
        #             # Find channels that affect multiple node types significantly
        #             threshold = 0.2  # 20% prediction change threshold
        #             shared_channels = []
                    
        #             for channel_idx in range(len(impacts['shared'])):
        #                 affected_types = sum(1 for node_type in impacts 
        #                                   if impacts[node_type][channel_idx] > threshold)
        #                 if affected_types > 1:
        #                     shared_channels.append(channel_idx)
                    
        #             logger.info(f"Found {len(shared_channels)} channels encoding multiple concepts")
        #             if shared_channels:
        #                 logger.info(f"Multi-concept channels: {shared_channels}")
            
        #     logger.info("\nVisualization files:")
        #     logger.info(f"- Model f channel sharing heatmap: {os.path.join(f_dir, 'channel_sharing.png')}")
        #     logger.info(f"- Model f class-specific analysis: {os.path.join(f_dir, 'class_channels.png')}")
        #     logger.info(f"- Model g channel sharing heatmap: {os.path.join(g_dir, 'channel_sharing.png')}")
        #     logger.info(f"- Model g class-specific analysis: {os.path.join(g_dir, 'class_channels.png')}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Create visualization
    plot_path = os.path.join(log_dir, f'memorization_analysis_{timestamp}.png')
    create_visualization(results_df, plot_path, args)
    
    # Save results
    results_df.to_csv(os.path.join(log_dir, 'results.csv'), index=False)
    
    # # Save neuron flipping results
    # with open(os.path.join(log_dir, 'neuron_results.pkl'), 'wb') as f:
    #     pickle.dump(neuron_results, f)
    
    logger.info("\nAnalysis complete!")
    logger.info(f"Results saved to: {log_dir}")
    logger.info(f"Memorization visualization saved as: {plot_path}")
    #logger.info("Individual neuron flipping plots saved for each homophily level")

if __name__ == '__main__':
    main()