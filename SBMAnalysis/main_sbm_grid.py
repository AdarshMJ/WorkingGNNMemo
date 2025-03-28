import argparse
import torch
import os
import logging
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
import networkx as nx

# Import existing functions from main.py and main_sbm.py
from main import (train_models,get_node_splits,
                 calculate_node_memorization_score)
from reliability_analysis import *
from utils import *

def load_grid_graphs(data_dir):
    """Load all graphs from the grid dataset."""
    with open(os.path.join(data_dir, "sbm_list.pkl"), "rb") as f:
        graphs = pickle.load(f)
    return graphs

def analyze_correlations(results_df, logger):
    """Analyze correlations between graph properties and memorization."""
    correlations = {}
    
    # Variables to analyze
    variables = {
        'homophily': 'Graph Homophily',
        'informativeness': 'Label Informativeness',
        'percent_memorized': 'Memorization Rate',
        'f_val_acc': 'Model F Validation Accuracy',
        'g_val_acc': 'Model G Validation Accuracy'
    }
    
    logger.info("\n=== Correlation Analysis ===")
    
    # Calculate correlations between all pairs
    for var1 in variables:
        for var2 in variables:
            if var1 >= var2:  # Skip duplicates and self-correlations
                continue
                
            # Calculate Spearman correlation
            rho, p_value = spearmanr(results_df[var1], results_df[var2])
            correlations[f'spearman_{var1}_{var2}'] = (rho, p_value)
            
            # Calculate Pearson correlation
            r, p_value_pearson = pearsonr(results_df[var1], results_df[var2])
            correlations[f'pearson_{var1}_{var2}'] = (r, p_value_pearson)
            
            logger.info(f"\n{variables[var1]} vs {variables[var2]}:")
            logger.info(f"Spearman correlation: ρ = {rho:.3f} (p = {p_value:.3e})")
            logger.info(f"Pearson correlation: r = {r:.3f} (p = {p_value_pearson:.3e})")
            
            # Interpret correlation strength
            strength = "strong" if abs(rho) > 0.7 else "moderate" if abs(rho) > 0.4 else "weak"
            logger.info(f"Correlation strength: {strength}")
    
    return correlations

def visualize_sbm_graphs(graphs, homophily_values, output_dir, logger):
    """
    Visualize SBM graphs with different homophily values.
    
    Args:
        graphs: List of PyG data objects
        homophily_values: List of target homophily values to visualize
        output_dir: Directory to save visualizations
        logger: Logger object
    """
    # Create visualization directory
    vis_dir = os.path.join(output_dir, 'graph_visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Find graphs with homophily values closest to target values
    target_graphs = {}
    for target in homophily_values:
        closest_idx = min(range(len(graphs)), key=lambda i: abs(graphs[i].homophily - target))
        target_graphs[target] = {
            'graph': graphs[closest_idx],
            'actual_homophily': graphs[closest_idx].homophily
        }
    
    logger.info(f"\nVisualizing graphs with homophily values: {homophily_values}")
    
    for target, graph_info in target_graphs.items():
        data = graph_info['graph']
        actual_homophily = graph_info['actual_homophily']
        
        # Convert PyG graph to NetworkX for visualization
        edges = data.edge_index.t().numpy()
        G = nx.Graph()
        G.add_nodes_from(range(data.num_nodes))
        G.add_edges_from(edges)
        
        # Get node labels
        labels = data.y.numpy()
        
        # Create a mapping from node to community
        node_community = {}
        for i, label in enumerate(labels):
            node_community[i] = int(label)
        
        # Set node attributes
        nx.set_node_attributes(G, node_community, 'community')
        
        # Create a figure with a good size
        plt.figure(figsize=(12, 10))
        
        # Define a colormap for the communities
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        node_colors = [colors[node_community[node]] for node in G.nodes()]
        
        # Use community-aware layout
        pos = nx.spring_layout(G, seed=42)
        
        # Draw the graph with appropriate styling
        nx.draw_networkx_nodes(G, pos, node_size=50, node_color=node_colors, alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.3)
        
        # Set title and labels
        plt.title(f"SBM Graph with Homophily = {actual_homophily:.2f}", fontsize=16)
        
        # Create a custom legend
        for i, color in enumerate(colors):
            plt.scatter([], [], c=color, label=f'Class {i}')
        plt.legend(fontsize=12, title="Node Classes")
        
        # Remove axes
        plt.axis('off')
        
        # Save figure
        filename = os.path.join(vis_dir, f"sbm_homophily_{actual_homophily:.2f}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved visualization for homophily={actual_homophily:.2f} to {filename}")
    
    logger.info(f"All graph visualizations saved to {vis_dir}")

def create_correlation_heatmap(results_df, save_path):
    """Create a heatmap of correlations between variables."""
    variables = ['homophily', 'informativeness', 'percent_memorized', 
                'f_val_acc', 'g_val_acc']
    
    # Calculate correlation matrix
    corr_matrix = results_df[variables].corr(method='spearman')
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu', vmin=-1, vmax=1, 
                center=0, fmt='.2f', square=True)
    
    # Customize labels
    labels = ['Homophily', 'Informativeness', 'Memorization', 
              'Model F Acc', 'Model G Acc']
    plt.xticks(np.arange(len(labels)) + 0.5, labels, rotation=45, ha='right')
    plt.yticks(np.arange(len(labels)) + 0.5, labels, rotation=0)
    
    plt.title('Spearman Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_memorization_heatmap(results_df, save_path, args):
    """
    Create visualizations of memorization rates:
    1. 2D heatmap of memorization percentage vs homophily and informativeness
    2. Label informativeness vs Rate of memorization
    3. Homophily vs Rate of memorization
    
    Args:
        results_df: DataFrame with columns ['homophily', 'informativeness', 'percent_memorized']
        save_path: Path to save the visualization
        args: Script arguments
    """
    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Plot 1: 2D scatter plot with color representing memorization rate (original plot)
    scatter = axes[0].scatter(
        results_df['homophily'],
        results_df['informativeness'],
        c=results_df['percent_memorized'],
        cmap='viridis',
        s=100,  # marker size
        alpha=0.7
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axes[0])
    cbar.set_label('Memorization Rate (%)', fontsize=12)
    
    # Customize plot
    axes[0].set_xlabel('Graph Homophily', fontsize=12)
    axes[0].set_ylabel('Label Informativeness', fontsize=12)
    axes[0].set_title(f'Homophily vs Informativeness vs Memorization\nModel: {args.model_type.upper()}', fontsize=14)
    
    # Add annotations for each point
    if len(results_df) <= 50:  # Only annotate if not too crowded
        for idx, row in results_df.iterrows():
            axes[0].annotate(
                f"{row['percent_memorized']:.1f}%",
                (row['homophily'], row['informativeness']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, alpha=0.8
            )
    
    # Add grid
    axes[0].grid(True, linestyle='--', alpha=0.3)
    
    # Plot 2: Label informativeness vs Rate of memorization
    axes[1].scatter(
        results_df['informativeness'],
        results_df['percent_memorized'],
        c=results_df['homophily'],  # Color by homophily
        cmap='coolwarm',
        s=100,
        alpha=0.7
    )
    
    # Add color bar
    cbar2 = plt.colorbar(
        plt.cm.ScalarMappable(
            norm=plt.Normalize(
                results_df['homophily'].min(), 
                results_df['homophily'].max()
            ), 
            cmap='coolwarm'
        ), 
        ax=axes[1]
    )
    cbar2.set_label('Homophily', fontsize=12)
    
    # Add smoothed trend line
    if len(results_df) >= 5:
        try:
            from scipy.stats import linregress
            from scipy import interpolate
            
            # Sort by informativeness for smooth line
            sorted_data = results_df.sort_values('informativeness')
            x = sorted_data['informativeness']
            y = sorted_data['percent_memorized']
            
            # Linear regression
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            axes[1].plot(
                x, intercept + slope * x, 
                'r--', 
                label=f'Linear trend (r²={r_value**2:.2f})'
            )
            
            # Add trend info in text
            axes[1].text(
                0.05, 0.95, 
                f'Correlation: {r_value:.2f}\np-value: {p_value:.4f}', 
                transform=axes[1].transAxes,
                fontsize=10, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
            )
            
            axes[1].legend(loc='lower right')
        except Exception as e:
            # If curve fitting fails, just continue without the trend line
            pass
    
    axes[1].set_xlabel('Label Informativeness', fontsize=12)
    axes[1].set_ylabel('Memorization Rate (%)', fontsize=12)
    axes[1].set_title('Label Informativeness vs Memorization Rate', fontsize=14)
    axes[1].grid(True, linestyle='--', alpha=0.3)
    
    # Plot 3: Homophily vs Rate of memorization
    axes[2].scatter(
        results_df['homophily'],
        results_df['percent_memorized'],
        c=results_df['informativeness'],  # Color by informativeness
        cmap='plasma',
        s=100,
        alpha=0.7
    )
    
    # Add color bar
    cbar3 = plt.colorbar(
        plt.cm.ScalarMappable(
            norm=plt.Normalize(
                results_df['informativeness'].min(), 
                results_df['informativeness'].max()
            ), 
            cmap='plasma'
        ), 
        ax=axes[2]
    )
    cbar3.set_label('Label Informativeness', fontsize=12)
    
    # Add smoothed trend line
    if len(results_df) >= 5:
        try:
            # Sort by homophily for smooth line
            sorted_data = results_df.sort_values('homophily')
            x = sorted_data['homophily']
            y = sorted_data['percent_memorized']
            
            # Linear regression
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            axes[2].plot(
                x, intercept + slope * x, 
                'r--', 
                label=f'Linear trend (r²={r_value**2:.2f})'
            )
            
            # Add trend info in text
            axes[2].text(
                0.05, 0.95, 
                f'Correlation: {r_value:.2f}\np-value: {p_value:.4f}', 
                transform=axes[2].transAxes,
                fontsize=10, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
            )
            
            axes[2].legend(loc='lower right')
        except Exception as e:
            # If curve fitting fails, just continue without the trend line
            pass
    
    axes[2].set_xlabel('Graph Homophily', fontsize=12)
    axes[2].set_ylabel('Memorization Rate (%)', fontsize=12)
    axes[2].set_title('Homophily vs Memorization Rate', fontsize=14)
    axes[2].grid(True, linestyle='--', alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save additional individual plots
    base_path = save_path.rsplit('.', 1)[0]
    
    # Plot 1 individual
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        results_df['homophily'],
        results_df['informativeness'],
        c=results_df['percent_memorized'],
        cmap='viridis',
        s=100,
        alpha=0.7
    )
    plt.colorbar(scatter, label='Memorization Rate (%)')
    plt.xlabel('Graph Homophily', fontsize=12)
    plt.ylabel('Label Informativeness', fontsize=12)
    plt.title(f'Homophily vs Informativeness vs Memorization\nModel: {args.model_type.upper()}', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{base_path}_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2 individual
    plt.figure(figsize=(10, 8))
    plt.scatter(
        results_df['informativeness'],
        results_df['percent_memorized'],
        c=results_df['homophily'],
        cmap='coolwarm',
        s=100,
        alpha=0.7
    )
    plt.colorbar(
        plt.cm.ScalarMappable(
            norm=plt.Normalize(
                results_df['homophily'].min(), 
                results_df['homophily'].max()
            ), 
            cmap='coolwarm'
        ),
        label='Homophily'
    )
    plt.xlabel('Label Informativeness', fontsize=12)
    plt.ylabel('Memorization Rate (%)', fontsize=12)
    plt.title('Label Informativeness vs Memorization Rate', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{base_path}_informativeness.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3 individual
    plt.figure(figsize=(10, 8))
    plt.scatter(
        results_df['homophily'],
        results_df['percent_memorized'],
        c=results_df['informativeness'],
        cmap='plasma',
        s=100,
        alpha=0.7
    )
    plt.colorbar(
        plt.cm.ScalarMappable(
            norm=plt.Normalize(
                results_df['informativeness'].min(), 
                results_df['informativeness'].max()
            ), 
            cmap='plasma'
        ),
        label='Label Informativeness'
    )
    plt.xlabel('Graph Homophily', fontsize=12)
    plt.ylabel('Memorization Rate (%)', fontsize=12)
    plt.title('Homophily vs Memorization Rate', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{base_path}_homophily.png", dpi=300, bbox_inches='tight')
    plt.close()

def analyze_node_properties(data, memorization_scores, model_f, model_g, nodes_dict, output_dir, device, logger):
    """
    Analyze node properties including centrality measures and reliability.
    """
    # Create analysis directory
    analysis_dir = os.path.join(output_dir, 'node_analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Extract memorization scores for candidate nodes
    candidate_scores = np.array(memorization_scores['candidate']['mem_scores'])
    node_types = ['candidate'] * len(candidate_scores)
    
    # Calculate centrality measures
    logger.info("\nCalculating centrality measures...")
    centrality_measures = compute_centrality_measures(
        data.edge_index,
        data.num_nodes
    )
    
    # Extract centrality measures for candidate nodes
    candidate_centralities = {
        measure: values[nodes_dict['candidate']]
        for measure, values in centrality_measures.items()
    }
    
    # Create centrality analysis plots
    logger.info("Creating centrality analysis plots...")
    
    # 1. Main centrality analysis plot
    plot_path = os.path.join(analysis_dir, 'centrality_analysis.png')
    plot_centrality_analysis(
        candidate_centralities,
        candidate_scores,
        node_types,
        plot_path
    )
    
    # 2. Centrality vs memorization plot
    plot_path_2 = os.path.join(analysis_dir, 'centrality_vs_memorization.png')
    plot_centrality_vs_memorization(
        candidate_centralities,
        candidate_scores,
        plot_path_2
    )
    
    # Create centrality summary
    summary_df = create_centrality_summary(
        candidate_centralities,
        candidate_scores
    )
    summary_df.to_csv(os.path.join(analysis_dir, 'centrality_summary.csv'))
    
    # Perform reliability analysis
    logger.info("\nPerforming reliability analysis...")
    reliability_results = analyze_reliability_vs_memorization(
        model_f=model_f,
        model_g=model_g,
        data=data,
        node_scores=memorization_scores,
        device=device
    )
    
    # Create reliability plots
    logger.info("Creating reliability analysis plots...")
    reliability_df = plot_reliability_analysis(
        reliability_results,
        os.path.join(analysis_dir, 'reliability_analysis.png')
    )
    reliability_df.to_csv(os.path.join(analysis_dir, 'reliability_summary.csv'))
    
    # Calculate delta entropy using kd_retention
    logger.info("\nCalculating delta entropy...")
    delta_entropy_values = kd_retention(model_f, data, device=device)[nodes_dict['candidate']]
    
    # Analyze delta entropy
    entropy_results = analyze_delta_entropy(
        delta_entropy_values,
        candidate_scores,
        node_types
    )
    
    # Create delta entropy plots
    logger.info("Creating delta entropy analysis plots...")
    plot_delta_entropy_analysis(
        entropy_results,
        os.path.join(analysis_dir, 'delta_entropy_analysis.png'),
        title=f"Delta Entropy Analysis (Homophily={data.homophily:.2f})"
    )
    
    # Create combined analysis plot
    logger.info("Creating combined analysis plot...")
    plot_combined_analysis(
        centrality_results=candidate_centralities,
        delta_entropy_results=entropy_results,
        save_path=os.path.join(analysis_dir, 'combined_analysis.png')
    )
    
    # Log analysis results
    logger.info("\nAnalysis Results:")
    logger.info("\nCentrality Measures:")
    for measure in candidate_centralities:
        corr, p_value = spearmanr(candidate_centralities[measure], candidate_scores)
        logger.info(f"{measure.capitalize()} correlation: ρ={corr:.3f} (p={p_value:.3e})")
    
    logger.info("\nReliability Analysis:")
    for node_type in reliability_results:
        mem_rel = reliability_results[node_type]['memorized_reliability']
        non_mem_rel = reliability_results[node_type]['non_memorized_reliability']
        pvalue = reliability_results[node_type]['stat_test']['pvalue']
        
        if len(mem_rel) > 0 and len(non_mem_rel) > 0:
            logger.info(f"\n{node_type.capitalize()} nodes:")
            logger.info(f"Memorized nodes reliability: {np.mean(mem_rel):.3f} ± {np.std(mem_rel):.3f}")
            logger.info(f"Non-memorized nodes reliability: {np.mean(non_mem_rel):.3f} ± {np.std(non_mem_rel):.3f}")
            logger.info(f"Mann-Whitney U test p-value: {pvalue:.3e}")
    
    return {
        'centrality': candidate_centralities,
        'reliability': reliability_results,
        'delta_entropy': entropy_results,
        'summary': {
            'centrality': summary_df,
            'reliability': reliability_df
        }
    }

def plot_combined_analysis(centrality_results, delta_entropy_results, save_path):
    """
    Create a combined analysis plot with correlations and summary statistics.
    
    Args:
        centrality_results: Dictionary containing centrality measures
        delta_entropy_results: Dictionary containing delta entropy analysis results
        save_path: Path to save the visualization
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Prepare data for correlation matrix
    data = {}
    for measure, values in centrality_results.items():
        data[measure] = values
    # Use delta_entropy directly from results if it exists
    if isinstance(delta_entropy_results, dict) and 'delta_entropy' in delta_entropy_results:
        data['delta_entropy'] = delta_entropy_results['delta_entropy']
    elif isinstance(delta_entropy_results, np.ndarray):
        data['delta_entropy'] = delta_entropy_results
    df = pd.DataFrame(data)
    
    # Calculate correlation matrix
    corr_matrix = df.corr(method='spearman')
    
    # Plot correlation heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu', vmin=-1, vmax=1, 
                center=0, ax=ax1, fmt='.2f')
    ax1.set_title('Measure Correlations')
    
    # Calculate summary statistics
    summary_stats = []
    for col in df.columns:
        stats = {
            'Measure': col,
            'Mean': df[col].mean(),
            'Std': df[col].std(),
            'Min': df[col].min(),
            'Max': df[col].max()
        }
        summary_stats.append(stats)
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df = summary_df.round(3)
    
    # Plot summary table
    table = ax2.table(cellText=summary_df.values,
                     colLabels=summary_df.columns,
                     cellLoc='center',
                     loc='center',
                     bbox=[0.1, 0.1, 0.8, 0.8])
    
    # Adjust table style
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Hide axes for table subplot
    ax2.axis('off')
    ax2.set_title('Summary Statistics')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing grid graphs')
    parser.add_argument('--model_type', type=str, default='gcn',
                       choices=['gcn', 'gat', 'graphconv'])
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--gat_heads', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_passes', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='results/grid_analysis')
    parser.add_argument('--visualize_graphs', action='store_true',
                       help='Visualize graphs with different homophily values')
    
    args = parser.parse_args()
    
    # Setup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f'grid_analysis_{args.model_type}_{timestamp}'
    log_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup logging
    logger = logging.getLogger('grid_analysis')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(log_dir, 'analysis.log'))
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load all graphs
    logger.info(f"Loading graphs from {args.data_dir}")
    graphs = load_grid_graphs(args.data_dir)
    logger.info(f"Loaded {len(graphs)} graphs")
    
    # Visualize graphs with different homophily values if requested
    if args.visualize_graphs:
        homophily_values = [0.0, 0.3, 0.5, 0.7, 1.0]
        visualize_sbm_graphs(graphs, homophily_values, log_dir, logger)
    
    # Process each graph
    results = []
    node_property_results = []
    for idx, data in enumerate(tqdm(graphs, desc="Processing graphs")):
        logger.info(f"\nProcessing graph {idx + 1}/{len(graphs)}")
        logger.info(f"Homophily: {data.homophily:.4f}")
        logger.info(f"Informativeness: {data.informativeness:.4f}")
        
        # Move individual tensors to device
        data.x = data.x.to(device)
        data.edge_index = data.edge_index.to(device)
        data.y = data.y.to(device)
        data.train_mask = data.train_mask.to(device)
        data.val_mask = data.val_mask.to(device)
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
        
        # Store results for candidate nodes
        results.append({
            'homophily': float(data.homophily),
            'informativeness': float(data.informativeness),
            'percent_memorized': node_scores['candidate']['percentage_above_threshold'],
            'avg_memorization': node_scores['candidate']['avg_score'],
            'num_memorized': node_scores['candidate']['nodes_above_threshold'],
            'total_nodes': len(node_scores['candidate']['mem_scores']),
            'f_val_acc': float(f_val_acc),
            'g_val_acc': float(g_val_acc)
        })
        
        # Add node property analysis
        property_analysis = analyze_node_properties(
            data=data,
            memorization_scores=node_scores,
            model_f=model_f,
            model_g=model_g,
            nodes_dict=nodes_dict,
            output_dir=os.path.join(log_dir, f'graph_{idx}'),
            device=device,
            logger=logger
        )
        
        node_property_results.append({
            'homophily': float(data.homophily),
            'informativeness': float(data.informativeness),
            'analysis': property_analysis
        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Perform correlation analysis
    correlations = analyze_correlations(results_df, logger)
    
    # Create correlation heatmap
    heatmap_path = os.path.join(log_dir, 'correlation_heatmap.png')
    create_correlation_heatmap(results_df, heatmap_path)
    logger.info(f"\nCorrelation heatmap saved to: {heatmap_path}")
    
    # Save correlation results
    corr_df = pd.DataFrame({
        'correlation_type': [k for k in correlations.keys()],
        'value': [v[0] for v in correlations.values()],
        'p_value': [v[1] for v in correlations.values()]
    })
    corr_df.to_csv(os.path.join(log_dir, 'correlations.csv'), index=False)
    
    # Create and save visualization
    plot_path = os.path.join(log_dir, 'memorization_heatmap.png')
    create_memorization_heatmap(results_df, plot_path, args)
    
    # Save results
    results_df.to_csv(os.path.join(log_dir, 'grid_results.csv'), index=False)
    
    logger.info("\nAnalysis complete!")
    logger.info(f"Results saved to: {log_dir}")
    logger.info(f"Visualization saved as: {plot_path}")

    # Save node property results
    with open(os.path.join(log_dir, 'node_property_results.pkl'), 'wb') as f:
        pickle.dump(node_property_results, f)

    # Perform category analysis
    logger.info("\nPerforming Category-based Analysis...")
    category_dir = os.path.join(log_dir, 'category_analysis')
    os.makedirs(category_dir, exist_ok=True)

    # Import category analysis function
    from analyze_grid_categories import analyze_grid_categories

    # Run category analysis
    summary_df = analyze_grid_categories(
        results_df=results_df,
        output_dir=category_dir,
        min_samples=5  # Adjust this based on your data
    )

    # Log category analysis results
    logger.info("\nCategory Analysis Summary:")
    for _, row in summary_df.iterrows():
        logger.info(f"\nHomophily: {row['Homophily_Category']}, Informativeness: {row['Informativeness_Category']}")
        logger.info(f"Number of samples: {row['Num_Samples']}")
        logger.info(f"Average memorization: {row['Avg_Memorization']:.2f}%")
        logger.info(f"Spearman correlation with homophily: {row['Spearman_Homophily']:.3f} (p={row['Spearman_H_PValue']:.3e})")
        logger.info(f"Spearman correlation with informativeness: {row['Spearman_Info']:.3f} (p={row['Spearman_I_PValue']:.3e})")

    # Save category analysis results
    summary_df.to_csv(os.path.join(category_dir, 'category_summary.csv'), index=False)
    logger.info(f"\nCategory analysis results saved to: {category_dir}")

if __name__ == '__main__':
    main()
