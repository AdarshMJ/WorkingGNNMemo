import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Union, Tuple, Any
from scipy import stats
import os
from collections import defaultdict

def convert_edge_index_to_adj_list(edge_index: torch.Tensor) -> Dict[int, List[int]]:
    """
    Convert PyG edge_index to an adjacency list representation for efficient neighbor lookup.
    
    Args:
        edge_index: PyG edge index tensor of shape [2, num_edges]
        
    Returns:
        Dictionary mapping node indices to lists of their neighbors
    """
    adj_list = defaultdict(list)
    
    for i in range(edge_index.shape[1]):
        source = edge_index[0, i].item()
        target = edge_index[1, i].item()
        adj_list[source].append(target)
        
    return adj_list

def calculate_weighted_label_heterophily(
    data,
    similarity_measure: str = 'cosine',
    gaussian_sigma: float = 1.0,
    epsilon: float = 1e-6,
    device = None
) -> torch.Tensor:
    """
    Calculate weighted label heterophily (conflict) score for each node.
    
    Args:
        data: PyG Data object containing node features, edge indices and labels
        similarity_measure: Method to compute feature similarity ('cosine' or 'gaussian')
        gaussian_sigma: Standard deviation if using gaussian similarity
        epsilon: Small value to prevent division by zero
        device: PyTorch device
        
    Returns:
        Tensor of shape [num_nodes] containing weighted conflict scores
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    num_nodes = data.x.size(0)
    x = data.x.to(device)
    y = data.y.to(device)
    edge_index = data.edge_index.to(device)
    
    # Pre-calculate adjacency list for efficient neighbor lookup
    adj_list = convert_edge_index_to_adj_list(edge_index)
    
    # Initialize weighted conflict scores
    weighted_conflict_scores = torch.zeros(num_nodes, device=device)
    
    # Calculate scores for each node
    for i in range(num_nodes):
        x_i = x[i]  # Central node features
        y_i = y[i]  # Central node label
        neighbors = adj_list[i]  # Get neighbors
        
        if not neighbors:  # Skip isolated nodes
            continue
            
        total_weighted_conflict = 0.0
        total_similarity = 0.0
        
        # Process all neighbors
        for j in neighbors:
            x_j = x[j]  # Neighbor features
            y_j = y[j]  # Neighbor label
            
            # Calculate feature similarity
            if similarity_measure == 'cosine':
                # Rescale cosine similarity from [-1, 1] to [0, 1]
                s_ij = (1.0 + F.cosine_similarity(
                    x_i.unsqueeze(0), x_j.unsqueeze(0)
                ).item()) / 2.0
            elif similarity_measure == 'gaussian':
                # Gaussian similarity
                squared_dist = torch.sum((x_i - x_j) ** 2).item()
                s_ij = torch.exp(torch.tensor(-squared_dist / (gaussian_sigma ** 2))).item()
            else:
                raise ValueError(f"Unknown similarity measure: {similarity_measure}")
            
            # Determine label conflict (1.0 if labels differ, 0.0 otherwise)
            conflict_j = 1.0 if y_j != y_i else 0.0
            
            # Update totals
            total_weighted_conflict += s_ij * conflict_j
            total_similarity += s_ij
            
        # Calculate normalized score
        if total_similarity > 0:
            weighted_conflict_scores[i] = total_weighted_conflict / (total_similarity + epsilon)
    
    return weighted_conflict_scores

def analyze_weighted_heterophily_scores(
    weighted_conflict_scores: torch.Tensor,
    node_scores: Dict[str, Dict],
    nodes_dict: Dict[str, List[int]],
    threshold: float = 0.5,
    node_types_to_analyze: List[str] = None
) -> Dict[str, Any]:
    """
    Analyze weighted heterophily scores by comparing memorized vs non-memorized nodes.
    
    Args:
        weighted_conflict_scores: Tensor of shape [num_nodes] with conflict scores
        node_scores: Dictionary containing memorization scores by node type
        nodes_dict: Dictionary mapping node types to lists of indices
        threshold: Threshold for memorization
        node_types_to_analyze: List of node types to analyze (default: None, analyze all)
        
    Returns:
        Dictionary with analysis results
    """
    results = {}
    
    # Default to analyzing all available node types
    if node_types_to_analyze is None:
        node_types_to_analyze = list(node_scores.keys())
        
    weighted_conflict_scores_np = weighted_conflict_scores.cpu().numpy()
    
    for node_type in node_types_to_analyze:
        if node_type not in node_scores:
            continue
            
        node_data = node_scores[node_type]['raw_data']
        memorized_mask = node_data['mem_score'] > threshold
        node_indices = node_data['node_idx'].values
        
        conflict_scores = np.array([weighted_conflict_scores_np[idx] for idx in node_indices])
        
        # Split by memorization
        mem_conflict = conflict_scores[memorized_mask]
        non_mem_conflict = conflict_scores[~memorized_mask]
        
        # Calculate statistics
        mem_stats = {
            'count': len(mem_conflict),
            'mean': np.mean(mem_conflict),
            'std': np.std(mem_conflict),
            'values': mem_conflict.tolist()  # Store all values for plotting
        }
        
        non_mem_stats = {
            'count': len(non_mem_conflict),
            'mean': np.mean(non_mem_conflict),
            'std': np.std(non_mem_conflict),
            'values': non_mem_conflict.tolist()
        }
        
        # Perform statistical test if both groups have data
        if len(mem_conflict) > 0 and len(non_mem_conflict) > 0:
            t_stat, p_val = stats.ttest_ind(mem_conflict, non_mem_conflict, equal_var=False)
            
            # Calculate effect size (Cohen's d)
            mean_diff = np.mean(mem_conflict) - np.mean(non_mem_conflict)
            pooled_std = np.sqrt((np.std(mem_conflict)**2 + np.std(non_mem_conflict)**2) / 2)
            effect_size = abs(mean_diff) / pooled_std if pooled_std > 0 else 0
            
            # Interpret effect size
            if effect_size < 0.2:
                effect_size_interp = "negligible"
            elif effect_size < 0.5:
                effect_size_interp = "small"
            elif effect_size < 0.8:
                effect_size_interp = "medium"
            else:
                effect_size_interp = "large"
                
            stat_test = {
                't_statistic': t_stat,
                'p_value': p_val,
                'effect_size': effect_size,
                'effect_size_interpretation': effect_size_interp
            }
        else:
            stat_test = None
        
        results[node_type] = {
            'memorized': mem_stats,
            'non_memorized': non_mem_stats,
            'stat_test': stat_test
        }
    
    return results

def plot_weighted_heterophily_comparison(
    results: Dict[str, Dict],
    save_path: str,
    similarity_measure: str = 'cosine',
    title_suffix: str = ""
) -> None:
    """
    Create visualization comparing weighted heterophily scores between 
    memorized and non-memorized nodes.
    
    Args:
        results: Dictionary with results from analyze_weighted_heterophily_scores
        save_path: Path to save the plot
        similarity_measure: Method used to compute feature similarity
        title_suffix: Additional text to add to plot title
    """
    # Create a grid of plots - one for each node type
    node_types = list(results.keys())
    n_plots = len(node_types)
    
    if n_plots == 0:
        return  # No data to plot
    
    # Determine grid size: max 2 columns
    n_cols = min(2, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    plt.figure(figsize=(n_cols * 6, n_rows * 5))
    
    # Color definition for memorized vs non-memorized
    colors = {
        'Memorized': '#FF9999',     # Light red
        'Non-memorized': '#66B2FF'  # Light blue
    }
    
    for i, node_type in enumerate(node_types):
        plt.subplot(n_rows, n_cols, i + 1)
        
        result = results[node_type]
        
        # Create DataFrame for plotting
        data = []
        
        # Add memorized nodes
        for val in result['memorized']['values']:
            data.append({'Group': 'Memorized', 'Weighted Heterophily': val})
        
        # Add non-memorized nodes
        for val in result['non_memorized']['values']:
            data.append({'Group': 'Non-memorized', 'Weighted Heterophily': val})
        
        df = pd.DataFrame(data)
        
        # Create the box plot
        if not df.empty:
            sns.boxplot(x='Group', y='Weighted Heterophily', data=df, 
                       palette=colors)
            
            # Add individual points for better visualization
            sns.stripplot(x='Group', y='Weighted Heterophily', data=df, 
                         size=4, color='.3', alpha=0.6)
            
            # Get current axis limits
            y_min, y_max = plt.ylim()
            
            # Calculate better position for annotations - above the plot area
            annotation_y = y_max + (y_max - y_min) * 0.05
            
            # Add means to plot with improved positioning
            mem_mean = result['memorized']['mean']
            mem_count = result['memorized']['count']
            non_mem_mean = result['non_memorized']['mean']
            non_mem_count = result['non_memorized']['count']
            
            # Set new y-limit to make room for annotations
            plt.ylim(y_min, y_max + (y_max - y_min) * 0.2)
            
            # Add mean values as text above the boxplots
            plt.text(0, annotation_y, 
                    f"Mean: {mem_mean:.4f}\nN: {mem_count}", 
                    ha='center', va='bottom',
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
                    
            plt.text(1, annotation_y, 
                    f"Mean: {non_mem_mean:.4f}\nN: {non_mem_count}", 
                    ha='center', va='bottom',
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
            
            # Add p-value and effect size if available
            if result['stat_test']:
                p_val = result['stat_test']['p_value']
                effect = result['stat_test']['effect_size_interpretation']
                
                # Add significance markers
                sig_marker = ''
                if p_val < 0.001:
                    sig_marker = '***'
                elif p_val < 0.01:
                    sig_marker = '**'
                elif p_val < 0.05:
                    sig_marker = '*'
                    
                plt.title(f"{node_type.capitalize()} Nodes\np = {p_val:.4e} {sig_marker} (Effect: {effect})")
            else:
                plt.title(f"{node_type.capitalize()} Nodes")
                
            plt.grid(True, linestyle='--', alpha=0.7, axis='y')
            plt.ylabel('Weighted Label Heterophily')
    
    # Add global title
    similarity_text = 'Cosine Similarity' if similarity_measure == 'cosine' else 'Gaussian Similarity'
    plt.suptitle(f"Weighted Label Heterophily using {similarity_text}\n{title_suffix}", 
                fontsize=14, y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def run_weighted_heterophily_analysis(
    data,
    node_scores: Dict[str, Dict],
    nodes_dict: Dict[str, List[int]],
    save_dir: str,
    timestamp: str, 
    model_type: str,
    dataset_name: str,
    threshold: float = 0.5,
    similarity_measure: str = 'cosine',
    gaussian_sigma: float = 1.0,
    epsilon: float = 1e-6,
    device = None,
    logger = None
) -> Dict[str, Any]:
    """
    Run comprehensive weighted label heterophily analysis and create visualizations.
    
    Args:
        data: PyG Data object
        node_scores: Dictionary containing memorization scores
        nodes_dict: Dictionary mapping node types to lists of indices
        save_dir: Directory to save plots
        timestamp: Timestamp string for filenames
        model_type: Model type (e.g., 'gcn')
        dataset_name: Dataset name
        threshold: Threshold for determining memorized nodes
        similarity_measure: Method to compute feature similarity
        gaussian_sigma: Standard deviation if using gaussian similarity
        epsilon: Small value to prevent division by zero
        device: PyTorch device
        logger: Logger object
        
    Returns:
        Dictionary with analysis results
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    if logger:
        logger.info("\nPerforming Weighted Label Heterophily analysis...")
        logger.info(f"Using {similarity_measure} similarity measure")
        
    # Calculate weighted heterophily scores for all nodes
    weighted_scores = calculate_weighted_label_heterophily(
        data=data,
        similarity_measure=similarity_measure,
        gaussian_sigma=gaussian_sigma,
        epsilon=epsilon,
        device=device
    )
    
    # Analyze scores for each node type
    analysis_results = analyze_weighted_heterophily_scores(
        weighted_conflict_scores=weighted_scores,
        node_scores=node_scores,
        nodes_dict=nodes_dict,
        threshold=threshold
    )
    
    # Create plots
    plot_path = os.path.join(
        save_dir,
        f'weighted_heterophily_{similarity_measure}_{model_type}_{dataset_name}_{timestamp}.png'
    )
    
    title_suffix = f"Dataset: {dataset_name}, Model: {model_type}"
    
    plot_weighted_heterophily_comparison(
        results=analysis_results,
        save_path=plot_path,
        similarity_measure=similarity_measure,
        title_suffix=title_suffix
    )
    
    # Log results
    if logger:
        logger.info("\nWeighted Label Heterophily Analysis Results:")
        
        for node_type, result in analysis_results.items():
            logger.info(f"\n{node_type.capitalize()} Nodes:")
            logger.info(f"  Memorized nodes (n={result['memorized']['count']}): {result['memorized']['mean']:.4f} ± {result['memorized']['std']:.4f}")
            logger.info(f"  Non-memorized nodes (n={result['non_memorized']['count']}): {result['non_memorized']['mean']:.4f} ± {result['non_memorized']['std']:.4f}")
            
            if result['stat_test']:
                p_val = result['stat_test']['p_value']
                t_stat = result['stat_test']['t_statistic']
                effect = result['stat_test']['effect_size']
                effect_interp = result['stat_test']['effect_size_interpretation']
                
                logger.info(f"  Welch's t-test: t = {t_stat:.4f}, p = {p_val:.4e}")
                
                # Format p-value with significance markers
                sig_marker = ''
                if p_val < 0.001:
                    sig_marker = '***'
                elif p_val < 0.01:
                    sig_marker = '**'
                elif p_val < 0.05:
                    sig_marker = '*'
                    
                logger.info(f"  Statistical significance: {sig_marker}")
                logger.info(f"  Effect size (Cohen's d): {effect:.4f} ({effect_interp})")
        
        logger.info(f"\nPlot saved to: {plot_path}")
    
    return {
        'weighted_scores': weighted_scores.cpu().numpy(),
        'analysis': analysis_results
    }