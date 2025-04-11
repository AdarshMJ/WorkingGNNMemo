import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional
import os

def extract_layer_representations(
    model: torch.nn.Module,
    data,
    device: torch.device,
    first_layer_idx: int = 0,
    return_logits: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Extract representations after the first layer and final hidden layer.
    
    Args:
        model: The trained GNN model
        data: PyG Data object containing node features and edge indices
        device: PyTorch device
        first_layer_idx: Index of the first layer to extract (default: 0)
        return_logits: Whether to also return the final logits
        
    Returns:
        Dictionary containing 'first_layer', 'last_hidden', and optionally 'logits'
    """
    model.eval()
    with torch.no_grad():
        # Extract first layer representation
        logits, first_layer_repr = model(data.x.to(device), data.edge_index.to(device), 
                                         return_node_emb=True, embedding_layer=first_layer_idx)
        
        # Extract last hidden layer representation
        # This is the last representation before the final classification layer
        _, last_hidden_repr = model(data.x.to(device), data.edge_index.to(device), 
                                    return_node_emb=True)
        
        result = {
            'first_layer': first_layer_repr,
            'last_hidden': last_hidden_repr
        }
        
        if return_logits:
            result['logits'] = logits
            
        return result

def calculate_cosine_distance_node_wise(
    first_repr: torch.Tensor,
    last_repr: torch.Tensor
) -> torch.Tensor:
    """
    Calculate cosine distance between first and last layer representations for each node.
    
    Args:
        first_repr: Tensor of shape [num_nodes, hidden_dim1]
        last_repr: Tensor of shape [num_nodes, hidden_dim2]
        
    Returns:
        Tensor of shape [num_nodes] containing cosine distances
    """
    distances = torch.zeros(first_repr.shape[0], device=first_repr.device)
    
    for i in range(first_repr.shape[0]):
        h1_i = first_repr[i].unsqueeze(0)  # Shape: [1, hidden_dim1]
        z_i = last_repr[i].unsqueeze(0)    # Shape: [1, hidden_dim2]
        
        # Calculate cosine distance: 1 - cosine_similarity
        distances[i] = 1.0 - F.cosine_similarity(h1_i, z_i)
    
    return distances

def calculate_class_average_divergence(
    divergences: torch.Tensor,
    labels: torch.Tensor,
    indices: List[int]
) -> Dict[int, float]:
    """
    Calculate average divergence per class based on specified indices.
    
    Args:
        divergences: Tensor of shape [num_nodes] with divergence scores
        labels: Tensor of shape [num_nodes] with class labels
        indices: List of indices to consider (e.g., training set)
        
    Returns:
        Dictionary mapping class IDs to average divergence values
    """
    class_avg = {}
    unique_classes = torch.unique(labels[indices]).cpu().numpy()
    
    for cls in unique_classes:
        # Find indices for this class
        cls_indices = [i for i in indices if labels[i].item() == cls]
        
        # Calculate average divergence if there are examples
        if cls_indices:
            avg_div = divergences[cls_indices].mean().item()
            class_avg[int(cls)] = avg_div
    
    return class_avg

def calculate_relative_divergence(
    divergences: torch.Tensor,
    class_avg_divergence: Dict[int, float],
    labels: torch.Tensor,
    epsilon: float = 1e-6
) -> torch.Tensor:
    """
    Calculate relative divergence by dividing each node's divergence 
    by the average divergence of its class.
    
    Args:
        divergences: Tensor of shape [num_nodes] with divergence scores
        class_avg_divergence: Dictionary mapping class IDs to average divergence
        labels: Tensor of shape [num_nodes] with class labels
        epsilon: Small value to avoid division by zero
        
    Returns:
        Tensor of shape [num_nodes] containing relative divergences
    """
    relative_divs = torch.zeros_like(divergences)
    
    for i in range(divergences.shape[0]):
        label = int(labels[i].item())
        if label in class_avg_divergence:
            avg_div = class_avg_divergence[label]
            # Avoid division by zero
            if abs(avg_div) < epsilon:
                relative_divs[i] = float('nan')  # Mark as undefined
            else:
                relative_divs[i] = divergences[i].item() / avg_div
        else:
            relative_divs[i] = float('nan')  # No average for this class
    
    return relative_divs

def analyze_feature_divergence(
    model_f: torch.nn.Module,
    data,
    nodes_dict: Dict[str, List[int]],
    node_scores: Dict[str, Dict],
    threshold: float = 0.5,
    device: torch.device = None,
    first_layer_idx: int = 0
) -> Dict:
    """
    Analyze feature divergence between first and last layer for nodes based on memorization.
    
    Args:
        model_f: The trained GNN model
        data: PyG Data object containing node features, edge indices and labels
        nodes_dict: Dictionary mapping node types to lists of indices
        node_scores: Dictionary containing memorization scores
        threshold: Threshold for determining memorized nodes
        device: PyTorch device
        first_layer_idx: Index of the first layer to extract (default: 0)
        
    Returns:
        Dictionary with analysis results
    """
    if device is None:
        device = next(model_f.parameters()).device
    
    # Extract candidate nodes and training indices
    candidate_nodes = nodes_dict['candidate']
    train_indices_f = nodes_dict['shared'] + candidate_nodes
    
    # 1. Extract intermediate representations
    representations = extract_layer_representations(
        model_f, data, device, first_layer_idx=first_layer_idx
    )
    first_layer_repr = representations['first_layer']
    last_hidden_repr = representations['last_hidden']
    
    # 2. Calculate node-wise divergence (cosine distance)
    node_divergences = calculate_cosine_distance_node_wise(
        first_layer_repr, last_hidden_repr
    )
    
    # 3. Calculate class-average divergence for the training set
    class_avg_divergence = calculate_class_average_divergence(
        node_divergences, data.y, train_indices_f
    )
    
    # 4. Calculate relative divergence
    relative_divergences = calculate_relative_divergence(
        node_divergences, class_avg_divergence, data.y
    )
    
    # 5. Analyze candidate nodes
    # Get candidate nodes' memorization scores
    candidate_scores = []
    for idx in candidate_nodes:
        for node_info in node_scores['candidate']['raw_data'].itertuples():
            if node_info.node_idx == idx:
                candidate_scores.append(node_info.mem_score)
                break
    
    candidate_scores = np.array(candidate_scores)
    
    # Separate memorized and non-memorized candidate nodes
    memorized_mask = candidate_scores > threshold
    non_memorized_mask = ~memorized_mask
    
    memorized_indices = [candidate_nodes[i] for i in range(len(candidate_nodes)) if memorized_mask[i]]
    non_memorized_indices = [candidate_nodes[i] for i in range(len(candidate_nodes)) if non_memorized_mask[i]]
    
    # Extract relative divergence scores for each group
    rel_div_memorized = relative_divergences[memorized_indices].cpu().numpy()
    rel_div_non_memorized = relative_divergences[non_memorized_indices].cpu().numpy()
    
    # Remove NaN values
    rel_div_memorized = rel_div_memorized[~np.isnan(rel_div_memorized)]
    rel_div_non_memorized = rel_div_non_memorized[~np.isnan(rel_div_non_memorized)]
    
    # 6. Statistical analysis
    stats_result = {
        'memorized': {
            'count': len(rel_div_memorized),
            'mean': np.mean(rel_div_memorized),
            'std': np.std(rel_div_memorized),
            'values': rel_div_memorized
        },
        'non_memorized': {
            'count': len(rel_div_non_memorized),
            'mean': np.mean(rel_div_non_memorized),
            'std': np.std(rel_div_non_memorized),
            'values': rel_div_non_memorized
        }
    }
    
    # Perform Welch's t-test if both groups have data
    if len(rel_div_memorized) > 0 and len(rel_div_non_memorized) > 0:
        t_stat, p_val = stats.ttest_ind(
            rel_div_memorized, 
            rel_div_non_memorized, 
            equal_var=False
        )
        
        # Calculate effect size (Cohen's d)
        mean_diff = np.mean(rel_div_memorized) - np.mean(rel_div_non_memorized)
        pooled_std = np.sqrt(
            (np.std(rel_div_memorized)**2 + np.std(rel_div_non_memorized)**2) / 2
        )
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
        
        stats_result['test'] = {
            't_statistic': t_stat,
            'p_value': p_val,
            'effect_size': effect_size,
            'effect_size_interpretation': effect_size_interp
        }
    
    # Collect all results
    result = {
        'node_divergences': node_divergences.cpu().numpy(),
        'relative_divergences': relative_divergences.cpu().numpy(),
        'class_avg_divergence': class_avg_divergence,
        'stats': stats_result
    }
    
    return result

def plot_feature_divergence_comparison(
    results: Dict,
    save_path: str,
    title_suffix: str = ""
) -> None:
    """
    Create visualization comparing feature divergence between memorized and non-memorized nodes.
    
    Args:
        results: Results dictionary from analyze_feature_divergence
        save_path: Path to save the plot
        title_suffix: Additional text to add to plot title
    """
    plt.figure(figsize=(10, 6))
    
    stats = results['stats']
    
    # Create a DataFrame for seaborn
    data = []
    
    # Add memorized nodes
    for value in stats['memorized']['values']:
        data.append({'Group': 'Memorized', 'Relative Divergence': value})
    
    # Add non-memorized nodes
    for value in stats['non_memorized']['values']:
        data.append({'Group': 'Non-memorized', 'Relative Divergence': value})
    
    df = pd.DataFrame(data)
    
    # Create the box plot
    sns.boxplot(x='Group', y='Relative Divergence', data=df)
    
    # Add individual points for better visualization
    sns.stripplot(x='Group', y='Relative Divergence', data=df, 
                 size=4, color='.3', alpha=0.6)
    
    # Set y-axis to logarithmic scale if the range is large
    if (df['Relative Divergence'].max() / df['Relative Divergence'].min() > 10) and \
       (df['Relative Divergence'].min() > 0):
        plt.yscale('log')
    
    # Add statistics as text annotations
    mem_mean = stats['memorized']['mean']
    non_mem_mean = stats['non_memorized']['mean']
    
    plt.text(0, plt.ylim()[1]*0.9, f"Mean: {mem_mean:.4f}\nN: {stats['memorized']['count']}", 
             ha='center', va='top')
    plt.text(1, plt.ylim()[1]*0.9, f"Mean: {non_mem_mean:.4f}\nN: {stats['non_memorized']['count']}", 
             ha='center', va='top')
    
    # Add p-value if statistical test was performed
    if 'test' in stats:
        p_val = stats['test']['p_value']
        effect = stats['test']['effect_size_interpretation']
        
        # Add stars for significance
        sig_marker = ''
        if p_val < 0.001:
            sig_marker = '***'
        elif p_val < 0.01:
            sig_marker = '**'
        elif p_val < 0.05:
            sig_marker = '*'
        
        plt.title(f"Feature Divergence Comparison\np = {p_val:.4e} {sig_marker} (Effect: {effect})")
    else:
        plt.title("Feature Divergence Comparison")
    
    if title_suffix:
        plt.suptitle(title_suffix, fontsize=10)
        plt.subplots_adjust(top=0.85)
    
    plt.ylabel('Relative Feature Divergence\n(First Layer vs. Last Hidden Layer)')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Save plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_divergence_vs_memorization(
    results: Dict,
    node_scores: Dict,
    candidate_indices: List[int],
    save_path: str,
    title_suffix: str = ""
) -> None:
    """
    Create scatter plot of feature divergence vs memorization score.
    
    Args:
        results: Results dictionary from analyze_feature_divergence
        node_scores: Dictionary containing memorization scores
        candidate_indices: List of indices for candidate nodes
        save_path: Path to save the plot
        title_suffix: Additional text to add to plot title
    """
    plt.figure(figsize=(10, 6))
    
    # Extract data
    relative_divergences = results['relative_divergences']
    
    # Get memorization scores for candidate nodes
    candidate_scores = []
    candidate_divs = []
    
    for idx in candidate_indices:
        for node_info in node_scores['candidate']['raw_data'].itertuples():
            if node_info.node_idx == idx:
                mem_score = node_info.mem_score
                rel_div = relative_divergences[idx]
                
                # Skip NaN values
                if not np.isnan(rel_div):
                    candidate_scores.append(mem_score)
                    candidate_divs.append(rel_div)
                break
    
    # Create scatter plot
    plt.scatter(candidate_scores, candidate_divs, alpha=0.7, s=30, c='blue', edgecolors='black')
    
    # Add trend line
    if len(candidate_scores) > 1:
        z = np.polyfit(candidate_scores, candidate_divs, 1)
        p = np.poly1d(z)
        plt.plot(
            sorted(candidate_scores), 
            p(sorted(candidate_scores)), 
            "r--", 
            alpha=0.8,
            label=f"Trend: y = {z[0]:.4f}x + {z[1]:.4f}"
        )
        
        # Calculate correlation
        correlation, p_value = stats.pearsonr(candidate_scores, candidate_divs)
        plt.text(
            0.05, 0.95, 
            f"Correlation: {correlation:.4f}\nP-value: {p_value:.4e}", 
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', alpha=0.1)
        )
    
    # Add vertical line at threshold 0.5
    plt.axvline(x=0.5, color='k', linestyle='--', alpha=0.7, label='Threshold = 0.5')
    
    plt.xlabel('Memorization Score')
    plt.ylabel('Relative Feature Divergence')
    plt.title('Feature Divergence vs. Memorization Score')
    
    if title_suffix:
        plt.suptitle(title_suffix, fontsize=10)
        plt.subplots_adjust(top=0.85)
    
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='best')
    
    # Save plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_node_category_divergence_comparison(
    relative_divergences: np.ndarray,
    nodes_dict: Dict[str, List[int]],
    node_scores: Dict[str, Dict],
    save_path: str,
    threshold: float = 0.5,
    title_suffix: str = ""
) -> None:
    """
    Create box plots comparing relative divergence between memorized and non-memorized nodes for each node category.
    
    Args:
        relative_divergences: Array of relative divergence values for all nodes
        nodes_dict: Dictionary mapping node types to lists of indices
        node_scores: Dictionary containing memorization scores
        save_path: Path to save the plot
        threshold: Threshold for determining memorized nodes
        title_suffix: Additional text to add to plot title
    """
    # Create a figure with a grid of subplots - one for each node category
    node_types = ['candidate', 'shared', 'independent', 'extra']
    # Filter only available node types
    available_types = [nt for nt in node_types if nt in node_scores]
    
    if not available_types:
        return  # No data to plot
    
    # Determine grid size: max 2 columns
    n_plots = len(available_types)
    n_cols = min(2, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    plt.figure(figsize=(n_cols * 6, n_rows * 5))
    
    # Color definition
    colors = {
        'Memorized': '#FF9999',     # Light red
        'Non-memorized': '#66B2FF'  # Light blue
    }
    
    # Collect all data for overall statistics
    all_data = pd.DataFrame()
    
    # Create a subplot for each node type
    for i, node_type in enumerate(available_types):
        plt.subplot(n_rows, n_cols, i + 1)
        
        # Get node indices and memorization scores for this category
        if node_type not in nodes_dict or node_type not in node_scores:
            continue
        
        node_indices = nodes_dict[node_type]
        node_data = node_scores[node_type]['raw_data']
        
        # Create DataFrame for this node type
        plot_data = []
        
        # Process each node
        for idx in node_indices:
            # Get mem score for this node
            mem_score = None
            for node_info in node_data.itertuples():
                if node_info.node_idx == idx:
                    mem_score = node_info.mem_score
                    break
                    
            if mem_score is None:
                continue
                
            # Get relative divergence
            rel_div = relative_divergences[idx]
            if np.isnan(rel_div):
                continue
                
            # Determine if memorized
            group = 'Memorized' if mem_score > threshold else 'Non-memorized'
            
            plot_data.append({
                'Node Type': node_type.capitalize(),
                'Group': group,
                'Relative Divergence': rel_div,
                'Memorization Score': mem_score
            })
        
        # Skip if no data for this node type
        if not plot_data:
            continue
            
        # Create DataFrame and append to overall data
        df = pd.DataFrame(plot_data)
        all_data = pd.concat([all_data, df], ignore_index=True)
        
        # Create box plot
        bp = sns.boxplot(x='Group', y='Relative Divergence', data=df)
        
        # Add strip plot on top for individual points
        sns.stripplot(x='Group', y='Relative Divergence', data=df, 
                     size=4, color='.3', alpha=0.6, jitter=True)
        
        # Calculate statistics
        mem_data = df[df['Group'] == 'Memorized']['Relative Divergence']
        non_mem_data = df[df['Group'] == 'Non-memorized']['Relative Divergence']
        
        # Add statistics as text
        mem_mean = mem_data.mean() if not mem_data.empty else float('nan')
        non_mem_mean = non_mem_data.mean() if not non_mem_data.empty else float('nan')
        
        # Perform t-test if enough data
        if len(mem_data) > 1 and len(non_mem_data) > 1:
            t_stat, p_val = stats.ttest_ind(mem_data, non_mem_data, equal_var=False)
            
            # Add significance marker
            sig_marker = ''
            if p_val < 0.001:
                sig_marker = '***'
            elif p_val < 0.01:
                sig_marker = '**'
            elif p_val < 0.05:
                sig_marker = '*'
                
            plt.title(f"{node_type.capitalize()} Nodes\np = {p_val:.4e} {sig_marker}")
        else:
            plt.title(f"{node_type.capitalize()} Nodes")
        
        # Add mean values as text
        if len(mem_data) > 0:
            plt.text(0, plt.ylim()[1]*0.9, f"Mean: {mem_mean:.4f}\nN: {len(mem_data)}", 
                    ha='center', va='top')
        if len(non_mem_data) > 0:
            plt.text(1, plt.ylim()[1]*0.9, f"Mean: {non_mem_mean:.4f}\nN: {len(non_mem_data)}", 
                    ha='center', va='top')
        
        plt.ylabel('Relative Feature Divergence')
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    plt.tight_layout()
    
    # Add global title if provided
    if title_suffix:
        plt.suptitle(title_suffix, fontsize=14, y=1.05)
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def run_feature_divergence_analysis(
    model_f: torch.nn.Module,
    data,
    nodes_dict: Dict[str, List[int]],
    node_scores: Dict[str, Dict],
    save_dir: str,
    timestamp: str,
    model_type: str,
    dataset_name: str,
    threshold: float = 0.5,
    device: torch.device = None,
    logger = None
) -> Dict:
    """
    Run comprehensive feature divergence analysis and create visualizations.
    
    Args:
        model_f: The trained GNN model
        data: PyG Data object
        nodes_dict: Dictionary mapping node types to lists of indices
        node_scores: Dictionary containing memorization scores
        save_dir: Directory to save plots
        timestamp: Timestamp string to include in filenames
        model_type: Model type string (e.g., 'gcn')
        dataset_name: Dataset name string
        threshold: Threshold for determining memorized nodes
        device: PyTorch device
        logger: Logger object for printing results
        
    Returns:
        Dictionary with analysis results
    """
    if logger:
        logger.info("\nPerforming Feature vs. Aggregated Feature Divergence analysis...")
    
    # Run main analysis
    results = analyze_feature_divergence(
        model_f=model_f,
        data=data,
        nodes_dict=nodes_dict,
        node_scores=node_scores,
        threshold=threshold,
        device=device
    )
    
    # Create visualizations
    plot_path_1 = os.path.join(
        save_dir, 
        f'feature_divergence_comparison_{model_type}_{dataset_name}_{timestamp}.png'
    )
    
    plot_path_2 = os.path.join(
        save_dir,
        f'feature_divergence_vs_memorization_{model_type}_{dataset_name}_{timestamp}.png'
    )
    
    plot_path_3 = os.path.join(
        save_dir,
        f'node_category_divergence_comparison_{model_type}_{dataset_name}_{timestamp}.png'
    )
    
    title_suffix = f"Dataset: {dataset_name}, Model: {model_type}"
    
    plot_feature_divergence_comparison(
        results=results,
        save_path=plot_path_1,
        title_suffix=title_suffix
    )
    
    plot_divergence_vs_memorization(
        results=results,
        node_scores=node_scores,
        candidate_indices=nodes_dict['candidate'],
        save_path=plot_path_2,
        title_suffix=title_suffix
    )
    
    plot_node_category_divergence_comparison(
        relative_divergences=results['relative_divergences'],
        nodes_dict=nodes_dict,
        node_scores=node_scores,
        save_path=plot_path_3,
        threshold=threshold,
        title_suffix=title_suffix
    )
    
    # Log results if logger is provided
    if logger:
        stats = results['stats']
        
        logger.info("\nFeature Divergence Analysis Results:")
        logger.info(f"Memorized nodes (n={stats['memorized']['count']}):")
        logger.info(f"  Mean relative divergence: {stats['memorized']['mean']:.4f}")
        logger.info(f"  Standard deviation: {stats['memorized']['std']:.4f}")
        
        logger.info(f"\nNon-memorized nodes (n={stats['non_memorized']['count']}):")
        logger.info(f"  Mean relative divergence: {stats['non_memorized']['mean']:.4f}")
        logger.info(f"  Standard deviation: {stats['non_memorized']['std']:.4f}")
        
        if 'test' in stats:
            logger.info("\nStatistical Test Results:")
            logger.info(f"  Welch's t-test: t = {stats['test']['t_statistic']:.4f}, p = {stats['test']['p_value']:.4e}")
            
            # Format p-value with significance markers
            p_val = stats['test']['p_value']
            sig_marker = ''
            if p_val < 0.001:
                sig_marker = '***'
            elif p_val < 0.01:
                sig_marker = '**'
            elif p_val < 0.05:
                sig_marker = '*'
                
            logger.info(f"  Statistical significance: {sig_marker}")
            logger.info(f"  Effect size (Cohen's d): {stats['test']['effect_size']:.4f} ({stats['test']['effect_size_interpretation']})")
        
        logger.info(f"\nPlots saved to:")
        logger.info(f"  - {plot_path_1}")
        logger.info(f"  - {plot_path_2}")
        logger.info(f"  - {plot_path_3}")
    
    return results