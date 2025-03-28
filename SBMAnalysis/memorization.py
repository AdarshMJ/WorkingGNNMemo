import torch
import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def verify_node_access(node_idx: int, node_type: str, nodes_dict: Dict[str, List[int]]):
    """Verify that node belongs to the correct set"""
    # Check if valid node type
    valid_types = {'shared', 'candidate', 'independent', 'extra'}
    if node_type not in valid_types:
        raise ValueError(f"Invalid node type: {node_type}. Must be one of {valid_types}")
        
    # Check if node exists in the specified type's list
    if node_idx not in nodes_dict[node_type]:
        raise ValueError(f"Node {node_idx} is not in the {node_type} set")
    
    return True  # Return True if validation passes

def calculate_node_memorization_score(
    model_f,
    model_g,
    data,
    nodes_dict,
    device: torch.device = None,
    logger=None,
    num_passes: int = 1  # Number of forward passes to average over
) -> Dict[str, Dict]:
    """Calculate memorization scores using confidence differences between models"""
    model_f.eval()
    model_g.eval()
    
    if device is None:
        device = next(model_f.parameters()).device
    
    results = {}
    
    # Process each node type
    for node_type, nodes in nodes_dict.items():
        if node_type in ['val', 'test']:  # Skip validation and test nodes
            continue
            
        # Lists to store scores and predictions
        mem_scores = []
        f_confidences = []
        g_confidences = []
        correct_f = 0
        correct_g = 0
        
        all_scores = []
        
        # Run multiple passes and average if requested
        for _ in range(num_passes):
            # Get model predictions
            with torch.no_grad():
                out_f = torch.softmax(model_f(data.x.to(device), data.edge_index.to(device)), dim=1)
                out_g = torch.softmax(model_g(data.x.to(device), data.edge_index.to(device)), dim=1)
            
            # Calculate scores for each node
            for node_idx in nodes:
                true_label = data.y[node_idx].item()
                prob_f = out_f[node_idx, true_label].item()
                prob_g = out_g[node_idx, true_label].item()
                
                # Calculate memorization score as difference in confidence
                mem_score = prob_f - prob_g
                
                # Track predictions for accuracy
                pred_f = out_f[node_idx].argmax().item()
                pred_g = out_g[node_idx].argmax().item()
                
                if pred_f == true_label:
                    correct_f += 1
                if pred_g == true_label:
                    correct_g += 1
                
                all_scores.append({
                    'node_idx': node_idx,
                    'node_type': node_type,
                    'true_label': true_label,
                    'pred_f': pred_f,
                    'pred_g': pred_g,
                    'conf_f': prob_f,
                    'conf_g': prob_g,
                    'mem_score': mem_score
                })
                
                mem_scores.append(mem_score)
                f_confidences.append(prob_f)
                g_confidences.append(prob_g)
        
        # Skip node type if no valid scores were calculated
        if not mem_scores:
            if logger:
                logger.warning(f"No valid scores calculated for {node_type} nodes")
            continue
        
        # Calculate average scores
        avg_score = np.mean(mem_scores)
        accuracy_f = correct_f / (len(nodes) * num_passes) if nodes else 0
        accuracy_g = correct_g / (len(nodes) * num_passes) if nodes else 0
        
        # Calculate nodes above threshold (0.5)
        nodes_above_threshold = sum(1 for score in mem_scores if score > 0.5)
        percentage_above_threshold = (nodes_above_threshold / len(mem_scores)) * 100 if mem_scores else 0
        
        # Store results
        results[node_type] = {
            'mem_scores': mem_scores,
            'f_confidences': f_confidences,
            'g_confidences': g_confidences,
            'avg_score': avg_score,
            'accuracy_f': accuracy_f,
            'accuracy_g': accuracy_g,
            'nodes_above_threshold': nodes_above_threshold,
            'percentage_above_threshold': percentage_above_threshold,
            'raw_data': pd.DataFrame(all_scores)
        }
        
        if logger:
            logger.info(f"Node type: {node_type}")
            logger.info(f"  Average memorization score: {avg_score:.4f}")
            logger.info(f"  Model f accuracy: {accuracy_f:.4f}")
            logger.info(f"  Model g accuracy: {accuracy_g:.4f}")
            logger.info(f"  Average f confidence: {np.mean(f_confidences):.4f}")
            logger.info(f"  Average g confidence: {np.mean(g_confidences):.4f}")
            logger.info(f"  Nodes with mem score > 0.5: {nodes_above_threshold}/{len(mem_scores)} ({percentage_above_threshold:.1f}%)")
    
    return results

def plot_node_memorization_analysis(
    node_scores: Dict[str, Dict],
    save_path: str,
    title_suffix="",
    node_types_to_plot: List[str] = None
):
    """
    Plot node memorization analysis results based on confidence score differences
    Args:
        node_scores: Dictionary containing scores for each node type
        save_path: Path to save the plot
        title_suffix: Additional text to add to plot titles
        node_types_to_plot: List of node types to include in histogram (e.g., ['shared', 'candidate'])
                          If None, all node types will be plotted
    """
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Model confidence comparison scatter plot (only for candidate nodes)
    plt.subplot(2, 2, 1)
    
    # Get candidate node data
    if 'candidate' in node_scores:
        f_confidences = node_scores['candidate']['f_confidences']
        g_confidences = node_scores['candidate']['g_confidences']
        mem_scores = node_scores['candidate']['mem_scores']
        
        # Add y=x line in red
        min_val = min(min(f_confidences), min(g_confidences))
        max_val = max(max(f_confidences), max(g_confidences))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='y=x')
        
        # Create scatter plot with viridis colormap
        scatter = plt.scatter(f_confidences, g_confidences, 
                            c=mem_scores, cmap='viridis', 
                            alpha=0.6, s=50)
        plt.colorbar(scatter, label='Memorization Score')
    
    plt.xlabel('Model f Confidence')
    plt.ylabel('Model g Confidence')
    plt.title('Confidence Comparison (Candidate Nodes)')
    plt.legend()
    
    # Plot 2: Histogram for specified node types
    plt.subplot(2, 2, 2)

    # Color and label definitions
    colors = {'candidate': 'blue', 'independent': 'orange', 'extra': 'green', 'shared': 'red'}
    labels = {'candidate': '$S_C$', 'independent': '$S_I$', 'extra': '$S_E$', 'shared': '$S_S$'}

    # If no specific types are provided, plot all available types
    if node_types_to_plot is None:
        node_types_to_plot = list(node_scores.keys())
    
    # Plot histogram instead of KDE
    num_bins = 20
    threshold = 0.5
    
    # Create histogram with frequency counts
    for node_type in node_types_to_plot:
        if node_type in node_scores:
            scores = node_scores[node_type]['mem_scores']
            mean_score = node_scores[node_type]['avg_score']
            nodes_above = node_scores[node_type]['nodes_above_threshold']
            total_nodes = len(scores)
            percentage_above = (nodes_above / total_nodes) * 100
            
            # Plot histogram with frequency counts
            plt.hist(scores, bins=num_bins, alpha=0.5, color=colors[node_type],
                     label=f"{labels[node_type]} ({total_nodes} nodes, {nodes_above} > 0.5, {percentage_above:.1f}%)")
    
    # Add vertical line at the threshold
    plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Threshold = {threshold}')
    
    # Set up plot appearance
    plt.xlabel('Memorization Score (f - g confidence)')
    plt.ylabel('Frequency Count')
    title = 'Frequency Distribution of Memorization Scores'
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    
    # Plot 3: Model f confidence distribution
    plt.subplot(2, 2, 3)
    for node_type in node_types_to_plot:
        if node_type in node_scores:
            confidences = node_scores[node_type]['f_confidences']
            mean_conf = np.mean(confidences)
            
            plt.hist(confidences, bins=num_bins, alpha=0.5, color=colors[node_type],
                     label=f"{labels[node_type]} (mean={mean_conf:.3f})")
    
    plt.xlabel('Model f Confidence')
    plt.ylabel('Frequency Count')
    plt.title('Frequency Distribution of Model f Confidence Scores')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper left')
    
    # Plot 4: Model g confidence distribution
    plt.subplot(2, 2, 4)
    for node_type in node_types_to_plot:
        if node_type in node_scores:
            confidences = node_scores[node_type]['g_confidences']
            mean_conf = np.mean(confidences)
            
            plt.hist(confidences, bins=num_bins, alpha=0.5, color=colors[node_type],
                     label=f"{labels[node_type]} (mean={mean_conf:.3f})")
    
    plt.xlabel('Model g Confidence')
    plt.ylabel('Frequency Count')
    plt.title('Frequency Distribution of Model g Confidence Scores')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper left')

    # Add title suffix if provided
    if title_suffix:
        plt.suptitle(title_suffix, fontsize=14)
        plt.subplots_adjust(top=0.85)  # Make room for the suptitle

    # Save the complete figure with all subplots
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()