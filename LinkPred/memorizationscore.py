import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Dict, List
from torch_geometric.utils import negative_sampling

def get_edge_splits(train_data, val_data=None, test_data=None, split_ratios=(0.50, 0.25, 0.25)):
    """
    Split the positive and negative edges into shared, candidate and independent sets.
    
    Args:
        train_data: PyG data object with training edges
        val_data: Optional PyG data for validation
        test_data: Optional PyG data for test
        split_ratios: Tuple of (shared, candidate, independent) ratios
        
    Returns:
        Dictionary with edge splits
    """
    # Ensure the ratios sum to 1
    assert sum(split_ratios) == 1.0, "Split ratios must sum to 1.0"
    
    # Get training positive edges (edge_label_index where edge_label=1)
    pos_edge_indices = train_data.edge_label_index[:, train_data.edge_label == 1]
    
    # Generate negative edges for training
    num_neg = pos_edge_indices.size(1)
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index,
        num_nodes=train_data.num_nodes,
        num_neg_samples=num_neg,
        method='sparse')
    
    # Sort edges for determinism
    pos_edges = pos_edge_indices.t().cpu().numpy()
    neg_edges = neg_edge_index.t().cpu().numpy()
    
    # Calculate split sizes
    num_pos_edges = len(pos_edges)
    num_neg_edges = len(neg_edges)
    
    # Calculate split sizes for positive edges
    shared_size_pos = int(split_ratios[0] * num_pos_edges)
    candidate_size_pos = int(split_ratios[1] * num_pos_edges)
    # independent_size_pos will be the remainder
    
    # Calculate split sizes for negative edges
    shared_size_neg = int(split_ratios[0] * num_neg_edges)
    candidate_size_neg = int(split_ratios[1] * num_neg_edges)
    # independent_size_neg will be the remainder
    
    # Split positive edges
    shared_pos = pos_edges[:shared_size_pos]
    candidate_pos = pos_edges[shared_size_pos:shared_size_pos + candidate_size_pos]
    independent_pos = pos_edges[shared_size_pos + candidate_size_pos:]
    
    # Split negative edges
    shared_neg = neg_edges[:shared_size_neg]
    candidate_neg = neg_edges[shared_size_neg:shared_size_neg + candidate_size_neg]
    independent_neg = neg_edges[shared_size_neg + candidate_size_neg:]
    
    # Create edge dictionaries with both positive and negative edges
    edges_dict = {
        'shared': {
            'pos': shared_pos,
            'neg': shared_neg
        },
        'candidate': {
            'pos': candidate_pos,
            'neg': candidate_neg
        },
        'independent': {
            'pos': independent_pos,
            'neg': independent_neg
        }
    }
    
    # Add validation edges if provided
    if val_data is not None:
        val_pos_indices = val_data.edge_label_index[:, val_data.edge_label == 1]
        val_neg_indices = val_data.edge_label_index[:, val_data.edge_label == 0]
        
        edges_dict['val'] = {
            'pos': val_pos_indices.t().cpu().numpy(),
            'neg': val_neg_indices.t().cpu().numpy()
        }
    
    # Add test edges if provided
    if test_data is not None:
        test_pos_indices = test_data.edge_label_index[:, test_data.edge_label == 1]
        test_neg_indices = test_data.edge_label_index[:, test_data.edge_label == 0]
        
        # Use some test positive edges as "extra" edges - similar to using test nodes as extra nodes
        extra_size = min(len(candidate_pos), len(test_pos_indices.t().cpu().numpy()))
        extra_pos = test_pos_indices.t().cpu().numpy()[:extra_size]
        extra_neg = test_neg_indices.t().cpu().numpy()[:extra_size]
        
        edges_dict['test'] = {
            'pos': test_pos_indices.t().cpu().numpy(),
            'neg': test_neg_indices.t().cpu().numpy()
        }
        
        edges_dict['extra'] = {
            'pos': extra_pos,
            'neg': extra_neg
        }
    
    return edges_dict

def calculate_edge_memorization_score(
    model_f,
    model_g,
    data,
    edges_dict,
    device,
    logger=None,
    threshold=0.5  # Threshold for binary prediction (edge/non-edge)
):
    """
    Calculate memorization scores for edges using confidence differences between models.
    
    Args:
        model_f: Model trained on shared+candidate edges
        model_g: Model trained on shared+independent edges
        data: PyG data object with node features and graph structure
        edges_dict: Dictionary with edge splits
        device: Device to run computation on
        logger: Optional logger for printing results
        threshold: Threshold for binary prediction (default 0.5)
    
    Returns:
        Dictionary with memorization scores for each edge type
    """
    # Ensure models are in evaluation mode
    model_f.eval()
    model_g.eval()
    
    results = {}
    
    # Process each edge type
    for edge_type in ['shared', 'candidate', 'independent', 'extra']:
        if edge_type not in edges_dict:
            if logger:
                logger.info(f"Skipping {edge_type} edges - not found in edges_dict")
            continue
        
        # Create data structures to store results
        pos_mem_scores = []  # Memorization scores for positive edges
        neg_mem_scores = []  # Memorization scores for negative edges
        
        pos_f_confidences = []  # Model f confidences for positive edges
        pos_g_confidences = []  # Model g confidences for positive edges
        neg_f_confidences = []  # Model f confidences for negative edges
        neg_g_confidences = []  # Model g confidences for negative edges
        
        # Accuracy tracking
        pos_correct_f = 0  # Correct predictions by model f for positive edges
        pos_correct_g = 0  # Correct predictions by model g for positive edges
        neg_correct_f = 0  # Correct predictions by model f for negative edges
        neg_correct_g = 0  # Correct predictions by model g for negative edges
        
        all_scores = []  # All scores for detailed analysis
        
        # Get node embeddings for both models
        with torch.no_grad():
            z_f = model_f.encode(data.x.to(device), data.edge_index.to(device))
            z_g = model_g.encode(data.x.to(device), data.edge_index.to(device))
            
            # Process positive edges
            if len(edges_dict[edge_type]['pos']) > 0:
                pos_edges = torch.tensor(edges_dict[edge_type]['pos'], dtype=torch.long).t().to(device)
                
                # Get predictions from both models
                pos_pred_f = model_f.decode(z_f, pos_edges).sigmoid()
                pos_pred_g = model_g.decode(z_g, pos_edges).sigmoid()
                
                # Calculate memorization scores for each positive edge
                for i in range(pos_edges.size(1)):
                    # Get edge endpoints
                    src, dst = pos_edges[0, i].item(), pos_edges[1, i].item()
                    
                    # Get prediction confidences for positive label (edge exists)
                    conf_f = pos_pred_f[i].item()
                    conf_g = pos_pred_g[i].item()
                    
                    # Calculate memorization score as difference in confidence
                    mem_score = conf_f - conf_g
                    
                    # Track if predictions are correct (above threshold for positive edges)
                    if conf_f > threshold:
                        pos_correct_f += 1
                    if conf_g > threshold:
                        pos_correct_g += 1
                    
                    # Store results
                    pos_mem_scores.append(mem_score)
                    pos_f_confidences.append(conf_f)
                    pos_g_confidences.append(conf_g)
                    
                    # Store detailed information
                    all_scores.append({
                        'edge': (src, dst),
                        'edge_type': edge_type,
                        'label': 1,  # Positive edge
                        'conf_f': conf_f,
                        'conf_g': conf_g,
                        'mem_score': mem_score,
                        'pred_f': 1 if conf_f > threshold else 0,
                        'pred_g': 1 if conf_g > threshold else 0,
                    })
            
            # Process negative edges
            if len(edges_dict[edge_type]['neg']) > 0:
                neg_edges = torch.tensor(edges_dict[edge_type]['neg'], dtype=torch.long).t().to(device)
                
                # Get predictions from both models
                neg_pred_f = model_f.decode(z_f, neg_edges).sigmoid()
                neg_pred_g = model_g.decode(z_g, neg_edges).sigmoid()
                
                # Calculate memorization scores for each negative edge
                for i in range(neg_edges.size(1)):
                    # Get edge endpoints
                    src, dst = neg_edges[0, i].item(), neg_edges[1, i].item()
                    
                    # Get prediction confidences
                    conf_f = neg_pred_f[i].item()
                    conf_g = neg_pred_g[i].item()
                    
                    # For negative edges, confidence is (1 - prediction) since we want confidence in non-edge
                    inv_conf_f = 1.0 - conf_f
                    inv_conf_g = 1.0 - conf_g
                    
                    # Calculate memorization score for negative edges (confidence in non-edge)
                    mem_score = inv_conf_f - inv_conf_g
                    
                    # Track if predictions are correct (below threshold for negative edges)
                    if conf_f <= threshold:
                        neg_correct_f += 1
                    if conf_g <= threshold:
                        neg_correct_g += 1
                    
                    # Store results
                    neg_mem_scores.append(mem_score)
                    neg_f_confidences.append(inv_conf_f)  # Store inverted confidences for negative edges
                    neg_g_confidences.append(inv_conf_g)
                    
                    # Store detailed information
                    all_scores.append({
                        'edge': (src, dst),
                        'edge_type': edge_type,
                        'label': 0,  # Negative edge
                        'conf_f': inv_conf_f,  # Store inverted for consistency
                        'conf_g': inv_conf_g,
                        'mem_score': mem_score,
                        'pred_f': 0 if conf_f <= threshold else 1,
                        'pred_g': 0 if conf_g <= threshold else 1,
                    })
        
        # Calculate statistics
        num_pos_edges = len(pos_mem_scores)
        num_neg_edges = len(neg_mem_scores)
        
        # Calculate percentage of edges above threshold (0.5)
        pos_above_threshold = sum(1 for score in pos_mem_scores if score > 0.5)
        pos_percentage_above = (pos_above_threshold / num_pos_edges) * 100 if num_pos_edges > 0 else 0
        
        neg_above_threshold = sum(1 for score in neg_mem_scores if score > 0.5)
        neg_percentage_above = (neg_above_threshold / num_neg_edges) * 100 if num_neg_edges > 0 else 0
        
        # Calculate accuracies
        pos_acc_f = pos_correct_f / num_pos_edges if num_pos_edges > 0 else 0
        pos_acc_g = pos_correct_g / num_pos_edges if num_pos_edges > 0 else 0
        neg_acc_f = neg_correct_f / num_neg_edges if num_neg_edges > 0 else 0
        neg_acc_g = neg_correct_g / num_neg_edges if num_neg_edges > 0 else 0
        
        # Store results
        results[edge_type] = {
            'positive_edges': {
                'mem_scores': pos_mem_scores,
                'f_confidences': pos_f_confidences,
                'g_confidences': pos_g_confidences,
                'avg_score': np.mean(pos_mem_scores) if pos_mem_scores else 0,
                'accuracy_f': pos_acc_f,
                'accuracy_g': pos_acc_g,
                'above_threshold': pos_above_threshold,
                'percentage_above': pos_percentage_above,
                'count': num_pos_edges
            },
            'negative_edges': {
                'mem_scores': neg_mem_scores,
                'f_confidences': neg_f_confidences,
                'g_confidences': neg_g_confidences,
                'avg_score': np.mean(neg_mem_scores) if neg_mem_scores else 0,
                'accuracy_f': neg_acc_f,
                'accuracy_g': neg_acc_g,
                'above_threshold': neg_above_threshold,
                'percentage_above': neg_percentage_above,
                'count': num_neg_edges
            },
            'all_scores': pd.DataFrame(all_scores)
        }
        
        if logger:
            logger.info(f"\nEdge Type: {edge_type}")
            
            logger.info(f"\n  Positive Edges ({num_pos_edges} edges):")
            logger.info(f"    Average memorization score: {results[edge_type]['positive_edges']['avg_score']:.4f}")
            logger.info(f"    Edges with score > 0.5: {pos_above_threshold} ({pos_percentage_above:.1f}%)")
            logger.info(f"    Accuracy - Model f: {pos_acc_f:.4f}, Model g: {pos_acc_g:.4f}")
            
            logger.info(f"\n  Negative Edges ({num_neg_edges} edges):")
            logger.info(f"    Average memorization score: {results[edge_type]['negative_edges']['avg_score']:.4f}")
            logger.info(f"    Edges with score > 0.5: {neg_above_threshold} ({neg_percentage_above:.1f}%)")
            logger.info(f"    Accuracy - Model f: {neg_acc_f:.4f}, Model g: {neg_acc_g:.4f}")
    
    return results

def plot_edge_memorization_analysis(
    edge_scores: Dict,
    save_path: str,
    title_suffix="",
    edge_types_to_plot: List[str] = None
):
    """
    Plot edge memorization analysis results based on confidence score differences.
    
    Args:
        edge_scores: Dictionary containing scores for each edge type
        save_path: Path to save the plot
        title_suffix: Additional text to add to plot titles
        edge_types_to_plot: List of edge types to include (e.g., ['shared', 'candidate'])
    """
    # Extract base path and extension
    base_path, ext = os.path.splitext(save_path)
    if not ext:
        ext = '.png'  # Default extension if none provided
    
    # Color and label definitions
    colors = {
        'candidate': 'blue', 
        'independent': 'orange', 
        'extra': 'green', 
        'shared': 'red'
    }
    
    labels = {
        'candidate': '$E_C$', 
        'independent': '$E_I$', 
        'extra': '$E_E$', 
        'shared': '$E_S$'
    }
    
    # If no specific types are provided, plot all available types
    if edge_types_to_plot is None:
        edge_types_to_plot = list(edge_scores.keys())
    
    # Filter out edge types that don't exist in the data
    edge_types_to_plot = [et for et in edge_types_to_plot if et in edge_scores]
    
    # Set a maximum figure size to avoid image size issues
    max_fig_width = 15
    max_fig_height = 12
    
    num_bins = 20
    threshold = 0.5
    
    # Create a figure with 4 subplots in a 2x2 grid
    plt.figure(figsize=(max_fig_width, max_fig_height))
    
    # 1. Plot full histogram for positive edges
    plt.subplot(221)
    
    # Create histogram with frequency counts for positive edges (full range)
    for edge_type in edge_types_to_plot:
        scores = edge_scores[edge_type]['positive_edges']['mem_scores']
        edges_above = edge_scores[edge_type]['positive_edges']['above_threshold']
        total_edges = edge_scores[edge_type]['positive_edges']['count']
        percentage_above = edge_scores[edge_type]['positive_edges']['percentage_above']
        
        if total_edges > 0:  # Only plot if there are edges
            plt.hist(scores, bins=num_bins, alpha=0.5, color=colors[edge_type],
                     label=f"{labels[edge_type]} ({edges_above}/{total_edges}, {percentage_above:.1f}%)")
    
    # Add vertical line at the threshold
    plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Threshold = {threshold}')
    
    # Set up plot appearance
    plt.xlabel('Memorization Score (f - g confidence)')
    plt.ylabel('Frequency Count')
    plt.title('Positive Edges - Full Distribution')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    
    # 2. Plot zoomed histogram for positive edges with high scores
    plt.subplot(222)
    
    # Create filtered data for each edge type that only includes scores > 0.2
    for edge_type in edge_types_to_plot:
        all_scores = edge_scores[edge_type]['positive_edges']['mem_scores']
        # Filter scores greater than 0.2 to focus on higher values
        high_scores = [score for score in all_scores if score > 0.2]
        if high_scores:  # Only plot if we have high scores
            plt.hist(high_scores, bins=15, alpha=0.7, color=colors[edge_type],
                    label=f"{labels[edge_type]}")
            
            # Highlight scores > 0.5
            very_high_scores = [score for score in all_scores if score > 0.5]
            if very_high_scores:
                plt.hist(very_high_scores, bins=5, color=colors[edge_type], 
                         edgecolor='black', linewidth=1.5, alpha=0.9)
    
    # Add vertical line at the threshold
    plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Threshold = {threshold}')
    
    # Set up plot appearance
    plt.xlabel('Memorization Score (f - g confidence)')
    plt.ylabel('Frequency Count')
    plt.title('Positive Edges - Zoomed View (Scores > 0.2)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlim(0.2, 1.0)  # Zoom in to better see high scores
    plt.legend(loc='upper right')
    
    # 3. Plot full histogram for negative edges
    plt.subplot(223)
    
    # Create histogram with frequency counts for negative edges
    for edge_type in edge_types_to_plot:
        scores = edge_scores[edge_type]['negative_edges']['mem_scores']
        edges_above = edge_scores[edge_type]['negative_edges']['above_threshold']
        total_edges = edge_scores[edge_type]['negative_edges']['count']
        percentage_above = edge_scores[edge_type]['negative_edges']['percentage_above']
        
        if total_edges > 0:  # Only plot if there are edges
            plt.hist(scores, bins=num_bins, alpha=0.5, color=colors[edge_type],
                     label=f"{labels[edge_type]} ({edges_above}/{total_edges}, {percentage_above:.1f}%)")
    
    # Add vertical line at the threshold
    plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Threshold = {threshold}')
    
    # Set up plot appearance
    plt.xlabel('Memorization Score (f - g confidence)')
    plt.ylabel('Frequency Count')
    plt.title('Negative Edges - Full Distribution')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    
    # 4. Plot zoomed histogram for negative edges with high scores
    plt.subplot(224)
    
    # Create filtered data for each edge type that only includes scores > 0.2
    for edge_type in edge_types_to_plot:
        all_scores = edge_scores[edge_type]['negative_edges']['mem_scores']
        # Filter scores greater than 0.2 to focus on higher values
        high_scores = [score for score in all_scores if score > 0.2]
        if high_scores:  # Only plot if we have high scores
            plt.hist(high_scores, bins=15, alpha=0.7, color=colors[edge_type],
                    label=f"{labels[edge_type]}")
            
            # Highlight scores > 0.5
            very_high_scores = [score for score in all_scores if score > 0.5]
            if very_high_scores:
                plt.hist(very_high_scores, bins=5, color=colors[edge_type], 
                         edgecolor='black', linewidth=1.5, alpha=0.9)
    
    # Add vertical line at the threshold
    plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Threshold = {threshold}')
    
    # Set up plot appearance
    plt.xlabel('Memorization Score (f - g confidence)')
    plt.ylabel('Frequency Count')
    plt.title('Negative Edges - Zoomed View (Scores > 0.2)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlim(0.2, 1.0)  # Zoom in to better see high scores
    plt.legend(loc='upper right')
    
    if title_suffix:
        plt.suptitle(title_suffix, fontsize=14)
        # Leave more space at top
        plt.subplots_adjust(top=0.88, hspace=0.3)
    
    # Adjust layout before saving
    plt.tight_layout()
    plt.savefig(f"{base_path}_mem_score_distribution{ext}", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Process data for bar charts
    edge_types_labels = []
    pos_values = []
    pos_counts = []
    neg_values = []
    neg_counts = []
    
    for edge_type in edge_types_to_plot:
        edge_types_labels.append(labels[edge_type])
        pos_values.append(edge_scores[edge_type]['positive_edges']['percentage_above'])
        pos_counts.append(edge_scores[edge_type]['positive_edges']['above_threshold'])
        neg_values.append(edge_scores[edge_type]['negative_edges']['percentage_above'])
        neg_counts.append(edge_scores[edge_type]['negative_edges']['above_threshold'])
    
    # Check if there's data to plot for bar charts
    if edge_types_labels:
        # Create bar charts for high memorization scores
        plt.figure(figsize=(max_fig_width, 8))
        
        # First subplot: Bar chart for positive edges above threshold
        plt.subplot(121)
        
        bars = plt.bar(edge_types_labels, pos_values, color=[colors[et] for et in edge_types_to_plot])
        plt.ylabel('Percentage of Edges > 0.5 (%)')
        plt.title('Positive Edges with High Memorization Score')
        
        # Add count labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{pos_counts[i]}',
                    ha='center', va='bottom', rotation=0)
        
        # Second subplot: Bar chart for negative edges above threshold
        plt.subplot(122)
        
        bars = plt.bar(edge_types_labels, neg_values, color=[colors[et] for et in edge_types_to_plot])
        plt.ylabel('Percentage of Edges > 0.5 (%)')
        plt.title('Negative Edges with High Memorization Score')
        
        # Add count labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{neg_counts[i]}',
                    ha='center', va='bottom', rotation=0)
        
        if title_suffix:
            plt.suptitle(title_suffix, fontsize=14)
            plt.subplots_adjust(top=0.85)
        
        #plt.tight_layout()
        #try:
        #    plt.savefig(f"{base_path}_high_score_analysis{ext}", dpi=300, bbox_inches='tight')
        #except ValueError as e:
         #   print(f"Warning: Could not save high score analysis plot: {e}")
            # Try with a smaller DPI if the image is too large
          #  plt.savefig(f"{base_path}_high_score_analysis{ext}", dpi=100, bbox_inches='tight')
        #plt.close()
    
    # Create a separate figure for the table
    if edge_types_to_plot:  # Only create if we have edge types to show
        try:
            plt.figure(figsize=(10, min(6, 1 + len(edge_types_to_plot))))
            table_data = []
            table_columns = ['Edge Type', 'Positive Edges > 0.5', '% of Total', 'Negative Edges > 0.5', '% of Total']
            
            for edge_type in edge_types_to_plot:
                pos_above = edge_scores[edge_type]['positive_edges']['above_threshold']
                pos_total = edge_scores[edge_type]['positive_edges']['count']
                pos_pct = edge_scores[edge_type]['positive_edges']['percentage_above']
                
                neg_above = edge_scores[edge_type]['negative_edges']['above_threshold']
                neg_total = edge_scores[edge_type]['negative_edges']['count']
                neg_pct = edge_scores[edge_type]['negative_edges']['percentage_above']
                
                table_data.append([
                    labels[edge_type], 
                    f"{pos_above}/{pos_total}", 
                    f"{pos_pct:.1f}%",
                    f"{neg_above}/{neg_total}",
                    f"{neg_pct:.1f}%"
                ])
            
            # Create table
            plt.axis('off')  # Turn off axis
            table = plt.table(
                cellText=table_data,
                colLabels=table_columns,
                loc='center',
                cellLoc='center',
                bbox=[0.1, 0.1, 0.8, 0.65]  # Better placement
            )
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 1.5)
            plt.title('Summary of High Memorization Score Edges (> 0.5)', pad=20)
            
            if title_suffix:
                plt.suptitle(title_suffix, fontsize=14, y=0.95)
                
            plt.savefig(f"{base_path}_summary_table{ext}", dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Warning: Could not create summary table: {e}")
    
    # Create scatter plot visualization for confidence comparison
    if 'candidate' in edge_scores:
        try:
            plt.figure(figsize=(max_fig_width, 6))
            
            # Positive edges
            plt.subplot(121)
            f_confidences = edge_scores['candidate']['positive_edges']['f_confidences']
            g_confidences = edge_scores['candidate']['positive_edges']['g_confidences']
            mem_scores = edge_scores['candidate']['positive_edges']['mem_scores']
            
            if len(f_confidences) > 0:
                # Add y=x line in red
                min_val = min(min(f_confidences), min(g_confidences))
                max_val = max(max(f_confidences), max(g_confidences))
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='y=x')
                
                # Create scatter plot with viridis colormap
                # If too many points, use fewer points to avoid memory issues
                max_points = 1000
                if len(f_confidences) > max_points:
                    indices = np.random.choice(len(f_confidences), max_points, replace=False)
                    f_conf_sample = [f_confidences[i] for i in indices]
                    g_conf_sample = [g_confidences[i] for i in indices]
                    mem_score_sample = [mem_scores[i] for i in indices]
                    scatter = plt.scatter(f_conf_sample, g_conf_sample, 
                                        c=mem_score_sample, cmap='viridis', 
                                        alpha=0.6, s=50)
                else:
                    scatter = plt.scatter(f_confidences, g_confidences, 
                                        c=mem_scores, cmap='viridis', 
                                        alpha=0.6, s=50)
                
                plt.colorbar(scatter, label='Memorization Score')
                
                plt.xlabel('Model f Confidence')
                plt.ylabel('Model g Confidence')
                plt.title('Positive Edges - Candidate Nodes')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            # Negative edges
            plt.subplot(122)
            f_confidences = edge_scores['candidate']['negative_edges']['f_confidences']
            g_confidences = edge_scores['candidate']['negative_edges']['g_confidences']
            mem_scores = edge_scores['candidate']['negative_edges']['mem_scores']
            
            if len(f_confidences) > 0:
                # Add y=x line in red
                min_val = min(min(f_confidences), min(g_confidences))
                max_val = max(max(f_confidences), max(g_confidences))
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='y=x')
                
                # Create scatter plot with viridis colormap
                # If too many points, use fewer points to avoid memory issues
                max_points = 1000
                if len(f_confidences) > max_points:
                    indices = np.random.choice(len(f_confidences), max_points, replace=False)
                    f_conf_sample = [f_confidences[i] for i in indices]
                    g_conf_sample = [g_confidences[i] for i in indices]
                    mem_score_sample = [mem_scores[i] for i in indices]
                    scatter = plt.scatter(f_conf_sample, g_conf_sample, 
                                        c=mem_score_sample, cmap='viridis', 
                                        alpha=0.6, s=50)
                else:
                    scatter = plt.scatter(f_confidences, g_confidences, 
                                        c=mem_scores, cmap='viridis', 
                                        alpha=0.6, s=50)
                
                plt.colorbar(scatter, label='Memorization Score')
                
                plt.xlabel('Model f Confidence')
                plt.ylabel('Model g Confidence')
                plt.title('Negative Edges - Candidate Nodes')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            if title_suffix:
                plt.suptitle(title_suffix, fontsize=14)
                plt.subplots_adjust(top=0.85)
            
            plt.tight_layout()
            plt.savefig(f"{base_path}_confidence_comparison{ext}", dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Warning: Could not create confidence comparison plot: {e}")
    
    return f"{base_path}_mem_score_distribution{ext}"