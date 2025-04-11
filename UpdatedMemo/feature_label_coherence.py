import torch
import numpy as np
import networkx as nx
from torch_geometric.utils import k_hop_subgraph
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
import matplotlib.patches as mpatches
from scipy.stats import mannwhitneyu
from nodeli import calculate_h_adj
from nli_analysis import calculate_local_nli, calculate_dataset_homophily
import os
import warnings
warnings.filterwarnings('ignore')

def calculate_feature_divergence(data, node_idx):
    """Calculate how a node's features diverge from expected features for its label"""
    # Get node's features and label
    node_features = data.x[node_idx].cpu().numpy()
    node_label = data.y[node_idx].item()
    
    # Get average features for this label (excluding the node itself)
    same_label_mask = (data.y == node_label) & (torch.arange(data.num_nodes) != node_idx)
    same_label_nodes = same_label_mask.nonzero().flatten()
    
    if len(same_label_nodes) == 0:
        return None  # Can't calculate divergence if there are no other nodes with same label
        
    expected_features = data.x[same_label_nodes].mean(dim=0).cpu().numpy()
    
    # Calculate Jensen-Shannon divergence
    try:
        # Normalize to probability distributions
        node_features_norm = node_features / (node_features.sum() + 1e-10)
        expected_features_norm = expected_features / (expected_features.sum() + 1e-10)
        
        js_div = jensenshannon(node_features_norm, expected_features_norm)
        
        # If result is nan, use an alternate calculation
        if np.isnan(js_div):
            # Calculate cosine similarity instead
            cosine_sim = np.dot(node_features, expected_features) / (
                np.linalg.norm(node_features) * np.linalg.norm(expected_features) + 1e-10)
            # Convert similarity to a distance in [0,1]
            js_div = 1 - cosine_sim
    except:
        js_div = None
    
    return js_div

def calculate_flc_score(data, node_idx, k_hops=2, beta=0.6, gamma=0.4):
    """
    Calculate Feature-Label Coherence score.
    
    Args:
        data: PyG data object
        node_idx: Index of node to analyze
        k_hops: Number of hops for local NLI calculation
        beta: Weight for NLI component (default: 0.6)
        gamma: Weight for feature divergence component (default: 0.4)
    
    Returns:
        FLC score (higher means better coherence, lower means more likely to require memorization)
    """
    # Calculate local NLI
    nli_score, _, _ = calculate_local_nli(data, node_idx, k_hops)
    
    # Calculate feature divergence
    js_div = calculate_feature_divergence(data, node_idx)
    
    if None in (nli_score, js_div):
        return None
    
    # Higher NLI is good, higher divergence is bad
    flc_score = (beta * nli_score) - (gamma * js_div)
    
    return flc_score

def analyze_memorization_vs_flc(data, node_scores, k_hops=2, beta=0.6, gamma=0.4):
    """Analyze relationship between memorization scores and FLC metric"""
    results = {}
    
    # Calculate dataset homophily for context
    homophily_ratio = calculate_dataset_homophily(data)
    
    results['dataset_info'] = {
        'homophily_ratio': homophily_ratio,
        'is_homophilic': homophily_ratio > 0.5
    }
    
    for node_type in ['shared', 'candidate', 'independent', 'extra']:
        if node_type not in node_scores:
            continue
            
        nodes_data = node_scores[node_type]['raw_data']
        memorized_nodes = nodes_data[nodes_data['mem_score'] > 0.5]['node_idx'].tolist()
        non_memorized_nodes = nodes_data[nodes_data['mem_score'] <= 0.5]['node_idx'].tolist()
        
        # Calculate FLC scores for both groups
        memorized_flc = []
        non_memorized_flc = []
        
        # Process memorized nodes
        for node in memorized_nodes:
            flc_score = calculate_flc_score(data, node, k_hops, beta, gamma)
            if flc_score is not None:
                memorized_flc.append({
                    'flc_score': flc_score,
                    'node_idx': node
                })
        
        # Process non-memorized nodes
        for node in non_memorized_nodes:
            flc_score = calculate_flc_score(data, node, k_hops, beta, gamma)
            if flc_score is not None:
                non_memorized_flc.append({
                    'flc_score': flc_score,
                    'node_idx': node
                })
        
        # Store results
        results[node_type] = {
            'memorized': memorized_flc,
            'non_memorized': non_memorized_flc
        }
        
    return results

def plot_memorization_flc_comparison(results, save_path, k_hops):
    """
    Create separate visualization files comparing FLC scores between memorized and non-memorized nodes
    for each node type
    """
    # Extract base path and extension
    base_path, ext = os.path.splitext(save_path)
    if not ext:
        ext = '.png'  # Default extension if none provided
    
    # Get dataset homophily information
    dataset_info = results.pop('dataset_info', {'homophily_ratio': None, 'is_homophilic': None})
    homophily_ratio = dataset_info.get('homophily_ratio')
    is_homophilic = dataset_info.get('is_homophilic')
    
    # Get node types to plot
    node_types = [nt for nt in ['shared', 'candidate', 'independent', 'extra'] if nt in results]
    
    if len(node_types) == 0:
        print("Warning: No valid node types found in results")
        return pd.DataFrame()
    
    # Color scheme
    colors = {
        'Memorized': '#FF9999',     # Light red
        'Non-memorized': '#66B2FF'  # Light blue
    }
    
    summary_data = []
    
    # Create separate plot for each node type
    for node_type in node_types:
        data = results[node_type]
        
        # Create figure for this node type
        plt.figure(figsize=(8, 6))
        
        # Prepare data for plotting
        mem_scores = [item['flc_score'] for item in data['memorized']]
        non_mem_scores = [item['flc_score'] for item in data['non_memorized']]
        
        if len(mem_scores) == 0 and len(non_mem_scores) == 0:
            plt.text(0.5, 0.5, f'No data available for {node_type} nodes',
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title(f'{node_type.capitalize()} Nodes - FLC Distribution')
            plt.tight_layout()
            # Save plot for this node type
            node_path = f"{base_path}_{node_type}_flc{ext}"
            plt.savefig(node_path, dpi=300, bbox_inches='tight')
            plt.close()
            continue
            
        # Calculate statistics
        if len(mem_scores) > 0 and len(non_mem_scores) > 0:
            try:
                stat, pvalue = mannwhitneyu(mem_scores, non_mem_scores, alternative='two-sided')
            except ValueError:
                pvalue = 1.0  # Set default p-value if test fails
        else:
            pvalue = 1.0
            
        mem_mean = np.mean(mem_scores) if len(mem_scores) > 0 else 0
        non_mem_mean = np.mean(non_mem_scores) if len(non_mem_scores) > 0 else 0
        
        # Create box plots
        plot_data = [mem_scores, non_mem_scores]
        labels = ['Memorized', 'Non-memorized']
        
        # Add box plots
        bp = plt.boxplot(plot_data, positions=[1, 2], widths=0.6, 
                        patch_artist=True, showfliers=False)
        
        # Customize box plots
        for box, color in zip(bp['boxes'], colors.values()):
            box.set(facecolor=color, alpha=0.8)
            
        # Add individual points for better visibility
        for i, (scores, pos) in enumerate(zip([mem_scores, non_mem_scores], [1, 2])):
            if len(scores) > 0:
                plt.scatter([pos] * len(scores), scores, 
                          alpha=0.4, color=list(colors.values())[i], 
                          s=30, zorder=3)
        
        # Add statistical annotation
        max_y = max(max(mem_scores, default=-np.inf), max(non_mem_scores, default=-np.inf))
        min_y = min(min(mem_scores, default=np.inf), min(non_mem_scores, default=np.inf))
        y_range = max_y - min_y
        if y_range == 0:
            y_range = 1.0  # Default if all scores are the same
        y_pos = max_y + 0.05 * y_range
        
        significance = ''
        if pvalue < 0.001:
            significance = '***'
        elif pvalue < 0.01:
            significance = '**'
        elif pvalue < 0.05:
            significance = '*'
        
        if significance and len(mem_scores) > 0 and len(non_mem_scores) > 0:
            plt.plot([1, 1, 2, 2], 
                    [y_pos, y_pos + 0.02 * y_range, y_pos + 0.02 * y_range, y_pos], 
                    color='black', lw=1.5)
            plt.text(1.5, y_pos + 0.03 * y_range, significance, 
                    ha='center', va='bottom')
        
        # Customize plot
        plt.title(f'{node_type.capitalize()} Nodes - FLC Distribution\n' +
                 f'(Memorized: n={len(mem_scores)}, μ={mem_mean:.3f} | ' +
                 f'Non-memorized: n={len(non_mem_scores)}, μ={non_mem_mean:.3f})')
        
        # Add homophily information to subtitle
        if homophily_ratio is not None:
            homophily_text = f"Homophilic (h={homophily_ratio:.2f})" if is_homophilic else f"Heterophilic (h={homophily_ratio:.2f})"
            plt.suptitle(f'{homophily_text} - {k_hops}-hop neighborhood',
                       fontsize=12)
            plt.subplots_adjust(top=0.88)  # Make room for the suptitle
            
        plt.ylabel('Feature-Label Coherence Score')
        plt.xticks([1, 2], labels)
        
        # Add grid for better readability
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Add p-value annotation
        plt.text(0.98, 0.02, f'p-value: {pvalue:.2e}',
                transform=plt.gca().transAxes,
                horizontalalignment='right',
                verticalalignment='bottom',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        # Add legend
        legend_elements = [mpatches.Patch(facecolor=color, label=label, alpha=0.7)
                          for label, color in colors.items()]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        # Save plot for this node type
        node_path = f"{base_path}_{node_type}_flc{ext}"
        plt.savefig(node_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Add to summary data
        if len(mem_scores) > 0 and len(non_mem_scores) > 0:
            try:
                _, pvalue = mannwhitneyu(mem_scores, non_mem_scores, alternative='two-sided')
            except ValueError:
                pvalue = 1.0
        else:
            pvalue = 1.0
                
        summary_data.extend([
            {
                'Node Type': node_type,
                'Group': 'Memorized',
                'Mean FLC': np.mean(mem_scores) if mem_scores else np.nan,
                'Sample Size': len(mem_scores),
                'p-value': pvalue
            },
            {
                'Node Type': node_type,
                'Group': 'Non-memorized',
                'Mean FLC': np.mean(non_mem_scores) if non_mem_scores else np.nan,
                'Sample Size': len(non_mem_scores),
                'p-value': pvalue
            }
        ])
    
    return pd.DataFrame(summary_data)

def perform_statistical_tests(results):
    """Perform statistical tests comparing FLC scores between memorized and non-memorized nodes"""
    stats_results = {}
    
    for node_type, data in results.items():
        if node_type == 'dataset_info':
            continue
            
        memorized_scores = [item['flc_score'] for item in data['memorized']]
        non_memorized_scores = [item['flc_score'] for item in data['non_memorized']]
        
        if len(memorized_scores) > 0 and len(non_memorized_scores) > 0:
            # Perform Mann-Whitney U test
            statistic, pvalue = stats.mannwhitneyu(
                memorized_scores,
                non_memorized_scores,
                alternative='two-sided'
            )
            
            # Calculate effect size (Cohen's d)
            mean_diff = np.mean(memorized_scores) - np.mean(non_memorized_scores)
            pooled_std = np.sqrt((np.std(memorized_scores)**2 + np.std(non_memorized_scores)**2) / 2)
            effect_size = abs(mean_diff) / pooled_std if pooled_std != 0 else 0
            
            stats_results[node_type] = {
                'statistic': statistic,
                'pvalue': pvalue,
                'effect_size': effect_size,
                'mean_memorized': np.mean(memorized_scores),
                'mean_non_memorized': np.mean(non_memorized_scores),
                'n_memorized': len(memorized_scores),
                'n_non_memorized': len(non_memorized_scores)
            }
            
    return stats_results

def analyze_flc_components(data, node_scores, k_hops=2):
    """
    Analyze the individual components of the FLC score (NLI and feature divergence)
    to understand their contributions to the overall score
    """
    results = {}
    
    for node_type in ['shared', 'candidate', 'independent', 'extra']:
        if node_type not in node_scores:
            continue
            
        nodes_data = node_scores[node_type]['raw_data']
        memorized_mask = nodes_data['mem_score'] > 0.5
        memorized_nodes = nodes_data.loc[memorized_mask, 'node_idx'].tolist()
        non_memorized_nodes = nodes_data.loc[~memorized_mask, 'node_idx'].tolist()
        
        # Initialize component data
        comp_data = {
            'memorized': {'nli': [], 'feature_div': []},
            'non_memorized': {'nli': [], 'feature_div': []}
        }
        
        # Process memorized nodes
        for node in memorized_nodes:
            # Calculate local NLI
            nli_score, _, _ = calculate_local_nli(data, node, k_hops)
            # Calculate feature divergence
            js_div = calculate_feature_divergence(data, node)
            
            if None not in (nli_score, js_div):
                comp_data['memorized']['nli'].append(nli_score)
                comp_data['memorized']['feature_div'].append(js_div)
        
        # Process non-memorized nodes
        for node in non_memorized_nodes:
            # Calculate local NLI
            nli_score, _, _ = calculate_local_nli(data, node, k_hops)
            # Calculate feature divergence
            js_div = calculate_feature_divergence(data, node)
            
            if None not in (nli_score, js_div):
                comp_data['non_memorized']['nli'].append(nli_score)
                comp_data['non_memorized']['feature_div'].append(js_div)
        
        results[node_type] = comp_data
    
    return results

def plot_component_contributions(component_results, save_path):
    """
    Create visualization showing the contribution of each FLC component
    to memorization vs non-memorization
    """
    # Extract base path and extension
    base_path, ext = os.path.splitext(save_path)
    if not ext:
        ext = '.png'  # Default extension if none provided
        
    # Component names and nice labels
    components = ['nli', 'feature_div']
    component_labels = ['Node Label Informativeness', 'Feature Divergence']
    
    for node_type, data in component_results.items():
        # Skip if insufficient data
        if (not data['memorized']['nli'] or 
            not data['non_memorized']['nli']):
            continue
            
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for i, (comp, label) in enumerate(zip(components, component_labels)):
            # Get data
            mem_vals = data['memorized'][comp]
            non_mem_vals = data['non_memorized'][comp]
            
            # Calculate means for bar chart
            mem_mean = np.mean(mem_vals)
            non_mem_mean = np.mean(non_mem_vals)
            mem_std = np.std(mem_vals)
            non_mem_std = np.std(non_mem_vals)
            
            # Statistical test
            try:
                _, pvalue = stats.mannwhitneyu(mem_vals, non_mem_vals, alternative='two-sided')
            except ValueError:
                pvalue = 1.0
                
            # Plot bars
            ax = axes[i]
            x = [0, 1]
            means = [mem_mean, non_mem_mean]
            stds = [mem_std, non_mem_std]
            
            ax.bar(x, means, width=0.6, yerr=stds, capsize=5, 
                  color=['#FF9999', '#66B2FF'], alpha=0.7)
            
            # Add significance marker if applicable
            if pvalue < 0.05:
                max_y = max(mem_mean + mem_std, non_mem_mean + non_mem_std)
                marker = '***' if pvalue < 0.001 else ('**' if pvalue < 0.01 else '*')
                ax.plot([0, 0, 1, 1], [max_y*1.1, max_y*1.15, max_y*1.15, max_y*1.1], 'k-')
                ax.text(0.5, max_y*1.16, marker, ha='center')
            
            # Labels
            ax.set_title(label)
            ax.set_xticks(x)
            ax.set_xticklabels(['Memorized', 'Non-mem.'])
            ax.set_ylabel('Value')
            
            # Add mean values on bars
            for j, v in enumerate(means):
                ax.text(x[j], v + stds[j] + 0.02, f'{v:.3f}', 
                      ha='center', fontsize=9)
            
            # Add p-value
            ax.text(0.5, 0.02, f'p={pvalue:.3e}', 
                   ha='center', transform=ax.transAxes, fontsize=8)
        
        plt.suptitle(f'{node_type.capitalize()} Nodes - FLC Component Analysis', fontsize=14)
        plt.tight_layout()
        
        # Save plot
        comp_path = f"{base_path}_components_{node_type}{ext}"
        plt.savefig(comp_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return