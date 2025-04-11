import torch
import numpy as np
import networkx as nx
from torch_geometric.utils import k_hop_subgraph
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from nodeli import li_node
from scipy import stats
import matplotlib.patches as mpatches
from scipy.stats import mannwhitneyu
import warnings
import os
warnings.filterwarnings('ignore')

def calculate_local_nli(data, node_idx, k_hops=2):
    """Calculate NLI score for a k-hop neighborhood around a node"""
    # Get k-hop subgraph
    subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=[node_idx], 
        num_hops=k_hops, 
        edge_index=data.edge_index,
        relabel_nodes=True,
        num_nodes=data.num_nodes
    )
    
    # Convert to NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(range(len(subset)))
    edge_list = edge_index.t().tolist()
    G.add_edges_from(edge_list)
    
    # Get labels for subgraph nodes
    labels = data.y[subset].cpu().numpy()
    
    # Calculate NLI for subgraph
    try:
        nli_score = li_node(G, labels)
    except:
        nli_score = None
        
    return nli_score, len(subset), len(edge_list)

def calculate_dataset_homophily(data):
    """Calculate edge homophily ratio for the entire dataset"""
    edge_index = data.edge_index
    labels = data.y
    
    # Count edges with same labels
    same_label_edges = (labels[edge_index[0]] == labels[edge_index[1]]).sum().item()
    total_edges = edge_index.size(1)
    
    homophily_ratio = same_label_edges / total_edges
    return homophily_ratio

def analyze_memorization_vs_nli(data, node_scores, k_hops=2):
    """Analyze relationship between memorization and local NLI scores"""
    results = {}
    
    # Calculate dataset homophily
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
        
        # Calculate NLI scores for both groups
        memorized_nli = []
        non_memorized_nli = []
        
        # Process memorized nodes
        for node in memorized_nodes:
            nli_score, num_nodes, num_edges = calculate_local_nli(data, node, k_hops)
            if nli_score is not None:
                memorized_nli.append({
                    'nli_score': nli_score,
                    'subgraph_size': num_nodes,
                    'edge_count': num_edges
                })
        
        # Process non-memorized nodes
        for node in non_memorized_nodes:
            nli_score, num_nodes, num_edges = calculate_local_nli(data, node, k_hops)
            if nli_score is not None:
                non_memorized_nli.append({
                    'nli_score': nli_score,
                    'subgraph_size': num_nodes,
                    'edge_count': num_edges
                })
        
        # Store results
        results[node_type] = {
            'memorized': memorized_nli,
            'non_memorized': non_memorized_nli
        }
        
    return results

def plot_memorization_nli_comparison(results, save_path, k_hops):
    """
    Create separate visualization files comparing NLI scores between memorized and non-memorized nodes
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
        mem_scores = [item['nli_score'] for item in data['memorized']]
        non_mem_scores = [item['nli_score'] for item in data['non_memorized']]
        
        if len(mem_scores) == 0 and len(non_mem_scores) == 0:
            plt.text(0.5, 0.5, f'No data available for {node_type} nodes',
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title(f'{node_type.capitalize()} Nodes - NLI Distribution')
            plt.tight_layout()
            # Save plot for this node type
            node_path = f"{base_path}_{node_type}_nli{ext}"
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
        plt.title(f'{node_type.capitalize()} Nodes - NLI Distribution\n' +
                 f'(Memorized: n={len(mem_scores)}, μ={mem_mean:.3f} | ' +
                 f'Non-memorized: n={len(non_mem_scores)}, μ={non_mem_mean:.3f})')
        
        # Add homophily information to subtitle
        if homophily_ratio is not None:
            homophily_text = f"Homophilic (h={homophily_ratio:.2f})" if is_homophilic else f"Heterophilic (h={homophily_ratio:.2f})"
            plt.suptitle(f'{homophily_text} - {k_hops}-hop neighborhood',
                       fontsize=12)
            plt.subplots_adjust(top=0.88)  # Make room for the suptitle
            
        plt.ylabel('Node Label Informativeness')
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
        node_path = f"{base_path}_{node_type}_nli{ext}"
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
                'Mean NLI': np.mean(mem_scores) if mem_scores else np.nan,
                'Sample Size': len(mem_scores),
                'p-value': pvalue
            },
            {
                'Node Type': node_type,
                'Group': 'Non-memorized',
                'Mean NLI': np.mean(non_mem_scores) if non_mem_scores else np.nan,
                'Sample Size': len(non_mem_scores),
                'p-value': pvalue
            }
        ])
    
    return pd.DataFrame(summary_data)

def perform_statistical_tests(results):
    """Perform statistical tests comparing NLI scores between memorized and non-memorized nodes"""
    stats_results = {}
    
    for node_type, data in results.items():
        if node_type == 'dataset_info':
            continue
            
        memorized_scores = [item['nli_score'] for item in data['memorized']]
        non_memorized_scores = [item['nli_score'] for item in data['non_memorized']]
        
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