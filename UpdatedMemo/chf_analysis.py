import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from chfmeasure import CFH_measure
import seaborn as sns
from tqdm import tqdm
from scipy import stats

def analyze_memorization_vs_cfh(data, node_scores, device, save_path=None):
    """
    Analyze the relationship between memorization scores and the CFH measure.
    
    Args:
        data: PyG data object
        node_scores: Dict of memorization scores by node type
        device: torch device
        save_path: Path to save the plot
    
    Returns:
        Dictionary with CFH measure results for each node type
    """
    # Create dummy args object to pass to CFH_measure
    class Args:
        def __init__(self, device):
            self.device = device
    
    args = Args(device)
    
    # Calculate graph-level CFH measure and node-level CFH values
    h_g_norm, h_vi_norm = CFH_measure(args, data, count_self=False, measure='CFH')
    
    results = {}
    
    # For each node type, compare CFH values for memorized vs non-memorized nodes
    for node_type in ['candidate', 'shared', 'independent', 'extra']:
        if node_type not in node_scores:
            continue
            
        # Get node indices and memorization scores
        node_data = node_scores[node_type]['raw_data']
        node_indices = node_data['node_idx'].values
        mem_scores = node_data['mem_score'].values
        
        # Get CFH values for these nodes
        cfh_values = [h_vi_norm[idx] for idx in node_indices]
        
        # Separate nodes by memorization threshold
        memorized_mask = mem_scores > 0.5
        memorized_cfh = np.array([cfh_values[i] for i in range(len(cfh_values)) if memorized_mask[i]])
        non_memorized_cfh = np.array([cfh_values[i] for i in range(len(cfh_values)) if not memorized_mask[i]])
        
        # Perform statistical test (Mann-Whitney U)
        if len(memorized_cfh) > 0 and len(non_memorized_cfh) > 0:
            stat, pval = stats.mannwhitneyu(memorized_cfh, non_memorized_cfh)
            # Calculate effect size (Cohen's d)
            mean_diff = np.mean(memorized_cfh) - np.mean(non_memorized_cfh)
            pooled_std = np.sqrt((np.std(memorized_cfh)**2 + np.std(non_memorized_cfh)**2) / 2)
            effect_size = abs(mean_diff) / pooled_std if pooled_std > 0 else 0
        else:
            stat, pval, effect_size = None, None, None
            
        # Store results
        results[node_type] = {
            'node_indices': node_indices,
            'cfh_values': cfh_values,
            'mem_scores': mem_scores,
            'memorized_cfh': memorized_cfh,
            'non_memorized_cfh': non_memorized_cfh,
            'stat_test': {
                'statistic': stat,
                'pvalue': pval,
                'effect_size': effect_size
            },
            'graph_cfh': h_g_norm
        }
        
    # Create visualization if save_path is provided
    if save_path:
        plot_memorization_cfh_comparison(results, save_path)
        
    return results

def plot_memorization_cfh_comparison(results, save_path):
    """
    Create boxplot comparing CFH values for memorized vs non-memorized nodes across node types
    """
    plt.figure(figsize=(12, 8))
    
    # Prepare data for box plots
    box_data = []
    labels = []
    colors = []
    
    for node_type in ['candidate', 'shared', 'independent', 'extra']:
        if node_type not in results:
            continue
            
        # Add memorized nodes
        if len(results[node_type]['memorized_cfh']) > 0:
            box_data.append(results[node_type]['memorized_cfh'])
            labels.append(f"{node_type}\nmemorized")
            colors.append('red')
            
        # Add non-memorized nodes
        if len(results[node_type]['non_memorized_cfh']) > 0:
            box_data.append(results[node_type]['non_memorized_cfh'])
            labels.append(f"{node_type}\nnon-memorized")
            colors.append('blue')
    
    # Create boxplot
    box_parts = plt.boxplot(box_data, patch_artist=True, labels=labels)
    
    # Color the boxes
    for box, color in zip(box_parts['boxes'], colors):
        box.set(facecolor=color, alpha=0.6)
    
    # Add statistical significance annotations
    y_max = max([max(data) if len(data) > 0 else -np.inf for data in box_data])
    y_min = min([min(data) if len(data) > 0 else np.inf for data in box_data])
    y_range = y_max - y_min
    
    box_idx = 0
    for node_type in ['candidate', 'shared', 'independent', 'extra']:
        if node_type not in results:
            continue
            
        # Check if we have both memorized and non-memorized nodes
        if len(results[node_type]['memorized_cfh']) > 0 and len(results[node_type]['non_memorized_cfh']) > 0:
            pval = results[node_type]['stat_test']['pvalue']
            
            # Add significance annotation
            if pval is not None:
                sig_text = ""
                if pval < 0.001:
                    sig_text = "***"
                elif pval < 0.01:
                    sig_text = "**"
                elif pval < 0.05:
                    sig_text = "*"
                else:
                    sig_text = "n.s."
                
                # Calculate effect size text
                effect_size = results[node_type]['stat_test']['effect_size']
                if effect_size is not None:
                    if effect_size < 0.2:
                        effect_text = "negligible"
                    elif effect_size < 0.5:
                        effect_text = "small"
                    elif effect_size < 0.8:
                        effect_text = "medium"
                    else:
                        effect_text = "large"
                        
                    sig_text += f"\n(d={effect_size:.2f}, {effect_text})"
                
                #plt.text(box_idx + 1.5, y_max + 0.05 * y_range, 
                 #        sig_text, horizontalalignment='center',
                  #       weight='bold')
            
            box_idx += 2
        else:
            # If only one type exists (either memorized or non-memorized)
            if len(results[node_type]['memorized_cfh']) > 0 or len(results[node_type]['non_memorized_cfh']) > 0:
                box_idx += 1
    
    # Add graph-level CFH as a horizontal line
    graph_cfh = results[list(results.keys())[0]]['graph_cfh']
    plt.axhline(y=graph_cfh, color='green', linestyle='--', 
                label=f'Graph CFH = {graph_cfh:.3f}')
    
    plt.ylabel('CFH Measure (h_vi_norm)')
    plt.title('Comparison of CFH Measure for Memorized vs Non-memorized Nodes')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def perform_statistical_tests(results):
    """
    Perform statistical tests comparing memorized and non-memorized nodes.
    
    Args:
        results: Dictionary with CFH measure results
        
    Returns:
        Dictionary with statistical test results
    """
    stats_results = {}
    
    for node_type, data in results.items():
        memorized_cfh = data['memorized_cfh']
        non_memorized_cfh = data['non_memorized_cfh']
        
        # Skip if no data in either group
        if len(memorized_cfh) == 0 or len(non_memorized_cfh) == 0:
            stats_results[node_type] = {
                'n_memorized': len(memorized_cfh),
                'n_non_memorized': len(non_memorized_cfh),
                'mean_memorized': np.mean(memorized_cfh) if len(memorized_cfh) > 0 else None,
                'mean_non_memorized': np.mean(non_memorized_cfh) if len(non_memorized_cfh) > 0 else None,
                'statistic': None,
                'pvalue': None,
                'effect_size': None
            }
            continue
        
        # Perform Mann-Whitney U test
        statistic, pvalue = stats.mannwhitneyu(memorized_cfh, non_memorized_cfh)
        
        # Calculate effect size (Cohen's d)
        mean_diff = np.mean(memorized_cfh) - np.mean(non_memorized_cfh)
        pooled_std = np.sqrt((np.std(memorized_cfh)**2 + np.std(non_memorized_cfh)**2) / 2)
        effect_size = abs(mean_diff) / pooled_std if pooled_std > 0 else 0
        
        stats_results[node_type] = {
            'n_memorized': len(memorized_cfh),
            'n_non_memorized': len(non_memorized_cfh),
            'mean_memorized': np.mean(memorized_cfh),
            'mean_non_memorized': np.mean(non_memorized_cfh),
            'statistic': statistic,
            'pvalue': pvalue,
            'effect_size': effect_size
        }
    
    return stats_results

def analyze_correlation(results):
    """
    Analyze correlation between memorization scores and CFH values.
    
    Args:
        results: Dictionary with CFH measure results
        
    Returns:
        DataFrame with correlation results
    """
    correlation_data = []
    
    for node_type, data in results.items():
        # Calculate correlation
        r, p = stats.pearsonr(data['mem_scores'], data['cfh_values'])
        
        correlation_data.append({
            'Node Type': node_type,
            'Correlation': r,
            'P-value': p,
            'Sample Size': len(data['mem_scores'])
        })
    
    return pd.DataFrame(correlation_data)