import torch
import numpy as np
import matplotlib.pyplot as plt
from model import MaxDataDimGNN
import torch.nn.functional as F
import os
from scipy import stats

def get_embeddings(model, data, device):
    """Get embeddings from the second-to-last layer of the GNN model"""
    model.eval()
    with torch.no_grad():
        # Forward pass through model, requesting embeddings from last hidden layer
        _, embeddings = model(data.x.to(device), data.edge_index.to(device), return_node_emb=True)
    return embeddings

def calculate_max_dim_for_nodes(embeddings, node_indices, max_dim_lr=1e-3, max_dim_iter=25, device='cpu'):
    """Calculate maximal data dimensionality for specific nodes"""
    if len(node_indices) == 0:
        return None
        
    # Get embeddings for specific nodes
    node_embeddings = embeddings[node_indices]
    hidden_size = node_embeddings.shape[1]
    
    # Initialize MaxDataDimGNN model
    v_model = MaxDataDimGNN(size=hidden_size, acts=node_embeddings).to(device)
    optimizer = torch.optim.Adam(v_model.parameters(), lr=max_dim_lr)
    
    # Optimize for maximum dimensionality
    max_dims = []
    final_dims = None
    
    for ep in range(max_dim_iter):
        optimizer.zero_grad()
        dims = v_model()
        loss = -torch.sum(dims)  # Maximize dimensionality
        loss.backward()
        optimizer.step()
        
        max_dims.append(dims.detach().cpu())
        final_dims = dims.detach().cpu()
    
    return final_dims

def perform_dimensionality_statistical_tests(node_scores, dim_results, logger=None):
    """
    Perform statistical tests to analyze the significance of dimensionality differences
    between memorized and non-memorized nodes.
    
    Args:
        node_scores: Dictionary containing memorization scores for each node type
        dim_results: Dictionary containing dimensionality values for each node category
        logger: Logger object for output
    
    Returns:
        Dictionary containing statistical test results
    """
    stats_results = {}
    
    for node_type in ['shared', 'candidate', 'independent', 'extra']:
        if node_type not in node_scores:
            continue
            
        # Get dimensionality values for memorized and non-memorized nodes
        scores_df = node_scores[node_type]['raw_data']
        mem_mask = scores_df['mem_score'] > 0.5
        
        memorized_dim = dim_results.get(f'{node_type}_memorized')
        non_memorized_dim = dim_results.get(f'{node_type}_non_memorized')
        
        if memorized_dim is None or non_memorized_dim is None:
            continue
            
        # Perform Mann-Whitney U test (non-parametric)
        try:
            statistic, pvalue = stats.mannwhitneyu(
                [memorized_dim] * sum(mem_mask),
                [non_memorized_dim] * sum(~mem_mask),
                alternative='two-sided'
            )
        except ValueError:
            # Handle case where one group is empty
            statistic, pvalue = None, None
        
        # Calculate effect size (Cohen's d)
        n1 = sum(mem_mask)
        n2 = sum(~mem_mask)
        if n1 > 0 and n2 > 0:
            # Pooled standard deviation formula
            s_pooled = np.sqrt(((n1 - 1) * np.std([memorized_dim] * n1)**2 + 
                              (n2 - 1) * np.std([non_memorized_dim] * n2)**2) / 
                             (n1 + n2 - 2))
            effect_size = abs(memorized_dim - non_memorized_dim) / s_pooled
        else:
            effect_size = None
        
        stats_results[node_type] = {
            'n_memorized': sum(mem_mask),
            'n_non_memorized': sum(~mem_mask),
            'mean_memorized': memorized_dim,
            'mean_non_memorized': non_memorized_dim,
            'statistic': statistic,
            'pvalue': pvalue,
            'effect_size': effect_size
        }
        
        if logger:
            logger.info(f"\n{node_type.capitalize()} Nodes Dimensionality Analysis:")
            logger.info(f"Sample sizes: {sum(mem_mask)} memorized, {sum(~mem_mask)} non-memorized")
            logger.info(f"Mean dimensionality:")
            logger.info(f"  - Memorized nodes: {memorized_dim:.4f}")
            logger.info(f"  - Non-memorized nodes: {non_memorized_dim:.4f}")
            if pvalue is not None:
                logger.info(f"Mann-Whitney U test: statistic={statistic:.4f}, p-value={pvalue:.4e}")
            
            if effect_size is not None:
                # Interpret effect size
                if effect_size < 0.2:
                    effect_interp = "negligible"
                elif effect_size < 0.5:
                    effect_interp = "small"
                elif effect_size < 0.8:
                    effect_interp = "medium"
                else:
                    effect_interp = "large"
                logger.info(f"Effect size (Cohen's d): {effect_size:.4f} ({effect_interp})")
    
    return stats_results

def plot_memorization_dimensionality(node_scores, model_f, model_g, data, save_path, device):
    """Create bar plot comparing dimensionality of memorized vs non-memorized nodes"""
    plt.figure(figsize=(12, 6))
    
    # Get embeddings from both models
    f_embeddings = get_embeddings(model_f, data, device)
    g_embeddings = get_embeddings(model_g, data, device)
    
    results = {}
    node_types = ['shared', 'candidate', 'independent', 'extra']
    bar_positions = np.arange(len(node_types)) * 3
    
    for i, node_type in enumerate(node_types):
        if node_type not in node_scores:
            continue
            
        # Split nodes into memorized and non-memorized
        scores_df = node_scores[node_type]['raw_data']
        mem_mask = scores_df['mem_score'] > 0.5
        
        memorized_nodes = scores_df[mem_mask]['node_idx'].values
        non_memorized_nodes = scores_df[~mem_mask]['node_idx'].values
        
        # Choose which model's embeddings to use
        if node_type == 'independent':
            embeddings = g_embeddings
        elif node_type in ['candidate', 'shared']:
            embeddings = f_embeddings
        else:  # 'extra' nodes - average of both models
            f_dims = calculate_max_dim_for_nodes(f_embeddings, scores_df['node_idx'].values, device=device)
            g_dims = calculate_max_dim_for_nodes(g_embeddings, scores_df['node_idx'].values, device=device)
            if f_dims is not None and g_dims is not None:
                results[f'{node_type}_memorized'] = (
                    torch.mean(f_dims[mem_mask]).item() + torch.mean(g_dims[mem_mask]).item()
                ) / 2
                results[f'{node_type}_non_memorized'] = (
                    torch.mean(f_dims[~mem_mask]).item() + torch.mean(g_dims[~mem_mask]).item()
                ) / 2
            continue
            
        # Calculate dimensionality
        mem_dims = calculate_max_dim_for_nodes(embeddings, memorized_nodes, device=device)
        non_mem_dims = calculate_max_dim_for_nodes(embeddings, non_memorized_nodes, device=device)
        
        if mem_dims is not None:
            results[f'{node_type}_memorized'] = torch.mean(mem_dims).item()
        if non_mem_dims is not None:
            results[f'{node_type}_non_memorized'] = torch.mean(non_mem_dims).item()
    
    # Create grouped bar plot
    bar_width = 0.35
    
    memorized_values = [results.get(f'{nt}_memorized', 0) for nt in node_types]
    non_memorized_values = [results.get(f'{nt}_non_memorized', 0) for nt in node_types]
    
    plt.bar(bar_positions - bar_width/2, memorized_values, bar_width, 
            label='Memorized (>0.5)', color='blue', alpha=0.7)
    plt.bar(bar_positions + bar_width/2, non_memorized_values, bar_width,
            label='Non-memorized (≤0.5)', color='orange', alpha=0.7)
    
    plt.xlabel('Node Type')
    plt.ylabel('Average Max Data Dimensionality')
    plt.title('Max Data Dimensionality: Memorized vs Non-memorized Nodes')
    plt.xticks(bar_positions, node_types)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value labels on top of bars
    def add_value_labels(positions, values, offset):
        for pos, val in zip(positions, values):
            if val != 0:  # Only add label if value exists
                plt.text(pos + offset, val, f'{val:.3f}', 
                        ha='center', va='bottom')
    
    add_value_labels(bar_positions, memorized_values, -bar_width/2)
    add_value_labels(bar_positions, non_memorized_values, bar_width/2)
    
    # Perform statistical tests
    stats_results = perform_dimensionality_statistical_tests(node_scores, results, logger=None)
    
    # Add statistical significance markers to plot if significant
    for node_type in node_types:
        if node_type in stats_results and stats_results[node_type]['pvalue'] is not None:
            if stats_results[node_type]['pvalue'] < 0.01:
                marker = '***'
            elif stats_results[node_type]['pvalue'] < 0.05:
                marker = '**'
            elif stats_results[node_type]['pvalue'] < 0.1:
                marker = '*'
            else:
                continue
                
            # Add significance marker
            idx = node_types.index(node_type)
            max_height = max(
                results.get(f'{node_type}_memorized', 0),
                results.get(f'{node_type}_non_memorized', 0)
            )
            plt.text(bar_positions[idx], max_height + 0.02, marker,
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return results, stats_results