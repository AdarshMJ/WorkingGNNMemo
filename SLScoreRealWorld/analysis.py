import torch
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch_geometric.utils import to_networkx, k_hop_subgraph
from scipy import stats
from typing import Dict, List, Tuple
import matplotlib.colors as mcolors
from itertools import combinations
from nodeli import li_node  # Import the li_node function from your nodeli.py

def create_node_visualization(
    edge_index: torch.Tensor,
    nodes_dict: Dict[str, List[int]],
    memorization_scores: Dict[str, Dict],
    save_path: str,
    title: str = "Node Types and Memorization Scores",
    figsize=(14, 8)  # Increased figure width to accommodate legend
):
    """
    Create a visualization of the graph with nodes colored by their type and memorization score.
    """
    # Create NetworkX graph from edge_list
    edge_list = edge_index.t().tolist()
    G = nx.Graph()
    G.add_edges_from(edge_list)
    max_node = max(max(edge_index[0]), max(edge_index[1]))
    G.add_nodes_from(range(max_node + 1))
    
    # Create position layout
    pos = nx.spring_layout(G, k=1/np.sqrt(G.number_of_nodes()), seed=42)
    
    # Create figure and axis with dark background
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor('#1f1f1f')
    fig.set_facecolor('#1f1f1f')
    
    # Create a color map for node types - using more distinct colors
    type_colors = {
        'shared': '#3498db',      # Bright blue
        'candidate': '#e74c3c',   # Bright red
        'independent': '#2ecc71', # Bright green
        'extra': '#9b59b6'       # Purple
    }
    
    # Draw edges first (in light gray)
    nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='#666666', ax=ax)
    
    # Track all scores for colorbar scaling
    all_scores = []
    
    # Draw nodes for each type
    for node_type, nodes in nodes_dict.items():
        if node_type in memorization_scores:
            scores = memorization_scores[node_type]['mem_scores']
            score_dict = {node: score for node, score in zip(nodes, scores)}
            all_scores.extend(scores)
            
            # Border color based on node type
            border_color = type_colors[node_type]
            
            # Draw nodes with larger size and distinct borders
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=nodes,
                node_color=[score_dict[n] for n in nodes],
                cmap='viridis',  # Changed to viridis colormap
                node_size=150,   # Increased node size
                alpha=0.9,       # Increased opacity
                label=node_type,
                ax=ax,
                edgecolors=border_color,
                linewidths=2     # Thicker border
            )
    
    # Add colorbar with improved styling
    if all_scores:
        norm = plt.Normalize(vmin=min(all_scores), vmax=max(all_scores))
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])
        # Position colorbar to the right of the plot with some padding
        cbar = plt.colorbar(sm, ax=ax, label='Memorization Score', pad=0.02)
        cbar.ax.yaxis.label.set_color('white')
        cbar.ax.tick_params(colors='white')
    
    # Style the plot
    plt.title(title, color='white', pad=20)
    
    # Position legend outside the plot on the right with no overlap
    legend = ax.legend(bbox_to_anchor=(1.25, 1.0), 
                      loc='upper left', 
                      frameon=True)
    legend.get_frame().set_facecolor('#1f1f1f')
    legend.get_frame().set_edgecolor('white')
    for text in legend.get_texts():
        text.set_color('white')
    
    ax.set_axis_off()
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save with extra right margin to ensure legend is visible
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#1f1f1f', 
                bbox_extra_artists=[legend], pad_inches=0.5)
    plt.close()

def analyze_score_clustering(
    edge_index: torch.Tensor,
    nodes_dict: Dict[str, List[int]],
    memorization_scores: Dict[str, Dict],
    save_path: str
) -> Dict[str, float]:
    """
    Analyze whether nodes with similar memorization scores cluster together.
    
    Args:
        edge_index: PyG edge_index tensor
        nodes_dict: Dictionary mapping node types to node indices
        memorization_scores: Dictionary containing memorization scores for each node type
        save_path: Path to save the analysis plots
    
    Returns:
        Dictionary containing clustering metrics
    """
    # Create NetworkX graph from edge_list
    edge_list = edge_index.t().tolist()
    G = nx.Graph()
    G.add_edges_from(edge_list)
    
    # Add any isolated nodes
    max_node = max(max(edge_index[0]), max(edge_index[1]))
    G.add_nodes_from(range(max_node + 1))
    
    # Create a mapping of node index to memorization score
    score_map = {}
    for node_type, data in memorization_scores.items():
        for node, score in zip(nodes_dict[node_type], data['mem_scores']):
            score_map[node] = score
    
    # Calculate local clustering metrics
    local_scores = []
    neighbor_scores = []
    
    for node in G.nodes():
        if node in score_map:
            node_score = score_map[node]
            # Get neighbors' scores
            neighbors = list(G.neighbors(node))
            if neighbors:
                neighbor_scores_list = [
                    score_map[n] for n in neighbors
                    if n in score_map
                ]
                if neighbor_scores_list:  # Only add if we found valid neighbor scores
                    neighbor_avg = np.mean(neighbor_scores_list)
                    local_scores.append(node_score)
                    neighbor_scores.append(neighbor_avg)
    
    if not local_scores or not neighbor_scores:
        return {
            'correlation': 0.0,
            'p_value': 1.0,
            'warning': 'Insufficient data for correlation analysis'
        }
    
    # Calculate correlation between node scores and neighbor scores
    correlation = stats.pearsonr(local_scores, neighbor_scores)
    
    # Plot the relationship
    plt.figure(figsize=(8, 6))
    plt.scatter(local_scores, neighbor_scores, alpha=0.5)
    plt.xlabel("Node Memorization Score")
    plt.ylabel("Average Neighbor Memorization Score")
    plt.title("Node Score vs Average Neighbor Score")
    
    # Add correlation line
    z = np.polyfit(local_scores, neighbor_scores, 1)
    p = np.poly1d(z)
    plt.plot(local_scores, p(local_scores), "r--", alpha=0.8)
    
    plt.text(0.05, 0.95, f'Correlation: {correlation[0]:.3f}\np-value: {correlation[1]:.3f}',
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'correlation': correlation[0],
        'p_value': correlation[1]
    }

def analyze_distance_effects(
    edge_index: torch.Tensor,
    nodes_dict: Dict[str, List[int]],
    memorization_scores: Dict[str, Dict],
    save_path: str,
    max_distance: int = 5
) -> Dict[str, Dict[int, float]]:
    """
    Analyze how memorization scores correlate with graph distance from candidate nodes.
    """
    # Create NetworkX graph from edge_list
    edge_list = edge_index.t().tolist()
    G = nx.Graph()
    G.add_edges_from(edge_list)
    
    # Add any isolated nodes
    max_node = max(max(edge_index[0]), max(edge_index[1]))
    G.add_nodes_from(range(max_node + 1))
    
    # Create score mapping
    score_map = {}
    for node_type, data in memorization_scores.items():
        for node, score in zip(nodes_dict[node_type], data['mem_scores']):
            score_map[node] = score
    
    # Calculate distances from candidate nodes and corresponding scores
    distances_scores = {d: [] for d in range(max_distance + 1)}
    
    for candidate_node in nodes_dict['candidate']:
        try:
            lengths = nx.single_source_shortest_path_length(G, candidate_node, cutoff=max_distance)
            for node, distance in lengths.items():
                if node in score_map and node != candidate_node:
                    distances_scores[distance].append(score_map[node])
        except nx.NetworkXError:
            continue
    
    # Calculate average scores at each distance
    avg_scores = {d: np.mean(scores) if scores else np.nan 
                 for d, scores in distances_scores.items()}
    std_scores = {d: np.std(scores) if scores else np.nan 
                 for d, scores in distances_scores.items()}
    
    # Plot distance decay
    plt.figure(figsize=(10, 6))
    distances = list(avg_scores.keys())
    means = list(avg_scores.values())
    stds = list(std_scores.values())
    
    plt.errorbar(distances, means, yerr=stds, fmt='o-', capsize=5)
    plt.xlabel("Hop Distance from Candidate Nodes")
    plt.ylabel("Average Memorization Score")
    plt.title("Memorization Score Decay with Graph Distance")
    
    # Add trend line
    valid_idx = ~np.isnan(means)
    if np.any(valid_idx):
        valid_distances = np.array(distances)[valid_idx]
        valid_means = np.array(means)[valid_idx]
        if len(valid_distances) > 1:  # Need at least 2 points for line fitting
            z = np.polyfit(valid_distances, valid_means, 1)
            p = np.poly1d(z)
            plt.plot(valid_distances, p(valid_distances), "r--", alpha=0.8, 
                    label=f'Trend (slope: {z[0]:.3f})')
            plt.legend()
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'avg_scores': avg_scores,
        'std_scores': std_scores
    }

def perform_statistical_tests(
    nodes_dict: Dict[str, List[int]],
    memorization_scores: Dict[str, Dict],
) -> Dict[str, Dict]:
    """
    Perform statistical tests on memorization scores.
    
    Args:
        nodes_dict: Dictionary mapping node types to node indices
        memorization_scores: Dictionary containing memorization scores for each node type
    
    Returns:
        Dictionary containing test results
    """
    results = {
        'pairwise_ttests': {},
        'correlation_tests': {}
    }
    
    # Perform pairwise t-tests between node types
    node_types = list(memorization_scores.keys())
    for type1, type2 in combinations(node_types, 2):
        scores1 = memorization_scores[type1]['mem_scores']
        scores2 = memorization_scores[type2]['mem_scores']
        
        t_stat, p_val = stats.ttest_ind(scores1, scores2)
        effect_size = abs(np.mean(scores1) - np.mean(scores2)) / np.sqrt(
            (np.var(scores1) + np.var(scores2)) / 2
        )  # Cohen's d
        
        results['pairwise_ttests'][f'{type1}_vs_{type2}'] = {
            't_statistic': t_stat,
            'p_value': p_val,
            'effect_size': effect_size,
            'mean_diff': np.mean(scores1) - np.mean(scores2)
        }
    
    return results

def analyze_neighbor_composition(
    edge_index: torch.Tensor,
    nodes_dict: Dict[str, List[int]],
    memorization_scores: Dict[str, Dict],
    save_path: str
) -> Dict:
    """
    Analyze the composition of neighbors for each node type and their memorization scores.
    """
    # Create NetworkX graph
    edge_list = edge_index.t().tolist()
    G = nx.Graph()
    G.add_edges_from(edge_list)
    
    # Create node type mapping
    node_type_map = {}
    for node_type, nodes in nodes_dict.items():
        for node in nodes:
            node_type_map[node] = node_type
    
    # Create score mapping
    score_map = {}
    for node_type, data in memorization_scores.items():
        for node, score in zip(nodes_dict[node_type], data['mem_scores']):
            score_map[node] = score
    
    # Initialize results dictionary
    composition_stats = {node_type: {
        'neighbor_types': {nt: [] for nt in nodes_dict.keys()},
        'neighbor_scores': {nt: [] for nt in nodes_dict.keys()}
    } for node_type in nodes_dict.keys()}
    
    # Analyze neighbors for each node type
    for node_type, nodes in nodes_dict.items():
        for node in nodes:
            neighbors = list(G.neighbors(node))
            for neighbor in neighbors:
                if neighbor in node_type_map:
                    neighbor_type = node_type_map[neighbor]
                    composition_stats[node_type]['neighbor_types'][neighbor_type].append(neighbor)
                    if neighbor in score_map:
                        composition_stats[node_type]['neighbor_scores'][neighbor_type].append(
                            score_map[neighbor]
                        )
    
    # Calculate statistics
    summary_stats = {}
    for node_type in nodes_dict.keys():
        total_neighbors = sum(len(v) for v in composition_stats[node_type]['neighbor_types'].values())
        if total_neighbors == 0:
            continue
            
        type_proportions = {
            nt: len(neighbors)/total_neighbors 
            for nt, neighbors in composition_stats[node_type]['neighbor_types'].items()
        }
        
        avg_neighbor_scores = {
            nt: np.mean(scores) if scores else np.nan
            for nt, scores in composition_stats[node_type]['neighbor_scores'].items()
        }
        
        summary_stats[node_type] = {
            'type_proportions': type_proportions,
            'avg_neighbor_scores': avg_neighbor_scores
        }
    
    # Visualization
    plt.figure(figsize=(15, 6))
    
    # Plot 1: Neighbor type composition
    plt.subplot(1, 2, 1)
    node_types = list(nodes_dict.keys())
    x = np.arange(len(node_types))
    width = 0.15
    
    for i, neighbor_type in enumerate(node_types):
        proportions = [summary_stats[nt]['type_proportions'].get(neighbor_type, 0) 
                      for nt in node_types]
        plt.bar(x + i*width, proportions, width, 
                label=f'{neighbor_type} neighbors',
                alpha=0.7)
    
    plt.xlabel('Node Type')
    plt.ylabel('Proportion of Neighbors')
    plt.title('Neighbor Type Composition')
    plt.xticks(x + width*1.5, node_types)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 2: Average neighbor scores
    plt.subplot(1, 2, 2)
    for i, node_type in enumerate(node_types):
        scores = [summary_stats[nt]['avg_neighbor_scores'].get(node_type, np.nan) 
                 for nt in node_types]
        plt.bar(x + i*width, scores, width, 
                label=f'{node_type} neighbors',
                alpha=0.7)
    
    plt.xlabel('Node Type')
    plt.ylabel('Average Memorization Score')
    plt.title('Average Neighbor Memorization Scores')
    plt.xticks(x + width*1.5, node_types)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return summary_stats

def analyze_distance_based_memorization(
    edge_index: torch.Tensor,
    nodes_dict: Dict[str, List[int]],
    memorization_scores: Dict[str, Dict],
    save_path: str,
    max_distance: int = 5
) -> Dict:
    """
    Analyze memorization scores as a function of distance from nearest candidate node.
    """
    # Create NetworkX graph
    edge_list = edge_index.t().tolist()
    G = nx.Graph()
    G.add_edges_from(edge_list)
    
    # Create score mapping
    score_map = {}
    for node_type, data in memorization_scores.items():
        for node, score in zip(nodes_dict[node_type], data['mem_scores']):
            score_map[node] = score
    
    # Initialize distance-based scores
    distance_scores = {
        'shared': {d: [] for d in range(max_distance + 1)},
        'independent': {d: [] for d in range(max_distance + 1)}
    }
    
    # Calculate minimum distance to any candidate node for each node
    for node_type in ['shared', 'independent']:
        for node in nodes_dict[node_type]:
            min_distance = float('inf')
            
            # Find minimum distance to any candidate node
            for candidate in nodes_dict['candidate']:
                try:
                    distance = nx.shortest_path_length(G, node, candidate)
                    min_distance = min(min_distance, distance)
                except nx.NetworkXNoPath:
                    continue
            
            if min_distance <= max_distance and node in score_map:
                distance_scores[node_type][min_distance].append(score_map[node])
    
    # Calculate statistics
    avg_scores = {
        node_type: {
            d: np.mean(scores) if scores else np.nan
            for d, scores in distances.items()
        }
        for node_type, distances in distance_scores.items()
    }
    
    std_scores = {
        node_type: {
            d: np.std(scores) if scores else np.nan
            for d, scores in distances.items()
        }
        for node_type, distances in distance_scores.items()
    }
    
    # Visualization
    plt.figure(figsize=(10, 6))
    
    colors = {'shared': '#3498db', 'independent': '#e74c3c'}
    labels = {'shared': 'Shared Nodes', 'independent': 'Independent Nodes'}
    
    for node_type in ['shared', 'independent']:
        distances = list(avg_scores[node_type].keys())
        means = list(avg_scores[node_type].values())
        stds = list(std_scores[node_type].values())
        
        valid_idx = ~np.isnan(means)
        valid_distances = np.array(distances)[valid_idx]
        valid_means = np.array(means)[valid_idx]
        valid_stds = np.array(stds)[valid_idx]
        
        if len(valid_distances) > 0:
            plt.errorbar(valid_distances, valid_means, yerr=valid_stds,
                        fmt='o-', color=colors[node_type], label=labels[node_type],
                        capsize=5, alpha=0.7)
    
    plt.xlabel('Distance from Nearest Candidate Node')
    plt.ylabel('Average Memorization Score')
    plt.title('Memorization Score vs Distance from Candidate Nodes')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'avg_scores': avg_scores,
        'std_scores': std_scores
    }

def analyze_structural_properties(
    edge_index: torch.Tensor,
    nodes_dict: Dict[str, List[int]],
    memorization_scores: Dict[str, Dict],
    save_path: str
) -> Dict:
    """
    Analyze structural properties of nodes and their neighborhoods.
    Computes: degree, clustering coefficient, betweenness centrality,
    and neighborhood diversity metrics.
    """
    # Create NetworkX graph
    edge_list = edge_index.t().tolist()
    G = nx.Graph()
    G.add_edges_from(edge_list)
    
    # Add isolated nodes
    max_node = max(max(edge_index[0]), max(edge_index[1]))
    G.add_nodes_from(range(max_node + 1))
    
    # Calculate global metrics
    clustering = nx.clustering(G)
    betweenness = nx.betweenness_centrality(G)
    degrees = dict(G.degree())
    
    # Initialize results dictionary
    results = {node_type: {
        'degree': [],
        'clustering': [],
        'betweenness': [],
        'neighborhood_diversity': [],
        'memorization_scores': [],
        'neighborhood_memorization': []
    } for node_type in nodes_dict.keys()}
    
    # Create node type mapping for neighborhood analysis
    node_type_map = {}
    for node_type, nodes in nodes_dict.items():
        for node in nodes:
            node_type_map[node] = node_type
    
    # Create score mapping
    score_map = {}
    for node_type, data in memorization_scores.items():
        for node, score in zip(nodes_dict[node_type], data['mem_scores']):
            score_map[node] = score
    
    # Collect metrics for each node type
    for node_type, nodes in nodes_dict.items():
        for node in nodes:
            if node in G:
                # Basic metrics
                results[node_type]['degree'].append(degrees[node])
                results[node_type]['clustering'].append(clustering[node])
                results[node_type]['betweenness'].append(betweenness[node])
                results[node_type]['memorization_scores'].append(score_map[node])
                
                # Analyze neighborhood diversity
                neighbors = list(G.neighbors(node))
                if neighbors:
                    neighbor_types = [node_type_map.get(n, 'unknown') for n in neighbors]
                    type_counts = {t: neighbor_types.count(t) for t in set(neighbor_types)}
                    diversity = len(type_counts) / len(nodes_dict.keys())  # Normalized by possible types
                    results[node_type]['neighborhood_diversity'].append(diversity)
                    
                    # Calculate average memorization score of neighbors
                    neighbor_scores = [score_map[n] for n in neighbors if n in score_map]
                    if neighbor_scores:
                        results[node_type]['neighborhood_memorization'].append(np.mean(neighbor_scores))
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Structural Properties Analysis')
    
    # Plot 1: Degree Distribution
    ax = axes[0, 0]
    for node_type in nodes_dict.keys():
        sns.kdeplot(
            results[node_type]['degree'],
            ax=ax,
            label=node_type,
            fill=True,
            alpha=0.3
        )
    ax.set_xlabel('Node Degree')
    ax.set_ylabel('Density')
    ax.set_title('Degree Distribution by Node Type')
    ax.legend()
    
    # Plot 2: Clustering Coefficient vs Memorization
    ax = axes[0, 1]
    for node_type in nodes_dict.keys():
        ax.scatter(
            results[node_type]['clustering'],
            results[node_type]['memorization_scores'],
            alpha=0.5,
            label=node_type
        )
    ax.set_xlabel('Clustering Coefficient')
    ax.set_ylabel('Memorization Score')
    ax.set_title('Clustering Coefficient vs Memorization')
    ax.legend()
    
    # Plot 3: Betweenness Centrality vs Memorization
    ax = axes[1, 0]
    for node_type in nodes_dict.keys():
        ax.scatter(
            results[node_type]['betweenness'],
            results[node_type]['memorization_scores'],
            alpha=0.5,
            label=node_type
        )
    ax.set_xlabel('Betweenness Centrality')
    ax.set_ylabel('Memorization Score')
    ax.set_title('Betweenness Centrality vs Memorization')
    ax.legend()
    
    # Plot 4: Neighborhood Diversity vs Memorization
    ax = axes[1, 1]
    for node_type in nodes_dict.keys():
        ax.scatter(
            results[node_type]['neighborhood_diversity'],
            results[node_type]['memorization_scores'],
            alpha=0.5,
            label=node_type
        )
    ax.set_xlabel('Neighborhood Diversity')
    ax.set_ylabel('Memorization Score')
    ax.set_title('Neighborhood Diversity vs Memorization')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate summary statistics
    summary_stats = {}
    metrics = ['degree', 'clustering', 'betweenness', 'neighborhood_diversity', 
               'memorization_scores', 'neighborhood_memorization']
    
    for node_type in nodes_dict.keys():
        summary_stats[node_type] = {}
        for metric in metrics:
            values = results[node_type][metric]
            if values:
                summary_stats[node_type][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values)
                }
    
    return summary_stats

def analyze_local_nli(
    edge_index: torch.Tensor,
    nodes_dict: Dict[str, List[int]],
    memorization_scores: Dict[str, Dict],
    node_labels: torch.Tensor,
    save_path: str,
    k_hops: int = 3
) -> Dict:
    """
    Analyze local node label informativeness (NLI) for each node type
    using k-hop subgraphs around each node.
    """
    # Create NetworkX graph
    edge_list = edge_index.t().tolist()
    G = nx.Graph()
    G.add_edges_from(edge_list)
    
    # Initialize results
    results = {node_type: {
        'nli_scores': [],
        'memorization_scores': [],
        'avg_nli': 0.0,
        'valid_nodes': 0,  # Track number of valid calculations
        'failed_nodes': 0  # Track number of failed calculations
    } for node_type in nodes_dict.keys()}
    
    # Process each node type
    for node_type, nodes in nodes_dict.items():
        for node in nodes:
            try:
                # Get k-hop subgraph
                subset, edge_index_sub, mapping, edge_mask = k_hop_subgraph(
                    node_idx=node,
                    num_hops=k_hops,
                    edge_index=edge_index,
                    relabel_nodes=True,
                    num_nodes=len(node_labels)
                )
                
                # Skip if subgraph is too small
                if len(subset) <= 1:
                    results[node_type]['failed_nodes'] += 1
                    continue
                
                # Create subgraph and remap labels
                sub_G = nx.Graph()
                sub_G.add_edges_from(edge_index_sub.t().tolist())
                
                # Skip if no edges
                if sub_G.number_of_edges() == 0:
                    results[node_type]['failed_nodes'] += 1
                    continue
                
                sub_labels = node_labels[subset].cpu().numpy()
                
                # Calculate NLI for subgraph
                nli = li_node(sub_G, sub_labels)
                
                # Only store valid NLI scores
                if np.isfinite(nli):
                    results[node_type]['nli_scores'].append(nli)
                    results[node_type]['memorization_scores'].append(
                        memorization_scores[node_type]['mem_scores'][nodes.index(node)]
                    )
                    results[node_type]['valid_nodes'] += 1
                else:
                    results[node_type]['failed_nodes'] += 1
                    
            except Exception as e:
                results[node_type]['failed_nodes'] += 1
                continue
        
        # Calculate average NLI
        if results[node_type]['nli_scores']:
            results[node_type]['avg_nli'] = np.mean(results[node_type]['nli_scores'])
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Plot 1: NLI Distribution by Node Type
    plt.subplot(1, 3, 1)
    valid_types = []
    for node_type in nodes_dict.keys():
        if results[node_type]['nli_scores']:
            sns.kdeplot(
                results[node_type]['nli_scores'],
                label=f"{node_type} (n={len(results[node_type]['nli_scores'])})",
                fill=True,
                alpha=0.3
            )
            valid_types.append(node_type)
    
    plt.xlabel('Local NLI Score')
    plt.ylabel('Density')
    plt.title('Distribution of Local NLI Scores')
    if valid_types:
        plt.legend()
    
    # Plot 2: NLI vs Memorization Scatter
    plt.subplot(1, 3, 2)
    for node_type in valid_types:
        plt.scatter(
            results[node_type]['nli_scores'],
            results[node_type]['memorization_scores'],
            alpha=0.5,
            label=node_type
        )
    plt.xlabel('Local NLI Score')
    plt.ylabel('Memorization Score')
    plt.title('NLI vs Memorization')
    if valid_types:
        plt.legend()
    
    # Plot 3: Box Plot of NLI by Node Type
    plt.subplot(1, 3, 3)
    data = [results[nt]['nli_scores'] for nt in valid_types]
    if data and all(len(d) > 0 for d in data):
        plt.boxplot(data, labels=valid_types)
        plt.ylabel('Local NLI Score')
        plt.title('NLI Score Distribution by Node Type')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate correlations only for valid data
    correlations = {}
    for node_type in valid_types:
        if len(results[node_type]['nli_scores']) > 1:
            correlation = stats.pearsonr(
                results[node_type]['nli_scores'],
                results[node_type]['memorization_scores']
            )
            correlations[node_type] = {
                'correlation': correlation[0],
                'p_value': correlation[1]
            }
    
    results['correlations'] = correlations
    return results

def run_post_hoc_analysis(
    edge_index: torch.Tensor,
    nodes_dict: Dict[str, List[int]],
    memorization_scores: Dict[str, Dict],
    output_dir: str,
    node_labels: torch.Tensor = None,  # Add node_labels parameter
) -> Dict:
    """
    Run all post-hoc analyses and save visualizations.
    
    Args:
        edge_index: PyG edge_index tensor
        nodes_dict: Dictionary mapping node types to node indices
        memorization_scores: Dictionary containing memorization scores for each node type
        output_dir: Directory to save analysis outputs
        node_labels: Tensor containing original node labels (optional)
        
    Returns:
        Dictionary containing analysis results
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # 1. Create node visualization
    viz_path = os.path.join(output_dir, 'node_visualization.png')
    create_node_visualization(edge_index, nodes_dict, memorization_scores, viz_path)
    results['visualization_path'] = viz_path
    
    # 2. Analyze score clustering
    clustering_path = os.path.join(output_dir, 'score_clustering.png')
    clustering_results = analyze_score_clustering(
        edge_index, nodes_dict, memorization_scores, clustering_path
    )
    results['clustering'] = clustering_results
    
    # 3. Analyze neighbor composition
    neighbor_path = os.path.join(output_dir, 'neighbor_composition.png')
    neighbor_results = analyze_neighbor_composition(
        edge_index, nodes_dict, memorization_scores, neighbor_path
    )
    results['neighbor_composition'] = neighbor_results
    
    # 4. Analyze structural properties
    structural_path = os.path.join(output_dir, 'structural_properties.png')
    structural_results = analyze_structural_properties(
        edge_index, nodes_dict, memorization_scores, structural_path
    )
    results['structural_properties'] = structural_results
    
    # 5. Analyze distance-based memorization
    distance_mem_path = os.path.join(output_dir, 'distance_memorization.png')
    distance_mem_results = analyze_distance_based_memorization(
        edge_index, nodes_dict, memorization_scores, distance_mem_path
    )
    results['distance_memorization'] = distance_mem_results
    
    # 6. Analyze general distance effects (original)
    distance_path = os.path.join(output_dir, 'distance_effects.png')
    distance_results = analyze_distance_effects(
        edge_index, nodes_dict, memorization_scores, distance_path
    )
    results['distance_effects'] = distance_results
    
    # 7. Perform statistical tests
    statistical_results = perform_statistical_tests(nodes_dict, memorization_scores)
    results['statistical_tests'] = statistical_results
    
    # Create statistical test visualization
    stats_viz_path = os.path.join(output_dir, 'statistical_tests.png')
    
    # Visualize statistical test results
    plt.figure(figsize=(12, 6))
    
    # Plot 1: Boxplot of memorization scores by node type
    plt.subplot(1, 2, 1)
    data_to_plot = []
    labels = []
    for node_type in memorization_scores:
        data_to_plot.append(memorization_scores[node_type]['mem_scores'])
        labels.append(node_type)
    
    plt.boxplot(data_to_plot, labels=labels)
    plt.title('Distribution of Memorization Scores by Node Type')
    plt.ylabel('Memorization Score')
    plt.xticks(rotation=45)
    
    # Plot 2: Heatmap of p-values from t-tests
    plt.subplot(1, 2, 2)
    node_types = list(memorization_scores.keys())
    n_types = len(node_types)
    p_values = np.zeros((n_types, n_types))
    np.fill_diagonal(p_values, 1)
    
    for i, type1 in enumerate(node_types):
        for j, type2 in enumerate(node_types):
            if i < j:
                key = f'{type1}_vs_{type2}'
                p_val = statistical_results['pairwise_ttests'][key]['p_value']
                p_values[i, j] = p_val
                p_values[j, i] = p_val
    
    sns.heatmap(p_values, annot=True, fmt='.3f', 
                xticklabels=node_types, yticklabels=node_types,
                cmap='RdYlBu_r', vmin=0, vmax=1)
    plt.title('P-values from Pairwise T-tests')
    
    plt.tight_layout()
    plt.savefig(stats_viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    results['statistical_visualization'] = stats_viz_path
    
    # Add NLI analysis if node labels are provided
    if node_labels is not None:
        nli_path = os.path.join(output_dir, 'nli_analysis.png')
        nli_results = analyze_local_nli(
            edge_index=edge_index,
            nodes_dict=nodes_dict,
            memorization_scores=memorization_scores,
            node_labels=node_labels,
            save_path=nli_path
        )
        results['nli_analysis'] = nli_results
    
    return results