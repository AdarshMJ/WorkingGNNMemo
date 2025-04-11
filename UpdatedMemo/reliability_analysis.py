import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import entropy
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Data
import pandas as pd
from scipy.stats import mannwhitneyu
import networkx as nx
from torch_geometric.utils import to_networkx
from tqdm import tqdm

def kd_retention(model, data: Data, noise_level: float = 0.1, device=None):
    """
    Calculate reliability scores using KD retention through entropy differences
    Args:
        model: The GNN model to evaluate
        data: PyG Data object containing graph data
        noise_level: Standard deviation of Gaussian noise (default: 0.1)
        device: Device to run computations on
    Returns:
        delta_entropy: Normalized entropy differences as reliability scores
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    # Get original predictions and entropies
    with torch.no_grad():
        out_teacher = model(data.x.to(device), data.edge_index.to(device))
        data_teacher = F.softmax(out_teacher, dim=-1).cpu().numpy()
        weight_t = np.array([entropy(dt) for dt in data_teacher])
        
        # Add Gaussian noise to features
        feats_noise = copy.deepcopy(data.x)
        feats_noise += torch.randn_like(feats_noise) * noise_level
        data_noise = Data(x=feats_noise, edge_index=data.edge_index).to(device)
        
        # Get predictions on noisy data
        out_noise = model(data_noise.x, data_noise.edge_index)
        out_noise = F.softmax(out_noise, dim=-1).cpu().numpy()
        
        # Calculate entropy differences
        weight_s = np.abs(np.array([entropy(on) for on in out_noise]) - weight_t)
        delta_entropy = weight_s / np.max(weight_s)
    
    return delta_entropy

def label_perturbation_retention(model, data: Data, nodes_to_perturb, perturb_ratio: float = 0.05, 
                               lr: float = 0.01, weight_decay: float = 5e-4, num_epochs: int = 10, device=None):
    """
    Calculate reliability scores using entropy differences between predictions on original vs perturbed labels
    Args:
        model: The GNN model to evaluate
        data: PyG Data object containing graph data
        nodes_to_perturb: List of node indices that can be perturbed
        perturb_ratio: Ratio of labels to perturb (default: 0.05)
        lr: Learning rate for optimizer (default: 0.01)
        weight_decay: Weight decay for optimizer (default: 5e-4)
        num_epochs: Number of epochs to train on perturbed labels (default: 1)
        device: Device to run computations on
    Returns:
        delta_entropy: Normalized entropy differences when evaluated on perturbed labels
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Store original model state
    original_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    # Get original predictions and entropies
    model.eval()
    with torch.no_grad():
        out_orig = model(data.x.to(device), data.edge_index.to(device))
        data_orig = F.softmax(out_orig, dim=-1).cpu().numpy()
        weight_t = np.array([entropy(dt) for dt in data_orig])
    
    # Create perturbed labels data
    data_perturbed = copy.deepcopy(data)
    num_classes = data.y.max().item() + 1
    num_to_perturb = int(len(nodes_to_perturb) * perturb_ratio)
    nodes_to_perturb = np.random.choice(nodes_to_perturb, num_to_perturb, replace=False)
    
    # Create perturbed labels by randomly selecting a different class
    for node in nodes_to_perturb:
        current_label = data_perturbed.y[node].item()
        possible_labels = list(range(num_classes))
        possible_labels.remove(current_label)
        new_label = np.random.choice(possible_labels)
        data_perturbed.y[node] = new_label
    
    # Create a temporary copy of the model for training
    temp_model = copy.deepcopy(model)
    for param in temp_model.parameters():
        param.requires_grad = True
    temp_model.train()
    
    # Train model for specified number of epochs on perturbed labels
    optimizer = torch.optim.Adam(temp_model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Training loop with progress bar
    pbar = tqdm(range(num_epochs), desc='Label perturbation training')
    for _ in pbar:
        optimizer.zero_grad()
        out_perturbed = temp_model(data_perturbed.x.to(device), data_perturbed.edge_index.to(device))
        loss = F.cross_entropy(out_perturbed[nodes_to_perturb], data_perturbed.y[nodes_to_perturb].to(device))
        loss.backward()
        optimizer.step()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Get predictions with updated model
    temp_model.eval()
    with torch.no_grad():
        out_perturbed = temp_model(data.x.to(device), data.edge_index.to(device))
        out_perturbed = F.softmax(out_perturbed, dim=-1).cpu().numpy()
    
    # Calculate entropy differences
    weight_s = np.abs(np.array([entropy(op) for op in out_perturbed]) - weight_t)
    delta_entropy = weight_s / np.max(weight_s) if np.max(weight_s) > 0 else weight_s
    
    # Restore original model state
    model.load_state_dict(original_state)
    
    return delta_entropy

def analyze_reliability_vs_memorization(
    model_f,
    model_g,
    data,
    node_scores,
    noise_level: float = 0.1,
    perturb_ratio: float = 0.05,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    num_epochs: int = 1,
    device=None
):
    """
    Analyze relationship between node reliability and memorization scores using both
    feature perturbation and label perturbation approaches
    """
    # Calculate feature perturbation reliability scores
    reliability_f_feat = kd_retention(model_f, data, noise_level, device)
    reliability_g_feat = kd_retention(model_g, data, noise_level, device)
    
    # Calculate label perturbation reliability scores
    # For model f: perturb shared + candidate nodes
    f_nodes_to_perturb = []
    for node_type in ['shared', 'candidate']:
        if node_type in node_scores:
            f_nodes_to_perturb.extend(node_scores[node_type]['raw_data']['node_idx'].tolist())
    
    # For model g: perturb shared + independent nodes
    g_nodes_to_perturb = []
    for node_type in ['shared', 'independent']:
        if node_type in node_scores:
            g_nodes_to_perturb.extend(node_scores[node_type]['raw_data']['node_idx'].tolist())
    
    reliability_f_label = label_perturbation_retention(
        model_f, data, f_nodes_to_perturb, perturb_ratio, lr, weight_decay, num_epochs, device
    )
    reliability_g_label = label_perturbation_retention(
        model_g, data, g_nodes_to_perturb, perturb_ratio, lr, weight_decay, num_epochs, device
    )
    
    results = {}
    
    for node_type, scores in node_scores.items():
        if node_type in ['val', 'test']:
            continue
            
        node_data = scores['raw_data']
        memorized_mask = node_data['mem_score'] > 0.5
        
        # Get feature perturbation reliability scores
        rel_f_feat = [reliability_f_feat[idx] for idx in node_data['node_idx']]
        rel_g_feat = [reliability_g_feat[idx] for idx in node_data['node_idx']]
        rel_scores_feat = np.mean([rel_f_feat, rel_g_feat], axis=0)
        
        # Get label perturbation reliability scores
        rel_f_label = [reliability_f_label[idx] for idx in node_data['node_idx']]
        rel_g_label = [reliability_g_label[idx] for idx in node_data['node_idx']]
        rel_scores_label = np.mean([rel_f_label, rel_g_label], axis=0)
        
        # Split into memorized vs non-memorized
        mem_rel_feat = rel_scores_feat[memorized_mask]
        non_mem_rel_feat = rel_scores_feat[~memorized_mask]
        mem_rel_label = rel_scores_label[memorized_mask]
        non_mem_rel_label = rel_scores_label[~memorized_mask]
        
        # Perform statistical tests for both approaches
        if len(mem_rel_feat) > 0 and len(non_mem_rel_feat) > 0:
            stat_feat, pvalue_feat = mannwhitneyu(mem_rel_feat, non_mem_rel_feat, alternative='two-sided')
            stat_label, pvalue_label = mannwhitneyu(mem_rel_label, non_mem_rel_label, alternative='two-sided')
        else:
            stat_feat, pvalue_feat = None, None
            stat_label, pvalue_label = None, None
            
        results[node_type] = {
            'feature_perturbation': {
                'reliability_scores': rel_scores_feat,
                'memorized_reliability': mem_rel_feat,
                'non_memorized_reliability': non_mem_rel_feat,
                'stat_test': {
                    'statistic': stat_feat,
                    'pvalue': pvalue_feat
                }
            },
            'label_perturbation': {
                'reliability_scores': rel_scores_label,
                'memorized_reliability': mem_rel_label,
                'non_memorized_reliability': non_mem_rel_label,
                'stat_test': {
                    'statistic': stat_label,
                    'pvalue': pvalue_label
                }
            }
        }
    
    return results

def plot_reliability_analysis(results, save_path: str):
    """Create visualization comparing delta entropy values between memorized and non-memorized nodes
    for both feature perturbation and label perturbation approaches"""
    n_types = len(results)
    fig = plt.figure(figsize=(15, 10 * n_types))
    
    colors = {
        'Memorized': '#FF9999',     # Light red
        'Non-memorized': '#66B2FF'  # Light blue
    }
    
    for idx, (node_type, data) in enumerate(results.items(), 1):
        # Plot feature perturbation results
        plt.subplot(n_types, 2, 2*idx-1)
        feat_data = data['feature_perturbation']
        
        mem_scores = feat_data['memorized_reliability']
        non_mem_scores = feat_data['non_memorized_reliability']
        
        plot_data = [mem_scores, non_mem_scores]
        
        # Add box plots
        bp = plt.boxplot(plot_data, positions=[1, 2], widths=0.6,
                        patch_artist=True, showfliers=False)
        
        # Customize box plots
        for box, color in zip(bp['boxes'], colors.values()):
            box.set(facecolor=color, alpha=0.8)
        
        # Add individual points with jitter
        for i, (scores, pos) in enumerate(zip([mem_scores, non_mem_scores], [1, 2])):
            if len(scores) > 0:
                x_jitter = np.random.normal(pos, 0.04, size=len(scores))
                plt.scatter(x_jitter, scores,
                          alpha=0.4, color=list(colors.values())[i],
                          s=30, zorder=3)
        
        plt.title(f'{node_type.capitalize()} Nodes - Feature Delta Entropy\n' +
                 f'(Memorized: n={len(mem_scores)}, μ={np.mean(mem_scores):.3f} | ' +
                 f'Non-memorized: n={len(non_mem_scores)}, μ={np.mean(non_mem_scores):.3f})')
        plt.ylabel('Delta Entropy (|H(noise) - H(orig)|)')
        plt.xticks([1, 2], ['Memorized', 'Non-memorized'])
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Add p-value for feature perturbation
        if feat_data['stat_test']['pvalue'] is not None:
            plt.text(0.98, 0.02, f"p-value: {feat_data['stat_test']['pvalue']:.6f}",
                    transform=plt.gca().transAxes,
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        # Plot label perturbation results
        plt.subplot(n_types, 2, 2*idx)
        label_data = data['label_perturbation']
        
        mem_scores = label_data['memorized_reliability']
        non_mem_scores = label_data['non_memorized_reliability']
        
        plot_data = [mem_scores, non_mem_scores]
        
        # Add box plots
        bp = plt.boxplot(plot_data, positions=[1, 2], widths=0.6,
                        patch_artist=True, showfliers=False)
        
        # Customize box plots
        for box, color in zip(bp['boxes'], colors.values()):
            box.set(facecolor=color, alpha=0.8)
        
        # Add individual points with jitter
        for i, (scores, pos) in enumerate(zip([mem_scores, non_mem_scores], [1, 2])):
            if len(scores) > 0:
                x_jitter = np.random.normal(pos, 0.04, size=len(scores))
                plt.scatter(x_jitter, scores,
                          alpha=0.4, color=list(colors.values())[i],
                          s=30, zorder=3)
        
        plt.title(f'{node_type.capitalize()} Nodes - Label Delta Entropy\n' +
                 f'(Memorized: n={len(mem_scores)}, μ={np.mean(mem_scores):.3f} | ' +
                 f'Non-memorized: n={len(non_mem_scores)}, μ={np.mean(non_mem_scores):.3f})')
        plt.ylabel('Delta Entropy (|H(perturbed) - H(orig)|)')
        plt.xticks([1, 2], ['Memorized', 'Non-memorized'])
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Add p-value for label perturbation
        if label_data['stat_test']['pvalue'] is not None:
            plt.text(0.98, 0.02, f"p-value: {label_data['stat_test']['pvalue']:.6f}",
                    transform=plt.gca().transAxes,
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    plt.suptitle('Reliability Analysis: Feature vs Label Perturbation\n' +
                 'Higher delta entropy values indicate larger changes under perturbations',
                 fontsize=14, y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return pd.DataFrame({
        'Node Type': list(results.keys()),
        'Mean Memorized Feature Reliability': [np.mean(data['feature_perturbation']['memorized_reliability']) for data in results.values()],
        'Mean Non-memorized Feature Reliability': [np.mean(data['feature_perturbation']['non_memorized_reliability']) for data in results.values()],
        'Feature P-value': [data['feature_perturbation']['stat_test']['pvalue'] for data in results.values()],
        'Mean Memorized Label Reliability': [np.mean(data['label_perturbation']['memorized_reliability']) for data in results.values()],
        'Mean Non-memorized Label Reliability': [np.mean(data['label_perturbation']['non_memorized_reliability']) for data in results.values()],
        'Label P-value': [data['label_perturbation']['stat_test']['pvalue'] for data in results.values()]
    })

def analyze_centrality_measures(data, node_scores, save_path: str):
    """
    Analyze and plot various centrality measures vs memorization scores for candidate nodes
    Args:
        data: PyG Data object containing graph data
        node_scores: Dictionary containing memorization scores
        save_path: Base path to save plots
    """
   
    
    # Convert to networkx graph
    G = to_networkx(data, to_undirected=True)
    
    # Get candidate nodes and their memorization scores
    candidate_data = node_scores['candidate']['raw_data']
    node_indices = candidate_data['node_idx'].values
    mem_scores = candidate_data['mem_score'].values
    
    # Calculate centrality measures
    centrality_measures = {
        'degree': nx.degree_centrality(G),
        'betweenness': nx.betweenness_centrality(G),
        'eigenvector': nx.eigenvector_centrality(G, max_iter=1000),
        'pagerank': nx.pagerank(G)
    }
    
    # Create plots
    for measure_name, measure_dict in centrality_measures.items():
        plt.figure(figsize=(8, 6))
        
        # Extract centrality values for candidate nodes
        centrality_values = [measure_dict[idx] for idx in node_indices]
        
        # Create scatter plot
        plt.scatter(centrality_values, mem_scores, alpha=0.6)
        
        # Add trend line
        z = np.polyfit(centrality_values, mem_scores, 1)
        p = np.poly1d(z)
        plt.plot(sorted(centrality_values), p(sorted(centrality_values)), "r--", alpha=0.8)
        
        # Calculate correlation
        correlation = np.corrcoef(centrality_values, mem_scores)[0,1]
        
        plt.xlabel(f'{measure_name.title()} Centrality')
        plt.ylabel('Memorization Score')
        plt.title(f'{measure_name.title()} Centrality vs Memorization\nCorrelation: {correlation:.3f}')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        measure_path = save_path.replace('.png', f'_{measure_name}_centrality.png')
        plt.savefig(measure_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    return centrality_measures