import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from scipy import stats
import os
from typing import Dict, List, Tuple, Union, Any

class MLP(nn.Module):
    """
    Simple MLP model for node classification using only node features.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
    def reset_parameters(self) -> None:
        """Reset model parameters."""
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

def train_feature_mlp(
    data,
    train_indices: List[int],
    hidden_dim: int = 64,
    epochs: int = 100,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    device = None,
    logger = None
) -> nn.Module:
    """
    Train a simple MLP on node features without using graph structure.
    
    Args:
        data: PyG Data object containing node features and labels
        train_indices: List of node indices to use for training
        hidden_dim: Size of hidden dimension in the MLP
        epochs: Number of training epochs
        lr: Learning rate for optimization
        weight_decay: Weight decay for regularization
        device: PyTorch device to use
        logger: Optional logger for recording training progress
        
    Returns:
        Trained MLP model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert train indices to torch tensor if needed
    if not isinstance(train_indices, torch.Tensor):
        train_indices = torch.tensor(train_indices, device=device)
    
    # Create MLP model
    input_dim = data.x.shape[1]
    num_classes = int(data.y.max().item()) + 1
    model = MLP(input_dim, hidden_dim, num_classes).to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Create train mask
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    train_mask[train_indices] = True
    
    # Extract training data
    x_train = data.x[train_mask].to(device)
    y_train = data.y[train_mask].to(device)
    
    # Set model to training mode
    model.train()
    
    # Training loop
    if logger:
        logger.info(f"Training feature-only MLP for {epochs} epochs...")
        
    pbar = tqdm(range(epochs), desc='Training MLP')
    for epoch in pbar:
        optimizer.zero_grad()
        
        # Forward pass
        out = model(x_train)
        loss = F.cross_entropy(out, y_train)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        # Calculate training accuracy
        pred = out.argmax(dim=1)
        acc = (pred == y_train).sum().item() / len(y_train)
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{acc:.4f}"})
            if logger:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Acc: {acc:.4f}")
    
    return model

def calculate_confidences(
    gnn_model,
    mlp_model,
    data,
    device=None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate confidence scores using the GNN model and the baseline MLP model.
    
    Args:
        gnn_model: Trained GNN model (model_f)
        mlp_model: Trained baseline MLP model
        data: PyG Data object
        device: PyTorch device to use
        
    Returns:
        Tuple of (GNN confidences, MLP confidences, Alignment scores)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Ensure models are in evaluation mode
    gnn_model.eval()
    mlp_model.eval()
    
    # Get true labels
    y_true = data.y.to(device)
    
    with torch.no_grad():
        # Get GNN predictions
        gnn_logits = gnn_model(data.x.to(device), data.edge_index.to(device))
        gnn_probs = F.softmax(gnn_logits, dim=-1)
        
        # Get MLP predictions
        mlp_logits = mlp_model(data.x.to(device))
        mlp_probs = F.softmax(mlp_logits, dim=-1)
        
        # Extract confidence for true class
        conf_gnn = torch.zeros(data.num_nodes, device=device)
        conf_mlp = torch.zeros(data.num_nodes, device=device)
        
        for i in range(data.num_nodes):
            true_label = y_true[i]
            conf_gnn[i] = gnn_probs[i, true_label]
            conf_mlp[i] = mlp_probs[i, true_label]
    
    # Calculate alignment score
    alignment_score = conf_gnn - conf_mlp
    
    # Convert to numpy for analysis
    conf_gnn_np = conf_gnn.cpu().numpy()
    conf_mlp_np = conf_mlp.cpu().numpy()
    alignment_score_np = alignment_score.cpu().numpy()
    
    return conf_gnn_np, conf_mlp_np, alignment_score_np

def analyze_alignment_scores(
    conf_gnn: np.ndarray,
    conf_mlp: np.ndarray,
    alignment_score: np.ndarray,
    node_scores: Dict[str, Dict],
    nodes_dict: Dict[str, List[int]],
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Analyze alignment scores and confidence differences.
    
    Args:
        conf_gnn: GNN confidence scores for true labels
        conf_mlp: MLP confidence scores for true labels
        alignment_score: Difference between GNN and MLP confidences
        node_scores: Dictionary containing memorization scores by node type
        nodes_dict: Dictionary mapping node types to lists of indices
        threshold: Threshold for memorization
        
    Returns:
        Dictionary with analysis results
    """
    results = {}
    
    # Process each node type
    for node_type, nodes in nodes_dict.items():
        if node_type in ['val', 'test']:  # Skip validation and test nodes
            continue
        
        # Skip node type if not in node_scores
        if node_type not in node_scores:
            continue
        
        # Get memorization scores from node_scores
        node_data = node_scores[node_type]['raw_data']
        node_indices = node_data['node_idx'].values
        mem_scores = node_data['mem_score'].values
        
        # Create masks for memorized vs non-memorized
        memorized_mask = mem_scores > threshold
        non_memorized_mask = ~memorized_mask
        
        # Get alignment scores and confidences for this node type
        align_scores = np.array([alignment_score[idx] for idx in node_indices])
        feature_conf = np.array([conf_mlp[idx] for idx in node_indices])
        gnn_conf = np.array([conf_gnn[idx] for idx in node_indices])
        
        # Split by memorization
        align_mem = align_scores[memorized_mask]
        align_non_mem = align_scores[non_memorized_mask]
        
        feat_mem = feature_conf[memorized_mask]
        feat_non_mem = feature_conf[non_memorized_mask]
        
        gnn_mem = gnn_conf[memorized_mask]
        gnn_non_mem = gnn_conf[non_memorized_mask]
        
        # Perform statistical tests if both groups have data
        if len(align_mem) > 0 and len(align_non_mem) > 0:
            # For alignment scores
            t_align, p_align = stats.ttest_ind(align_mem, align_non_mem, equal_var=False)
            
            # Calculate effect size (Cohen's d)
            mean_diff_align = np.mean(align_mem) - np.mean(align_non_mem)
            pooled_std_align = np.sqrt((np.std(align_mem)**2 + np.std(align_non_mem)**2) / 2)
            effect_size_align = abs(mean_diff_align) / pooled_std_align if pooled_std_align > 0 else 0
            
            # For feature confidence
            t_feat, p_feat = stats.ttest_ind(feat_mem, feat_non_mem, equal_var=False)
            
            # Calculate effect size (Cohen's d)
            mean_diff_feat = np.mean(feat_mem) - np.mean(feat_non_mem)
            pooled_std_feat = np.sqrt((np.std(feat_mem)**2 + np.std(feat_non_mem)**2) / 2)
            effect_size_feat = abs(mean_diff_feat) / pooled_std_feat if pooled_std_feat > 0 else 0
        else:
            t_align, p_align = None, None
            effect_size_align = None
            t_feat, p_feat = None, None
            effect_size_feat = None
        
        # Store results
        results[node_type] = {
            'align_all': align_scores,
            'align_memorized': align_mem,
            'align_non_memorized': align_non_mem,
            'feature_all': feature_conf,
            'feature_memorized': feat_mem,
            'feature_non_memorized': feat_non_mem,
            'gnn_all': gnn_conf,
            'gnn_memorized': gnn_mem,
            'gnn_non_memorized': gnn_non_mem,
            'align_stats': {
                'memorized': {
                    'count': len(align_mem),
                    'mean': np.mean(align_mem) if len(align_mem) > 0 else float('nan'),
                    'std': np.std(align_mem) if len(align_mem) > 0 else float('nan'),
                },
                'non_memorized': {
                    'count': len(align_non_mem),
                    'mean': np.mean(align_non_mem) if len(align_non_mem) > 0 else float('nan'),
                    'std': np.std(align_non_mem) if len(align_non_mem) > 0 else float('nan'),
                },
                't_statistic': t_align,
                'p_value': p_align,
                'effect_size': effect_size_align
            },
            'feature_stats': {
                'memorized': {
                    'count': len(feat_mem),
                    'mean': np.mean(feat_mem) if len(feat_mem) > 0 else float('nan'),
                    'std': np.std(feat_mem) if len(feat_mem) > 0 else float('nan'),
                },
                'non_memorized': {
                    'count': len(feat_non_mem),
                    'mean': np.mean(feat_non_mem) if len(feat_non_mem) > 0 else float('nan'),
                    'std': np.std(feat_non_mem) if len(feat_non_mem) > 0 else float('nan'),
                },
                't_statistic': t_feat,
                'p_value': p_feat,
                'effect_size': effect_size_feat
            }
        }
    
    return results

def plot_alignment_comparison(
    results: Dict[str, Dict],
    save_dir: str,
    timestamp: str,
    model_type: str,
    dataset_name: str
) -> None:
    """
    Create visualizations comparing alignment scores and feature confidence 
    between memorized and non-memorized nodes.
    
    Args:
        results: Dictionary with results from analyze_alignment_scores
        save_dir: Directory to save the plots
        timestamp: Timestamp string for filenames
        model_type: Model type (e.g., 'gcn')
        dataset_name: Dataset name
    """
    # Plot alignment score comparison for each node type
    color_scheme = {
        'Memorized': '#FF9999',     # Light red
        'Non-memorized': '#66B2FF'  # Light blue
    }
    
    for metric_name, metric_key, ylabel in [
        ('alignment', 'align', 'GNN Confidence - MLP Confidence'),
        ('feature', 'feature', 'MLP Feature Confidence')
    ]:
        # Create a figure with grid of plots - one for each node type
        node_types = list(results.keys())
        n_plots = len(node_types)
        
        if n_plots == 0:
            continue
        
        # Determine grid size
        n_cols = min(2, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        plt.figure(figsize=(n_cols * 6, n_rows * 5))
        
        for i, node_type in enumerate(node_types):
            plt.subplot(n_rows, n_cols, i + 1)
            
            result = results[node_type]
            stats_key = f'{metric_key}_stats'
            
            # Create DataFrame for plotting
            data = []
            
            # Add memorized nodes
            mem_data = result[f'{metric_key}_memorized']
            for val in mem_data:
                data.append({'Group': 'Memorized', 'Value': val})
            
            # Add non-memorized nodes
            non_mem_data = result[f'{metric_key}_non_memorized']
            for val in non_mem_data:
                data.append({'Group': 'Non-memorized', 'Value': val})
            
            df = pd.DataFrame(data)
            
            # Create the box plot
            if not df.empty:
                sns.boxplot(x='Group', y='Value', data=df, palette=color_scheme)
                
                # Add strip plot for individual points
                sns.stripplot(x='Group', y='Value', data=df, 
                            size=4, color='.3', alpha=0.6, jitter=True)
                
                # Add statistics
                mem_mean = result[stats_key]['memorized']['mean']
                non_mem_mean = result[stats_key]['non_memorized']['mean']
                mem_count = result[stats_key]['memorized']['count']
                non_mem_count = result[stats_key]['non_memorized']['count']
                p_val = result[stats_key]['p_value']
                
                # Add significance marker if p-value is available
                sig_marker = ''
                if p_val is not None:
                    if p_val < 0.001:
                        sig_marker = '***'
                    elif p_val < 0.01:
                        sig_marker = '**'
                    elif p_val < 0.05:
                        sig_marker = '*'
                
                # Add means to plot
                if mem_count > 0:
                    plt.text(0, plt.ylim()[1] * 0.9, 
                            f"Mean: {mem_mean:.4f}\nN: {mem_count}", 
                            ha='center', va='top')
                if non_mem_count > 0:
                    plt.text(1, plt.ylim()[1] * 0.9, 
                            f"Mean: {non_mem_mean:.4f}\nN: {non_mem_count}", 
                            ha='center', va='top')
                
                # Add effect size if available
                effect_size = result[stats_key]['effect_size']
                effect_interp = "N/A"
                if effect_size is not None:
                    if effect_size < 0.2:
                        effect_interp = "negligible"
                    elif effect_size < 0.5:
                        effect_interp = "small"
                    elif effect_size < 0.8:
                        effect_interp = "medium"
                    else:
                        effect_interp = "large"
                
                if p_val is not None:
                    plt.title(f"{node_type.capitalize()} Nodes\np = {p_val:.4e} {sig_marker} (Effect: {effect_interp})")
                else:
                    plt.title(f"{node_type.capitalize()} Nodes")
                
                plt.ylabel(ylabel)
                plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        plt.tight_layout()
        plt.suptitle(f"{metric_name.capitalize()} Analysis - {dataset_name}, {model_type.upper()}", 
                    fontsize=14, y=1.02)
        
        # Save the plot
        save_path = os.path.join(
            save_dir, 
            f'{metric_name}_analysis_{model_type}_{dataset_name}_{timestamp}.png'
        )
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def run_aggregation_alignment_analysis(
    model_f,
    data,
    nodes_dict: Dict[str, List[int]],
    node_scores: Dict[str, Dict],
    save_dir: str,
    timestamp: str,
    model_type: str,
    dataset_name: str,
    threshold: float = 0.5,
    mlp_hidden_dim: int = 64,
    mlp_epochs: int = 100,
    mlp_lr: float = 0.01,
    device=None,
    logger=None
) -> Dict[str, Any]:
    """
    Run comprehensive aggregation alignment analysis.
    
    Args:
        model_f: Trained GNN model
        data: PyG Data object
        nodes_dict: Dictionary mapping node types to lists of indices
        node_scores: Dictionary containing memorization scores
        save_dir: Directory to save plots
        timestamp: Timestamp string for filenames
        model_type: Model type (e.g., 'gcn')
        dataset_name: Dataset name
        threshold: Threshold for determining memorized nodes
        mlp_hidden_dim: Hidden dimension for baseline MLP
        mlp_epochs: Number of epochs to train baseline MLP
        mlp_lr: Learning rate for baseline MLP
        device: PyTorch device
        logger: Logger object
        
    Returns:
        Dictionary with analysis results
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if logger:
        logger.info("\nPerforming Aggregation Alignment analysis...")
    
    # Get training indices for model_f (S_S U S_C)
    train_indices_f = nodes_dict['shared'] + nodes_dict['candidate']
    
    # Train baseline MLP model
    mlp_model = train_feature_mlp(
        data=data,
        train_indices=train_indices_f,
        hidden_dim=mlp_hidden_dim,
        epochs=mlp_epochs,
        lr=mlp_lr,
        device=device,
        logger=logger
    )
    
    # Calculate confidences and alignment scores
    conf_gnn, conf_mlp, alignment_score = calculate_confidences(
        gnn_model=model_f,
        mlp_model=mlp_model,
        data=data,
        device=device
    )
    
    # Analyze results
    analysis_results = analyze_alignment_scores(
        conf_gnn=conf_gnn,
        conf_mlp=conf_mlp,
        alignment_score=alignment_score,
        node_scores=node_scores,
        nodes_dict=nodes_dict,
        threshold=threshold
    )
    
    # Create visualizations
    plot_alignment_comparison(
        results=analysis_results,
        save_dir=save_dir,
        timestamp=timestamp,
        model_type=model_type,
        dataset_name=dataset_name
    )
    
    # Log results if logger is provided
    if logger:
        logger.info("\nAggregation Alignment Analysis Results:")
        
        for node_type, result in analysis_results.items():
            logger.info(f"\n{node_type.capitalize()} Nodes:")
            
            # Alignment score results
            align_stats = result['align_stats']
            logger.info("\n  Alignment Score (GNN - MLP Confidence):")
            logger.info(f"    Memorized nodes (n={align_stats['memorized']['count']}): {align_stats['memorized']['mean']:.4f} ± {align_stats['memorized']['std']:.4f}")
            logger.info(f"    Non-memorized nodes (n={align_stats['non_memorized']['count']}): {align_stats['non_memorized']['mean']:.4f} ± {align_stats['non_memorized']['std']:.4f}")
            
            if align_stats['p_value'] is not None:
                logger.info(f"    Welch's t-test: t = {align_stats['t_statistic']:.4f}, p = {align_stats['p_value']:.4e}")
                
                # Format p-value with significance markers
                p_val = align_stats['p_value']
                sig_marker = ''
                if p_val < 0.001:
                    sig_marker = '***'
                elif p_val < 0.01:
                    sig_marker = '**'
                elif p_val < 0.05:
                    sig_marker = '*'
                    
                logger.info(f"    Statistical significance: {sig_marker}")
                
                # Interpret effect size
                effect_size = align_stats['effect_size']
                effect_interp = "negligible"
                if effect_size >= 0.8:
                    effect_interp = "large"
                elif effect_size >= 0.5:
                    effect_interp = "medium"
                elif effect_size >= 0.2:
                    effect_interp = "small"
                
                logger.info(f"    Effect size (Cohen's d): {effect_size:.4f} ({effect_interp})")
            
            # Feature confidence results
            feat_stats = result['feature_stats']
            logger.info("\n  MLP Feature Confidence:")
            logger.info(f"    Memorized nodes (n={feat_stats['memorized']['count']}): {feat_stats['memorized']['mean']:.4f} ± {feat_stats['memorized']['std']:.4f}")
            logger.info(f"    Non-memorized nodes (n={feat_stats['non_memorized']['count']}): {feat_stats['non_memorized']['mean']:.4f} ± {feat_stats['non_memorized']['std']:.4f}")
            
            if feat_stats['p_value'] is not None:
                logger.info(f"    Welch's t-test: t = {feat_stats['t_statistic']:.4f}, p = {feat_stats['p_value']:.4e}")
                
                # Format p-value with significance markers
                p_val = feat_stats['p_value']
                sig_marker = ''
                if p_val < 0.001:
                    sig_marker = '***'
                elif p_val < 0.01:
                    sig_marker = '**'
                elif p_val < 0.05:
                    sig_marker = '*'
                    
                logger.info(f"    Statistical significance: {sig_marker}")
                
                # Interpret effect size
                effect_size = feat_stats['effect_size']
                effect_interp = "negligible"
                if effect_size >= 0.8:
                    effect_interp = "large"
                elif effect_size >= 0.5:
                    effect_interp = "medium"
                elif effect_size >= 0.2:
                    effect_interp = "small"
                
                logger.info(f"    Effect size (Cohen's d): {effect_size:.4f} ({effect_interp})")
        
        # Log plot paths
        logger.info("\nPlots saved to:")
        logger.info(f"  - {os.path.join(save_dir, f'alignment_analysis_{model_type}_{dataset_name}_{timestamp}.png')}")
        logger.info(f"  - {os.path.join(save_dir, f'feature_analysis_{model_type}_{dataset_name}_{timestamp}.png')}")
    
    # Return combined results
    return {
        'analysis': analysis_results,
        'confidence': {
            'gnn': conf_gnn,
            'mlp': conf_mlp,
            'alignment': alignment_score
        },
        'mlp_model': mlp_model
    }