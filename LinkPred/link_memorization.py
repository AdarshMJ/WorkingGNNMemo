import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import roc_auc_score
import random
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GATConv, GraphConv
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import negative_sampling
import logging
from datetime import datetime
import os.path as osp
from dataloader import load_npz_dataset, process_heterophilic_dataset_for_link_prediction, get_heterophilic_datasets

# Set up device
if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class LinkGNN(torch.nn.Module):
    """Base class for GNN models for link prediction."""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, model_type='gcn', **kwargs):
        super().__init__()
        self.model_type = model_type.lower()
        
        # Select the appropriate GNN layer based on model type
        if self.model_type == 'gcn':
            GNNLayer = GCNConv
            self.kwargs = {}
        elif self.model_type == 'gat':
            GNNLayer = GATConv
            self.kwargs = {'heads': kwargs.get('heads', 4)}
        elif self.model_type == 'graphconv':
            GNNLayer = GraphConv
            self.kwargs = {}
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create encoder layers
        self.convs = torch.nn.ModuleList()
        
        # Input layer
        self.convs.append(GNNLayer(in_channels, hidden_channels, **self.kwargs))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            if self.model_type == 'gat':
                # For GAT, consider the number of heads in hidden dimension
                self.convs.append(
                    GNNLayer(hidden_channels * self.kwargs['heads'], 
                             hidden_channels, **self.kwargs)
                )
            else:
                self.convs.append(
                    GNNLayer(hidden_channels, hidden_channels, **self.kwargs)
                )
        
        # Output layer
        if self.model_type == 'gat' and num_layers > 1:
            self.convs.append(
                GNNLayer(hidden_channels * self.kwargs['heads'], 
                         out_channels, **self.kwargs)
            )
        else:
            self.convs.append(
                GNNLayer(hidden_channels, out_channels, **self.kwargs)
            )

    def encode(self, x, edge_index):
        """Encode node features to embeddings."""
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = x.relu()
            if self.model_type != 'gat':  # GAT already applied activation in the layer
                x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        
        # Last layer without dropout
        x = self.convs[-1](x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        """Predict links based on node embeddings."""
        # Dot product of node embeddings for the edges
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        """Decode all possible edges."""
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

def setup_logging(model_type, dataset_name, timestamp=None):
    """Set up logging directory and file."""
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create directory structure
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    log_dir = os.path.join(results_dir, f"{model_type}_{dataset_name}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup logger
    logger = logging.getLogger('link_memorization')
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers
    
    # Create file handler
    log_file = os.path.join(log_dir, f'link_memorization_{model_type}_{dataset_name}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    
    return logger, log_dir

def load_dataset(dataset_name, transform):
    """Load PyG dataset with transformation."""
    if dataset_name.lower() in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root='data', name=dataset_name, transform=transform)
        return dataset
    elif dataset_name.lower() in get_heterophilic_datasets():
        # Load heterophilic dataset
        data = load_npz_dataset(dataset_name)
        
        # Apply normalization and device transformation first
        normalize_transform = T.Compose([
            T.NormalizeFeatures(),
            T.ToDevice(device)
        ])
        data = normalize_transform(data)
        
        # Now apply the RandomLinkSplit
        link_split = T.RandomLinkSplit(
            num_val=0.05, 
            num_test=0.1, 
            is_undirected=True,
            add_negative_train_samples=False
        )
        
        # Apply the split and return data as a "dataset" with a single entry
        train_data, val_data, test_data = link_split(data)
        return [(train_data, val_data, test_data)]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

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

def create_training_data(data, edges_dict, edge_type):
    """Create training data for model f (shared+candidate) or model g (shared+independent)."""
    if edge_type not in ['f', 'g']:
        raise ValueError("edge_type must be 'f' or 'g'")
    
    # Common shared edges
    shared_pos_edges = torch.tensor(edges_dict['shared']['pos'], dtype=torch.long).t().to(device)
    shared_neg_edges = torch.tensor(edges_dict['shared']['neg'], dtype=torch.long).t().to(device)
    
    # Select appropriate edges based on model type
    if edge_type == 'f':
        # Model f: shared + candidate
        other_pos_edges = torch.tensor(edges_dict['candidate']['pos'], dtype=torch.long).t().to(device)
        other_neg_edges = torch.tensor(edges_dict['candidate']['neg'], dtype=torch.long).t().to(device)
    else:
        # Model g: shared + independent
        other_pos_edges = torch.tensor(edges_dict['independent']['pos'], dtype=torch.long).t().to(device)
        other_neg_edges = torch.tensor(edges_dict['independent']['neg'], dtype=torch.long).t().to(device)
    
    # Combine edges
    pos_edge_index = torch.cat([shared_pos_edges, other_pos_edges], dim=1)
    neg_edge_index = torch.cat([shared_neg_edges, other_neg_edges], dim=1)
    
    # Set up edge indices for training
    train_edges = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    edge_labels = torch.cat([
        torch.ones(pos_edge_index.size(1), dtype=torch.float, device=device),
        torch.zeros(neg_edge_index.size(1), dtype=torch.float, device=device)
    ])
    
    return train_edges, edge_labels

def train_link_prediction_model(model, optimizer, data, train_edges, edge_labels, epochs=100):
    """Train a link prediction model."""
    criterion = torch.nn.BCEWithLogitsLoss()
    best_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Get node embeddings
        z = model.encode(data.x, data.edge_index)
        
        # Make predictions
        out = model.decode(z, train_edges).view(-1)
        loss = criterion(out, edge_labels)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        # Save best model
        if loss < best_loss:
            best_loss = loss
            best_model_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}')
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model

@torch.no_grad()
def evaluate_model(model, data, edge_label_index, edge_label):
    """Evaluate model using AUC-ROC."""
    model.eval()
    z = model.encode(data.x, data.edge_index)
    pred = model.decode(z, edge_label_index).sigmoid()
    return roc_auc_score(edge_label.cpu().numpy(), pred.cpu().numpy())

def train_models(data_full, edges_dict, args, logger):
    """
    Train model f (on shared+candidate) and model g (on shared+independent) with multiple seeds.
    
    Args:
        data_full: Original data object with all graph information
        edges_dict: Dictionary with edge splits
        args: Arguments dictionary with model parameters
        logger: Logger object
    
    Returns:
        Tuple of (model_f, model_g) - best models based on validation accuracy
    """
    # Extract training data
    train_data, val_data, test_data = data_full[0]
    
    # Create training data for model f (shared+candidate)
    train_edges_f, edge_labels_f = create_training_data(train_data, edges_dict, 'f')
    
    # Create training data for model g (shared+independent)
    train_edges_g, edge_labels_g = create_training_data(train_data, edges_dict, 'g')
    
    # Log information about the splits
    logger.info("\nTraining Set Information:")
    logger.info(f"Model f training edges: {len(edge_labels_f)}")
    logger.info(f"  - Positive edges: {torch.sum(edge_labels_f).item()}")
    logger.info(f"  - Negative edges: {len(edge_labels_f) - torch.sum(edge_labels_f).item()}")
    
    logger.info(f"\nModel g training edges: {len(edge_labels_g)}")
    logger.info(f"  - Positive edges: {torch.sum(edge_labels_g).item()}")
    logger.info(f"  - Negative edges: {len(edge_labels_g) - torch.sum(edge_labels_g).item()}")
    
    # Create models
    in_channels = train_data.num_features
    hidden_channels = args.hidden_dim
    out_channels = args.out_dim
    
    logger.info("\nModel Architecture Details:")
    logger.info(f"Model Type: {args.model_type.upper()}")
    logger.info(f"Input Features: {in_channels}")
    logger.info(f"Hidden Dimensions: {hidden_channels}")
    logger.info(f"Output Dimensions: {out_channels}")
    logger.info(f"Number of Layers: {args.num_layers}")
    
    # Define seeds for multiple training runs
    seeds = [42, 123, 456]
    logger.info(f"Training with seeds: {seeds}")
    
    # Lists to store models and validation scores
    f_models = []
    g_models = []
    f_val_aucs = []
    g_val_aucs = []
    
    # Train multiple models with different seeds
    for seed_idx, seed in enumerate(seeds):
        logger.info(f"\nTraining models with seed {seed} ({seed_idx+1}/{len(seeds)})...")
        
        # Set the seed for reproducibility
        set_seed(seed)
        
        # Initialize models for this seed
        model_f = LinkGNN(
            in_channels=in_channels, 
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=args.num_layers,
            model_type=args.model_type,
            heads=args.heads if args.model_type.lower() == 'gat' else None
        ).to(device)
        
        model_g = LinkGNN(
            in_channels=in_channels, 
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=args.num_layers,
            model_type=args.model_type,
            heads=args.heads if args.model_type.lower() == 'gat' else None
        ).to(device)
        
        # Create optimizers
        optimizer_f = torch.optim.Adam(model_f.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_g = torch.optim.Adam(model_g.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        # Train models
        logger.info("Training model f (shared+candidate)...")
        model_f = train_link_prediction_model(
            model_f, optimizer_f, train_data, train_edges_f, edge_labels_f, epochs=args.epochs
        )
        
        logger.info("Training model g (shared+independent)...")
        model_g = train_link_prediction_model(
            model_g, optimizer_g, train_data, train_edges_g, edge_labels_g, epochs=args.epochs
        )
        
        # Evaluate on validation set
        val_edge_index = torch.cat([
            torch.tensor(edges_dict['val']['pos'], dtype=torch.long).t().to(device),
            torch.tensor(edges_dict['val']['neg'], dtype=torch.long).t().to(device)
        ], dim=1)
        
        val_edge_label = torch.cat([
            torch.ones(len(edges_dict['val']['pos']), device=device),
            torch.zeros(len(edges_dict['val']['neg']), device=device)
        ])
        
        # Evaluate models on validation set
        val_auc_f = evaluate_model(model_f, train_data, val_edge_index, val_edge_label)
        val_auc_g = evaluate_model(model_g, train_data, val_edge_index, val_edge_label)
        
        logger.info(f"Validation AUC - Model f: {val_auc_f:.4f}, Model g: {val_auc_g:.4f}")
        
        # Store models and validation scores
        f_models.append(model_f.state_dict())
        g_models.append(model_g.state_dict())
        f_val_aucs.append(val_auc_f)
        g_val_aucs.append(val_auc_g)

    # Select best models based on validation AUC
    best_f_idx = np.argmax(f_val_aucs)
    best_g_idx = np.argmax(g_val_aucs)
    
    logger.info("\nSelecting best models based on validation AUC:")
    logger.info(f"Best model f seed: {seeds[best_f_idx]}, Validation AUC: {f_val_aucs[best_f_idx]:.4f}")
    logger.info(f"Best model g seed: {seeds[best_g_idx]}, Validation AUC: {g_val_aucs[best_g_idx]:.4f}")
    
    # Initialize models with the best weights
    best_model_f = LinkGNN(
        in_channels=in_channels, 
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=args.num_layers,
        model_type=args.model_type,
        heads=args.heads if args.model_type.lower() == 'gat' else None
    ).to(device)
    
    best_model_g = LinkGNN(
        in_channels=in_channels, 
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=args.num_layers,
        model_type=args.model_type,
        heads=args.heads if args.model_type.lower() == 'gat' else None
    ).to(device)
    
    best_model_f.load_state_dict(f_models[best_f_idx])
    best_model_g.load_state_dict(g_models[best_g_idx])
    
    return best_model_f, best_model_g

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
    
    num_bins = 20
    threshold = 0.5
    
    # Plot 1: Memorization score distribution for positive edges
    plt.figure(figsize=(12, 8))
    plt.subplot(211)
    
    # Create histogram with frequency counts for positive edges
    for edge_type in edge_types_to_plot:
        if edge_type in edge_scores:
            scores = edge_scores[edge_type]['positive_edges']['mem_scores']
            mean_score = edge_scores[edge_type]['positive_edges']['avg_score']
            edges_above = edge_scores[edge_type]['positive_edges']['above_threshold']
            total_edges = edge_scores[edge_type]['positive_edges']['count']
            percentage_above = edge_scores[edge_type]['positive_edges']['percentage_above']
            
            if total_edges > 0:  # Only plot if there are edges
                plt.hist(scores, bins=num_bins, alpha=0.5, color=colors[edge_type],
                         label=f"{labels[edge_type]} pos ({total_edges} edges, {edges_above} > 0.5, {percentage_above:.1f}%)")
    
    # Add vertical line at the threshold
    plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Threshold = {threshold}')
    
    # Set up plot appearance
    plt.xlabel('Memorization Score (f - g confidence)')
    plt.ylabel('Frequency Count')
    plt.title('Memorization Score Distribution - Positive Edges')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    
    # Plot 2: Memorization score distribution for negative edges
    plt.subplot(212)
    
    # Create histogram with frequency counts for negative edges
    for edge_type in edge_types_to_plot:
        if edge_type in edge_scores:
            scores = edge_scores[edge_type]['negative_edges']['mem_scores']
            mean_score = edge_scores[edge_type]['negative_edges']['avg_score']
            edges_above = edge_scores[edge_type]['negative_edges']['above_threshold']
            total_edges = edge_scores[edge_type]['negative_edges']['count']
            percentage_above = edge_scores[edge_type]['negative_edges']['percentage_above']
            
            if total_edges > 0:  # Only plot if there are edges
                plt.hist(scores, bins=num_bins, alpha=0.5, color=colors[edge_type],
                         label=f"{labels[edge_type]} neg ({total_edges} edges, {edges_above} > 0.5, {percentage_above:.1f}%)")
    
    # Add vertical line at the threshold
    plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Threshold = {threshold}')
    
    # Set up plot appearance
    plt.xlabel('Memorization Score (f - g confidence)')
    plt.ylabel('Frequency Count')
    plt.title('Memorization Score Distribution - Negative Edges')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    
    if title_suffix:
        plt.suptitle(title_suffix, fontsize=14)
        plt.subplots_adjust(top=0.85)
    
    plt.tight_layout()
    plt.savefig(f"{base_path}_mem_score_distribution{ext}", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot confidence comparison scatter plots for candidate edges
    if 'candidate' in edge_scores:
        plt.figure(figsize=(15, 6))
        
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
    
    return f"{base_path}_mem_score_distribution{ext}"

def main():
    """Main function to run link prediction memorization analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Link Prediction Memorization Score')
    
    # Dataset and model parameters
    parser.add_argument('--dataset', type=str, default='Cora',
                        choices=['Cora', 'Citeseer', 'Pubmed'] + get_heterophilic_datasets(),
                        help='Dataset name')
    parser.add_argument('--model_type', type=str, default='gcn',
                        choices=['gcn', 'gat', 'graphconv'],
                        help='GNN model type')
    
    # Model hyperparameters
    parser.add_argument('--hidden_dim', type=int, default=32,
                        help='Hidden dimension size')
    parser.add_argument('--out_dim', type=int, default=32,
                        help='Output dimension size')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of GNN layers')
    parser.add_argument('--heads', type=int, default=4,
                        help='Number of attention heads for GAT')
    
    # Training parameters
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Splitting parameters
    parser.add_argument('--split_ratios', nargs='+', type=float, default=[0.50, 0.25, 0.25],
                        help='Split ratios for [shared, candidate, independent] edges')
    
    args = parser.parse_args()
    
    # Ensure split ratios sum to 1
    split_sum = sum(args.split_ratios)
    if abs(split_sum - 1.0) > 1e-6:
        args.split_ratios = [ratio / split_sum for ratio in args.split_ratios]
        print(f"Normalized split ratios to sum to 1.0: {args.split_ratios}")
    
    # Set random seed
    set_seed(args.seed)
    
    # Set up logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger, log_dir = setup_logging(args.model_type, args.dataset, timestamp)
    
    logger.info(f"Link Prediction Memorization Analysis")
    logger.info(f"Dataset: {args.dataset}, Model: {args.model_type}")
    logger.info(f"Using device: {device}")
    logger.info(f"Split ratios: {args.split_ratios}")
    
    # Create data transformation
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                          add_negative_train_samples=False),
    ])
    
    # Load dataset
    logger.info(f"Loading {args.dataset} dataset...")
    dataset = load_dataset(args.dataset, transform)
    
    # Apply transformation and get data
    data_full = dataset
    train_data, val_data, test_data = data_full[0]
    
    # Log dataset statistics
    logger.info(f"Dataset loaded:")
    logger.info(f"  Number of nodes: {train_data.num_nodes}")
    logger.info(f"  Number of edges: {train_data.edge_index.size(1)}")
    logger.info(f"  Number of node features: {train_data.num_features}")
    
    # Get edge splits
    logger.info("Creating edge splits...")
    edges_dict = get_edge_splits(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        split_ratios=tuple(args.split_ratios)
    )
    
    # Log edge split statistics
    for edge_type in edges_dict:
        if edge_type in ['shared', 'candidate', 'independent']:
            pos_edges = edges_dict[edge_type]['pos']
            neg_edges = edges_dict[edge_type]['neg']
            logger.info(f"  {edge_type.capitalize()} edges: {len(pos_edges)} positive, {len(neg_edges)} negative")
    
    # Train models
    logger.info("Training models...")
    model_f, model_g = train_models(data_full, edges_dict, args, logger)
    
    # Calculate memorization scores
    logger.info("Calculating edge memorization scores...")
    edge_scores = calculate_edge_memorization_score(
        model_f=model_f,
        model_g=model_g,
        data=train_data,
        edges_dict=edges_dict,
        device=device,
        logger=logger
    )
    
    # Create visualization
    logger.info("Creating visualizations...")
    plot_filename = f'link_memorization_{args.model_type}_{args.dataset}_{timestamp}.png'
    plot_path = os.path.join(log_dir, plot_filename)
    
    plot_edge_memorization_analysis(
        edge_scores=edge_scores,
        save_path=plot_path,
        title_suffix=f"Link Prediction Memorization - {args.dataset}, {args.model_type.upper()}",
        edge_types_to_plot=['shared', 'candidate', 'independent', 'extra']
    )
    
    logger.info(f"Memorization score plots saved to: {log_dir}")
    
    # Save models
    torch.save(model_f.state_dict(), os.path.join(log_dir, 'model_f.pt'))
    torch.save(model_g.state_dict(), os.path.join(log_dir, 'model_g.pt'))
    logger.info(f"Models saved to: {log_dir}")
    
    return edge_scores

if __name__ == "__main__":
    main()