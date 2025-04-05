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
from model import *
from torch_geometric.datasets import Planetoid,Actor, WikipediaNetwork,WebKB
from train import *
from torch_geometric.utils import negative_sampling
from memorizationscore  import *
import logging
from datetime import datetime
import os.path as osp
from dataloader import load_npz_dataset, process_heterophilic_dataset_for_link_prediction, get_heterophilic_datasets
from scipy import stats

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
        
        # Apply the split
        train_data, val_data, test_data = link_split(data)
        
        # Return the data in the same format as Planetoid datasets
        return [(train_data, val_data, test_data)]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")



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



def main():
    """Main function to run link prediction memorization analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Link Prediction Memorization Score')
    
    # Dataset and model parameters
    parser.add_argument('--dataset', type=str, default='Cora',
                        choices=['Cora', 'Citeseer', 'Pubmed', 'Cornell', 'Texas','Wisconsin','Chameleon','Squirrel',"Actor"] + get_heterophilic_datasets(),
                        help='Dataset name')
    parser.add_argument('--model_type', type=str, default='gcn',
                        choices=['gcn', 'gat', 'graphconv'],
                        help='GNN model type')
    
    # Model hyperparameters
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension size')
    parser.add_argument('--out_dim', type=int, default=32,
                        help='Output dimension size')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of GNN layers')
    parser.add_argument('--heads', type=int, default=4,
                        help='Number of attention heads for GAT')
    
    # Training parameters
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0,
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