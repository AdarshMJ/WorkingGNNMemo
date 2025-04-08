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
from model import LinkGNN, LinkGNN_MLP  # Import the new model class
from torch_geometric.datasets import Planetoid,Actor, WikipediaNetwork,WebKB
from train import *

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
    
    # Select appropriate edges based on model type
    if edge_type == 'f':
        # Model f: shared + candidate
        other_pos_edges = torch.tensor(edges_dict['candidate']['pos'], dtype=torch.long).t().to(device)
        edge_categories = ['shared', 'candidate']
    else:
        # Model g: shared + independent
        other_pos_edges = torch.tensor(edges_dict['independent']['pos'], dtype=torch.long).t().to(device)
        edge_categories = ['shared', 'independent']
    
    # Combine positive edges
    pos_edge_index = torch.cat([shared_pos_edges, other_pos_edges], dim=1)
    
    # Get initial negative edges (will be used only for the first epoch)
    shared_neg_edges = torch.tensor(edges_dict['shared']['neg'], dtype=torch.long).t().to(device)
    
    if edge_type == 'f':
        other_neg_edges = torch.tensor(edges_dict['candidate']['neg'], dtype=torch.long).t().to(device)
    else:
        other_neg_edges = torch.tensor(edges_dict['independent']['neg'], dtype=torch.long).t().to(device)
    
    # Combine initial negative edges
    neg_edge_index = torch.cat([shared_neg_edges, other_neg_edges], dim=1)
    
    # Set up edge indices for training
    train_edges = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    edge_labels = torch.cat([
        torch.ones(pos_edge_index.size(1), dtype=torch.float, device=device),
        torch.zeros(neg_edge_index.size(1), dtype=torch.float, device=device)
    ])
    
    # Store edge counts for per-epoch negative sampling
    edge_counts = {
        'shared': shared_neg_edges.size(1),
        edge_categories[1]: other_neg_edges.size(1)
    }
    
    # Return positive edges separately to allow per-epoch negative sampling
    return train_edges, edge_labels, pos_edge_index, edge_counts, edges_dict, edge_categories



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
    parser.add_argument('--decoder', type=str, default='dot',
                        choices=['dot', 'mlp'],
                        help='Link decoder type: dot product or MLP')
    
    # Model hyperparameters
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension size')
    parser.add_argument('--out_dim', type=int, default=32,
                        help='Output dimension size')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of GNN layers')
    parser.add_argument('--heads', type=int, default=4,
                        help='Number of attention heads for GAT')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--seed', type=int, nargs='+', default=[42, 123, 456],
                        help='Random seed(s) for reproducibility (can provide multiple)')
    parser.add_argument('--per_epoch_sampling', action='store_true',
                        help='Enable per-epoch negative sampling (default: False)')
    
    # Splitting parameters
    parser.add_argument('--split_ratios', nargs='+', type=float, default=[0.50, 0.25, 0.25],
                        help='Split ratios for [shared, candidate, independent] edges')
    
    args = parser.parse_args()
    
    # Ensure split ratios sum to 1
    split_sum = sum(args.split_ratios)
    if abs(split_sum - 1.0) > 1e-6:
        args.split_ratios = [ratio / split_sum for ratio in args.split_ratios]
        print(f"Normalized split ratios to sum to 1.0: {args.split_ratios}")
    
    # Set up logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = f"{args.model_type}_{args.decoder}" if args.decoder == 'mlp' else args.model_type
    logger, log_dir = setup_logging(model_name, args.dataset, timestamp)
    
    logger.info(f"Link Prediction Memorization Analysis")
    logger.info(f"Dataset: {args.dataset}, Model: {args.model_type}, Decoder: {args.decoder}")
    logger.info(f"Using device: {device}")
    logger.info(f"Split ratios: {args.split_ratios}")
    logger.info(f"Random seeds: {args.seed}")
    
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
    
    # Train models for each seed
    logger.info("Training models with multiple seeds...")
    best_val_acc_f = 0
    best_val_acc_g = 0
    best_models = {'f': None, 'g': None}
    best_seed = None

    # Prepare validation edge index and labels
    val_edge_index = torch.cat([
        torch.tensor(edges_dict['val']['pos'], dtype=torch.long).t().to(device),
        torch.tensor(edges_dict['val']['neg'], dtype=torch.long).t().to(device)
    ], dim=1)
    
    val_edge_label = torch.cat([
        torch.ones(len(edges_dict['val']['pos']), device=device),
        torch.zeros(len(edges_dict['val']['neg']), device=device)
    ])
    
    # Train models for each seed and select the best one
    for seed in args.seed:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training with seed {seed}")
        logger.info(f"{'='*50}")
        
        # Create a copy of args with the current seed
        current_args = argparse.Namespace(**vars(args))
        current_args.seed = seed
        
        # Train models using the fixed version that respects the provided seed
        model_f, model_g = train_models_fixed(data_full, edges_dict, current_args, logger)
        
        # Evaluate on validation set
        val_auc_f = evaluate_model(model_f, train_data, val_edge_index, val_edge_label)
        val_auc_g = evaluate_model(model_g, train_data, val_edge_index, val_edge_label)
        
        logger.info(f"Seed {seed} - Model f validation AUC: {val_auc_f:.4f}")
        logger.info(f"Seed {seed} - Model g validation AUC: {val_auc_g:.4f}")
        
        # Check if this is the best model so far
        if val_auc_f > best_val_acc_f and val_auc_g > best_val_acc_g:
            best_val_acc_f = val_auc_f
            best_val_acc_g = val_auc_g
            # Save the best models' state dictionaries
            best_models['f'] = {k: v.cpu().clone() for k, v in model_f.state_dict().items()}
            best_models['g'] = {k: v.cpu().clone() for k, v in model_g.state_dict().items()}
            best_seed = seed
            logger.info(f"New best models found with seed {seed}")
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Best models found with seed {best_seed}")
    logger.info(f"Model f validation AUC: {best_val_acc_f:.4f}")
    logger.info(f"Model g validation AUC: {best_val_acc_g:.4f}")
    logger.info(f"{'='*50}")
    
    # Recreate the models with the architecture from the best performing run
    current_args = argparse.Namespace(**vars(args))
    current_args.seed = best_seed
    
    # Initialize new models with the same architecture
    if args.decoder == 'mlp':
        model_f = LinkGNN_MLP(
            in_channels=train_data.num_features, 
            hidden_channels=args.hidden_dim,
            out_channels=args.out_dim,
            num_layers=args.num_layers,
            model_type=args.model_type,
            dropout=args.dropout,
            heads=args.heads if args.model_type == 'gat' else None
        ).to(device)
        
        model_g = LinkGNN_MLP(
            in_channels=train_data.num_features, 
            hidden_channels=args.hidden_dim,
            out_channels=args.out_dim,
            num_layers=args.num_layers,
            model_type=args.model_type,
            dropout=args.dropout,
            heads=args.heads if args.model_type == 'gat' else None
        ).to(device)
    else:
        model_f = LinkGNN(
            in_channels=train_data.num_features, 
            hidden_channels=args.hidden_dim,
            out_channels=args.out_dim,
            num_layers=args.num_layers,
            model_type=args.model_type,
            heads=args.heads if args.model_type == 'gat' else None
        ).to(device)
        
        model_g = LinkGNN(
            in_channels=train_data.num_features, 
            hidden_channels=args.hidden_dim,
            out_channels=args.out_dim,
            num_layers=args.num_layers,
            model_type=args.model_type,
            heads=args.heads if args.model_type == 'gat' else None
        ).to(device)
    
    # Load the best weights
    model_f.load_state_dict({k: v.to(device) for k, v in best_models['f'].items()})
    model_g.load_state_dict({k: v.to(device) for k, v in best_models['g'].items()})
    
    # Calculate memorization scores using the best models
    logger.info("Calculating edge memorization scores using best models...")
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
    plot_filename = f'link_memorization_{model_name}_{args.dataset}_{timestamp}.png'
    plot_path = os.path.join(log_dir, plot_filename)
    
    plot_edge_memorization_analysis(
        edge_scores=edge_scores,
        save_path=plot_path,
        title_suffix=f"Link Prediction Memorization - {args.dataset}, {args.model_type.upper()}, {args.decoder.upper()} decoder (seed {best_seed})",
        edge_types_to_plot=['shared', 'candidate', 'independent', 'extra']
    )
    
    logger.info(f"Memorization score plots saved to: {log_dir}")
    
    # Save best models
    torch.save(model_f.state_dict(), os.path.join(log_dir, f'model_f_seed{best_seed}.pt'))
    torch.save(model_g.state_dict(), os.path.join(log_dir, f'model_g_seed{best_seed}.pt'))
    logger.info(f"Best models (seed {best_seed}) saved to: {log_dir}")
    
    return edge_scores

if __name__ == "__main__":
    main()