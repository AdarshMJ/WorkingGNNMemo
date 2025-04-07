import argparse
import torch
import os
import logging
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import networkx as nx
from scipy import sparse
import pickle
from torch_geometric.utils import negative_sampling
import torch_geometric.transforms as T

# Import local modules
from dataset import CustomDataset
from model import LinkGNN, LinkGNN_MLP
from link_memorization import set_seed, setup_logging
from memorizationscore import get_edge_splits, calculate_edge_memorization_score
from nodeli import li_node

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_and_process_dataset(args, dataset_name, logger):
    """Load synthetic Cora dataset and prepare it for link prediction"""
    # Construct full path to dataset
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "syn-cora")
    dataset = CustomDataset(root=root_dir, name=dataset_name, setting="gcn")
    
    # Convert to PyG format
    edge_index = torch.from_numpy(np.vstack(dataset.adj.nonzero())).long()
    
    # Convert sparse features to dense numpy array
    if sparse.issparse(dataset.features):
        x = torch.from_numpy(dataset.features.todense()).float()
    else:
        x = torch.from_numpy(dataset.features).float()
    
    y = torch.from_numpy(dataset.labels).long()
    
    # Convert to networkx for informativeness calculation
    G = nx.Graph()
    G.add_nodes_from(range(len(y)))
    G.add_edges_from(edge_index.t().numpy())
    
    # Calculate label informativeness using existing function
    informativeness = li_node(G, dataset.labels)
    
    # Calculate homophily (edge homophily)
    edges = edge_index.t().numpy()
    same_label = dataset.labels[edges[:, 0]] == dataset.labels[edges[:, 1]]
    homophily = same_label.mean()
    
    # Create a data object with the necessary attributes for link prediction
    data = type('Data', (), {
        'x': x,
        'y': y,
        'edge_index': edge_index,
        'num_nodes': len(y),
        'num_features': x.shape[1],
        'informativeness': informativeness,
        'homophily': homophily
    })()
    
    # Log dataset statistics
    logger.info(f"\nDataset Statistics:")
    logger.info(f"Number of nodes: {data.num_nodes}")
    logger.info(f"Number of edges: {len(edges)}")
    logger.info(f"Number of features: {data.num_features}")
    logger.info(f"Number of classes: {len(torch.unique(y))}")
    logger.info(f"Homophily: {homophily:.4f}")
    logger.info(f"Label Informativeness: {informativeness:.4f}")
    
    return data

def prepare_link_prediction_data(data):
    """
    Prepare data for link prediction by creating train/val/test splits
    """
    # Get all edges and create train/val/test splits
    edge_index = data.edge_index
    num_edges = edge_index.size(1)
    
    # Create random permutation
    perm = torch.randperm(num_edges)
    
    # 80% train, 10% val, 10% test
    train_size = int(0.8 * num_edges)
    val_size = int(0.1 * num_edges)
    
    # Create edge splits
    train_edges = edge_index[:, perm[:train_size]]
    val_edges = edge_index[:, perm[train_size:train_size+val_size]]
    test_edges = edge_index[:, perm[train_size+val_size:]]
    
    # Generate negative edges for training, validation and test sets
    train_neg_edges = negative_sampling(
        edge_index=edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=train_edges.size(1),
        method='sparse'
    )
    
    val_neg_edges = negative_sampling(
        edge_index=edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=val_edges.size(1),
        method='sparse'
    )
    
    test_neg_edges = negative_sampling(
        edge_index=edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=test_edges.size(1),
        method='sparse'
    )
    
    # Create data objects for train/val/test with proper edge label formats for link prediction
    train_data = type('Data', (), {
        'x': data.x,
        'y': data.y,
        'edge_index': train_edges,
        'num_nodes': data.num_nodes,
        'num_features': data.num_features,
        # Edge label index contains both positive and negative edges for training
        'edge_label_index': torch.cat([train_edges, train_neg_edges], dim=1),
        # Edge label has 1 for positive edges and 0 for negative edges
        'edge_label': torch.cat([torch.ones(train_edges.size(1)), torch.zeros(train_neg_edges.size(1))])
    })()
    
    val_data = type('Data', (), {
        'x': data.x,
        'y': data.y,
        'edge_index': train_edges,  # Use training graph structure
        'num_nodes': data.num_nodes,
        'num_features': data.num_features,
        'edge_label_index': torch.cat([val_edges, val_neg_edges], dim=1),
        'edge_label': torch.cat([torch.ones(val_edges.size(1)), torch.zeros(val_neg_edges.size(1))])
    })()
    
    test_data = type('Data', (), {
        'x': data.x,
        'y': data.y,
        'edge_index': train_edges,  # Use training graph structure
        'num_nodes': data.num_nodes,
        'num_features': data.num_features,
        'edge_label_index': torch.cat([test_edges, test_neg_edges], dim=1),
        'edge_label': torch.cat([torch.ones(test_edges.size(1)), torch.zeros(test_neg_edges.size(1))])
    })()
    
    return train_data, val_data, test_data

def get_model(args, num_features):
    """Get appropriate model based on args"""
    if args.decoder == 'mlp':
        model_class = LinkGNN_MLP
    else:  # Default to dot product decoder
        model_class = LinkGNN
    
    model = model_class(
        in_channels=num_features,
        hidden_channels=args.hidden_dim,
        out_channels=args.out_dim,
        num_layers=args.num_layers,
        model_type=args.model_type,
        dropout=args.dropout,
        heads=args.heads if args.model_type == 'gat' else None
    ).to(device)
    
    return model

def train_step(model, optimizer, train_edges, edge_labels, data):
    """Train model for one step"""
    model.train()
    optimizer.zero_grad()
    
    # Encode
    z = model.encode(data.x, data.edge_index)
    
    # Decode
    out = model.decode(z, train_edges)
    
    # Loss
    loss = torch.nn.BCEWithLogitsLoss()(out, edge_labels)
    
    # Backward
    loss.backward()
    optimizer.step()
    
    return loss.item()

@torch.no_grad()
def test(model, data, pos_edge_index, neg_edge_index):
    """Test model on validation/test edges"""
    model.eval()
    
    # Encode nodes
    z = model.encode(data.x, data.edge_index)
    
    # Get predictions for positive and negative edges
    pos_pred = model.decode(z, pos_edge_index).sigmoid()
    neg_pred = model.decode(z, neg_edge_index).sigmoid()
    
    # Combine predictions and true labels
    pred = torch.cat([pos_pred, neg_pred], dim=0)
    true = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))], dim=0)
    
    # Calculate accuracy
    correct = ((pred > 0.5) == true).sum().item()
    acc = correct / true.size(0)
    
    return acc

def train_models(data_full, edges_dict, args, logger=None):
    """Train models f and g on different edge subsets"""
    # Create training data for model f (shared+candidate)
    train_data, val_data, test_data = data_full
    
    # Create train edges for models f and g
    train_edges_f, edge_labels_f = create_training_data(train_data, edges_dict, 'f')
    train_edges_g, edge_labels_g = create_training_data(train_data, edges_dict, 'g')
    
    # Get validation edges
    val_pos_edges = torch.tensor(edges_dict['val']['pos'], dtype=torch.long).t().to(device)
    val_neg_edges = torch.tensor(edges_dict['val']['neg'], dtype=torch.long).t().to(device)
    
    # Initialize models
    model_f = get_model(args, train_data.num_features)
    model_g = get_model(args, train_data.num_features)
    
    # Optimizers
    optimizer_f = torch.optim.Adam(model_f.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_g = torch.optim.Adam(model_g.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Training loop
    best_val_acc_f = 0
    best_val_acc_g = 0
    best_model_f = None
    best_model_g = None
    
    for epoch in range(1, args.epochs + 1):
        # Train model f
        loss_f = train_step(model_f, optimizer_f, train_edges_f, edge_labels_f, train_data)
        
        # Train model g
        loss_g = train_step(model_g, optimizer_g, train_edges_g, edge_labels_g, train_data)
        
        # Validate
        val_acc_f = test(model_f, val_data, val_pos_edges, val_neg_edges)
        val_acc_g = test(model_g, val_data, val_pos_edges, val_neg_edges)
        
        # Save best models
        if val_acc_f > best_val_acc_f:
            best_val_acc_f = val_acc_f
            best_model_f = {k: v.cpu().clone() for k, v in model_f.state_dict().items()}
        
        if val_acc_g > best_val_acc_g:
            best_val_acc_g = val_acc_g
            best_model_g = {k: v.cpu().clone() for k, v in model_g.state_dict().items()}
        
        if epoch % 10 == 0 and logger:
            logger.info(f'Epoch {epoch:03d}: Loss F: {loss_f:.4f}, Loss G: {loss_g:.4f}, '
                      f'Val Acc F: {val_acc_f:.4f}, Val Acc G: {val_acc_g:.4f}')
    
    # Load best models
    model_f.load_state_dict({k: v.to(device) for k, v in best_model_f.items()})
    model_g.load_state_dict({k: v.to(device) for k, v in best_model_g.items()})
    
    # Final validation
    final_val_acc_f = test(model_f, val_data, val_pos_edges, val_neg_edges)
    final_val_acc_g = test(model_g, val_data, val_pos_edges, val_neg_edges)
    
    if logger:
        logger.info(f'Final validation accuracy - Model F: {final_val_acc_f:.4f}, Model G: {final_val_acc_g:.4f}')
    
    return model_f, model_g, final_val_acc_f, final_val_acc_g

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

def create_homophily_informativeness_plot(results_df, save_path, title, args):
    """Create scatter plot of homophily vs informativeness colored by memorization rate"""
    plt.figure(figsize=(10, 8))
    
    scatter = plt.scatter(
        results_df['homophily'],
        results_df['informativeness'],
        c=results_df['percent_memorized'],
        cmap='viridis',
        s=100,
        alpha=0.7
    )
    
    plt.xlabel('Homophily', fontsize=12)
    plt.ylabel('Label Informativeness', fontsize=12)
    plt.title(f'{title}\nModel: {args.model_type.upper()}, Decoder: {args.decoder.upper()}', fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Memorization Rate (%)', fontsize=10)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Add text labels with exact memorization percentages
    for i, row in results_df.iterrows():
        plt.annotate(
            f"{row['percent_memorized']:.1f}%",
            (row['homophily'], row['informativeness']),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center'
        )
    
    # Save plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    
    # Dataset parameters
    parser.add_argument('--homophily_levels', type=float, nargs='+', default=[0.0, 0.3, 0.5, 0.7, 1.0],
                       help='Homophily levels to analyze')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='gcn',
                       choices=['gcn', 'gat', 'graphconv'],
                       help='GNN model type')
    parser.add_argument('--decoder', type=str, default='dot',
                      choices=['dot', 'mlp'],
                      help='Link decoder type: dot product or MLP')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden dimension size')
    parser.add_argument('--out_dim', type=int, default=64,
                       help='Output embedding dimension')
    parser.add_argument('--num_layers', type=int, default=2,
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
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--split_ratios', nargs='+', type=float, default=[0.50, 0.25, 0.25],
                       help='Split ratios for [shared, candidate, independent] edges')
    parser.add_argument('--output_dir', type=str, default='results/syncora_link_prediction',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Setup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f'syncora_link_{args.model_type}_{args.decoder}_{timestamp}'
    log_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup logging
    logger = logging.getLogger('syncora_link_prediction')
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear any existing handlers
    
    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, 'analysis.log'))
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    
    # Set random seed
    set_seed(args.seed)
    
    logger.info(f"Syncora Link Prediction Memorization Analysis")
    logger.info(f"Using device: {device}")
    logger.info(f"Model: {args.model_type}, Decoder: {args.decoder}")
    logger.info(f"Homophily levels: {args.homophily_levels}")
    logger.info(f"Split ratios: {args.split_ratios}")
    
    # Ensure split ratios sum to 1
    split_sum = sum(args.split_ratios)
    if abs(split_sum - 1.0) > 1e-6:
        args.split_ratios = [ratio / split_sum for ratio in args.split_ratios]
        logger.info(f"Normalized split ratios to sum to 1.0: {args.split_ratios}")
    
    # Initialize results containers
    results_pos = []  # For positive edges
    results_neg = []  # For negative edges
    
    # Select dataset files based on homophily levels
    dataset_files = [f'h{h:.2f}-r1' for h in args.homophily_levels]
    
    for dataset_name in tqdm(dataset_files, desc="Processing datasets"):
        logger.info(f"\nProcessing dataset: {dataset_name}")
        
        # Load and process dataset
        try:
            data = load_and_process_dataset(args, dataset_name, logger)
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}")
            continue
        
        # Move data to device
        data.x = data.x.to(device)
        data.edge_index = data.edge_index.to(device)
        
        # Prepare data for link prediction
        try:
            train_data, val_data, test_data = prepare_link_prediction_data(data)
            data_full = [train_data, val_data, test_data]
        except Exception as e:
            logger.error(f"Error preparing link prediction data for {dataset_name}: {e}")
            continue
        
        # Get edge splits
        try:
            edges_dict = get_edge_splits(
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                split_ratios=tuple(args.split_ratios)
            )
        except Exception as e:
            logger.error(f"Error getting edge splits for {dataset_name}: {e}")
            continue
        
        # Log edge split statistics
        for edge_type in ['shared', 'candidate', 'independent']:
            pos_edges = edges_dict[edge_type]['pos']
            neg_edges = edges_dict[edge_type]['neg']
            logger.info(f"  {edge_type.capitalize()} edges: {len(pos_edges)} positive, {len(neg_edges)} negative")
        
        # Train models
        try:
            model_f, model_g, f_val_acc, g_val_acc = train_models(
                data_full=data_full,
                edges_dict=edges_dict,
                args=args,
                logger=logger
            )
        except Exception as e:
            logger.error(f"Error training models for {dataset_name}: {e}")
            continue
        
        # Calculate memorization scores
        try:
            edge_scores = calculate_edge_memorization_score(
                model_f=model_f,
                model_g=model_g,
                data=train_data,
                edges_dict=edges_dict,
                device=device,
                logger=logger
            )
        except Exception as e:
            logger.error(f"Error calculating memorization scores for {dataset_name}: {e}")
            continue
        
        # Store results for positive edges
        results_pos.append({
            'dataset': dataset_name,
            'homophily': float(data.homophily),
            'informativeness': float(data.informativeness),
            'percent_memorized': edge_scores['candidate']['positive_edges']['percentage_above'],
            'avg_memorization': edge_scores['candidate']['positive_edges']['avg_score'],
            'num_memorized': edge_scores['candidate']['positive_edges']['above_threshold'],
            'total_edges': edge_scores['candidate']['positive_edges']['count'],
            'f_val_acc': float(f_val_acc),
            'g_val_acc': float(g_val_acc)
        })
        
        # Store results for negative edges
        results_neg.append({
            'dataset': dataset_name,
            'homophily': float(data.homophily),
            'informativeness': float(data.informativeness),
            'percent_memorized': edge_scores['candidate']['negative_edges']['percentage_above'],
            'avg_memorization': edge_scores['candidate']['negative_edges']['avg_score'],
            'num_memorized': edge_scores['candidate']['negative_edges']['above_threshold'],
            'total_edges': edge_scores['candidate']['negative_edges']['count'],
            'f_val_acc': float(f_val_acc),
            'g_val_acc': float(g_val_acc)
        })
        
        # Save model checkpoints
        torch.save(model_f.state_dict(), os.path.join(log_dir, f'model_f_{dataset_name}.pt'))
        torch.save(model_g.state_dict(), os.path.join(log_dir, f'model_g_{dataset_name}.pt'))
        
    # Convert results to DataFrames
    results_df_pos = pd.DataFrame(results_pos)
    results_df_neg = pd.DataFrame(results_neg)
    
    # Save results as CSV
    results_df_pos.to_csv(os.path.join(log_dir, 'results_positive_edges.csv'), index=False)
    results_df_neg.to_csv(os.path.join(log_dir, 'results_negative_edges.csv'), index=False)
    
    # Create visualization for positive edges
    if len(results_pos) > 0:
        plot_path_pos = os.path.join(log_dir, f'memorization_positive_edges_{timestamp}.png')
        create_homophily_informativeness_plot(
            results_df=results_df_pos,
            save_path=plot_path_pos,
            title='Link Memorization Analysis - Positive Edges',
            args=args
        )
        logger.info(f"Positive edges memorization visualization saved as: {plot_path_pos}")
    
    # Create visualization for negative edges
    if len(results_neg) > 0:
        plot_path_neg = os.path.join(log_dir, f'memorization_negative_edges_{timestamp}.png')
        create_homophily_informativeness_plot(
            results_df=results_df_neg,
            save_path=plot_path_neg,
            title='Link Memorization Analysis - Negative Edges',
            args=args
        )
        logger.info(f"Negative edges memorization visualization saved as: {plot_path_neg}")
    
    logger.info("\nAnalysis complete!")
    logger.info(f"Results saved to: {log_dir}")

if __name__ == '__main__':
    main()