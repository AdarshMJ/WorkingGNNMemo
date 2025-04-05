import torch
from link_memorization import *
from sklearn.metrics import roc_auc_score
import numpy as np
from model import *


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