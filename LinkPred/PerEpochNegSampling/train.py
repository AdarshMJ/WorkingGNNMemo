import torch
from link_memorization import *
from sklearn.metrics import roc_auc_score
import numpy as np
from model import LinkGNN, LinkGNN_MLP  # Import both model classes
from torch_geometric.utils import negative_sampling

def train_link_prediction_model(model, optimizer, data, train_edges, edge_labels, epochs=100, pos_edge_index=None, edge_counts=None, edges_dict=None, edge_categories=None):
    """
    Train a link prediction model with per-epoch negative sampling.
    Args:
        model: The model to train
        optimizer: Optimizer to use
        data: Graph data
        train_edges: Initial edge indices for training (will be updated each epoch if pos_edge_index is provided)
        edge_labels: Initial edge labels for training (will be updated each epoch if pos_edge_index is provided)
        epochs: Number of training epochs
        pos_edge_index: Positive edges to use for training (for per-epoch negative sampling)
        edge_counts: Number of negative edges to sample per category
        edges_dict: Dictionary with edge splits
        edge_categories: Categories of edges to use ['shared', 'candidate'] or ['shared', 'independent']
    """
    criterion = torch.nn.BCEWithLogitsLoss()
    best_loss = float('inf')
    best_model_state = None
    
    # Determine if we should use per-epoch negative sampling
    use_per_epoch_sampling = (pos_edge_index is not None and 
                             edge_counts is not None and 
                             edges_dict is not None and 
                             edge_categories is not None)
    
    # Function to create edge overlap check sets
    def get_edge_sets(edges_dict):
        all_pos_edges_set = set()
        for category in ['shared', 'candidate', 'independent', 'val', 'test']:
            if category in edges_dict:
                # Add positive edges as tuples for set operations
                all_pos_edges_set.update({tuple(edge) for edge in edges_dict[category]['pos']})
        return all_pos_edges_set
    
    # Get all positive edges as a set for checking overlap
    if use_per_epoch_sampling:
        all_pos_edges_set = get_edge_sets(edges_dict)
        print(f"Total positive edges (all categories): {len(all_pos_edges_set)}")
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Get node embeddings
        z = model.encode(data.x, data.edge_index)
        
        # Use per-epoch negative sampling if all required data is provided
        if use_per_epoch_sampling:
            # Generate new negative edges for each category
            all_neg_edges = []
            neg_edges_by_category = {}
            
            # Sample negative edges for each category
            for category in edge_categories:
                num_neg_samples = edge_counts[category]
                
                # Get all edges to exclude from sampling
                existing_edges = torch.cat([data.edge_index, pos_edge_index], dim=1).t().cpu().numpy()
                existing_edges_set = set(map(tuple, existing_edges))
                
                # Store all sampled negative edges as tuples for checking overlap
                neg_edges_set = set()
                
                attempts = 0
                max_attempts = num_neg_samples * 10  # Allow multiple attempts to find valid negatives
                
                while len(neg_edges_set) < num_neg_samples and attempts < max_attempts:
                    # Sample candidate negative edges
                    neg_edge_index = negative_sampling(
                        edge_index=data.edge_index,
                        num_nodes=data.num_nodes,
                        num_neg_samples=num_neg_samples - len(neg_edges_set),
                        method='sparse'
                    ).t().cpu().numpy()
                    
                    # Filter edges that already exist in any category
                    for edge in neg_edge_index:
                        edge_tuple = tuple(edge)
                        if edge_tuple not in existing_edges_set and edge_tuple not in neg_edges_set:
                            neg_edges_set.add(edge_tuple)
                    
                    attempts += len(neg_edge_index)
                
                # Convert set of tuples back to a list of edges
                neg_edges = list(neg_edges_set)
                
                if len(neg_edges) < num_neg_samples:
                    print(f"Warning: Could only sample {len(neg_edges)}/{num_neg_samples} negative edges for {category} after {attempts} attempts")
                
                # Convert to tensor and store
                neg_edge_tensor = torch.tensor(np.array(neg_edges), dtype=torch.long).t().to(device)
                all_neg_edges.append(neg_edge_tensor)
                neg_edges_by_category[category] = neg_edge_tensor
                
                # Verify no overlap with positive edges
                overlap = neg_edges_set.intersection(all_pos_edges_set)
                if overlap:
                    print(f"WARNING: Found {len(overlap)} negative edges that overlap with positive edges in {category}")
            
            # Combine all negative edges
            if len(all_neg_edges) > 0:
                neg_edge_index = torch.cat(all_neg_edges, dim=1)
            else:
                # Fallback if no valid negative edges were found
                print(f"WARNING: No valid negative edges found, using random sampling")
                neg_edge_index = negative_sampling(
                    edge_index=data.edge_index,
                    num_nodes=data.num_nodes,
                    num_neg_samples=sum(edge_counts.values()),
                    method='sparse'
                ).to(device)
            
            # Optional debugging: Print negative edge counts per category
            if epoch == 0 or (epoch + 1) % 20 == 0:
                for cat in edge_categories:
                    if cat in neg_edges_by_category:
                        tensor = neg_edges_by_category[cat]
                        size = tensor.size(1) if tensor.dim() > 1 else 1
                        print(f"Epoch {epoch+1}: Generated {size} {cat} negative edges")
                print(f"Epoch {epoch+1}: Total negative edges: {neg_edge_index.size(1)}")
            
            # Create combined edge index and labels for this epoch
            current_train_edges = torch.cat([pos_edge_index, neg_edge_index], dim=1)
            current_edge_labels = torch.cat([
                torch.ones(pos_edge_index.size(1), dtype=torch.float, device=device),
                torch.zeros(neg_edge_index.size(1), dtype=torch.float, device=device)
            ])
            
            # Make predictions on the edges for this epoch
            out = model.decode(z, current_train_edges).view(-1)
            loss = criterion(out, current_edge_labels)
        else:
            # Use fixed edges for the entire training
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

def train_model_single_seed(data_full, edges_dict, args, logger, model_type='f'):
    """
    Train a single model with the provided seed.
    
    Args:
        data_full: Original data object with all graph information
        edges_dict: Dictionary with edge splits
        args: Arguments dictionary with model parameters
        logger: Logger object
        model_type: Type of model to train ('f' or 'g')
        
    Returns:
        Trained model
    """
    # Extract training data
    train_data, val_data, test_data = data_full[0]
    
    # Create training data based on model type
    if model_type == 'f':
        # Model f: shared + candidate
        train_edges, edge_labels, pos_edge_index, edge_counts, edges_dict_copy, edge_categories = create_training_data(train_data, edges_dict, 'f')
        model_name = "f (shared+candidate)"
    elif model_type == 'g': 
        # Model g: shared + independent
        train_edges, edge_labels, pos_edge_index, edge_counts, edges_dict_copy, edge_categories = create_training_data(train_data, edges_dict, 'g')
        model_name = "g (shared+independent)"
    else:
        raise ValueError(f"Invalid model_type: {model_type}. Use 'f' or 'g'")
    
    # Initialize the model
    in_channels = train_data.num_features
    hidden_channels = args.hidden_dim
    out_channels = args.out_dim
    
    logger.info(f"\nTraining model {model_name} with seed {args.seed}")
    logger.info(f"Training edges: {len(edge_labels)}")
    logger.info(f"  - Positive edges: {torch.sum(edge_labels).item()}")
    logger.info(f"  - Negative edges: {len(edge_labels) - torch.sum(edge_labels).item()}")
    logger.info(f"  - Using per-epoch negative sampling for categories: {edge_categories}")
    
    for category in edge_categories:
        logger.info(f"    - {category}: {edge_counts[category]} negative edges per epoch")
    
    # Set seed for reproducibility
    if hasattr(torch, 'set_deterministic'):
        torch.set_deterministic(True)
    
    # Set the seed (using the one from args)
    set_seed(args.seed)
    
    # Create model - select class based on decoder type
    if hasattr(args, 'decoder') and args.decoder == 'mlp':
        model = LinkGNN_MLP(
            in_channels=in_channels, 
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=args.num_layers,
            model_type=args.model_type,
            dropout=args.dropout if hasattr(args, 'dropout') else 0.5,
            heads=args.heads if args.model_type.lower() == 'gat' else None
        ).to(device)
        logger.info(f"Using LinkGNN_MLP with MLP decoder")
    else:
        model = LinkGNN(
            in_channels=in_channels, 
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=args.num_layers,
            model_type=args.model_type,
            heads=args.heads if args.model_type.lower() == 'gat' else None
        ).to(device)
        logger.info(f"Using LinkGNN with dot product decoder")
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Train model with per-epoch negative sampling
    logger.info(f"Training model {model_name} with per-epoch negative sampling...")
    model = train_link_prediction_model(
        model=model, 
        optimizer=optimizer, 
        data=train_data, 
        train_edges=train_edges, 
        edge_labels=edge_labels, 
        epochs=args.epochs,
        pos_edge_index=pos_edge_index,
        edge_counts=edge_counts,
        edges_dict=edges_dict_copy,
        edge_categories=edge_categories
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
    
    val_auc = evaluate_model(model, train_data, val_edge_index, val_edge_label)
    logger.info(f"Model {model_name} - Validation AUC: {val_auc:.4f}")
    
    return model

def train_models_fixed(data_full, edges_dict, args, logger):
    """
    Fixed version of train_models that uses the provided seed directly
    rather than running with multiple hardcoded seeds.
    
    Args:
        data_full: Original data object with all graph information
        edges_dict: Dictionary with edge splits
        args: Arguments dictionary with model parameters
        logger: Logger object
    
    Returns:
        Tuple of (model_f, model_g)
    """
    # Extract training data
    train_data, val_data, test_data = data_full[0]
    
    # Log model architecture details
    in_channels = train_data.num_features
    hidden_channels = args.hidden_dim
    out_channels = args.out_dim
    
    logger.info("\nModel Architecture Details:")
    logger.info(f"Model Type: {args.model_type.upper()}")
    logger.info(f"Decoder Type: {args.decoder if hasattr(args, 'decoder') else 'dot'}")
    logger.info(f"Input Features: {in_channels}")
    logger.info(f"Hidden Dimensions: {hidden_channels}")
    logger.info(f"Output Dimensions: {out_channels}")
    logger.info(f"Number of Layers: {args.num_layers}")
    logger.info(f"Learning Rate: {args.lr}")
    logger.info(f"Weight Decay: {args.weight_decay}")
    logger.info(f"Dropout Rate: {args.dropout if hasattr(args, 'dropout') else 0.5}")
    logger.info(f"Using seed: {args.seed}")
    
    # Train each model with the provided seed
    model_f = train_model_single_seed(data_full, edges_dict, args, logger, model_type='f')
    
    # Important: Reset the seed before training model_g to ensure proper initialization
    set_seed(args.seed)
    model_g = train_model_single_seed(data_full, edges_dict, args, logger, model_type='g')
    
    return model_f, model_g