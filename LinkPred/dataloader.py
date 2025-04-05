import numpy as np
import torch
from torch_geometric.data import Data
import os.path as osp
import scipy.sparse as sp
from torch_geometric.transforms import LargestConnectedComponents, RandomNodeSplit
import logging

def load_npz_dataset(dataset_name, data_dir='data'):
    """
    Load a heterophilic dataset from a .npz file.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'cornell', 'texas')
        data_dir: Directory containing the dataset files
    
    Returns:
        PyG Data object
    """
    # Handle dataset naming
    if dataset_name.lower() == 'chameleon':
        npz_file = 'chameleon_filtered.npz'
    elif dataset_name.lower() == 'squirrel':
        npz_file = 'squirrel_filtered.npz'
    else:
        npz_file = f'{dataset_name.lower()}.npz'
    
    # Construct file path
    file_path = osp.join(data_dir, npz_file)
    
    # Check if file exists
    if not osp.exists(file_path):
        raise FileNotFoundError(f"Dataset file {file_path} not found")
    
    # Load the .npz file using the provided conversion code
    try:
        data = np.load(file_path, allow_pickle=True)
        print(f"Converting {dataset_name} to PyG dataset...")
        
        # Extract features, labels, and edges
        if 'node_features' in data:
            x = torch.tensor(data['node_features'], dtype=torch.float)
        elif 'features' in data:
            x = torch.tensor(data['features'], dtype=torch.float)
        else:
            # Try alternative keys
            keys = list(data.keys())
            if 'x' in keys:
                x = torch.tensor(data['x'], dtype=torch.float)
            elif 'attr_matrix' in data:
                x = torch.tensor(data['attr_matrix'], dtype=torch.float)
            else:
                raise ValueError(f"Could not find node features in {npz_file}")
        
        # Extract labels
        if 'node_labels' in data:
            y = torch.tensor(data['node_labels'], dtype=torch.long)
        elif 'labels' in data:
            y = torch.tensor(data['labels'], dtype=torch.long)
        else:
            # Try alternative keys
            keys = list(data.keys())
            if 'y' in keys:
                y = torch.tensor(data['y'], dtype=torch.long)
            elif 'class_labels' in data:
                y = torch.tensor(data['class_labels'], dtype=torch.long)
            else:
                raise ValueError(f"Could not find node labels in {npz_file}")
        
        # Ensure y is properly shaped
        if y.ndim > 1 and y.shape[1] > 1:
            # If one-hot encoded, convert to class indices
            y = y.argmax(dim=1)
        
        # Extract edges
        if 'edges' in data:
            edge_index = torch.tensor(data['edges'], dtype=torch.long).t().contiguous()
        elif 'edge_index' in data:
            edge_index = torch.tensor(data['edge_index'], dtype=torch.long)
        elif 'adj_matrix' in data:
            adj_matrix = data['adj_matrix']
            if sp.issparse(adj_matrix):
                adj_matrix = adj_matrix.tocoo()
                edge_index = torch.tensor(np.vstack((adj_matrix.row, adj_matrix.col)), dtype=torch.long)
            else:
                # Convert dense adjacency matrix to edge index
                adj_matrix = sp.csr_matrix(adj_matrix)
                adj_matrix = adj_matrix.tocoo()
                edge_index = torch.tensor(np.vstack((adj_matrix.row, adj_matrix.col)), dtype=torch.long)
        else:
            # Try alternative keys for adjacency
            keys = list(data.keys())
            if 'network' in keys:
                adj_matrix = data['network']
                if sp.issparse(adj_matrix):
                    adj_matrix = adj_matrix.tocoo()
                else:
                    adj_matrix = sp.csr_matrix(adj_matrix)
                    adj_matrix = adj_matrix.tocoo()
                edge_index = torch.tensor(np.vstack((adj_matrix.row, adj_matrix.col)), dtype=torch.long)
            elif 'adj' in keys:
                adj_matrix = data['adj']
                if sp.issparse(adj_matrix):
                    adj_matrix = adj_matrix.tocoo()
                else:
                    adj_matrix = sp.csr_matrix(adj_matrix)
                    adj_matrix = adj_matrix.tocoo()
                edge_index = torch.tensor(np.vstack((adj_matrix.row, adj_matrix.col)), dtype=torch.long)
            else:
                raise ValueError(f"Could not find edges or adjacency matrix in {npz_file}")
        
        # Extract masks if available, otherwise create them
        if 'train_masks' in data and 'val_masks' in data and 'test_masks' in data:
            try:
                train_mask = torch.tensor(data['train_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
                val_mask = torch.tensor(data['val_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
                test_mask = torch.tensor(data['test_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
                
                # If multi-dimensional, take first split
                if train_mask.dim() > 1:
                    train_mask = train_mask[0]
                if val_mask.dim() > 1:
                    val_mask = val_mask[0]
                if test_mask.dim() > 1:
                    test_mask = test_mask[0]
            except Exception as e:
                print(f"Error processing masks, will create new splits: {e}")
                train_mask = val_mask = test_mask = None
        else:
            train_mask = val_mask = test_mask = None
            
        # Create PyG data object
        if train_mask is not None and val_mask is not None and test_mask is not None:
            pyg_data = Data(x=x, edge_index=edge_index, y=y, 
                           train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        else:
            pyg_data = Data(x=x, edge_index=edge_index, y=y)
        
        # Set number of classes
        num_classes = len(torch.unique(y))
        pyg_data.num_classes = num_classes
        
        # Apply LargestConnectedComponents transformation
        print(f"Selecting the LargestConnectedComponent...")
        transform = LargestConnectedComponents()
        pyg_data = transform(pyg_data)
        
        # If masks not provided, create them
        if train_mask is None or val_mask is None or test_mask is None:
            print("\nSplitting dataset into train/val/test...")
            transform2 = RandomNodeSplit(split="train_rest", num_val=0.2, num_test=0.2)
            pyg_data = transform2(pyg_data)
        
        print(f"Dataset loaded: {pyg_data}")
        print(f"Number of features: {pyg_data.num_features}")
        print(f"Number of classes: {pyg_data.num_classes}")
        print("Done!..")
        
        return pyg_data
        
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        raise

def process_heterophilic_dataset_for_link_prediction(data, val_ratio=0.05, test_ratio=0.1):
    """
    Process a heterophilic dataset for link prediction task.
    
    Args:
        data: PyG Data object
        val_ratio: Ratio of edges to use for validation
        test_ratio: Ratio of edges to use for testing
    
    Returns:
        Processed PyG Data object ready for link prediction
    """
    # Add train_mask, val_mask, test_mask if they don't exist
    if not hasattr(data, 'train_mask') or not hasattr(data, 'val_mask') or not hasattr(data, 'test_mask'):
        num_nodes = data.num_nodes
        indices = torch.randperm(num_nodes)
        
        # Calculate split sizes
        test_size = int(num_nodes * test_ratio)
        val_size = int(num_nodes * val_ratio)
        
        # Create masks
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        test_mask[indices[:test_size]] = True
        val_mask[indices[test_size:test_size + val_size]] = True
        train_mask[indices[test_size + val_size:]] = True
        
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
    
    return data

def get_heterophilic_datasets():
    """Return a list of available heterophilic datasets."""
    return ['cornell', 'texas', 'wisconsin', 'chameleon', 'squirrel', 'actor', 
            'amazon_ratings', 'roman_empire', 'questions', 'tolokers', 'minesweeper']

if __name__ == "__main__":
    # Test loading a dataset
    try:
        data = load_npz_dataset('cornell')
        print(f"Loaded cornell dataset successfully")
        print(f"Number of nodes: {data.num_nodes}")
        print(f"Number of edges: {data.edge_index.size(1)}")
        print(f"Number of features: {data.num_features}")
        print(f"Number of classes: {data.num_classes}")
        
        # Test processing for link prediction
        processed_data = process_heterophilic_dataset_for_link_prediction(data)
        print(f"Processed data for link prediction")
    except Exception as e:
        print(f"Error loading dataset: {e}")