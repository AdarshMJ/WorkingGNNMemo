import torch
from torch_geometric.nn import GCNConv, GATConv, GraphConv
from torch_geometric.nn import MessagePassing


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


class LinkGNN_MLP(torch.nn.Module):
    """
    GNN model for link prediction with MLP-based link prediction layer.
    Uses GNN layers for node encoding and an MLP for link prediction.
    This may improve performance on heterophilic graphs.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, 
                 model_type='gcn', dropout=0.61, **kwargs):
        super().__init__()
        self.model_type = model_type.lower()
        self.dropout = dropout
        self.out_channels = out_channels
        
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
        
        # Output layer for node embeddings
        if self.model_type == 'gat' and num_layers > 1:
            self.convs.append(
                GNNLayer(hidden_channels * self.kwargs['heads'], 
                         out_channels, **self.kwargs)
            )
        else:
            self.convs.append(
                GNNLayer(hidden_channels, out_channels, **self.kwargs)
            )
        
        # MLP link predictor - takes the combined node embeddings and predicts link likelihood
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(out_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, 1)
        )

    def encode(self, x, edge_index):
        """Encode node features to embeddings."""
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = x.relu()
            if self.model_type != 'gat':  # GAT already applied activation in the layer
                x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        
        # Last layer without dropout
        x = self.convs[-1](x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        """
        Predict links using an MLP on element-wise product of node embeddings.
        This provides more expressive power than a simple dot product.
        """
        # Get node embeddings for the edges
        src_embeddings = z[edge_label_index[0]]
        dst_embeddings = z[edge_label_index[1]]
        
        # Element-wise product as in the DGL implementation
        edge_features = src_embeddings * dst_embeddings
        
        # Pass through MLP predictor
        return self.predictor(edge_features).squeeze(-1)

    def decode_all(self, z):
        """
        Decode all possible edges.
        Note: This is computationally expensive for large graphs
        as it requires computing predictions for all node pairs.
        """
        num_nodes = z.size(0)
        device = z.device
        
        # Pre-filter potential edges using dot product for efficiency
        prob_adj = z @ z.t()
        potential_edges = (prob_adj > 0).nonzero(as_tuple=False)
        
        # Only compute MLP predictions for potential edges
        src_indices = potential_edges[:, 0]
        dst_indices = potential_edges[:, 1]
        
        edge_index = torch.stack([src_indices, dst_indices], dim=0)
        scores = self.decode(z, edge_index)
        
        # Return edges with positive scores
        positive_mask = scores > 0
        return edge_index[:, positive_mask]
    
    def decode_batch(self, z, batch_size=10000):
        """
        Decode all edges in batches for memory efficiency.
        Useful for very large graphs.
        """
        num_nodes = z.size(0)
        device = z.device
        edge_indices = []
        
        for i in range(0, num_nodes, batch_size):
            rows = torch.arange(i, min(i + batch_size, num_nodes), device=device)
            cols = torch.arange(num_nodes, device=device)
            
            # Create grid of all pairs for this batch
            row_grid, col_grid = torch.meshgrid(rows, cols)
            
            # Convert to edge index format
            src = row_grid.flatten()
            dst = col_grid.flatten()
            
            edge_index = torch.stack([src, dst], dim=0)
            scores = self.decode(z, edge_index)
            
            # Only keep positive scores
            positive_mask = scores > 0
            edge_indices.append(edge_index[:, positive_mask])
        
        return torch.cat(edge_indices, dim=1) if edge_indices else torch.zeros((2, 0), device=device)
