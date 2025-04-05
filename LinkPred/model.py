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
