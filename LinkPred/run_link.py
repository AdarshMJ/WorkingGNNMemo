import os.path as osp

import torch
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from sklearn.decomposition import PCA
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
import numpy as np
import random
import os

# Create a directory for saving visualizations
os.makedirs('visualizations', exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def visualize_graph_split(train_data, val_data, test_data):
    """Visualize the train/val/test split of the graph."""
    plt.figure(figsize=(18, 6))
    
    # Convert to networkx graphs
    G_train = to_networkx(train_data, to_undirected=True)
    
    # Get val and test positive edges
    val_pos_edges = val_data.edge_label_index[:, val_data.edge_label == 1].t().cpu().numpy()
    test_pos_edges = test_data.edge_label_index[:, test_data.edge_label == 1].t().cpu().numpy()
    
    # Create a combined graph for visualization
    G_full = nx.Graph()
    G_full.add_nodes_from(range(train_data.num_nodes))
    
    # Add all edges with different colors
    train_edges = train_data.edge_index.t().cpu().numpy()
    
    # Plot the three subplots
    plt.subplot(131)
    pos = nx.spring_layout(G_train, seed=42)  # Use same layout for all graphs
    
    nx.draw_networkx_nodes(G_train, pos, node_size=30, node_color='lightblue')
    nx.draw_networkx_edges(G_train, pos, width=0.5, alpha=0.5)
    plt.title(f'Training Graph\n({len(train_edges)} edges)')
    plt.axis('off')
    
    plt.subplot(132)
    nx.draw_networkx_nodes(G_train, pos, node_size=30, node_color='lightblue')
    nx.draw_networkx_edges(G_train, pos, width=0.5, alpha=0.2)  # Draw training edges faded
    # Draw validation edges
    edge_list = [(int(u), int(v)) for u, v in val_pos_edges]
    nx.draw_networkx_edges(G_full, pos, edgelist=edge_list, width=1.5, edge_color='green')
    plt.title(f'Validation Positive Edges\n({len(val_pos_edges)} edges)')
    plt.axis('off')
    
    plt.subplot(133)
    nx.draw_networkx_nodes(G_train, pos, node_size=30, node_color='lightblue')
    nx.draw_networkx_edges(G_train, pos, width=0.5, alpha=0.2)  # Draw training edges faded
    # Draw test edges
    edge_list = [(int(u), int(v)) for u, v in test_pos_edges]
    nx.draw_networkx_edges(G_full, pos, edgelist=edge_list, width=1.5, edge_color='red')
    plt.title(f'Test Positive Edges\n({len(test_pos_edges)} edges)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('visualizations/graph_split.png', dpi=300)
    plt.close()
    print("Graph split visualization saved to 'visualizations/graph_split.png'")


def visualize_negative_sampling(train_data):
    """Visualize the negative sampling process."""
    plt.figure(figsize=(12, 5))
    
    # Convert to networkx graph
    G = to_networkx(train_data, to_undirected=True)
    
    # Create negative samples
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
        num_neg_samples=min(100, train_data.num_nodes), method='sparse')
    
    pos = nx.spring_layout(G, seed=42)
    
    plt.subplot(121)
    nx.draw_networkx_nodes(G, pos, node_size=30, node_color='lightblue')
    nx.draw_networkx_edges(G, pos, width=0.5)
    plt.title('Positive Edges (Existing)')
    plt.axis('off')
    
    plt.subplot(122)
    nx.draw_networkx_nodes(G, pos, node_size=30, node_color='lightblue')
    # Create a temporary graph to draw negative edges
    G_neg = nx.Graph()
    G_neg.add_nodes_from(G.nodes())
    neg_edges = [(int(u), int(v)) for u, v in neg_edge_index.t().cpu().numpy()]
    G_neg.add_edges_from(neg_edges)
    nx.draw_networkx_edges(G_neg, pos, width=0.5, edge_color='red', style='dashed')
    plt.title('Negative Edges (Sampled Non-edges)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('visualizations/negative_sampling.png', dpi=300)
    plt.close()
    print("Negative sampling visualization saved to 'visualizations/negative_sampling.png'")


def visualize_model_performance(model, test_data):
    """Visualize the model's prediction performance."""
    plt.figure(figsize=(15, 5))
    
    model.eval()
    
    with torch.no_grad():
        z = model.encode(test_data.x, test_data.edge_index)
        out = model.decode(z, test_data.edge_label_index).sigmoid()
        
        predictions = out.cpu().numpy()
        labels = test_data.edge_label.cpu().numpy()
        
        # ROC curve
        plt.subplot(131)
        fpr, tpr, _ = roc_curve(labels, predictions)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        
        # Precision-Recall curve
        plt.subplot(132)
        precision, recall, _ = precision_recall_curve(labels, predictions)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        
        # Score distribution
        plt.subplot(133)
        pos_scores = predictions[labels == 1]
        neg_scores = predictions[labels == 0]
        
        plt.hist(pos_scores, bins=20, alpha=0.5, density=True, color='green', label='Positive edges')
        plt.hist(neg_scores, bins=20, alpha=0.5, density=True, color='red', label='Negative edges')
        plt.xlabel('Prediction Score')
        plt.ylabel('Density')
        plt.title('Score Distribution')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('visualizations/model_performance.png', dpi=300)
    plt.close()
    print("Model performance visualization saved to 'visualizations/model_performance.png'")


def visualize_embedding_space(model, test_data):
    """Visualize node embeddings in 2D space with edges."""
   
    
    plt.figure(figsize=(10, 8))
    
    model.eval()
    
    with torch.no_grad():
        # Get node embeddings
        z = model.encode(test_data.x, test_data.edge_index)
        z_np = z.cpu().numpy()
        
        # Reduce dimensionality to 2D
        pca = PCA(n_components=2)
        z_2d = pca.fit_transform(z_np)
        
        # Plot node embeddings
        plt.scatter(z_2d[:, 0], z_2d[:, 1], s=10, c='lightblue')
        
        # Plot a sample of edges from the test set (positive)
        pos_edge_indices = test_data.edge_label_index[:, test_data.edge_label == 1]
        
        # Sample edges for visualization (to avoid clutter)
        max_edges = 100
        if pos_edge_indices.shape[1] > max_edges:
            idx = torch.randperm(pos_edge_indices.shape[1])[:max_edges]
            pos_edge_indices = pos_edge_indices[:, idx]
        
        # Draw edges
        for i in range(pos_edge_indices.shape[1]):
            src, dst = pos_edge_indices[0, i], pos_edge_indices[1, i]
            plt.plot([z_2d[src, 0], z_2d[dst, 0]], 
                     [z_2d[src, 1], z_2d[dst, 1]], 
                     'k-', alpha=0.1)
        
        plt.title('2D Visualization of Node Embeddings with Test Edges')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
    
    plt.tight_layout()
    plt.savefig('visualizations/embedding_space.png', dpi=300)
    plt.close()
    print("Embedding space visualization saved to 'visualizations/embedding_space.png'")

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')


print(f'Using device: {device}')

###All hyperparams#####
hidden_channels = 32
out_channels = 32
num_layers = 2
learning_rate = 0.01
num_epochs = 100
weight_decay = 0.0



transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device),
    T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                      add_negative_train_samples=False),
])
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset = Planetoid(path, name='Cora', transform=transform)
print(dataset[0])
# After applying the `RandomLinkSplit` transform, the data is transformed from
# a data object to a list of tuples (train_data, val_data, test_data), with
# each element representing the corresponding split.
train_data, val_data, test_data = dataset[0]

# Visualize the train/val/test split of the graph
visualize_graph_split(train_data, val_data, test_data)

# Visualize negative sampling
visualize_negative_sampling(train_data)

set_seed(42)
class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


model = Net(dataset.num_features, hidden_channels, out_channels).to(device)
print(model)
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = torch.nn.BCEWithLogitsLoss()


def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)

    # We perform a new round of negative sampling for every training epoch:
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat([
        train_data.edge_label,
        train_data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)

    out = model.decode(z, edge_label_index).view(-1)
    loss = criterion(out, edge_label)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())


best_val_auc = final_test_auc = 0
for epoch in range(1, 101):
    loss = train()
    val_auc = test(val_data)
    test_auc = test(test_data)
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        final_test_auc = test_auc
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, '
          f'Test: {test_auc:.4f}')

print(f'Final Test: {final_test_auc:.4f}')

# Add model performance and embedding visualizations after training
visualize_model_performance(model, test_data)
visualize_embedding_space(model, test_data)

z = model.encode(test_data.x, test_data.edge_index)
final_edge_index = model.decode_all(z)

