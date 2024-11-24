from torch_geometric.nn import to_hetero
import torch.nn.functional as F
from torch import Tensor
from torch import nn
from torch_geometric.data import Data, HeteroData
import torch

from gs import GS
from gat import GAT
from gatv2 import GATv2
from han import HAN
from gin import GIN
from gine import GINE
from decoder import Decoder

class Encoder(nn.Module):
    def __init__(self, hidden_channels, data, model_name):
        super().__init__()

        self.model_name = model_name
        # Define a dictionary that maps model names to class constructors
        model_classes = {
            'gs': GS,
            'gat': GAT,
            'gatv2': GATv2,
            'gin': GIN,
            'gine': GINE
        }

        if isinstance(data, HeteroData):
            self.node_lin = nn.ModuleList()
            self.node_emb = nn.ModuleList()
            # for each node_type
            node_types = data.node_types
            # linear transformers and node embeddings based on the num_features and num_nodes of the node_types
            for i, node_type in enumerate(node_types):
                if data.is_cuda:
                    self.node_lin.append(nn.Linear(data[node_type].num_features, hidden_channels).cuda())
                    self.node_emb.append(nn.Embedding(data[node_type].num_nodes, hidden_channels).cuda())
                else:
                    self.node_lin.append(nn.Linear(data[node_type].num_features, hidden_channels))
                    self.node_emb.append(nn.Embedding(data[node_type].num_nodes, hidden_channels))
        else:
            if data.is_cuda:
                self.node_lin = nn.Linear(data.num_features, hidden_channels).cuda()
                self.node_emb = nn.Linear(data.num_nodes, hidden_channels).cuda()
            else:
                self.node_lin = nn.Linear(data.num_features, hidden_channels)
                self.node_emb = nn.Linear(data.num_nodes, hidden_channels)

        # Instantiate homogeneous model:
        if self.model_name == 'han':
            self.model = HAN(hidden_channels, data.metadata())
        else:
            self.model = model_classes[model_name](hidden_channels)

        # instantiate the predictor class
        self.decoder = Decoder()

    def forward(self, data, seed_edge_type, is_directed, emb=False) -> Tensor:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if isinstance(data, HeteroData):
            self.x_dict = {}
            for i, node_type in enumerate(data.node_types):
                # Add node IDs if they don't exist
                if not hasattr(data[node_type], 'n_id'):
                    data[node_type].n_id = torch.arange(data[node_type].num_nodes, device=device)
                
                # Ensure all tensors are on the correct device
                data[node_type].x = data[node_type].x.to(device)
                data[node_type].n_id = data[node_type].n_id.to(device)
                
                # Move model components to device
                self.node_lin[i] = self.node_lin[i].to(device)
                self.node_emb[i] = self.node_emb[i].to(device)
                
                self.x_dict[node_type] = (self.node_lin[i](data[node_type].x) + 
                                        self.node_emb[i](data[node_type].n_id))

            # Process edge indices
            edge_index_dict = {}
            for edge_type in data.edge_types:
                if not hasattr(data[edge_type], 'edge_index'):
                    continue
                    
                edge_index = data[edge_type].edge_index
                if not isinstance(edge_index, torch.Tensor):
                    edge_index = torch.tensor(edge_index, dtype=torch.long, device=device)
                else:
                    edge_index = edge_index.to(device)
                    
                if edge_index.dim() != 2 or edge_index.size(0) != 2:
                    edge_index = edge_index.t().contiguous()
                
                edge_index_dict[edge_type] = edge_index

            # Process the data through the model
            self.x_dict = self.model(self.x_dict, edge_index_dict)
                
            if emb:
                return self.x_dict
                
        else:
            # Handle homogeneous graphs
            if not hasattr(data, 'n_id'):
                data.n_id = torch.arange(data.num_nodes, device=device)
            
            # Move all tensors and model components to device
            data.x = data.x.to(device)
            data.n_id = data.n_id.to(device)
            self.node_lin = self.node_lin.to(device)
            self.node_emb = self.node_emb.to(device)
            
            x = self.node_lin(data.x) + self.node_emb(data.n_id)
            
            # Process edge index
            edge_index = data.edge_index.to(device)
            if not isinstance(edge_index, torch.Tensor):
                edge_index = torch.tensor(edge_index, dtype=torch.long, device=device)
            
            if edge_index.dim() != 2 or edge_index.size(0) != 2:
                edge_index = edge_index.t().contiguous()
                
            x = self.model(x, edge_index)
            self.x = x
            if emb:
                return self.x

        # Create predictions
        preds = torch.empty(0, device=device)

        if isinstance(data, HeteroData):
            source_node_emb = self.x_dict[seed_edge_type[0]]
            target_node_emb = self.x_dict[seed_edge_type[2]]
            edge_label_index = data[seed_edge_type].edge_label_index.to(device)
            pred = self.decoder(source_node_emb, target_node_emb, edge_label_index)
            preds = torch.cat((preds, pred.unsqueeze(0)), dim=1)
        else:
            edge_label_index = data.edge_label_index.to(device)
            pred = self.decoder(x, x, edge_label_index)
            preds = torch.cat((preds, pred.unsqueeze(0)), dim=1)
            
        return preds.squeeze(dim=0)