import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GS(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.convs = None
        self.is_hetero = False

    def create_hetero_convs(self, metadata):
        """Create separate convolutions for each edge type."""
        self.convs = nn.ModuleDict()
        
        for edge_type in metadata[1]:  # edge_types
            # Create 4 layers of convolutions for this edge type
            for i in range(1, 5):
                in_channels = self.hidden_channels if i > 1 else (-1, -1)
                self.convs[f"{edge_type}_{i}"] = SAGEConv(
                    in_channels=in_channels,
                    out_channels=self.hidden_channels,
                    aggr='mean'
                ).to(self.device)
        
        self.is_hetero = True

    def forward(self, x_dict, edge_index_dict):
        try:
            if isinstance(x_dict, dict):
                if not self.is_hetero:
                    metadata = (list(x_dict.keys()), list(edge_index_dict.keys()))
                    self.create_hetero_convs(metadata)

                # Move to device and print shapes
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                print("\nInput node feature shapes:")
                for k, v in x_dict.items():
                    print(f"{k}: {v.shape}")

                # Process edge indices
                edge_index_dict = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) 
                    else torch.tensor(v, dtype=torch.long, device=self.device)
                    for k, v in edge_index_dict.items()
                }
                
                # Ensure correct edge index format and print info
                for edge_type in edge_index_dict:
                    if edge_index_dict[edge_type].shape[0] != 2:
                        edge_index_dict[edge_type] = edge_index_dict[edge_type].t().contiguous()
                    print(f"\nEdge type {edge_type}:")
                    print(f"  Shape: {edge_index_dict[edge_type].shape}")
                    print(f"  Max source index: {edge_index_dict[edge_type][0].max()}")
                    print(f"  Max target index: {edge_index_dict[edge_type][1].max()}")

                # Message passing layers
                x_dicts = [x_dict]  # Store all intermediate representations
                
                # Process all 4 layers
                for layer in range(1, 5):
                    x_dict_current = {}
                    
                    # Initialize tensors for each node type using their original sizes
                    for node_type, features in x_dicts[-1].items():
                        x_dict_current[node_type] = torch.zeros(
                            (features.size(0), self.hidden_channels),
                            device=self.device
                        )
                    
                    # Aggregate messages for each edge type
                    message_counts = {node_type: 0 for node_type in x_dict_current}
                    
                    for edge_type in edge_index_dict:
                        src_type, _, dst_type = edge_type
                        edge_index = edge_index_dict[edge_type]
                        conv = self.convs[f"{edge_type}_{layer}"]
                        
                        # Get source features
                        source_x = x_dicts[-1][src_type]
                        
                        # Compute messages
                        messages = conv(source_x, edge_index)
                        x_dict_current[dst_type] += messages
                        message_counts[dst_type] += 1
                    
                    # Average the messages and apply activation
                    for node_type in x_dict_current:
                        if message_counts[node_type] > 0:
                            x_dict_current[node_type] = x_dict_current[node_type] / message_counts[node_type]
                        x_dict_current[node_type] = F.relu(x_dict_current[node_type])
                    
                    # Add residual connection if not first layer
                    if layer > 1:
                        for node_type in x_dict_current:
                            x_dict_current[node_type] = x_dict_current[node_type] + x_dicts[-1][node_type]
                    
                    # Store this layer's output
                    x_dicts.append(x_dict_current)
                
                # Return final layer output
                return x_dicts[-1]

            else:
                raise ValueError("Only heterogeneous graphs are supported in this implementation")

        except Exception as e:
            print("\nError in GraphSAGE forward pass:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            if isinstance(x_dict, dict):
                print("\nNode types and shapes:")
                for k, v in x_dict.items():
                    print(f"{k}: {v.shape}")
                print("\nEdge types and shapes:")
                for k, v in edge_index_dict.items():
                    print(f"{k}: {v.shape}")
            raise e

    def __repr__(self):
        return f'GS(hidden_channels={self.hidden_channels})'