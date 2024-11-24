import torch
from torch import nn
import torch.nn.functional as F

class MessageAwareGATLayer(nn.Module):
    """GAT layer that handles different source and target dimensions."""
    def __init__(self, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        # Attention components
        self.attn_src = nn.Linear(hidden_channels, hidden_channels)
        self.attn_dst = nn.Linear(hidden_channels, hidden_channels)
        self.attn_score = nn.Linear(hidden_channels * 2, 1)
        
    def forward(self, src_feats, dst_feats, edge_index):
        """
        src_feats: features of source nodes
        dst_feats: features of target nodes
        edge_index: [2, num_edges] tensor of edge indices
        """
        # Get source and target nodes for edges
        src_idx, dst_idx = edge_index
        
        # Get features for connected nodes
        src = src_feats[src_idx]  # [num_edges, channels]
        dst = dst_feats[dst_idx]  # [num_edges, channels]
        
        # Transform features
        src_transformed = self.attn_src(src)
        dst_transformed = self.attn_dst(dst)
        
        # Compute attention scores
        attn_input = torch.cat([src_transformed, dst_transformed], dim=-1)
        attn_weights = F.leaky_relu(self.attn_score(attn_input))
        attn_weights = F.softmax(attn_weights, dim=0)
        
        # Apply attention to source features
        messages = src * attn_weights
        
        # Aggregate messages for each target node
        out = torch.zeros_like(dst_feats)
        out.index_add_(0, dst_idx, messages)
        
        return out

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.layers = None
        self.is_initialized = False

    def _init_layers(self, metadata):
        """Initialize layers for all edge types."""
        self.layers = nn.ModuleDict()
        self.node_transforms = nn.ModuleDict()
        
        # Create layers for each edge type
        for edge_type in metadata[1]:
            self.layers[str(edge_type)] = nn.ModuleList([
                MessageAwareGATLayer(self.hidden_channels).to(self.device)
                for _ in range(4)  # 4 layers
            ])
        
        # Create node-specific transformations
        for node_type in metadata[0]:
            self.node_transforms[node_type] = nn.Sequential(
                nn.Linear(self.hidden_channels, self.hidden_channels),
                nn.LayerNorm(self.hidden_channels),
                nn.ReLU(),
                nn.Dropout(0.1)
            ).to(self.device)
        
        self.is_initialized = True

    def _process_single_layer(self, x_dict, edge_index_dict, layer_idx):
        """Process a single layer of the network."""
        next_x = {
            node_type: torch.zeros_like(features)
            for node_type, features in x_dict.items()
        }
        message_counts = {
            node_type: torch.zeros(features.size(0), device=features.device)
            for node_type, features in x_dict.items()
        }
        
        # Process each edge type
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            layer = self.layers[str(edge_type)][layer_idx]
            
            try:
                # Get valid edge mask
                src_idx, dst_idx = edge_index
                valid_edges = (src_idx < x_dict[src_type].size(0)) & (dst_idx < x_dict[dst_type].size(0))
                
                if valid_edges.any():
                    # Process only valid edges
                    valid_edge_index = edge_index[:, valid_edges]
                    
                    # Compute messages
                    messages = layer(
                        x_dict[src_type],
                        x_dict[dst_type],
                        valid_edge_index
                    )
                    
                    # Accumulate messages
                    next_x[dst_type] += messages
                    message_counts[dst_type] += torch.ones_like(messages[:, 0])
            
            except Exception as e:
                print(f"\nError processing edge type {edge_type} in layer {layer_idx}:")
                print(f"Source shape: {x_dict[src_type].shape}")
                print(f"Target shape: {x_dict[dst_type].shape}")
                print(f"Edge index shape: {edge_index.shape}")
                raise e
        
        # Average messages and apply transformations
        for node_type in x_dict:
            # Avoid division by zero
            mask = message_counts[node_type] > 0
            if mask.any():
                next_x[node_type][mask] = next_x[node_type][mask] / message_counts[node_type][mask].unsqueeze(1)
            
            # Apply node-specific transformation
            next_x[node_type] = self.node_transforms[node_type](next_x[node_type])
        
        return next_x

    def forward(self, x_dict, edge_index_dict):
        try:
            # Initialize if needed
            if not self.is_initialized:
                metadata = (list(x_dict.keys()), list(edge_index_dict.keys()))
                self._init_layers(metadata)

            # Move to device
            x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
            edge_index_dict = {k: v.to(self.device) for k, v in edge_index_dict.items()}

            # Print input shapes
            print("\nInput shapes:")
            for k, v in x_dict.items():
                print(f"{k}: {v.shape}")
            for k, v in edge_index_dict.items():
                print(f"Edge {k}: {v.shape}")

            # Process through layers
            current_x = x_dict
            for layer_idx in range(4):
                next_x = self._process_single_layer(current_x, edge_index_dict, layer_idx)
                
                # Add residual connection after first layer
                if layer_idx > 0:
                    for node_type in next_x:
                        next_x[node_type] = next_x[node_type] + current_x[node_type]
                
                current_x = next_x

            return current_x

        except Exception as e:
            print("\nError in GAT forward pass:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            for k, v in x_dict.items():
                print(f"{k} shape: {v.shape}")
            for k, v in edge_index_dict.items():
                print(f"Edge {k} shape: {v.shape}")
            raise e

    def __repr__(self):
        return f'GAT(hidden_channels={self.hidden_channels})'