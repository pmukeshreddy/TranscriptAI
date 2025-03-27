import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import GATv2Conv

from .attention import StructureAwareattenation
from .transformer import StructureAwareTransformerEncoder, StructureAwareTransformerLayer
from .refinement import CoordinateRefinementModule

class RNAGraphTransformerWithRefinement(nn.Module):
    """
    Graph transformer model for RNA 3D structure prediction with iterative refinement.
    """
    def __init__(
        self,
        node_feature,
        edge_feature,
        hidden_dim=128,
        num_gcn_layers=8,
        num_transformer_layers=7,
        num_heads=4,
        drop_out=0.1,
        max_length=300,
        num_iterations=3,
        max_distance=20.0
    ):
        super(RNAGraphTransformerWithRefinement, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_gcn_layers = num_gcn_layers
        self.max_length = max_length
        self.num_transformer_layer = num_transformer_layers
        self.node_embedding = nn.Linear(node_feature, hidden_dim)
        self.edge_embedding = nn.Linear(edge_feature, hidden_dim)
        self.num_iterations = num_iterations
        
        # GNN layers
        self.gcn_layers = nn.ModuleList()
        self.layer_norm_gcn = nn.ModuleList()
        for i in range(num_gcn_layers):
            self.gcn_layers.append(
                GATv2Conv(
                    hidden_dim,
                    hidden_dim//num_heads,
                    heads=num_heads,
                    edge_dim=hidden_dim,
                    concat=True,
                    dropout=drop_out
                )
            )
            self.layer_norm_gcn.append(nn.LayerNorm(hidden_dim))
        
        # Transformer components
        encoder_layer = StructureAwareTransformerLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=drop_out
        )
        self.transformer_encoder = StructureAwareTransformerEncoder(
            encoder_layer,
            self.num_transformer_layer
        )
        
        # Structure-aware attention
        self.structure_aware_attention = StructureAwareattenation(
            self.hidden_dim,
            num_heads,
            max_distance=max_distance
        )
        
        # Coordinate prediction components
        self.initial_coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.GELU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_out),
            nn.Linear(hidden_dim, 3)
        )
        
        self.coord_refinement = CoordinateRefinementModule(hidden_dim)
        self.pos_dropout = nn.Dropout(drop_out)
        self.position_encoding = self._create_position_encoding(max_length, hidden_dim)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.position_encoding = self.position_encoding.to(device)
        
        self.final_layer_norm = nn.LayerNorm(hidden_dim)

    def _create_position_encoding(self, max_length, d_model):
        """
        Create sinusoidal positional encodings.
        
        Args:
            max_length: Maximum sequence length
            d_model: Dimensionality of the model
            
        Returns:
            Positional encoding matrix of shape [max_length, d_model]
        """
        position = torch.arange(max_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        pos_enc = torch.zeros(max_length, d_model)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        
        return pos_enc
    
    def _apply_position_encoding(self, x, batch):
        """
        Apply positional encoding to input features.
        
        Args:
            x: Input features
            batch: Batch assignment of nodes (for graphs)
            
        Returns:
            Features with positional encoding added
        """
        if batch is None:
            if x.dim() == 2:
                seq_len = min(x.size(0), self.max_length)
                x = x + self.position_encoding[:seq_len]
            else:
                batch_size, seq_len, _ = x.size()
                seq_len = min(seq_len, self.max_length)

                x_truncated = x.clone()
                x_truncated[:, :seq_len, :] = x[:, :seq_len, :] + self.position_encoding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
                x = x_truncated

        else:
            for graph_idx in batch.unique():
                mask = (graph_idx == batch)
                nodes_count = mask.sum().item()
                seq_len = min(nodes_count, self.max_length)
                indices = torch.where(mask)[0][:seq_len]
                x[indices] = x[indices] + self.position_encoding[:seq_len]
        return self.pos_dropout(x)
    
    def forward(self, data):
        """
        Forward pass through the model.
        
        Args:
            data: PyTorch Geometric data object containing graph information
            
        Returns:
            final_coords: Final predicted coordinates
            all_coords: List of coordinate predictions from each refinement iteration
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Embed node and edge features
        x = self.node_embedding(x)
        if edge_attr is not None:
            edge_attr = self.edge_embedding(edge_attr)

        # Process through GNN layers with residual connections
        layer_outputs = [x]
        for i, gcn_layer in enumerate(self.gcn_layers):
            identity = x
            if edge_attr is not None:
                x_new = gcn_layer(x, edge_index, edge_attr=edge_attr)
            else:
                x_new = gcn_layer(x, edge_index)
            x_new = F.dropout(x_new, p=0.1, training=self.training)
            x_new = x_new + identity
            x_new = F.gelu(x_new)
            x_new = self.layer_norm_gcn[i](x_new)

            # Add residual connection every 2 layers
            if i >= 2 and i % 2 == 0:
                x_new = x_new + layer_outputs[-2]

            x = x_new
            layer_outputs.append(x)

        # Final layer normalization
        x = self.final_layer_norm(x)
        
        # Handle batched graphs
        if batch is not None:
            max_nodes = torch.bincount(batch).max().item()
            batch_size = batch.max().item() + 1
            x_padded = torch.zeros(batch_size, max_nodes, self.hidden_dim, device=x.device)
            padding_mask = torch.ones(batch_size, max_nodes, dtype=torch.bool, device=x.device)

            # Create padded batch and mask
            for i in range(batch_size):
                mask = (batch == i)
                nodes_i = mask.sum().item()
                x_padded[i, :nodes_i] = x[mask]
                padding_mask[i, :nodes_i] = False
                
            # Apply positional encoding
            x_padded = self._apply_position_encoding(x_padded, None)
            
            # Initial coordinate prediction
            all_coords = []
            curr_embedding = self.transformer_encoder(x_padded, src_key_padding_mask=padding_mask)
            curr_coords = torch.zeros(batch_size, max_nodes, 3, device=x.device)
            
            for i in range(batch_size):
                mask = (batch == i)
                nodes_i = mask.sum().item()
                initial_coords = self.initial_coord_mlp(curr_embedding[i, :nodes_i])
                curr_coords[i, :nodes_i] = initial_coords
                
            all_coords.append(curr_coords)
            
            # Iterative refinement of coordinates
            for iter_idx in range(1, self.num_iterations):
                attention_weights = self.structure_aware_attention(curr_coords, padding_mask)
                curr_embeddings = self.transformer_encoder(
                    x_padded, 
                    attention_weights=attention_weights,
                    src_key_padding_mask=padding_mask
                )
                refined_coords = self.coord_refinement(curr_embeddings, curr_coords)
                curr_coords = refined_coords
                all_coords.append(curr_coords)
                
            # Extract final coordinates for each node
            final_coords = torch.zeros_like(x[:, :3])
            for i in range(batch_size):
                mask = (batch == i)
                nodes_i = mask.sum().item()
                final_coords[mask] = all_coords[-1][i, :nodes_i]
                
            return final_coords, all_coords
            
        # Handle single graph (non-batched)
        else:
            x = self._apply_position_encoding(x, None)
            x_batched = x.unsqueeze(0)

            all_coords = []

            # Initial coordinate prediction
            curr_embedding = self.transformer_encoder(x_batched)
            curr_embeddings = curr_embedding.squeeze(0)
            curr_coords = self.initial_coord_mlp(curr_embeddings)
            all_coords.append(curr_coords.unsqueeze(0))
            
            # Iterative refinement of coordinates
            for iter_idx in range(1, self.num_iterations):
                # Generate structure-aware attention weights from current coordinates
                attention_weights = self.structure_aware_attention(
                    curr_coords.unsqueeze(0)  # Add batch dimension
                )
                
                # Process through transformer with structure-aware attention
                curr_embeddings = self.transformer_encoder(
                    x_batched,
                    attention_weights=attention_weights
                ).squeeze(0)  # Remove batch dimension
                
                # Refine coordinates
                refined_coords = self.coord_refinement(
                    curr_embeddings, 
                    curr_coords
                )
                curr_coords = refined_coords
                all_coords.append(curr_coords.unsqueeze(0))
            
            return curr_coords, all_coords