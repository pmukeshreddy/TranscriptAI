import torch
import torch.nn as nn

class CoordinateRefinementModule(nn.Module):
    """
    Module for refining 3D coordinates based on node embeddings and previous coordinate predictions.
    """
    def __init__(self, hidden_dim):
        super(CoordinateRefinementModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.refine_mlp = nn.Sequential(
            nn.Linear(hidden_dim+3, hidden_dim*2),
            nn.GELU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 3)
        )
        
    def forward(self, node_embeddings, previous_coords):
        """
        Refine the coordinates using node embeddings and previous coordinate predictions.
        
        Args:
            node_embeddings: Tensor of shape [batch_size, seq_len, hidden_dim] containing node features
            previous_coords: Tensor of shape [batch_size, seq_len, 3] containing previous coordinate predictions
            
        Returns:
            Tensor of shape [batch_size, seq_len, 3] containing refined coordinates
        """
        combined_features = torch.cat([node_embeddings, previous_coords], dim=-1)
        combined_residual = self.refine_mlp(combined_features)
        output = previous_coords + combined_residual
        return output