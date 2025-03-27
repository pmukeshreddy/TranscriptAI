import torch
import torch.nn as nn

class StructureAwareattenation(nn.Module):
    """
    Structure-aware attention module that considers the 3D spatial distances between residues
    to create attention weights.
    """
    def __init__(self, hidden_state, num_heads, max_distance=20.0):
        super(StructureAwareattenation, self).__init__()
        self.hidden_state = hidden_state
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.distance_projection = nn.Sequential(
            nn.Linear(1, hidden_state//2),
            nn.GELU(),
            nn.Linear(hidden_state//2, 1)
        )
        
    def forward(self, coords, padding_mask=None):
        """
        Generate attention weights based on 3D distances between residues.
        
        Args:
            coords: Tensor of shape [batch_size, seq_len, 3] containing 3D coordinates
            padding_mask: Optional boolean mask indicating padded positions
            
        Returns:
            Tensor of shape [batch_size * num_heads, seq_len, seq_len] containing attention weights
        """
        batch_size, seq_len, _ = coords.shape
        
        # Compute pairwise distances
        coords_i = coords.unsqueeze(2)  # [batch, seq, 1, 3]
        coords_j = coords.unsqueeze(1)  # [batch, 1, seq, 3]
        distance = torch.sqrt(((coords_i - coords_j) ** 2).sum(dim=-1) + 1e-8)  # [batch, seq, seq]
        
        # Normalize and convert to attention weights
        distance = torch.clamp(distance / self.max_distance, 0, 1)
        distance = distance.unsqueeze(-1)  # [batch, seq, seq, 1]
        attention = self.distance_projection(distance)  # [batch, seq, seq, 1]
        attention = torch.exp(-3.0 * attention)  # [batch, seq, seq, 1]
        attention = attention.squeeze(-1)  # [batch, seq, seq]
        
        # Create a new tensor with the correct shape for multihead attention
        result = torch.zeros(batch_size * self.num_heads, seq_len, seq_len, device=coords.device)
        
        # Fill it with the attention weights
        for b in range(batch_size):
            for h in range(self.num_heads):
                idx = b * self.num_heads + h
                result[idx] = attention[b]
        
        # Apply padding mask if provided
        if padding_mask is not None:
            for b in range(batch_size):
                mask_b = padding_mask[b]  # [seq_len]
                
                # Create row and column masks
                row_mask = mask_b.unsqueeze(1).expand(seq_len, seq_len)  # [seq_len, seq_len]
                col_mask = mask_b.unsqueeze(0).expand(seq_len, seq_len)  # [seq_len, seq_len]
                combined = row_mask | col_mask  # [seq_len, seq_len]
                
                # Apply to all heads for this batch
                for h in range(self.num_heads):
                    idx = b * self.num_heads + h
                    result[idx].masked_fill_(combined, -1e9)
        
        return result