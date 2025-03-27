import torch
import torch.nn as nn
import torch.nn.functional as F

class StructureAwareTransformerLayer(nn.Module):
    """
    A single transformer layer that can incorporate structure-aware attention.
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super(StructureAwareTransformerLayer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.activation = F.gelu
        
    def forward(self, src, attention_weight=None, src_key_padding_mask=None):
        """
        Process inputs through the transformer layer.
        
        Args:
            src: Input tensor of shape [batch_size, seq_len, d_model]
            attention_weight: Optional pre-computed attention weights
            src_key_padding_mask: Optional padding mask for source sequence
            
        Returns:
            Output tensor with the same shape as src
        """
        src2 = self.norm1(src)
        if attention_weight is not None:
            q = k = v = src2
            src2 = self.self_attn(q, k, v, attn_mask=attention_weight, 
                                 key_padding_mask=src_key_padding_mask)[0]
        else:
            src2 = self.self_attn(src2, src2, src2, key_padding_mask=src_key_padding_mask)[0]
        src = src + src2
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + src2
        return src


class StructureAwareTransformerEncoder(nn.Module):
    """
    Stack of transformer encoder layers that can incorporate structure-aware attention.
    """
    def __init__(self, encoder_layer, num_layers):
        super(StructureAwareTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        
    def forward(self, src, attention_weights=None, src_key_padding_mask=None):
        """
        Process inputs through the transformer encoder stack.
        
        Args:
            src: Input tensor of shape [batch_size, seq_len, d_model]
            attention_weights: Optional pre-computed attention weights
            src_key_padding_mask: Optional padding mask for source sequence
            
        Returns:
            Output tensor with the same shape as src after passing through all layers
        """
        output = src
        for layer in self.layers:
            output = layer(output, attention_weights, src_key_padding_mask)
        return output