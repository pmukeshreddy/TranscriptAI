import torch
import torch.nn.functional as F
from torch_geometric.data import Data

class RNA3DGraphBuilder:
    """
    Converts RNA sequence and 3D coordinates into graph representation for GNN processing.
    """
    def __init__(self, distance_threshold=10.0, include_backbone=True, include_base_pairing=True):
        self.distance_threshold = distance_threshold
        self.include_backbone = include_backbone
        self.include_base_pairing = include_base_pairing
        self.edge_types = {
            "backbone": 0,
            "base_pair": 1,
            "spatial_proximity": 2
        }
        
    def build_graph(self, sequence, coordinates):
        """
        Build a graph from RNA sequence and 3D coordinates.
        
        Args:
            sequence: Tensor of nucleotide indices
            coordinates: Tensor of 3D coordinates with shape [seq_length, num_atoms, 3]
                       or [seq_length, 3]
        
        Returns:
            torch_geometric.data.Data object containing the graph representation
        """
        seq_len = len(sequence)
        
        # Handle empty coordinates case
        if coordinates.numel() == 0:
            x = F.one_hot(sequence, num_classes=4).float()
            pos_feature = torch.arange(seq_len, dtype=torch.float).view(-1, 1) / seq_len
            x = torch.cat([x, pos_feature], dim=1)
            
            # Create minimal graph with self-loops
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            edge_attr = torch.tensor([[0, 0, 0]], dtype=torch.float)
            centers = torch.zeros((seq_len, 3))  # Default positions
            
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=centers, num_nodes=seq_len)
            
        # One-hot encode nucleotide sequence
        x = F.one_hot(sequence, num_classes=4).float()
        
        # Get coordinate centers if we have per-atom coordinates
        if len(coordinates.shape) == 3:
            centers = coordinates.mean(dim=1)
        else:
            centers = coordinates
            
        edge_index = []
        edge_attr = []
        
        # Add backbone edges
        if self.include_backbone:
            for i in range(seq_len-1):
                edge_index.append([i, i+1])
                edge_index.append([i+1, i])
                edge_attr.append([1, 0, 0])
                edge_attr.append([1, 0, 0])
                
        # Add spatial proximity and base-pairing edges
        for i in range(seq_len-2):
            for j in range(i+2, min(seq_len, len(centers))):
                if j < seq_len:
                    distance = torch.norm(centers[i].float() - centers[j].float(), p=2)
                    if distance < self.distance_threshold:
                        edge_type = self._determine_edge_type(sequence[i], sequence[j])
                        edge_index.append([i, j])
                        edge_index.append([j, i])
                        if edge_type == "base_pair":
                            edge_attr.append([0, 1, 0])
                            edge_attr.append([0, 1, 0])
                        else:
                            edge_attr.append([0, 0, 1])
                            edge_attr.append([0, 0, 1])
        
        # Handle case with no edges
        if not edge_index:
            edge_index = [[0, 0]]
            edge_attr = [[0, 0, 0]]
            
        # Convert to PyTorch tensors
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        # Add positional feature to node features
        pos_feature = torch.arange(seq_len, dtype=torch.float).view(-1, 1) / seq_len
        x = torch.cat([x, pos_feature], dim=1)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=centers, num_nodes=seq_len)
    
    def _determine_edge_type(self, nt1, nt2):
        """
        Determine the type of edge between two nucleotides.
        
        Args:
            nt1: First nucleotide index
            nt2: Second nucleotide index
            
        Returns:
            Edge type as string: "base_pair" or "spatial_approximation"
        """
        if (nt1 == 0 and nt2 == 2) or (nt1 == 2 and nt2 == 0):  # A-U
            return "base_pair"
        elif (nt1 == 1 and nt2 == 3) or (nt1 == 3 and nt2 == 1):  # C-G
            return "base_pair"
        else:
            return "spatial_approximation"