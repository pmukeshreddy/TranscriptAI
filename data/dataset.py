import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .graph_builder import RNA3DGraphBuilder

class RNADataset(Dataset):
    """
    Dataset for RNA 3D structure data.
    Converts RNA sequences and their 3D coordinates into graph representations.
    """
    def __init__(self, df, graph_builder=None, sequence_col="sequence", coord_col="coordinates"):
        """
        Initialize the dataset.
        
        Args:
            df: Pandas DataFrame containing RNA data
            graph_builder: RNA3DGraphBuilder instance (created if None)
            sequence_col: Column name for sequences
            coord_col: Column name for coordinates
        """
        self.df = df
        self.graph_builder = graph_builder or RNA3DGraphBuilder()
        self.sequence_col = sequence_col
        self.coord_col = coord_col
        self.tokens = {nt: i for i, nt in enumerate("ACGU")}
        
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.df)
    
    def prepare_data(self, sequence, coordinates, max_len=300):
        """
        Prepare sequence and coordinate data.
        
        Args:
            sequence: RNA sequence as tensor of nucleotide indices
            coordinates: 3D coordinates as tensor
            max_len: Maximum sequence length to consider
            
        Returns:
            Processed sequence and normalized coordinates
        """
        if len(sequence) > max_len:
            sequence = sequence[:max_len]
            coordinates = coordinates[:max_len]
        
        # Convert numpy array to PyTorch tensor if it's not already
        if isinstance(coordinates, np.ndarray):
            coordinates = torch.tensor(coordinates, dtype=torch.float)
            
        # Normalize coordinates
        normalized_coordinates = (coordinates - coordinates.mean(dim=0)) / (coordinates.std(dim=0) + 1e-8)
        return sequence, normalized_coordinates
        
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            PyTorch Geometric Data object representing the RNA structure as a graph
        """
        row = self.df.iloc[idx]
        seq = row[self.sequence_col]
        seq_indices = [self.tokens.get(nt, 0) for nt in seq]
        seq_tensor = torch.tensor(seq_indices, dtype=torch.long) 
        
        coords = row[self.coord_col]
        # Make sure coords is a tensor before passing to prepare_data
        if not isinstance(coords, torch.Tensor):
            coords_tensor = torch.tensor(coords, dtype=torch.float)
        else:
            coords_tensor = coords
            
        seq_tensor, coords_tensor = self.prepare_data(seq_tensor, coords_tensor)
        graph = self.graph_builder.build_graph(seq_tensor, coords_tensor)
        return graph