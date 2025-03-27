import pandas as pd
import torch
import numpy as np

def load_rna_data(sequences_path, labels_path):
    """
    Load RNA sequence and label data from CSV files.
    
    Args:
        sequences_path: Path to sequences CSV file
        labels_path: Path to labels CSV file
        
    Returns:
        DataFrame containing combined RNA data
    """
    # Load sequences and labels
    sequences_df = pd.read_csv(sequences_path)
    sequences_df = sequences_df.dropna()
    
    labels_df = pd.read_csv(labels_path)
    labels_df = labels_df.dropna()
    
    # Extract main ID for joining
    labels_df["main_id"] = labels_df["ID"].apply(lambda x: x.split("_")[0])
    
    # Get coordinates for each sequence
    xyz_coordinates = []
    for m_id in sequences_df["target_id"]:
        df = labels_df[labels_df["main_id"] == m_id]
        xyz = df[["x_1", "y_1", "z_1"]].to_numpy().astype("float32")
        xyz_coordinates.append(xyz)
    
    # Create combined dataframe
    data = {
        "sequnce": sequences_df["sequence"].to_list(),
        "temporal_difference": sequences_df["temporal_cutoff"].to_list(),
        "description": sequences_df["description"].to_list(),
        "all_sequences": sequences_df["all_sequences"].to_list(),
        "coordinates": xyz_coordinates
    }
    
    df = pd.DataFrame(data)
    return df

def clean_rna_data(df):
    """
    Clean RNA data by removing invalid sequences and empty coordinates.
    
    Args:
        df: DataFrame containing RNA data
        
    Returns:
        Cleaned DataFrame
    """
    # Check for characters other than A, C, U, G in each sequence
    df['contains_invalid'] = df['sequnce'].apply(check_invalid_characters)
    
    # Remove sequences with invalid characters
    df = df[~df['contains_invalid']].drop(columns=['contains_invalid'])
    
    # Check for empty coordinates
    df['has_empty_coordinates'] = df['coordinates'].apply(is_empty_coordinates)
    
    # Remove rows with empty coordinates
    df = df[~df['has_empty_coordinates']].copy()
    df.drop(columns=['has_empty_coordinates'], inplace=True)
    
    return df

def check_invalid_characters(sequence):
    """
    Check if a sequence contains characters other than A, C, U, G.
    
    Args:
        sequence: RNA sequence string
        
    Returns:
        True if sequence contains invalid characters, False otherwise
    """
    valid_characters = set("ACUG")
    return any(char not in valid_characters for char in sequence)

def is_empty_coordinates(coordinates):
    """
    Check if coordinates are empty or None.
    
    Args:
        coordinates: Coordinate array or list
        
    Returns:
        True if coordinates are empty, False otherwise
    """
    # Check if coordinates is None or empty list
    if coordinates is None or len(coordinates) == 0:
        return True
    # Check if any coordinate in the list is None or empty
    return any(coord is None or len(coord) == 0 for coord in coordinates)

def split_by_date(df, train_cutoff_date, test_cutoff_date):
    """
    Split data by temporal cutoff date.
    
    Args:
        df: DataFrame containing RNA data
        train_cutoff_date: Cutoff date for training data (e.g., "2020-01-01")
        test_cutoff_date: Cutoff date for test data (e.g., "2022-05-01")
        
    Returns:
        train_df: Training DataFrame
        test_df: Testing DataFrame
    """
    train_cutoff = pd.Timestamp(train_cutoff_date)
    test_cutoff = pd.Timestamp(test_cutoff_date)
    
    train_indices = [i for i, d in enumerate(df["temporal_difference"]) 
                    if pd.Timestamp(d) <= train_cutoff]
    
    test_indices = [i for i, d in enumerate(df["temporal_difference"]) 
                   if pd.Timestamp(d) > train_cutoff and pd.Timestamp(d) <= test_cutoff]
    
    train_df = df.iloc[train_indices].copy()
    test_df = df.iloc[test_indices].copy()
    
    return train_df, test_df

def fix_sequence_coordinate_mismatch(df, sequence_col='sequnce', coord_col='coordinates', strategy='trim'):
    """
    Fix mismatches between sequence length and coordinate count.
    
    Args:
        df: DataFrame containing sequence and coordinate data
        sequence_col: Column name for sequences
        coord_col: Column name for coordinates
        strategy: Strategy to fix mismatches ('pad', 'trim', or 'drop')
        
    Returns:
        DataFrame with fixed mismatches
    """
    fixed_df = df.copy()
    mismatches = []
    
    for idx, row in fixed_df.iterrows():
        sequence = row[sequence_col]
        coords = row[coord_col]
        
        seq_length = len(sequence)
        coord_count = coords.shape[0]
        
        if seq_length != coord_count:
            if strategy == 'drop':
                fixed_df.drop(idx, inplace=True)
                mismatches.append({'index': idx, 'action': 'dropped row'})
            
            elif strategy == 'trim':
                # Trim to the shorter length
                min_length = min(seq_length, coord_count)
                fixed_df.at[idx, sequence_col] = sequence[:min_length]
                fixed_df.at[idx, coord_col] = coords[:min_length]
                mismatches.append({'index': idx, 'action': 'trimmed to length ' + str(min_length)})
            
            elif strategy == 'pad':
                if seq_length > coord_count:
                    # Pad coordinates to match sequence length
                    # Check if coords is a torch tensor or numpy array
                    if isinstance(coords, torch.Tensor):
                        padding = torch.zeros((seq_length - coord_count, coords.shape[1]), 
                                             dtype=coords.dtype, device=coords.device)
                        fixed_df.at[idx, coord_col] = torch.cat([coords, padding], dim=0)
                    else:
                        # For numpy arrays
                        padding = np.zeros((seq_length - coord_count, coords.shape[1]), 
                                          dtype=coords.dtype)
                        fixed_df.at[idx, coord_col] = np.concatenate([coords, padding], axis=0)
                    
                    mismatches.append({'index': idx, 'action': 'padded coordinates'})
                else:
                    # Pad sequence to match coordinate count
                    fixed_df.at[idx, sequence_col] = sequence + 'N' * (coord_count - seq_length)
                    mismatches.append({'index': idx, 'action': 'padded sequence'})
    
    return fixed_df