import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def calculate_d0(L_ref):
    """
    Calculate the d0 distance scaling factor used in TM-score.
    
    Args:
        L_ref (int): Number of residues in reference structure
        
    Returns:
        float: d0 scaling factor
    """
    if L_ref >= 30:
        return 0.6 * ((L_ref - 0.5) ** 0.5) - 2.5
    elif L_ref >= 24:
        return 0.7  # For Lref 24-29
    elif L_ref >= 20:
        return 0.6  # For Lref 20-23
    elif L_ref >= 16:
        return 0.5  # For Lref 16-19
    elif L_ref >= 12:
        return 0.4  # For Lref 12-15
    else:
        return 0.3  # For Lref < 12

def calculate_tm_score(pred_coords, target_coords, device='cuda'):
    """
    Calculate the TM-score between predicted and target coordinates.
    
    Args:
        pred_coords: Predicted coordinates [N, 3]
        target_coords: Target coordinates [N, 3]
        device: Device for calculation
        
    Returns:
        float: TM-score (ranges from 0 to 1, higher is better)
    """
    # Center the structures at origin
    p_center = torch.mean(pred_coords, dim=0)
    t_center = torch.mean(target_coords, dim=0)
    
    p_centered = pred_coords - p_center
    t_centered = target_coords - t_center
    
    # Calculate covariance matrix for optimal rotation
    covar = torch.matmul(p_centered.T, t_centered)
    
    # Calculate d0 based on sequence length
    L_ref = target_coords.shape[0]
    d0 = calculate_d0(L_ref)
    d0_tensor = torch.tensor(d0, device=device)
    
    try:
        # SVD decomposition (differentiable in PyTorch)
        U, _, V = torch.linalg.svd(covar)
        
        # Calculate rotation matrix
        rotation = torch.matmul(V, U.T)
        
        # Check if we need to flip the last row (ensure right-handedness)
        det = torch.det(rotation)
        if det < 0:
            V_adjusted = V.clone()
            V_adjusted[-1] = -V_adjusted[-1]
            rotation = torch.matmul(V_adjusted, U.T)
        
        # Apply rotation to align predicted structure
        p_rotated = torch.matmul(p_centered, rotation)
        
        # Calculate distances between corresponding atoms
        distances = torch.sqrt(torch.sum((p_rotated - t_centered)**2, dim=1) + 1e-8)
        
        # Calculate TM-score sum
        tm_sum = torch.sum(1.0 / (1.0 + (distances / d0_tensor)**2))
        
        # Normalize by length
        tm_score = tm_sum / L_ref
        
    except Exception as e:
        print(f"SVD did not converge: {e}")
        # Fall back to simpler calculation without rotation
        distances = torch.sqrt(torch.sum((pred_coords - target_coords)**2, dim=1) + 1e-8)
        tm_sum = torch.sum(1.0 / (1.0 + (distances / d0_tensor)**2))
        tm_score = tm_sum / L_ref
    
    return tm_score.item()

def get_per_length_tm_scores(tm_scores, lengths):
    """
    Analyze TM-scores by sequence length groups.
    
    Args:
        tm_scores: List of TM-scores
        lengths: List of sequence lengths
        
    Returns:
        dict: TM-scores grouped by length ranges
    """
    # Define length bins
    length_bins = {
        "0-50": [],
        "51-100": [],
        "101-200": [],
        "201-300": [],
        "301+": []
    }
    
    # Assign TM-scores to bins
    for tm, length in zip(tm_scores, lengths):
        if length <= 50:
            length_bins["0-50"].append(tm)
        elif length <= 100:
            length_bins["51-100"].append(tm)
        elif length <= 200:
            length_bins["101-200"].append(tm)
        elif length <= 300:
            length_bins["201-300"].append(tm)
        else:
            length_bins["301+"].append(tm)
    
    # Calculate statistics for each bin
    results = {}
    for bin_name, scores in length_bins.items():
        if scores:
            results[bin_name] = {
                "count": len(scores),
                "mean": np.mean(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores)
            }
        else:
            results[bin_name] = {
                "count": 0,
                "mean": 0,
                "std": 0,
                "min": 0,
                "max": 0
            }
    
    return results

def visualize_tm_score_results(overall_tm, length_results, output_path="tm_score_results.png"):
    """
    Generate visualization of TM-score validation results.
    
    Args:
        overall_tm: Overall mean TM-score
        length_results: Results by length bins
        output_path: Path to save the visualization
    """
    plt.figure(figsize=(12, 8))
    
    # Plot overall TM-score
    plt.subplot(2, 2, 1)
    plt.text(0.5, 0.5, f"Overall TM-score: {overall_tm:.4f}", 
             horizontalalignment='center', verticalalignment='center', fontsize=14)
    plt.axis('off')
    
    # Plot TM-scores by length
    plt.subplot(2, 2, 2)
    bins = []
    means = []
    errors = []
    counts = []
    
    for bin_name, stats in length_results.items():
        if stats["count"] > 0:
            bins.append(bin_name)
            means.append(stats["mean"])
            errors.append(stats["std"])
            counts.append(stats["count"])
    
    x = range(len(bins))
    plt.bar(x, means, yerr=errors, capsize=5)
    plt.xticks(x, bins, rotation=45)
    plt.ylabel("Mean TM-score")
    plt.title("TM-score by Sequence Length")
    
    # Plot counts by length bin
    plt.subplot(2, 2, 3)
    plt.bar(x, counts)
    plt.xticks(x, bins, rotation=45)
    plt.ylabel("Count")
    plt.title("Sample Count by Length Bin")
    
    # Plot min/max by length bin
    plt.subplot(2, 2, 4)
    mins = [stats["min"] for bin_name, stats in length_results.items() if stats["count"] > 0]
    maxs = [stats["max"] for bin_name, stats in length_results.items() if stats["count"] > 0]
    
    plt.plot(x, mins, 'ro-', label='Min')
    plt.plot(x, maxs, 'go-', label='Max')
    plt.plot(x, means, 'bo--', label='Mean')
    plt.xticks(x, bins, rotation=45)
    plt.ylabel("TM-score")
    plt.title("Min/Max TM-scores by Length")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Results visualization saved to {output_path}")
    plt.close()

def validate_rna_model_tm_score(model, data_loader, device, model_name="RNA_Model", 
                              save_dir="validation_results", visualize=True):
    """
    Validate an RNA 3D structure prediction model using TM-score.
    
    Args:
        model: The RNA structure prediction model
        data_loader: DataLoader containing validation data
        device: Device to run validation on
        model_name: Name of the model for saving results
        save_dir: Directory to save validation results
        visualize: Whether to generate visualization
        
    Returns:
        dict: Validation results
    """
    model.eval()
    all_tm_scores = []
    all_lengths = []
    
    # Create results directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Validating TM-score")
        
        for batch in progress_bar:
            batch = batch.to(device)
            
            # Get model predictions
            final_coords, all_coords = model(batch)
            
            # Use the final refined coordinates
            final_prediction = all_coords[-1]
            
            # Calculate TM-score for each graph in the batch
            batch_tm_scores = []
            
            if batch.batch is not None:
                # Handle batched graphs
                batch_size = batch.batch.max().item() + 1
                
                for j in range(batch_size):
                    mask = (batch.batch == j)
                    nodes_j = mask.sum().item()
                    
                    # Extract predicted coordinates for this graph
                    if len(final_prediction.shape) == 2 and j == 0:  
                        # Single graph case
                        pred_coords = final_prediction[:nodes_j]
                    else:  
                        # Batch case
                        pred_coords = final_prediction[j, :nodes_j]
                    
                    # Get target coordinates
                    target_coords = batch.pos[mask]
                    
                    # Calculate TM-score
                    tm_score = calculate_tm_score(pred_coords, target_coords, device)
                    batch_tm_scores.append(tm_score)
                    all_tm_scores.append(tm_score)
                    all_lengths.append(nodes_j)
                
                # Update progress bar with average TM-score for this batch
                if batch_tm_scores:
                    avg_batch_tm = sum(batch_tm_scores) / len(batch_tm_scores)
                    progress_bar.set_postfix({"TM-score": f"{avg_batch_tm:.4f}"})
                
            else:
                # Handle single graph
                pred_coords = final_prediction.squeeze(0) if final_prediction.dim() > 2 else final_prediction
                target_coords = batch.pos
                
                tm_score = calculate_tm_score(pred_coords, target_coords, device)
                all_tm_scores.append(tm_score)
                all_lengths.append(target_coords.shape[0])
                
                progress_bar.set_postfix({"TM-score": f"{tm_score:.4f}"})
    
    # Calculate overall results
    mean_tm_score = np.mean(all_tm_scores)
    std_tm_score = np.std(all_tm_scores)
    median_tm_score = np.median(all_tm_scores)
    min_tm_score = np.min(all_tm_scores)
    max_tm_score = np.max(all_tm_scores)
    
    # Calculate length-specific results
    length_results = get_per_length_tm_scores(all_tm_scores, all_lengths)
    
    # Save numerical results
    results = {
        "overall": {
            "mean_tm_score": mean_tm_score,
            "std_tm_score": std_tm_score,
            "median_tm_score": median_tm_score,
            "min_tm_score": min_tm_score,
            "max_tm_score": max_tm_score,
            "sample_count": len(all_tm_scores)
        },
        "by_length": length_results
    }
    
    # Save results as text file
    results_path = os.path.join(save_dir, f"{model_name}_tm_score_results.txt")
    with open(results_path, 'w') as f:
        f.write(f"===== TM-score Validation Results for {model_name} =====\n\n")
        f.write(f"Total samples: {len(all_tm_scores)}\n")
        f.write(f"Mean TM-score: {mean_tm_score:.4f}\n")
        f.write(f"Std Dev: {std_tm_score:.4f}\n")
        f.write(f"Median: {median_tm_score:.4f}\n")
        f.write(f"Min: {min_tm_score:.4f}\n")
        f.write(f"Max: {max_tm_score:.4f}\n\n")
        
        f.write("===== Results by Sequence Length =====\n")
        for bin_name, stats in length_results.items():
            f.write(f"{bin_name} ({stats['count']} samples):\n")
            f.write(f"  Mean: {stats['mean']:.4f}\n")
            f.write(f"  Std Dev: {stats['std']:.4f}\n")
            f.write(f"  Min: {stats['min']:.4f}\n")
            f.write(f"  Max: {stats['max']:.4f}\n\n")
    
    print(f"Results saved to {results_path}")
    
    # Create visualization
    if visualize:
        vis_path = os.path.join(save_dir, f"{model_name}_tm_score_vis.png")
        visualize_tm_score_results(mean_tm_score, length_results, vis_path)
    
    return results