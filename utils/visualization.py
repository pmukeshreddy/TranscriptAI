import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import io
import base64
import torch
import networkx as nx

def visualize_rna_graph(graph, sequence=None):
    """
    Visualize an RNA graph in 2D and 3D
    
    Args:
        graph: PyG Data object representing the RNA graph
        sequence: Original RNA sequence (optional)
    """
    # 1. Create a NetworkX graph for 2D visualization
    G = nx.Graph()
    
    # Add nodes
    for i in range(graph.num_nodes):
        label = sequence[i] if sequence and i < len(sequence) else str(i)
        G.add_node(i, label=label)
    
    # Add edges with their types
    edge_index = graph.edge_index.t().numpy()
    edge_attr = graph.edge_attr.numpy()
    
    edge_colors = []
    for i, (src, tgt) in enumerate(edge_index):
        if src <= tgt:  # Add each edge only once
            edge_type = edge_attr[i]
            if edge_type[0] == 1:
                edge_color = 'green'  # Backbone
                edge_label = 'backbone'
            elif edge_type[1] == 1:
                edge_color = 'red'  # Base pair
                edge_label = 'base_pair'
            else:
                edge_color = 'blue'  # Spatial proximity
                edge_label = 'spatial'
                
            G.add_edge(src, tgt, color=edge_color, label=edge_label)
            edge_colors.append(edge_color)
    
    # Create figure with two subplots: 2D and 3D
    fig = plt.figure(figsize=(15, 6), dpi=100)
    
    # 2D Graph visualization
    ax1 = fig.add_subplot(121)
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue', edgecolors='black')
    
    # Draw edges with colors
    for edge, color in zip(G.edges(), edge_colors):
        nx.draw_networkx_edges(G, pos, edgelist=[edge], width=2, edge_color=color)
    
    # Add labels
    labels = {node: data['label'] for node, data in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels, font_size=12)
    
    ax1.set_title("2D RNA Graph Visualization", fontsize=14, weight='bold')
    ax1.axis('off')
    
    # 3D visualization if coordinates are available
    if hasattr(graph, 'pos') and graph.pos is not None:
        ax2 = fig.add_subplot(122, projection='3d')
        pos_3d = graph.pos.numpy()
        
        # Plot nodes
        ax2.scatter(pos_3d[:, 0], pos_3d[:, 1], pos_3d[:, 2], 
                   s=100, c='lightblue', edgecolors='black', alpha=0.8)
        
        # Plot edges
        for i, (src, tgt) in enumerate(edge_index):
            if src <= tgt:  # Plot each edge only once
                edge_type = edge_attr[i]
                if edge_type[0] == 1:
                    color = 'green'  # Backbone
                elif edge_type[1] == 1:
                    color = 'red'  # Base pair
                else:
                    color = 'blue'  # Spatial proximity
                    
                ax2.plot([pos_3d[src, 0], pos_3d[tgt, 0]], 
                         [pos_3d[src, 1], pos_3d[tgt, 1]], 
                         [pos_3d[src, 2], pos_3d[tgt, 2]], color=color, linewidth=2)
        
        # Add node labels
        for i, (x, y, z) in enumerate(pos_3d):
            label = sequence[i] if sequence and i < len(sequence) else str(i)
            ax2.text(x + 0.1, y + 0.1, z + 0.1, label, fontsize=10)
            
        ax2.set_title("3D RNA Structure", fontsize=14, weight='bold')
        ax2.set_xlabel("X", fontsize=12)
        ax2.set_ylabel("Y", fontsize=12)
        ax2.set_zlabel("Z", fontsize=12)
        
        # Add grid for better spatial perception
        ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig

def visualize_rna_structure(coords, sequence=None, title="RNA 3D Structure", 
                            backbone_only=False, highlight_pairs=True):
    """
    Create an enhanced visualization of RNA 3D structure with improved clarity.
    
    Args:
        coords: Tensor of shape [N, 3] containing 3D coordinates
        sequence: Optional RNA sequence string for labeling nucleotides
        title: Title for the visualization
        backbone_only: If True, only show backbone connections
        highlight_pairs: If True, try to identify and highlight base pairs
        
    Returns:
        Matplotlib figure object
    """
    # Convert tensor to numpy if needed
    if isinstance(coords, torch.Tensor):
        coords_np = coords.detach().cpu().numpy()
    else:
        coords_np = coords
    
    # Create figure with proper size and DPI for clear visualization
    fig = plt.figure(figsize=(10, 8), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a better color map for nucleotides if sequence is provided
    if sequence:
        # Assign colors to different nucleotides
        nuc_colors = {
            'A': '#FF5733',  # Red-orange for Adenine
            'C': '#33A8FF',  # Blue for Cytosine
            'G': '#33FF57',  # Green for Guanine
            'U': '#F3FF33'   # Yellow for Uracil
        }
        
        # Create color array for nodes
        node_colors = []
        for i, nuc in enumerate(sequence):
            if i < len(coords_np):  # Ensure we don't go out of bounds
                node_colors.append(nuc_colors.get(nuc, 'lightgrey'))
    else:
        # Default color if no sequence provided
        node_colors = ['lightblue'] * len(coords_np)
    
    # Plot nodes with nucleotide-specific colors and improved visibility
    scatter = ax.scatter(
        coords_np[:, 0], 
        coords_np[:, 1], 
        coords_np[:, 2],
        s=80,  # Larger point size
        c=node_colors,
        edgecolors='black',
        alpha=0.8,
        zorder=3  # Ensure nodes are on top
    )
    
    # Plot backbone connections
    for i in range(len(coords_np) - 1):
        ax.plot(
            [coords_np[i, 0], coords_np[i+1, 0]],
            [coords_np[i, 1], coords_np[i+1, 1]],
            [coords_np[i, 2], coords_np[i+1, 2]],
            color='forestgreen',
            linewidth=2.5,
            alpha=0.8,
            zorder=2
        )
    
    # Try to identify and highlight potential base pairs if requested
    if highlight_pairs and sequence and len(sequence) > 4 and not backbone_only:
        # Find complementary base pairs
        pairs = []
        complementary = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
        
        # Look for complementary bases within a reasonable distance
        for i in range(len(sequence)):
            for j in range(i + 3, len(sequence)):  # Skip very close neighbors
                if i < len(coords_np) and j < len(coords_np) and sequence[i] in complementary:
                    # Check if bases are complementary
                    if sequence[j] == complementary[sequence[i]]:
                        # Calculate distance
                        dist = np.linalg.norm(coords_np[i] - coords_np[j])
                        # Use distance threshold based on realistic base pair distances
                        if dist < 12.0:  # Typical base pair distance in RNA
                            pairs.append((i, j))
        
        # Plot base pairs with dotted lines
        for i, j in pairs:
            ax.plot(
                [coords_np[i, 0], coords_np[j, 0]],
                [coords_np[i, 1], coords_np[j, 1]],
                [coords_np[i, 2], coords_np[j, 2]],
                color='red',
                linewidth=1.5,
                linestyle='--',
                alpha=0.7,
                zorder=1
            )
    
    # Add nucleotide labels if sequence is provided and structure is not too large
    if sequence and len(sequence) <= 50 and not backbone_only:  # Only label if not too crowded
        for i, (x, y, z) in enumerate(coords_np):
            if i < len(sequence):
                # Offset the label slightly from the point for better visibility
                ax.text(
                    x + 0.1, 
                    y + 0.1, 
                    z + 0.1, 
                    sequence[i],
                    fontsize=10,
                    weight='bold',
                    backgroundcolor='white',
                    alpha=0.7
                )
    
    # Improve visual appearance
    ax.set_title(title, fontsize=14, weight='bold')
    ax.set_xlabel('X', fontsize=12, labelpad=10)
    ax.set_ylabel('Y', fontsize=12, labelpad=10)
    ax.set_zlabel('Z', fontsize=12, labelpad=10)
    
    # Add subtle grid for better spatial perception
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set equal aspect ratio for better 3D visualization
    max_range = max([
        np.ptp(coords_np[:, 0]),
        np.ptp(coords_np[:, 1]),
        np.ptp(coords_np[:, 2])
    ])
    mid_x = np.mean(coords_np[:, 0])
    mid_y = np.mean(coords_np[:, 1])
    mid_z = np.mean(coords_np[:, 2])
    
    # Set limits with some padding
    padding = max_range * 0.2  # 20% padding for better visibility
    ax.set_xlim(mid_x - max_range/2 - padding, mid_x + max_range/2 + padding)
    ax.set_ylim(mid_y - max_range/2 - padding, mid_y + max_range/2 + padding)
    ax.set_zlim(mid_z - max_range/2 - padding, mid_z + max_range/2 + padding)
    
    # Add legend if sequence is provided
    if sequence:
        # Create legend elements
        legend_elements = []
        
        # Nucleotide legend
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                               markerfacecolor=nuc_colors['A'], markersize=10, label='A - Adenine'))
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                               markerfacecolor=nuc_colors['C'], markersize=10, label='C - Cytosine'))
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                               markerfacecolor=nuc_colors['G'], markersize=10, label='G - Guanine'))
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                               markerfacecolor=nuc_colors['U'], markersize=10, label='U - Uracil'))
        
        # Backbone legend
        legend_elements.append(plt.Line2D([0], [0], color='forestgreen', lw=2.5, label='Backbone'))
        
        # Base pairs legend if enabled
        if highlight_pairs and not backbone_only:
            legend_elements.append(plt.Line2D([0], [0], color='red', lw=1.5, 
                                   linestyle='--', label='Base Pairs'))
        
        # Add legend with smaller font to save space
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.7)
    
    # Set optimal viewing angle for better structure visibility
    ax.view_init(elev=20, azim=135)
    
    # Use a white background with a subtle frame
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('lightgrey')
    
    plt.tight_layout()
    return fig

def visualize_refinement_process(all_coords, sequence=None):
    """
    Create an improved visualization showing the refinement process across iterations.
    
    Args:
        all_coords: List of coordinate tensors from each refinement iteration
        sequence: Optional RNA sequence for nucleotide labeling
        
    Returns:
        Matplotlib figure object
    """
    # Determine number of iterations and create appropriate figure size
    num_iterations = len(all_coords)
    fig = plt.figure(figsize=(4 * num_iterations, 5), dpi=100)
    
    # Create a better colormap for iterations
    # Using a sequential colormap from light to dark
    cmap = plt.cm.Blues
    colors = [cmap(0.3 + 0.7 * i/(num_iterations-1) if num_iterations > 1 else 0.7) for i in range(num_iterations)]
    
    # Process each iteration
    all_coords_np = []
    for i in range(num_iterations):
        # Convert coordinates to numpy
        if isinstance(all_coords[i], torch.Tensor):
            coords_np = all_coords[i].detach().cpu().numpy()
        else:
            coords_np = all_coords[i]
            
        # Handle batched outputs
        if len(coords_np.shape) > 2:
            if coords_np.shape[0] == 1:  # Single graph in batch
                coords_np = coords_np[0]
            else:
                # Take first graph in batch
                coords_np = coords_np[0]
        
        all_coords_np.append(coords_np)
    
    # Calculate global limits for consistent scaling across all subplots
    all_coords_flat = np.vstack(all_coords_np)
    global_min = np.min(all_coords_flat, axis=0)
    global_max = np.max(all_coords_flat, axis=0)
    global_mid = (global_min + global_max) / 2
    global_range = np.max(global_max - global_min)
    padding = global_range * 0.2  # 20% padding
    
    # Plot each iteration
    for i in range(num_iterations):
        ax = fig.add_subplot(1, num_iterations, i+1, projection='3d')
        coords_np = all_coords_np[i]
        
        # Plot nodes with iteration-specific color
        ax.scatter(
            coords_np[:, 0], 
            coords_np[:, 1], 
            coords_np[:, 2],
            s=50,  # Smaller points for the refinement view
            c=[colors[i]] * len(coords_np),
            edgecolors='black',
            alpha=0.8,
            zorder=3
        )
        
        # Plot backbone with consistent color
        for j in range(len(coords_np) - 1):
            ax.plot(
                [coords_np[j, 0], coords_np[j+1, 0]],
                [coords_np[j, 1], coords_np[j+1, 1]],
                [coords_np[j, 2], coords_np[j+1, 2]],
                color='green',
                linewidth=1.5,
                alpha=0.7,
                zorder=2
            )
        
        # Add nucleotide labels if sequence is provided and not too many nodes
        if sequence and len(sequence) <= 30 and i == num_iterations - 1:  # Only in final iteration
            for j, (x, y, z) in enumerate(coords_np):
                if j < len(sequence):
                    ax.text(
                        x + 0.1, 
                        y + 0.1, 
                        z + 0.1, 
                        sequence[j],
                        fontsize=8,
                        alpha=0.7
                    )
        
        # Set titles and labels
        if i == 0:
            title = "Initial Prediction"
        elif i == num_iterations - 1:
            title = "Final Structure"
        else:
            title = f"Refinement Step {i}"
        
        ax.set_title(title, fontsize=12, weight='bold')
        
        # Use smaller font for axis labels in the refinement view
        ax.set_xlabel('X', fontsize=9, labelpad=5)
        ax.set_ylabel('Y', fontsize=9, labelpad=5)
        ax.set_zlabel('Z', fontsize=9, labelpad=5)
        
        # Set consistent limits across all subplots
        ax.set_xlim(global_mid[0] - global_range/2 - padding, global_mid[0] + global_range/2 + padding)
        ax.set_ylim(global_mid[1] - global_range/2 - padding, global_mid[1] + global_range/2 + padding)
        ax.set_zlim(global_mid[2] - global_range/2 - padding, global_mid[2] + global_range/2 + padding)
        
        # Use consistent viewing angle
        ax.view_init(elev=20, azim=135)
        
        # Add subtle grid
        ax.grid(True, alpha=0.2, linestyle=':')
        
        # White background with subtle frame
        ax.set_facecolor('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('lightgrey')
    
    # Add a common title
    plt.suptitle("RNA Structure Refinement Process", fontsize=14, weight='bold', y=0.98)
    
    plt.tight_layout()
    return fig

def find_potential_base_pairs(sequence, coords, distance_threshold=8.0):
    """
    Find potential base pairs based on complementary bases and distance.
    
    Args:
        sequence: RNA sequence string
        coords: Coordinates array of shape [N, 3]
        distance_threshold: Maximum distance to consider for base pairs
        
    Returns:
        List of (i, j) index pairs for potential base pairs
    """
    # Define complementary base pairs
    complements = {
        'A': 'U',
        'U': 'A',
        'G': 'C',
        'C': 'G'
    }
    
    pairs = []
    
    # Look for potential base pairs (skip adjacent nucleotides)
    for i in range(len(sequence)):
        for j in range(i + 3, len(sequence)):  # Skip very close neighbors
            # Check if bases are complementary
            if sequence[i] in complements and sequence[j] == complements[sequence[i]]:
                # Check distance between nucleotides
                if i < len(coords) and j < len(coords):
                    dist = np.linalg.norm(coords[i] - coords[j])
                    if dist < distance_threshold:
                        pairs.append((i, j))
    
    return pairs

def create_graph_visualization(sequence, coords, edges=None):
    """
    Create a 2D graph visualization of the RNA structure.
    
    Args:
        sequence: RNA sequence string
        coords: 3D coordinates
        edges: Optional list of (i, j) edges to include
        
    Returns:
        Matplotlib figure
    """
    # Convert coordinates to numpy if needed
    if isinstance(coords, torch.Tensor):
        coords_np = coords.detach().cpu().numpy()
    else:
        coords_np = coords
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add nodes
    for i in range(len(sequence)):
        if i < len(coords_np):
            G.add_node(i, label=sequence[i])
    
    # Add backbone edges
    for i in range(len(sequence) - 1):
        if i < len(coords_np) and i + 1 < len(coords_np):
            G.add_edge(i, i + 1, color='green', weight=2)
    
    # Add additional edges if provided
    if edges:
        for i, j in edges:
            if i < len(coords_np) and j < len(coords_np):
                G.add_edge(i, j, color='red', weight=1)
    
    # Create figure
    fig = plt.figure(figsize=(10, 8), dpi=100)
    ax = fig.add_subplot(111)
    
    # Set positions using a spring layout
    pos = nx.spring_layout(G, seed=42)
    
    # Get edge colors and weights
    edge_colors = [G[u][v]['color'] for u, v in G.edges()]
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    
    # Draw the graph
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_weights)
    
    # Draw nodes with nucleotide-specific colors
    node_colors = []
    for node in G.nodes():
        nuc = G.nodes[node]['label']
        if nuc == 'A':
            node_colors.append('#FF5733')
        elif nuc == 'C':
            node_colors.append('#33A8FF')
        elif nuc == 'G':
            node_colors.append('#33FF57')
        elif nuc == 'U':
            node_colors.append('#F3FF33')
        else:
            node_colors.append('lightgrey')
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.8,
                          edgecolors='black')
    
    # Draw labels
    labels = {node: G.nodes[node]['label'] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')
    
    # Add title and remove axes
    plt.title("RNA Structure Graph Representation", fontsize=14, weight='bold')
    plt.axis('off')
    
    return fig

def fig_to_base64(fig):
    """
    Convert a matplotlib figure to base64 encoded string.
    
    Args:
        fig: Matplotlib figure object
        
    Returns:
        Base64 encoded string of the figure
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

def validate_rna_dataset(dataset, df, num_samples=3):
    """
    Validate an RNADataset by examining several samples from it.
    
    Args:
        dataset: The RNADataset instance
        df: Original DataFrame used to create the dataset
        num_samples: Number of samples to check (default: 3)
    """
    print(f"Dataset contains {len(dataset)} samples")
    
    # Check a few random samples
    indices = torch.randperm(len(dataset))[:min(num_samples, len(dataset))]
    
    for i, idx in enumerate(indices):
        print(f"\n{'='*50}")
        print(f"Sample {i+1} (index {idx.item()}):")
        
        # Get the graph from dataset
        graph = dataset[idx.item()]  # Convert PyTorch tensor to integer
        
        # Get original data from DataFrame
        row = df.iloc[idx.item()]  # Convert PyTorch tensor to integer
        sequence = row[dataset.sequence_col]
        
        print(f"Original sequence: {sequence}")
        print(f"Sequence length: {len(sequence)}")
        
        # Basic graph validation
        print(f"\nGraph properties:")
        print(f"- Number of nodes: {graph.num_nodes}")
        print(f"- Node features shape: {graph.x.shape}")
        print(f"- Number of edges: {graph.edge_index.shape[1]}")
        print(f"- Edge attributes shape: {graph.edge_attr.shape}")
        
        # Check if node count matches sequence length
        if graph.num_nodes != len(sequence):
            print(f"WARNING: Number of nodes ({graph.num_nodes}) doesn't match sequence length ({len(sequence)})")
        
        # Check edge types
        if graph.edge_attr.shape[0] > 0:
            backbone_edges = (graph.edge_attr[:, 0] == 1).sum().item()
            base_pair_edges = (graph.edge_attr[:, 1] == 1).sum().item()
            spatial_edges = (graph.edge_attr[:, 2] == 1).sum().item()
            
            print(f"\nEdge types count:")
            print(f"- Backbone edges: {backbone_edges}")
            print(f"- Base-pair edges: {base_pair_edges}")
            print(f"- Spatial proximity edges: {spatial_edges}")
        
        # Check node features (should be one-hot encoding + position)
        if graph.x is not None:
            num_features = graph.x.shape[1]
            expected_features = 5  # 4 for one-hot nucleotides + 1 for position
            
            if num_features != expected_features:
                print(f"WARNING: Number of node features ({num_features}) doesn't match expected ({expected_features})")
            
            # Check if one-hot encoding makes sense
            one_hot_part = graph.x[:, :4]
            one_hot_sum = one_hot_part.sum(dim=1)
            if not torch.all(one_hot_sum == 1):
                print("WARNING: One-hot encoding is not valid (sum not equal to 1)")
        
        # Visualize the first graph if possible
        if i == 0:
            try:
                visualize_rna_graph(graph, sequence)
            except Exception as e:
                print(f"Could not visualize graph: {e}")