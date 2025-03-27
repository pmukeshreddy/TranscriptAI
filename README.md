# RNA 3D Structure Prediction with Graph Transformers

This repository contains a deep learning model for predicting 3D structures of RNA molecules using a combination of Graph Neural Networks (GNNs) and Transformer architectures with iterative refinement.

## Overview

Predicting the three-dimensional structure of RNA molecules is crucial for understanding their biological functions and interactions. This project implements a novel approach that combines graph representation learning with structure-aware attention mechanisms to generate accurate 3D coordinates for RNA nucleotides.

### Key Features

- **Graph-based RNA representation**: Converts RNA sequences into graph structures with nucleotide-level features and different edge types (backbone, base-pairing, spatial proximity)
- **Structure-aware attention**: Utilizes 3D spatial information to guide the attention mechanism
- **Iterative coordinate refinement**: Progressively refines predicted coordinates through multiple iterations
- **TM-score evaluation**: Assesses model quality using Template Modeling score, a standard metric in structural biology
- **Web application**: User-friendly interface for predicting RNA structures from sequences

## Model Architecture

The model consists of several key components:

1. **Graph Neural Network (GNN)**: Processes the initial RNA graph representation using Graph Attention Networks (GATv2)
2. **Transformer Encoder**: Enhances node representations using self-attention mechanisms
3. **Structure-Aware Attention**: Modulates attention based on predicted spatial distances
4. **Coordinate Refinement Module**: Iteratively refines the 3D coordinates based on node embeddings
5. **Evaluation Metrics**: Computes TM-scores to assess the quality of predicted structures

## Directory Structure

```
/
├── models/                      # Model definitions
│   ├── __init__.py
│   ├── attention.py             # Structure-aware attention implementation  
│   ├── transformer.py           # Transformer layer and encoder
│   ├── refinement.py            # Coordinate refinement module
│   ├── egnn.py                  # E(n) Equivariant Graph Neural Network
│   └── rna_transformer.py       # Main model implementation
│
├── data/                        # Data processing utilities
│   ├── __init__.py
│   ├── graph_builder.py         # Converts RNA sequences to graph representations
│   ├── dataset.py               # PyTorch dataset for RNA structures
│   └── utils.py                 # Utility functions for data processing
│
├── utils/                       # Utility functions
│   ├── __init__.py
│   ├── evaluation.py            # TM-score computation and model validation
│   └── visualization.py         # Visualization tools for RNA structures
│
├── app.py                       # Flask web application
├── templates/                   # HTML templates for web interface
│   └── index.html               # Main page template
│
├── static/                      # Static assets for web interface
│   └── css/
│       └── styles.css           # CSS styling
│
├── Dockerfile                   # Docker configuration
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

### Option 1: Using Docker

```bash
# Build the Docker image
docker build -t rna-structure-prediction .

# Run the container
docker run -p 5000:5000 rna-structure-prediction
```

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rna-structure-prediction.git
cd rna-structure-prediction

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

## Usage

### Web Application

After running the application, navigate to `http://localhost:5000` in your web browser to access the user interface. You can:

1. Enter an RNA sequence (using A, C, G, U nucleotides)
2. Click "Predict Structure" to generate a 3D model
3. View the predicted structure and refinement process
4. Download the structure in PDB format for further analysis

### Python API

```python
import torch
from models import RNAGraphTransformerWithRefinement
from data import RNA3DGraphBuilder, RNADataset

# Initialize model
model = RNAGraphTransformerWithRefinement(
    node_feature=5,
    edge_feature=3,
    hidden_dim=128,
    num_gcn_layers=8,
    num_transformer_layers=7,
    num_heads=4,
    drop_out=0.1,
    max_length=300,
    num_iterations=3
)

# Load pre-trained weights
model.load_state_dict(torch.load('models/rna_model.pt'))
model.eval()

# Prepare data for a single RNA sequence
sequence = "GGCUAGAUCAGCUUGAUUAGCUAGCC"
tokens = {nt: i for i, nt in enumerate("ACGU")}
seq_indices = [tokens.get(nt, 0) for nt in sequence]
seq_tensor = torch.tensor(seq_indices, dtype=torch.long)

# Create initial coordinates and graph
fake_coords = torch.zeros((len(sequence), 3), dtype=torch.float)
graph_builder = RNA3DGraphBuilder()
graph = graph_builder.build_graph(seq_tensor, fake_coords)

# Predict 3D structure
with torch.no_grad():
    final_coords, all_coords = model(graph)

# final_coords contains the predicted 3D coordinates
# all_coords contains intermediate refinement steps
```

## Evaluation

The model can be evaluated using TM-scores (Template Modeling scores), which assess the similarity between predicted and experimentally determined structures:

```python
from utils.evaluation import validate_rna_model_tm_score
from torch_geometric.loader import DataLoader

# Load validation dataset
val_dataset = RNADataset(val_df, graph_builder)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Run validation
results = validate_rna_model_tm_score(
    model, 
    val_loader, 
    device,
    model_name="RNA_Graph_Transformer",
    save_dir="validation_results"
)

# Print results
print(f"Mean TM-score: {results['overall']['mean_tm_score']:.4f}")
```

## Performance

The model was trained on RNA structures from various sources and evaluated using TM-scores.

TM-scores range from 0 to 1:
- 0-0.17: Random structures
- 0.5+: Structures with the same fold
- 0.9+: Very high similarity

Different RNA lengths affect prediction accuracy, with shorter sequences generally having higher accuracy.

## Web Application

The included web application provides an intuitive interface for RNA structure prediction:

- Simple RNA sequence input
- Interactive 3D visualization of predicted structures
- Visualization of the refinement process
- PDB file export for compatibility with molecular visualization software

## Dependencies

- PyTorch (>=1.9.0)
- PyTorch Geometric (>=2.0.0)
- Flask
- NumPy
- Matplotlib
- Pandas

See `requirements.txt` for a complete list of dependencies.

## Limitations

- Performance decreases for very long RNA sequences (>300 nucleotides)
- Pseudoknots and complex tertiary interactions may not be modeled accurately
- Prediction quality depends on the availability of similar structures in the training data

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@article{rna-transformer,
  title={RNA 3D Structure Prediction with Graph Transformers and Iterative Refinement},
  author={Your Name},
  journal={Journal Name},
  year={2025}
}
```

## Acknowledgments

- The model architecture builds upon recent advances in graph neural networks and transformers for biomolecular structure prediction
- The evaluation code is inspired by methods used in protein structure assessment
