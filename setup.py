import os
import sys
import shutil
from pathlib import Path

def create_directory_structure():
    """Create the necessary directory structure for the application."""
    print("Creating directory structure...")
    
    # Create main directories
    dirs = [
        "models",
        "data",
        "utils",
        "static/css",
        "templates",
        "temp"
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        
    # Create __init__.py files in each module directory
    for module_dir in ["models", "data", "utils"]:
        init_file = os.path.join(module_dir, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w") as f:
                f.write("# Auto-generated __init__.py\n")
    
    print("Directory structure created.")

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import torch
        import torch_geometric
        import torch_scatter
        import flask
        import matplotlib
        import numpy
        
        print("All dependencies found.")
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install required dependencies with: pip install -r requirements.txt")
        return False

def main():
    """Main setup function."""
    print("Setting up RNA Structure Prediction Web Application...")
    
    # Create directory structure
    create_directory_structure()
    
    # Check dependencies
    if not check_dependencies():
        print("Setup incomplete. Please install required dependencies.")
        return
    
    print("\nSetup complete!")
    print("\nTo run the application, use: python app.py")
    print("Then open your browser and navigate to: http://localhost:5000")

if __name__ == "__main__":
    main()
