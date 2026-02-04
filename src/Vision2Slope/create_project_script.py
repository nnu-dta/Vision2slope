"""
Vision2Slope Project Creation Script
Run this script to create the full project structure and all files.
"""

import os
from pathlib import Path

# Project file contents
FILES = {
    "vision2slope/__init__.py": '''"""
Vision2Slope: Integrated Pipeline for Road Slope Analysis
========================================================

A comprehensive pipeline for road slope analysis from street view images.

Author: Cubics Yang
Date: June 2025
"""

from .config import PipelineConfig, VisualizationConfig
from .pipeline import Vision2SlopePipeline
from .models import SegmentationModel
from .detectors import SkewDetector
from .correctors import ImageCorrector
from .analyzers import RoadSlopeAnalyzer
from .utils import Utils
from .core.types import ProcessingResult, ProcessingStatus, ProcessingStage

__version__ = "1.0.0"
__all__ = [
    "PipelineConfig",
    "VisualizationConfig",
    "Vision2SlopePipeline",
    "SegmentationModel",
    "SkewDetector",
    "ImageCorrector",
    "RoadSlopeAnalyzer",
    "Utils",
    "ProcessingResult",
    "ProcessingStatus",
    "ProcessingStage",
]
''',

    "requirements.txt": '''torch>=2.0.0
transformers>=4.30.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
Pillow>=10.0.0
scikit-image>=0.21.0
scikit-learn>=1.3.0
tqdm>=4.65.0
''',

    "README.md": '''# Vision2Slope

For full documentation, see the README.md file in the artifacts above.
''',

    ".gitignore": '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Project specific
output/
results/
*.log
'''
}

def create_project(base_dir="vision2slope_project"):
    """Create the full project structure."""
    base_path = Path(base_dir)
    
    print(f"Creating project in: {base_path.absolute()}")
    
    # Create directory structure
    dirs = [
        "vision2slope",
        "examples",
        "tests",
        "docs",
    ]
    
    for dir_name in dirs:
        dir_path = base_path / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")
    
    # Create files
    for file_path, content in FILES.items():
        full_path = base_path / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ Created file: {full_path}")
    
    # Create placeholder files
    placeholder_files = [
        "tests/__init__.py",
        "docs/API.md",
    ]
    
    for file_path in placeholder_files:
        full_path = base_path / file_path
        full_path.touch()
        print(f"✓ Created placeholder: {full_path}")
    
    print("\n" + "="*60)
    print("Project created successfully!")
    print("="*60)
    print("\nPlease follow the steps below to complete project setup:")
    print("\n1. Enter the project directory:")
    print(f"   cd {base_dir}")
    print("\n2. Copy the full contents of the following files from the artifacts above into the corresponding locations:")
    print("   - vision2slope/config.py")
    print("   - vision2slope/data_types.py")
    print("   - vision2slope/utils.py")
    print("   - vision2slope/models.py")
    print("   - vision2slope/detectors.py")
    print("   - vision2slope/correctors.py")
    print("   - vision2slope/analyzers.py")
    print("   - vision2slope/visualizers.py")
    print("   - vision2slope/pipeline.py")
    print("   - vision2slope/cli.py")
    print("   - main.py")
    print("   - examples/example_usage.py")
    print("   - setup.py")
    print("\n3. Install dependencies:")
    print("   pip install -r requirements.txt")
    print("\n4. Install the project:")
    print("   pip install -e .")
    print("\n5. Run the example:")
    print("   python main.py --help")

if __name__ == "__main__":
    create_project()
