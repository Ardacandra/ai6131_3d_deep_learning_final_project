# AI6131-3D Deep Learning Final Project

This repository contains the final project for NTU MSAI's **AI6131-3D Deep Learning** course. The project focuses on 3D object recognition and reconstruction using deep learning techniques on the ShapeNet dataset.

## Prerequisites

Before getting started, ensure you have the following installed:
- **Conda** (Miniconda or Anaconda) - for environment management
- **Git** - for cloning the repository
- **NVIDIA CUDA 11.8+** (recommended) - for GPU acceleration with PyTorch
- **HuggingFace API Token** - to access the ShapeNet dataset (get one at https://huggingface.co/settings/tokens)

## Setup Instructions

### 1. Clone the Repository

Download a copy of this project to your local machine:

```bash
git clone https://github.com/Ardacandra/ai6131_3d_deep_learning_final_project.git
cd ai6131_3d_deep_learning_final_project
```

### 2. Set Up the Conda Environment

Create an isolated Python environment with all necessary dependencies. This ensures your project won't conflict with other Python packages on your system.

```bash
# Create a new conda environment with Python 3.10
conda create -n ai6131_3d_deep_learning_final_project python=3.10 -y

# Activate the environment
conda activate ai6131_3d_deep_learning_final_project

# Install PyTorch with CUDA 11.8 support for GPU acceleration
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install all other project dependencies from requirements.txt
pip install -r requirements.txt
```

### 3. Configure HuggingFace Authentication

The ShapeNet dataset is hosted on HuggingFace, so you need to provide your access token:

1. **Create a `.env` file** by copying the example template:
   ```bash
   cp .env.example .env
   ```

2. **Add your HuggingFace token** to the `.env` file:
   - Get your token from https://huggingface.co/settings/tokens
   - Open `.env` in a text editor and set:
     ```
     HF_TOKEN=your_token_here
     ```

### 4. Project Configuration

All project-wide settings are centralized in **`config.py`**. This is the single source of truth for:

- **ShapeNet Categories** - Define which object categories to work with
- **Data Paths** - Dataset and output directory locations
- **Visualization Settings** - Sample sizes, figure dimensions, etc.
- **File Formats** - References to 3D file formats (OBJ, BinVOX, etc.)

### 5. Download and Prepare the Dataset

Download the ShapeNet dataset locally. This script uses your HF token to authenticate with HuggingFace:

```bash
python prepare_dataset.py
```

The dataset will be downloaded to `./data/shapenet_v2_subset/`. Unzip them with the following script:

```bash
cd ./data/shapenet_v2_subset
for f in *.zip; do unzip "$f"; done
cd ../..
```

### 6. Explore and Visualize the Dataset

To understand how 3D objects are represented in the dataset and visualize sample models, run the visualization script:

```bash
python visualize_dataset.py
```

This script will:
- **Scan the dataset** and display statistics (number of models per category)
- **Show file formats** used in the dataset (OBJ mesh, BinVOX voxels, etc.)
- **Generate mesh visualizations** showing sample models from each category as 3D point clouds with wireframe edges
- **Generate voxel visualizations** showing the volumetric representation of models
- **Save all visualizations** to the `./out/` directory as PNG files

**Output files** (for each category in `SHAPENET_CATEGORIES`):
- `visualization_<category>_models.png` - Mesh visualizations
- `visualization_<category>_voxels.png` - Voxel grid visualizations