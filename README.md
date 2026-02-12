# AI6131-3D Deep Learning Final Project

This repository contains the final project for NTU MSAI's **AI6131-3D Deep Learning** course. The project focuses on 3D object recognition and reconstruction using deep learning techniques on the ShapeNet dataset.

## Prerequisites

Before getting started, ensure you have the following installed:
- **Conda** (Miniconda or Anaconda) - for environment management
- **Git** - for cloning the repository
- **NVIDIA CUDA 12.1+** (recommended) - for GPU acceleration with PyTorch
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
# Create a new conda environment with Python 3.9
conda create -n ai6131_3d_deep_learning_final_project python=3.9 -y

# Activate the environment
conda activate ai6131_3d_deep_learning_final_project

# Install PyTorch with CUDA 12.1 support for GPU acceleration
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

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

- **ShapeNet Categories** - Define which object categories to work with (default: Airplane, Chair)
- **Data Paths** - Dataset directory (`./data/shapenet_v2_subset/`) and output directories
- **Visualization Settings** - Sample sizes, figure dimensions, voxel downsampling, etc.
- **File Formats** - References to 3D file formats (OBJ, BinVOX, JSON)
- **DeepSDF Preprocessing Settings** - All parameters for SDF generation (samples, variance, voting, etc.)

To customize settings, simply edit `config.py` and the changes will be applied across all scripts automatically.

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

## Data Preprocessing for DeepSDF

To train a DeepSDF model for 3D shape reconstruction, preprocess the ShapeNet dataset into signed distance fields (SDF).

### Overview

The preprocessing pipeline implements the methodology from the DeepSDF paper (Park et al., CVPR 2019):

- **Mesh Normalization** - Scale each mesh to a unit sphere
- **Surface Sampling** - Sample 500,000 spatial points with 47% density near the surface
- **Proper Orientation** - Handle non-watertight meshes using visible surface sampling with virtual cameras (100 viewpoints)
- **SDF Computation** - Compute signed distance values using k-nearest neighbor voting (11 neighbors)
- **Output** - Save as NPZ format with separate positive/negative samples

**All preprocessing parameters are configured in `config.py`** under `DEEPSDF_SETTINGS`. Default values:
- `num_spatial_samples`: 500,000
- `surface_variance`: 0.005
- `num_votes`: 11 (k-NN neighbors for sign voting)
- `num_views`: 100 (virtual camera viewpoints)
- `output_format`: "npz"

### Running the Preprocessing

**Basic usage - process all categories (uses defaults from `config.py`):**

```bash
python prepare_deepsdf.py
```

Each output `sdf.npz` file contains:
- **pos**: Positive SDF samples (outside the mesh) - shape: (N, 4) [x, y, z, sdf_value]
- **neg**: Negative SDF samples (inside the mesh) - shape: (M, 4) [x, y, z, sdf_value]

## References

### DeepSDF

**Park et al., "DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation"** (CVPR 2019)

### ShapeNet Dataset

**Chang et al., "ShapeNet: An Information-Rich 3D Model Repository"** (arXiv:1512.02101, 2015)