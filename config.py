"""
Central configuration file for the ShapeNet 3D Deep Learning project.
All project-wide settings should be defined here and imported by other modules.
"""

from pathlib import Path

# Data directory configuration
DATA_DIR = Path("./data/shapenet_v2_subset")
OUTPUT_DIR = Path("./out/")

# ShapeNet category definitions
# Format: {synset_id: category_name}
SHAPENET_CATEGORIES = {
    "02691156": "Airplane",
    "02747177": "Chair",
    # Uncomment to add more categories:
    # "04379243": "Table",
    # "02958343": "Car",
}

# Synset IDs for dataset download (used by prepare_dataset.py)
TARGET_SYNSETS = list(SHAPENET_CATEGORIES.keys())

# Visualization settings
VISUALIZATION_SETTINGS = {
    "sample_models_per_category": 4,
    "sample_voxels_per_category": 2,
    "mesh_figure_size": (16, 12),
    "mesh_dpi": 100,
    "voxel_figure_size": (14, 6),
    "voxel_dpi": 100,
    "voxel_downsampling_target": 32,  # Downsample voxel grids to 32x32x32
    "mesh_sample_faces": 500,  # Number of faces to sample for edge visualization
}

# File format settings
FILE_FORMATS = {
    "mesh": "model_normalized.obj",
    "mesh_material": "model_normalized.mtl",
    "voxel_surface": "model_normalized.surface.binvox",
    "voxel_solid": "model_normalized.solid.binvox",
    "metadata": "model_normalized.json",
}

# DeepSDF preprocessing settings
DEEPSDF_SETTINGS = {
    "num_spatial_samples": 500000,  # Total spatial samples per mesh
    "surface_variance": 0.005,      # Variance for near-surface sampling
    "near_surface_ratio": 47.0 / 50.0,  # Ratio of near-surface to random samples
    "output_format": "npz",         # Output format: "npz" or "npy"
    "output_dir": Path("./data/shapenet_sdf/"),  # Output directory for SDF data
    "num_votes": 11,                # Number of neighbors for SDF voting
    "num_views": 100,               # Number of virtual camera views
    "bounding_cube_dim": 2.0,       # Bounding cube dimension
}
