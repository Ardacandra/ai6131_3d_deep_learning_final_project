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

# DeepSDF dataset defaults
DEEPSDF_DATASET = {
    "extensions": [".npz", ".npy"],
}

# DeepSDF model defaults
DEEPSDF_MODEL = {
    "latent_size": 64,
    "hidden_size": 256,
    "num_layers": 6,
    "dims": None,
    "dropout": None,
    "dropout_prob": 0.0,
    "norm_layers": (),
    "latent_in": (),
    "weight_norm": False,
    "xyz_in_all": False,
    "use_tanh": False,
    "latent_dropout": False,
}

# DeepSDF training defaults
DEEPSDF_TRAINING = {
    "data_root": Path("./data/shapenet_sdf"),
    "latent_size": 64,
    "hidden_size": 256,
    "lr": 1e-4,
    "epochs": 100,
    "batch_points": 2048,
    "samples_per_scene": 2048,
    "scenes_per_batch": 1,
    "batch_split": 1,
    "clamp_dist": 0.1,
    "code_regularization": True,
    "code_regularization_lambda": 1e-4,
    "code_bound": None,
    "code_init_stddev": 1.0,
    "grad_clip_norm": None,
    "log_frequency": 10,
    "snapshot_frequency": 10,
    "additional_snapshots": [],
    "lr_schedules": None,
    "save_dir": Path("./deepsdf_checkpoints"),
}

# DeepSDF evaluation defaults
DEEPSDF_EVALUATION = {
    "checkpoint": Path("./deepsdf_checkpoints/deepsdf_latest.pth"),
    "data_root": Path("./data/shapenet_sdf"),
    "gt_data_root": Path("./data/shapenet_v2_subset"),
    "resolution": 128,  # Grid resolution for marching cubes
    "num_sample_points": 10000,  # Number of points to sample from meshes
    "batch_size": 32768,  # Batch size for SDF queries
    "max_shapes": None,  # Maximum number of shapes to evaluate (None for all)
    "output_file": "out/deepsdf_evaluation_results.json",  # Output file for detailed results
    "percentile": 90.0,  # Percentile for mesh accuracy metric
}
