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
    "03001627": "Chair",
    "04379243": "Table",
}

# Synset IDs for dataset download (used by prepare_dataset.py)
TARGET_SYNSETS = list(SHAPENET_CATEGORIES.keys())

# Visualization settings
VISUALIZATION_SETTINGS = {
    "sample_models_per_category": 4,
    "sample_voxels_per_category": 2,
    "voxel_downsampling_target": 72,  # Downsample voxel grids to 72x72x72
}

# DeepSDF preprocessing settings
DEEPSDF_SETTINGS = {
    "num_spatial_samples": 500000,  # Total spatial samples per mesh
    "surface_variance_primary": 0.005,  # Broader near-surface normal offset variance
    "surface_variance_secondary": 0.0005,  # Tighter near-surface normal offset variance
    "surface_sample_ratio_primary": 0.5,  # Fraction of offsets drawn from the primary variance
    "surface_offset_clip_multiplier": 3.0,  # Clamp offsets to avoid extreme outliers
    "near_surface_ratio": 47.0 / 50.0,  # Fraction of total samples that are near-surface (47:3 from DeepSDF paper)
    "output_format": "npz",         # Output format: "npz" or "npy"
    "output_dir": Path("./data/shapenet_sdf/"),  # Output directory for SDF data
    "num_votes": 11,                # Number of neighbors for SDF voting
    "sign_ambiguity_threshold": 1e-4,  # Fallback threshold for ambiguous local sign votes
    "sign_vote_consensus_threshold": 0.35,  # Require consistent neighbor vote polarity for confident sign
    "use_contains_sign_fallback": True,  # Use mesh.contains for ambiguous signs on watertight meshes
    "contains_batch_size": 8192,     # Batch size for optional mesh.contains fallback
    "far_field_distance_threshold": 0.08,  # Treat far points as outside when sign confidence is low
    "num_views": 100,               # Number of virtual camera views
    "bounding_cube_dim": 2.0,       # Bounding cube dimension
    "objects_per_category": 50,      # Max objects to preprocess per category (None = all)
    "random_seed": 42,              # Seed for reproducible object selection
}

# DeepSDF dataset defaults
DEEPSDF_DATASET = {
    "extensions": [".npz", ".npy"],
}

# DeepSDF model defaults
DEEPSDF_MODEL = {
    "latent_size": 32,
    "hidden_size": 256,
    "num_layers": 8,
    "dims": None,
    "dropout": None,
    "dropout_prob": 0.0,
    "norm_layers": (),
    "latent_in": [4],
    "weight_norm": True,
    "xyz_in_all": False,
    "use_tanh": False,
    "latent_dropout": False,
}

# DeepSDF training defaults
DEEPSDF_TRAINING = {
    "data_root": Path("./data/shapenet_sdf"),
    "latent_size": 32,
    "hidden_size": 256,
    "lr": 3e-4,
    "random_seed": 42,
    "epochs": 2000,
    "batch_points": 16384,
    "samples_per_scene": 4096,
    "scenes_per_batch": 16,
    "batch_split": 1,
    "clamp_dist": 0.1,
    "code_regularization": True,
    "code_regularization_lambda": 1e-4,
    "code_bound": 1.0,
    "code_init_stddev": 1.0,
    "grad_clip_norm": 1.0,
    "log_frequency": 10,
    "snapshot_frequency": 100,
    "additional_snapshots": [],
    "lr_schedules": [
        {
            "type": "step",
            "initial": 3e-4,
            "interval": 500,   # every 500 epochs
            "factor": 0.5      # multiply LR by 0.5
        }
    ],
    "save_dir": Path("./out/deepsdf"),
}

# DeepSDF evaluation defaults
DEEPSDF_EVALUATION = {
    "checkpoint": Path("./out/deepsdf/deepsdf_latest.pth"),
    "data_root": Path("./data/shapenet_sdf"),
    "gt_data_root": Path("./data/shapenet_v2_subset"),
    "resolution": 128,  # Grid resolution for marching cubes
    "num_sample_points": 10000,  # Number of points to sample from meshes
    "batch_size": 32768,  # Batch size for SDF queries
    "max_shapes": None,  # Maximum number of shapes to evaluate (None for all)
    "output_file": Path("./out/deepsdf/deepsdf_evaluation_results.json"),  # Output file for detailed results
    "percentile": 90.0,  # Percentile for mesh accuracy metric
}
