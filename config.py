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
    "num_votes": 21,                # Number of neighbors for SDF voting (higher → fewer flip errors)
    "sign_ambiguity_threshold": 5e-4,  # Fallback threshold for ambiguous local sign votes
    "sign_vote_consensus_threshold": 0.65,  # Require stronger consensus before trusting projection sign
    "use_contains_sign_fallback": True,  # Use mesh.contains for sign assignment on watertight meshes
    "contains_batch_size": 8192,     # Batch size for mesh.contains calls
    "far_field_distance_threshold": 0.05,  # Treat far points as outside when sign confidence is low
    "num_views": 100,               # Number of virtual camera views
    "bounding_cube_dim": 2.0,       # Bounding cube dimension
    "objects_per_category": 50,      # Max objects to preprocess per category (None = all)
    "random_seed": 42,              # Seed for reproducible object selection
    # Optional manual selection of exact model IDs per category.
    # If a category has a non-empty list here, preprocessing will use only those IDs
    # for that category (in the order provided) and ignore objects_per_category.
    # "selected_model_ids": {
    #     # airplane
    #     "02691156": [
    #         "b4a420a55d3db8aca89fa467f217f46",
    #         "3af52163a2d0551d91637951367b1518",
    #         "6e4570ef29d420e17099115060cea9b5",
    #     ],
    #     # chair
    #     "03001627": [
    #         "5283a98b5c693e64ebefe6b1d594ad2e",
    #         "c747e6ceb1a6faaa3074f48b99186254",
    #         "5df875e6f0cc0e37f838a2212356e267",
    #     ],
    #     # table
    #     "04379243": [
    #         "e08d1cd0dc7dc73db9d7c2fc41e80228",
    #         "fff492e352c8cb336240c88cd4684446",
    #         "696beb1883be838cc955e5ed03ef3a2f",
    #     ]
    # },
    "selected_model_ids": None,
}

# DeepSDF dataset defaults
DEEPSDF_DATASET = {
    "extensions": [".npz", ".npy"],
}

# DeepSDF model defaults
DEEPSDF_MODEL = {
    "latent_size": 256,
    "hidden_size": 512,
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
    "latent_size": 256,
    "hidden_size": 512,
    "lr": 5e-4,
    "random_seed": 42,
    "epochs": 2000,
    "batch_points": 32768,
    "samples_per_scene": 8192,
    "scenes_per_batch": 10,
    "batch_split": 1,
    "clamp_dist": 1.0,
    "code_regularization": True,
    "code_regularization_lambda": 1e-5,
    "code_bound": 1.0,
    "code_init_stddev": 1.0,
    "grad_clip_norm": 1.0,
    "log_frequency": 10,
    "snapshot_frequency": 500,
    "additional_snapshots": [],
    "lr_schedules": [
        {
            "type": "step",
            "initial": 5e-4,
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

# VQ-DeepSDF model defaults (kept separate from baseline DeepSDF settings)
DEEPSDF_VQ_MODEL = {
    "latent_size": 256,
    "hidden_size": 512,
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
    # VQ bottleneck settings
    "num_codebooks": 16,
    "codebook_size": 24,
    "code_dim": 16,
    "codebook_init_scale": 0.1,
    # EMA codebook update settings
    "ema_decay": 0.99,           # Decay factor for cluster-size and embed-sum EMAs
    "dead_code_threshold": 1.0,  # Codes with EMA cluster size below this are reset
}

# VQ-DeepSDF training defaults
DEEPSDF_VQ_TRAINING = {
    "data_root": Path("./data/shapenet_sdf"),
    "latent_size": 256,
    "hidden_size": 512,
    "lr": 5e-4,
    "random_seed": 42,
    "epochs": 2000,
    "batch_points": 32768,
    "samples_per_scene": 8192,
    "scenes_per_batch": 10,
    "batch_split": 1,
    "clamp_dist": 1.0,
    "grad_clip_norm": 1.0,
    "log_frequency": 10,
    "snapshot_frequency": 500,
    "additional_snapshots": [],
    # VQ losses
    "commitment_weight": 0.25,
    "codebook_weight": 1.0,
    "entropy_weight": 1e-3,
    # Shape-latent initialization before quantization
    "shape_latent_init_stddev": 1.0,
    "shape_latent_bound": 1.0,
    "lr_schedules": [
        {
            "type": "step",
            "initial": 5e-4,
            "interval": 500,
            "factor": 0.5,
        }
    ],
    "save_dir": Path("./out/deepsdf_vq"),
}

# VQ-DeepSDF AR Transformer prior defaults
DEEPSDF_VQ_PRIOR = {
    # Architecture
    "d_model": 256,          # Embedding / model dimension
    "n_heads": 4,            # Attention heads
    "n_layers": 3,           # Number of causal transformer layers
    "ffn_multiplier": 4,     # Feed-forward hidden width = d_model * ffn_multiplier
    "dropout": 0.4,          # High dropout on attention + FFN to prevent memorisation
    # Training
    "lr": 1e-3,
    "epochs": 2000,
    "batch_size": 32,        # Number of shapes per batch
    "random_seed": 42,
    "grad_clip_norm": 1.0,
    "log_frequency": 50,
    "snapshot_frequency": 500,
    "vq_checkpoint": Path("./out/deepsdf_vq/deepsdf_vq_latest.pth"),
    "save_dir": Path("./out/deepsdf_vq_prior"),
    "lr_schedules": [
        {
            "type": "step",
            "initial": 1e-3,
            "interval": 500,
            "factor": 0.5,
        }
    ],
}

# VQ-DeepSDF prior evaluation defaults
DEEPSDF_VQ_PRIOR_EVALUATION = {
    "prior_checkpoint": Path("./out/deepsdf_vq_prior/prior_latest.pth"),
    "vq_checkpoint": Path("./out/deepsdf_vq/deepsdf_vq_latest.pth"),
    "gt_data_root": Path("./data/shapenet_v2_subset"),
    "n_samples": 50,          # Number of shapes to generate from the prior
    "temperature": 1.0,       # Sampling temperature
    "resolution": 128,        # Marching cubes resolution
    "num_sample_points": 10000,
    "batch_size": 32768,
    "output_file": Path("./out/deepsdf_vq_prior/prior_evaluation_results.json"),
}

# VQ-DeepSDF Gaussian baseline evaluation defaults
DEEPSDF_VQ_GAUSSIAN_BASELINE_EVALUATION = {
    "vq_checkpoint": Path("./out/deepsdf_vq/deepsdf_vq_latest.pth"),
    "gt_data_root": Path("./data/shapenet_v2_subset"),
    "n_samples": 50,          # Number of shapes to generate
    "std": 1.0,               # Standard deviation of the Gaussian noise
    "resolution": 128,        # Marching cubes resolution
    "num_sample_points": 10000,
    "batch_size": 32768,
    "output_file": Path("./out/deepsdf_vq_gaussian_baseline/gaussian_baseline_evaluation_results.json"),
}

# VQ-DeepSDF evaluation defaults
DEEPSDF_VQ_EVALUATION = {
    "checkpoint": Path("./out/deepsdf_vq/deepsdf_vq_latest.pth"),
    "data_root": Path("./data/shapenet_sdf"),
    "gt_data_root": Path("./data/shapenet_v2_subset"),
    "resolution": 128,
    "num_sample_points": 10000,
    "batch_size": 32768,
    "max_shapes": None,
    "output_file": Path("./out/deepsdf_vq/deepsdf_vq_evaluation_results.json"),
    "percentile": 90.0,
}
