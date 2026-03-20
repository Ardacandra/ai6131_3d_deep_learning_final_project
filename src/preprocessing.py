"""
DeepSDF Data Preprocessing Pipeline.

This implementation emphasizes geometry-aware SDF labels for thin structures,
openings, and holes by:
- Normalizing meshes to the unit sphere
- Sampling dense oriented surface points
- Creating near-surface samples by offsetting along local normals
- Estimating signs for broader spatial samples from local surface geometry
- Saving training data in NPY/NPZ format
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from scipy.spatial import cKDTree
import trimesh
from tqdm import tqdm

from config import DEEPSDF_SETTINGS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MeshNormalizer:
    """Normalize mesh to unit sphere"""
    
    @staticmethod
    def normalize(mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, np.ndarray, float]:
        """
        Normalize mesh to unit sphere.
        
        Args:
            mesh: Input trimesh object
            
        Returns:
            normalized_mesh: Mesh scaled to unit sphere
            offset: Translation offset applied
            scale: Scaling factor applied
        """
        # Compute centroid
        centroid = mesh.vertices.mean(axis=0)
        mesh.vertices -= centroid
        
        # Compute max distance from origin
        max_dist = np.max(np.linalg.norm(mesh.vertices, axis=1))
        
        # Scale to unit sphere
        scale = 1.0 / max_dist if max_dist > 0 else 1.0
        mesh.vertices *= scale
        
        return mesh, centroid, scale


class SurfaceSampler:
    """Sample points on mesh surface with proper orientation"""
    
    def __init__(self, mesh: trimesh.Trimesh, num_views: Optional[int] = None):
        """
        Initialize surface sampler.
        
        Args:
            mesh: Input trimesh
            num_views: Number of virtual camera views for visibility determination
        """
        self.mesh = mesh
        self.num_views = num_views or DEEPSDF_SETTINGS["num_views"]
        self.vertices = None
        self.normals = None
        
    def get_equidistant_sphere_points(self, radius: float = 1.1) -> np.ndarray:
        """
        Generate equidistant points on sphere using Fibonacci sphere algorithm.
        
        Args:
            radius: Radius of sphere
            
        Returns:
            Array of points on sphere surface
        """
        points = np.zeros((self.num_views, 3))
        offset = 2.0 / self.num_views
        increment = np.pi * (3.0 - np.sqrt(5.0))
        
        for i in range(self.num_views):
            y = ((i * offset) - 1) + (offset / 2)
            r = np.sqrt(1 - y**2)
            phi = (i + 1) * increment
            
            x = np.cos(phi) * r
            z = np.sin(phi) * r
            
            points[i] = radius * np.array([x, y, z])
        
        return points
    
    def sample_surface_with_orientation(self, num_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample surface points with proper orientation (facing camera).
        
        Uses multiple virtual camera viewpoints to determine which triangles are
        visible and oriented correctly.
        
        Args:
            num_samples: Number of surface points to sample (default from config)
            
        Returns:
            surface_points: Sampled surface points
            surface_normals: Surface normals at sampled points
        """
        num_samples = num_samples or DEEPSDF_SETTINGS["num_spatial_samples"]
        logger.info(f"Sampling {num_samples} surface points with proper orientation...")
        
        # Use trimesh's precomputed face normals.  After fix_normals() in
        # preprocess_mesh these are consistently outward-facing for watertight
        # meshes.  A vectorised radial-alignment check reduces residual errors
        # on non-watertight geometry without a slow per-face Python loop.
        face_normals = self.mesh.face_normals.copy()

        # Sanity check global orientation: after normalization around origin,
        # outward normals should generally point away from the center.
        face_centers = self.mesh.vertices[self.mesh.faces].mean(axis=1)
        radial_alignment = np.einsum("ij,ij->i", face_normals, face_centers)
        if np.median(radial_alignment) < 0.0:
            face_normals = -face_normals
            logger.debug("Flipped face normals globally to enforce outward orientation")

        # Sample triangles weighted by area (vectorised).
        area_list = self.mesh.area_faces
        area_list = area_list / area_list.sum()

        sampled_face_indices = np.random.choice(
            len(self.mesh.faces),
            size=num_samples,
            p=area_list,
        )

        # Vectorised barycentric point sampling.
        r1 = np.random.rand(num_samples)
        r2 = np.random.rand(num_samples)
        flip = r1 + r2 > 1.0
        r1[flip], r2[flip] = 1.0 - r1[flip], 1.0 - r2[flip]
        r3 = 1.0 - r1 - r2

        tri_verts = self.mesh.vertices[self.mesh.faces[sampled_face_indices]]
        surface_points = (
            r3[:, None] * tri_verts[:, 0]
            + r1[:, None] * tri_verts[:, 1]
            + r2[:, None] * tri_verts[:, 2]
        )
        surface_normals_arr = face_normals[sampled_face_indices]

        return surface_points, surface_normals_arr


class SpatialPointSampler:
    """Sample spatial points with higher density near surface"""
    
    @staticmethod
    def sample_near_surface(
        surface_points: np.ndarray,
        surface_normals: np.ndarray,
        num_samples: int,
        primary_variance: Optional[float] = None,
        secondary_variance: Optional[float] = None,
        primary_ratio: Optional[float] = None,
        clip_multiplier: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample points near surface by moving along oriented surface normals.

        This keeps the sign locally consistent around cavities and openings,
        which isotropic xyz perturbations tend to blur.
        
        Args:
            surface_points: Surface points to perturb
            surface_normals: Outward-oriented normals for the sampled surface points
            num_samples: Number of spatial samples to generate
            primary_variance: Broader Gaussian variance for offsets
            secondary_variance: Tighter Gaussian variance for offsets
            primary_ratio: Fraction of samples drawn from the primary variance
            clip_multiplier: Sigma multiplier used to clamp extreme offsets
            
        Returns:
            Tuple of sampled spatial points and their signed distances
        """
        if primary_variance is None:
            primary_variance = DEEPSDF_SETTINGS["surface_variance_primary"]
        if secondary_variance is None:
            secondary_variance = DEEPSDF_SETTINGS["surface_variance_secondary"]
        if primary_ratio is None:
            primary_ratio = DEEPSDF_SETTINGS["surface_sample_ratio_primary"]
        if clip_multiplier is None:
            clip_multiplier = DEEPSDF_SETTINGS["surface_offset_clip_multiplier"]
        logger.info(f"Sampling {num_samples} spatial points near surface...")

        if len(surface_points) == 0:
            return np.empty((0, 3)), np.empty((0,), dtype=np.float32)

        num_primary = int(round(num_samples * primary_ratio))
        num_primary = min(max(num_primary, 0), num_samples)
        num_secondary = num_samples - num_primary

        def _sample_group(count: int, variance: float) -> Tuple[np.ndarray, np.ndarray]:
            if count <= 0:
                return np.empty((0, 3)), np.empty((0,), dtype=np.float32)

            indices = np.random.randint(0, len(surface_points), size=count)
            sigma = np.sqrt(variance)
            offsets = np.random.normal(0.0, sigma, size=count)
            if clip_multiplier is not None and clip_multiplier > 0:
                offsets = np.clip(offsets, -clip_multiplier * sigma, clip_multiplier * sigma)

            sampled_points = surface_points[indices] + surface_normals[indices] * offsets[:, None]
            return sampled_points, offsets.astype(np.float32)

        primary_points, primary_sdf = _sample_group(num_primary, primary_variance)
        secondary_points, secondary_sdf = _sample_group(num_secondary, secondary_variance)

        spatial_points = np.vstack([primary_points, secondary_points])
        sdf_values = np.concatenate([primary_sdf, secondary_sdf])
        return spatial_points, sdf_values
    
    @staticmethod
    def sample_random_cube(
        num_samples: int,
        bounding_cube_dim: Optional[float] = None
    ) -> np.ndarray:
        """
        Sample random points from bounding cube.
        
        Args:
            num_samples: Number of random samples
            bounding_cube_dim: Size of bounding cube (default from config)
            
        Returns:
            Random spatial points
        """
        bounding_cube_dim = bounding_cube_dim or DEEPSDF_SETTINGS["bounding_cube_dim"]
        logger.info(f"Sampling {num_samples} random points in cube...")
        
        half_dim = bounding_cube_dim / 2.0
        return np.random.uniform(-half_dim, half_dim, size=(num_samples, 3))


class SDFComputer:
    """Compute signed distance fields"""
    
    def __init__(self, mesh: trimesh.Trimesh, surface_points: np.ndarray, surface_normals: np.ndarray):
        """
        Initialize SDF computer.
        
        Args:
            mesh: Input trimesh
            surface_points: Surface points with proper orientation
            surface_normals: Surface normals at sampled points
        """
        self.mesh = mesh
        self.surface_tree = cKDTree(surface_points)
        self.surface_points = surface_points
        self.surface_normals = surface_normals
        self.mesh_is_watertight = bool(mesh.is_watertight)
    
    def compute(
        self,
        spatial_points: np.ndarray,
        num_votes: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute signed distance field values.

        Unsigned distance is approximated with the nearest oriented surface sample,
        while the sign is estimated from a weighted local projection onto nearby
        surface normals. This is substantially more faithful to cavities and thin
        structures than a mesh-center heuristic.
        
        Args:
            spatial_points: Points at which to compute SDF
            num_votes: Number of neighbors used for local sign voting
            
        Returns:
            Array of signed distance values
        """
        if len(spatial_points) == 0:
            return np.empty((0,), dtype=np.float32)

        num_votes = num_votes or DEEPSDF_SETTINGS["num_votes"]
        num_votes = max(1, min(num_votes, len(self.surface_points)))
        logger.info("Computing signed distance field values...")

        # Find k-nearest oriented surface samples.
        distances, indices = self.surface_tree.query(spatial_points, k=num_votes)

        if np.isscalar(distances):
            distances = np.array([[distances]], dtype=np.float32)
            indices = np.array([[indices]], dtype=np.int64)
        elif distances.ndim == 1:
            distances = distances[:, None]
            indices = indices[:, None]

        neighbor_points = self.surface_points[indices]
        neighbor_normals = self.surface_normals[indices]
        deltas = spatial_points[:, None, :] - neighbor_points
        projected_distances = np.sum(deltas * neighbor_normals, axis=2)

        weights = 1.0 / np.maximum(distances, 1e-8)
        weight_sum = np.sum(weights, axis=1)
        weighted_projection = np.sum(projected_distances * weights, axis=1) / weight_sum

        vote_signs = np.sign(projected_distances)
        vote_signs[vote_signs == 0.0] = 1.0
        vote_consensus = np.abs(np.sum(vote_signs * weights, axis=1) / weight_sum)

        unsigned_distance = distances[:, 0]

        # For watertight meshes, mesh.contains is the most reliable sign oracle.
        # Applying it to ALL spatial points eliminates both "confidently wrong"
        # labels that the ambiguity filter cannot catch, and ambiguous ones, in
        # a single pass.
        if DEEPSDF_SETTINGS.get("use_contains_sign_fallback", True) and self.mesh_is_watertight:
            try:
                contains_batch_size = max(1, int(DEEPSDF_SETTINGS.get("contains_batch_size", 8192)))
                inside_mask = np.zeros(len(spatial_points), dtype=bool)
                for start in range(0, len(spatial_points), contains_batch_size):
                    end = start + contains_batch_size
                    inside_mask[start:end] = self.mesh.contains(spatial_points[start:end])
                signs = np.where(inside_mask, -1.0, 1.0)
                return (signs * unsigned_distance).astype(np.float32)
            except Exception as exc:
                logger.debug("mesh.contains unavailable for full sign pass: %s", exc)

        # Fallback for non-watertight meshes: local normal-projection voting
        # with ambiguity resolution.
        nearest_projection = projected_distances[:, 0]
        sign_source = weighted_projection.copy()
        ambiguity_threshold = DEEPSDF_SETTINGS["sign_ambiguity_threshold"]
        consensus_threshold = DEEPSDF_SETTINGS["sign_vote_consensus_threshold"]
        ambiguous = (np.abs(sign_source) < ambiguity_threshold) | (vote_consensus < consensus_threshold)

        if np.any(ambiguous):
            ambiguous_idx = np.where(ambiguous)[0]
            far_field_threshold = DEEPSDF_SETTINGS.get("far_field_distance_threshold", 0.05)
            far_mask = unsigned_distance[ambiguous_idx] > far_field_threshold
            if np.any(far_mask):
                sign_source[ambiguous_idx[far_mask]] = 1.0
            unresolved_idx = ambiguous_idx[~far_mask]
            sign_source[unresolved_idx] = nearest_projection[unresolved_idx]

        signs = np.where(sign_source < 0.0, -1.0, 1.0)

        return (signs * unsigned_distance).astype(np.float32)


class DeepSDFPreprocessor:
    """Main preprocessing pipeline for DeepSDF"""
    
    def __init__(
        self,
        num_spatial_samples: Optional[int] = None,
        surface_variance: Optional[float] = None,
        surface_variance_secondary: Optional[float] = None,
        output_format: Optional[str] = None
    ):
        """
        Initialize preprocessor.
        
        Args:
            num_spatial_samples: Total spatial samples to generate (default from config)
            surface_variance: Primary variance for near-surface sampling
            surface_variance_secondary: Secondary tighter variance for near-surface sampling
            output_format: Output format 'npz' or 'npy' (default from config)
        """
        self.num_spatial_samples = (
            num_spatial_samples if num_spatial_samples is not None else DEEPSDF_SETTINGS["num_spatial_samples"]
        )
        self.surface_variance = (
            surface_variance if surface_variance is not None else DEEPSDF_SETTINGS["surface_variance_primary"]
        )
        self.surface_variance_secondary = (
            surface_variance_secondary
            if surface_variance_secondary is not None
            else DEEPSDF_SETTINGS["surface_variance_secondary"]
        )
        self.output_format = output_format or DEEPSDF_SETTINGS["output_format"]
        
        # Ratio of near-surface to random samples (47:3 from paper)
        self.near_surface_ratio = DEEPSDF_SETTINGS["near_surface_ratio"]
    
    def preprocess_mesh(
        self,
        mesh_path: str,
        output_path: str
    ) -> dict:
        """
        Preprocess a single mesh.
        
        Args:
            mesh_path: Path to input mesh (OBJ, etc.)
            output_path: Path to save processed data
            
        Returns:
            Dictionary with preprocessing statistics
        """
        logger.info(f"Processing mesh: {mesh_path}")
        
        # Load mesh
        mesh = trimesh.load(mesh_path)
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Could not load mesh from {mesh_path}")

        # Repair mesh geometry and fix normal orientation before any sampling.
        # fix_normals() ensures consistent outward-facing normals for watertight
        # meshes, which directly improves both the offset-based near-surface
        # labels and the normal-projection sign voter.
        trimesh.repair.fix_normals(mesh)

        stats = {
            'input_file': mesh_path,
            'output_file': output_path,
            'num_vertices': len(mesh.vertices),
            'num_faces': len(mesh.faces),
            'is_watertight': bool(mesh.is_watertight),
        }
        
        # Normalize mesh
        logger.info("Normalizing mesh to unit sphere...")
        mesh, offset, scale = MeshNormalizer.normalize(mesh)
        stats['offset'] = offset.tolist()
        stats['scale'] = float(scale)
        
        # Sample surface with proper orientation
        sampler = SurfaceSampler(mesh)
        num_surface_samples = int(self.num_spatial_samples * self.near_surface_ratio)
        surface_points, surface_normals = sampler.sample_surface_with_orientation(num_surface_samples)
        
        # Sample spatial points
        num_near_surface = int(self.num_spatial_samples * self.near_surface_ratio)
        num_random = self.num_spatial_samples - num_near_surface

        near_surface_points, near_surface_sdf = SpatialPointSampler.sample_near_surface(
            surface_points,
            surface_normals,
            num_near_surface,
            primary_variance=self.surface_variance,
            secondary_variance=self.surface_variance_secondary,
        )

        # For watertight meshes, override the offset-based near-surface signs
        # with mesh.contains to correct any residual flipped-normal artefacts.
        if mesh.is_watertight and DEEPSDF_SETTINGS.get("use_contains_sign_fallback", True):
            try:
                contains_batch_size = max(1, int(DEEPSDF_SETTINGS.get("contains_batch_size", 8192)))
                inside_mask = np.zeros(len(near_surface_points), dtype=bool)
                for start in range(0, len(near_surface_points), contains_batch_size):
                    end = start + contains_batch_size
                    inside_mask[start:end] = mesh.contains(near_surface_points[start:end])
                corrected = np.where(
                    inside_mask,
                    -np.abs(near_surface_sdf),
                    np.abs(near_surface_sdf),
                ).astype(np.float32)
                n_flipped = int(np.sum(np.sign(corrected) != np.sign(near_surface_sdf)))
                if n_flipped:
                    logger.debug("Near-surface sign correction: flipped %d / %d labels", n_flipped, len(near_surface_sdf))
                near_surface_sdf = corrected
            except Exception as exc:
                logger.debug("mesh.contains near-surface sign correction failed: %s", exc)

        random_points = SpatialPointSampler.sample_random_cube(num_random)

        # Compute SDF values for the broader spatial samples with local geometry voting.
        sdf_computer = SDFComputer(mesh, surface_points, surface_normals)
        random_sdf = sdf_computer.compute(random_points)

        # Combine all spatial points and SDF values.
        spatial_points = np.vstack([near_surface_points, random_points])
        sdf_values = np.concatenate([near_surface_sdf, random_sdf])
        logger.info(f"Total spatial points: {len(spatial_points)}")
        
        # Save output
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.output_format == "npz":
            # Separate positive and negative SDF values
            pos_mask = sdf_values >= 0
            pos_data = np.column_stack([spatial_points[pos_mask], sdf_values[pos_mask]])
            neg_data = np.column_stack([spatial_points[~pos_mask], sdf_values[~pos_mask]])
            
            np.savez(output_path, pos=pos_data, neg=neg_data)
            stats['format'] = 'npz'
            stats['num_positive_samples'] = len(pos_data)
            stats['num_negative_samples'] = len(neg_data)
        else:
            # Save as NPY with XYZS format
            data = np.column_stack([spatial_points, sdf_values])
            np.save(output_path, data)
            stats['format'] = 'npy'
            stats['num_samples'] = len(data)
        
        logger.info(f"✅ Saved to: {output_path}")
        
        return stats
    
    def preprocess_dataset(
        self,
        dataset_dir: str,
        output_dir: str,
        category: Optional[str] = None,
        objects_per_category: Optional[int] = None,
        random_seed: Optional[int] = None,
        selected_model_ids: Optional[Dict[str, List[str]]] = None,
    ) -> dict:
        """
        Preprocess entire dataset.

        Iterates through candidate meshes in each category and keeps processing
        until the number of successfully prepared objects reaches
        ``objects_per_category`` (when specified). Failed meshes are skipped and
        do not count toward the quota. The selected model IDs are written to
        ``<output_dir>/selected_samples.json`` after processing finishes.

        Args:
            dataset_dir: Root dataset directory
            output_dir: Output directory for processed meshes
            category: Specific category to process (optional)
            objects_per_category: Max objects to preprocess per category.
                ``None`` (default) processes every available model.
            random_seed: Seed for reproducible object selection.
            selected_model_ids: Optional mapping of category_id -> ordered list of
                model IDs to preprocess. When provided for a category, this manual
                selection takes precedence over ``objects_per_category``.

        Returns:
            Dictionary with overall statistics
        """
        dataset_dir = Path(dataset_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(random_seed)
        selected_model_ids = (
            selected_model_ids
            if selected_model_ids is not None
            else DEEPSDF_SETTINGS.get("selected_model_ids", {})
        )
        selected_model_ids = selected_model_ids or {}

        all_stats = []
        selected_by_category: Dict[str, List[str]] = {}

        # Find all mesh files
        if category:
            category_dirs = [dataset_dir / category]
        else:
            category_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])

        # Collect available models per category.
        # Maps category_name -> list of (model_id, obj_file) tuples
        category_models: Dict[str, List[tuple]] = {}
        for category_dir in category_dirs:
            if not category_dir.is_dir():
                continue

            category_name = category_dir.name
            available = []
            for model_dir in sorted(category_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                models_dir = model_dir / "models"
                if not models_dir.exists():
                    continue
                obj_files = list(models_dir.glob("model_normalized.obj"))
                if obj_files:
                    available.append((model_dir.name, obj_files[0]))

            if not available:
                continue

            category_models[category_name] = available

        # Process meshes while ensuring successful count hits per-category quota.
        total_meshes = 0
        for category_name, available_models in category_models.items():
            selected_successful_ids: List[str] = []
            target_count = objects_per_category
            manual_ids = selected_model_ids.get(category_name, [])

            # Keep deterministic selection order when a seed is provided.
            # If no seed is provided, keep sorted filesystem order.
            models = list(available_models)
            if manual_ids:
                by_model_id = {model_id: (model_id, obj_file) for model_id, obj_file in models}
                missing_ids = [model_id for model_id in manual_ids if model_id not in by_model_id]
                if missing_ids:
                    logger.warning(
                        "Category %s has %d requested model IDs not found: %s",
                        category_name,
                        len(missing_ids),
                        ", ".join(missing_ids),
                    )

                models = [by_model_id[model_id] for model_id in manual_ids if model_id in by_model_id]
                target_count = len(models)
            elif random_seed is not None:
                perm = rng.permutation(len(models))
                models = [models[idx] for idx in perm]

            logger.info(
                f"\n📁 Processing category: {category_name} "
                f"(available={len(models)}, target={target_count if target_count is not None else 'All'})"
            )

            for model_id, obj_file in tqdm(models, desc=f"Category {category_name}"):
                if target_count is not None and len(selected_successful_ids) >= target_count:
                    break

                # Create output structure
                output_category_dir = output_dir / category_name / model_id
                output_file = output_category_dir / f"sdf.{self.output_format}"

                # Skip if already processed
                if output_file.exists():
                    logger.debug(f"Skipping {model_id} (already exists)")
                    selected_successful_ids.append(model_id)
                    total_meshes += 1
                    continue

                try:
                    stats = self.preprocess_mesh(str(obj_file), str(output_file))
                    stats['category'] = category_name
                    stats['model_id'] = model_id
                    all_stats.append(stats)
                    selected_successful_ids.append(model_id)
                    total_meshes += 1
                except Exception as e:
                    logger.warning(f"Failed to process {model_id}: {e}")

            if target_count is not None and len(selected_successful_ids) < target_count:
                logger.warning(
                    "Category %s reached only %d/%d successful objects",
                    category_name,
                    len(selected_successful_ids),
                    target_count,
                )

            selected_by_category[category_name] = selected_successful_ids

        # Write successful sample manifest after processing.
        manifest_path = output_dir / "selected_samples.json"
        with open(manifest_path, "w") as f:
            json.dump(selected_by_category, f, indent=2)
        logger.info(f"Saved selected samples manifest: {manifest_path}")
        
        # Summary statistics
        summary = {
            'total_meshes_processed': total_meshes,
            'output_directory': str(output_dir),
            'output_format': self.output_format,
            'selected_samples_file': str(manifest_path),
            'statistics': all_stats
        }
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Preprocessing complete!")
        logger.info(f"Total meshes processed: {total_meshes}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"{'='*60}\n")
        
        return summary


def main():
    """Example usage"""
    from config import DATA_DIR
    
    # Initialize preprocessor
    preprocessor = DeepSDFPreprocessor(
        num_spatial_samples=500000,
        surface_variance=0.005,
        surface_variance_secondary=0.0005,
        output_format="npz"
    )
    
    # Process dataset
    preprocessor.preprocess_dataset(
        dataset_dir=str(DATA_DIR),
        output_dir="./data/shapenet_sdf/",
    )


if __name__ == "__main__":
    main()
