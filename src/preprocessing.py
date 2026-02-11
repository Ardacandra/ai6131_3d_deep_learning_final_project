"""
DeepSDF Data Preprocessing Pipeline

This module implements the data preparation method from the DeepSDF paper:
- Normalizes meshes to unit sphere
- Samples 500,000 spatial points with higher density near surface
- Handles non-watertight meshes using visible surface sampling
- Computes signed distance fields (SDF)
- Outputs training data in NPY/NPZ format

Based on: Park et al., "DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation"
C++ reference: Facebook Research preprocessing scripts
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional
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
        
        # Get virtual camera positions
        views = self.get_equidistant_sphere_points()
        
        # For each face, track which camera sees it
        face_normals = np.zeros_like(self.mesh.face_normals)
        face_visibility = {}
        
        for face_idx in range(len(self.mesh.faces)):
            face = self.mesh.faces[face_idx]
            v0, v1, v2 = self.mesh.vertices[face]
            
            # Compute face normal using right-hand rule
            normal = np.cross(v1 - v0, v2 - v0)
            normal_length = np.linalg.norm(normal)
            if normal_length > 1e-8:
                normal = normal / normal_length
            else:
                normal = np.array([0., 0., 1.])  # Fallback for degenerate faces
            
            # Majority voting: count how many views see this normal as front-facing
            # Front-facing means dot(normal, view_direction) > 0
            front_facing_votes = 0
            face_center = (v0 + v1 + v2) / 3.0
            
            for view in views:
                view_dir = (view - face_center) / (np.linalg.norm(view - face_center) + 1e-8)
                dot_product = np.dot(normal, view_dir)
                
                if dot_product > 0:  # Normal points towards this view (outward)
                    front_facing_votes += 1
            
            # If majority of views see it as back-facing, flip the normal
            if front_facing_votes < len(views) // 2:
                normal = -normal
            
            face_normals[face_idx] = normal
        
        # Sample points from surface weighted by triangle area
        area_list = []
        for face_idx, face in enumerate(self.mesh.faces):
            v0, v1, v2 = self.mesh.vertices[face]
            area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
            area_list.append(area)
        
        area_list = np.array(area_list)
        area_list = area_list / area_list.sum()
        
        # Sample triangles according to area
        sampled_face_indices = np.random.choice(
            len(self.mesh.faces), 
            size=num_samples, 
            p=area_list
        )
        
        surface_points = []
        surface_normals_list = []
        
        for face_idx in sampled_face_indices:
            face = self.mesh.faces[face_idx]
            v0, v1, v2 = self.mesh.vertices[face]
            
            # Random barycentric coordinates
            r1, r2 = np.random.rand(2)
            if r1 + r2 > 1:
                r1, r2 = 1 - r1, 1 - r2
            
            point = (1 - r1 - r2) * v0 + r1 * v1 + r2 * v2
            surface_points.append(point)
            surface_normals_list.append(face_normals[face_idx])
        
        return np.array(surface_points), np.array(surface_normals_list)


class SpatialPointSampler:
    """Sample spatial points with higher density near surface"""
    
    @staticmethod
    def sample_near_surface(
        surface_points: np.ndarray,
        num_samples: int,
        variance: Optional[float] = None,
        second_variance: Optional[float] = None
    ) -> np.ndarray:
        """
        Sample points near surface with Gaussian perturbation.
        
        Args:
            surface_points: Surface points to perturb
            num_samples: Number of spatial samples to generate
            variance: Primary variance for near-surface sampling (default from config)
            second_variance: Secondary variance for exploration (default from config * 0.1)
            
        Returns:
            Sampled spatial points
        """
        variance = variance or DEEPSDF_SETTINGS["surface_variance"]
        second_variance = second_variance or (DEEPSDF_SETTINGS["surface_variance"] / 10)
        logger.info(f"Sampling {num_samples} spatial points near surface...")
        
        spatial_points = []
        
        # Sample near surface with two different variances
        num_primary = num_samples // 2
        num_secondary = num_samples - num_primary
        
        # Primary samples - closer to surface
        for point in surface_points[:num_primary]:
            perturbation = np.random.normal(0, np.sqrt(variance), 3)
            spatial_points.append(point + perturbation)
        
        # Secondary samples - further from surface
        for point in surface_points[:num_secondary]:
            perturbation = np.random.normal(0, np.sqrt(second_variance), 3)
            spatial_points.append(point + perturbation)
        
        return np.array(spatial_points)
    
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
    
    def compute(
        self,
        spatial_points: np.ndarray,
        num_votes: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute signed distance field values.
        
        Uses majority voting by checking if points are closer or further from mesh center
        compared to their nearest surface neighbors.
        
        Args:
            spatial_points: Points at which to compute SDF
            num_votes: Number of neighbors for majority voting (default from config)
            
        Returns:
            Array of signed distance values
        """
        num_votes = num_votes or DEEPSDF_SETTINGS["num_votes"]
        logger.info("Computing signed distance field values...")
        
        # Find k-nearest neighbors
        distances, indices = self.surface_tree.query(spatial_points, k=num_votes)
        
        # Compute mesh center for inside/outside determination
        mesh_center = self.surface_points.mean(axis=0)
        
        sdf_values = []
        
        for i, point in enumerate(tqdm(spatial_points, desc="Computing SDF")):
            nearest_indices = indices[i]
            nearest_distances = distances[i]
            
            if np.isscalar(nearest_distances):
                nearest_distances = np.array([nearest_distances])
            
            # Use median distance as magnitude
            distance_mag = np.median(nearest_distances)
            
            # Majority voting for sign determination based on distance from mesh center
            # Points closer to center than their surface neighbors = inside (negative)
            # Points further from center than their surface neighbors = outside (positive)
            num_inside_votes = 0
            
            dist_to_center_point = np.linalg.norm(point - mesh_center)
            
            for nn_idx in nearest_indices:
                if nn_idx < len(self.surface_points):
                    surface_point = self.surface_points[nn_idx]
                    dist_to_center_surface = np.linalg.norm(surface_point - mesh_center)
                    
                    # If point is closer to center than surface point, it's likely inside
                    if dist_to_center_point < dist_to_center_surface:
                        num_inside_votes += 1
            
            # Assign sign based on majority vote
            # More votes for inside = negative SDF, more votes for outside = positive SDF
            sign = -1.0 if num_inside_votes > num_votes // 2 else 1.0
            sdf = sign * distance_mag
            
            sdf_values.append(sdf)
        
        return np.array(sdf_values)


class DeepSDFPreprocessor:
    """Main preprocessing pipeline for DeepSDF"""
    
    def __init__(
        self,
        num_spatial_samples: Optional[int] = None,
        surface_variance: Optional[float] = None,
        output_format: Optional[str] = None
    ):
        """
        Initialize preprocessor.
        
        Args:
            num_spatial_samples: Total spatial samples to generate (default from config)
            surface_variance: Variance for surface sampling (default from config)
            output_format: Output format 'npz' or 'npy' (default from config)
        """
        self.num_spatial_samples = num_spatial_samples or DEEPSDF_SETTINGS["num_spatial_samples"]
        self.surface_variance = surface_variance or DEEPSDF_SETTINGS["surface_variance"]
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
        
        stats = {
            'input_file': mesh_path,
            'output_file': output_path,
            'num_vertices': len(mesh.vertices),
            'num_faces': len(mesh.faces),
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
        
        near_surface_points = SpatialPointSampler.sample_near_surface(
            surface_points,
            num_near_surface,
            variance=self.surface_variance,
            second_variance=self.surface_variance / 10
        )
        
        random_points = SpatialPointSampler.sample_random_cube(num_random)
        
        # Combine all spatial points
        spatial_points = np.vstack([near_surface_points, random_points])
        logger.info(f"Total spatial points: {len(spatial_points)}")
        
        # Compute SDF values
        sdf_computer = SDFComputer(mesh, surface_points, surface_normals)
        sdf_values = sdf_computer.compute(spatial_points)
        
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
        category: Optional[str] = None
    ) -> dict:
        """
        Preprocess entire dataset.
        
        Args:
            dataset_dir: Root dataset directory
            output_dir: Output directory for processed meshes
            category: Specific category to process (optional)
            
        Returns:
            Dictionary with overall statistics
        """
        dataset_dir = Path(dataset_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_stats = []
        
        # Find all mesh files
        if category:
            category_dirs = [dataset_dir / category]
        else:
            category_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
        
        total_meshes = 0
        for category_dir in category_dirs:
            if not category_dir.is_dir():
                continue
            
            category_name = category_dir.name
            logger.info(f"\n📁 Processing category: {category_name}")
            
            # Find all model directories
            for model_dir in tqdm(sorted(category_dir.iterdir()), desc=f"Category {category_name}"):
                if not model_dir.is_dir():
                    continue
                
                model_id = model_dir.name
                models_dir = model_dir / "models"
                
                if not models_dir.exists():
                    continue
                
                # Find OBJ file
                obj_files = list(models_dir.glob("model_normalized.obj"))
                if not obj_files:
                    continue
                
                obj_file = obj_files[0]
                
                # Create output structure
                output_category_dir = output_dir / category_name / model_id
                output_file = output_category_dir / f"sdf.{self.output_format}"
                
                try:
                    stats = self.preprocess_mesh(str(obj_file), str(output_file))
                    stats['category'] = category_name
                    stats['model_id'] = model_id
                    all_stats.append(stats)
                    total_meshes += 1
                except Exception as e:
                    logger.warning(f"Failed to process {model_id}: {e}")
        
        # Summary statistics
        summary = {
            'total_meshes_processed': total_meshes,
            'output_directory': str(output_dir),
            'output_format': self.output_format,
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
        output_format="npz"
    )
    
    # Process dataset
    preprocessor.preprocess_dataset(
        dataset_dir=str(DATA_DIR),
        output_dir="./data/shapenet_sdf/",
    )


if __name__ == "__main__":
    main()
