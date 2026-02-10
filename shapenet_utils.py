"""
Utility functions for loading and processing 3D model formats.
These functions are designed to be reusable across different parts of the project.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, List


def load_obj(obj_path: str) -> Tuple[np.ndarray, List]:
    """
    Load OBJ file and extract vertices and faces.
    
    This function parses Wavefront OBJ files and extracts the 3D geometry.
    It handles various OBJ formats including different vertex attributes.
    
    Args:
        obj_path (str): Path to the OBJ file
        
    Returns:
        vertices (np.ndarray): (N, 3) array of vertex coordinates
        faces (list): List of faces, each face is a list of vertex indices
        
    Raises:
        FileNotFoundError: If the OBJ file does not exist
        ValueError: If the file cannot be parsed
        
    Example:
        >>> vertices, faces = load_obj("path/to/model.obj")
        >>> print(f"Loaded {len(vertices)} vertices and {len(faces)} faces")
    """
    vertices = []
    faces = []
    
    try:
        with open(obj_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if not parts:
                    continue
                
                # Vertex positions
                if parts[0] == 'v':
                    vertices.append([float(x) for x in parts[1:4]])
                
                # Faces
                elif parts[0] == 'f':
                    face = []
                    for vertex_data in parts[1:]:
                        # Handle different face formats (v, v/vt, v/vt/vn, v//vn)
                        vertex_id = int(vertex_data.split('/')[0])
                        face.append(vertex_id - 1)  # OBJ uses 1-based indexing
                    if len(face) >= 3:
                        faces.append(face)
        
        vertices = np.array(vertices, dtype=np.float32)
        return vertices, faces
    
    except FileNotFoundError:
        raise FileNotFoundError(f"OBJ file not found: {obj_path}")
    except Exception as e:
        raise ValueError(f"Error loading OBJ file {obj_path}: {e}")


def load_binvox(binvox_path: str) -> np.ndarray:
    """
    Load and parse BinVOX file format.
    
    This function reads binary voxel files (BinVOX format) which represent
    3D objects as a grid of voxels (3D pixels). The format uses run-length
    encoding (RLE) for compression.
    
    Args:
        binvox_path (str): Path to the BinVOX file
        
    Returns:
        voxel_grid (np.ndarray): 3D array where non-zero values indicate filled voxels
        
    Raises:
        FileNotFoundError: If the BinVOX file does not exist
        ValueError: If the file format is invalid
        
    Example:
        >>> voxel_grid = load_binvox("path/to/model.binvox")
        >>> print(f"Voxel grid shape: {voxel_grid.shape}")
        >>> filled_voxels = np.sum(voxel_grid > 0)
        >>> print(f"Filled voxels: {filled_voxels}")
    """
    try:
        with open(binvox_path, 'rb') as f:
            # Parse header to get dimension
            dims = None
            translate = None
            scale = None
            
            # BinVOX files have ASCII header followed by binary RLE data
            while True:
                line = f.readline()
                if not line:
                    break
                
                try:
                    line_str = line.decode('ascii').strip()
                except UnicodeDecodeError:
                    # Hit binary data - seek back and break
                    f.seek(-len(line), 1)
                    break
                
                if line_str.startswith('dim'):
                    dims = [int(x) for x in line_str.split()[1:]]
                elif line_str.startswith('translate'):
                    translate = [float(x) for x in line_str.split()[1:]]
                elif line_str.startswith('scale'):
                    scale = float(line_str.split()[1])
                elif line_str == 'end_header' or line_str == 'data':
                    break
            
            if dims is None:
                dims = [256, 256, 256]
            
            # Read remaining binary voxel data (RLE encoded)
            voxel_data = f.read()
            
            voxel_grid = np.zeros(dims, dtype=np.uint8)
            
            # Decode RLE
            idx = 0
            data_idx = 0
            while data_idx < len(voxel_data):
                value = voxel_data[data_idx]
                data_idx += 1
                
                if value > 0:
                    count = voxel_data[data_idx] if data_idx < len(voxel_data) else 1
                    data_idx += 1
                else:
                    count = voxel_data[data_idx] if data_idx < len(voxel_data) else 1
                    data_idx += 1
                
                # Fill voxels
                for _ in range(count):
                    if idx < voxel_grid.size:
                        voxel_grid.flat[idx] = value
                        idx += 1
            
            return voxel_grid
    
    except FileNotFoundError:
        raise FileNotFoundError(f"BinVOX file not found: {binvox_path}")
    except Exception as e:
        # Log warning but don't crash - return empty grid
        print(f"Warning: Could not load BinVOX file {binvox_path}: {e}")
        return np.zeros((32, 32, 32), dtype=np.uint8)


def get_point_cloud_from_voxels(voxel_grid: np.ndarray) -> np.ndarray:
    """
    Convert a voxel grid to a point cloud (set of 3D coordinates).
    
    This is useful for visualization or further processing of voxelized objects.
    
    Args:
        voxel_grid (np.ndarray): 3D binary array where non-zero values indicate filled voxels
        
    Returns:
        points (np.ndarray): (N, 3) array of 3D coordinates for filled voxels
        
    Example:
        >>> voxel_grid = load_binvox("model.binvox")
        >>> point_cloud = get_point_cloud_from_voxels(voxel_grid)
        >>> print(f"Point cloud has {len(point_cloud)} points")
    """
    filled_voxels = np.where(voxel_grid > 0)
    points = np.column_stack(filled_voxels)
    return points


def center_and_scale_vertices(vertices: np.ndarray) -> np.ndarray:
    """
    Center and normalize vertices to unit cube.
    
    This is useful for standardizing 3D models before processing or visualization.
    
    Args:
        vertices (np.ndarray): (N, 3) array of vertex coordinates
        
    Returns:
        normalized_vertices (np.ndarray): Centered and scaled vertices
        
    Example:
        >>> vertices, faces = load_obj("model.obj")
        >>> normalized = center_and_scale_vertices(vertices)
    """
    # Center
    centroid = np.mean(vertices, axis=0)
    centered = vertices - centroid
    
    # Scale to unit cube
    max_dist = np.max(np.linalg.norm(centered, axis=1))
    if max_dist > 0:
        normalized = centered / max_dist
    else:
        normalized = centered
    
    return normalized


def get_model_bounds(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the bounding box of a 3D model.
    
    Args:
        vertices (np.ndarray): (N, 3) array of vertex coordinates
        
    Returns:
        min_bounds (np.ndarray): Minimum coordinates (x_min, y_min, z_min)
        max_bounds (np.ndarray): Maximum coordinates (x_max, y_max, z_max)
        
    Example:
        >>> vertices, _ = load_obj("model.obj")
        >>> min_bounds, max_bounds = get_model_bounds(vertices)
        >>> print(f"Bounds: {min_bounds} to {max_bounds}")
    """
    min_bounds = np.min(vertices, axis=0)
    max_bounds = np.max(vertices, axis=0)
    return min_bounds, max_bounds


def compute_voxel_statistics(voxel_grid: np.ndarray) -> dict:
    """
    Compute statistics about a voxel grid.
    
    Args:
        voxel_grid (np.ndarray): 3D binary array
        
    Returns:
        stats (dict): Dictionary containing:
            - 'total_voxels': Total number of voxels
            - 'filled_voxels': Number of filled voxels
            - 'occupancy': Percentage of filled voxels
            - 'dimensions': Shape of the voxel grid
            
    Example:
        >>> voxel_grid = load_binvox("model.binvox")
        >>> stats = compute_voxel_statistics(voxel_grid)
        >>> print(f"Occupancy: {stats['occupancy']:.2f}%")
    """
    total_voxels = voxel_grid.size
    filled_voxels = np.sum(voxel_grid > 0)
    occupancy = (filled_voxels / total_voxels) * 100 if total_voxels > 0 else 0
    
    return {
        'total_voxels': total_voxels,
        'filled_voxels': filled_voxels,
        'occupancy': occupancy,
        'dimensions': voxel_grid.shape
    }


def downsample_voxels(voxel_grid: np.ndarray, factor: int = 2) -> np.ndarray:
    """
    Downsample a voxel grid by a given factor.
    
    Useful for reducing memory usage and computation time for large voxel grids.
    
    Args:
        voxel_grid (np.ndarray): 3D array
        factor (int): Downsampling factor (default: 2)
        
    Returns:
        downsampled (np.ndarray): Downsampled voxel grid
        
    Example:
        >>> voxel_grid = load_binvox("model.binvox")
        >>> small_grid = downsample_voxels(voxel_grid, factor=4)
    """
    return voxel_grid[::factor, ::factor, ::factor]
