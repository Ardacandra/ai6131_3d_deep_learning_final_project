"""
Visualization and exploration script for ShapeNet 3D dataset.
Explores the dataset structure and visualizes sample 3D objects.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import glob
from typing import Dict, List, Tuple
import struct

# Import reusable utilities and configuration
from src.shapenet_utils import load_obj, load_binvox, compute_voxel_statistics
from config import (
    DATA_DIR,
    OUTPUT_DIR,
    SHAPENET_CATEGORIES,
    VISUALIZATION_SETTINGS,
    FILE_FORMATS,
)


class ShapeNetExplorer:
    """Explore and visualize ShapeNet dataset"""
    
    def __init__(self, data_dir: Path = None, output_dir: Path = None):
        self.data_dir = data_dir or DATA_DIR
        self.output_dir = output_dir or (OUTPUT_DIR / "viz")
        self.category_stats = {}
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {self.output_dir.resolve()}")
        
    def scan_dataset(self) -> Dict:
        """Scan the dataset and collect statistics"""
        print("=" * 60)
        print("SCANNING SHAPENET DATASET")
        print("=" * 60)
        
        stats = {}
        
        for category_id in sorted(os.listdir(self.data_dir)):
            category_path = self.data_dir / category_id
            
            if not category_path.is_dir() or category_id.startswith('.'):
                continue
            
            category_name = SHAPENET_CATEGORIES.get(category_id, "Unknown")
            model_dirs = [d for d in os.listdir(category_path) 
                         if (category_path / d).is_dir() and not d.startswith('.')]
            
            print(f"\n📁 Category: {category_name} (ID: {category_id})")
            print(f"   Number of models: {len(model_dirs)}")
            
            # Analyze available file types
            file_types = {}
            total_size = 0
            
            for model_id in model_dirs[:5]:  # Sample first 5 models
                model_path = category_path / model_id / "models"
                if model_path.exists():
                    for file in os.listdir(model_path):
                        ext = Path(file).suffix
                        if ext:
                            file_types[ext] = file_types.get(ext, 0) + 1
                            total_size += (model_path / file).stat().st_size
            
            print(f"   Sample file types found:")
            for ext, count in sorted(file_types.items()):
                print(f"      - {ext}: {count} files")
            
            # Sample first model details
            if model_dirs:
                sample_model = model_dirs[0]
                model_path = category_path / sample_model / "models"
                print(f"\n   Sample model: {sample_model}")
                if model_path.exists():
                    files = os.listdir(model_path)
                    print(f"   Files: {', '.join(sorted(files))}")
            
            stats[category_id] = {
                'name': category_name,
                'num_models': len(model_dirs),
                'model_dirs': model_dirs
            }
        
        self.category_stats = stats
        return stats
    
    def visualize_models(self, num_samples: int = 4):
        """Visualize sample models from each category"""
        print("\n" + "=" * 60)
        print("VISUALIZING SAMPLE MODELS")
        print("=" * 60)
        
        for category_id, stats in self.category_stats.items():
            category_name = stats['name']
            model_dirs = stats['model_dirs']
            
            print(f"\n📊 Visualizing {category_name} models...")
            
            # Select sample models
            sample_indices = np.linspace(0, len(model_dirs) - 1, 
                                        min(num_samples, len(model_dirs)), 
                                        dtype=int)
            
            fig = plt.figure(figsize=(16, 12))
            fig.suptitle(f'ShapeNet {category_name} - Sample Models', 
                        fontsize=16, fontweight='bold')
            
            for plot_idx, model_idx in enumerate(sample_indices):
                model_id = model_dirs[int(model_idx)]
                model_path = self.data_dir / category_id / model_id / "models"
                obj_file = model_path / "model_normalized.obj"
                
                if not obj_file.exists():
                    continue
                
                print(f"  Loading model {plot_idx + 1}/{len(sample_indices)}: {model_id}")
                
                # Load vertices
                vertices, faces = load_obj(str(obj_file))
                
                # Create subplot
                ax = fig.add_subplot(2, 2, plot_idx + 1, projection='3d')
                
                # Plot vertices
                if len(vertices) > 0:
                    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                             c='steelblue', marker='o', s=1, alpha=0.6)
                    
                    # Plot edges (sample for visualization clarity)
                    if len(faces) > 0:
                        sample_faces = np.random.choice(len(faces), 
                                                       min(500, len(faces)), 
                                                       replace=False)
                        for face_idx in sample_faces:
                            face = faces[face_idx]
                            # Draw edges of the face
                            for i in range(len(face)):
                                v1 = vertices[face[i]]
                                v2 = vertices[face[(i + 1) % len(face)]]
                                ax.plot([v1[0], v2[0]], [v1[1], v2[1]], 
                                       [v1[2], v2[2]], 'b-', alpha=0.3, linewidth=0.5)
                    
                    # Set labels and title
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    ax.set_title(f'Model {plot_idx + 1}\n{model_id[:16]}...')
                    
                    # Auto scale
                    max_range = np.array([vertices[:, 0].max() - vertices[:, 0].min(),
                                        vertices[:, 1].max() - vertices[:, 1].min(),
                                        vertices[:, 2].max() - vertices[:, 2].min()]).max() / 2.0
                    mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
                    mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
                    mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5
                    
                    ax.set_xlim(mid_x - max_range, mid_x + max_range)
                    ax.set_ylim(mid_y - max_range, mid_y + max_range)
                    ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            plt.tight_layout()
            output_file = self.output_dir / f"visualization_{category_name.lower()}_models.png"
            plt.savefig(output_file, dpi=100, bbox_inches='tight')
            print(f"✅ Visualization saved as: {output_file}")
            plt.close()
    
    def visualize_voxels(self, num_samples: int = 2):
        """Visualize voxel representations"""
        print("\n" + "=" * 60)
        print("VISUALIZING VOXEL REPRESENTATIONS")
        print("=" * 60)
        
        for category_id, stats in self.category_stats.items():
            category_name = stats['name']
            model_dirs = stats['model_dirs']
            
            print(f"\n🎲 Visualizing voxel data for {category_name}...")
            
            # Select sample models
            sample_indices = np.linspace(0, len(model_dirs) - 1, 
                                        min(num_samples, len(model_dirs)), 
                                        dtype=int)
            
            fig = plt.figure(figsize=(14, 6))
            fig.suptitle(f'ShapeNet {category_name} - Voxel Representations', 
                        fontsize=16, fontweight='bold')
            
            for plot_idx, model_idx in enumerate(sample_indices):
                model_id = model_dirs[int(model_idx)]
                model_path = self.data_dir / category_id / model_id / "models"
                
                # Try to load surface voxels
                binvox_file = model_path / "model_normalized.surface.binvox"
                
                if binvox_file.exists():
                    print(f"  Loading voxel model {plot_idx + 1}/{len(sample_indices)}: {model_id}")
                    
                    voxel_data = load_binvox(str(binvox_file))
                    
                    # Downsample for visualization
                    step = max(1, voxel_data.shape[0] // 32)
                    voxel_data_sampled = voxel_data[::step, ::step, ::step]
                    
                    ax = fig.add_subplot(1, 2, plot_idx + 1, projection='3d')
                    
                    # Get indices of filled voxels
                    filled_voxels = np.where(voxel_data_sampled > 0)
                    ax.scatter(filled_voxels[0], filled_voxels[1], filled_voxels[2],
                             c='coral', marker='s', s=20, alpha=0.8)
                    
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    ax.set_title(f'Voxel Model {plot_idx + 1}\n{model_id[:16]}...')
                    ax.set_box_aspect([1, 1, 1])
            
            plt.tight_layout()
            output_file = self.output_dir / f"visualization_{category_name.lower()}_voxels.png"
            plt.savefig(output_file, dpi=100, bbox_inches='tight')
            print(f"✅ Voxel visualization saved as: {output_file}")
            plt.close()
    
    def print_summary(self):
        """Print dataset summary"""
        print("\n" + "=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)
        
        total_models = sum(stats['num_models'] for stats in self.category_stats.values())
        print(f"\nTotal categories: {len(self.category_stats)}")
        print(f"Total models: {total_models}")
        
        for category_id, stats in self.category_stats.items():
            print(f"\n  • {stats['name']} ({category_id}): {stats['num_models']} models")
        
        print("\n📝 Data Representation Formats:")
        print("  • OBJ files: 3D mesh with vertices and faces")
        print("  • MTL files: Material definitions for OBJ")
        print("  • BinVOX files: Binary voxel grid format")
        print("    - surface.binvox: Surface voxel representation")
        print("    - solid.binvox: Solid voxel representation")
        print("  • JSON files: Metadata for each model")
        print("\n" + "=" * 60 + "\n")


def main():
    """Main execution"""
    # Initialize explorer with settings from config
    # You can override with: ShapeNetExplorer(data_dir=Path("./custom_data/"))
    explorer = ShapeNetExplorer()
    
    print(f"📂 Data source: {explorer.data_dir.resolve()}")
    print(f"💾 Output directory: {explorer.output_dir.resolve()}\n")
    
    # Scan and analyze dataset
    explorer.scan_dataset()
    
    # Print summary
    explorer.print_summary()
    
    # Visualize mesh models
    num_samples_models = VISUALIZATION_SETTINGS["sample_models_per_category"]
    print(f"💾 Generating {num_samples_models} 3D mesh visualizations per category...")
    explorer.visualize_models(num_samples=num_samples_models)
    
    # Visualize voxel representations
    num_samples_voxels = VISUALIZATION_SETTINGS["sample_voxels_per_category"]
    print(f"💾 Generating {num_samples_voxels} voxel grid visualizations per category...")
    explorer.visualize_voxels(num_samples=num_samples_voxels)
    
    print("\n✨ Visualization complete! Check the generated PNG files.")


if __name__ == "__main__":
    main()
