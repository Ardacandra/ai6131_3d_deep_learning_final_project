#!/usr/bin/env python3
"""
Visualization script for DeepSDF preprocessing results.

Displays:
- 3D point clouds colored by SDF sign (positive/negative)
- SDF value distribution histograms
- Point density near/far from surface
- Statistics verification
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    logger.warning("Plotly not installed. Interactive 3D plots will use matplotlib instead.")


def load_sdf_data(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load SDF data from NPZ file.
    
    Args:
        npz_path: Path to .npz file
        
    Returns:
        pos_data: Positive SDF samples [N, 4] with columns [x, y, z, sdf]
        neg_data: Negative SDF samples [M, 4] with columns [x, y, z, sdf]
    """
    data = np.load(npz_path)
    pos_data = data['pos']
    neg_data = data['neg']
    return pos_data, neg_data


def print_statistics(pos_data: np.ndarray, neg_data: np.ndarray):
    """Print summary statistics about the SDF data."""
    print("\n" + "="*70)
    print("SDF DATA STATISTICS")
    print("="*70)
    
    print(f"\nPositive SDF (outside mesh):")
    print(f"  Samples: {len(pos_data)}")
    if len(pos_data) > 0:
        print(f"  SDF range: [{np.min(pos_data[:, 3]):.4f}, {np.max(pos_data[:, 3]):.4f}]")
        print(f"  SDF mean: {np.mean(pos_data[:, 3]):.4f}")
        print(f"  SDF std: {np.std(pos_data[:, 3]):.4f}")
    else:
        print(f"  ⚠️  No positive samples found")
    
    print(f"\nNegative SDF (inside mesh):")
    print(f"  Samples: {len(neg_data)}")
    if len(neg_data) > 0:
        print(f"  SDF range: [{np.min(neg_data[:, 3]):.4f}, {np.max(neg_data[:, 3]):.4f}]")
        print(f"  SDF mean: {np.mean(neg_data[:, 3]):.4f}")
        print(f"  SDF std: {np.std(neg_data[:, 3]):.4f}")
    else:
        print(f"  ⚠️  No negative samples found")
    
    if len(pos_data) > 0 or len(neg_data) > 0:
        all_sdf = np.concatenate([pos_data[:, 3], neg_data[:, 3]])
        print(f"\nOverall SDF distribution:")
        print(f"  Total samples: {len(all_sdf)}")
        if len(all_sdf) > 0:
            print(f"  % outside (positive): {100 * len(pos_data) / len(all_sdf):.1f}%")
            print(f"  % inside (negative): {100 * len(neg_data) / len(all_sdf):.1f}%")
        
        print("\n✅ Sign Check: Positive values should be >= 0, Negative values should be < 0")
        if len(pos_data) > 0:
            pos_valid = np.all(pos_data[:, 3] >= 0)
            print(f"  Positive SDF >= 0: {'✅ PASS' if pos_valid else '❌ FAIL'}")
        if len(neg_data) > 0:
            neg_valid = np.all(neg_data[:, 3] < 0)
            print(f"  Negative SDF < 0: {'✅ PASS' if neg_valid else '❌ FAIL'}")
    print("="*70 + "\n")


def plot_sdf_distribution(pos_data: np.ndarray, neg_data: np.ndarray):
    """Plot SDF value distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of all SDF values
    all_sdf = np.concatenate([pos_data[:, 3], neg_data[:, 3]]) if (len(pos_data) > 0 or len(neg_data) > 0) else np.array([])
    if len(all_sdf) > 0:
        axes[0].hist(all_sdf, bins=100, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='SDF=0 (surface)')
    else:
        axes[0].text(0.5, 0.5, 'No data available', ha='center', va='center', transform=axes[0].transAxes)
    axes[0].set_xlabel('SDF Value')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of SDF Values')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Separate histograms
    if len(pos_data) > 0:
        axes[1].hist(pos_data[:, 3], bins=50, alpha=0.6, label='Positive (outside)', color='green', edgecolor='black')
    if len(neg_data) > 0:
        axes[1].hist(neg_data[:, 3], bins=50, alpha=0.6, label='Negative (inside)', color='red', edgecolor='black')
    if len(pos_data) == 0 and len(neg_data) == 0:
        axes[1].text(0.5, 0.5, 'No data available', ha='center', va='center', transform=axes[1].transAxes)
    axes[1].set_xlabel('SDF Value')
    axes[1].set_ylabel('Count')
    axes[1].set_title('SDF Distribution by Sign')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_3d_scatter_matplotlib(pos_data: np.ndarray, neg_data: np.ndarray):
    """Create 3D scatter plot using matplotlib."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Sample points if too many (for performance)
    if len(pos_data) > 0:
        sample_rate = max(1, len(pos_data) // 2000)
        pos_sample = pos_data[::sample_rate]
        
        # Plot points
        scatter_pos = ax.scatter(
            pos_sample[:, 0], pos_sample[:, 1], pos_sample[:, 2],
            c=pos_sample[:, 3], cmap='Greens', s=10, alpha=0.6, label='Positive (outside)'
        )
        cbar_pos = plt.colorbar(scatter_pos, ax=ax, pad=0.1, shrink=0.8)
        cbar_pos.set_label('Positive SDF')
    
    if len(neg_data) > 0:
        sample_rate = max(1, len(neg_data) // 2000)
        neg_sample = neg_data[::sample_rate]
        scatter_neg = ax.scatter(
            neg_sample[:, 0], neg_sample[:, 1], neg_sample[:, 2],
            c=neg_sample[:, 3], cmap='Reds_r', s=10, alpha=0.6, label='Negative (inside)'
        )
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('SDF Point Cloud (sampled)')
    ax.legend()
    
    return fig


def plot_3d_scatter_plotly(pos_data: np.ndarray, neg_data: np.ndarray) -> go.Figure:
    """Create interactive 3D scatter plot using plotly."""
    traces = []
    
    # Create positive trace if data exists
    if len(pos_data) > 0:
        sample_rate = max(1, len(pos_data) // 1000)
        pos_sample = pos_data[::sample_rate]
        
        trace_pos = go.Scatter3d(
            x=pos_sample[:, 0], y=pos_sample[:, 1], z=pos_sample[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=pos_sample[:, 3],
                colorscale='Greens',
                showscale=False,
                opacity=0.6
            ),
            name='Positive (outside)',
            text=[f'SDF: {sdf:.4f}' for sdf in pos_sample[:, 3]],
            hoverinfo='text'
        )
        traces.append(trace_pos)
    
    # Create negative trace if data exists
    if len(neg_data) > 0:
        sample_rate = max(1, len(neg_data) // 1000)
        neg_sample = neg_data[::sample_rate]
        
        trace_neg = go.Scatter3d(
            x=neg_sample[:, 0], y=neg_sample[:, 1], z=neg_sample[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=neg_sample[:, 3],
                colorscale='Reds_r',
                showscale=False,
                opacity=0.6
            ),
            name='Negative (inside)',
            text=[f'SDF: {sdf:.4f}' for sdf in neg_sample[:, 3]],
            hoverinfo='text'
        )
        traces.append(trace_neg)
    
    fig = go.Figure(data=traces)
    fig.update_layout(
        title='3D SDF Point Cloud (Interactive)',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        width=1200,
        height=800
    )
    
    return fig


def plot_spatial_distribution(pos_data: np.ndarray, neg_data: np.ndarray):
    """Plot spatial distribution statistics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Distance from origin for positive and negative
    if len(pos_data) > 0:
        pos_dist = np.linalg.norm(pos_data[:, :3], axis=1)
        axes[0, 0].hist(pos_dist, bins=50, alpha=0.6, color='green', label='Positive', edgecolor='black')
    if len(neg_data) > 0:
        neg_dist = np.linalg.norm(neg_data[:, :3], axis=1)
        axes[0, 0].hist(neg_dist, bins=50, alpha=0.6, color='red', label='Negative', edgecolor='black')
    
    axes[0, 0].set_xlabel('Distance from origin')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Spatial Distribution (distance from origin)')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # XYZ coordinate distributions
    for idx, axis in enumerate(['X', 'Y', 'Z']):
        ax = axes[(idx + 1) // 2, (idx + 1) % 2]
        if len(pos_data) > 0:
            ax.hist(pos_data[:, idx], bins=50, alpha=0.5, color='green', label='Positive', edgecolor='black')
        if len(neg_data) > 0:
            ax.hist(neg_data[:, idx], bins=50, alpha=0.5, color='red', label='Negative', edgecolor='black')
        ax.set_xlabel(f'{axis} coordinate')
        ax.set_ylabel('Count')
        ax.set_title(f'{axis} Coordinate Distribution')
        ax.legend()
        ax.grid(alpha=0.3)
    
    # Remove the 4th subplot
    fig.delaxes(axes[1, 1])
    
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Visualize DeepSDF preprocessing results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize a specific mesh
  python visualize_sdf.py data/shapenet_sdf/02747177/24884ef01e9a3832d2b12aa6a0f050b3/sdf.npz
  
  # Interactive viewing (requires plotly)
  python visualize_sdf.py data/shapenet_sdf/02747177/24884ef01e9a3832d2b12aa6a0f050b3/sdf.npz --interactive
  
  # Save plots to disk
  python visualize_sdf.py data/shapenet_sdf/02747177/24884ef01e9a3832d2b12aa6a0f050b3/sdf.npz --save ./output/
        """
    )
    
    parser.add_argument(
        'npz_file',
        help='Path to NPZ file containing SDF data'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Use interactive 3D plots (requires plotly)'
    )
    
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Save plots to directory instead of displaying'
    )
    
    args = parser.parse_args()
    
    # Validate file
    npz_path = Path(args.npz_file)
    if not npz_path.exists():
        logger.error(f"File not found: {npz_path}")
        return
    
    logger.info(f"Loading SDF data from {npz_path}...")
    pos_data, neg_data = load_sdf_data(str(npz_path))
    
    # Print statistics
    print_statistics(pos_data, neg_data)
    
    # Create output directory if saving
    if args.save:
        output_dir = Path(args.save)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: SDF Distribution
    logger.info("Creating SDF distribution plots...")
    fig1 = plot_sdf_distribution(pos_data, neg_data)
    if args.save:
        fig1.savefig(Path(args.save) / 'sdf_distribution.png', dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {Path(args.save) / 'sdf_distribution.png'}")
    else:
        plt.show()
    
    # Plot 2: 3D Scatter
    logger.info("Creating 3D point cloud visualization...")
    if args.interactive and HAS_PLOTLY:
        fig2 = plot_3d_scatter_plotly(pos_data, neg_data)
        if args.save:
            fig2.write_html(Path(args.save) / '3d_scatter_interactive.html')
            logger.info(f"Saved: {Path(args.save) / '3d_scatter_interactive.html'}")
        else:
            fig2.show()
    else:
        fig2 = plot_3d_scatter_matplotlib(pos_data, neg_data)
        if args.save:
            fig2.savefig(Path(args.save) / '3d_scatter.png', dpi=150, bbox_inches='tight')
            logger.info(f"Saved: {Path(args.save) / '3d_scatter.png'}")
        else:
            plt.show()
    
    # Plot 3: Spatial Distribution
    logger.info("Creating spatial distribution plots...")
    fig3 = plot_spatial_distribution(pos_data, neg_data)
    if args.save:
        fig3.savefig(Path(args.save) / 'spatial_distribution.png', dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {Path(args.save) / 'spatial_distribution.png'}")
    else:
        plt.show()
    
    logger.info("✅ Visualization complete!")


if __name__ == '__main__':
    main()
