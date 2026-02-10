#!/usr/bin/env python3
"""
Convenient CLI script to run DeepSDF preprocessing on the ShapeNet dataset.

Usage:
    python prepare_deepsdf.py [--category CATEGORY] [--output OUTPUT_DIR] [--num-samples NUM]
"""

import argparse
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing import DeepSDFPreprocessor
from config import DATA_DIR, SHAPENET_CATEGORIES, DEEPSDF_SETTINGS


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess ShapeNet dataset for DeepSDF training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all categories
  python prepare_deepsdf.py --output ./data/shapenet_sdf/
  
  # Process specific category
  python prepare_deepsdf.py --category Airplane --output ./data/shapenet_sdf/
  
  # Custom number of samples
  python prepare_deepsdf.py --num-samples 250000 --output ./data/shapenet_sdf/
        """
    )
    
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help=f"Specific category to process. Options: {', '.join(SHAPENET_CATEGORIES.values())}"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEEPSDF_SETTINGS["output_dir"]),
        help=f"Output directory for SDF data (default: {DEEPSDF_SETTINGS['output_dir']})"
    )
    
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEEPSDF_SETTINGS["num_spatial_samples"],
        help=f"Number of spatial samples per mesh (default: {DEEPSDF_SETTINGS['num_spatial_samples']})"
    )
    
    parser.add_argument(
        "--variance",
        type=float,
        default=DEEPSDF_SETTINGS["surface_variance"],
        help=f"Variance for near-surface sampling (default: {DEEPSDF_SETTINGS['surface_variance']})"
    )
    
    parser.add_argument(
        "--format",
        type=str,
        choices=["npz", "npy"],
        default=DEEPSDF_SETTINGS["output_format"],
        help=f"Output format (default: {DEEPSDF_SETTINGS['output_format']})"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # Validate category if specified
    if args.category:
        category_id = None
        for cid, cname in SHAPENET_CATEGORIES.items():
            if cname.lower() == args.category.lower():
                category_id = cid
                break
        
        if category_id is None:
            logger.error(f"Unknown category: {args.category}")
            logger.info(f"Available categories: {', '.join(SHAPENET_CATEGORIES.values())}")
            sys.exit(1)
        
        category = category_id
    else:
        category = None
    
    # Create preprocessor
    logger.info("=" * 70)
    logger.info("DeepSDF Data Preprocessing Pipeline")
    logger.info("=" * 70)
    logger.info(f"Dataset directory: {DATA_DIR}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Samples per mesh: {args.num_samples}")
    logger.info(f"Category: {args.category if args.category else 'All'}")
    logger.info(f"Output format: {args.format}")
    logger.info("=" * 70 + "\n")
    
    preprocessor = DeepSDFPreprocessor(
        num_spatial_samples=args.num_samples,
        surface_variance=args.variance,
        output_format=args.format
    )
    
    # Run preprocessing
    try:
        stats = preprocessor.preprocess_dataset(
            dataset_dir=str(DATA_DIR),
            output_dir=args.output,
            category=category
        )
        
        logger.info("\n✅ Preprocessing completed successfully!")
        logger.info(f"Total meshes processed: {stats['total_meshes_processed']}")
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Preprocessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
