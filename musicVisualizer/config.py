"""
config.py - Configuration Management for musicVisualizer

Loads configuration from .env file with sensible defaults.
Uses python-dotenv for environment variable loading.

Usage:
    import config

    # Access configuration values
    source_img = config.TEMPLATE_IMAGE
    target_img = config.PALETTE_IMAGE

    # Get dynamic paths
    model_path = config.get_model_path()  # e.g., 'models/spiderman_hybridTheory.pth'
    session = config.get_session_name('20260205_120000')  # e.g., 'spiderman_hybridTheory_20260205_120000'
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
_project_root = Path(__file__).parent
load_dotenv(_project_root / '.env')


# =============================================================================
# IMAGE PATHS
# =============================================================================

# Source image for triangulation (template)
TEMPLATE_IMAGE = os.getenv('TEMPLATE_IMAGE')

# Target image for color extraction (palette)
PALETTE_IMAGE = os.getenv('PALETTE_IMAGE')


# =============================================================================
# CLUSTERING PARAMETERS
# =============================================================================

# Number of K-means clusters for initial color extraction
NUM_CLUSTERS = int(os.getenv('NUM_CLUSTERS'))

# Number of distinct colors to select using LAB algorithm
NUM_DISTINCT = int(os.getenv('NUM_DISTINCT'))

# Density reduction parameter (higher = fewer/larger triangles)
DENSITY_REDUCTION = int(os.getenv('DENSITY_REDUCTION', '60'))


# =============================================================================
# DIRECTORY PATHS
# =============================================================================

MODELS_DIR = os.getenv('MODELS_DIR', 'models')
FEEDBACK_DIR = os.getenv('FEEDBACK_DIR', 'feedback_data')
TENSORBOARD_DIR = os.getenv('TENSORBOARD_DIR', 'runs')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', 'triangulatedImages')


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_image_basename(image_path):
    """
    Extract basename without extension from image path.

    Args:
        image_path: Path to image file

    Returns:
        str: Filename without extension

    Example:
        get_image_basename('originalImages/spiderman.jpg') -> 'spiderman'
        get_image_basename('hybridTheory.jpeg') -> 'hybridTheory'
    """
    return Path(image_path).stem


def get_model_name():
    """
    Generate model name based on template and palette basenames.

    Returns:
        str: Model name in format '{template}_{palette}'

    Example:
        If TEMPLATE_IMAGE='originalImages/spiderman.jpg' and
        PALETTE_IMAGE='hybridTheory.jpeg', returns 'spiderman_hybridTheory'
    """
    template_base = get_image_basename(TEMPLATE_IMAGE)
    palette_base = get_image_basename(PALETTE_IMAGE)
    return f'{template_base}_{palette_base}'


def get_model_path():
    """
    Generate full model path based on template and palette basenames.

    Returns:
        str: Full path like 'models/spiderman_hybridTheory.pth'
    """
    return f'{MODELS_DIR}/{get_model_name()}.pth'


def get_session_name(timestamp_str):
    """
    Generate session name based on template, palette, and timestamp.

    Args:
        timestamp_str: Timestamp string (e.g., '20260205_120000')

    Returns:
        str: Session name like 'spiderman_hybridTheory_20260205_120000'
    """
    return f'{get_model_name()}_{timestamp_str}'


def validate_config():
    """
    Validate that configured image paths exist.

    Returns:
        list: List of error messages (empty if all valid)
    """
    errors = []

    if not Path(TEMPLATE_IMAGE).exists():
        errors.append(f"TEMPLATE_IMAGE not found: {TEMPLATE_IMAGE}")
    if not Path(PALETTE_IMAGE).exists():
        errors.append(f"PALETTE_IMAGE not found: {PALETTE_IMAGE}")

    # Validate numeric ranges
    if NUM_CLUSTERS < 2:
        errors.append(f"NUM_CLUSTERS must be >= 2, got {NUM_CLUSTERS}")
    if NUM_DISTINCT < 1:
        errors.append(f"NUM_DISTINCT must be >= 1, got {NUM_DISTINCT}")
    if NUM_DISTINCT > NUM_CLUSTERS:
        errors.append(f"NUM_DISTINCT ({NUM_DISTINCT}) cannot exceed NUM_CLUSTERS ({NUM_CLUSTERS})")
    if DENSITY_REDUCTION < 1:
        errors.append(f"DENSITY_REDUCTION must be >= 1, got {DENSITY_REDUCTION}")

    return errors


def print_config():
    """Print current configuration for debugging."""
    print("=" * 60)
    print("musicVisualizer Configuration")
    print("=" * 60)
    print(f"  TEMPLATE_IMAGE:     {TEMPLATE_IMAGE}")
    print(f"  PALETTE_IMAGE:      {PALETTE_IMAGE}")
    print(f"  NUM_CLUSTERS:       {NUM_CLUSTERS}")
    print(f"  NUM_DISTINCT:       {NUM_DISTINCT}")
    print(f"  DENSITY_REDUCTION:  {DENSITY_REDUCTION}")
    print("-" * 60)
    print(f"  Model Name:      {get_model_name()}")
    print(f"  Model Path:      {get_model_path()}")
    print("=" * 60)


if __name__ == '__main__':
    print_config()
    errors = validate_config()
    if errors:
        print("\nConfiguration Errors:")
        for err in errors:
            print(f"  - {err}")
    else:
        print("\nConfiguration is valid!")
