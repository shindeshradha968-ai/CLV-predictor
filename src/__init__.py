"""
CLV Predictor - Customer Lifetime Value Prediction Package

This package provides tools for predicting customer lifetime value
using machine learning algorithms including Linear Regression,
Random Forest, and XGBoost.
"""

__version__ = "1.0.0"
__author__ = "CLV Predictor Team"

from pathlib import Path

# Define project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIG_DIR = PROJECT_ROOT / "src" / "config"

# Ensure directories exist
for directory in [DATA_DIR / "raw", DATA_DIR / "processed", DATA_DIR / "external", MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)