"""Package-anchored paths for assets and sample data."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets"
DATA_DIR = REPO_ROOT / "data"
