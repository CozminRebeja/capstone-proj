"""Package-anchored paths for assets and sample data."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets"
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "output"


def ensure_output_dir(*parts):
    """Return the root output directory, creating it when needed."""
    output_dir = OUTPUT_DIR.joinpath(*parts)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def resolve_output_path(path):
    """Resolve a caller-provided output path under the root output directory."""
    output_path = Path(path).expanduser()
    if output_path.is_absolute():
        return output_path.resolve()
    if output_path.parts and output_path.parts[0] == OUTPUT_DIR.name:
        return (REPO_ROOT / output_path).resolve()
    return (OUTPUT_DIR / output_path).resolve()


def resolve_output_dir(path):
    """Resolve and create a caller-provided output directory."""
    output_dir = resolve_output_path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
