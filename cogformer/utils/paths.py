"""Central source of truth for where generated artifacts (results) are stored.

All experiment outputs live under a single top-level ``outputs/`` tree, organized by
artifact type. Paths are resolved relative to the repository root (derived from this
file's location), so they are independent of the current working directory.

Layout::

    outputs/
    ├── figures/      # .pdf/.png (+ co-located per-cell metric .csv)
    ├── checkpoints/  # model weights (.pt / keras dirs)
    ├── data/         # raw prediction arrays (.npz)
    ├── tables/       # publication tables (.tex + summary .csv)
    └── docs/         # architecture diagrams, notation, algorithm tables

Use the helpers (``figures_dir``, ``checkpoints_dir`` …) to build sub-paths, e.g.::

    figures_dir("model_family", "cf", "ddm", "interaction", make=True)
    checkpoints_dir("model_class")
"""

from pathlib import Path

# cogformer/utils/paths.py -> parents[0]=utils, parents[1]=cogformer, parents[2]=repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = REPO_ROOT / "outputs"

FIGURES = OUTPUTS / "figures"
CHECKPOINTS = OUTPUTS / "checkpoints"
DATA = OUTPUTS / "data"
TABLES = OUTPUTS / "tables"
DOCS = OUTPUTS / "docs"


def _build(root: Path, parts, make: bool) -> Path:
    p = root.joinpath(*[str(x) for x in parts]) if parts else root
    if make:
        p.mkdir(parents=True, exist_ok=True)
    return p


def figures_dir(*parts, make: bool = False) -> Path:
    """Path under ``outputs/figures`` (e.g. ``figures_dir('model_family', 'cf', 'ddm')``)."""
    return _build(FIGURES, parts, make)


def checkpoints_dir(*parts, make: bool = False) -> Path:
    """Path under ``outputs/checkpoints`` (e.g. ``checkpoints_dir('model_family', 'ddm')``)."""
    return _build(CHECKPOINTS, parts, make)


def data_dir(*parts, make: bool = False) -> Path:
    """Path under ``outputs/data`` (raw prediction arrays, e.g. ``data_dir('predictions')``)."""
    return _build(DATA, parts, make)


def tables_dir(*parts, make: bool = False) -> Path:
    """Path under ``outputs/tables`` (LaTeX tables + summary CSVs)."""
    return _build(TABLES, parts, make)


def docs_dir(*parts, make: bool = False) -> Path:
    """Path under ``outputs/docs`` (architecture diagrams, notation, algorithms)."""
    return _build(DOCS, parts, make)
