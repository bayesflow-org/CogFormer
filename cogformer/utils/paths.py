"""Central source of truth for where generated artifacts (results) are stored.

All experiment outputs live under a single top-level ``outputs/`` tree, organized by
artifact type. Paths are resolved relative to the repository root (derived from this
file's location), so they are independent of the current working directory.

Layout::

    outputs/
    ├── figures/      # .pdf/.png plots
    ├── metrics/      # per-cell metric .csv (mirrors the figures/ layout)
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
METRICS = OUTPUTS / "metrics"
CHECKPOINTS = OUTPUTS / "checkpoints"
DATA = OUTPUTS / "data"
TABLES = OUTPUTS / "tables"
DOCS = OUTPUTS / "docs"

# Weights & Biases run logs live in ``<repo>/wandb``. wandb appends a ``wandb/``
# subfolder to whatever is passed as ``dir=``, so the value handed to ``wandb.init``
# is the repo root itself (see ``wandb_init_dir``). Anchoring here keeps every run
# in one place regardless of the current working directory.
WANDB = REPO_ROOT / "wandb"


def wandb_init_dir() -> str:
    """Value to pass as ``dir=`` to ``wandb.init`` so runs land in ``<repo>/wandb``.

    wandb writes runs to ``<dir>/wandb/run-*``, appending the ``wandb/`` subfolder
    itself, so this returns the repo root rather than ``WANDB``. Using it makes the
    run location independent of where the experiment script is launched from.
    """
    return str(REPO_ROOT)


def _build(root: Path, parts, make: bool) -> Path:
    p = root.joinpath(*[str(x) for x in parts]) if parts else root
    if make:
        p.mkdir(parents=True, exist_ok=True)
    return p


def figures_dir(*parts, make: bool = False) -> Path:
    """Path under ``outputs/figures`` (e.g. ``figures_dir('model_family', 'cf', 'ddm')``)."""
    return _build(FIGURES, parts, make)


def metrics_dir(*parts, make: bool = False) -> Path:
    """Path under ``outputs/metrics`` (per-cell metric CSVs, mirrors ``figures/``)."""
    return _build(METRICS, parts, make)


def metrics_mirror(path, make: bool = False) -> Path:
    """Map a path under ``outputs/figures`` to the parallel ``outputs/metrics`` tree.

    Metric CSVs live in their own tree that mirrors the figures layout, keeping
    figure directories free of data files. Accepts either a directory or a file
    path; anything not under ``outputs/figures`` is returned unchanged. When
    ``make`` is set, the resolved directory (the path itself, or the parent for a
    file path) is created.
    """
    path = Path(path)
    try:
        rel = path.relative_to(FIGURES)
    except ValueError:
        return path
    mapped = METRICS / rel
    if make:
        (mapped.parent if mapped.suffix else mapped).mkdir(parents=True, exist_ok=True)
    return mapped


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
