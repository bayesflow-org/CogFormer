# CogFormer
Generalized framework for multi-model amortized Bayesian inference

## Install

### Using pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Using uv

```bash
uv venv
uv sync
```

## Install for Devs

### Using pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev,test]
pre-commit install
```

### Using uv
```bash
uv venv
uv sync --extra dev --extra test
source .venv/bin/activate
pre-commit install
```
