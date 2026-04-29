# AGENTS.md

## Project

Python 3.11 ML research repo (SIGGRAPH 2026). Package manager: `uv`. No CI. No existing test suite despite pytest being installed.

## Setup

```sh
uv sync --frozen --all-groups   # installs base + training + dev + pytorch groups
```

Create a `.env` file at repo root — required before running any script:
```sh
MIB_POCOYO_DATASET_DIR=data/dataset
MIB_POCOYO_EXPERIMENT_DIR=data/exp
# Optional:
MLFLOW_TRACKING_URI=https://your-mlflow-server-uri
HF_TOKEN=your_hf_token   # only needed for private HF datasets
```

`pydantic-settings` reads vars with prefix `mib_{character_name}_`. Missing `.env` causes `ValidationError` at startup.

## Key Commands

| Task | Command |
|---|---|
| Lint + format (pre-commit) | `uv run pre-commit run` |
| Lint only | `uv run ruff check .` |
| Format only | `uv run ruff format .` |
| Run tests | `uv run pytest` |
| Train | `uv run python -m motion_inbetweening.scripts.train pocoyo --training-type best` |
| Evaluate (local ckpt) | `uv run python -m motion_inbetweening.scripts.test path/to/checkpoint [test_set]` |
| Evaluate (HF model) | `uv run python -m motion_inbetweening.scripts.test AnimajSAS/ais_bilstm_rig_controllers_values [test_set]` |

`--training-type` options: `best`, `custom`, `debug`, `from_file`. Model options: `lstm` (default), `citl`, `delta_interpolator`, `diffusion`. `test_set` options: `all` (default), `held_out_algorithmic`, `held_out_random`, `production`. Dataset path is resolved automatically from env vars — do not pass a path.

## Structure

```
motion_inbetweening/   # Main ML package
  scripts/             # Entrypoints: train.py, test.py, test_st.py
  config/              # Pydantic + OmegaConf config classes
  lightning_modules/   # PyTorch Lightning modules
  infra/               # HF Hub, paths, checkpoint loading
shared/                # Local utility package, imported as `from shared.domain...`
  domain/
  infrastructure/
  losses/
  rig/
  utils.py
```

`shared/` is not separately installed — it is importable directly as a local package.

## Toolchain Quirks

- **Configs**: OmegaConf + Pydantic used together; support `.from_yaml()` and `.model_dump()`.
- **Dataset download**: If dataset dir is missing, `ensure_hf_dataset()` auto-pulls from HuggingFace Hub.
- **Checkpoints**: After training, automatically converted to safetensors format.
- **Dual logging**: MLflow + TensorBoard both active during training.
- **`moviepy==1.0.3`** is pinned exactly — do not upgrade.
- **`node_modules/`** exists at root — prettier (npm) is used by pre-commit to format `*.md` files at 120 char width.

## Linting / Style

- Line length: **120**
- `F401` (unused imports) and `F841` (unused variables) are **not auto-fixed** — must be resolved manually.
- Print statements (`T20`) are **allowed**.
- Annotations (`ANN`) and docstrings (`D`) rules are **disabled**.
- Docstring style when present: **Google**.

## Commits

Commitizen is enforced via pre-commit `commit-msg` hook. Use conventional commits format: `feat:`, `fix:`, `chore:`, `refactor:`, etc.
