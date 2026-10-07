# Adaptive Interpolation-Synthesis for Motion In-Betweening on Keyframe-Based Animation

Official implementation of the method described in the following paper, including training and evaluation scripts:

> Raël et al., "Adaptive Interpolation-Synthesis for Motion In-Betweening on Keyframe-Based Animation",
> SIGGRAPH 2026 Conference Papers (2026)

[![Paper](https://img.shields.io/badge/Paper-link-blue)](https://arxiv.org/abs/2605.02742)
![Python](https://img.shields.io/badge/Python-3.x-green)
![uv](https://img.shields.io/badge/uv-package%20manager-lightgrey)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)
[![HF Model](https://img.shields.io/badge/🤗%20Model-AIS_BI_LSTM_v0-yellow)](https://huggingface.co/AnimajSAS/AIS_BI_LSTM_v0)
[![HF Dataset](https://img.shields.io/badge/🤗%20Dataset-mib_rig_controllers_values-yellow)](https://huggingface.co/datasets/AnimajSAS/mib_rig_controllers_values)

## Table of Contents

- [Installation](#installation)
- [Setup](#setup)
- [Training](#training)
- [Evaluation](#evaluation)
- [Maya plugin](#maya-plugin)
- [Development](#development)
- [Citation](#citation)

## Installation

This repo uses uv as the package manager. See the [uv documentation](https://docs.astral.sh/uv/) for installation
instructions.

Once installed, run:

```sh
uv sync --frozen --all-groups
```

## Setup

Create a `.env` file at the repo root:

```sh
# Dataset & experiment directories
export MIB_POCOYO_DATASET_DIR=data/dataset
export MIB_POCOYO_EXPERIMENT_DIR=data/exp

# Optional: MLflow tracking server
export MLFLOW_TRACKING_URI=https://your-mlflow-server-uri
```

If the dataset directory is missing, it will be automatically downloaded from
[AnimajSAS/mib_rig_controllers_values](https://huggingface.co/datasets/AnimajSAS/mib_rig_controllers_values) on HuggingFace.

## Training

```sh
uv run python -m motion_inbetweening.scripts.train pocoyo --training-type best
```

Trained in ~2 hours on a RTX 4070.

## Evaluation

### Evaluate local checkpoint

```sh
uv run python -m motion_inbetweening.scripts.test path/to/checkpoint
# e.g. uv run python -m motion_inbetweening.scripts.test data/exp/<exp_name>/checkpoints/safetensors
```

`path/to/checkpoint` should point to the `safetensors` directory, generated at the end of training.

### Evaluate HuggingFace model

We provide a pretrained model on HuggingFace at [AnimajSAS/AIS_BI_LSTM_v0](https://huggingface.co/AnimajSAS/AIS_BI_LSTM_v0):

```sh
uv run python -m motion_inbetweening.scripts.test AnimajSAS/AIS_BI_LSTM_v0
```

By default, all three test sets are evaluated. Use `--test-set` to run a specific one:

```sh
uv run python -m motion_inbetweening.scripts.test path/to/checkpoint --test-set held_out_algorithmic
```

Valid values for `--test-set`:

- `held_out_algorithmic` — in-house dataset with algorithmic block schedule
- `held_out_random` — in-house dataset with random masking
- `production` — production test set
- `all` (default) — runs all three test sets

## Maya plugin

The Maya plugin in-betweens the Pocoyo rig on your machine. It has two parts:

- a local inference server, which runs in this repo's uv environment and loads the model one time;
- a Maya shelf, which sends the keyed frames to the server and keys the predicted frames as breakdowns.

The plugin needs Autodesk Maya 2025 or later. It uses only the Python standard library inside Maya, so you do not
need to install packages in `mayapy`.

### 1. Start the inference server

Complete the [Installation](#installation) and [Setup](#setup) steps, then run this command from the repo root:

```sh
uv run python -m motion_inbetweening.scripts.serve
```

The first start downloads [AnimajSAS/AIS_BI_LSTM_v0](https://huggingface.co/AnimajSAS/AIS_BI_LSTM_v0) to
`MIB_POCOYO_EXPERIMENT_DIR`. Options:

- `--model path/to/safetensors`: use a local checkpoint
- `--port 8765`: change the port. Set the same URL in the `MIB_SERVER_URL` environment variable of Maya (for example
  `MIB_SERVER_URL=http://127.0.0.1:8766`).
- `--device cpu`: change the device. The CPU is fast enough for this model.

Keep the server open while you use the plugin.

### 2. Install the shelf buttons

Select a shelf in Maya, then drag `maya_plugin/install.mel` into the Maya viewport. The installer adds two buttons to
the current shelf:

- **In-between Pocoyo**: predicts the frames between the keys
- **Undo**: removes the keys of the last in-betweening

### 3. In-between an animation

1. Reference or open the Pocoyo rig: `assets/pcy/ch/ch0001/ch0001.ma`.
2. Key at least 2 poses. Key all the controllers at each pose.
3. Select a controller of the rig.
4. Select a time range in the timeline. If no range is selected, the plugin uses the playback range.
5. Click **In-between Pocoyo**.

The plugin keys all the frames between the first and the last keyed frames of the range, as breakdowns. It does not
change your keys. Click **Undo** to remove the predicted keys.

## Development

Run linting and formatting checks before committing:

```sh
uv run pre-commit run
```

## License

This project is licensed under the [Apache 2.0 License](LICENSE).
