import json
from datetime import datetime
from pathlib import Path

import torch
import yaml


def to_json_serializable(data):
    if isinstance(data, Path):
        return str(data)
    elif isinstance(data, list):
        return [to_json_serializable(d) for d in data]
    elif isinstance(data, dict):
        return {k: to_json_serializable(v) for k, v in data.items()}
    return data


def save_json(path, data):
    data = to_json_serializable(data)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def load_yaml(path):
    with open(path) as file:
        values = yaml.safe_load(file)
    return values


def save_yaml(path, data):
    with open(path, "w") as file:
        yaml.dump(data, file)


def get_device(device=None):
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_save_dir(config) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M")
    config.logging.experiment_name = f"{config.logging.experiment_name}_{timestamp}"
    save_dir = Path(config.logging.save_dir_experiments) / config.logging.experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def save_fig_lr_finder(lr_finder, save_dir):
    fig = lr_finder.plot(suggest=True)
    fig.suptitle(f"best lr found by tuner: {lr_finder.suggestion()}", fontsize=16)
    fig.savefig(save_dir / "lr_finder.png")
    return fig
