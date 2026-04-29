import json
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

from motion_inbetweening.config.controller_keyframes_preprocessing import (
    AttributeKeyframeDivisionStategy,
    TransformationName,
    TransformationsKeyframesDivisionStrategy,
)
from motion_inbetweening.config.train import Seq2SeqTrainingModuleConfig
from motion_inbetweening.infra.dev_settings import get_experiment_directory
from motion_inbetweening.lightning_modules.seq2seq_module import Seq2SeqModule
from shared.domain.entities.ip import PocoyoCharacterName
from shared.rig.config import ControllerConfig
from shared.rig.trainable_controllers import TrainableController, load_trainable_controllers, save_trainable_controllers


def save_safetensors(ckpt_path: Path, output_dir: Path) -> None:
    """Exports a .ckpt checkpoint into `output_dir` as three files:

    - `model.safetensors`: model weights (prefixed 'model/') and normalizer tensors (prefixed 'normalizer/').
    - `trainable_controllers.yaml`: trainable controllers.
    - `model.json`: config and metadata.
    """
    # weights_only=False required: .ckpt stores arbitrary Python objects (Pydantic models, etc.)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    hparams: dict = checkpoint.get("hyper_parameters", {})
    state_dict: dict[str, torch.Tensor] = checkpoint["state_dict"]

    tensors: dict[str, torch.Tensor] = {f"model/{key}": tensor.contiguous() for key, tensor in state_dict.items()}

    mean_normalizer: torch.Tensor | None = hparams.get("mean_pose_normalizer")
    std_normalizer: torch.Tensor | None = hparams.get("std_pose_normalizer")

    if mean_normalizer is not None:
        tensors["normalizer/mean"] = mean_normalizer.contiguous()
    if std_normalizer is not None:
        tensors["normalizer/std"] = std_normalizer.contiguous()

    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, output_dir / "model.safetensors")

    config: Seq2SeqTrainingModuleConfig = hparams["config"]
    controllers_config: ControllerConfig = hparams["controllers_config"]

    # Handle absolute paths in controllers config
    controllers_config.default_controllers_path = Path(
        str(controllers_config.default_controllers_path).replace("/workspaces/", "")
    )

    trainable_controllers: list[TrainableController] = hparams["trainable_controllers"]
    raw_strategy: TransformationsKeyframesDivisionStrategy | None = hparams.get(
        "transformations_keyframe_division_strategy"
    )

    save_trainable_controllers(trainable_controllers, output_dir / "trainable_controllers.yaml")

    meta = {
        "config": config.model_dump(mode="json"),
        "controllers_config": controllers_config.model_dump(mode="json"),
        "transformations_keyframe_division_strategy": (
            {str(k): str(v) for k, v in raw_strategy.items()} if raw_strategy is not None else None
        ),
        "has_mean_pose_normalizer": mean_normalizer is not None,
        "has_std_pose_normalizer": std_normalizer is not None,
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
    }

    (output_dir / "model.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_safetensors(checkpoint_dir: Path) -> Seq2SeqModule:
    """Loads a Seq2SeqModule from a checkpoint directory produced by export_ckpt_model.

    Expects `checkpoint_dir` to contain `model.safetensors`, `trainable_controllers.yaml`, and `model.json`.
    """
    meta: dict = json.loads((checkpoint_dir / "model.json").read_text(encoding="utf-8"))

    config = Seq2SeqTrainingModuleConfig.model_validate(meta["config"])
    controllers_config = ControllerConfig.model_validate(meta["controllers_config"])

    trainable_controllers = load_trainable_controllers(
        checkpoint_dir / "trainable_controllers.yaml", controllers_config.simplify_classes_path
    )

    raw_strategy: dict[str, str] | None = meta["transformations_keyframe_division_strategy"]
    transformations_keyframe_division_strategy: TransformationsKeyframesDivisionStrategy | None = (
        {TransformationName(k): AttributeKeyframeDivisionStategy(v) for k, v in raw_strategy.items()}
        if raw_strategy is not None
        else None
    )

    tensors = load_file(checkpoint_dir / "model.safetensors", device="cpu")

    mean_pose_normalizer = tensors["normalizer/mean"] if meta["has_mean_pose_normalizer"] else None
    std_pose_normalizer = tensors["normalizer/std"] if meta["has_std_pose_normalizer"] else None

    state_dict = {key[len("model/") :]: tensor for key, tensor in tensors.items() if key.startswith("model/")}

    module = Seq2SeqModule(
        config=config,
        controllers_config=controllers_config,
        trainable_controllers=trainable_controllers,
        mean_pose_normalizer=mean_pose_normalizer,
        std_pose_normalizer=std_pose_normalizer,
        transformations_keyframe_division_strategy=transformations_keyframe_division_strategy,
    )
    module.load_state_dict(state_dict)
    module.eval()

    return module


def load_safetensors_from_hub(repo_id: str) -> "Seq2SeqModule":
    """Downloads a checkpoint from a HuggingFace model repository to the experiment directory and loads it."""

    ckpt_name = repo_id.split("/")[-1]

    ckpt_dir = get_experiment_directory(PocoyoCharacterName.POCOYO) / ckpt_name

    if not ckpt_dir.exists():
        snapshot_download(repo_id=repo_id, repo_type="model", local_dir=ckpt_dir)

    return load_safetensors(ckpt_dir)
