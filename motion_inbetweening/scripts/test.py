"""Script to run qualitative evaluation for a single experiment across all test sets.
Takes the path to the safetensors checkpoint directory as input."""

import logging
import os
from enum import StrEnum
from pathlib import Path

import typer
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import MLFlowLogger

from motion_inbetweening.app_services.online_preprocessing.mask_generator import RandomUniformMaskGeneratorConfig
from motion_inbetweening.infra.configs.train import get_best_model_config
from motion_inbetweening.infra.dev_settings import get_dataset_directory
from motion_inbetweening.infra.hf_dataset import ensure_hf_dataset
from motion_inbetweening.infra.loading.safetensors import load_safetensors, load_safetensors_from_hub
from motion_inbetweening.infra.pytorch_lightning.data_module import MotionInbetweeningDatamodule
from shared.domain.services.ip import validate_character_name
from shared.utils import get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("motion_inbetweening.scripts.test")


load_dotenv()


class TestSet(StrEnum):
    HELD_OUT_ALGORITHMIC = "held_out_algorithmic"
    HELD_OUT_RANDOM = "held_out_random"
    PRODUCTION = "production"


def validate_test_set(test_set_str: str) -> TestSet:
    try:
        return TestSet(test_set_str)
    except ValueError as e:
        raise ValueError(f"Invalid test set: {test_set_str}. Valid options are: {[e.value for e in TestSet]}") from e


def get_mlflow_logger_for_existing_run(experiment_name: str, run_name: str) -> MLFlowLogger | None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        logger.info("MLFLOW_TRACKING_URI not set. Skipping MLflow logging.")
        return None
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.warning(f"MLflow experiment '{experiment_name}' not found. Skipping MLflow logging.")
        return None
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
    )
    if not runs:
        logger.warning(
            f"No MLflow run with name '{run_name}' in experiment '{experiment_name}'. Skipping MLflow logging."
        )
        return None
    return MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        run_id=runs[0].info.run_id,
    )


def _run_test_set(
    test_set: TestSet,
    train_module,
    character_name: str,
    dataset_dir: Path,
    trainable_controllers,
    device: str,
    run_name: str,
) -> None:
    config = get_best_model_config(model_name="lstm", character_name=character_name)

    match test_set:
        case TestSet.HELD_OUT_ALGORITHMIC:
            test_set_path = dataset_dir / "in_house_dataset"
        case TestSet.HELD_OUT_RANDOM:
            test_set_path = dataset_dir / "in_house_dataset"
            train_module.test_mask_generator_config = RandomUniformMaskGeneratorConfig(
                strategy_name="random_uniform", ratio_masked_frames=0.9
            )

        case TestSet.PRODUCTION:
            test_set_path = dataset_dir / "prod_test_dataset"

    if config.data_module.test_dataset is None:
        raise ValueError("Test dataset cannot be None for quantitative evaluation.")

    config.data_module.test_dataset.block_schedule_augmentation = None
    config.data_module.test_dataset.dataset_directory = test_set_path
    config.data_module.batch_size = 1
    data_module = MotionInbetweeningDatamodule(config.data_module, trainable_controllers)

    mlflow_logger = get_mlflow_logger_for_existing_run(config.logging.project_name, run_name)
    trainer = Trainer(
        accelerator=device,
        devices=1,
        max_epochs=config.training.max_epochs,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        val_check_interval=config.training.val_check_interval,
        logger=mlflow_logger if mlflow_logger is not None else False,
        precision=config.training.precision,
    )
    seed_everything(0)
    trainer.test(model=train_module, datamodule=data_module)


def main(checkpoint: str, test_set: str = "all") -> None:
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.exists():
        train_module = load_safetensors(checkpoint_path)
    else:
        train_module = load_safetensors_from_hub(checkpoint)
    device = get_device()
    character_name = validate_character_name(train_module.config.character_name)
    dataset_dir = get_dataset_directory(character_name)
    ensure_hf_dataset(dataset_dir)
    trainable_controllers = train_module.trainable_controllers
    run_name = checkpoint_path.parent.name if checkpoint_path.exists() else checkpoint.split("/")[-1]

    if test_set == "all":
        # Note: order is important to avoid overriding the block schedule to random for the other test sets
        test_sets = [TestSet.PRODUCTION, TestSet.HELD_OUT_ALGORITHMIC, TestSet.HELD_OUT_RANDOM]
    else:
        test_sets = [validate_test_set(test_set)]

    for test_set in test_sets:
        logger.info(f"Running evaluation on test set: {test_set}")
        _run_test_set(
            test_set=test_set,
            train_module=train_module,
            character_name=character_name,
            dataset_dir=dataset_dir,
            trainable_controllers=trainable_controllers,
            device=device,
            run_name=run_name,
        )


if __name__ == "__main__":
    typer.run(main)
