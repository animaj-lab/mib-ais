from pathlib import Path

import pandas as pd
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import BasePredictionWriter

from motion_inbetweening.config.controller_keyframes_preprocessing import ControllerKeyframesPreprocessingConfig
from motion_inbetweening.config.data import RelativeControllerTransformationConfig
from motion_inbetweening.domain.controller_keyframes import RigControllersKeyframes
from motion_inbetweening.domain.data.data_sample import AnimationSceneData
from motion_inbetweening.evaluation.curves_analysis import write_controller_curves
from motion_inbetweening.infra.paths.dataset import (
    to_controller_curves_dataframe_path,
    to_predict_controller_keyframes_path,
)
from motion_inbetweening.lightning_callbacks.write_interval import WriteInterval
from shared.domain.entities.rig_controllers_values import RigControllersValues
from shared.rig.trainable_controllers import TrainableController


class RigCSVWriter(BasePredictionWriter):
    def __init__(
        self,
        predict_dataset_dir: Path,
        experiment_dir: Path,
        trainable_controllers: list[TrainableController],
        relative_controller_transformations: dict[str, list[RelativeControllerTransformationConfig]],
        controller_keyframes_preprocessing_config: ControllerKeyframesPreprocessingConfig | None,
        write_interval: WriteInterval,
    ):
        super().__init__(write_interval)
        experiment_dir.mkdir(parents=True, exist_ok=True)
        self.predict_dataset_dir = predict_dataset_dir.resolve()
        self.output_dir = experiment_dir.resolve() / "qualitative_evaluation"
        self.csv_controller_values_dir = self.output_dir / "csv_rigs"
        self.trainable_controllers = trainable_controllers
        self.relative_controller_transformations = relative_controller_transformations
        self.df_controller_values = pd.read_csv(to_controller_curves_dataframe_path(self.predict_dataset_dir))
        self.write_controller_keyframes_curves = False
        if controller_keyframes_preprocessing_config is not None:
            self.write_controller_keyframes_curves = True
            controller_keyframes_path = to_predict_controller_keyframes_path(
                self.predict_dataset_dir, controller_keyframes_preprocessing_config.detection_algorithm
            )
            if not controller_keyframes_path.exists():
                raise FileNotFoundError(
                    f"{controller_keyframes_path} does not exist. "
                    "Writing controller keyframes curves is enabled and requires to have "
                    "computed ground truth controller keyframes csv."
                )
            self.df_controller_keyframes = pd.read_csv(controller_keyframes_path)

    def write_on_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        predictions: list[
            tuple[
                list[list[RigControllersValues]], list[list[RigControllersKeyframes]] | None, list[AnimationSceneData]
            ]
        ],
        batch_indices: list[list[list[int]]],
    ) -> None:
        for batch in predictions:
            list_preds_values, list_preds_controller_keyframes, list_scenes, list_details = batch
            for i, (predicted_controller_values, scene) in enumerate(zip(list_preds_values, list_scenes, strict=False)):
                if self.write_controller_keyframes_curves:
                    if list_preds_controller_keyframes is None:
                        raise ValueError(
                            "Writing controller curves is enabled but no controller keyframes were predicted."
                        )
                    predicted_controller_keyframes = list_preds_controller_keyframes[i]
                else:
                    predicted_controller_keyframes = None
                write_controller_curves(
                    dataset_df_controller_values=self.df_controller_values,
                    dataset_df_controller_keyframes=self.df_controller_keyframes
                    if self.write_controller_keyframes_curves
                    else None,
                    scene=scene,
                    predicted_controller_values=predicted_controller_values,
                    predicted_controller_keyframes=predicted_controller_keyframes,
                    trainable_controllers=self.trainable_controllers,
                    relative_controller_transformations=self.relative_controller_transformations,
                    csv_rigs_dir=self.csv_controller_values_dir,
                )
