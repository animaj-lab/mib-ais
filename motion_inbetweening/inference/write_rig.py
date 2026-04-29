from pathlib import Path

from motion_inbetweening.config.data import RelativeControllerTransformationConfig
from motion_inbetweening.config.post_processing import UnmaskedFramesPostProcessingStrategy
from motion_inbetweening.domain.controller_keyframes import RigControllersKeyframes
from motion_inbetweening.domain.data.data_sample import AnimationSceneData
from motion_inbetweening.inference.post_processing import FilteringPlateauConfig, post_process_rig_prediction
from shared.domain.entities.rig_controllers_values import RigControllersValues, SceneRigControllersValues
from shared.utils import save_json


def write_scene_rig(scene_rig: SceneRigControllersValues, output_filepath: Path) -> None:
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_filepath, scene_rig.to_deepcopy_dict())


def post_process_and_write_rigs(
    output_filepath: Path,
    predicted_rigs_controllers_values: list[RigControllersValues],
    scene: AnimationSceneData,
    relative_controller_transformations: dict[str, list[RelativeControllerTransformationConfig]],
    do_controller_keyframes_reduction: bool,
    predicted_controller_keyframes: list[RigControllersKeyframes] | None,
    threshold_controller_keyframes_reduction: float | None,
    unmasked_frames_post_processing_strategy: UnmaskedFramesPostProcessingStrategy,
    filter_plateau_tolerance_config: FilteringPlateauConfig | None,
) -> None:
    output_rig_dict = post_process_rig_prediction(
        predicted_controller_values=predicted_rigs_controllers_values,
        scene=scene,
        relative_controller_transformations=relative_controller_transformations,
        do_controller_keyframes_reduction=do_controller_keyframes_reduction,
        predicted_controller_keyframes=predicted_controller_keyframes,
        threshold_controller_keyframes_reduction=threshold_controller_keyframes_reduction,
        unmasked_frames_post_processing_strategy=unmasked_frames_post_processing_strategy,
        filter_plateau_tolerance_config=filter_plateau_tolerance_config,
    )
    write_scene_rig(output_rig_dict, output_filepath)
