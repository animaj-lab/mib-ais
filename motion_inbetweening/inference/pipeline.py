from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from motion_inbetweening.app_services.online_preprocessing.batching import predict_collate_fn
from motion_inbetweening.app_services.online_preprocessing.predict import process_predict_data_sample
from motion_inbetweening.app_services.online_preprocessing.relative_sequence import (
    get_indices_for_relative_controller_transformations,
)
from motion_inbetweening.app_services.online_preprocessing.scene_splitting import FullSceneSplitting
from motion_inbetweening.config.post_processing import UnmaskedFramesPostProcessingStrategy
from motion_inbetweening.domain.data.data_sample import AnimationSceneData, PredictDataSample
from motion_inbetweening.domain.rig_controllers_values import scene_rig_controllers_values_to_dict
from motion_inbetweening.inference.post_processing import post_process_rig_prediction
from motion_inbetweening.infra.configs.train import get_relative_controller_transformations
from motion_inbetweening.lightning_modules.base_module import predict_step
from motion_inbetweening.lightning_modules.seq2seq_module import Seq2SeqModule
from shared.domain.entities.rig_controllers_values import RigControllersValues, SceneRigControllersValues
from shared.rig.rig_controllers import rig_to_vec, vec_to_rig
from shared.rig.trainable_controllers import TrainableController


@torch.no_grad()
def run_inference_pipeline(input_payload: dict, module: Seq2SeqModule, device: torch.device | str) -> dict:
    """Run the inference pipeline (preprocessing, inference, postprocessing) on a single sample.

    The first and the last frames of the input are the bounds of the scene. The model predicts all the frames between
    the bounds that are not in the input.

    Args:
        input_payload (dict): Non-validated input, frame ID -> controller name -> attribute name -> value. The
            controller names have no namespace. Each input frame contains all the trainable controllers.
        module (Seq2SeqModule): Trained module, already on `device`
        device (torch.device | str): Device that runs the model

    Returns:
        dict: The predicted frames only (the input frames are removed), in the same format as the input
    """
    scene_rig_controllers_values = SceneRigControllersValues.from_any(input_payload)
    if len(scene_rig_controllers_values) < 2:
        raise ValueError("The input must contain at least 2 frames.")

    predict_data_sample = scene_rig_controllers_values_to_predict_data_sample(
        scene_rig_controllers_values, module.trainable_controllers
    )

    relative_controller_transformations = get_relative_controller_transformations()
    scene_splitting = FullSceneSplitting(strategy_name="full_scene")
    relative_sequence_vectors, scene_data = process_predict_data_sample(
        predict_data_sample=predict_data_sample,
        scene_splitting=scene_splitting,
        indices_for_relative_controller_transformations=get_indices_for_relative_controller_transformations(
            module.trainable_controllers, relative_controller_transformations
        ),
    )
    padded_batch, _, torch_lengths, unmasked_frames_indices, _ = predict_collate_fn(
        [(relative_sequence_vectors, scene_data)], scene_splitting
    )

    predicted_sequences, _, _ = predict_step(
        padded_batch.to(device),
        torch_lengths.to(device),
        unmasked_frames_indices,
        module.pose_normalizer_training_module,
        module,
        module.mask_applier,
        module.predict_mask_generator_config,
    )

    predicted_sequence = predicted_sequences[0].detach().cpu()
    predicted_rigs = [
        RigControllersValues.from_any(vec_to_rig(module.trainable_controllers, frame_vec, defaultdict(dict)))
        for frame_vec in predicted_sequence
    ]
    output_scene_rig_controllers_values = post_process_rig_prediction(
        predicted_controller_values=predicted_rigs,
        scene=scene_data,
        relative_controller_transformations=relative_controller_transformations,
        do_controller_keyframes_reduction=False,
        predicted_controller_keyframes=None,
        threshold_controller_keyframes_reduction=None,
        unmasked_frames_post_processing_strategy=UnmaskedFramesPostProcessingStrategy.REMOVE,
        filter_plateau_tolerance_config=None,
    )
    return scene_rig_controllers_values_to_dict(output_scene_rig_controllers_values)


def scene_rig_controllers_values_to_predict_data_sample(
    scene_rig_controllers_values: SceneRigControllersValues, trainable_controllers: list[TrainableController]
) -> PredictDataSample:
    """Wrap the input frames into a PredictDataSample. The input frames are the unmasked frames."""
    first_frame_id = scene_rig_controllers_values.get_first_frame_id()
    last_frame_id = scene_rig_controllers_values.get_last_frame_id()
    scene_data = AnimationSceneData(
        name="local_inference",
        ma_file_path=Path(),
        input_rig=scene_rig_controllers_values,
        frame_range=(first_frame_id, last_frame_id),
        unmasked_frames_id=sorted(scene_rig_controllers_values.keys()),
    )
    pose_vectors = scene_rig_controllers_values_to_pose_vectors(scene_rig_controllers_values, trainable_controllers)
    return PredictDataSample(pose_vectors=pose_vectors, scene_data=scene_data)


def scene_rig_controllers_values_to_pose_vectors(
    scene_rig_controllers_values: SceneRigControllersValues, trainable_controllers: list[TrainableController]
) -> torch.Tensor:
    """Vectorize the input frames into a tensor of shape (scene_length, pose_vector_length).

    The frames that are not in the input stay at zero. The mask hides them from the model.
    """
    first_frame_id = scene_rig_controllers_values.get_first_frame_id()
    scene_length = scene_rig_controllers_values.get_last_frame_id() - first_frame_id + 1

    pose_vectors = None
    for frame_id, rig_controllers_values in scene_rig_controllers_values.items():
        vec = np.asarray(rig_to_vec(trainable_controllers, rig_controllers_values.to_deepcopy_dict()), dtype=np.float32)
        if pose_vectors is None:
            pose_vectors = np.zeros((scene_length, len(vec)), dtype=np.float32)
        pose_vectors[frame_id - first_frame_id] = vec

    return torch.from_numpy(pose_vectors)
