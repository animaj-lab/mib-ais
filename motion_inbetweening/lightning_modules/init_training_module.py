from motion_inbetweening.config.controller_keyframes_preprocessing import ControllerKeyframesPreprocessingConfig
from motion_inbetweening.config.train import DiffusionTrainingModuleConfig, Seq2SeqTrainingModuleConfig
from motion_inbetweening.lightning_modules.base_module import BaseModule
from motion_inbetweening.lightning_modules.diffusion_module import DiffusionModule
from motion_inbetweening.lightning_modules.seq2seq_module import Seq2SeqModule
from shared.rig.config import ControllerConfig
from shared.rig.normalizer import PoseNormalizer
from shared.rig.trainable_controllers import TrainableController


def init_training_module(
    config: Seq2SeqTrainingModuleConfig | DiffusionTrainingModuleConfig,
    controllers_config: ControllerConfig,
    trainable_controllers: list[TrainableController],
    pose_normalizer: PoseNormalizer | None,
    controller_keyframes_preprocessing_config: ControllerKeyframesPreprocessingConfig | None,
) -> BaseModule:
    if isinstance(config, Seq2SeqTrainingModuleConfig):
        if controller_keyframes_preprocessing_config is not None:
            transformations_keyframe_division_strategy = (
                controller_keyframes_preprocessing_config.transformations_keyframe_division_strategy
            )
        else:
            transformations_keyframe_division_strategy = None
        if pose_normalizer is not None:
            mean_pose_normalizer = pose_normalizer.mean
            std_pose_normalizer = pose_normalizer.std
        else:
            mean_pose_normalizer = None
            std_pose_normalizer = None
        return Seq2SeqModule(
            config,
            controllers_config,
            trainable_controllers,
            mean_pose_normalizer,
            std_pose_normalizer,
            transformations_keyframe_division_strategy,
        )
    elif isinstance(config, DiffusionTrainingModuleConfig):
        return DiffusionModule(config, controllers_config, trainable_controllers, pose_normalizer)
    else:
        raise NotImplementedError(f"Module {config.module_name} not implemented")
