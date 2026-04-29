from motion_inbetweening.app_services.online_preprocessing.mask_generator import (
    CondMIRandomMaskGeneratorConfig,
    ManualMaskGeneratorConfig,
    RandomBlockKeyframesMaskGeneratorConfig,
)
from motion_inbetweening.app_services.online_preprocessing.scene_splitting import (
    FixedLengthSplitting,
    RandomFixedLengthSplitting,
)
from motion_inbetweening.config.data import DatasetConfig, TrainDataModuleConfig
from motion_inbetweening.config.gaussian_diffusion import CosineNoiseSchedule, SpacedDiffusionConfig
from motion_inbetweening.config.inference_diffusion import InferenceDiffusionConfig
from motion_inbetweening.config.losses import BCEWithLogitsLoss, L1Loss, LossesConfig, LossesWeights
from motion_inbetweening.config.mdm import MDMUnetConfig
from motion_inbetweening.config.optimization import (
    OptimizationConfig,
    OptimizerConfig,
    RegularizationConfig,
    SchedulerConfig,
)
from motion_inbetweening.config.train import (
    CallbacksConfig,
    DiffusionTrainingModuleConfig,
    EarlyStoppingParams,
    GlobalTrainingConfig,
    MaskingConfig,
    ModelCheckpointParams,
    SceneWriterParams,
    SchedulerSamplerConfig,
    TrainConfig,
)
from motion_inbetweening.domain.mask_applier import ConcatMode, FillMode, MaskApplierConfig
from motion_inbetweening.infra.configs.post_processing import get_default_post_processing_config
from motion_inbetweening.infra.dev_settings import get_dataset_directory
from motion_inbetweening.infra.paths.dataset import to_predict_dir
from shared.domain.entities.ip import CharacterName

from .train import get_controllers_config, get_logging_config, get_relative_controller_transformations


def get_callbacks_config() -> CallbacksConfig:
    model_checkpoint_params = ModelCheckpointParams(every_n_epochs=1)
    early_stopping_params = EarlyStoppingParams(monitor="val/loss", mode="min", patience=16, min_delta=0.0001)
    scene_writer_params = SceneWriterParams(video_resolution=512, fps=25)
    return CallbacksConfig(
        model_checkpoint_params=model_checkpoint_params,
        use_early_stopping=False,
        early_stopping_params=early_stopping_params,
        scene_writer_params=scene_writer_params,
        write_controller_keyframes_curves=False,
    )


def get_optimization_config() -> OptimizationConfig:
    optimizer = OptimizerConfig(name="AdamW", lr=1e-05, tune_lr=True, params={"weight_decay": 0.001})
    lr_scheduler = SchedulerConfig(name="CyclicLR", params={"base_lr": 1e-05, "max_lr": 0.0003, "step_size_up": 500})
    regularization = RegularizationConfig(
        last_layer_regularization=False,
        last_layer_regularization_weight=0.001,
        full_model_regularization=False,
        full_model_regularization_weight=1e-08,
    )
    return OptimizationConfig(optimizer=optimizer, lr_scheduler=lr_scheduler, regularization=regularization)


def get_random_block_keyframes_mask_generator_config(
    ratio_masked_block_keyframes: float,
) -> RandomBlockKeyframesMaskGeneratorConfig:
    return RandomBlockKeyframesMaskGeneratorConfig(
        strategy_name="random_block_keyframes", ratio_masked_block_keyframes=ratio_masked_block_keyframes
    )


def get_cond_mi_random_mask_generator_config() -> CondMIRandomMaskGeneratorConfig:
    return CondMIRandomMaskGeneratorConfig(strategy_name="cond_mi_random")


def get_masking_config() -> MaskingConfig:
    mask_generator = get_cond_mi_random_mask_generator_config()
    predict_mask_generator = ManualMaskGeneratorConfig(strategy_name="manual")
    test_mask_generator = get_random_block_keyframes_mask_generator_config(0)
    test_mask_generator = get_random_block_keyframes_mask_generator_config(0.0)
    masking_applier = MaskApplierConfig(concat_mode=ConcatMode.CONCAT_DIM, fill_mode=FillMode.ZEROS)
    return MaskingConfig(
        mask_generator=mask_generator,
        predict_mask_generator=predict_mask_generator,
        test_mask_generator=test_mask_generator,
        mask_applier=masking_applier,
    )


def get_inference_diffusion_config() -> InferenceDiffusionConfig:
    return InferenceDiffusionConfig(
        use_ema_model=False, do_imputation=False, imputation_params=None, do_guidance=False, guidance_params=None
    )


def get_losses_config() -> LossesConfig:
    return LossesConfig(
        pose_loss=L1Loss(reduction="none"),
        speed_loss=L1Loss(reduction="none"),
        acceleration_loss=L1Loss(reduction="mean"),
        jerk_loss=L1Loss(reduction="mean"),
        controller_keyframes_prediction_loss=BCEWithLogitsLoss(reduction="none", pos_weight=1.0),
        weights=LossesWeights(
            pose=1.0,
            speed=0.0,
            acceleration=0.0,
            jerk=0.0,
            controller_keyframes_prediction=0.0,
            frame_weights=None,
            controller_keyframes_weights=None,
        ),
        auto_weighting_loss=False,
    )


def get_diffusion_training_module_config(character_name: CharacterName) -> DiffusionTrainingModuleConfig:
    model_config = get_unet_mdm_config()
    diffusion_config = get_spaced_diffusion_config()
    masking_config = get_masking_config()
    optimization_config = get_optimization_config()
    scheduler_config = get_scheduler_config()
    inference_diffusion_config = get_inference_diffusion_config()
    losses = get_losses_config()
    do_controller_keyframes_prediction = False
    post_processing_config = get_default_post_processing_config(do_controller_keyframes_prediction)
    return DiffusionTrainingModuleConfig(
        character_name=character_name,
        model=model_config,
        diffusion=diffusion_config,
        masking=masking_config,
        optimization=optimization_config,
        scheduler=scheduler_config,
        inference=inference_diffusion_config,
        losses=losses,
        do_controller_keyframes_prediction=do_controller_keyframes_prediction,
        post_processing=post_processing_config,
    )


def get_scheduler_config():
    return SchedulerSamplerConfig(schedule_sampler_type="uniform")


def get_spaced_diffusion_config() -> SpacedDiffusionConfig:
    noise_schedule = CosineNoiseSchedule(name="cosine", offset=0.008, exponent=2)
    return SpacedDiffusionConfig(
        noise_schedule=noise_schedule,
        steps=1000,
        use_ddim=False,
        predict_xstart=True,
        learn_sigma=False,
        sigma_small=True,
        rescale_timesteps=False,
        lambda_pose=1.0,
        lambda_vel=0.0,
        use_random_proj=False,
        fp16=True,
        apply_zero_mask=False,
        time_weighted_loss=False,
        train_x0_as_eps=False,
        train_keypoint_mask="none",
    )


def get_unet_mdm_config():
    return MDMUnetConfig(
        modeltype="unet",
        latent_dim=256,
        dim_mults=(2, 2, 2, 2),
        attention=False,
        ablation=None,
        legacy=False,
        emb_trans_dec=False,
        adagn=True,
        zero=True,
        arch="unet",
        unet_out_mult=1,
        keyframe_conditioned=True,
        zero_keyframe_loss=False,
        final_type=1,
    )


def get_main_predict_dataset_config(character_name: CharacterName) -> DatasetConfig:
    predict_dir = to_predict_dir(get_dataset_directory(character_name), "test_subset")
    return DatasetConfig(
        dataset_directory=predict_dir,
        subset_size=-1,
        relative_controller_transformations=get_relative_controller_transformations(),
        block_schedule_augmentation=None,
    )


def get_diffusion_data_module_config(use_manual_predict: bool, character_name: CharacterName) -> TrainDataModuleConfig:
    train_dataset = DatasetConfig(
        dataset_directory=get_dataset_directory(character_name),
        subset_size=-1,
        relative_controller_transformations=get_relative_controller_transformations(),
        block_schedule_augmentation=None,
    )
    test_dataset = DatasetConfig(
        dataset_directory=get_dataset_directory(character_name),
        subset_size=-1,
        relative_controller_transformations=get_relative_controller_transformations(),
        block_schedule_augmentation=None,
    )
    if use_manual_predict:
        raise NotImplementedError("Manual predict dataset is not implemented yet")
    else:
        predict_dataset = get_main_predict_dataset_config(character_name)
    data_module = TrainDataModuleConfig(
        batch_size=32,
        validation_ratio=0.2,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        predict_dataset=predict_dataset,
        num_workers=0,
        do_normalization=False,
        train_scene_splitting=RandomFixedLengthSplitting(strategy_name="random_fixed_length", length=224),
        test_scene_splitting=FixedLengthSplitting(strategy_name="fixed_length", length=224),
        predict_scene_splitting=FixedLengthSplitting(strategy_name="fixed_length", length=224),
        controller_keyframes_preprocessing_config=None,
    )
    return data_module


def get_diffusion_default_config(character_name: CharacterName) -> GlobalTrainingConfig:
    logging = get_logging_config(character_name)
    data_module = get_diffusion_data_module_config(False, character_name)
    training_module = get_diffusion_training_module_config(character_name)
    train = TrainConfig(
        max_epochs=800, precision="16-mixed", accumulate_grad_batches=1, val_check_interval=1.0, gradient_clip_val=0.0
    )
    callbacks = get_callbacks_config()
    controllers_config = get_controllers_config(character_name)
    return GlobalTrainingConfig(
        data_module=data_module,
        training=train,
        logging=logging,
        callbacks=callbacks,
        controllers=controllers_config,
        train_module=training_module,
        seed=42,
    )


def get_diffusion_debug_config(character_name: CharacterName) -> GlobalTrainingConfig:
    """Debug config, used to test the training pipeline

    Args:
        remote (bool): Whether to use the remote or local config

    Returns:
        GlobalTrainingConfig: The debug config
    """
    config = get_diffusion_default_config(character_name)
    config.logging.project_name = f"MIB_{character_name}_dev"
    config.logging.experiment_name = "debug_diffusion"
    config.training.max_epochs = 10
    config.train_module.optimization.optimizer.tune_lr = False
    config.data_module.train_dataset.subset_size = -1
    config.callbacks.model_checkpoint_params.every_n_epochs = 10
    return config


def get_diffusion_best_model_config(character_name: CharacterName) -> GlobalTrainingConfig:
    """Config of current best model

    Returns:
        GlobalTrainingConfig: The config of the best model
    """
    config = get_diffusion_default_config(character_name)
    config.logging.project_name = "MIB_paper_SOTA_best_models"
    config.logging.experiment_name = "best_diffusion_model"
    config.callbacks.model_checkpoint_params.every_n_epochs = 1
    assert isinstance(config.train_module, DiffusionTrainingModuleConfig), "Expected DiffusionTrainingModuleConfig"
    config.train_module.model = MDMUnetConfig(
        modeltype="unet",
        latent_dim=512,
        dim_mults=(1, 1, 1, 1),
        attention=False,
        ablation=None,
        legacy=False,
        emb_trans_dec=False,
        adagn=True,
        zero=True,
        arch="unet",
        unet_out_mult=1,
        keyframe_conditioned=True,
        zero_keyframe_loss=False,
        final_type=1,
    )
    config.train_module.optimization.lr_scheduler = SchedulerConfig(
        name="CyclicLR", params={"base_lr": 1.497e-07, "max_lr": 0.0003342, "step_size_up": 100, "mode": "triangular2"}
    )
    config.data_module.batch_size = 16
    config.training.max_epochs = 2000
    return config
