from motion_inbetweening.app_services.online_preprocessing.mask_generator import (
    CITLRandomMaskGeneratorConfig,
    DeltaInterpolatorMaskGeneratorConfig,
    ManualMaskGeneratorConfig,
    RandomBlockKeyframesMaskGeneratorConfig,
)
from motion_inbetweening.app_services.online_preprocessing.scene_splitting import (
    FixedLengthSplitting,
    FullSceneSplitting,
    RandomFixedLengthSplitting,
)
from motion_inbetweening.config.block_keyframes_schedule_modifier import (
    BlockKeyframesScheduleAddition,
    BlockKeyframesScheduleDeletion,
    BlockKeyframesScheduleDeterministicShift,
    BlockKeyframesScheduleRandomShift,
    BlockScheduleAugmentation,
    BlockScheduleProbabilisticAugmentation,
)
from motion_inbetweening.config.data import DatasetConfig, RelativeControllerTransformationConfig, TrainDataModuleConfig
from motion_inbetweening.config.losses import BCEWithLogitsLoss, L1Loss, LossesConfig, LossesWeights
from motion_inbetweening.config.model import (
    CITLModelConfig,
    DeltaInterpolatorModelConfig,
    ExplicitInterpolationExtrapolationResidualLSTMModelConfig,
    LastLayerMode,
    ModelConfig,
    TransformerModelConfig,
)
from motion_inbetweening.config.optimization import (
    OptimizationConfig,
    OptimizerConfig,
    RegularizationConfig,
    SchedulerConfig,
)
from motion_inbetweening.config.train import (
    CallbacksConfig,
    EarlyStoppingParams,
    GlobalTrainingConfig,
    LoggingConfig,
    MaskingConfig,
    ModelCheckpointParams,
    SceneWriterParams,
    Seq2SeqTrainingModuleConfig,
    TrainConfig,
)
from motion_inbetweening.domain.mask_applier import ConcatMode, FillMode, MaskApplierConfig
from motion_inbetweening.infra.configs.post_processing import get_default_post_processing_config
from motion_inbetweening.infra.dev_settings import get_dataset_directory, get_experiment_directory
from motion_inbetweening.infra.paths.controller_configs import (
    get_pcy_default_controllers_path,
    get_pcy_trainable_controllers_path,
)
from motion_inbetweening.infra.paths.dataset import to_predict_dir
from shared.domain.entities.ip import CharacterName
from shared.rig.config import ControllerConfig


def get_logging_config(character_name: CharacterName) -> LoggingConfig:
    tags = {"scene_splitting": "random_fixed_length", "scene_splitting_length": "150"}
    return LoggingConfig(
        project_name=f"MIB_{character_name}_default",
        experiment_name="default",
        save_dir_experiments=get_experiment_directory(character_name),
        save_model_to_mlflow=False,
        hyperparameters={},
        tags=tags,
        upload_to_mlflow=True,
    )


def get_relative_controller_transformations() -> dict[str, list[RelativeControllerTransformationConfig]]:
    train_range_filter_threshold = 100
    test_range_filter_threshold = 100
    return {
        "x_main_CTRL": [
            RelativeControllerTransformationConfig(
                attribute="translateX",
                train_range_filter_threshold=train_range_filter_threshold,
                test_range_filter_threshold=test_range_filter_threshold,
            ),
            RelativeControllerTransformationConfig(
                attribute="translateY",
                train_range_filter_threshold=train_range_filter_threshold,
                test_range_filter_threshold=test_range_filter_threshold,
            ),
            RelativeControllerTransformationConfig(
                attribute="translateZ",
                train_range_filter_threshold=train_range_filter_threshold,
                test_range_filter_threshold=test_range_filter_threshold,
            ),
        ]
    }


def get_main_predict_dataset_config(character_name: CharacterName) -> DatasetConfig:
    predict_dir = to_predict_dir(get_dataset_directory(character_name), "test_subset")
    return DatasetConfig(
        dataset_directory=predict_dir,
        subset_size=-1,
        relative_controller_transformations=get_relative_controller_transformations(),
        block_schedule_augmentation=None,
    )


def get_block_schedule_augmentation_training_config() -> BlockScheduleAugmentation:
    return [
        BlockScheduleProbabilisticAugmentation(modifier=BlockKeyframesScheduleDeletion(percentage=10), proba=0.1),
        BlockScheduleProbabilisticAugmentation(modifier=BlockKeyframesScheduleDeletion(percentage=20), proba=0.05),
        BlockScheduleProbabilisticAugmentation(modifier=BlockKeyframesScheduleDeletion(percentage=30), proba=0.01),
        BlockScheduleProbabilisticAugmentation(
            modifier=BlockKeyframesScheduleDeterministicShift(shift_size=1), proba=0.2
        ),
        BlockScheduleProbabilisticAugmentation(
            modifier=BlockKeyframesScheduleDeterministicShift(shift_size=2), proba=0.1
        ),
        BlockScheduleProbabilisticAugmentation(
            modifier=BlockKeyframesScheduleDeterministicShift(shift_size=3), proba=0.05
        ),
        BlockScheduleProbabilisticAugmentation(
            modifier=BlockKeyframesScheduleDeterministicShift(shift_size=-1), proba=0.2
        ),
        BlockScheduleProbabilisticAugmentation(
            modifier=BlockKeyframesScheduleDeterministicShift(shift_size=-2), proba=0.1
        ),
        BlockScheduleProbabilisticAugmentation(
            modifier=BlockKeyframesScheduleDeterministicShift(shift_size=-3), proba=0.05
        ),
        BlockScheduleProbabilisticAugmentation(modifier=BlockKeyframesScheduleAddition(percentage=10), proba=0.1),
        BlockScheduleProbabilisticAugmentation(modifier=BlockKeyframesScheduleAddition(percentage=30), proba=0.05),
        BlockScheduleProbabilisticAugmentation(modifier=BlockKeyframesScheduleAddition(percentage=50), proba=0.01),
        BlockScheduleProbabilisticAugmentation(
            modifier=BlockKeyframesScheduleRandomShift(min_shift_size=-3, max_shift_size=3), proba=0.01
        ),
        BlockScheduleProbabilisticAugmentation(modifier=BlockKeyframesScheduleAddition(percentage=40), proba=0.01),
        BlockScheduleProbabilisticAugmentation(modifier=BlockKeyframesScheduleAddition(percentage=50), proba=0.01),
        BlockScheduleProbabilisticAugmentation(modifier=BlockKeyframesScheduleAddition(percentage=60), proba=0.01),
        BlockScheduleProbabilisticAugmentation(
            modifier=BlockKeyframesScheduleDeterministicShift(shift_size=4), proba=0.01
        ),
        BlockScheduleProbabilisticAugmentation(
            modifier=BlockKeyframesScheduleDeterministicShift(shift_size=5), proba=0.01
        ),
        BlockScheduleProbabilisticAugmentation(
            modifier=BlockKeyframesScheduleDeterministicShift(shift_size=6), proba=0.01
        ),
    ]


def get_data_module_config(character_name: CharacterName) -> TrainDataModuleConfig:
    train_dataset = DatasetConfig(
        dataset_directory=get_dataset_directory(character_name) / "in_house_dataset",
        subset_size=-1,
        relative_controller_transformations=get_relative_controller_transformations(),
        block_schedule_augmentation=get_block_schedule_augmentation_training_config(),
    )
    test_dataset = DatasetConfig(
        dataset_directory=get_dataset_directory(character_name) / "in_house_dataset",
        subset_size=-1,
        relative_controller_transformations=get_relative_controller_transformations(),
        block_schedule_augmentation=None,
    )
    data_module = TrainDataModuleConfig(
        batch_size=32,
        validation_ratio=0.2,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        predict_dataset=None,
        num_workers=0,
        do_normalization=False,
        train_scene_splitting=RandomFixedLengthSplitting(strategy_name="random_fixed_length", length=224),
        test_scene_splitting=FixedLengthSplitting(strategy_name="fixed_length", length=224),
        predict_scene_splitting=FullSceneSplitting(strategy_name="full_scene"),
        controller_keyframes_preprocessing_config=None,
    )
    return data_module


def get_cyclicLR_scheduler_config() -> SchedulerConfig:
    return SchedulerConfig(
        name="CyclicLR", params={"base_lr": 1e-06, "max_lr": 0.001, "step_size_up": 100, "mode": "triangular2"}
    )


def get_model_config(model_name: str) -> ModelConfig:
    if model_name == "transformer":
        return get_transformer_config()
    elif model_name == "lstm":
        return get_lstm_config()
    elif model_name == "citl":
        return get_citl_model_config()
    elif model_name == "delta_interpolator":
        return get_delta_interpolator_model_config()
    else:
        raise ValueError(f"Model {model_name} not recognized")


def get_transformer_config() -> TransformerModelConfig:
    return TransformerModelConfig(
        d_model=2048, nhead=4, num_encoder_layers=1, num_decoder_layers=2, dim_feedforward=1024, dropout=0.0
    )


def get_lstm_config() -> ExplicitInterpolationExtrapolationResidualLSTMModelConfig:
    return ExplicitInterpolationExtrapolationResidualLSTMModelConfig(
        hidden_size=1024, num_layers=2, dropout=0.2, last_layer_mode=LastLayerMode.AIS
    )


def get_citl_model_config() -> CITLModelConfig:
    return CITLModelConfig(
        embed_size=512,
        max_length=224,
        heads=4,
        key_layers=4,
        interm_layers=4,
        dec_layers=4,
        dropout=0.25,
        last_layer_mode=LastLayerMode.ORIGINAL,
    )


def get_delta_interpolator_model_config() -> DeltaInterpolatorModelConfig:
    return DeltaInterpolatorModelConfig(
        num_blocks_enc=6,
        num_layers_enc=3,
        layer_width_enc=1024,
        num_blocks_dec=0,
        num_layers_dec=1,
        layer_width_dec=1024,
        dropout=0.2,
        embedding_dim=32,
        embedding_size=256,
        embedding_num=2,
        num_heads=8,
        delta_mode="interpolator",
        input_delta_mode="last_pose",
        last_layer_mode=LastLayerMode.ORIGINAL,
    )


def get_random_block_keyframes_mask_generator_config(
    ratio_masked_block_keyframes: float,
) -> RandomBlockKeyframesMaskGeneratorConfig:
    return RandomBlockKeyframesMaskGeneratorConfig(
        strategy_name="random_block_keyframes", ratio_masked_block_keyframes=ratio_masked_block_keyframes
    )


def get_citl_mask_generator_config() -> CITLRandomMaskGeneratorConfig:
    return CITLRandomMaskGeneratorConfig(strategy_name="citl_random")


def get_delta_interpolator_mask_generator_config() -> DeltaInterpolatorMaskGeneratorConfig:
    return DeltaInterpolatorMaskGeneratorConfig(strategy_name="delta_interpolator", period=5)


def get_masking_config(model_name) -> MaskingConfig:
    if model_name == "lstm":
        mask_generator = get_random_block_keyframes_mask_generator_config(0)
    elif model_name == "citl":
        mask_generator = get_citl_mask_generator_config()
    elif model_name == "delta_interpolator":
        mask_generator = get_delta_interpolator_mask_generator_config()
    else:
        raise ValueError(f"Masking config for model {model_name} not recognized")
    predict_mask_generator = ManualMaskGeneratorConfig(strategy_name="manual")
    test_mask_generator = get_random_block_keyframes_mask_generator_config(0)
    if model_name == "lstm" or model_name == "citl":
        masking_applier = MaskApplierConfig(concat_mode=ConcatMode.CONCAT_DIM, fill_mode=FillMode.ZEROS)
    elif model_name == "delta_interpolator":
        masking_applier = MaskApplierConfig(concat_mode=ConcatMode.CONCAT_DIM, fill_mode=FillMode.SLERP)
    else:
        raise ValueError(f"Masking applier for model {model_name} not recognized")
    return MaskingConfig(
        mask_generator=mask_generator,
        predict_mask_generator=predict_mask_generator,
        test_mask_generator=test_mask_generator,
        mask_applier=masking_applier,
    )


def get_optimization_config() -> OptimizationConfig:
    optimizer = OptimizerConfig(name="AdamW", lr=0.0001, tune_lr=False, params={"weight_decay": 0.001})
    lr_scheduler = get_cyclicLR_scheduler_config()
    regularization = RegularizationConfig(
        last_layer_regularization=False,
        last_layer_regularization_weight=0.001,
        full_model_regularization=False,
        full_model_regularization_weight=1e-08,
    )
    return OptimizationConfig(optimizer=optimizer, lr_scheduler=lr_scheduler, regularization=regularization)


def get_training_module_config(model_name: str, character_name: CharacterName) -> Seq2SeqTrainingModuleConfig:
    model_config = get_model_config(model_name)
    masking_config = get_masking_config(model_name)
    optimization_config = get_optimization_config()
    losses_config = get_losses_config()
    do_controller_keyframes_prediction = False
    post_processing_config = get_default_post_processing_config(do_controller_keyframes_prediction)
    training_module = Seq2SeqTrainingModuleConfig(
        character_name=character_name,
        module_name="seq2seq",
        model=model_config,
        masking=masking_config,
        optimization=optimization_config,
        losses=losses_config,
        do_controller_keyframes_prediction=do_controller_keyframes_prediction,
        post_processing=post_processing_config,
    )
    return training_module


def get_losses_config() -> LossesConfig:
    return LossesConfig(
        pose_loss=L1Loss(reduction="none"),
        speed_loss=L1Loss(reduction="mean"),
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
        auto_weighting_loss=True,
    )


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


def get_controllers_config(character_name: CharacterName) -> ControllerConfig:
    return ControllerConfig(
        default_controllers_path=get_pcy_default_controllers_path(character_name),
        trainable_controllers_path=get_pcy_trainable_controllers_path(character_name),
        simplify_classes_path=None,
    )


def get_default_config(model_name: str, character_name: CharacterName) -> GlobalTrainingConfig:
    logging = get_logging_config(character_name)
    data_module = get_data_module_config(character_name)
    training_module = get_training_module_config(model_name, character_name)
    train = TrainConfig(
        max_epochs=600, precision="16-mixed", accumulate_grad_batches=1, val_check_interval=1.0, gradient_clip_val=0.0
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


def get_debug_config(model_name: str, character_name: CharacterName) -> GlobalTrainingConfig:
    """Debug config, used to test the training pipeline

    Args:
        remote (bool): Whether to use the remote or local config

    Returns:
        GlobalTrainingConfig: The debug config
    """
    config = get_delta_interpolator_best_model_config(character_name)
    config.logging.project_name = f"MIB_{character_name}_dev"
    config.logging.experiment_name = "delta_interpolator_AIS"
    config.train_module.model.last_layer_mode = LastLayerMode.AIS
    config.train_module.losses.auto_weighting_loss = True
    config.training.max_epochs = 1
    config.train_module.optimization.optimizer.tune_lr = False
    config.data_module.train_dataset.subset_size = -1
    config.callbacks.model_checkpoint_params.every_n_epochs = 1
    return config


def get_best_model_config(model_name: str, character_name: CharacterName) -> GlobalTrainingConfig:
    """Config of current best model

    Returns:
        GlobalTrainingConfig: The config of the best model
    """
    if model_name == "lstm":
        config = get_lstm_best_model_config(character_name)
    elif model_name == "citl":
        config = get_best_citl_model_config(character_name)
    elif model_name == "delta_interpolator":
        config = get_delta_interpolator_best_model_config(character_name)
    else:
        raise ValueError(f"Best model config for {model_name} not recognized")
    config.logging.project_name = "MIB_paper_SOTA_best_models"
    config.logging.experiment_name = f"best_{model_name}"
    config.callbacks.model_checkpoint_params.every_n_epochs = 1
    return config


def get_lstm_best_model_config(character_name: CharacterName) -> GlobalTrainingConfig:
    """Config of current best LSTM model

    Returns:
        GlobalTrainingConfig: The config of the best LSTM model
    """
    config = get_default_config("lstm", character_name)
    config.train_module.model.hidden_size = 512
    config.train_module.model.num_layers = 2
    config.train_module.model.dropout = 0.3003
    config.train_module.optimization.lr_scheduler = SchedulerConfig(
        name="CyclicLR", params={"base_lr": 6.897e-07, "max_lr": 0.001805, "step_size_up": 100, "mode": "triangular2"}
    )
    config.data_module.batch_size = 64
    config.training.max_epochs = 800
    return config


def get_best_citl_model_config(character_name: CharacterName) -> GlobalTrainingConfig:
    """Config of current best CITL model

    Returns:
        GlobalTrainingConfig: The config of the best CITL model
    """
    config = get_default_config("citl", character_name)
    config.train_module.model.embed_size = 1024
    config.train_module.model.max_length = 224
    config.train_module.model.heads = 8
    config.train_module.model.key_layers = 2
    config.train_module.model.interm_layers = 2
    config.train_module.model.dec_layers = 4
    config.train_module.model.dropout = 0.01757
    config.train_module.optimization.optimizer.lr = 3.1232112144020485e-05
    config.train_module.optimization.lr_scheduler = SchedulerConfig(
        name="ReduceLROnPlateau", params={"mode": "min", "factor": 0.9868, "patience": 100}
    )
    config.data_module.batch_size = 32
    config.training.max_epochs = 800
    return config


def get_delta_interpolator_best_model_config(character_name: CharacterName) -> GlobalTrainingConfig:
    """Config of current best Delta Interpolator model

    Returns:
        GlobalTrainingConfig: The config of the best Delta Interpolator model
    """
    config = get_default_config("delta_interpolator", character_name)
    config.train_module.model = DeltaInterpolatorModelConfig(
        num_blocks_enc=5,
        num_layers_enc=1,
        layer_width_enc=512,
        num_blocks_dec=0,
        num_layers_dec=1,
        layer_width_dec=1024,
        dropout=0.7152,
        embedding_dim=32,
        embedding_size=256,
        embedding_num=2,
        num_heads=4,
        delta_mode="interpolator",
        input_delta_mode="last_pose",
        last_layer_mode=LastLayerMode.ORIGINAL,
    )
    config.train_module.optimization.lr_scheduler = SchedulerConfig(
        name="ReduceLROnPlateau", params={"mode": "min", "factor": 0.5446, "patience": 15}
    )
    config.train_module.optimization.optimizer.lr = 4.4582123677323057e-05
    config.data_module.batch_size = 16
    config.training.max_epochs = 800
    return config
