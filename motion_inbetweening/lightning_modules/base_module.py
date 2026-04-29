import functools
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import MLFlowLogger
from torch import optim

from motion_inbetweening.app_services.online_preprocessing.mask_generator import MaskGeneratorConfig, generate_mask
from motion_inbetweening.config.train import DiffusionTrainingModuleConfig, Seq2SeqTrainingModuleConfig
from motion_inbetweening.controller_keyframes.vectorization import controller_keyframes_from_vector
from motion_inbetweening.domain.controller_keyframes import RigControllersKeyframes
from motion_inbetweening.domain.data.data_sample import AnimationSceneData
from motion_inbetweening.domain.mask_applier import MaskApplier
from motion_inbetweening.losses.factory import loss_factory
from motion_inbetweening.metric.npss import NPSS
from motion_inbetweening.metric.shifted_distance import compute_batch_shifted_motion_distance
from shared.domain.entities.rig_controllers_values import RigControllersValues
from shared.losses.weighted_loss import MultiTaskLoss
from shared.rig.config import ControllerConfig
from shared.rig.default_controllers import load_default_controllers
from shared.rig.normalizer import PoseNormalizer, apply_normalization_padded_sequences
from shared.rig.rig_controllers import vec_to_rig
from shared.rig.trainable_controllers import TrainableController


def vec_to_rig_wrapper(frame_vec, trainable_controllers):
    return RigControllersValues.from_any(vec_to_rig(trainable_controllers, frame_vec, defaultdict(dict)))


class BaseModule(pl.LightningModule):
    def __init__(
        self,
        config: Seq2SeqTrainingModuleConfig | DiffusionTrainingModuleConfig,
        controllers_config: ControllerConfig,
        trainable_controllers: list[TrainableController],
        mean_pose_normalizer: torch.Tensor | None,
        std_pose_normalizer: torch.Tensor | None,
    ) -> None:
        super().__init__()
        self.config = config
        self.mask_generator_config = config.masking.mask_generator
        self.mask_applier = MaskApplier(config.masking.mask_applier)
        self.test_mask_generator_config = config.masking.test_mask_generator
        self.predict_mask_generator_config = config.masking.predict_mask_generator
        self.trainable_controllers = trainable_controllers
        self.default_controllers = load_default_controllers(controllers_config.default_controllers_path)
        self.lr = self.config.optimization.optimizer.lr
        self.mask_generator_config = config.masking.mask_generator
        self.mask_applier = MaskApplier(config.masking.mask_applier)
        self.predict_mask_generator_config = config.masking.predict_mask_generator
        self.save_hyperparameters(logger=True)
        if mean_pose_normalizer is not None and std_pose_normalizer is not None:
            self.pose_normalizer_training_module = PoseNormalizer(mean_pose_normalizer, std_pose_normalizer)
        else:
            self.pose_normalizer_training_module = None
        self.auto_weighting_loss = None
        self.pose_controllers_losses, num_losses = self.create_pose_controllers_losses()
        if self.config.losses.auto_weighting_loss:
            self.auto_weighting_loss = MultiTaskLoss(n_losses=num_losses)
        self.npss = NPSS()

    def configure_optimizers(self):
        optimizer_class = getattr(optim, self.config.optimization.optimizer.name)
        optimizer = optimizer_class(
            params=filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
            **self.config.optimization.optimizer.params,
        )
        if self.auto_weighting_loss is not None:
            criterion_params = filter(lambda p: p.requires_grad, self.auto_weighting_loss.parameters())
            optimizer.add_param_group({"params": criterion_params})
        optimizer_dict = {"optimizer": optimizer}
        if self.config.optimization.lr_scheduler is not None:
            lr_scheduler_class = getattr(optim.lr_scheduler, self.config.optimization.lr_scheduler.name)
            optimizer_dict["lr_scheduler"] = lr_scheduler_class(
                optimizer, **self.config.optimization.lr_scheduler.params
            )
        if self.config.optimization.lr_scheduler.name == "ReduceLROnPlateau":
            optimizer_dict["monitor"] = "val/loss"
        return optimizer_dict

    def create_pose_controllers_losses(self) -> tuple[list[torch.nn.Module], int]:
        pose_loss_function = loss_factory(self.config.losses.pose_loss)
        pose_controllers_losses = []
        num_losses = 0
        for controller in self.trainable_controllers:
            for _ in controller.transformations:
                pose_controllers_losses.append(pose_loss_function)
                num_losses += 1
        return (pose_controllers_losses, num_losses)

    def test_step(self, batch, batch_idx):
        (
            ground_truth_sequence,
            animation_keyframes,
            block_keyframes,
            padding_mask,
            controller_keyframes,
            sequence_lengths,
        ) = batch
        mask = generate_mask(
            config=self.test_mask_generator_config,
            movement_torch=ground_truth_sequence,
            sequence_lengths=sequence_lengths,
            animation_keyframes=animation_keyframes,
            block_keyframes=block_keyframes,
            list_unmasked_frames=None,
        )
        list_unmasked_frames = []
        for mask_seq in mask:
            list_unmasked_frames.append([i for i, m in enumerate(mask_seq) if m == 0])
        predicted_sequence, predicted_keyframes, _ = self.sequence_inbetweening(
            ground_truth_sequence, mask, sequence_lengths=sequence_lengths
        )
        if self.pose_normalizer_training_module is not None:
            predicted_sequence = self.pose_normalizer_training_module.denormalize(predicted_sequence)
            ground_truth_sequence = self.pose_normalizer_training_module.denormalize(ground_truth_sequence)
        shifted_distance = compute_batch_shifted_motion_distance(
            ground_truth_sequences=ground_truth_sequence,
            predicted_sequences=predicted_sequence,
            animation_keyframes=animation_keyframes,
            unmasked_frames=list_unmasked_frames,
        )
        self.log("test/shifted_distance", shifted_distance, prog_bar=True, on_step=False, on_epoch=True)
        self.npss.update(predicted_sequence, ground_truth_sequence, sequence_lengths=sequence_lengths)
        if self.do_controller_keyframes_prediction:
            controller_keyframes_int = (controller_keyframes > self.threshold).int()
            predicted_keyframes = torch.sigmoid(predicted_keyframes)
            for metric_name, metric in self.controller_keyframes_metrics_quantitative.items():
                metric(predicted_keyframes, controller_keyframes_int)
                self.log(f"test/{metric_name}", metric, prog_bar=True, on_step=False, on_epoch=True)
            for _, metric in self.controller_keyframes_metrics_plot.items():
                metric.update(predicted_keyframes, controller_keyframes_int)

    def on_test_epoch_end(self):
        if self.do_controller_keyframes_prediction:
            for metric_name, metric in self.controller_keyframes_metrics_plot.items():
                fig_, _ = metric.plot()
                if isinstance(self.logger, MLFlowLogger):
                    self.logger.experiment.log_figure(
                        self.logger.run_id, fig_, f"controller_keyframes_curves/{metric_name}.png"
                    )
        self.log("test/npss", self.npss.compute())
        self.npss.reset()

    def predict_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[int]], list[AnimationSceneData]],
        batch_idx,
    ) -> tuple[list[list[RigControllersValues]], list[list[RigControllersKeyframes]] | None, list[AnimationSceneData]]:
        input_sequence, _, sequence_lengths, list_unmasked_frames_indices, scenes = batch
        batch_predicted_sequences, batch_predicted_keyframes, details = predict_step(
            input_sequence,
            sequence_lengths,
            list_unmasked_frames_indices,
            self.pose_normalizer_training_module,
            self,
            self.mask_applier,
            self.predict_mask_generator_config,
        )
        all_scene_predicted_rigs = []
        for predicted_sequence in batch_predicted_sequences:
            predicted_sequence = predicted_sequence.detach().cpu()
            convert_func = functools.partial(vec_to_rig_wrapper, trainable_controllers=self.trainable_controllers)
            with ProcessPoolExecutor(2) as executor:
                predicted_scene_rigs = list(executor.map(convert_func, predicted_sequence))
            all_scene_predicted_rigs.append(predicted_scene_rigs)
        if self.do_controller_keyframes_prediction:
            batch_predicted_keyframes_normalized = torch.sigmoid(batch_predicted_keyframes)
            all_scene_predicted_controller_keyframes = []
            for predicted_controller_keyframes_vectorized in batch_predicted_keyframes_normalized:
                predicted_controller_keyframes = [
                    controller_keyframes_from_vector(
                        trainable_controllers=self.trainable_controllers,
                        transformations_keyframe_division_strategy=self.controller_keyframes_projection.transformations_keyframes_division_strategy,
                        vec=vec,
                    )
                    for vec in predicted_controller_keyframes_vectorized.cpu().numpy()
                ]
                all_scene_predicted_controller_keyframes.append(predicted_controller_keyframes)
        else:
            all_scene_predicted_controller_keyframes = None
        return (all_scene_predicted_rigs, all_scene_predicted_controller_keyframes, scenes, details)

    def sequence_inbetweening(
        self, input_sequence: torch.Tensor, mask: torch.Tensor, sequence_lengths: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError("This method should be implemented in the child class")

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, *args, **kwargs):
        """
        Set weights_only to False by default to be able to load the whole model (default changed to True in pytorch
            2.6.0)
        """
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return super().load_from_checkpoint(checkpoint_path, *args, **kwargs)


def predict_step(
    input_sequence: torch.Tensor,
    sequence_lengths: torch.Tensor,
    list_unmasked_frames_indices: list[list[int]],
    pose_normalizer: PoseNormalizer | None,
    module: BaseModule,
    mask_applier: MaskApplier,
    predict_mask_generator_config: MaskGeneratorConfig,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply model inference: generate the mask from the list of unmasked frames indices, apply the mask to the input
    sequence, and run the model on the masked sequence.

    There is also an optional step of normalization/denormalization

    Args:
        input_sequence (torch.Tensor): Input movement, with shape (batch_size, seq_len, pose_dim)
        sequence_lengths (torch.Tensor): Sequence lengths, with shape (batch_size)
        list_unmasked_frames_indices (list[list[int]]): List of unmasked frames indices for each sequence in the batch
        pose_normalizer (PoseNormalizer | None): Pose normalizer to apply normalization/denormalization
        model (torch.nn.Module) : Model to use for inference
        mask_applier (MaskApplier): Mask applier
        predict_mask_generator_config (MaskGeneratorConfig): Mask generator

    Returns:
        tuple[torch.Tensor, torch.Tensor | None]: Predicted movement of shape (batch_size, seq_len, pose_dim) and
            optionally predicted keyframes of the same shape
    """
    if pose_normalizer is not None:
        input_sequence = apply_normalization_padded_sequences(input_sequence, sequence_lengths, pose_normalizer)
    mask = generate_mask(
        config=predict_mask_generator_config,
        movement_torch=input_sequence,
        sequence_lengths=sequence_lengths,
        list_unmasked_frames=list_unmasked_frames_indices,
    )
    batch_predicted_sequences, batch_predicted_keyframes, details = module.sequence_inbetweening(
        input_sequence=input_sequence, mask=mask, sequence_lengths=sequence_lengths
    )
    if pose_normalizer is not None:
        batch_predicted_sequences = pose_normalizer.denormalize(batch_predicted_sequences)
    return (batch_predicted_sequences, batch_predicted_keyframes, details)
