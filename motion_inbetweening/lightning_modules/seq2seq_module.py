import torch

from motion_inbetweening.app_services.online_preprocessing.mask_generator import generate_mask, generate_padding_mask
from motion_inbetweening.app_services.online_preprocessing.padding import apply_padding_mask
from motion_inbetweening.config.controller_keyframes_preprocessing import TransformationsKeyframesDivisionStrategy
from motion_inbetweening.config.train import Seq2SeqTrainingModuleConfig
from motion_inbetweening.controller_keyframes.metrics import create_metrics_for_controller_keyframes
from motion_inbetweening.controller_keyframes.prediction_loss import compute_keyframe_prediction_loss
from motion_inbetweening.controller_keyframes.projection import initialize_controller_keyframes_projection
from motion_inbetweening.controller_keyframes.weights import compute_controller_keyframes_weights
from motion_inbetweening.domain.mask_applier import FillMode, MaskApplier
from motion_inbetweening.domain.models.citl.inpainter import Inpainter
from motion_inbetweening.domain.models.delta_interpolator.modules.infill_transformer import (
    DeltaInterpolatorInfillTransformer,
)
from motion_inbetweening.domain.models.explicit_interpolation_extrapolation_residual_lstm import (
    ExplicitInterpolationExtrapolationResidualLSTM,
)
from motion_inbetweening.domain.models.factory import initialize_model
from motion_inbetweening.lightning_modules.base_module import BaseModule
from motion_inbetweening.lightning_modules.utils import reshape_sequence
from motion_inbetweening.losses.factory import loss_factory
from motion_inbetweening.losses.frame_weights import compute_frame_weights
from motion_inbetweening.losses.movement_losses import compute_acceleration_loss, compute_jerk_loss, compute_speed_loss
from shared.rig.config import ControllerConfig
from shared.rig.trainable_controllers import TrainableController, split_vectors_using_controllers


def extract_keyposes(ground_truth_sequence: torch.Tensor, mask: torch.Tensor):
    """
    Extracts key poses (unmasked frames) and their indices.

    Args:
        ground_truth_sequence (torch.Tensor): Tensor of shape (B, T, D)
        mask (torch.Tensor): Boolean tensor of shape (B, T), where False indicates unmasked frames

    Returns:
        keyposes (torch.Tensor): (B, N, D) tensor of unmasked poses
        keyframes (torch.Tensor): (B, N) tensor of indices of the unmasked poses
    """
    B, T, D = ground_truth_sequence.shape
    indices = torch.arange(T, device=ground_truth_sequence.device).unsqueeze(0).expand(B, -1)
    unmasked = ~mask
    keyposes = ground_truth_sequence[unmasked].view(B, -1, D)
    keyframes = indices[unmasked].view(B, -1)
    return (keyposes, keyframes)


class Seq2SeqModule(BaseModule):
    def __init__(
        self,
        config: Seq2SeqTrainingModuleConfig,
        controllers_config: ControllerConfig,
        trainable_controllers: list[TrainableController],
        mean_pose_normalizer: torch.Tensor | None,
        std_pose_normalizer: torch.Tensor | None,
        transformations_keyframe_division_strategy: TransformationsKeyframesDivisionStrategy | None,
    ):
        super().__init__(config, controllers_config, trainable_controllers, mean_pose_normalizer, std_pose_normalizer)
        self.do_controller_keyframes_prediction = config.do_controller_keyframes_prediction
        if config.do_controller_keyframes_prediction:
            if transformations_keyframe_division_strategy is None:
                raise ValueError(
                    "transformations_keyframe_division_strategy should be provided "
                    "if controller keyframes prediction is enabled"
                )
            self.controller_keyframes_projection = initialize_controller_keyframes_projection(
                trainable_controllers=trainable_controllers,
                transformations_keyframe_division_strategy=transformations_keyframe_division_strategy,
            )
            self.controller_keyframes_prediction_loss = loss_factory(config.losses.controller_keyframes_prediction_loss)
            self.threshold = 0.5
            self.controller_keyframes_metrics_quantitative, self.controller_keyframes_metrics_plot = (
                create_metrics_for_controller_keyframes(threshold=self.threshold)
            )
        self.model = initialize_model(
            model_config=config.model,
            trainable_controllers=trainable_controllers,
            do_controller_keyframes_prediction=config.do_controller_keyframes_prediction
            if config.do_controller_keyframes_prediction
            else False,
            controller_keyframes_dim=self.controller_keyframes_projection.dim_controller_keyframes
            if config.do_controller_keyframes_prediction
            else 0,
        )
        self.mask_generator_config = config.masking.mask_generator
        self.mask_applier = MaskApplier(config.masking.mask_applier)
        self.test_mask_generator_config = config.masking.test_mask_generator
        self.predict_mask_generator_config = config.masking.predict_mask_generator
        self.speed_loss = loss_factory(config.losses.speed_loss)
        self.acceleration_loss = loss_factory(config.losses.acceleration_loss)
        self.jerk_loss = loss_factory(config.losses.jerk_loss)

    def loss_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.IntTensor],
        prefix: str,
        log_details: bool,
    ) -> torch.Tensor:
        (
            ground_truth_sequence,
            animation_keyframes,
            block_keyframes,
            padding_mask,
            controller_keyframes,
            sequence_lengths,
        ) = batch
        mask = generate_mask(
            config=self.mask_generator_config,
            movement_torch=ground_truth_sequence,
            sequence_lengths=sequence_lengths,
            animation_keyframes=animation_keyframes,
            block_keyframes=block_keyframes,
            list_unmasked_frames=None,
        )
        if isinstance(self.model, ExplicitInterpolationExtrapolationResidualLSTM):
            predicted_sequence, _ = self.model(ground_truth_sequence, mask)
        elif isinstance(self.model, Inpainter):
            keyposes, keyframes = extract_keyposes(ground_truth_sequence, mask)
            predicted_sequence = self.model(
                keyposes=keyposes,
                keyframes=keyframes,
                frames=224,
                ground_truth_sequence=ground_truth_sequence,
                mask=mask,
            )
            predicted_keyframes = None
        elif isinstance(self.model, DeltaInterpolatorInfillTransformer):
            batch_size, seq_len, pose_dim = ground_truth_sequence.shape
            keyposes, keyframes = extract_keyposes(ground_truth_sequence, mask)
            input_poses = keyposes
            past_frame_indices = keyframes[:, 0].unsqueeze(1)
            key_frame_indices = keyframes[:, 1:-1]
            future_frame_indices = keyframes[:, -1].unsqueeze(1)
            target_frame_indices = torch.stack(
                [torch.tensor([i for i in range(seq_len) if i not in k], device=keyframes.device) for k in keyframes]
            )
            assert self.mask_applier.fill_mode == FillMode.SLERP
            interpolated_sequence = self.mask_applier.fill_masked_sequence(
                ground_truth_sequence, mask, trainable_controllers=self.trainable_controllers
            )
            if torch.isnan(interpolated_sequence).any():
                raise ValueError("Interpolated sequence contains NaN values")
            predicted_sequence = self.model(
                input_poses=input_poses,
                interpolated_sequence=interpolated_sequence,
                past_frame_indices=past_frame_indices,
                key_frame_indices=key_frame_indices,
                future_frame_indices=future_frame_indices,
                target_frame_indices=target_frame_indices,
                input_sequence=ground_truth_sequence,
                mask=mask,
            )
            predicted_keyframes = None
        else:
            masked_sequence = self.mask_applier.apply_mask(ground_truth_sequence.half(), mask)
            predicted_sequence, predicted_keyframes = self.model(masked_sequence, mask)
        predicted_sequence_padded = apply_padding_mask(predicted_sequence, padding_mask)
        gt_controllers_vectors = reshape_sequence(ground_truth_sequence)
        predict_controllers_vectors = reshape_sequence(predicted_sequence_padded)
        effective_batch_size = gt_controllers_vectors.size(0)
        frame_weights_reshape = None
        if self.config.losses.weights.frame_weights is not None:
            padding_mask_reshape = generate_padding_mask(ground_truth_sequence, sequence_lengths)
            frame_weights = compute_frame_weights(
                losses_frame_weights=self.config.losses.weights.frame_weights,
                mask=mask,
                animation_keyframes=animation_keyframes,
                padding_mask=padding_mask_reshape,
            )
            frame_weights_reshape = frame_weights.view(-1)
        controller_keyframes_weights_reshaped = None
        if self.config.losses.weights.controller_keyframes_weights is not None:
            padding_mask_reshape = generate_padding_mask(ground_truth_sequence, sequence_lengths)
            controller_keyframes_weights = compute_controller_keyframes_weights(
                controller_keyframes_vector=controller_keyframes,
                controller_keyframes_projection=self.controller_keyframes_projection,
                controller_keyframes_weight_function=self.config.losses.weights.controller_keyframes_weights.weight_function,
                padding_mask=padding_mask_reshape,
            )
            controller_keyframes_weights_reshaped = reshape_sequence(controller_keyframes_weights)
        pose_controllers_loss_value = self.compute_pose_controllers_loss(
            gt_controllers_vectors=gt_controllers_vectors,
            predict_controllers_vectors=predict_controllers_vectors,
            batch_size=effective_batch_size,
            prefix=prefix,
            log_details=log_details,
            frame_weights=frame_weights_reshape,
            controller_keyframes_weights=controller_keyframes_weights_reshaped,
        )
        self.log(f"{prefix}/pose_loss", pose_controllers_loss_value, prog_bar=False, on_step=False, on_epoch=True)
        loss = self.config.losses.weights.pose * pose_controllers_loss_value
        movement_loss = self.compute_movement_losses(
            ground_truth_sequence=ground_truth_sequence,
            predicted_sequence=predicted_sequence_padded,
            sequence_lengths=sequence_lengths,
            prefix=prefix,
        )
        loss += movement_loss
        if self.do_controller_keyframes_prediction:
            loss_value_controller_keyframes_prediction = compute_keyframe_prediction_loss(
                predicted_keyframes=predicted_keyframes,
                controller_keyframes=controller_keyframes,
                sequence_lengths=sequence_lengths,
                loss_function=self.controller_keyframes_prediction_loss,
            )
            self.log(
                f"{prefix}/controller_keyframes_loss",
                loss_value_controller_keyframes_prediction,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )
            loss += (
                self.config.losses.weights.controller_keyframes_prediction * loss_value_controller_keyframes_prediction
            )
        self.log(f"{prefix}/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def compute_pose_controllers_loss(
        self,
        gt_controllers_vectors: torch.Tensor,
        predict_controllers_vectors: torch.Tensor,
        batch_size: int,
        prefix: str,
        log_details: bool,
        frame_weights: torch.Tensor | None,
        controller_keyframes_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Compute the loss for the given ground truth and predicted controllers vectors
        Args:
            gt_controllers_vectors: ground truth controllers vectors of shape (batch_size * seq_len, pose_dim)
            predict_controllers_vectors: predicted controllers vectors of shape (batch_size * seq_len, pose_dim)
            batch_size: effective batch size
            prefix: prefix to use for logging
            log_details: whether to log the details of the loss
            frame_weights: frame weights to use for the loss of shape (batch_size * seq_len)
            controller_keyframes_weights: controller keyframes weights to use for the loss of shape
                (batch_size * seq_len, pose_dim)
        """
        split_gt, split_pred, coeffs, _, controller_names = split_vectors_using_controllers(
            self.trainable_controllers, gt_controllers_vectors, predict_controllers_vectors, False
        )
        if controller_keyframes_weights is not None:
            split_controller_keyframes_weights, _, _, _, _ = split_vectors_using_controllers(
                self.trainable_controllers, controller_keyframes_weights, controller_keyframes_weights, False
            )
        total_loss = 0
        total_losses = []
        for idx, (loss, pred, gt, coeff, controller_name) in enumerate(
            zip(self.pose_controllers_losses, split_pred, split_gt, coeffs, controller_names, strict=True)
        ):
            controller_keyframes_weight = (
                split_controller_keyframes_weights[idx] if controller_keyframes_weights is not None else None
            )
            loss_value = loss(pred, gt)
            if controller_keyframes_weight is not None:
                loss_value = loss_value * controller_keyframes_weight
            loss_value = loss_value.mean(dim=-1)
            if frame_weights is not None:
                loss_value = loss_value * frame_weights
            loss_value = loss_value.mean()
            if log_details:
                self.log(
                    f"{prefix}/details/loss_{controller_name}",
                    loss_value,
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    batch_size=batch_size,
                )
            if self.config.losses.auto_weighting_loss:
                total_losses.append(loss_value)
            else:
                total_loss += coeff * loss_value
        if self.config.losses.auto_weighting_loss:
            total_loss = self.auto_weighting_loss(total_losses, [])
        return total_loss

    def compute_movement_losses(
        self,
        ground_truth_sequence: torch.Tensor,
        predicted_sequence: torch.Tensor,
        sequence_lengths: torch.Tensor,
        prefix: str,
    ):
        """
        Compute the movement losses for the given ground truth and predicted sequences
        Args:
            ground_truth_sequence: ground truth sequence of shape (batch_size, seq_len, pose_dim)
            predicted_sequence: predicted sequence of shape (batch_size, seq_len, pose_dim)
            sequence_lengths: sequence lengths tensor of shape (batch_size)
            prefix: prefix to use for logging
        Returns:
            loss: total movement loss
        """
        loss = 0
        if self.config.losses.weights.speed > 0:
            speed_loss = self.compute_speed_loss(
                ground_truth_sequence=ground_truth_sequence,
                predicted_sequence=predicted_sequence,
                sequence_lengths=sequence_lengths,
            )
            self.log(f"{prefix}/speed_loss", speed_loss, prog_bar=False, on_step=False, on_epoch=True)
            loss += self.config.losses.weights.speed * speed_loss
        if self.config.losses.weights.acceleration > 0:
            acceleration_loss = self.compute_acceleration_loss(
                ground_truth_sequence=ground_truth_sequence,
                predicted_sequence=predicted_sequence,
                sequence_lengths=sequence_lengths,
            )
            self.log(f"{prefix}/acceleration_loss", acceleration_loss, prog_bar=False, on_step=False, on_epoch=True)
            loss += self.config.losses.weights.acceleration * acceleration_loss
        if self.config.losses.weights.jerk > 0:
            jerk_loss = self.compute_jerk_loss(
                ground_truth_sequence=ground_truth_sequence,
                predicted_sequence=predicted_sequence,
                sequence_lengths=sequence_lengths,
            )
            self.log(f"{prefix}/jerk_loss", jerk_loss, prog_bar=False, on_step=False, on_epoch=True)
            loss += self.config.losses.weights.jerk * jerk_loss
        return loss

    def compute_speed_loss(
        self, ground_truth_sequence: torch.Tensor, predicted_sequence: torch.Tensor, sequence_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the speed loss for the given ground truth and predicted sequences
        Args:
            ground_truth_sequence: ground truth sequence of shape (batch_size, seq_len, pose_dim)
            predicted_sequence: predicted sequence of shape (batch_size, seq_len, pose_dim)
            sequence_lengths: sequence lengths tensor of shape (batch_size)
        Returns:
            speed_loss: loss between the ground truth speed and predicted speed
        """
        return compute_speed_loss(
            loss_fn=self.speed_loss,
            ground_truth_sequence=ground_truth_sequence,
            predicted_sequence=predicted_sequence,
            sequence_lengths=sequence_lengths,
        )

    def compute_acceleration_loss(
        self, ground_truth_sequence: torch.Tensor, predicted_sequence: torch.Tensor, sequence_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the acceleration loss for the given ground truth and predicted sequences
        Args:
            ground_truth_sequence: ground truth sequence of shape (batch_size, seq_len, pose_dim)
            predicted_sequence: predicted sequence of shape (batch_size, seq_len, pose_dim)
            sequence_lengths: sequence lengths tensor of shape (batch_size)
        Returns:
            acceleration_loss: loss between the ground truth acceleration and predicted acceleration
        """
        return compute_acceleration_loss(
            loss_fn=self.acceleration_loss,
            ground_truth_sequence=ground_truth_sequence,
            predicted_sequence=predicted_sequence,
            sequence_lengths=sequence_lengths,
        )

    def compute_jerk_loss(
        self, ground_truth_sequence: torch.Tensor, predicted_sequence: torch.Tensor, sequence_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the jerk loss for the given ground truth and predicted sequences
        Args:
            ground_truth_sequence: ground truth sequence of shape (batch_size, seq_len, pose_dim)
            predicted_sequence: predicted sequence of shape (batch_size, seq_len, pose_dim)
            sequence_lengths: sequence lengths tensor of shape (batch_size)
        Returns:
            jerk_loss: loss between the ground truth jerk and predicted jerk
        """
        return compute_jerk_loss(
            loss_fn=self.jerk_loss,
            ground_truth_sequence=ground_truth_sequence,
            predicted_sequence=predicted_sequence,
            sequence_lengths=sequence_lengths,
        )

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss = self.loss_step(batch, prefix="train", log_details=False)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        loss = self.loss_step(batch, prefix="val", log_details=True)
        return loss

    def sequence_inbetweening(
        self, input_sequence: torch.Tensor, mask: torch.Tensor, sequence_lengths: torch.Tensor
    ) -> torch.Tensor:
        details = None
        if isinstance(self.model, ExplicitInterpolationExtrapolationResidualLSTM):
            batch_predicted_sequences, batch_predicted_keyframes, details = self.model(
                input_sequence, mask, return_details=True
            )
        elif isinstance(self.model, Inpainter):
            batch_predicted_sequences_list = []
            for i in range(input_sequence.size(0)):
                input_seq_i = input_sequence[i].unsqueeze(0)
                mask_i = mask[i].unsqueeze(0)
                keyposes, keyframes = extract_keyposes(input_seq_i, mask_i)
                pred_seq = self.model(
                    keyposes=keyposes, keyframes=keyframes, frames=224, ground_truth_sequence=input_seq_i, mask=mask_i
                )
                batch_predicted_sequences_list.append(pred_seq)
            batch_predicted_sequences = torch.cat(batch_predicted_sequences_list, dim=0)
            batch_predicted_keyframes = None
        elif isinstance(self.model, DeltaInterpolatorInfillTransformer):
            batch_predicted_sequences_list = []
            batch_size, seq_len, pose_dim = input_sequence.shape
            for i in range(batch_size):
                input_seq_i = input_sequence[i].unsqueeze(0)
                mask_i = mask[i].unsqueeze(0)
                keyposes, keyframes = extract_keyposes(input_seq_i, mask_i)
                input_poses = keyposes
                past_frame_indices = keyframes[:, 0].unsqueeze(1)
                key_frame_indices = keyframes[:, 1:-1]
                future_frame_indices = keyframes[:, -1].unsqueeze(1)
                target_frame_indices = torch.stack(
                    [
                        torch.tensor([j for j in range(seq_len) if j not in k], device=keyframes.device)
                        for k in keyframes
                    ]
                )
                assert self.mask_applier.fill_mode == FillMode.SLERP
                interpolated_sequence = self.mask_applier.fill_masked_sequence(
                    input_seq_i, mask_i, trainable_controllers=self.trainable_controllers
                )
                if torch.isnan(interpolated_sequence).any():
                    raise ValueError("Interpolated sequence contains NaN values")
                pred_seq = self.model(
                    input_poses=input_poses,
                    interpolated_sequence=interpolated_sequence,
                    past_frame_indices=past_frame_indices,
                    key_frame_indices=key_frame_indices,
                    future_frame_indices=future_frame_indices,
                    target_frame_indices=target_frame_indices,
                    input_sequence=input_seq_i,
                    mask=mask_i,
                )
                batch_predicted_sequences_list.append(pred_seq)
            batch_predicted_sequences = torch.cat(batch_predicted_sequences_list, dim=0)
            batch_predicted_keyframes = None
        else:
            masked_sequence = self.mask_applier.apply_mask(input_sequence, mask)
            batch_predicted_sequences, batch_predicted_keyframes = self.model(masked_sequence, mask)
        return (batch_predicted_sequences, batch_predicted_keyframes, details)
