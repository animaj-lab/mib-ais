import torch
from torch.optim.swa_utils import AveragedModel, get_ema_avg_fn

from motion_inbetweening.app_services.online_preprocessing.mask_generator import generate_mask, generate_padding_mask
from motion_inbetweening.config.gaussian_diffusion import SpacedDiffusionConfig
from motion_inbetweening.config.inference_diffusion import InferenceDiffusionConfig
from motion_inbetweening.config.mdm import MotionDiffusionModelConfig
from motion_inbetweening.config.train import DiffusionTrainingModuleConfig
from motion_inbetweening.diffusion_utils.conditioning import (
    BatchSequenceConditioning,
    update_conditioning_with_inference_params,
)
from motion_inbetweening.diffusion_utils.mdm_architectures.mdm_unet import MDM_UNET
from motion_inbetweening.diffusion_utils.resample import create_named_schedule_sampler
from motion_inbetweening.diffusion_utils.respace import SpacedDiffusion
from motion_inbetweening.domain.mask_applier import MaskApplier, MaskApplierConfig
from motion_inbetweening.domain.models.diffusion_model import load_diffusion_model
from motion_inbetweening.lightning_modules.base_module import BaseModule
from motion_inbetweening.losses.frame_weights import compute_frame_weights
from shared.rig.config import ControllerConfig
from shared.rig.normalizer import PoseNormalizer
from shared.rig.trainable_controllers import TrainableController, create_controllers_weights_vector, to_full_vector_size


class DiffusionModule(BaseModule):
    def __init__(
        self,
        config: DiffusionTrainingModuleConfig,
        controllers_config: ControllerConfig,
        trainable_controllers: list[TrainableController],
        pose_normalizer: PoseNormalizer | None,
    ):
        super().__init__(
            config, controllers_config, trainable_controllers, mean_pose_normalizer=None, std_pose_normalizer=None
        )
        self.pose_dim = to_full_vector_size(self.trainable_controllers)
        self.model = self._create_model(
            model_config=config.model,
            mask_applier_config=config.masking.mask_applier,
            learn_sigma=config.diffusion.learn_sigma,
        )
        self.diffusion = self._create_gaussian_diffusion(config.diffusion)
        self.schedule_sampler = create_named_schedule_sampler(config.scheduler.schedule_sampler_type, self.diffusion)
        self.mask_generator_config = config.masking.mask_generator
        self.mask_applier = MaskApplier(config.masking.mask_applier)
        self.predict_mask_generator_config = config.masking.predict_mask_generator
        self.use_fp16 = config.diffusion.fp16
        self.ema_alpha = 0.9999
        self.ema_model = AveragedModel(self.model, avg_fn=get_ema_avg_fn(self.ema_alpha))
        self.controllers_weights = create_controllers_weights_vector(controllers=self.trainable_controllers)
        self.inference_config = config.inference
        self.do_controller_keyframes_prediction = False

    def _create_model(
        self, model_config: MotionDiffusionModelConfig, mask_applier_config: MaskApplierConfig, learn_sigma: bool
    ) -> MDM_UNET:
        return load_diffusion_model(
            pose_dim=self.pose_dim,
            model_config=model_config,
            mask_applier_config=mask_applier_config,
            learn_sigma=learn_sigma,
        )

    def _create_gaussian_diffusion(self, spaced_diffusion_config: SpacedDiffusionConfig) -> SpacedDiffusion:
        return SpacedDiffusion(conf=spaced_diffusion_config, loss_conf=self.config.losses)

    def _set_inference_config(self, inference_config: InferenceDiffusionConfig) -> None:
        self.inference_config = inference_config

    def _loss_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.IntTensor],
        prefix: str,
    ):
        """
        Training step
        Args:
            batch: batch of data composed of: sequences, keyframes, padding_mask, lengths
            batch_idx: batch index"""
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
        conditioning = BatchSequenceConditioning.from_batch_attributes(
            sequence=ground_truth_sequence, mask=mask, sequence_lengths=sequence_lengths
        )
        input_motion = conditioning.observation_input_motion
        t, weights = self.schedule_sampler.sample(input_motion.shape[0], device=input_motion.device)
        controllers_weights = (
            self.controllers_weights.unsqueeze(0).unsqueeze(-1).expand(input_motion.shape).to(input_motion.device)
        )
        if self.config.losses.weights.frame_weights is not None:
            padding_mask_reshape = generate_padding_mask(ground_truth_sequence, sequence_lengths)
            frame_weights = compute_frame_weights(
                losses_frame_weights=self.config.losses.weights.frame_weights,
                mask=mask,
                animation_keyframes=animation_keyframes,
                padding_mask=padding_mask_reshape,
            )
            frame_weights = frame_weights.unsqueeze(1).expand(-1, self.pose_dim, -1)
        else:
            frame_weights = torch.ones_like(input_motion)
        losses = self.diffusion.training_losses(
            self.model,
            input_motion,
            t,
            model_kwargs=conditioning.to_model_kwargs(),
            controllers_weights=controllers_weights,
            frame_weights=frame_weights,
        )
        for loss_name, loss_value in losses.items():
            if loss_name != "loss":
                self.log(f"{prefix}/{loss_name}", loss_value.mean(), on_step=False, on_epoch=True, prog_bar=False)
        total_loss = (losses["loss"] * weights).mean()
        self.log(f"{prefix}/loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        return total_loss

    def optimizer_step(self, *args, **kwargs) -> None:
        super().optimizer_step(*args, **kwargs)
        self.ema_model.update_parameters(self.model)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.BoolTensor, torch.Tensor, torch.Tensor, torch.IntTensor], batch_idx
    ):
        return self._loss_step(batch, prefix="train")

    def validation_step(self, batch, batch_idx):
        return self._loss_step(batch, prefix="val")

    def sequence_inbetweening(
        self, input_sequence: torch.Tensor, mask: torch.Tensor, sequence_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        masked_sequence = self.mask_applier.fill_masked_sequence(sequence=input_sequence, mask=mask)
        batch_size, num_frames, pose_dim = masked_sequence.shape
        conditioning = BatchSequenceConditioning.from_batch_attributes(
            sequence=masked_sequence, mask=mask, sequence_lengths=sequence_lengths
        )
        model_kwargs = conditioning.to_model_kwargs()
        model_kwargs = update_conditioning_with_inference_params(model_kwargs, self.inference_config)
        sample_fn = self.diffusion.p_sample_loop
        inference_model = self.ema_model if self.inference_config.use_ema_model else self.model
        sample = sample_fn(
            model=inference_model,
            shape=(batch_size, pose_dim, num_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )
        return (sample.transpose(1, 2), None, None)
