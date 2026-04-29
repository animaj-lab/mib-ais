from typing import Self

import torch

from motion_inbetweening.config.inference_diffusion import InferenceDiffusionConfig


def lengths_to_mask(lengths, max_len):
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


class BatchSequenceConditioning:
    """
    BatchSequenceConditioning is a dataclass that represents the inbetwweening conditioning for a batch of sequences.
    Args:
        obs_x0: torch.Tensor of shape [batch_size, pose_dim, sequence_length] is the full observed sequence.
        obs_mask: torch.Tensor of shape [batch_size, pose_dim, sequence_length] is the mask of the observed sequence.
            1 if the pose is unmasked, 0 otherwise.
        sequence_length_mask: torch.Tensor of shape [batch_size, sequence_length] is the mask of the sequence length.
            1 if the frame is in the sequence, 0 if the frame corresponds to padding.

    """

    def __init__(self, observation_input_motion, observation_mask, sequence_length_mask):
        self.observation_input_motion = observation_input_motion
        self.observation_mask = observation_mask
        self.sequence_length_mask = sequence_length_mask

    @classmethod
    def from_batch_attributes(cls, sequence, mask, sequence_lengths) -> Self:
        observation_input_motion = sequence.transpose(1, 2)
        mask_reshaped = mask.unsqueeze(1).repeat(1, observation_input_motion.shape[1], 1)
        observation_mask = ~mask_reshaped
        sequence_length_mask = lengths_to_mask(sequence_lengths, observation_input_motion.shape[-1]).unsqueeze(1)
        return cls(observation_input_motion, observation_mask, sequence_length_mask)

    def to_model_kwargs(self):
        model_kwargs = {
            "obs_x0": self.observation_input_motion,
            "obs_mask": self.observation_mask,
            "y": {"mask": self.sequence_length_mask},
        }
        return model_kwargs


def update_conditioning_with_inference_params(model_kwargs: dict, inference_diffusion_config: InferenceDiffusionConfig):
    inference_dict = {
        "inpainted_motion": model_kwargs["obs_x0"],
        "inpainting_mask": model_kwargs["obs_mask"],
        "imputate": inference_diffusion_config.do_imputation,
        "replacement_distribution": str(inference_diffusion_config.imputation_params.replacement_distribution)
        if inference_diffusion_config.do_imputation
        else None,
        "stop_imputation_at": inference_diffusion_config.imputation_params.stop_imputation_at
        if inference_diffusion_config.do_imputation
        else None,
        "reconstruction_guidance": inference_diffusion_config.do_guidance,
        "stop_recguidance_at": inference_diffusion_config.guidance_params.stop_recguidance_at
        if inference_diffusion_config.do_guidance
        else None,
        "gradient_schedule": inference_diffusion_config.guidance_params.gradient_schedule
        if inference_diffusion_config.do_guidance
        else None,
        "diffusion_steps": inference_diffusion_config.guidance_params.diffusion_steps
        if inference_diffusion_config.do_guidance
        else None,
        "reconstruction_weight": inference_diffusion_config.guidance_params.reconstruction_weight
        if inference_diffusion_config.do_guidance
        else None,
    }
    model_kwargs["y"].update(inference_dict)
    return model_kwargs
