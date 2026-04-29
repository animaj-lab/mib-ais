import torch

from motion_inbetweening.config.losses import LossesFrameWeights


def compute_frame_weights(
    losses_frame_weights: LossesFrameWeights,
    mask: torch.Tensor,
    animation_keyframes: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute frame weights based on the losses frame weights configuration.
    Args:
        losses_frame_weights: LossesFrameWeights configuration.
        mask: Mask tensor of shape (n_batch, n_frames) where True indicates a masked frame and False indicates an
            unmasked frame.
        animation_keyframes: Keyframes tensor of shape (n_batch, n_frames) where True indicates an animation keyframe
            and False indicates a non-keyframe.
        padding_mask: Padding mask tensor of shape (n_batch, n_frames) where True indicates a padding frame and False
            indicates a non-padding frame.
    Returns:
        frame_weights: Frame weights tensor of shape (n_batch, n_frames) where the weights are computed based on the
            losses frame weights configuration.
    """
    frame_weights = (
        losses_frame_weights.unmasked_frames * ~mask
        + losses_frame_weights.masked_animation_keyframes * animation_keyframes * mask
        + losses_frame_weights.masked_non_animation_keyframes * mask * ~animation_keyframes
    )
    frame_weights *= ~padding_mask
    return frame_weights
