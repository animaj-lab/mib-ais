from collections.abc import Callable

import torch

from motion_inbetweening.controller_keyframes.projection import (
    ControllerKeyframesProjection,
    project_controller_keyframes_vector_to_pose_vector_dim,
)


def compute_controller_keyframes_weights(
    controller_keyframes_vector: torch.Tensor,
    controller_keyframes_projection: ControllerKeyframesProjection,
    controller_keyframes_weight_function: Callable[[torch.Tensor], torch.Tensor],
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the controller keyframes weights based on the controller keyframes values
    These weights are going to be applied to the pose loss
    Therefore we need to change the dimension of the controller keyframes to match the pose dimension
    Args:
    controller_keyframes (torch.Tensor): The controller keyframes tensor of shape
        (batch_size, seq_len, dim_controller_keyframes)
    controller_keyframes_projection (ControllerKeyframesProjection): The transformation to apply
        to the controller keyframes
    controller_keyframes_weight_function (Callable[[torch.Tensor], torch.Tensor]): The function to apply
        to the controller keyframes values
    mask (torch.Tensor): The mask tensor of shape (batch_size, seq_len) where 1 indicates padded frames
        and 0 indicates non-padded frames
    Returns:
    torch.Tensor: The controller keyframes weights tensor of shape (batch_size, seq_len, dim_pose_vector)
    """
    controller_keyframes_vector_reshaped = project_controller_keyframes_vector_to_pose_vector_dim(
        controller_keyframes_vector, controller_keyframes_projection
    )
    controller_keyframes_weights = controller_keyframes_weight_function(controller_keyframes_vector_reshaped)
    controller_keyframes_weights = controller_keyframes_weights * ~padding_mask.unsqueeze(-1)
    return controller_keyframes_weights
