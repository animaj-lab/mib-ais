import torch


def compute_keyframe_prediction_loss(
    predicted_keyframes: torch.Tensor,
    controller_keyframes: torch.Tensor,
    sequence_lengths: torch.Tensor,
    loss_function: torch.nn.Module,
) -> torch.Tensor:
    """
    Computes the loss for keyframe prediction by comparing the predicted keyframes with the actual controller keyframes.

    Args:
        predicted_keyframes (torch.Tensor): The predicted keyframes tensor of shape
            (batch_size, max_sequence_length, dim_controller_keyframe).
        controller_keyframes (torch.Tensor): The actual controller keyframes tensor of shape
            (batch_size, max_sequence_length, dim_controller_keyframe).
        sequence_lengths (torch.Tensor): A tensor containing the lengths of each sequence in the batch of shape
            (batch_size,).
        keyframe_predictor (ControllerKeyframesPredictor): An instance of the keyframe predictor which contains the
            loss function.

    Returns:
        torch.Tensor: The computed loss value for the keyframe prediction.
    """
    batch_size, max_sequence_length, _ = controller_keyframes.size()
    padding_mask_keyframe_predictor = torch.arange(max_sequence_length, device=controller_keyframes.device).expand(
        batch_size, max_sequence_length
    ).unsqueeze(-1) >= sequence_lengths.unsqueeze(1).unsqueeze(-1)
    loss_value_controller_keyframes_prediction = loss_function(predicted_keyframes, controller_keyframes)
    loss_value_controller_keyframes_prediction = loss_value_controller_keyframes_prediction.masked_fill(
        padding_mask_keyframe_predictor, 0
    )
    return loss_value_controller_keyframes_prediction.mean()
