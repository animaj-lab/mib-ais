import torch


def compute_batch_shifted_motion_distance(
    ground_truth_sequences: torch.Tensor,
    predicted_sequences: torch.Tensor,
    animation_keyframes: torch.BoolTensor,
    unmasked_frames: list[list[int]],
) -> float:
    """
    Compute the shifted motion distance between the ground truth and predicted sequences.
    Args:
        ground_truth_sequence: ground truth sequence of shape (batch_size, seq_len, pose_dim)
        predicted_sequence: predicted sequence of shape (batch_size, seq_len, pose_dim)
        animation_keyframes: animation keyframes tensor of shape (batch_size, seq_len) where True indicates a keyframe
            and False indicates a non-keyframe
        unmasked_frames: list of list of integers containing the unmasked frames for each sequence in the batch"""
    batch_size, seq_len, pose_dim = ground_truth_sequences.size()
    shifted_sequence_distances = []
    for i in range(batch_size):
        distance_sequence_i = compute_shifted_motion_distance(
            ground_truth_sequence=ground_truth_sequences[i],
            predicted_sequence=predicted_sequences[i],
            animation_keyframes=animation_keyframes[i],
            unmasked_frames=unmasked_frames[i],
        )
        shifted_sequence_distances.append(distance_sequence_i)
    mean_shifted_motion_distance = sum(shifted_sequence_distances) / batch_size
    return mean_shifted_motion_distance


def compute_shifted_motion_distance(
    ground_truth_sequence, predicted_sequence, animation_keyframes, unmasked_frames
) -> float:
    """
    Compute the shifted motion distance for a given sequence.
    Args:
        ground_truth_sequence: ground truth sequence of shape (seq_len, pose_dim)
        predicted_sequence: predicted sequence of shape (seq_len, pose_dim)
        animation_keyframes: animation keyframes tensor of shape (seq_len) where True indicates a keyframe and
        False indicates a non-keyframe
        unmasked_frames: list of integers containing the unmasked frames
    Returns:
        shifted_motion_distance: shifted motion distance for the given sequence"""
    shifted_motion_distance_i = 0
    num_frames_i = 0
    for j in range(len(unmasked_frames) - 1):
        start_frame = unmasked_frames[j]
        end_frame = unmasked_frames[j + 1]
        ground_truth_motion = ground_truth_sequence[start_frame : end_frame + 1]
        candidates_predicted_motion = get_candidates_motion(
            start_frame, end_frame, predicted_sequence, min_frame=unmasked_frames[0], max_frame=unmasked_frames[-1] + 1
        )
        closest_predicted_motion = find_closest_motion(ground_truth_motion, candidates_predicted_motion)
        frames_to_consider = compute_frames_to_consider(start_frame, end_frame, animation_keyframes)
        motion_distance = compute_motion_distance(ground_truth_motion, closest_predicted_motion, frames_to_consider)
        shifted_motion_distance_i += motion_distance
        num_frames_i += len(frames_to_consider)
    return shifted_motion_distance_i / num_frames_i


def get_candidates_motion(
    start_frame: int, end_frame: int, predicted_sequence: torch.Tensor, min_frame: int, max_frame: int
) -> list[torch.Tensor]:
    """
    Get the candidates predicted motion for a given ground truth motion.
    Args:
        start_frame: start frame index of the ground truth motion
        end_frame: end frame index of the ground truth motion
        predicted_sequence: predicted sequence of shape (seq_len, pose)
        min_frame: minimum frame index (left limit of the sequence)
        max_frame: maximum frame index (right limit of the sequence)
    Returns:
        candidates_predicted_motion: list of candidate predicted motions"""
    candidates_predicted_motion = []
    motion_length = end_frame - start_frame
    start_frame_search_min = max(start_frame - motion_length, min_frame)
    start_frame_search_max = min(end_frame + 1, max_frame - motion_length)
    for start_predicted_frame in range(start_frame_search_min, start_frame_search_max):
        predicted_motion = predicted_sequence[start_predicted_frame : start_predicted_frame + motion_length + 1]
        candidates_predicted_motion.append(predicted_motion)
    return candidates_predicted_motion


def find_closest_motion(
    ground_truth_motion: torch.Tensor, candidates_predicted_motion: list[torch.Tensor]
) -> torch.Tensor:
    """
    Find the closest predicted motion to the ground truth motion using parallelized operations.
    Args:
        ground_truth_motion: ground truth motion of shape (motion_length, pose_dim)
        candidates_predicted_motion: list of candidate predicted motions
    Returns:
        closest_predicted_motion: predicted motion closest to the ground truth motion"""
    candidates_predicted_motion_tensor = torch.stack(candidates_predicted_motion)
    distances = torch.norm(candidates_predicted_motion_tensor - ground_truth_motion.unsqueeze(0), p=1, dim=(1, 2))
    closest_index = torch.argmin(distances)
    closest_predicted_motion = candidates_predicted_motion_tensor[closest_index]
    return closest_predicted_motion


def compute_frames_to_consider(start_frame: int, end_frame: int, animation_keyframes: torch.BoolTensor) -> list[int]:
    """
    Compute the frames to consider for the motion distance computation.
    In the current heuristic, we consider the animation keyframes of the ground truth motion
    and their neighbors to take the tangent of the controller curves into account.
    Args:
        start_frame: start frame index of the ground truth motion
        end_frame: end frame index of the ground truth motion
        animation_keyframes: animation keyframes tensor of shape (seq_len) where True indicates a keyframe and False
            indicates a non-keyframe
    Returns:
        frames_to_consider: list of frame indices to consider for the motion distance computation"""
    frames_to_consider = list(range(start_frame, end_frame + 1))
    frames_to_consider = [frame - start_frame for frame in frames_to_consider]
    return frames_to_consider


def compute_motion_distance(
    ground_truth_motion: torch.Tensor, closest_predicted_motion: torch.Tensor, frames_to_consider: list[int]
) -> float:
    """
    Compute the motion distance between the ground truth and predicted motions.
    Args:
        ground_truth_motion: ground truth motion of shape (motion_length, pose_dim)
        closest_predicted_motion: predicted motion closest to the ground truth motion
        frames_to_consider: list of frame indices to consider for the motion distance computation
    Returns:
        motion_distance: motion distance between the ground truth and predicted motions"""
    motion_distance = torch.norm(
        ground_truth_motion[frames_to_consider] - closest_predicted_motion[frames_to_consider], p=1
    ).item()
    return motion_distance
