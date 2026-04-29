import torch
import torch._dynamo

from shared.rig.trainable_controllers import TrainableController
from shared.rig.utils import matrix_to_quaternion, matrix_to_rotation_6d, quaternion_to_matrix, rotation_6d_to_matrix


@torch.compile()
def lerp_slerp_interpolation(
    sequence: torch.Tensor, mask: torch.Tensor, trainable_controllers: list[TrainableController]
) -> torch.Tensor:
    """
    Interpolates masked sequences using linear or SLERP depending on transformation type.
    Fully optimized with slicing and torch.compile.
    """
    B, T, D = sequence.shape
    output = torch.zeros_like(sequence)
    slices = []
    start_index = 0
    for controller in trainable_controllers:
        for transformation in controller.transformations:
            size = transformation.to_vector_size()
            name = transformation.to_transformation_name()
            slices.append((start_index, start_index + size, name))
            start_index += size
    left_idx, right_idx = compute_left_right_indices(mask)
    for start, end, name in slices:
        sub_sequence = sequence[:, :, start:end]
        if name in ["Translation", "Scale"]:
            interpolated = linear_interpolation(sub_sequence, mask, left_idx, right_idx)
        elif name == "Rotation3D":
            interpolated = slerp_interpolation_6D(sub_sequence, mask, left_idx, right_idx)
        elif name == "Rotation1D":
            interpolated = slerp_interpolation_2D(sub_sequence, mask, left_idx, right_idx)
        else:
            raise ValueError(f"Unknown transformation name: {name}")
        output[:, :, start:end] = interpolated
    return output


def compute_left_right_indices(mask: torch.Tensor):
    B, T = mask.shape
    device = mask.device
    time = torch.arange(T, device=device).view(1, T).expand(B, T)
    known_time = torch.where(~mask, time, torch.full_like(time, -1))
    left_idx = known_time.clone()
    for t in range(1, T):
        left_idx[:, t] = torch.where(left_idx[:, t] == -1, left_idx[:, t - 1], left_idx[:, t])
    right_idx = known_time.clone()
    for t in reversed(range(T - 1)):
        right_idx[:, t] = torch.where(right_idx[:, t] == -1, right_idx[:, t + 1], right_idx[:, t])
    return (left_idx.clamp(min=0), right_idx.clamp(max=T - 1))


def linear_interpolation(sequence, mask, left_idx, right_idx):
    B, T, D = sequence.shape
    device = sequence.device
    time = torch.arange(T, device=device).view(1, T).expand(B, T)
    batch_idx = torch.arange(B, device=device).view(B, 1).expand(B, T)
    left_val = sequence[batch_idx, left_idx]
    right_val = sequence[batch_idx, right_idx]
    denom = (right_idx - left_idx).clamp(min=1).unsqueeze(-1)
    numer = (time - left_idx).unsqueeze(-1)
    alpha = numer / denom
    interpolated = (1 - alpha) * left_val + alpha * right_val
    output = sequence.clone()
    mask_expanded = mask.unsqueeze(-1).expand_as(sequence)
    output = sequence.clone()
    output[mask_expanded] = interpolated[mask_expanded]
    return output


def slerp_interpolation_6D(sequence, mask, left_idx, right_idx):
    B, T, D = sequence.shape
    assert D == 6
    device = sequence.device
    output = sequence.clone()
    rotmat = rotation_6d_to_matrix(sequence)
    quat = matrix_to_quaternion(rotmat)
    time = torch.arange(T, device=device).view(1, T).expand(B, T)
    batch_idx = torch.arange(B, device=device).view(B, 1).expand(B, T)
    q0 = quat[batch_idx, left_idx]
    q1 = quat[batch_idx, right_idx]
    dot = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = (q0 * q1).sum(-1, keepdim=True).clamp(-1.0, 1.0)
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    alpha = (time - left_idx).clamp(min=0).unsqueeze(-1).float() / (right_idx - left_idx).clamp(min=1).unsqueeze(
        -1
    ).float()
    near_zero = sin_theta_0 < 1e-06
    s0 = torch.where(near_zero, 1.0 - alpha, torch.sin((1 - alpha) * theta_0) / sin_theta_0)
    s1 = torch.where(near_zero, alpha, torch.sin(alpha * theta_0) / sin_theta_0)
    q_interp = s0 * q0 + s1 * q1
    r_interp = quaternion_to_matrix(q_interp)
    interp_6d = matrix_to_rotation_6d(r_interp)
    mask_expanded = mask.unsqueeze(-1).expand_as(sequence)
    output = sequence.clone()
    output[mask_expanded] = interp_6d[mask_expanded]
    return output
    return output


def slerp_interpolation_2D(sequence, mask, left_idx, right_idx):
    B, T, D = sequence.shape
    assert D == 2
    device = sequence.device
    output = sequence.clone()
    time = torch.arange(T, device=device).view(1, T).expand(B, T)
    batch_idx = torch.arange(B, device=device).view(B, 1).expand(B, T)
    v0 = sequence[batch_idx, left_idx]
    v1 = sequence[batch_idx, right_idx]
    dot = (v0 * v1).sum(-1).clamp(-1.0, 1.0)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    alpha = (time - left_idx).clamp(min=0).float() / (right_idx - left_idx).clamp(min=1).float()
    alpha = alpha.unsqueeze(-1)
    omega = omega.unsqueeze(-1)
    sin_omega = sin_omega.unsqueeze(-1)
    s0 = torch.sin((1 - alpha) * omega) / (sin_omega + 1e-06)
    s1 = torch.sin(alpha * omega) / (sin_omega + 1e-06)
    v_interp = s0 * v0 + s1 * v1
    mask_expanded = mask.unsqueeze(-1).expand_as(sequence)
    output = sequence.clone()
    output[mask_expanded] = v_interp[mask_expanded]
    return output
    return output
