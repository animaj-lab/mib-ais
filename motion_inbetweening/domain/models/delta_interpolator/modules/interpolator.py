import torch
from torch import nn

from shared.rig.utils import matrix_to_rotation_6d, quaternion_to_matrix


def slerp(q0, q1, t):
    """
    Spherical Linear Interpolation of quaternions
    https://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/slerp/index.htm
    :param q0: Start quats (w, x, y, z) : shape = (B, J, 4)
    :param q1: End quats (w, x, y, z) : shape = (B, J, 4)
    :param t:  Step (in [0, 1]) : shape = (B, T, J, 1)
    :return: Interpolated quat (w, x, y, z) : shape = (B, J, 4)
    """
    q0 = q0.unsqueeze(1)
    q1 = q1.unsqueeze(1)

    # Dot product
    cos_half_theta = torch.sum(q0 * q1, dim=-1, keepdim=True)

    # Make sure we take the shortest path :
    q1_antipodal = -q1
    q1 = torch.where(cos_half_theta > 0, q1, q1_antipodal)

    half_theta = torch.acos(cos_half_theta)
    # torch.sin must be safer here
    sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)
    ratio_a = torch.sin((1 - t) * half_theta) / sin_half_theta
    ratio_b = torch.sin(t * half_theta) / sin_half_theta

    qt = ratio_a * q0 + ratio_b * q1
    # If the angle was constant, prevent nans by picking the original quat:
    qt = torch.where(torch.abs(cos_half_theta) >= 1.0, q0, qt)
    return qt


class Interpolator(nn.Module):
    def __init__(self, nb_joints: int = None, space: str = "ortho6d"):
        super().__init__()
        self.nb_joints = nb_joints
        self.space = space

    def normalize(self, x, axis=-1, eps=1e-08):
        """
        Normalizes a tensor over some axis (axes)
        :param x: data tensor
        :param axis: axis(axes) along which to compute the norm
        :param eps: epsilon to prevent numerical instabilities
        :return: The normalized tensor
        """
        length = torch.sqrt(torch.sum(x * x, axis=axis, keepdims=True))
        res = x / (length + eps)
        return res

    def interpolate_local(
        self,
        lcl_r_mb: torch.Tensor,
        lcl_q_mb: torch.Tensor,
        n_past: int = 1,
        n_future: int = 1,
        target_timestamps: torch.Tensor = None,
    ):
        """
        Performs interpolation between 2 frames of an animation sequence.
        The 2 frames are indirectly specified through n_past and n_future.
        SLERP is performed on the quaternions
        LERP is performed on the root's positions.
        :param lcl_r_mb:  Local/Global root positions (B, T, 1, 3)
        :param lcl_q_mb:  Local quaternions (B, T, J, 4)
        :param n_past:    Number of frames of past context
        :param n_future:  Number of frames of future context
        :return: Interpolated root and quats
        """
        start_lcl_r_mb = lcl_r_mb[:, n_past - 1 : n_past, ...]
        end_lcl_r_mb = lcl_r_mb[:, -n_future : -n_future + 1 if n_future > 1 else None, ...]
        start_lcl_q_mb = lcl_q_mb[:, n_past - 1, :, :]
        end_lcl_q_mb = lcl_q_mb[:, -n_future, :, :]
        offset = end_lcl_r_mb - start_lcl_r_mb
        weights = target_timestamps - target_timestamps[0:1] + 1
        weights = weights / (weights[-1] + 1)
        weights = weights.type_as(lcl_r_mb)
        const_trans = start_lcl_r_mb.repeat([1, weights.shape[0], 1, 1])
        inter_lcl_r_mb = const_trans + weights.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * offset
        start = self.normalize(start_lcl_q_mb)
        end = self.normalize(end_lcl_q_mb)
        inter_lcl_q_mb = slerp(start, end, weights.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
        return (inter_lcl_r_mb, inter_lcl_q_mb)

    def interpolate_global(
        self, lcl_r_mb: torch.Tensor, n_past: int = 1, n_future: int = 1, target_timestamps: torch.Tensor = None
    ):
        """
        Performs interpolation between 2 frames of an animation sequence.
        The 2 frames are indirectly specified through n_past and n_future.
        SLERP is performed on the quaternions
        LERP is performed on the root's positions.
        :param lcl_r_mb:  Global positions (B, T, J, 3)
        :param n_past:    Number of frames of past context
        :param n_future:  Number of frames of future context
        :return: Interpolated positions
        """
        start_lcl_r_mb = lcl_r_mb[:, n_past - 1 : n_past, ...]
        end_lcl_r_mb = lcl_r_mb[:, -n_future : -n_future + 1 if n_future > 1 else None, ...]
        offset = end_lcl_r_mb - start_lcl_r_mb
        weights = target_timestamps - target_timestamps[0:1] + 1
        weights = weights / (weights[-1] + 1)
        weights = weights.type_as(lcl_r_mb)
        const_trans = start_lcl_r_mb.repeat([1, weights.shape[0], 1, 1])
        inter_lcl_r_mb = const_trans + weights.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * offset
        return inter_lcl_r_mb

    def forward(self, input_data):
        past_i = input_data["past_frame_indices"][-1]
        n_trans = input_data["target_frame_indices"].shape[0]
        future_i = input_data["future_frame_indices"][0] - n_trans
        if "key_frame_indices" in input_data.keys():
            past_i = input_data["past_frame_indices"][-1]
            n_trans = input_data["target_frame_indices"].shape[0]
            future_i = input_data["future_frame_indices"][0] - n_trans
            positions = torch.stack(
                [input_data["joint_positions"][:, 0, ...], input_data["joint_positions"][:, 1, ...]], dim=1
            )
            positions = self.interpolate_global(positions, target_timestamps=input_data["target_frame_indices"])
            return positions
        elif "joint_rotations" not in input_data.keys() and "joint_rotations_global" not in input_data.keys():
            past_i = input_data["past_frame_indices"][-1]
            n_trans = input_data["target_frame_indices"].shape[0]
            future_i = input_data["future_frame_indices"][0] - n_trans
            positions = torch.stack(
                [input_data["joint_positions"][:, past_i, ...], input_data["joint_positions"][:, future_i, ...]], dim=1
            )
            positions = self.interpolate_global(positions, target_timestamps=input_data["target_frame_indices"])
            return positions
        if self.space == "ortho6d":
            rotations = input_data["joint_rotations"]
            positions = torch.stack(
                [input_data["root_positions"][:, past_i, :], input_data["root_positions"][:, future_i, :]], dim=1
            )[:, :, None, :]
        else:
            rotations = input_data["joint_rotations_global"]
            positions = torch.stack(
                [input_data["joint_positions"][:, past_i, ...], input_data["joint_positions"][:, future_i, ...]], dim=1
            )
        rotations = torch.stack([rotations[:, past_i, ...], rotations[:, future_i, ...]], dim=1)
        root_position, joint_rotations = self.interpolate_local(
            positions, rotations, target_timestamps=input_data["target_frame_indices"]
        )
        if self.space == "ortho6d":
            batch_size, n_frames = joint_rotations.shape[:2]
            joint_rotations = quaternion_to_matrix(joint_rotations)
            joint_rotations = matrix_to_rotation_6d(joint_rotations.view(-1, 3, 3))
            joint_rotations = joint_rotations.view(batch_size, n_frames, self.nb_joints, 6).contiguous()
            joint_positions = root_position.repeat([1, 1, self.nb_joints, 1]).contiguous()
        else:
            joint_positions = root_position
        return (joint_positions, joint_rotations)


class InbetweenInterpolator(Interpolator):
    def forward(self, input_data: dict[str, torch.Tensor] = None, *args) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x the continuous input : BxTxNxC
        """
        src_time = input_data["input_frame_indices"].to(dtype=torch.int64)
        if "key_frame_indices" in input_data.keys():
            known_frames_full = (
                input_data["past_frame_indices"].tolist()
                + input_data["key_frame_indices"].tolist()
                + input_data["future_frame_indices"].tolist()
            )
            known_frames = sorted(set(known_frames_full))
            frame_builder = []
            for i in range(len(known_frames) - 1):
                input_data_chunk = {}
                dev = input_data["joint_positions"].device
                input_data_chunk["joint_positions"] = torch.stack(
                    [input_data["joint_positions"][:, i + 1], input_data["joint_positions"][:, i + 2]], axis=1
                )
                input_data_chunk["input_frame_indices"] = torch.tensor(
                    [input_data["input_frame_indices"][i + 1], input_data["input_frame_indices"][i + 2]], device=dev
                )
                input_data_chunk["past_frame_indices"] = torch.tensor(
                    [input_data_chunk["input_frame_indices"][0]], device=dev
                )
                input_data_chunk["key_frame_indices"] = None
                if known_frames[i] + 1 > known_frames[i + 1]:
                    continue
                if input_data_chunk["input_frame_indices"][0] + 1 == input_data_chunk["input_frame_indices"][1]:
                    frame_builder.append(input_data_chunk["joint_positions"][:, -2:])
                    continue
                input_data_chunk["target_frame_indices"] = torch.arange(
                    known_frames[i] + 1, known_frames[i + 1], device=dev
                )
                input_data_chunk["future_frame_indices"] = input_data["future_frame_indices"] * 0 + known_frames[i + 1]
                joint_positions_interpolator = super().forward(input_data_chunk)
                frame_builder.append(input_data_chunk["joint_positions"][:, 0].unsqueeze(1))
                frame_builder.append(joint_positions_interpolator)
            frame_builder.append(frame_builder[-1][:, -1:])
            joint_positions_interpolator = torch.cat(frame_builder, dim=1)
            return joint_positions_interpolator
        elif "joint_rotations" not in input_data.keys() and "joint_rotations_global" not in input_data.keys():
            joint_positions_interpolator = super().forward(input_data)
            joint_positions_interpolator = torch.cat(
                [
                    joint_positions_interpolator[:, :1].repeat([1, src_time.shape[0] - 1, 1, 1]),
                    joint_positions_interpolator,
                    joint_positions_interpolator[:, -1:],
                ],
                dim=1,
            )
            return joint_positions_interpolator
        joint_positions_interpolator, joint_rotations_interpolator = super().forward(input_data)
        joint_rotations_interpolator = torch.cat(
            [
                joint_rotations_interpolator[:, :1].repeat([1, src_time.shape[0] - 1, 1, 1]),
                joint_rotations_interpolator,
                joint_rotations_interpolator[:, -1:],
            ],
            dim=1,
        )
        joint_positions_interpolator = torch.cat(
            [
                joint_positions_interpolator[:, :1].repeat([1, src_time.shape[0] - 1, 1, 1]),
                joint_positions_interpolator,
                joint_positions_interpolator[:, -1:],
            ],
            dim=1,
        )
        return (joint_positions_interpolator, joint_rotations_interpolator)
