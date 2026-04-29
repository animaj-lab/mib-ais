from dataclasses import dataclass

import torch

from motion_inbetweening.config.controller_keyframes_preprocessing import (
    AttributeKeyframeDivisionStategy,
    TransformationsKeyframesDivisionStrategy,
    get_transformation_name,
)
from shared.rig.trainable_controllers import (
    TrainableController,
    Transformation,
    to_full_vector_size,
    to_vector_size,
    transformation_to_attributes_names,
)


@dataclass
class ControllerKeyframesProjection:
    """
    A class to transform controller keyframes into pose vectors.

    Attributes:
    ----------
    dim_controller_keyframes : int
        The dimensionality of the controller keyframes.
    dim_pose_vector : int
        The dimensionality of the pose vector.
    matrix_controller_keyframes_to_pose_vector : torch.Tensor
        A transformation matrix with shape (dim_pose_vector, dim_controller_keyframes)
        that maps controller keyframes dim to pose vectors dim.
    transformations_keyframes_division_strategy : TransformationsKeyframesDivisionStrategy
        Strategy for dividing keyframes into transformations.
    """

    dim_controller_keyframes: int
    dim_pose_vector: int
    matrix_controller_keyframes_to_pose_vector: torch.Tensor
    transformations_keyframes_division_strategy: TransformationsKeyframesDivisionStrategy


def initialize_controller_keyframes_projection(
    trainable_controllers: list[TrainableController],
    transformations_keyframe_division_strategy: TransformationsKeyframesDivisionStrategy,
) -> ControllerKeyframesProjection:
    dim_controller_keyframes = get_controller_keyframes_dim(
        trainable_controllers, transformations_keyframe_division_strategy
    )
    dim_pose_vector = to_full_vector_size(trainable_controllers)
    matrix_controller_keyframes_to_pose_vector = get_matrix_controller_keyframes_to_pose_vector(
        dim_controller_keyframes, dim_pose_vector, trainable_controllers, transformations_keyframe_division_strategy
    )
    return ControllerKeyframesProjection(
        dim_controller_keyframes=dim_controller_keyframes,
        dim_pose_vector=dim_pose_vector,
        matrix_controller_keyframes_to_pose_vector=matrix_controller_keyframes_to_pose_vector,
        transformations_keyframes_division_strategy=transformations_keyframe_division_strategy,
    )


def to_controller_keyframes_vector_size(
    transformation: Transformation, transformations_keyframe_division_strategy: TransformationsKeyframesDivisionStrategy
) -> int:
    if get_transformation_name(transformation) not in transformations_keyframe_division_strategy:
        raise ValueError(f"Missing keyframe division strategy for {transformation}")
    attribute_keys = transformation_to_attributes_names(transformation)
    keyframe_division_strategy = transformations_keyframe_division_strategy[get_transformation_name(transformation)]
    if keyframe_division_strategy == AttributeKeyframeDivisionStategy.ONE_FOR_EACH_ATTRIBUTE:
        return len(attribute_keys)
    elif keyframe_division_strategy == AttributeKeyframeDivisionStategy.ONE_FOR_ALL_ATTRIBUTES:
        return 1
    else:
        raise ValueError(f"Unknown keyframe division strategy: {keyframe_division_strategy}")


def get_controller_keyframes_dim(
    trainable_controllers: list[TrainableController],
    transformations_keyframe_division_strategy: TransformationsKeyframesDivisionStrategy,
) -> int:
    dim_controller_keyframes = 0
    for trainable_controller in trainable_controllers:
        for transformation in trainable_controller.transformations:
            dim_controller_keyframes += to_controller_keyframes_vector_size(
                transformation, transformations_keyframe_division_strategy
            )
    return dim_controller_keyframes


def get_matrix_controller_keyframes_to_pose_vector(
    dim_controller_keyframes: int,
    dim_pose_vector: int,
    trainable_controllers: list[TrainableController],
    transformations_keyframe_division_strategy: TransformationsKeyframesDivisionStrategy,
) -> torch.Tensor:
    index_pose_vector = 0
    index_controller_keyframe = 0
    matrix_controller_keyframes_to_pose_vector = torch.zeros(dim_pose_vector, dim_controller_keyframes)
    for trainable_controller in trainable_controllers:
        for transformation in trainable_controller.transformations:
            transformation_dim_in_controller_keyframes = to_controller_keyframes_vector_size(
                transformation, transformations_keyframe_division_strategy
            )
            transformation_dim_in_pose_vector = to_vector_size(transformation)
            keyframe_division_strategy = transformations_keyframe_division_strategy[
                get_transformation_name(transformation)
            ]
            if keyframe_division_strategy == AttributeKeyframeDivisionStategy.ONE_FOR_EACH_ATTRIBUTE:
                if transformation_dim_in_controller_keyframes != transformation_dim_in_pose_vector:
                    raise ValueError(
                        f"Dimension mismatch for transformation {transformation} and pose vector. "
                        f"dim_controller_keyframes: {transformation_dim_in_controller_keyframes}, "
                        f"dim_pose_vector: {transformation_dim_in_pose_vector}. "
                        f"If the number of attributes of this transformation is not equal to its vector size, "
                        f"you should use the keyframe division strategy ONE_FOR_ALL_ATTRIBUTES."
                    )
                for i in range(transformation_dim_in_pose_vector):
                    matrix_controller_keyframes_to_pose_vector[index_pose_vector + i, index_controller_keyframe + i] = 1
            elif keyframe_division_strategy == AttributeKeyframeDivisionStategy.ONE_FOR_ALL_ATTRIBUTES:
                for i in range(transformation_dim_in_pose_vector):
                    matrix_controller_keyframes_to_pose_vector[index_pose_vector + i, index_controller_keyframe] = 1
            index_pose_vector += transformation_dim_in_pose_vector
            index_controller_keyframe += transformation_dim_in_controller_keyframes
    return matrix_controller_keyframes_to_pose_vector


def project_controller_keyframes_vector_to_pose_vector_dim(
    controller_keyframes_vector: torch.Tensor, controller_keyframes_projection: ControllerKeyframesProjection
) -> torch.Tensor:
    """
    project_controller_keyframes_vector_to_pose_vector_dim projects the controller keyframe vector to the pose vector
        dimension.
    Args:
        controller_keyframes_vector (torch.Tensor): The controller keyframe vector tensor of shape
            (batch_size, seq_len, dim_controller_keyframes)
        controller_keyframes_projection (ControllerKeyframesProjection): The transformation to apply to the
        controller keyframes, it contains a matrix of shape
        (dim_pose_vector, dim_controller_keyframes) to project the controller keyframes to the pose vector dimension.
    Returns:
        torch.Tensor: The controller keyframe vector projected to the pose vector dimension of shape
            (batch_size, seq_len, dim_pose_vector)
    """
    matrix = controller_keyframes_projection.matrix_controller_keyframes_to_pose_vector.to(
        controller_keyframes_vector.device
    )
    return torch.matmul(controller_keyframes_vector, matrix.T)
