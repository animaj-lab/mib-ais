import torch

from motion_inbetweening.config.data import RelativeControllerTransformationConfig
from shared.rig.trainable_controllers import TrainableController, transformation_to_attributes_names


def get_indices_for_relative_controller_transformations(
    trainable_controllers: list[TrainableController],
    relative_controllers: dict[str, list[RelativeControllerTransformationConfig]],
) -> list[int]:
    indices = []
    relative_controller_names = []
    for controller_name, transformations in relative_controllers.items():
        for transformation in transformations:
            relative_controller_names.append(f"{controller_name}:{transformation.attribute}")
    controller_index = 0
    for controller in trainable_controllers:
        if controller.name in relative_controllers.keys():
            transformation_index = controller_index
            for transformation in controller.transformations:
                transformation_names = transformation_to_attributes_names(transformation)
                full_transformation_names = [f"{controller.name}:{name}" for name in transformation_names]
                local_index = transformation_index
                for transformation_name in full_transformation_names:
                    if transformation_name in relative_controller_names:
                        indices.append(local_index)
                    local_index += 1
                transformation_index += transformation.to_vector_size()
        controller_index += sum(transformation.to_vector_size() for transformation in controller.transformations)
    return indices


def get_relative_sequence_vectors(
    sequence_vectors: torch.Tensor, indices_for_relative_controller_transformations: list[int]
) -> torch.Tensor:
    for i in indices_for_relative_controller_transformations:
        sequence_vectors[:, i] = sequence_vectors[:, i] - sequence_vectors[0, i]
    return sequence_vectors
