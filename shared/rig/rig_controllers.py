from copy import deepcopy

import torch

from shared.rig import trainable_controllers
from shared.rig.trainable_controllers import TrainableController


def vec_to_rig(controllers: list[TrainableController], vec: torch.Tensor, default_rig: dict):
    rig = deepcopy(default_rig)
    vec = vec.view(-1)
    for controller in controllers:
        for transformation in controller.transformations:
            size = transformation.to_vector_size()
            controller.name = controller.name.replace("py_ch0001_1:", "")
            if controller.name in rig:
                rig[controller.name] |= trainable_controllers.controller_vec_to_rig(transformation, vec[:size])
            else:
                rig[controller.name] = trainable_controllers.controller_vec_to_rig(transformation, vec[:size])
            vec = vec[size:]
    return rig


def rig_to_vec(controllers: list[TrainableController], rig: dict[str, dict[str, float | int | bool]]) -> list[float]:
    """Vectorize one frame of rig controller values, in the order of the trainable controllers.

    The rig can contain more controllers and attributes than the trainable controllers. The function ignores them.

    Args:
        controllers (list[TrainableController]): Trainable controllers that define the vector layout
        rig (dict[str, dict[str, float | int | bool]]): Controller name (without namespace) -> attribute -> value

    Returns:
        list[float]: The pose vector of the frame

    Raises:
        KeyError: If a trainable controller or one of its attributes is missing from the rig
    """
    vec = []
    for controller in controllers:
        controller_name = trainable_controllers.remove_namespace(controller.name)
        if controller_name not in rig:
            raise KeyError(f"Controller '{controller_name}' is missing from the rig.")
        for transformation in controller.transformations:
            try:
                vec.extend(trainable_controllers.controller_rig_to_vec(transformation, rig[controller_name]))
            except KeyError as e:
                raise KeyError(f"Attribute {e} of controller '{controller_name}' is missing from the rig.") from e
    return vec
