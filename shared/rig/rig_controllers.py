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
