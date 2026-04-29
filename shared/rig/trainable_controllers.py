import math
import re
from enum import StrEnum
from pathlib import Path

import numpy as np
import pydantic
import torch

from shared.domain.entities.rig_controllers_values import AttributeName
from shared.rig.config import ControllerConfig
from shared.rig.utils import (
    euler_angles_to_matrix,
    matrix_to_euler_angles,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from shared.utils import load_yaml


class Transformation(pydantic.BaseModel):
    """A transformation is a group of attribute that are processed together in the model.

    For instance all the rotation attributes of a controller are grouped together during training.
    Then depending of the modelisation, we could learn 3 values (the 3 rotations), 6 value (a 6D modelisation)
    or 9 values (a 3x3 rotation matrix) for this single transformation.

    All transformations have a `coeff` attribute that is used to weight the loss of this transformation.

    """

    coeff: float

    def to_vector_size(self) -> int:
        return to_vector_size(self)

    def to_transformation_name(self) -> str:
        return to_transformation_name(self)


class TrainableController(pydantic.BaseModel):
    name: str
    display_name: str
    transformations: list[Transformation]


class Numerical(Transformation):
    pass


class Translation(Numerical):
    axis: str
    scale: float = 1.0
    center: float = 0.0
    do_clip: bool = True

    def __str__(self) -> str:
        return f"Translation(axis={self.axis}, scale={self.scale}, coeff={self.coeff}, do_clip={self.do_clip})"


class Rotation3D(Numerical):
    pass

    def __str__(self) -> str:
        return f"Rotation3D(coeff={self.coeff})"


class Rotation1D(Numerical):
    axis: str

    def __str__(self) -> str:
        return f"Rotation1D(axis={self.axis}, coeff={self.coeff})"


class Scale(Numerical):
    axis: str = "XYZ"
    scale: float = 1.0
    center: float = 1.0
    do_clip: bool = True

    def __str__(self) -> str:
        return f"Scale(axis={self.axis}, scale={self.scale}, coeff={self.coeff}, do_clip={self.do_clip})"


class Boolean(Transformation):
    attribute_name: str

    def __str__(self) -> str:
        return f"Boolean(attribute_name={self.attribute_name}, coeff={self.coeff})"


class DiscreteAttribute(Transformation):
    """A discrete attribute in maya is a member of a numerical enumeration.

    In our modelisation, it is possible to group several labels together, leading to less classes to be
    predicted by the model.

    The `aggregate_classes` attribute is a dictionary that gives the mapping from the original labels to the new ones.
    The set of keys are all the values of [0, N) where N is the number of items in the Maya enumeration. Each value is
    present only once.
    The set of values are also values of [0, N) with repetition allowed.
    (Note that in a future implementation, we could use a list of len N instead of a dictionary.)

    The `predicted_labels_id` is the inverse mapping, that gives for each predicted label a class representent
    in the original labels.

    The `num_classes` attribute is the number of classes that are predicted by the model. Please note that in the
    general case it is not equal to the number of items in the Maya enumeration.
    """

    attribute_name: str
    num_classes: int
    aggregate_classes: dict[int, int] = pydantic.Field(default_factory=lambda: {})
    predicted_labels_id: list[int] = pydantic.Field(default_factory=lambda: [])

    def __str__(self) -> str:
        return (
            f"DiscreteAttribute(attribute_name={self.attribute_name}, num_classes={self.num_classes}, "
            f"coeff={self.coeff})"
        )


def to_transformation_name(transformation: Transformation) -> str:
    match transformation:
        case Translation():
            return "Translation"
        case Rotation3D():
            return "Rotation3D"
        case Rotation1D():
            return "Rotation1D"
        case Scale():
            return "Scale"
        case DiscreteAttribute(attribute_name=attribute_name):
            return f"DiscreteAttribute_{attribute_name}"
        case Boolean(attribute_name=attribute_name):
            return f"Boolean_{attribute_name}"
        case _:
            raise ValueError(f"Unknown transformation type: {transformation}")


def to_vector_size(transformation: Transformation) -> int:
    match transformation:
        case Translation(axis=axis):
            return len(axis)
        case Rotation3D():
            return 6
        case Rotation1D():
            return 2
        case Scale(axis=axis):
            return len(axis)
        case DiscreteAttribute(num_classes=num_classes):
            return num_classes
        case Boolean():
            return 1
        case _:
            raise ValueError(f"Unknown controller type: {transformation}")


def controller_vec_to_rig(transformation: Transformation, vec: torch.Tensor) -> dict[str, int | float | bool]:
    match transformation:
        case Rotation3D():
            rotation_matrix = rotation_6d_to_matrix(vec)
            euler_angle_radians = matrix_to_euler_angles(rotation_matrix, convention="XYZ")
            euler_angle_degrees = [math.degrees(x) for x in euler_angle_radians]
            return {f"rotate{ax}": angle for ax, angle in zip("XYZ", euler_angle_degrees, strict=True)}
        case Rotation1D(axis=axis):
            value_radians = math.atan2(vec[1], vec[0])
            return {f"rotate{axis}": math.degrees(value_radians)}
        case Translation(axis=axis, scale=scale, center=center):
            vec = vec * scale + center
            return {f"translate{ax}": vec[i].item() for i, ax in enumerate(axis)}
        case Scale(axis=axis, scale=scale, center=center):
            vec = vec * scale + center
            return {f"scale{ax}": vec[i].item() for i, ax in enumerate(axis)}
        case DiscreteAttribute(predicted_labels_id=predicted_labels_id, attribute_name=attribute_name):
            vec = vec.cpu().to(torch.float32)
            class_num = int(np.argmax(a=vec))
            if predicted_labels_id:
                class_num = predicted_labels_id[class_num]
            return {attribute_name: class_num}
        case Boolean(attribute_name=attribute_name):
            vec = vec.cpu().to(torch.float32)
            return {attribute_name: bool(vec[0] > 0.5)}
        case _:
            raise ValueError(f"Unknown controller type: {transformation}")


def controller_rig_to_vec(transformation: Transformation, rig: dict[str, float]) -> list[float]:
    match transformation:
        case Rotation3D():
            output = []
            for ax in "XYZ":
                output.append(rig[f"rotate{ax}"])
            radians = torch.Tensor([math.radians(x) for x in output])
            rotation_matrix = euler_angles_to_matrix(radians, convention="XYZ")
            return list(matrix_to_rotation_6d(rotation_matrix).numpy())
        case Rotation1D(axis=axis):
            value = rig[f"rotate{axis}"]
            return convert_rotation1d_to_vector(value=value)
        case Boolean(attribute_name=attribute_name):
            return [int(rig[attribute_name])]
        case DiscreteAttribute(
            num_classes=num_classes,
            aggregate_classes=aggregate_classes,
            attribute_name=attribute_name,
            predicted_labels_id=predicted_labels_id,
        ):
            output = [0] * num_classes
            class_num = int(rig[attribute_name])
            if aggregate_classes:
                new_class_num = aggregate_classes[class_num]
                class_num = predicted_labels_id.index(new_class_num)
            output[class_num] = 1
            return list(output)
        case Scale(axis=axis, scale=scale, center=center, do_clip=do_clip):
            output = np.array([rig[f"scale{ax}"] for ax in axis], dtype=np.float64)
            output = (output - center) / scale
            if do_clip:
                output = output.clip(-1, 1)
            return list(output)
        case Translation(axis=axis, scale=scale, center=center, do_clip=do_clip):
            output = np.array([rig[f"translate{ax}"] for ax in axis], dtype=np.float64)
            output = (output - center) / scale
            if do_clip:
                output = output.clip(-1, 1)
            return list(output)
        case _:
            raise ValueError(f"Unknown controller type: {transformation}")


def transformation_to_attributes_names(transformation: Transformation) -> list[AttributeName]:
    match transformation:
        case Rotation3D():
            return [f"rotate{ax}" for ax in "XYZ"]
        case Rotation1D(axis=axis):
            return [f"rotate{axis}"]
        case Boolean():
            raise NotImplementedError("Boolean transformation is not supported.")
        case DiscreteAttribute():
            raise NotImplementedError("DiscreteAttribute transformation is not supported.")
        case Scale(axis=axis):
            return [f"scale{ax}" for ax in axis]
        case Translation(axis=axis):
            return [f"translate{ax}" for ax in axis]
        case _:
            raise ValueError(f"Unknown controller type: {transformation}")


def convert_rotation1d_to_vector(value: float) -> list[float]:
    value_radians = math.radians(value)
    return [math.cos(value_radians), math.sin(value_radians)]


def generate_random_values(transformation: Transformation, num_values: int) -> np.ndarray:
    match transformation:
        case Rotation3D():
            values = np.random.normal(-50, 30, num_values)
            values += np.random.normal(50, 30, num_values)
            values_normalized = _normalize_angles_array(values)
            return values_normalized
        case Rotation1D():
            values = np.random.normal(-50, 30, num_values)
            values += np.random.normal(50, 30, num_values)
            values_normalized = _normalize_angles_array(values)
            return values_normalized
        case Boolean():
            return np.random.randint(0, 2, num_values).tolist()
        case DiscreteAttribute(num_classes=num_classes, aggregate_classes=aggregate_classes):
            values = np.random.randint(0, num_classes, num_values).tolist()
            if aggregate_classes:
                values = np.array([aggregate_classes[v] for v in values])
            return values
        case Translation(scale=scale, center=center):
            values = np.random.normal(-0.5, 0.2, num_values)
            values += np.random.normal(0.5, 0.2, num_values)
            return values * scale + center
        case Scale(scale=scale, center=center):
            values = np.random.normal(-0.5, 0.2, num_values)
            values += np.random.normal(0.5, 0.2, num_values)
            return values * scale + center
        case _:
            raise ValueError(f"Unknown controller type: {transformation}")


def _parse_controller_def(controller_trainable_attr: str) -> Transformation:
    match = re.match(r"(?P<class_name>\w+)(\((?P<params_str>.*)\))?$", controller_trainable_attr)
    if not match:
        raise ValueError(f"Invalid class name format: {controller_trainable_attr}")
    class_name = match.group("class_name").strip()
    params_str = match.group("params_str")
    params = {}
    if params_str:
        params = dict(re.findall(r"(\w+)=([^,\s)]+)", params_str))
    match class_name:
        case "Translation":
            return Translation(**params)
        case "Rotation3D":
            return Rotation3D(**params)
        case "Scale":
            return Scale(**params)
        case "Rotation1D":
            return Rotation1D(**params)
        case "DiscreteAttribute":
            return DiscreteAttribute(**params)
        case _:
            raise ValueError(f"Invalid class name: {class_name}")


def load_trainable_controllers_from_conf(conf: ControllerConfig) -> list[TrainableController]:
    return load_trainable_controllers(conf.trainable_controllers_path, conf.simplify_classes_path)


def remove_namespace(object_name: str) -> str:
    """Removes the namespace from the object name.

    Args:
        object_name (str): 'namespace:object_name' or 'object_name'

    Returns:
        str: new object name without the namespace -> 'object_name'
    """
    return object_name.split(":")[-1]


def load_trainable_controllers(
    yaml_file_path: Path, simplification_yaml_file_path: Path | None
) -> list[TrainableController]:
    """load the yaml file and generate a AST of the custom language

    Note that previous version returned a dictionary.
    It was changed to a list of custom type to explicitly keep the order of the controllers.
    """
    output = []
    simplification_dict = {}
    if simplification_yaml_file_path is not None:
        simplification_dict = load_yaml(simplification_yaml_file_path)
    trainable_controllers = load_yaml(yaml_file_path)
    if not isinstance(trainable_controllers, dict):
        raise ValueError("Trainable controllers should be a dictionary.")
    for idx, (controller_name, v) in enumerate(trainable_controllers.items()):
        if "x_main_CTRL" in controller_name and idx != 0:
            raise ValueError("If x_main_CTRL is present in trainable controllers, it should be the first one.")
        if not isinstance(controller_name, str):
            raise ValueError("Controller name should be a string.")
        if not isinstance(v, list):
            raise ValueError(f"Controller {controller_name} should be a list.")
        controllers_types = []
        for controller_trainable_attr in v:
            try:
                controller_def = _parse_controller_def(controller_trainable_attr)
            except Exception as e:
                raise ValueError(
                    f"Error parsing controller {controller_trainable_attr} for controller {controller_name}"
                ) from e
            if isinstance(controller_def, DiscreteAttribute) and simplification_dict.get(controller_name):
                controller_def.aggregate_classes = simplification_dict[controller_name]
                controller_def.predicted_labels_id = sorted(list(set(controller_def.aggregate_classes.values())))
                controller_def.num_classes = len(controller_def.predicted_labels_id)
            controllers_types.append(controller_def)
        display_name = remove_namespace(controller_name).split("_CTRL")[0]
        output.append(
            TrainableController(name=controller_name, display_name=display_name, transformations=controllers_types)
        )
    return output


def serialize_trainable_controllers_as_yaml_str(trainable_controllers: list[TrainableController]) -> str:
    """Dump a list of TrainableController into the yaml format used in the trainable controllers files, as a string."""
    res = ""
    for tc in trainable_controllers:
        res += f"{tc.name}:\n"
        for transformation in tc.transformations:
            res += f"  - {transformation}\n"

    return res


def save_trainable_controllers(trainable_controllers: list[TrainableController], path: Path) -> None:
    """Save a list of TrainableController into a yaml file."""
    with open(path, "w") as f:
        f.write(serialize_trainable_controllers_as_yaml_str(trainable_controllers))


def _normalize_angles_array(array: np.ndarray) -> np.ndarray:
    return (array + 180) % 360 - 180


class LossType(StrEnum):
    CATEGORICAL = "categorical"
    NUMERICAL = "numerical"


def split_vectors_using_controllers(
    rig_controllers: list[TrainableController],
    gt_controllers_vectors: torch.Tensor,
    predict_controllers_vectors: torch.Tensor,
    format_for_test: bool,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[float], list[str], list[str]]:
    """Function used only in the training/validation/testing steps of the LightningModule."""
    split_gt = []
    split_pred = []
    coeffs = []
    losses_types = []
    controller_names = []
    previous_id = 0
    for controller in rig_controllers:
        for transformation in controller.transformations:
            match transformation:
                case Boolean() | DiscreteAttribute():
                    losses_types.append(LossType.CATEGORICAL)
                case Numerical():
                    losses_types.append(LossType.NUMERICAL)
            coeffs.append(transformation.coeff)
            controller_names.append(f"{controller.display_name}_{transformation.to_transformation_name()}")
            size = transformation.to_vector_size()
            split_gt_ = gt_controllers_vectors[:, previous_id : previous_id + size]
            split_pred_ = predict_controllers_vectors[:, previous_id : previous_id + size]
            if format_for_test and isinstance(transformation, DiscreteAttribute):
                split_gt_ = split_gt_.argmax(dim=1)
                split_pred_ = split_pred_.argmax(dim=1)
            split_gt.append(split_gt_)
            split_pred.append(split_pred_)
            previous_id += size
    return split_gt, split_pred, coeffs, losses_types, controller_names


def create_controllers_weights_vector(controllers: list[TrainableController]) -> torch.Tensor:
    """Create a vector of weights for each controller transformation.
    weights"""
    weights = []
    for controller in controllers:
        for transformation in controller.transformations:
            length = transformation.to_vector_size()
            for _ in range(length):
                weights.append(transformation.coeff)
    return torch.tensor(weights, dtype=torch.float32)


def to_full_vector_size(controllers: list[TrainableController]) -> int:
    return sum(
        [to_vector_size(transformation) for controller in controllers for transformation in controller.transformations]
    )


def compute_list_controllers_attributes_in_trainable(trainable_controllers: list[TrainableController]) -> list[str]:
    """
    Compute the list of controllers attributes in trainable controllers.
    """
    list_controllers_attributes_in_trainable = []
    for controller in trainable_controllers:
        for transformation in controller.transformations:
            for attribute in transformation_to_attributes_names(transformation):
                name = f"{controller.name}:{attribute}"
                list_controllers_attributes_in_trainable.append(name)
    return list_controllers_attributes_in_trainable
