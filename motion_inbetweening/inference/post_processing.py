from collections import defaultdict

from motion_inbetweening.config.base import BaseModel
from motion_inbetweening.config.data import RelativeControllerTransformationConfig
from motion_inbetweening.config.post_processing import UnmaskedFramesPostProcessingStrategy
from motion_inbetweening.domain.animation_curve import (
    AnimationCurveType,
    get_animation_curve_from_scene_rig_controllers_values,
    select_non_plateau_frames_ids_from_curve,
)
from motion_inbetweening.domain.controller_keyframes import RigControllersKeyframes, SceneRigControllersKeyframes
from motion_inbetweening.domain.data.data_sample import AnimationSceneData, get_all_frames_ids, get_unmasked_frames_ids
from motion_inbetweening.domain.rig_controllers_values import (
    remove_frames_from_scene_rig_controllers_values,
    scene_rig_controllers_values_to_dict,
)
from shared.domain.entities.rig_controllers_values import (
    FullSceneRigControllersValues,
    RigControllersValues,
    SceneRigControllersValues,
)


class FilteringPlateauConfig(BaseModel):
    """
    Configuration class for filtering plateau tolerance.

    Attributes:
        translation (float): The absolute tolerance value for filtering translation changes.
        rotation (float): The absolute tolerance value for filtering rotation changes.
        scale (float): The absolute tolerance value for filtering scale changes.
    """

    translation: float
    rotation: float
    scale: float


def get_tolerance_from_animation_curve(
    animation_curve_type: AnimationCurveType, filtering_plateau_tolerance_config: FilteringPlateauConfig
) -> float:
    """
    Get the tolerance value based on the animation curve type.
    """
    if animation_curve_type == AnimationCurveType.TRANSLATION:
        return filtering_plateau_tolerance_config.translation
    elif animation_curve_type == AnimationCurveType.ROTATION:
        return filtering_plateau_tolerance_config.rotation
    elif animation_curve_type == AnimationCurveType.SCALE:
        return filtering_plateau_tolerance_config.scale
    else:
        raise ValueError(f"Unknown animation curve type: {animation_curve_type}.")


def get_absolute_predicted_rigs(
    predicted_rigs: list[RigControllersValues],
    first_frame_rig: RigControllersValues,
    relative_controller_transformations: dict[str, list[RelativeControllerTransformationConfig]],
) -> list[RigControllersValues]:
    """
    Converts a list of predicted rigs with relative transformations to absolute transformations.

    Args:
        predicted_rigs (list[dict]): A list of predicted rigs, where each rig is represented as a dictionary.
        first_frame_rig (dict): The rig of the first frame, used as a reference for absolute transformations.
        relative_controller_transformations (dict[str, list[RelativeControllerTransformationConfig]]):
            A dictionary defining the controllers that are relative to the first frame.

    Returns:
        list[dict]: A list of rigs with absolute transformations.
    """
    absolute_predicted_rigs: list[RigControllersValues] = []
    for predicted_rig in predicted_rigs:
        absolute_predicted_rig = predicted_rig
        for controller_name, controller_transformations in relative_controller_transformations.items():
            for transformation_config in controller_transformations:
                transformation = transformation_config.attribute
                first_frame_rig_value = first_frame_rig[controller_name][transformation]
                first_frame_rig_value = float(first_frame_rig_value)
                current_value = float(predicted_rig[controller_name][transformation])
                absolute_predicted_rig = absolute_predicted_rig.set(
                    absolute_predicted_rig, controller_name, transformation, first_frame_rig_value + current_value
                )
        absolute_predicted_rigs.append(absolute_predicted_rig)
    return absolute_predicted_rigs


def reduce_controller_keyframes(
    rig_controllers_values: RigControllersValues, controller_keyframes: RigControllersKeyframes, threshold: float
) -> RigControllersValues:
    """
    Reduces the number of keyframes for each controller in the rig.

    Args:
        rig_controllers_values (RigDict): The rig controllers values.
        controller_keyframes (RigControllersKeyframes): The controller keyframes.
        threshold (float): The threshold for the reduction.

    Returns:
        RigDict: The reduced rig controllers values.
    """
    reduced_rig_controllers_values = defaultdict(dict)
    for controller_name, controller_values in rig_controllers_values.items():
        for transformation, values in controller_values.items():
            if controller_name not in controller_keyframes:
                raise KeyError(f"Controller '{controller_name}' not found in controller keyframes.")
            if transformation not in controller_keyframes[controller_name]:
                raise KeyError(f"Transformation '{transformation}' not found in controller keyframes.")
            controller_keyframe = controller_keyframes[controller_name][transformation]
            if controller_keyframe > threshold:
                reduced_rig_controllers_values[controller_name][transformation] = values
    return RigControllersValues.from_any(reduced_rig_controllers_values)


def reduce_controller_keyframes_in_scene(
    list_rig_controllers_values: list[RigControllersValues],
    list_rig_controllers_keyframes: list[RigControllersKeyframes],
    threshold: float,
) -> list[RigControllersValues]:
    reduced_trainable_controllers_values = []
    for trainable_controller_values, rig_controllers_keyframes in zip(
        list_rig_controllers_values, list_rig_controllers_keyframes, strict=True
    ):
        reduced_trainable_controller_values = reduce_controller_keyframes(
            trainable_controller_values, rig_controllers_keyframes, threshold
        )
        reduced_trainable_controllers_values.append(reduced_trainable_controller_values)
    return reduced_trainable_controllers_values


def get_scene_rig_controllers_values(
    rig_dict: list[RigControllersValues], scene: AnimationSceneData
) -> SceneRigControllersValues:
    scene_rig_controllers_values = {}
    for i, frame_id in enumerate(get_all_frames_ids(scene)):
        scene_rig_controllers_values[frame_id] = rig_dict[i]
    return SceneRigControllersValues.from_any(scene_rig_controllers_values)


def get_scene_rig_controllers_keyframes(
    rig_dict: list[RigControllersKeyframes], scene: AnimationSceneData
) -> SceneRigControllersKeyframes:
    scene_rig_controllers_keyframes = {}
    for i, frame_id in enumerate(get_all_frames_ids(scene)):
        scene_rig_controllers_keyframes[frame_id] = rig_dict[i]
    return SceneRigControllersKeyframes.from_any(scene_rig_controllers_keyframes)


def replace_full_controllers_values_at_unmasked_frames(
    scene_rig_controllers_values: SceneRigControllersValues, scene: AnimationSceneData
) -> SceneRigControllersValues:
    """
    Replace the controller values at unmasked frames in the scene rig with the input rig values.

    This function creates a deep copy of the provided scene rig controller values and updates the
    values at frames that are not masked with the corresponding values from the input rig in the scene.

    Args:
        scene_rig_controllers_values (SceneRigControllersValues): The original scene rig controller values.
        scene (AnimationSceneData): The animation scene data containing the input rig and mask information.

    Returns:
        SceneRigControllersValues: The updated scene rig controller values with replaced values at unmasked frames.
    """
    scene_rig_controllers_values_with_block = scene_rig_controllers_values.to_deepcopy_dict()
    for frame_id in get_unmasked_frames_ids(scene):
        scene_rig_controllers_values_with_block[frame_id] = scene.input_rig[frame_id]
    return SceneRigControllersValues.from_any(scene_rig_controllers_values_with_block)


def add_non_trainable_controllers_values_at_unmasked_frames(
    scene_rig_controllers_values: SceneRigControllersValues, scene: AnimationSceneData
) -> SceneRigControllersValues:
    """
    Adds non-trainable controller values to the scene rig controllers at unmasked frames.

    This function takes the existing scene rig controller values and adds the values of
    non-trainable controllers at frames that are not masked. It allows to keep the non-trainable
    controller values (e.g. facial discrete controllers, binary secondary controllers) from the
    input rig in the scene.

    Args:
        scene_rig_controllers_values (SceneRigControllersValues): The original scene rig
            controllers values.
        scene (AnimationSceneData): The animation scene data containing input rig and mask
            information.

    Returns:
        SceneRigControllersValues: The updated scene rig controllers values with non-trainable
        controller values added at unmasked frames.
    """
    scene_rig_controllers_values_with_block = scene_rig_controllers_values.to_deepcopy_dict()
    for frame_id in get_unmasked_frames_ids(scene):
        for controller_name, controller_values in scene.input_rig[frame_id].items():
            if controller_name not in scene_rig_controllers_values_with_block[frame_id]:
                scene_rig_controllers_values_with_block[frame_id][controller_name] = controller_values
            for transformation, value in controller_values.items():
                if transformation not in scene_rig_controllers_values_with_block[frame_id][controller_name]:
                    scene_rig_controllers_values_with_block[frame_id][controller_name][transformation] = value
    return SceneRigControllersValues.from_any(scene_rig_controllers_values_with_block)


def remove_unmasked_frames(
    scene_rig_controllers_values: SceneRigControllersValues, scene: AnimationSceneData
) -> SceneRigControllersValues:
    return remove_frames_from_scene_rig_controllers_values(scene_rig_controllers_values, get_unmasked_frames_ids(scene))


def filter_plateau_frames(
    full_scene_rig_controllers_values: FullSceneRigControllersValues,
    filtering_plateau_tolerance_config: FilteringPlateauConfig,
) -> SceneRigControllersValues:
    """
    Filters the plateau frames from the full scene rig controllers values.

    A plateau is a sequence of frames where the controller values remain constant.
    For such sequences, only the frames that deviate from the plateau based on the
    specified tolerance are retained.

    Args:
        full_scene_rig_controllers_values (FullSceneRigControllersValues): The full scene rig
            controllers values containing all frames and their respective controller values.
        filtering_plateau_tolerance_config (FilteringPlateauConfig): Configuration specifying
            the tolerance for filtering plateau frames.

    Returns:
        SceneRigControllersValues: The updated scene rig controllers values with plateau frames
        filtered out based on the specified tolerance.
    """
    output_scene_rig_controllers_values_dict = defaultdict(lambda: defaultdict(dict))
    controllers_attribute_tuples = full_scene_rig_controllers_values.to_controller_attribute_tuples()
    for controller_name, attribute_name in controllers_attribute_tuples:
        anim_curve = get_animation_curve_from_scene_rig_controllers_values(
            full_scene_rig_controllers_values, controller_name, attribute_name
        )
        tolerance = get_tolerance_from_animation_curve(anim_curve.transformation, filtering_plateau_tolerance_config)
        frames_ids_to_keep = select_non_plateau_frames_ids_from_curve(anim_curve, tolerance)
        for frame_id in frames_ids_to_keep:
            output_scene_rig_controllers_values_dict[frame_id][controller_name][attribute_name] = (
                full_scene_rig_controllers_values[frame_id][controller_name][attribute_name]
            )
    return SceneRigControllersValues.from_any(output_scene_rig_controllers_values_dict)


def post_process_unmasked_frames(
    scene_rig_controllers_values: SceneRigControllersValues,
    scene: AnimationSceneData,
    block_keyframes_post_processing_strategy: UnmaskedFramesPostProcessingStrategy,
) -> SceneRigControllersValues:
    """Handles the post processing of the unmasked frames.

    The different strategies are:
        - KEEP_PREDICTED: add the non trainable controllers values at unmasked frames
        - FORCE_INPUT: add the non trainable controllers values at unmasked frames and replace the predicted frames
        - REMOVE: remove the unmasked frames altogether
    """
    match block_keyframes_post_processing_strategy:
        case UnmaskedFramesPostProcessingStrategy.FORCE_INPUT:
            return replace_full_controllers_values_at_unmasked_frames(scene_rig_controllers_values, scene)
        case UnmaskedFramesPostProcessingStrategy.KEEP_PREDICTED:
            return add_non_trainable_controllers_values_at_unmasked_frames(scene_rig_controllers_values, scene)
        case UnmaskedFramesPostProcessingStrategy.REMOVE:
            return remove_unmasked_frames(scene_rig_controllers_values, scene)


def post_process_rig_prediction(
    predicted_controller_values: list[RigControllersValues],
    scene: AnimationSceneData,
    relative_controller_transformations: dict[str, list[RelativeControllerTransformationConfig]],
    do_controller_keyframes_reduction: bool,
    predicted_controller_keyframes: list[RigControllersKeyframes] | None,
    threshold_controller_keyframes_reduction: float | None,
    unmasked_frames_post_processing_strategy: UnmaskedFramesPostProcessingStrategy,
    filter_plateau_tolerance_config: FilteringPlateauConfig | None,
) -> SceneRigControllersValues:
    """
    Post-processes the predicted rig configurations for an animation scene.

    Args:
        predicted_rigs (list[RigDict]): A list of predicted rig configurations.
        scene (AnimationSceneData): The animation scene data containing input rig and frame range.
        relative_controller_transformations (dict[str, list[RelativeControllerTransformationConfig]]):
            A dictionary indicating the relative controllers
        predicted_controller_values (list[RigControllersValues]): A list of predicted controller values.
            A dictionary indicating the relative controllers and their transformations.
        do_controller_keyframes_reduction (bool): Flag indicating whether to reduce the number of keyframes for
            each controller.
        predicted_controller_keyframes (list[RigControllersKeyframes] | None): A list of predicted controller keyframes.
            Must be provided if `do_controller_keyframes_reduction` is True.
        threshold_controller_keyframes_reduction (float | None): Threshold for reducing controller keyframes.
            Must be provided if `do_controller_keyframes_reduction` is True.
        unmasked_frames_post_processing_strategy (UnmaskedFramesPostProcessingStrategy): Post-processing strategy for
            the unmasked frames.
        filter_plateau_tolerance_config (FilteringPlateauConfig | None): Tolerance configuration for filtering
            plateau frames. If None, plateau filtering is not applied.

    Returns
        SceneRigControllersValues: A dictionary representing the scene's controller values after applying
            the predictions and post-processing.
    """
    absolute_predicted_rigs = get_absolute_predicted_rigs(
        predicted_controller_values,
        scene.input_rig.get_first_frame_rig_controllers_values(),
        relative_controller_transformations,
    )
    if do_controller_keyframes_reduction:
        if predicted_controller_keyframes is None:
            raise ValueError("Controller keyframes must be provided when reducing controller keyframes.")
        if len(predicted_controller_keyframes) != len(predicted_controller_values):
            raise ValueError("Controller keyframes must have the same length as the predicted values.")
        if threshold_controller_keyframes_reduction is None:
            raise ValueError("Threshold for controller keyframes reduction must be provided.")
        reduced_rig_dict = reduce_controller_keyframes_in_scene(
            absolute_predicted_rigs, predicted_controller_keyframes, threshold_controller_keyframes_reduction
        )
    else:
        reduced_rig_dict = absolute_predicted_rigs
    scene_rig_controllers_values = get_scene_rig_controllers_values(reduced_rig_dict, scene)
    if filter_plateau_tolerance_config is not None:
        full_scene_rig_controllers_values = FullSceneRigControllersValues.from_any(
            scene_rig_controllers_values_to_dict(scene_rig_controllers_values)
        )
        scene_rig_controllers_values = filter_plateau_frames(
            full_scene_rig_controllers_values, filter_plateau_tolerance_config
        )
    return post_process_unmasked_frames(scene_rig_controllers_values, scene, unmasked_frames_post_processing_strategy)
