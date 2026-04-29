import torch

from motion_inbetweening.domain.data.data_sample import TrainDataSample
from shared.domain.entities.animation import EpisodeID, SceneID

VectorizedControllersValuesDict = dict[tuple[EpisodeID, SceneID], dict[str, torch.Tensor]]
VectorizedBlockKeyframesDict = dict[tuple[EpisodeID, SceneID], torch.Tensor]
VectorizedControllersKeyframesDict = dict[tuple[EpisodeID, SceneID], torch.Tensor]


def get_list_data_samples(
    controller_values_data: VectorizedControllersValuesDict,
    block_keyframes_data: VectorizedBlockKeyframesDict,
    controller_keyframes_data: VectorizedControllersKeyframesDict | None,
) -> list[TrainDataSample]:
    motion_data = []
    for episode_id, scene_id in controller_values_data.keys():
        controller_values_vector = controller_values_data[episode_id, scene_id]["vectors"]
        animation_keyframes_vector = controller_values_data[episode_id, scene_id]["animation_keyframes"]
        if (episode_id, scene_id) not in block_keyframes_data:
            continue
        block_keyframes_vector = block_keyframes_data[episode_id, scene_id]
        if controller_keyframes_data is not None:
            controller_keyframes_vector = controller_keyframes_data[episode_id, scene_id]
        else:
            controller_keyframes_vector = torch.zeros((controller_values_vector.shape[0], 1)).to(torch.float32)
        motion_data.append(
            TrainDataSample(
                episode_id=episode_id,
                scene_id=scene_id,
                vectors=controller_values_vector,
                block_keyframes=block_keyframes_vector,
                animation_keyframes=animation_keyframes_vector,
                controller_keyframes=controller_keyframes_vector,
            )
        )
    return motion_data
