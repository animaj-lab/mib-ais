from dataclasses import dataclass
from pathlib import Path

from motion_inbetweening.domain.data.data_sample import AnimationSceneData
from motion_inbetweening.domain.data.predict import PredictDataType
from motion_inbetweening.infra.paths.dataset import to_predict_video_path


@dataclass
class QualitativeEvaluationDirectories:
    input_predict_dataset_dir: Path
    output_predicted_rigs_dir: Path
    output_predicted_videos_dir: Path
    output_comparison_videos_dir: Path


def initialize_qualitative_evaluation_directories(
    input_predict_dataset_dir: Path, output_dir: Path
) -> QualitativeEvaluationDirectories:
    return QualitativeEvaluationDirectories(
        input_predict_dataset_dir=input_predict_dataset_dir.resolve(),
        output_predicted_rigs_dir=output_dir / "predicted_rigs",
        output_predicted_videos_dir=output_dir / "predicted_videos",
        output_comparison_videos_dir=output_dir / "comparison_videos",
    )


def get_ground_truth_video_filepath(
    qualitative_evaluation_directories: QualitativeEvaluationDirectories, scene_data: AnimationSceneData
) -> Path:
    return to_predict_video_path(
        qualitative_evaluation_directories.input_predict_dataset_dir, PredictDataType.ground_truth, scene_data.name
    )


def get_maya_baseline_video_filepath(
    qualitative_evaluation_directories: QualitativeEvaluationDirectories, scene_data: AnimationSceneData
) -> Path:
    return to_predict_video_path(
        qualitative_evaluation_directories.input_predict_dataset_dir, PredictDataType.maya_baseline, scene_data.name
    )


def get_predicted_rig_filepath(
    qualitative_evaluation_directories: QualitativeEvaluationDirectories, scene_data: AnimationSceneData
) -> Path:
    return qualitative_evaluation_directories.output_predicted_rigs_dir / scene_data.name / "rig.json"


def get_predicted_video_filepath(
    qualitative_evaluation_directories: QualitativeEvaluationDirectories, scene_data: AnimationSceneData
) -> Path:
    return qualitative_evaluation_directories.output_predicted_videos_dir / scene_data.name / "predicted.mp4"


def get_comparison_video_filepath(
    qualitative_evaluation_directories: QualitativeEvaluationDirectories, scene_data: AnimationSceneData
) -> Path:
    return qualitative_evaluation_directories.output_comparison_videos_dir / scene_data.name / "comparison.mp4"
