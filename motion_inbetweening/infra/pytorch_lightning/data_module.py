import logging

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split

from motion_inbetweening.app_services.online_preprocessing.batching import collate_fn, predict_collate_fn
from motion_inbetweening.app_services.online_preprocessing.scene_splitting import SceneSplitting
from motion_inbetweening.config.data import PredictDataModuleConfig, TrainDataModuleConfig
from motion_inbetweening.domain.data.data_sample import AnimationSceneData
from motion_inbetweening.domain.data.split_type import SplitType
from motion_inbetweening.infra.loading.dataset import load_and_instantiate_dataset, load_and_instantiate_predict_dataset
from shared.rig.normalizer import initialize_pose_normalizer
from shared.rig.trainable_controllers import TrainableController

logger = logging.getLogger(__name__)


class MotionInbetweeningDatamodule(pl.LightningDataModule):
    def __init__(
        self, config: TrainDataModuleConfig | PredictDataModuleConfig, trainable_controllers: list[TrainableController]
    ):
        super().__init__()
        self.config = config
        self.trainable_controllers = trainable_controllers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

    def setup(self, stage: str):
        if stage == "fit" or stage == "validate":
            if self.train_dataset is None or self.val_dataset is None:
                logger.info("Setting up train and validation datasets")
                if not isinstance(self.config, TrainDataModuleConfig):
                    raise TypeError("Passed predict data module config, but expected train data module config")
                train_val_set = load_and_instantiate_dataset(
                    dataset_config=self.config.train_dataset,
                    scene_splitting=self.config.train_scene_splitting,
                    trainable_controllers=self.trainable_controllers,
                    split_type=SplitType.TRAIN,
                    controller_keyframes_preprocessing_config=self.config.controller_keyframes_preprocessing_config,
                )
                val_size = int(self.config.validation_ratio * len(train_val_set))
                train_size = len(train_val_set) - val_size
                self.train_dataset, self.val_dataset = random_split(train_val_set, [train_size, val_size])
                if self.config.do_normalization:
                    training_data = self.train_dataset.dataset.list_data_samples
                    vectors = torch.cat([train_data_sample.vectors for train_data_sample in training_data], dim=0)
                    self.pose_normalizer = initialize_pose_normalizer(vectors)
                else:
                    self.pose_normalizer = None
        if stage == "test":
            self.pose_normalizer = None
            if self.test_dataset is None:
                logger.info("Setting up test dataset")
                if not isinstance(self.config, TrainDataModuleConfig):
                    raise TypeError("Passed predict data module config, but expected train data module config")
                if self.config.test_dataset is None:
                    raise ValueError("Test dataset is required for test stage")
                self.test_dataset = load_and_instantiate_dataset(
                    dataset_config=self.config.test_dataset,
                    scene_splitting=self.config.test_scene_splitting,
                    trainable_controllers=self.trainable_controllers,
                    split_type=SplitType.TEST,
                    controller_keyframes_preprocessing_config=self.config.controller_keyframes_preprocessing_config,
                )
        if stage == "predict":
            if self.predict_dataset is None:
                logger.info("Setting up predict dataset")
                if self.config.predict_dataset is None:
                    raise ValueError("Predict dataset is required for predict stage")
                self.predict_dataset = load_and_instantiate_predict_dataset(
                    predict_dataset_config=self.config.predict_dataset,
                    scene_splitting=self.config.predict_scene_splitting,
                    trainable_controllers=self.trainable_controllers,
                )

    def __collate_fn(
        self,
        batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        scene_splitting: SceneSplitting,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert isinstance(self.config, TrainDataModuleConfig), "use _predict_collate_fn for predict stage"
        return collate_fn(batch, scene_splitting, self.pose_normalizer)

    def _predict_collate_fn(
        self, batch: list[tuple[torch.Tensor, AnimationSceneData]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[int]], list[AnimationSceneData]]:
        assert self.config.predict_dataset is not None, "Predict dataset is required for predict stage"
        return predict_collate_fn(batch, self.config.predict_scene_splitting)

    def train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Train dataset is not set up")
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=lambda batch: self.__collate_fn(batch, self.config.train_scene_splitting),
            num_workers=self.config.num_workers,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            raise ValueError("Validation dataset is not set up")
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=lambda batch: self.__collate_fn(batch, self.config.train_scene_splitting),
            num_workers=self.config.num_workers,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            raise ValueError("Test dataset is not set up")
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            collate_fn=lambda batch: self.__collate_fn(batch, self.config.test_scene_splitting),
            num_workers=self.config.num_workers,
        )

    def predict_dataloader(self):
        if self.predict_dataset is None:
            raise ValueError("Predict dataset is not set up")
        return DataLoader(self.predict_dataset, batch_size=self.config.batch_size, collate_fn=self._predict_collate_fn)
