from motion_inbetweening.config.base import BaseModel


class OptimizerConfig(BaseModel):
    name: str
    lr: float
    tune_lr: bool
    params: dict[str, float]


class SchedulerConfig(BaseModel):
    name: str
    params: dict[str, float | str]


class RegularizationConfig(BaseModel):
    last_layer_regularization: bool
    last_layer_regularization_weight: float
    full_model_regularization: bool
    full_model_regularization_weight: float


class OptimizationConfig(BaseModel):
    optimizer: OptimizerConfig
    lr_scheduler: SchedulerConfig
    regularization: RegularizationConfig
