from enum import StrEnum
from typing import Self

import pydantic

from motion_inbetweening.config.base import MutableBaseModel


class ReplacementDistribution(StrEnum):
    CONDITIONAL = "conditional"
    MARGINAL = "marginal"


class ImputationParams(MutableBaseModel):
    replacement_distribution: ReplacementDistribution
    stop_imputation_at: int


class GuidanceParams(MutableBaseModel):
    stop_recguidance_at: int
    gradient_schedule: str
    diffusion_steps: int
    reconstruction_weight: float


class InferenceDiffusionConfig(MutableBaseModel):
    use_ema_model: bool
    do_imputation: bool
    imputation_params: ImputationParams | None
    do_guidance: bool
    guidance_params: GuidanceParams | None

    @pydantic.model_validator(mode="after")
    def validate_imputation(self) -> Self:
        if self.do_imputation and self.imputation_params is None:
            raise ValueError("Imputation params must be provided if do_imputation is True")
        return self

    @pydantic.model_validator(mode="after")
    def validate_guidance(self) -> Self:
        if self.do_guidance and self.guidance_params is None:
            raise ValueError("Guidance params must be provided if do_guidance is True")
        return self

    def get_subdirectory(self) -> str:
        subdirectory = f"{('ema' if self.use_ema_model else 'no_ema')}"
        if self.do_imputation:
            subdirectory += f"_imputation_stop_at_{self.imputation_params.stop_imputation_at}"
        if self.do_guidance:
            subdirectory += (
                f"_guidance_stop_at_{self.guidance_params.stop_recguidance_at}"
                f"_rec_weight_{self.guidance_params.reconstruction_weight}"
            )
        return subdirectory
