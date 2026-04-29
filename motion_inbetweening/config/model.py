from enum import StrEnum, auto

from motion_inbetweening.config.base import BaseModel


class UNetTransformerModelConfig(BaseModel):
    dim: int = 256
    dim_mults: tuple
    attention: bool
    adagn: bool
    zero: bool
    added_input_channels: int


class InputResidualConnectionConfig(BaseModel):
    add_input_residual_connection_values: bool
    add_input_residual_connections_controller_keyframes: bool


class BiDirectionalLSTMModelConfig(BaseModel):
    hidden_size: int
    num_layers: int
    dropout: float
    input_residual_connection: InputResidualConnectionConfig


class ResidualLSTMModelConfig(BaseModel):
    hidden_size: int
    num_layers: int


class LastLayerMode(StrEnum):
    DIRECT_SYNTHESIS = auto()
    LEARNED_INTERPOLATION_ONLY = auto()
    PREVIOUS = auto()
    LINEAR = auto()
    SLERP = auto()
    INTERP_PLUS_SYNTHESIS_NO_GATE = auto()
    OFFSET = auto()
    AIS = auto()
    ORIGINAL = auto()
    FIXED_BETA = auto()


class ExplicitInterpolationExtrapolationResidualLSTMModelConfig(BaseModel):
    hidden_size: int
    num_layers: int
    dropout: float
    last_layer_mode: LastLayerMode


class TransformerModelConfig(BaseModel):
    d_model: int
    nhead: int
    num_encoder_layers: int
    num_decoder_layers: int
    dim_feedforward: int
    dropout: float


class CITLModelConfig(BaseModel):
    embed_size: int
    max_length: int
    heads: int
    key_layers: int
    interm_layers: int
    dec_layers: int
    dropout: float
    last_layer_mode: LastLayerMode


class DeltaInterpolatorModelConfig(BaseModel):
    num_blocks_enc: int
    num_layers_enc: int
    layer_width_enc: int
    num_blocks_dec: int
    num_layers_dec: int
    layer_width_dec: int
    dropout: float
    embedding_dim: int
    embedding_size: int
    embedding_num: int
    num_heads: int
    delta_mode: str
    input_delta_mode: str
    last_layer_mode: LastLayerMode


ModelConfig = (
    UNetTransformerModelConfig
    | BiDirectionalLSTMModelConfig
    | TransformerModelConfig
    | ResidualLSTMModelConfig
    | ExplicitInterpolationExtrapolationResidualLSTMModelConfig
    | CITLModelConfig
    | DeltaInterpolatorModelConfig
)
