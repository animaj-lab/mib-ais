from motion_inbetweening.config.model import (
    BiDirectionalLSTMModelConfig,
    CITLModelConfig,
    DeltaInterpolatorModelConfig,
    ExplicitInterpolationExtrapolationResidualLSTMModelConfig,
    LastLayerMode,
    ModelConfig,
    ResidualLSTMModelConfig,
    TransformerModelConfig,
    UNetTransformerModelConfig,
)
from motion_inbetweening.domain.models.citl.inpainter import Inpainter
from motion_inbetweening.domain.models.delta_interpolator.modules.infill_transformer import (
    DeltaInterpolatorInfillTransformer,
)
from motion_inbetweening.domain.models.explicit_interpolation_extrapolation_residual_lstm import (
    ExplicitInterpolationExtrapolationResidualLSTM,
)
from motion_inbetweening.domain.models.lstm import BiDirectionalLSTMModel
from motion_inbetweening.domain.models.residual_lstm import BidirectionalResidualLSTM
from motion_inbetweening.domain.models.transformer import TransformerModel
from motion_inbetweening.domain.models.unet_transformer import UNetTransformerModel
from shared.rig.trainable_controllers import TrainableController, to_full_vector_size


def initialize_model(
    model_config: ModelConfig,
    trainable_controllers: list[TrainableController],
    do_controller_keyframes_prediction: bool,
    controller_keyframes_dim: int,
) -> (
    BidirectionalResidualLSTM
    | ExplicitInterpolationExtrapolationResidualLSTM
    | UNetTransformerModel
    | TransformerModel
    | BiDirectionalLSTMModel
    | Inpainter
):
    pose_dim = to_full_vector_size(trainable_controllers)
    match model_config:
        case UNetTransformerModelConfig():
            if do_controller_keyframes_prediction:
                raise NotImplementedError("Controller keyframes prediction not implemented for UNetTransformerModel")
            return UNetTransformerModel(config=model_config, pose_dim=pose_dim)
        case BiDirectionalLSTMModelConfig():
            return BiDirectionalLSTMModel(
                config=model_config,
                pose_dim=pose_dim,
                do_controller_keyframes_prediction=do_controller_keyframes_prediction,
                controller_keyframes_dim=controller_keyframes_dim,
            )
        case ResidualLSTMModelConfig(hidden_size=hidden_size, num_layers=num_layers):
            if do_controller_keyframes_prediction:
                raise NotImplementedError("Controller keyframes prediction not implemented for ResidualLSTMModel")
            return BidirectionalResidualLSTM(
                input_size=pose_dim + 1, hidden_size=hidden_size, output_size=pose_dim, num_layers=num_layers
            )
        case ExplicitInterpolationExtrapolationResidualLSTMModelConfig(
            hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, last_layer_mode=last_layer_mode
        ):
            if do_controller_keyframes_prediction:
                raise NotImplementedError(
                    "Controller keyframes prediction not implemented for PreviousAndNextResidualLSTMDirectInputModel"
                )
            if last_layer_mode == LastLayerMode.SLERP:
                return ExplicitInterpolationExtrapolationResidualLSTM(
                    input_size=pose_dim + 1,
                    hidden_size=hidden_size,
                    output_size=pose_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                    last_layer_mode=last_layer_mode,
                    trainable_controllers=trainable_controllers,
                )
            else:
                return ExplicitInterpolationExtrapolationResidualLSTM(
                    input_size=pose_dim + 1,
                    hidden_size=hidden_size,
                    output_size=pose_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                    last_layer_mode=last_layer_mode,
                )
        case TransformerModelConfig():
            if do_controller_keyframes_prediction:
                raise NotImplementedError("Controller keyframes prediction not implemented for TransformerModel")
            return TransformerModel(config=model_config, pose_dim=pose_dim)
        case CITLModelConfig():
            return Inpainter(
                pose_dim=pose_dim,
                embed_size=model_config.embed_size,
                max_length=model_config.max_length,
                heads=model_config.heads,
                key_layers=model_config.key_layers,
                interm_layers=model_config.interm_layers,
                dec_layers=model_config.dec_layers,
                dropout=model_config.dropout,
                last_layer_mode=model_config.last_layer_mode,
            )
        case DeltaInterpolatorModelConfig():
            return DeltaInterpolatorInfillTransformer(
                pose_dim=pose_dim,
                num_blocks_enc=model_config.num_blocks_enc,
                num_layers_enc=model_config.num_layers_enc,
                layer_width_enc=model_config.layer_width_enc,
                num_blocks_dec=model_config.num_blocks_dec,
                num_layers_dec=model_config.num_layers_dec,
                layer_width_dec=model_config.layer_width_dec,
                dropout=model_config.dropout,
                embedding_dim=model_config.embedding_dim,
                embedding_size=model_config.embedding_size,
                embedding_num=model_config.embedding_num,
                num_heads=model_config.num_heads,
                delta_mode=model_config.delta_mode,
                input_delta_mode=model_config.input_delta_mode,
                last_layer_mode=model_config.last_layer_mode,
            )
        case _:
            raise NotImplementedError(f"Model {model_config} not implemented")
