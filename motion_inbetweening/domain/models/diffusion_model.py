from motion_inbetweening.config.mdm import MDMConfig, MDMDiTConfig, MDMUnetConfig, MotionDiffusionModelConfig
from motion_inbetweening.diffusion_utils.mdm_architectures.mdm_unet import MDM_UNET
from motion_inbetweening.domain.mask_applier import MaskApplierConfig


def load_diffusion_model(
    pose_dim: int, model_config: MotionDiffusionModelConfig, mask_applier_config: MaskApplierConfig, learn_sigma: bool
):
    if isinstance(model_config, MDMConfig):
        raise ValueError("MDM is not implemented yet")
    elif isinstance(model_config, MDMUnetConfig):
        model = MDM_UNET(
            modeltype=model_config.modeltype,
            pose_dim=pose_dim,
            latent_dim=model_config.latent_dim,
            dim_mults=model_config.dim_mults,
            attention=model_config.attention,
            ablation=model_config.ablation,
            legacy=model_config.legacy,
            emb_trans_dec=model_config.emb_trans_dec,
            adagn=model_config.adagn,
            zero=model_config.zero,
            arch=model_config.arch,
            unet_out_mult=model_config.unet_out_mult,
            keyframe_conditioned=model_config.keyframe_conditioned,
            zero_keyframe_loss=model_config.zero_keyframe_loss,
            concat_mode=mask_applier_config.concat_mode,
            learn_sigma=learn_sigma,
            final_type=model_config.final_type,
        )
    elif isinstance(model_config, MDMDiTConfig):
        raise ValueError("MDM DiT is not implemented yet")
    else:
        raise ValueError(f"Unsupported model config: {model_config}")
    return model
