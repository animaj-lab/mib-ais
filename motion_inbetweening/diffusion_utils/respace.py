from copy import deepcopy

import numpy as np
import torch as th

from motion_inbetweening.config.gaussian_diffusion import (
    GaussianDiffusionConfig,
    LossType,
    ModelMeanType,
    ModelVarType,
    SpacedDiffusionConfig,
)
from motion_inbetweening.config.losses import LossesConfig

from .gaussian_diffusion import GaussianDiffusion, get_named_beta_schedule


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(f"cannot create exactly {num_timesteps} steps with an integer stride")
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(f"cannot divide section of {size} steps into {section_count}")
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.

    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, conf: SpacedDiffusionConfig, loss_conf: LossesConfig):
        steps = conf.steps
        if conf.use_ddim:
            timestep_respacing = "ddim100"
        else:
            timestep_respacing = ""
        if not timestep_respacing:
            timestep_respacing = [steps]
        betas = get_named_beta_schedule(conf.noise_schedule, steps)
        loss_type = LossType.MSE if not conf.learn_sigma else LossType.RESCALED_MSE
        model_mean_type = ModelMeanType.EPSILON if not conf.predict_xstart else ModelMeanType.START_X
        model_var_type = (
            (ModelVarType.FIXED_LARGE if not conf.sigma_small else ModelVarType.FIXED_SMALL)
            if not conf.learn_sigma
            else ModelVarType.LEARNED_RANGE
        )
        from motion_inbetweening.losses.factory import loss_factory

        pose_loss_function = loss_factory(class_name=loss_conf.pose_loss)
        speed_loss_function = loss_factory(class_name=loss_conf.speed_loss)
        gaussian_diffusion_config = GaussianDiffusionConfig(
            betas=betas,
            model_mean_type=model_mean_type,
            model_var_type=model_var_type,
            loss_type=loss_type,
            pose_loss_function=pose_loss_function,
            speed_loss_function=speed_loss_function,
            rescale_timesteps=conf.rescale_timesteps,
            lambda_pose=loss_conf.weights.pose,
            lambda_speed=loss_conf.weights.speed,
            use_random_proj=conf.use_random_proj,
            fp16=conf.fp16,
            apply_zero_mask=conf.apply_zero_mask,
            time_weighted_loss=conf.time_weighted_loss,
            train_x0_as_eps=conf.train_x0_as_eps,
            train_keypoint_mask=conf.train_keypoint_mask,
        )
        self.use_timesteps = set(space_timesteps(steps, timestep_respacing))
        self.timestep_map = []
        self.original_num_steps = len(gaussian_diffusion_config.betas)
        base_diffusion = GaussianDiffusion(gaussian_diffusion_config)
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        new_conf = deepcopy(gaussian_diffusion_config)
        new_conf.betas = np.array(new_betas)
        super().__init__(new_conf)

    def p_mean_variance(self, model, *args, **kwargs):
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(self, model, *args, **kwargs):
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def condition_mean(self, cond_fn, *args, **kwargs):
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(model, self.timestep_map, self.rescale_timesteps, self.original_num_steps)

    def _scale_timesteps(self, t):
        return t


class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, timestep, **kwargs):
        map_tensor = th.tensor(self.timestep_map, device=timestep.device, dtype=timestep.dtype)
        new_timstep = map_tensor[timestep]
        if self.rescale_timesteps:
            new_timstep = new_timstep.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_timstep, **kwargs)
