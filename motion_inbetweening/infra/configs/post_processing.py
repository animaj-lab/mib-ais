from motion_inbetweening.config.post_processing import PostProcessingConfig


def get_default_post_processing_config(do_controller_keyframes_prediction: bool) -> PostProcessingConfig:
    if do_controller_keyframes_prediction:
        return PostProcessingConfig(
            do_controller_keyframes_reduction=True, threshold_controller_keyframes_reduction=0.5
        )
    else:
        return PostProcessingConfig(
            do_controller_keyframes_reduction=False, threshold_controller_keyframes_reduction=None
        )
