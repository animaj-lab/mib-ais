import torch

from motion_inbetweening.config.losses import BCEWithLogitsLoss, L1Loss, MSELoss, TorchLossFunction


def loss_factory(class_name: TorchLossFunction) -> torch.nn.Module:
    match class_name:
        case L1Loss(reduction=reduction):
            return torch.nn.L1Loss(reduction=reduction)
        case MSELoss(reduction=reduction):
            return torch.nn.MSELoss(reduction=reduction)
        case BCEWithLogitsLoss(reduction=reduction, pos_weight=pos_weight):
            return torch.nn.BCEWithLogitsLoss(reduction=reduction, pos_weight=torch.tensor(pos_weight))
        case _:
            raise ValueError(f"Unknown loss function {class_name}")
