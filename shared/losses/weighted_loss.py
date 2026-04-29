import torch
import torch.nn as nn


class AutoWeightedLossBase(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, reg_losses: list[torch.Tensor], cls_losses: list[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement this method.")


class MultiTaskLoss(AutoWeightedLossBase):
    """
    Implementation of paper https://arxiv.org/pdf/1805.06334
    from https://github.com/Mikoto10032/AutomaticWeightedLoss/
    """

    def __init__(self, n_losses: int):
        super().__init__()
        params = torch.ones((n_losses, 1))
        self.params = torch.nn.Parameter(params)
        self.epsilon = torch.tensor(1e-06, device=self.params.device)

    def forward(self, reg_losses: list[torch.Tensor], cls_losses: list[torch.Tensor]) -> torch.Tensor:
        loss_sum = torch.tensor(0.0, device=self.params.device)
        for i, loss in enumerate(reg_losses + cls_losses):
            param = self.params[i]
            param = torch.max(param, self.epsilon).squeeze()
            loss_sum += 0.5 / param**2 * loss + torch.log(1 + param**2)
        return loss_sum
