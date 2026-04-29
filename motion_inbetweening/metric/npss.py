import torch
import torch.fft
from torchmetrics import Metric


class NPSS(Metric):
    def __init__(self, eps=1e-09):
        super().__init__()
        self.add_state("gt_power", default=[], dist_reduce_fx="cat")
        self.add_state("pred_power", default=[], dist_reduce_fx="cat")
        self.eps = eps

    def update(self, preds: torch.Tensor, target: torch.Tensor, sequence_lengths: torch.Tensor):
        """
        preds, target: (B, T, D)
        sequence_lengths: (B,) with actual length of each sequence (unpadded)
        """
        preds = preds.float()
        target = target.float()
        preds = preds.reshape(preds.shape[0], preds.shape[1], -1)
        target = target.reshape(target.shape[0], target.shape[1], -1)
        B = preds.shape[0]
        for b in range(B):
            L = sequence_lengths[b].item()
            if L <= 1:
                continue
            pred_seq = preds[b, :L, :]
            target_seq = target[b, :L, :]
            pred_power = torch.abs(torch.fft.fft(pred_seq, dim=0)) ** 2
            gt_power = torch.abs(torch.fft.fft(target_seq, dim=0)) ** 2
            self.pred_power.append(pred_power.T)
            self.gt_power.append(gt_power.T)

    def compute(self):
        npss_scores = []
        for pred_power, gt_power in zip(self.pred_power, self.gt_power, strict=False):
            gt_total_power = torch.sum(gt_power, dim=1)
            pred_total_power = torch.sum(pred_power, dim=1)
            gt_norm = gt_power / (gt_total_power.unsqueeze(1) + self.eps)
            pred_norm = pred_power / (pred_total_power.unsqueeze(1) + self.eps)
            cdf_gt = torch.cumsum(gt_norm, dim=1)
            cdf_pred = torch.cumsum(pred_norm, dim=1)
            emd = torch.sum(torch.abs(cdf_gt - cdf_pred), dim=1)
            weighted_emd = torch.sum(emd * gt_total_power) / (torch.sum(gt_total_power) + self.eps)
            npss_scores.append(weighted_emd)
        return torch.mean(torch.stack(npss_scores))
