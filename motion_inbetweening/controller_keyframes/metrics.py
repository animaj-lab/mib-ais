import torch
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryPrecisionRecallCurve,
    BinaryRecall,
    BinaryROC,
)


def create_metrics_for_controller_keyframes(threshold: float) -> torch.nn.ModuleDict:
    """
    Create the metrics for the controller keyframes prediction
    Returns:
    torch.nn.ModuleDict: The dictionary containing the metrics
    """
    quantitative_metrics = torch.nn.ModuleDict(
        {
            "accuracy": BinaryAccuracy(threshold=threshold),
            "precision": BinaryPrecision(threshold=threshold),
            "recall": BinaryRecall(threshold=threshold),
            "f1": BinaryF1Score(threshold=threshold),
        }
    )
    plot_metrics = torch.nn.ModuleDict(
        {"PRCurve": BinaryPrecisionRecallCurve(thresholds=20), "ROCCurve": BinaryROC(thresholds=20)}
    )
    return (quantitative_metrics, plot_metrics)
