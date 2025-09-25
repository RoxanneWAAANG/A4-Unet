import torch
import torch.nn.functional as F


def dice_coeff(pred: torch.Tensor, target: torch.Tensor, reduce_batch_first: bool = True, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Dice coefficient for binary masks. Expects pred/target of shape [B, H, W] or [B, 1, H, W] (floats 0/1).
    """
    if pred.ndim == 4 and pred.size(1) == 1:
        pred = pred[:, 0]
    if target.ndim == 4 and target.size(1) == 1:
        target = target[:, 0]

    intersect = torch.sum(pred * target, dim=[-1, -2])
    union = torch.sum(pred, dim=[-1, -2]) + torch.sum(target, dim=[-1, -2])
    dice = (2.0 * intersect + epsilon) / (union + epsilon)
    if reduce_batch_first:
        return dice.mean()
    return dice


def multiclass_dice_coeff(pred_onehot: torch.Tensor, target_onehot: torch.Tensor, reduce_batch_first: bool = True, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Dice for multi-class masks provided as one-hot: shape [B, C, H, W].
    """
    assert pred_onehot.shape == target_onehot.shape
    intersect = torch.sum(pred_onehot * target_onehot, dim=[-1, -2])
    union = torch.sum(pred_onehot, dim=[-1, -2]) + torch.sum(target_onehot, dim=[-1, -2])
    dice_per_class = (2.0 * intersect + epsilon) / (union + epsilon)
    # average over classes, then optionally over batch
    dice_per_sample = dice_per_class.mean(dim=1)
    if reduce_batch_first:
        return dice_per_sample.mean()
    return dice_per_sample


def dice_loss(probs: torch.Tensor, targets: torch.Tensor, multiclass: bool = False, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Dice loss that works for binary (probs: [B,H,W]) and multi-class (probs: [B,C,H,W]) cases.
    Targets should be [B,H,W] (long) for multi-class or float mask for binary.
    """
    if not multiclass:
        if probs.ndim == 4 and probs.size(1) == 1:
            probs = probs[:, 0]
        if targets.ndim == 4 and targets.size(1) == 1:
            targets = targets[:, 0]
        probs_bin = (probs > 0.5).float()
        return 1.0 - dice_coeff(probs_bin, targets.float(), reduce_batch_first=True, epsilon=epsilon)
    else:
        # probs expected as softmax outputs [B,C,H,W]
        one_hot = F.one_hot(targets, probs.size(1)).permute(0, 3, 1, 2).float()
        # hard prediction to compute dice (to match prior pipeline)
        pred_onehot = F.one_hot(probs.argmax(dim=1), probs.size(1)).permute(0, 3, 1, 2).float()
        return 1.0 - multiclass_dice_coeff(pred_onehot, one_hot, reduce_batch_first=True, epsilon=epsilon)


