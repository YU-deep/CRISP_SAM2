import torch
import torch.nn as nn
import torch.nn.functional as F


class CrispSamLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.1, gamma=0.1, omega1=20 / 21, omega2=1 / 21):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha, requires_grad=True))
        self.beta = nn.Parameter(torch.tensor(beta, requires_grad=True))
        self.gamma = nn.Parameter(torch.tensor(gamma, requires_grad=True))
        self.omega1 = omega1
        self.omega2 = omega2

    def focal_loss(self, inputs, targets, gamma=2):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = (1 - pt) ** gamma * BCE_loss
        return F_loss.mean()

    def dice_loss(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection) / (inputs.sum() + targets.sum())
        return 1 - dice

    def forward(self, masks_original_pred, masks_refined_pred, masks_gt, iou_pred=None, iou_gt=None, obj_pred=None, obj_gt=None):

        L_focal_original = self.focal_loss(masks_original_pred, masks_gt)
        L_dice_original = self.dice_loss(masks_original_pred, masks_gt)
        L_seg_original = self.omega1 * L_focal_original + self.omega2 * L_dice_original

        L_focal_refined = self.focal_loss(masks_refined_pred, masks_gt)
        L_dice_refined = self.dice_loss(masks_refined_pred, masks_gt)
        L_seg_refined = self.omega1 * L_focal_refined + self.omega2 * L_dice_refined

        if iou_pred is not None:
            if iou_gt is None:
                iou_gt = torch.ones_like(iou_pred)
            L_mae = F.l1_loss(iou_pred, iou_gt)
        else:
            L_mae = torch.tensor(0.0, requires_grad=False)

        if obj_pred is not None:
            if obj_gt is None:
                obj_gt = torch.ones(obj_pred.size(0), dtype=torch.long, device=obj_pred.device)
            L_ce = F.cross_entropy(obj_pred, obj_gt)
        else:
            L_ce = torch.tensor(0.0, requires_grad=False)

        total_loss = self.alpha * L_seg_original + (
                    1 - self.alpha) * L_seg_refined + self.beta * L_mae + self.gamma * L_ce
        return L_seg_original, L_seg_refined, total_loss

