import torch
import torch.nn.functional as F


class FocalLoss(torch.nn.Module):
    def __init__(self, reduce=True):
        super(FocalLoss, self).__init__()
        self.beta = 0.25
        self.gamma = 2.0
        self.reduce = reduce

    def forward(self, inputs, targets, weights=None):
        mae_loss = F.l1_loss(inputs, targets, reduction='none')
        # mae_loss = F.smooth_l1_loss(inputs, targets, reduction='none')
        F_loss = torch.sigmoid(self.beta * mae_loss) ** self.gamma * mae_loss
        if weights is not None:
            F_loss *= weights
        return torch.mean(F_loss) if self.reduce else torch.sum(F_loss)
