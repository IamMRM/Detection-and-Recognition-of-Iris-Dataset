import torch
from torch import nn


class ContrastiveLoss(nn.Module):

    def __init__(self, margin=1):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, out1, out2, labels):
        distance = torch.sqrt(torch.sum(torch.pow(out1 - out2, 2), dim=1, keepdim=True))
        loss = labels * torch.pow(distance, 2) + (1 - labels) * torch.pow(torch.clamp(self.margin - distance, min=0), 2)
        return torch.mean(loss) * 0.5
