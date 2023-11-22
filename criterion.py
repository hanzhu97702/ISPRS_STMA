import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=0,):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight=weight, ignore_index=ignore_index,
                                   reduction='mean')

    def forward(self, input, target):
        return self.nll_loss(F.log_softmax(input, dim=1), target)
