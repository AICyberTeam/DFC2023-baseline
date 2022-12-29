import torch
import torch.nn as nn


class CriterionDSN(nn.Module):
    def __init__(self, ignore_index=255, use_weight=True, reduce=True):
        super(CriterionDSN, self).__init__()
        self.ignore_index = ignore_index
        self.l1loss = torch.nn.SmoothL1Loss(reduce=reduce)
        if not reduce:
            print("disabled the reduce.")

    def forward(self, preds, dsms):
        loss = self.l1loss(preds, dsms)
        return loss

