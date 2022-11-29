import torch
import torch.nn as nn


class CriterionDSN(nn.Module):
    def __init__(self, ignore_index=255, use_weight=True, reduce=True):
        super(CriterionDSN, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduce=reduce)
        self.l1loss = torch.nn.SmoothL1Loss(reduce=reduce)
        if not reduce:
            print("disabled the reduce.")

    def forward(self, preds, dsms):
        huber_c = 0.2 * torch.max(preds - dsms)

        diff = dsms - preds
        valid_mask = (dsms > 0).detach()
        diff = diff[valid_mask].abs()
        huber_mask = (diff > huber_c).detach()

        diff_sq = diff[huber_mask] ** 2
        loss = torch.cat((diff, diff_sq)).mean()

        # loss = self.l1loss(preds, dsms)
        return loss

