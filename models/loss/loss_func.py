import torch.nn as nn


class JointMSELoss(nn.Module):
    def __init__(self, use_target_weight=False, use_target_average=False, weight=1.0):
        super(JointMSELoss, self).__init__()
        self.use_target_weight = use_target_weight
        self.use_target_average = use_target_average
        self.weight = weight

        if self.use_target_average:
            self.criterion = nn.MSELoss(reduction='none')
        else:
            self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, predictions, targets):
        pred_hms = predictions['heatmaps']
        tgt_hms = targets['heatmaps']
        tgt_weight = targets['heatmap_weights']

        batch_size = pred_hms.size(0)
        num_joints = pred_hms.size(1)

        pred_hms = pred_hms.reshape((batch_size, num_joints, -1)).split(1, 1)
        tgt_hms = tgt_hms.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            pred_hms_i = pred_hms[idx].squeeze()
            tgt_hms_i = tgt_hms[idx].squeeze()
            if self.use_target_weight:
                loss += self.criterion(
                    pred_hms_i.mul(tgt_weight[:, idx]),
                    tgt_hms_i.mul(tgt_weight[:, idx])
                )
            else:
                loss += self.criterion(pred_hms_i, tgt_hms_i)

        if self.use_target_average:
            divisor = tgt_weight.sum()
            if divisor > 0:
                loss = loss.mean(dim=-1).sum() / divisor
            else:
                loss = loss.mean(dim=-1).sum() * 0
        else:
            loss = loss / num_joints

        return loss * self.weight
