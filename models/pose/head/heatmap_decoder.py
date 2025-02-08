import torch
import torch.nn as nn
import torch.nn.functional as F


class HeatmapDecoder(nn.Module):
    def __init__(self, embed_dim, upsample_ratio):
        super(HeatmapDecoder, self).__init__()

        self.upsample_ratio = upsample_ratio
        self.embed_dim = embed_dim

        # init parameters
        self.conv_weight = nn.Parameter(torch.ones(1, 1, 3, 3))
        self.conv_bias = nn.Parameter(torch.zeros(1))

        # init weights
        nn.init.normal_(self.conv_weight, std=0.001)
        nn.init.constant_(self.conv_bias, 0)

    def forward(self, x, embeds, fast_weights):
        """Forward function."""
        conv_weight = fast_weights.get('conv_weight', self.conv_weight)
        conv_bias = fast_weights.get('conv_bias', self.conv_bias)

        # up-sample spatial features
        B, C, H, W = x.shape
        new_H, new_W = int(H * self.upsample_ratio), int(W * self.upsample_ratio)
        x_up = F.interpolate(x,
                             size=(new_H, new_W),
                             mode='bilinear',
                             align_corners=False)

        # compute cosine-sim
        x_dot = torch.matmul(embeds, x_up.reshape(B, -1, new_H * new_W))
        x_dot = x_dot.reshape(B, -1, new_H, new_W)
        x_sim = F.layer_norm(x_dot, (new_H, new_W), eps=1e-8)

        # spatial-wise conv
        num_joints = x_sim.shape[1]
        conv_weight = conv_weight.expand(num_joints, -1, -1, -1)
        conv_bias = conv_bias.expand(num_joints)
        heatmaps = F.conv2d(x_sim, conv_weight, conv_bias, stride=1, padding=1, groups=num_joints)

        return heatmaps, x_up
