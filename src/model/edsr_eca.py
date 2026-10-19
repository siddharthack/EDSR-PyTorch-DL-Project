"""
EDSR-ECA: the repository's EDSR with Efficient Channel Attention (ECA) added
inside every residual block.

ECA comes from "ECA-Net: Efficient Channel Attention for Deep Convolutional
Neural Networks" (Wang et al., CVPR 2020, https://arxiv.org/abs/1910.03151).

This file is EDSR (src/model/edsr.py) with exactly ONE change: each residual
block gets an ECA layer between its second convolution and the skip addition.
Everything else is identical to the EDSR baseline: head, 16 blocks, body conv,
long skip connection, x4 sub-pixel upsampler, mean shift, 64 feature channels.
There are no residual groups (that was RCAN).

    Baseline EDSR block:   x -> conv -> ReLU -> conv ------------> * res_scale -> + x
    EDSR-ECA block:        x -> conv -> ReLU -> conv -> ECA -----> * res_scale -> + x

ECA layer (adds only k parameters per block, k=3 by default):
    feature maps (C x H x W)
      -> global average pool            -> C numbers (one summary per channel)
      -> 1D conv (kernel k) over the C numbers, then Sigmoid -> C weights in (0, 1)
      -> multiply every feature map by its weight

Unlike RCAN's CALayer there is no squeeze/expand bottleneck (C -> C/r -> C).
"""
from model import common

import torch.nn as nn


def make_model(args, parent=False):
    return EDSRECA(args)


class ECALayer(nn.Module):
    """Efficient Channel Attention.

    kernel_size is the number of neighbouring channels that influence each
    channel's weight. k=3 is what ECA-Net's adaptive rule gives for C=64
    (k = odd(|log2(C)/2 + 1/2|) = odd(3.5) = 3).
    """

    def __init__(self, kernel_size=3):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("ECA kernel_size must be odd")
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # (B, C, H, W) -> (B, C, 1, 1) -> (B, 1, C): channels become a 1D sequence
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.sigmoid(self.conv(y))            # (B, 1, C) channel weights
        y = y.transpose(-1, -2).unsqueeze(-1)     # (B, C, 1, 1)
        return x * y


class ECAResBlock(nn.Module):
    """EDSR ResBlock (common.ResBlock) with an ECA layer after the second conv."""

    def __init__(self, conv, n_feats, kernel_size, eca_kernel=3,
                 bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super().__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)
        m.append(ECALayer(eca_kernel))

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class EDSRECA(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super().__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        eca_kernel = getattr(args, "eca_kernel", 3)
        res_scale = getattr(args, "res_scale", 1)
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # head: RGB (3 channels) -> n_feats feature maps
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # body: n_resblocks ECA residual blocks + one conv
        m_body = [
            ECAResBlock(conv, n_feats, kernel_size, eca_kernel=eca_kernel,
                        act=act, res_scale=res_scale)
            for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # tail: sub-pixel upsampling + back to RGB
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size),
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x  # long skip connection

        x = self.tail(res)
        x = self.add_mean(x)
        return x


def count_parameters(model, trainable_only=True):
    """Number of (trainable) parameters; MeanShift layers are frozen and excluded."""
    return sum(p.numel() for p in model.parameters()
               if p.requires_grad or not trainable_only)
