# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from collections import OrderedDict
from .hr_layers import upsample, ConvBlock, ChannelAttention
from einops import rearrange
class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1, use_pose_corre=False):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for
        self.use_pose_corre = use_pose_corre
        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)
        if use_pose_corre:
            for idx in range(3, -1, -1):
                if idx == 3 or idx == 2:
                    self.convs[("upconv", idx)] = ConvBlock(256, 256)
                    self.convs[("channel_attention", idx)] = ChannelAttention(self.num_ch_enc[idx] + 256)
                    if idx == 2:
                        self.convs[("pose_corre", idx)] = nn.Conv2d(self.num_ch_enc[idx] + 256, 64, 3, padding=1)
                    else:
                        self.convs[("pose_corre", idx)] = nn.Conv2d(self.num_ch_enc[idx] + 256, 256, 3, padding=1)
                else:
                    self.convs[("upconv", idx)] = ConvBlock(64, 64)
                    if idx == 0:
                        self.convs[("pose_corre", idx)] = nn.Conv2d(64, 64, 3, padding=1)
                    else:
                        self.convs[("channel_attention", idx)] = ChannelAttention(self.num_ch_enc[idx] + 64)
                        self.convs[("pose_corre", idx)] = nn.Conv2d(self.num_ch_enc[idx] + 64, 64, 3,
                                                                             padding=1)
            self.convs[("upconv", 4, 0)] = ConvBlock(64, 32)
            self.convs[("upconv", 4, 1)] = nn.Conv2d(32, 3, 3, padding=1)
        self.relu = nn.ReLU()

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)
            if self.use_pose_corre and i == 1:
                # 此时out的尺寸为12x256x6x20
                middle_out = out.detach()
                for idx in range(3, -1, -1):
                    if idx == 0:
                        middle_out = upsample(self.convs[("upconv", idx)](middle_out))
                    else:
                        middle_out = torch.cat([upsample(self.convs[("upconv", idx)](middle_out)), input_features[0][idx]],1)
                        middle_out = self.convs[("channel_attention", idx)](middle_out)
                    middle_out = self.convs[("pose_corre", idx)](middle_out)
                    middle_out = self.relu(middle_out)
                middle_out = self.convs[("upconv", 4, 0)](middle_out)
                middle_out = upsample(middle_out)
                middle_out = self.convs[("upconv", 4, 1)](middle_out)

        out = out.mean(3).mean(2)
        # out的尺寸为12x12
        if self.use_pose_corre:
            # middle_out的尺寸为12x3x192x640
            pose_corre_out = middle_out - out[:,3:6].unsqueeze(2).unsqueeze(3).detach()
            # pose_corre_out的尺寸为12x3x192x640
            b, _, h, w = middle_out.size()
            pose_corre_translation = 0.01 * pose_corre_out
            # pose_corre_axisangle和pose_corre_translation的尺寸为12x3x192x640
            b, c, h, w = pose_corre_translation.size()
            pose_corre_translation = rearrange(pose_corre_translation, 'b c h w -> b h w c', b=b, c=c, h=h, w=w)
            # pose_corre_axisangle和pose_corre_translation的尺寸为12x2x192x640x3
        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]
        if self.use_pose_corre:
            return axisangle, translation, pose_corre_translation
        else:
            return axisangle, translation