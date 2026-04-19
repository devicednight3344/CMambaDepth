# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *
from .hr_layers import MambaModule

class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True, use_channel_mamba=False):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.use_channel_mamba = use_channel_mamba
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        if self.use_channel_mamba:
            input_channels = [[64,64,256],[64,64,128],[64,64,64]]
            hide_channels = [[64,64,64],[64,64,64],[32,32,32]]
            channel_mamba_sqe = [3,3,3]
            d_model = [14,14,14]
            mamba_sqe = [3,3,3]
            mamba_sqe_list = [[1,1,1],[1,1,1],[1,1,1]]
            out_channel = [256,128,64]
            downsample_size = [[3,2,0],[2,1,0],[1,0,0]]
            d_state = [16,16,16]
            mamba_num = [2,2,2]
            num = 0
        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            if self.use_channel_mamba and i != 0 and i != 1:
                self.convs[("channel_mamba_1", i)] = MambaModule(input_channels[num], hide_channels[num],
                                                                 channel_mamba_sqe[num], d_model[num], mamba_sqe[num],
                                                                 mamba_sqe_list[num], out_channel[num],
                                                                 downsample_size[num], mamba_num[num], d_state[num])
                num = num + 1
            else:
                # upconv_1
                num_ch_in = self.num_ch_dec[i]
                if self.use_skips and i > 0:
                    num_ch_in += self.num_ch_enc[i - 1]
                num_ch_out = self.num_ch_dec[i]
                self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            if self.use_channel_mamba and i != 0 and i != 1:
                featured_list = input_features[:2]
                featured_list.append(x)
                x = self.convs[("channel_mamba_1", i)](featured_list)
            else:
                x = [upsample(x)]
                if self.use_skips and i > 0:
                    x += [input_features[i - 1]]
                x = torch.cat(x, 1)
                x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs