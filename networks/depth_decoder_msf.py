# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

#depth_decoder
from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *
from networks.HAM import HAM

class DepthDecoder_MSF(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True, use_channel_mamba=False,
                 use_channel_mamba_2=False, use_HAM=False, dataset="kitti"):
        super(DepthDecoder_MSF, self).__init__()

        self.num_output_channels = num_output_channels
        self.scales = scales

        self.num_ch_enc = num_ch_enc        #features in encoder, [64, 18, 36, 72, 144]
        self.use_channel_mamba = use_channel_mamba
        self.use_channel_mamba_2 = use_channel_mamba_2
        # decoder
        self.convs = OrderedDict()
        self.use_HAM = use_HAM
        if self.use_channel_mamba_2:
            # my_train_2025_0301
            input_channels = num_ch_enc
            hide_channel = 20
            hide_channels = [4, 4, 4, 4, 4]
            out_channel = 50
            out_channels = num_ch_enc
            channel_mamba_d_model = 4
            mamba_sqe = 5
            if dataset == "ddad":
                height = 48
            else:
                height = 24
            Anchor = 2
            mamba_channel_list = [10, 10, 10, 10, 10]
            channel_mamba_downsample_size = [2, 1, 0, 1, 2]
            mamba_num = 2
            self.convs["f_channel_mamba"] = MambaModule_2(input_channels, hide_channel, hide_channels, out_channel,
                                                          out_channels, channel_mamba_d_model, mamba_sqe,
                                                          mamba_channel_list, Anchor, height,
                                                          channel_mamba_downsample_size, mamba_num, d_conv=1,
                                                          dt_rank=2)
        self.convs[("parallel_conv"), 0, 1] = ConvBlock(self.num_ch_enc[1], self.num_ch_enc[1])
        self.convs[("parallel_conv"), 0, 2] = ConvBlock(self.num_ch_enc[2], self.num_ch_enc[2])
        self.convs[("parallel_conv"), 0, 3] = ConvBlock(self.num_ch_enc[3], self.num_ch_enc[3])
        self.convs[("parallel_conv"), 0, 4] = ConvBlock(self.num_ch_enc[4], self.num_ch_enc[4])

        self.convs[("conv1x1", 0, 2_1)] = ConvBlock1x1(self.num_ch_enc[2], self.num_ch_enc[1])
        self.convs[("conv1x1", 0, 3_2)] = ConvBlock1x1(self.num_ch_enc[3], self.num_ch_enc[2])
        self.convs[("conv1x1", 0, 3_1)] = ConvBlock1x1(self.num_ch_enc[3], self.num_ch_enc[1])
        self.convs[("conv1x1", 0, 4_3)] = ConvBlock1x1(self.num_ch_enc[4], self.num_ch_enc[3])
        self.convs[("conv1x1", 0, 4_2)] = ConvBlock1x1(self.num_ch_enc[4], self.num_ch_enc[2])
        self.convs[("conv1x1", 0, 4_1)] = ConvBlock1x1(self.num_ch_enc[4], self.num_ch_enc[1])

        self.convs[("parallel_conv"), 1, 1] = ConvBlock(self.num_ch_enc[1], self.num_ch_enc[1])
        self.convs[("parallel_conv"), 1, 2] = ConvBlock(self.num_ch_enc[2], self.num_ch_enc[2])
        self.convs[("parallel_conv"), 1, 3] = ConvBlock(self.num_ch_enc[3], self.num_ch_enc[3])
        if self.use_channel_mamba:
            self.convs[("parallel_conv_mamba"), 1, 1] = ConvBlock(self.num_ch_enc[1], self.num_ch_enc[1])
            self.convs[("parallel_conv_mamba"), 1, 2] = ConvBlock(self.num_ch_enc[2], self.num_ch_enc[2])
            self.convs[("parallel_conv_mamba"), 1, 3] = ConvBlock(self.num_ch_enc[3], self.num_ch_enc[3])
        self.convs[("conv1x1", 1, 2_1)] = ConvBlock1x1(self.num_ch_enc[2], self.num_ch_enc[1])
        self.convs[("conv1x1", 1, 3_2)] = ConvBlock1x1(self.num_ch_enc[3], self.num_ch_enc[2])
        self.convs[("conv1x1", 1, 3_1)] = ConvBlock1x1(self.num_ch_enc[3], self.num_ch_enc[1])

        self.convs[("parallel_conv"), 2, 1] = ConvBlock(self.num_ch_enc[1], self.num_ch_enc[1])
        self.convs[("parallel_conv"), 2, 2] = ConvBlock(self.num_ch_enc[2], self.num_ch_enc[2])
        self.convs[("conv1x1", 2, 2_1)] = ConvBlock1x1(self.num_ch_enc[2], self.num_ch_enc[1])
        if self.use_channel_mamba:
            self.convs[("parallel_conv_mamba"), 2, 1] = ConvBlock(self.num_ch_enc[1], self.num_ch_enc[1])
            self.convs[("parallel_conv_mamba"), 2, 2] = ConvBlock(self.num_ch_enc[2], self.num_ch_enc[2])
        self.convs[("parallel_conv"), 3, 0] = ConvBlock(self.num_ch_enc[0], self.num_ch_enc[0])
        self.convs[("parallel_conv"), 3, 1] = ConvBlock(self.num_ch_enc[1], self.num_ch_enc[1])
        if self.use_channel_mamba:
            self.convs[("parallel_conv_mamba"), 3, 1] = ConvBlock(self.num_ch_enc[1], self.num_ch_enc[1])
        self.convs[("conv1x1", 3, 1_0)] = ConvBlock1x1(self.num_ch_enc[1], self.num_ch_enc[0])

        self.convs[("parallel_conv"), 4, 0] = ConvBlock(self.num_ch_enc[0], 32)
        if self.use_channel_mamba:
            self.convs[("parallel_conv_mamba"), 4, 0] = ConvBlock(self.num_ch_enc[0], 32)
        self.convs[("parallel_conv"), 5, 0] = ConvBlock(32, 16)
        self.convs[("dispconv", 0)] = Conv3x3(16, self.num_output_channels)
        if use_HAM:
            in_channels = [18, 36, 72, 18, 36, 18, 32]  # [18, 36, 72, 18, 36, 18, 64]
            num_heads = [1, 2, 4, 1, 2, 1, 2]
            Ret_depths = [2, 3, 4, 2, 2, 2, 1]
            self.convs["AdaRM_d0_1"] = HAM(in_channels[0], num_heads[0],Ret_depths[0], index="d0_1")
            self.convs["AdaRM_d0_2"] = HAM(in_channels[1], num_heads[1], Ret_depths[1], index="d0_2")
            self.convs["AdaRM_d0_3"] = HAM(in_channels[2], num_heads[2], Ret_depths[2], index="d0_3")
            self.convs["AdaRM_d1_1"] = HAM(in_channels[3], num_heads[3], Ret_depths[3], index="d1_1")
            self.convs["AdaRM_d1_2"] = HAM(in_channels[4], num_heads[4], Ret_depths[4], index="d1_2")
            self.convs["AdaRM_d2_1"] = HAM(in_channels[5], num_heads[5], Ret_depths[5], index="d2_1")
            self.convs["AdaRM_d3_1"] = HAM(in_channels[6], num_heads[6], Ret_depths[6], index="d3_1")
        # Mamba层
        if self.use_channel_mamba:
            # my_train_2025_0301
            in_channel = [72, 108, 144, 54, 72, 36, 128]# [18, 36, 72, 18, 36, 18, 64]
            hide_channel = [36, 27, 24, 27, 18, 18, 24]
            channel_mamba_d_model = [9, 9, 8, 5, 4, 5, 8]
            channel_mamba_sqe = [4, 3, 2, 3, 2, 2, 2]
            linear_in_channel = [18, 24, 18, 12, 12, 6, 12]
            out_channels = [[9, 9, 9], [18, 18], [72], [9, 9], [36], [18], [64]]
            # channel_mamba_pacth_size = [None, None, 2, 2]
            self.expanded_ratio = [3, 2, 1, 2, 1, 1, 1]
            self.out_mamba_sqe_num = [3, 2, 1, 2, 1, 1, 1]
            d_model_origin = [18, 36, 72, 18, 36, 18, 64]
            downsample_size = [2, 1, 0, 1, 0, 1, 1]
            self.out_sqe_num = [3, 2, 1, 2, 1, 1, 1]
            mamba_sqe_list = [[1, 1, 1, 1], [1, 1, 1], [1, 1], [1, 1, 1], [1, 1], [1, 1], [1, 1]]
            mamba_channel_list = [[6, 6, 6], [12, 12], [18], [6, 6], [12], [6], [12]]
            out_channel = [18, 36, 72, 18, 36, 18, 64]
            mamba_num = [1, 1, 1, 1, 1, 1, 1]
            self.convs["mamba_d0_1_msf"] = MambaModule(in_channel[0], hide_channel[0], channel_mamba_d_model[0],
                                                 channel_mamba_sqe[0], mamba_sqe_list[0], mamba_channel_list[0],
                                                 linear_in_channel[0], out_channel[0], out_channels[0],
                                                 self.out_sqe_num[0], self.out_mamba_sqe_num[0], d_model_origin[0],
                                                 downsample_size[0], mamba_num[0], index="d0_1")
            self.convs["mamba_d0_2_msf"] = MambaModule(in_channel[1], hide_channel[1], channel_mamba_d_model[1],
                                                 channel_mamba_sqe[1], mamba_sqe_list[1], mamba_channel_list[1],
                                                 linear_in_channel[1], out_channel[1], out_channels[1],
                                                 self.out_sqe_num[1], self.out_mamba_sqe_num[1], d_model_origin[1],
                                                 downsample_size[1], mamba_num[1], index="d0_2")
            self.convs["mamba_d0_3_msf"] = MambaModule(in_channel[2], hide_channel[2], channel_mamba_d_model[2],
                                                 channel_mamba_sqe[2], mamba_sqe_list[2], mamba_channel_list[2],
                                                 linear_in_channel[2], out_channel[2], out_channels[2],
                                                 self.out_sqe_num[2], self.out_mamba_sqe_num[2], d_model_origin[2],
                                                 downsample_size[2], mamba_num[2], index="d0_3")
            self.convs["mamba_d1_1_msf"] = MambaModule(in_channel[3], hide_channel[3], channel_mamba_d_model[3],
                                                 channel_mamba_sqe[3], mamba_sqe_list[3], mamba_channel_list[3],
                                                 linear_in_channel[3], out_channel[3], out_channels[3],
                                                 self.out_sqe_num[3], self.out_mamba_sqe_num[3], d_model_origin[3],
                                                 downsample_size[3], mamba_num[3], index="d1_1")
            self.convs["mamba_d1_2_msf"] = MambaModule(in_channel[4], hide_channel[4], channel_mamba_d_model[4],
                                                 channel_mamba_sqe[4], mamba_sqe_list[4], mamba_channel_list[4],
                                                 linear_in_channel[4], out_channel[4], out_channels[4],
                                                 self.out_sqe_num[4], self.out_mamba_sqe_num[4], d_model_origin[4],
                                                 downsample_size[4], mamba_num[4], index="d1_2")
            self.convs["mamba_d2_1_msf"] = MambaModule(in_channel[5], hide_channel[5], channel_mamba_d_model[5],
                                                 channel_mamba_sqe[5], mamba_sqe_list[5], mamba_channel_list[5],
                                                 linear_in_channel[5], out_channel[5], out_channels[5],
                                                 self.out_sqe_num[5], self.out_mamba_sqe_num[5], d_model_origin[5],
                                                 downsample_size[5], mamba_num[5], index="d2_1")
            self.convs["mamba_d3_0_msf"] = MambaModule(in_channel[6], hide_channel[6], channel_mamba_d_model[6],
                                                 channel_mamba_sqe[6], mamba_sqe_list[6], mamba_channel_list[6],
                                                 linear_in_channel[6], out_channel[6], out_channels[6],
                                                 self.out_sqe_num[6], self.out_mamba_sqe_num[6], d_model_origin[6],
                                                 downsample_size[6], mamba_num[6], index="d3_0")
        # # 其他各层的dispconv
        # self.convs[("dispconv", 1)] = Conv3x3(32, self.num_output_channels)
        # self.convs[("dispconv", 2)] = Conv3x3(18, self.num_output_channels)
        # self.convs[("dispconv", 3)] = Conv3x3(36, self.num_output_channels)
        # self.convs[("dispconv", 4)] = Conv3x3(72, self.num_output_channels)
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features, use_channel_mamba, use_channel_mamba_2, use_HAM):
        self.outputs = {}
        # features in encoder
        if use_channel_mamba_2:
            output_features = self.convs["f_channel_mamba"](input_features)
            e4 = output_features[4]
            e3 = output_features[3]
            e2 = output_features[2]
            e1 = output_features[1]
            e0 = output_features[0]
        else:
            e4 = input_features[4]
            e3 = input_features[3]
            e2 = input_features[2]
            e1 = input_features[1]
            e0 = input_features[0]

        d0_1 = self.convs[("parallel_conv"), 0, 1](e1)
        d0_2 = self.convs[("parallel_conv"), 0, 2](e2)
        d0_3 = self.convs[("parallel_conv"), 0, 3](e3)
        d0_4 = self.convs[("parallel_conv"), 0, 4](e4)

        d0_2_1 = updown_sample(d0_2, 2)
        d0_3_2 = updown_sample(d0_3, 2)
        d0_3_1 = updown_sample(d0_3, 4)
        d0_4_3 = updown_sample(d0_4, 2)
        d0_4_2 = updown_sample(d0_4, 4)
        d0_4_1 = updown_sample(d0_4, 8)

        d0_2_1 = self.convs[("conv1x1", 0, 2_1)](d0_2_1)
        d0_3_2 = self.convs[("conv1x1", 0, 3_2)](d0_3_2)
        d0_3_1 = self.convs[("conv1x1", 0, 3_1)](d0_3_1)
        d0_4_3 = self.convs[("conv1x1", 0, 4_3)](d0_4_3)
        d0_4_2 = self.convs[("conv1x1", 0, 4_2)](d0_4_2)
        d0_4_1 = self.convs[("conv1x1", 0, 4_1)](d0_4_1)

        if not use_channel_mamba:
            d0_1_msf = d0_1 + d0_2_1 + d0_3_1 + d0_4_1  # 通道数为18，48x160
            d0_2_msf = d0_2 + d0_3_2 + d0_4_2  # 通道数为36
            d0_3_msf = d0_3 + d0_4_3  # 通道数为72
        else:
            # mytrain_2025_0301
            middle_features_map = torch.cat([d0_3_1, d0_2_1, d0_1], 1)  # 12x72x48x160
            d0_1_msf = self.convs["mamba_d0_1_msf"](middle_features_map, d0_4_1, [d0_3_1, d0_2_1, d0_1],
                                                    self.expanded_ratio[0],
                                                    self.out_sqe_num[0])
            middle_features_map = torch.cat([d0_3_2, d0_2], 1)  # 12x108x24x80
            d0_2_msf = self.convs["mamba_d0_2_msf"](middle_features_map, d0_4_2, [d0_3_2, d0_2], self.expanded_ratio[1],
                                                    self.out_sqe_num[1])
            d0_3_msf = self.convs["mamba_d0_3_msf"](d0_3, d0_4_3, [d0_3], self.expanded_ratio[2],
                                                        self.out_sqe_num[2])

        if not use_channel_mamba:
            d1_1 = self.convs[("parallel_conv"), 1, 1](d0_1_msf)
            d1_2 = self.convs[("parallel_conv"), 1, 2](d0_2_msf)
            d1_3 = self.convs[("parallel_conv"), 1, 3](d0_3_msf)
        else:
            d1_1 = self.convs[("parallel_conv_mamba"), 1, 1](d0_1_msf)
            d1_2 = self.convs[("parallel_conv_mamba"), 1, 2](d0_2_msf)
            d1_3 = self.convs[("parallel_conv_mamba"), 1, 3](d0_3_msf)
        # # 第 4号 disp
        # self.outputs[("disp", 4)] = self.sigmoid(self.convs[("dispconv", 4)](d1_3))
        if use_HAM:
            d1_1 = self.convs["AdaRM_d0_1"](d1_1)
            d1_2 = self.convs["AdaRM_d0_2"](d1_2)
            d1_3 = self.convs["AdaRM_d0_3"](d1_3)
        d1_2_1 = updown_sample(d1_2, 2)
        d1_3_2 = updown_sample(d1_3, 2)
        d1_3_1 = updown_sample(d1_3, 4)

        d1_2_1 = self.convs[("conv1x1", 1, 2_1)](d1_2_1)
        d1_3_2 = self.convs[("conv1x1", 1, 3_2)](d1_3_2)
        d1_3_1 = self.convs[("conv1x1", 1, 3_1)](d1_3_1)
        if not use_channel_mamba:
            d1_1_msf = d1_1 + d1_2_1 + d1_3_1  # 通道数为18
            d1_2_msf = d1_2 + d1_3_2  # 通道数为36
        else:
            # mytrain_2025_0301
            middle_features_map = torch.cat([d1_2_1, d1_1], 1)  # 12x54x48x160
            d1_1_msf = self.convs["mamba_d1_1_msf"](middle_features_map, d1_3_1, [d1_2_1, d1_1], self.expanded_ratio[3],
                                                    self.out_sqe_num[3])
            d1_2_msf = self.convs["mamba_d1_2_msf"](d1_2, d1_3_2, [d1_2], self.expanded_ratio[4],
                                                        self.out_sqe_num[4])
        if not use_channel_mamba:
            d2_1 = self.convs[("parallel_conv"), 2, 1](d1_1_msf)
            d2_2 = self.convs[("parallel_conv"), 2, 2](d1_2_msf)
        else:
            d2_1 = self.convs[("parallel_conv_mamba"), 2, 1](d1_1_msf)
            d2_2 = self.convs[("parallel_conv_mamba"), 2, 2](d1_2_msf)
        # # 第 3 号 disp
        # self.outputs[("disp", 3)] = self.sigmoid(self.convs[("dispconv", 3)](d2_2))
        # if use_HAM:
        #     d2_1 = self.convs["AdaRM_d1_1"](d2_1)
        #     d2_2 = self.convs["AdaRM_d1_2"](d2_2)
        d2_2_1 = updown_sample(d2_2, 2)

        d2_2_1 = self.convs[("conv1x1", 2, 2_1)](d2_2_1)
        if not use_channel_mamba:
            d2_1_msf = d2_1 + d2_2_1  # 通道数为18
        else:
            # mytrain_2025_0301
            d2_1_msf = self.convs["mamba_d2_1_msf"](d2_1, d2_2_1, [d2_1], self.expanded_ratio[5],
                                                  self.out_sqe_num[5])
        if not use_channel_mamba:
            d3_1 = self.convs[("parallel_conv"), 3, 1](d2_1_msf)
        else:
            d3_1 = self.convs[("parallel_conv_mamba"), 3, 1](d2_1_msf)
        # # 第 2 号 disp
        # self.outputs[("disp", 2)] = self.sigmoid(self.convs[("dispconv", 2)](d3_1))
        # if use_HAM:
        #     d3_1 = self.convs["AdaRM_d2_1"](d3_1)
        d3_0 = self.convs[("parallel_conv"), 3, 0](e0)
        d3_1_0 = updown_sample(d3_1, 2)
        d3_1_0 = self.convs[("conv1x1", 3, 1_0)](d3_1_0)

        if not use_channel_mamba:
            d3_0_msf = d3_0 + d3_1_0  # 通道数为64
        else:
            d3_0_msf = self.convs["mamba_d3_0_msf"](d3_0, d3_1_0, [d3_1_0], self.expanded_ratio[6],
                                                  self.out_sqe_num[6])

        if not use_channel_mamba:
            d4_0 = self.convs[("parallel_conv"), 4, 0](d3_0_msf)
        else:
            d4_0 = self.convs[("parallel_conv_mamba"), 4, 0](d3_0_msf)
        # # 第 1 号 disp
        # self.outputs[("disp", 1)] = self.sigmoid(self.convs[("dispconv", 1)](d4_0))
        # if use_HAM:
        #     d4_0 = self.convs["AdaRM_d3_1"](d4_0)
        d4_0 = updown_sample(d4_0, 2)
        d5 = self.convs[("parallel_conv"), 5, 0](d4_0)
        self.outputs[("disp", 0)] = self.sigmoid(self.convs[("dispconv", 0)](d5))

        return self.outputs  # single-scale depth