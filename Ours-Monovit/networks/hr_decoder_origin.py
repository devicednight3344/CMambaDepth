from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from .hr_layers import *


class DepthDecoder(nn.Module):
    def __init__(self, ch_enc = [64,128,216,288,288], scales=range(4),num_ch_enc = [ 64, 64, 128, 256, 512 ], num_output_channels=1, use_channel_mamba=False):
        super(DepthDecoder, self).__init__()
        self.num_output_channels = num_output_channels
        self.num_ch_enc = num_ch_enc
        self.ch_enc = ch_enc
        self.scales = scales
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.convs = nn.ModuleDict()
        self.use_channel_mamba = use_channel_mamba

        # decoder
        self.convs = nn.ModuleDict()
        
        # feature fusion
        self.convs["f4"] = Attention_Module(self.ch_enc[4]  , num_ch_enc[4])
        self.convs["f3"] = Attention_Module(self.ch_enc[3]  , num_ch_enc[3])
        self.convs["f2"] = Attention_Module(self.ch_enc[2]  , num_ch_enc[2])
        self.convs["f1"] = Attention_Module(self.ch_enc[1]  , num_ch_enc[1])
        


        self.all_position = ["01", "11", "21", "31", "02", "12", "22", "03", "13", "04"]
        self.attention_position = ["31", "22", "13", "04"]
        self.non_attention_position = ["01", "11", "21", "02", "12", "03"]
        if self.use_channel_mamba:
            # channel_mamba_sqe = [4, 6, 8, 12]
            channel_mamba_d_model = [64, 32, 16, 16]
            num = 0
        for j in range(5):
            for i in range(5 - j):
                # upconv 0
                num_ch_in = num_ch_enc[i]
                if i == 0 and j != 0:
                    num_ch_in /= 2
                num_ch_out = num_ch_in / 2
                self.convs["X_{}{}_Conv_0".format(i, j)] = ConvBlock(num_ch_in, num_ch_out)

                # X_04 upconv 1, only add X_04 convolution
                if i == 0 and j == 4:
                    num_ch_in = num_ch_out
                    num_ch_out = self.num_ch_dec[i]
                    self.convs["X_{}{}_Conv_1".format(i, j)] = ConvBlock(num_ch_in, num_ch_out)
        # declare fSEModule and original module
        for index in self.attention_position:
            row = int(index[0])
            col = int(index[1])
            if self.use_channel_mamba:
                # self.convs["X_" + index + "_channel_mamba"] = MambaModule(num_ch_enc[row + 1] // 2, self.num_ch_enc[row]
                #                                                     + self.num_ch_dec[row + 1] * (col - 1), channel_mamba_sqe[num])
                self.convs["X_" + index + "_channel_mamba"] = MambaModule(num_ch_enc[row + 1] // 2, self.num_ch_enc[row]
                                                                          + self.num_ch_dec[row + 1] * (col - 1),
                                                                          channel_mamba_d_model[num], index)
                num = num + 1
            else:
                self.convs["X_" + index + "_attention"] = fSEModule(num_ch_enc[row + 1] // 2, self.num_ch_enc[row]
                                                                         + self.num_ch_dec[row + 1] * (col - 1))
        for index in self.non_attention_position:
            row = int(index[0])
            col = int(index[1])
            if col == 1:
                self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)] = ConvBlock(num_ch_enc[row + 1] // 2 +
                                                                        self.num_ch_enc[row], self.num_ch_dec[row + 1])
            else:
                self.convs["X_"+index+"_downsample"] = Conv1x1(num_ch_enc[row+1] // 2 + self.num_ch_enc[row]
                                                                        + self.num_ch_dec[row+1]*(col-1), self.num_ch_dec[row + 1] * 2)
                self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)] = ConvBlock(self.num_ch_dec[row + 1] * 2, self.num_ch_dec[row + 1])

        for i in range(4):
            self.convs["dispconv{}".format(i)] = Conv3x3(self.num_ch_dec[i], self.num_output_channels)
                

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def nestConv(self, conv, high_feature, low_features):
        conv_0 = conv[0]
        conv_1 = conv[1]
        assert isinstance(low_features, list)
        high_features = [upsample(conv_0(high_feature))]
        for feature in low_features:
            high_features.append(feature)
        high_features = torch.cat(high_features, 1)
        if len(conv) == 3:
            high_features = conv[2](high_features)
        return conv_1(high_features)

    def forward(self, input_features):
        outputs = {}
        feat={}
        feat[4] = self.convs["f4"](input_features[4])
        feat[3] = self.convs["f3"](input_features[3])
        feat[2] = self.convs["f2"](input_features[2])
        feat[1] = self.convs["f1"](input_features[1])
        feat[0] = input_features[0]
        features = {}
        for i in range(5):
            features["X_{}0".format(i)] = feat[i]
        # Network architecture
        for index in self.all_position:
            row = int(index[0])
            col = int(index[1])

            low_features = []
            for i in range(col):
                low_features.append(features["X_{}{}".format(row, i)])

            # add fSE block to decoder
            if index in self.attention_position:
                if self.use_channel_mamba:
                    features["X_" + index] = self.convs["X_" + index + "_channel_mamba"](
                        self.convs["X_{}{}_Conv_0".format(row + 1, col - 1)](
                            features["X_{}{}".format(row + 1, col - 1)]), low_features)
                else:
                    features["X_"+index] = self.convs["X_" + index + "_attention"](
                    self.convs["X_{}{}_Conv_0".format(row+1, col-1)](features["X_{}{}".format(row+1, col-1)]), low_features)
            elif index in self.non_attention_position:
                conv = [self.convs["X_{}{}_Conv_0".format(row + 1, col - 1)],
                        self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)]]
                if col != 1:
                    conv.append(self.convs["X_" + index + "_downsample"])
                features["X_" + index] = self.nestConv(conv, features["X_{}{}".format(row+1, col-1)], low_features)
                if index in ["01", "02", "03"] and len(input_features) == 6:
                    features["X_" + index] = features["X_" + index] + input_features[5]

        x = features["X_04"]
        x = self.convs["X_04_Conv_0"](x)
        x = self.convs["X_04_Conv_1"](upsample(x))
        outputs[("disp", 0)] = self.sigmoid(self.convs["dispconv0"](x))
        outputs[("disp", 1)] = self.sigmoid(self.convs["dispconv1"](features["X_04"]))
        outputs[("disp", 2)] = self.sigmoid(self.convs["dispconv2"](features["X_13"]))
        outputs[("disp", 3)] = self.sigmoid(self.convs["dispconv3"](features["X_22"]))
        # features["X_04"]的尺寸为torch.Size([1, 32, 96, 320])
        # features["X_03"]的尺寸为torch.Size([1, 32, 96, 320])
        # features["X_02"]的尺寸为torch.Size([1, 32, 96, 320])
        # features["X_01"]的尺寸为torch.Size([1, 32, 96, 320])
        # features["X_00"]的尺寸为torch.Size([1, 64, 96, 320])
        # torch.save(features["X_04"], "img/X_04.pt")
        # torch.save(features["X_03"], "img/X_03.pt")
        # torch.save(features["X_02"], "img/X_02.pt")
        # torch.save(features["X_01"], "img/X_01.pt")
        # torch.save(features["X_00"], "img/X_00.pt")
        # torch.save(features["X_13"], "img/X_13.pt")
        # torch.save(features["X_22"], "img/X_22.pt")
        # torch.save(features["X_31"], "img/X_31.pt")
        # torch.save(features["X_10"], "img/X_10.pt")
        # torch.save(features["X_20"], "img/X_20.pt")
        # torch.save(features["X_30"], "img/X_30.pt")
        # torch.save(features["X_40"], "img/X_40.pt")
        # torch.save(features["X_12"], "img/X_12.pt")
        # torch.save(features["X_21"], "img/X_21.pt")
        # torch.save(features["X_11"], "img/X_11.pt")
        return outputs
        
