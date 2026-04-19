from __future__ import absolute_import, division, print_function

from .hr_layers import *
from networks.HAM import HAM

class DepthDecoder(nn.Module):
    def __init__(self, ch_enc=[64, 128, 216, 288, 288], scales=range(4), num_ch_enc=[64, 64, 128, 256, 512],
                 num_output_channels=1, use_channel_mamba=False, use_channel_mamba_2=False, use_HAM=False):
        super(DepthDecoder, self).__init__()
        self.num_output_channels = num_output_channels
        self.num_ch_enc = num_ch_enc
        self.ch_enc = ch_enc
        self.scales = scales
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.use_channel_mamba = use_channel_mamba
        self.use_channel_mamba_2 = use_channel_mamba_2
        self.use_HAM = use_HAM
        # decoder
        self.convs = nn.ModuleDict()
        if use_HAM:
            in_channels = [256, 256]
            num_heads = [4, 4]
            Ret_depths = [4, 4]
            self.convs["AdaRM_d3_0"] = HAM(in_channels[0], num_heads[0], Ret_depths[0], index="d3_0")
            self.convs["AdaRM_d4_0"] = HAM(in_channels[1], num_heads[1], Ret_depths[1], index="d4_0")
            in_channels = [128, 128, 128]
            num_heads = [2, 2, 2]
            Ret_depths = [3, 3, 3]
            self.convs["AdaRM_d2_0"] = HAM(in_channels[0], num_heads[0], Ret_depths[0], index="d2_0")
            self.convs["AdaRM_d2_1"] = HAM(in_channels[1], num_heads[1], Ret_depths[1], index="d2_1")
            self.convs["AdaRM_d3_1"] = HAM(in_channels[2], num_heads[2], Ret_depths[2], index="d3_1")
        if self.use_channel_mamba_2:
            input_channels = ch_enc
            hide_channel = 20
            hide_channels = [4, 4, 4, 4, 4]
            out_channel = 100
            out_channels = num_ch_enc
            channel_mamba_d_model = 4
            mamba_sqe = 5
            height = 24
            Anchor = 2
            mamba_channel_list = [20, 20, 20, 20, 20]
            channel_mamba_downsample_size = [2, 1, 0, 1, 2]
            mamba_num = 2
            self.convs["f_channel_mamba"] = MambaModule_2(input_channels, hide_channel, hide_channels, out_channel,
                                                          out_channels, channel_mamba_d_model, mamba_sqe,
                                                          mamba_channel_list, Anchor, height,
                                                          channel_mamba_downsample_size, mamba_num, d_conv=1,
                                                          dt_rank=2)
        else:
            # feature fusion
            self.convs["f4"] = Attention_Module(self.ch_enc[4], num_ch_enc[4])
            self.convs["f3"] = Attention_Module(self.ch_enc[3], num_ch_enc[3])
            self.convs["f2"] = Attention_Module(self.ch_enc[2], num_ch_enc[2])
            self.convs["f1"] = Attention_Module(self.ch_enc[1], num_ch_enc[1])

        self.all_position = ["01", "11", "21", "31", "02", "12", "22", "03", "13", "04"]
        self.attention_position = ["31", "22", "13", "04"]
        self.non_attention_position = ["01", "11", "21", "02", "12", "03"]
        if self.use_channel_mamba:
            hide_channel = [80, 60, 96, 60]
            channel_mamba_d_model = [14, 14, 14, 9]
            channel_mamba_sqe = [4, 3, 4, 5]
            linear_in_channel = [64, 48, 48, 32]
            out_channels = [[128], [64], [32, 32], [16, 16]]
            # channel_mamba_pacth_size = [None, None, 2, 2]
            self.expanded_ratio = [1, 2, 3, 4]
            num = 0
            self.out_mamba_sqe_num = [1, 1, 2, 2]
            d_model_origin = [256,128,64,32]
            downsample_size = [0, 0, 1, 1]
            self.out_sqe_num = [2, 1, 2, 2]
            mamba_sqe_list = [[2, 2], [1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1, 1]]
            mamba_channel_list = [[64], [48], [24, 24], [16, 16]]
            self.convs["X_00_Conv_extra"] = Conv1x1(64, 32)

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
                if index == "04":
                    in_channel = num_ch_enc[row + 1] // 2 + 32 + self.num_ch_dec[row + 1] * (col - 1)
                else:
                    in_channel = num_ch_enc[row + 1] // 2 + self.num_ch_enc[row] + self.num_ch_dec[row + 1] * (col - 1)
                out_channel = num_ch_enc[row + 1] // 2
                # linear_in_channel = in_channel // channel_mamba_sqe[num] * self.out_sqe_num[col-1]
                self.convs["X_" + index + "_channel_mamba"] = MambaModule(in_channel, hide_channel[num],
                                                                          channel_mamba_d_model[num],
                                                                          channel_mamba_sqe[num], mamba_sqe_list[col-1],
                                                                          mamba_channel_list[col-1], linear_in_channel[col-1],
                                                                          out_channel, out_channels[col-1],self.out_sqe_num[col-1],
                                                                          self.out_mamba_sqe_num[col-1], d_model_origin[col-1],
                                                                          downsample_size[num])
                num = num + 1
            else:
                self.convs["X_" + index + "_attention"] = fSEModule(num_ch_enc[row + 1] // 2, self.num_ch_enc[row]
                                                                    + self.num_ch_dec[row + 1] * (col - 1))
        for index in self.non_attention_position:
            row = int(index[0])
            col = int(index[1])
            if col == 1:
                self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)] = ConvBlock(num_ch_enc[row + 1] // 2 +
                                                                                 self.num_ch_enc[row],
                                                                                 self.num_ch_dec[row + 1])
            else:
                self.convs["X_" + index + "_downsample"] = Conv1x1(num_ch_enc[row + 1] // 2 + self.num_ch_enc[row]
                                                                   + self.num_ch_dec[row + 1] * (col - 1),
                                                                   self.num_ch_dec[row + 1] * 2)
                self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)] = ConvBlock(self.num_ch_dec[row + 1] * 2,
                                                                                 self.num_ch_dec[row + 1])

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
        feat = {}
        if self.use_channel_mamba_2:
            output_features = self.convs["f_channel_mamba"](input_features)
            feat[4] = output_features[4]
            feat[3] = output_features[3]
            feat[2] = output_features[2]
            feat[1] = output_features[1]
            feat[0] = output_features[0]
        else:
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
                if self.use_channel_mamba and index == "04" and i == 0:
                    features["X_{}{}".format(row, i)] = self.convs["X_00_Conv_extra"](features["X_{}{}".format(row, i)])
                if self.use_HAM and index != "04":
                    if index in ["31", "22"]:
                        features["X_{}{}".format(row, i)] = self.convs["AdaRM_d{}_{}".format(row, i)](features["X_{}{}".format(row, i)])
                low_features.append(features["X_{}{}".format(row, i)])

            # add fSE block to decoder
            if index in self.attention_position:
                if self.use_channel_mamba:
                    high_feature = upsample(self.convs["X_{}{}_Conv_0".format(row + 1, col - 1)](
                            features["X_{}{}".format(row + 1, col - 1)]))
                    if self.use_HAM and (index == "31" or index == "22"):
                        high_feature = self.convs["AdaRM_d{}_{}".format(row + 1, col - 1)](high_feature)
                    if index == "31" or index == "22":
                        features_identity = [low_features[-1]]
                        features_map = torch.cat(low_features, 1)
                    else:
                        features_identity = [low_features[-1], high_feature]
                        features_map = torch.cat(low_features, 1)
                    # 12x384x24x80
                    features["X_" + index] = self.convs["X_" + index + "_channel_mamba"](features_map, high_feature,
                                                                                         features_identity,
                                                                                         self.expanded_ratio[col-1],
                                                                                         self.out_sqe_num[col-1],index)
                else:
                    features["X_" + index] = self.convs["X_" + index + "_attention"](
                        self.convs["X_{}{}_Conv_0".format(row + 1, col - 1)](
                            features["X_{}{}".format(row + 1, col - 1)]), low_features,index)
            elif index in self.non_attention_position:
                conv = [self.convs["X_{}{}_Conv_0".format(row + 1, col - 1)],
                        self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)]]
                if col != 1:
                    conv.append(self.convs["X_" + index + "_downsample"])
                features["X_" + index] = self.nestConv(conv, features["X_{}{}".format(row + 1, col - 1)], low_features)

        x = features["X_04"]
        x = self.convs["X_04_Conv_0"](x)
        x = self.convs["X_04_Conv_1"](upsample(x))
        outputs[("disp", 0)] = self.sigmoid(self.convs["dispconv0"](x))
        outputs[("disp", 1)] = self.sigmoid(self.convs["dispconv1"](features["X_04"]))
        outputs[("disp", 2)] = self.sigmoid(self.convs["dispconv2"](features["X_13"]))
        outputs[("disp", 3)] = self.sigmoid(self.convs["dispconv3"](features["X_22"]))
        return outputs
