from __future__ import absolute_import, division, print_function

import torch
import networks
from thop import clever_format
from thop import profile

import time
import PIL.Image as pil
from torchvision import transforms

def profile_once(encoder, decoder, x):
    x_e = x[0, :, :, :].unsqueeze(0)
    x_d = encoder(x_e)
    flops_e, params_e = profile(encoder, inputs=(x_e.unsqueeze(0)), verbose=False)
    flops_d, params_d = profile(decoder, inputs=(x_d, True, True, True), verbose=False)

    flops, params = clever_format([flops_e + flops_d, params_e + params_d], "%.3f")
    flops_e, params_e = clever_format([flops_e, params_e], "%.3f")
    flops_d, params_d = clever_format([flops_d, params_d], "%.3f")

    return flops, params, flops_e, params_e, flops_d, params_d

device = torch.device("cuda")
encoder = networks.hrnet18(False)
depth_decoder = networks.DepthDecoder_MSF(encoder.num_ch_enc, [0], num_output_channels=1,
                                          use_channel_mamba=True,
                                          use_channel_mamba_2=True,
                                          use_HAM=True)
encoder.to(device)
depth_decoder.to(device)
input_color = torch.randn(1,3,192,640).to(device)
encoder.eval()
depth_decoder.eval()
flops, params, flops_e, params_e, flops_d, params_d = profile_once(encoder, depth_decoder, input_color)
print("\n  " + ("flops: {0}, params: {1}, flops_e: {2}, params_e:{3}, flops_d:{4}, params_d:{5}").format(flops, params, flops_e, params_e, flops_d, params_d))

image_path = "D:\题目资料\单目\img/0000000002.png"
# Load image and preprocess
input_image = pil.open(image_path).convert('RGB')
original_width, original_height = input_image.size
input_image = input_image.resize((640, 192), pil.LANCZOS)
input_image = transforms.ToTensor()(input_image).unsqueeze(0)
# PREDICTION
input_image = input_image.cuda()
##         推断时间
inference_time1 = time.time()

features = encoder(input_image)
outputs = depth_decoder(features, True, True, True)

inference_time2 = time.time()
print("推断时间：{:.4f}秒".format(inference_time2-inference_time1))
# 4090上推断时间为0.0991秒(创新点全加)，基线的4090推断时间为0.0402秒