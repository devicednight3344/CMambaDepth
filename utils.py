# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import os
import hashlib
import zipfile
from six.moves import urllib
import torch
import torch.nn as nn
import numpy as np


class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k = np.stack((edge_kx, edge_ky))

        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)
        
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.edge_conv(x) 
        out = out.contiguous().view(-1, 2, x.size(2), x.size(3))
  
        return out


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d


def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)


def download_model_if_doesnt_exist(model_name):
    """If pretrained kitti model doesn't exist, download and unzip it
    """
    # values are tuples of (<google cloud URL>, <md5 checksum>)
    download_paths = {
        }

    if not os.path.exists("models"):
        os.makedirs("models")

    model_path = os.path.join("models", model_name)

    def check_file_matches_md5(checksum, fpath):
        if not os.path.exists(fpath):
            return False
        with open(fpath, 'rb') as f:
            current_md5checksum = hashlib.md5(f.read()).hexdigest()
        return current_md5checksum == checksum

    # see if we have the model already downloaded...
    if not os.path.exists(os.path.join(model_path, "encoder.pth")):

        model_url, required_md5checksum = download_paths[model_name]

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("-> Downloading pretrained model to {}".format(model_path + ".zip"))
            urllib.request.urlretrieve(model_url, model_path + ".zip")

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("   Failed to download a file which matches the checksum - quitting")
            quit()

        print("   Unzipping model...")
        with zipfile.ZipFile(model_path + ".zip", 'r') as f:
            f.extractall(model_path)

        print("   Model unzipped to {}".format(model_path))

def create_recons_loss_guide_mask(batch_size, indices, edge_region, disp):
    # indices尺寸为torch.Size([12, 17554, 2])
    # edge_region尺寸为torch.Size([12, 1, 192, 640])
    # disp寸为torch.Size([12, 1, 192, 640])
    device = torch.device("cuda")
    edge_region = edge_region.squeeze(1)
    disp = disp.squeeze(1) # disp是没有梯度的
    middle_disp = edge_region * disp
    # middle_disp的尺寸为12, 192, 640
    center_x = indices[:, :, 0]
    center_y = indices[:, :, 1]
    mask = torch.where(center_x == 1, torch.tensor(0, device=device), torch.tensor(1, device=device))
    # center_x和center_y和mask的尺寸为12x17554
    center_x = center_x * mask
    center_y = center_y * mask
    batch_indices_1 = torch.arange(batch_size)[:, None]
    center_value = disp[batch_indices_1, center_x, center_y]
    center_value = center_value.unsqueeze(2)
    # center_value的尺寸为torch.Size([12, 17554, 1])
    offsets = torch.tensor([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1],
                            [1, -1], [1, 0], [1, 1]], device=torch.device("cuda"))
    batch_indices_2 = torch.arange(batch_size)[:, None, None]
    # 扩展索引矩阵以便进行偏移运算
    expanded_index_matrix = indices[:, :, None, :] + offsets
    # expanded_index_matrix的尺寸为torch.Size([12, 17554, 8, 2])
    # 提取 x 和 y 坐标
    x_indices = expanded_index_matrix[..., 0]
    y_indices = expanded_index_matrix[..., 1]
    # x_indices的尺寸为torch.Size([12, 17554, 8])
    # 使用高级索引从 matrix 中提取值
    windows_value = middle_disp[batch_indices_2, x_indices, y_indices]
    # windows_value的尺寸为torch.Size([12, 17554, 8])
    # comparison = windows_value < center_value
    comparison = windows_value >= center_value
    # comparison的尺寸为torch.Size([12, 17554, 8])
    expanded_index_matrix_reshaped = expanded_index_matrix.reshape(batch_size, -1, 2)
    # expanded_index_matrix_reshaped的尺寸为torch.Size([12, 17554 * 8, 2])
    comparison_reshaped = comparison.reshape(batch_size, -1)
    # comparison_reshaped的尺寸为torch.Size([12, 17554 * 8])
    comparison_reshaped_01value = comparison_reshaped.to(torch.int64)
    comparison_reshaped_01value = comparison_reshaped_01value.unsqueeze(2)
    # comparison_reshaped_01value的尺寸为torch.Size([12, 17554 * 8, 1])
    selected_coords = expanded_index_matrix_reshaped * comparison_reshaped_01value
    # selected_coords的尺寸为torch.Size([12, 17554 * 8, 2])
    x_positions = selected_coords[..., 0]
    y_positions = selected_coords[..., 1]
    # x_positions的尺寸为torch.Size([12, 17554 * 8])
    # 创建 edge_region 的副本，并修改副本
    edge_region_copy = edge_region.clone()
    edge_region_copy[batch_indices_1, x_positions, y_positions] = 0
    # edge_region的尺寸为torch.Size([12, 192, 640])
    # np.all和torch.all只有全是True才会置为1
    return edge_region_copy.unsqueeze(1)


def create_recons_loss_guide_mask_2(batch_size, indices, edge_region, disp):
    # indices尺寸为torch.Size([12, 17554, 2])
    # edge_region尺寸为torch.Size([12, 1, 192, 640])
    # disp寸为torch.Size([12, 1, 192, 640])
    device = torch.device("cuda")
    edge_region = edge_region.squeeze(1)
    disp = disp.squeeze(1) # disp是没有梯度的
    middle_disp = edge_region * disp
    # middle_disp的尺寸为12, 192, 640
    center_x = indices[:, :, 0]
    center_y = indices[:, :, 1]
    mask = torch.where(center_x == 1, torch.tensor(0, device=device), torch.tensor(1, device=device))
    # center_x和center_y和mask的尺寸为12x17554
    center_x = center_x * mask
    center_y = center_y * mask
    batch_indices_1 = torch.arange(batch_size)[:, None]
    center_value = disp[batch_indices_1, center_x, center_y]
    center_value = center_value.unsqueeze(2)
    # center_value的尺寸为torch.Size([12, 17554, 1])
    offsets = torch.tensor([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1],
                            [1, -1], [1, 0], [1, 1]], device=torch.device("cuda"))
    batch_indices_2 = torch.arange(batch_size)[:, None, None]
    # 扩展索引矩阵以便进行偏移运算
    expanded_index_matrix = indices[:, :, None, :] + offsets
    # expanded_index_matrix的尺寸为torch.Size([12, 17554, 8, 2])
    # 提取 x 和 y 坐标
    x_indices = expanded_index_matrix[..., 0]
    y_indices = expanded_index_matrix[..., 1]
    # x_indices的尺寸为torch.Size([12, 17554, 8])
    # 使用高级索引从 matrix 中提取值
    windows_value = middle_disp[batch_indices_2, x_indices, y_indices]
    # windows_value的尺寸为torch.Size([12, 17554, 8])
    # comparison = windows_value < center_value
    comparison = windows_value <= center_value
    # comparison的尺寸为torch.Size([12, 17554, 8])
    expanded_index_matrix_reshaped = expanded_index_matrix.reshape(batch_size, -1, 2)
    # expanded_index_matrix_reshaped的尺寸为torch.Size([12, 17554 * 8, 2])
    comparison_reshaped = comparison.reshape(batch_size, -1)
    # comparison_reshaped的尺寸为torch.Size([12, 17554 * 8])
    comparison_reshaped_01value = comparison_reshaped.to(torch.int64)
    comparison_reshaped_01value = comparison_reshaped_01value.unsqueeze(2)
    # comparison_reshaped_01value的尺寸为torch.Size([12, 17554 * 8, 1])
    selected_coords = expanded_index_matrix_reshaped * comparison_reshaped_01value
    # selected_coords的尺寸为torch.Size([12, 17554 * 8, 2])
    x_positions = selected_coords[..., 0]
    y_positions = selected_coords[..., 1]
    # x_positions的尺寸为torch.Size([12, 17554 * 8])
    # 创建 edge_region 的副本，并修改副本
    edge_region_copy = edge_region.clone()
    edge_region_copy[batch_indices_1, x_positions, y_positions] = 0
    # edge_region的尺寸为torch.Size([12, 192, 640])
    # np.all和torch.all只有全是True才会置为1
    return edge_region_copy.unsqueeze(1)