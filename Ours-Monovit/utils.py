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
        "mono_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip",
             "a964b8356e08a02d009609d9e3928f7c"),
        "stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_640x192.zip",
             "3dfb76bcff0786e4ec07ac00f658dd07"),
        "mono+stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192.zip",
             "c024d69012485ed05d7eaa9617a96b81"),
        "mono_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_no_pt_640x192.zip",
             "9c2f071e35027c895a4728358ffc913a"),
        "stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_no_pt_640x192.zip",
             "41ec2de112905f85541ac33a854742d1"),
        "mono+stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_no_pt_640x192.zip",
             "46c3b824f541d143a45c37df65fbab0a"),
        "mono_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_1024x320.zip",
             "0ab0766efdfeea89a0d9ea8ba90e1e63"),
        "stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_1024x320.zip",
             "afc2f2126d70cf3fdf26b550898b501a"),
        "mono+stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_1024x320.zip",
             "cdc5fc9b23513c07d5b19235d9ef08f7"),
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
    use_feature_guide_loss = True
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
    if use_feature_guide_loss:
        comparison_2 = windows_value < center_value
    # comparison的尺寸为torch.Size([12, 17554, 8])
    expanded_index_matrix_reshaped = expanded_index_matrix.reshape(batch_size, -1, 2)
    # expanded_index_matrix_reshaped的尺寸为torch.Size([12, 17554 * 8, 2])
    comparison_reshaped = comparison.reshape(batch_size, -1)
    if use_feature_guide_loss:
        comparison_reshaped_2 = comparison_2.reshape(batch_size, -1)
    # comparison_reshaped的尺寸为torch.Size([12, 17554 * 8])
    comparison_reshaped_01value = comparison_reshaped.to(torch.int64)
    comparison_reshaped_01value = comparison_reshaped_01value.unsqueeze(2)
    if use_feature_guide_loss:
        comparison_reshaped_01value_2 = comparison_reshaped_2.to(torch.int64)
        comparison_reshaped_01value_2 = comparison_reshaped_01value_2.unsqueeze(2)
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
