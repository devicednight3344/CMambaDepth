from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.utils import save_image
import cv2
from einops import rearrange

def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    # axisangle和translation的尺寸为12x1x3
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

class ConvBlockDepth(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlockDepth, self).__init__()

        self.conv = DepthConv3x3(in_channels, out_channels)
        self.nonlin = nn.GELU()

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class DepthConv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(DepthConv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        # self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=3, groups=int(out_channels), bias=False)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)
        # self.conv = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=3, padding=3 // 2, groups=int(out_channels), bias=False)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

def to_optimise_max(matrix, top_factor=0.2):
    # 获取每个 batch 上最后两个维度的大小
    batch_size, _, height, width = matrix.shape

    # 展平最后两个维度并计算每个 batch 的 top 20% 的数值索引
    flattened = matrix.view(batch_size, -1)  # 变形为 (12, 192*640)

    # 计算每个 batch 上前 20% 的元素数量
    num_top = int(top_factor * flattened.size(1))

    # 使用 topk 获取前 20% 数值的索引
    top_k_values, top_k_indices = torch.topk(flattened, num_top, dim=1)
    # top_k_indices的尺寸为12xnum_top
    # 将一维的索引转换为二维坐标
    # 计算行和列的坐标
    rows = torch.div(top_k_indices, width, rounding_mode='trunc')  # 行坐标
    cols = top_k_indices % width  # 列坐标

    # 将结果存储为 (batch_size, num_top, 2) 的列表
    top_k_coords = torch.stack((rows, cols), dim=-1)
    # top_k_coords的尺寸为12x24576x2
    return top_k_coords, top_k_indices

class BackprojectDepth_corre(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth_corre, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K, top_k_indices):
        # self.pix_coords尺寸是[12, 3, 122880]
        batch_size, num_top = top_k_indices.size()
        batch_indices = torch.arange(batch_size).view(-1, 1).expand(-1, num_top)  # (batch_size, num_top)
        # self.pix_coords尺寸是[12, 3, 122880]
        pix_coords_2 = self.pix_coords[batch_indices, :, top_k_indices].transpose(1,2)
        # self.pix_coords尺寸是[12, 3, num_top]
        cam_points = torch.matmul(inv_K[:, :3, :3], pix_coords_2)
        # cam_points的尺寸为torch.Size([12, 3, num_top])
        # depth尺寸是torch.Size([12, 1, 192, 640])
        depth = depth.view(self.batch_size, 1, -1)
        # depth尺寸是12x1x122880
        depth = depth[batch_indices, :, top_k_indices].transpose(1,2)
        # depth尺寸是[12, 1, num_top]
        cam_points = depth * cam_points
        # cam_points的尺寸为torch.Size([12, 3, num_top])
        ones_2 = self.ones[batch_indices, :, top_k_indices].transpose(1, 2)
        # self.ones的尺寸为torch.Size([12, 1, num_top])
        cam_points = torch.cat([cam_points, ones_2], 1)
        # cam_points的尺寸为torch.Size([12, 4, num_top])
        return cam_points

class Project3D_inv_corre(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D_inv_corre, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, axisangle, translation):
        # axisangle的尺寸为12x1x3，K的尺寸是4x4
        # translation的尺寸为12x192x640x3
        # points尺寸是torch.Size([12, 4, 122880])
        b, _, n = points.size()
        R = rot_from_axisangle(axisangle)
        t = translation.clone()
        R = R.transpose(1, 2)
        # R的尺寸为12x4x4
        t *= -1
        t = t.transpose(2,3).transpose(1,2).reshape(b, 3, n)
        # t的尺寸为12x3x122880
        middle_points = points[:, :3] + t
        # middle_points的尺寸为12x3x122880
        middle_points = R[:,:3,:3] @ middle_points
        # middle_points的尺寸为12x3x122880
        cam_points = torch.matmul(K[:3,:3], middle_points)
        # cam_points尺寸是12x3x122880

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords

class Project3D_corre(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D_corre, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, axisangle, translation):
        # axisangle的尺寸为12x1x3，K的尺寸是4x4
        # translation的尺寸为12x192x640x3
        # points尺寸是torch.Size([12, 4, 122880])
        b, _, n = points.size()
        R = rot_from_axisangle(axisangle)
        # R的尺寸为12x4x4
        t = translation.clone()
        t = t.transpose(2,3).transpose(1,2).reshape(b, 3, n)
        # t的尺寸为12x3x122880
        middle_points = R[:, :3, :3] @ points[:, :3]
        # middle_points的尺寸为12x3x122880
        middle_points = middle_points + t
        # middle_points的尺寸为12x3x122880
        cam_points = torch.matmul(K[:3,:3], middle_points)
        # cam_points尺寸是12x3x122880

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points

class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]#P的尺寸是12x3x4

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords

def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")

def get_smooth_loss_guide_mask(disp, img, mask3_Threshold=None):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)
    # grad_disp_x尺寸为torch.Size([12, 1, 192, 639])
    # grad_disp_y尺寸为torch.Size([12, 1, 191, 640]
    # Res_guide_mask1尺寸为12, 1, 192, 640
    # 先计算平滑损失图
    batch_size, _, height, width = disp.shape
    device = torch.device("cuda")
    middle_grad_disp = torch.zeros(batch_size, 1, height, width, device=device)
    middle_grad_disp[:, :, 1:(height-1), 1:(width-1)] = grad_disp_x[:, :, 1:(height-1), 1:(width-1)]\
                                                        + grad_disp_y[:, :, 1:(height-1), 1:(width-1)]
    # middle_grad_disp = middle_grad_disp / torch.max(middle_grad_disp)
    smooth_loss_guide_mask = torch.where(middle_grad_disp > mask3_Threshold, torch.tensor(1, device=device),
                                         torch.tensor(0, device=device))
    return smooth_loss_guide_mask
def compute_affinity(feature, kernel_size):
    # feature的尺寸为12x32x192x640
    pad = kernel_size // 2
    feature = F.normalize(feature, dim=1)
    # feature的尺寸为12x32x192x640
    unfolded = F.pad(feature, [pad] * 4).unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
    # unfold(2, kernel_size, 1)：在高度维度 H进行滑动窗口操作，提取每个kernel_size的局部块。
    # unfold(3, kernel_size, 1)：在宽度维度 W进行滑动窗口操作，
    # 最终unfolded变成一个kernel_size × kernel_size的局部邻域张量
    # unfolded的尺寸为torch.Size([1, 32, 192, 640, 3, 3])
    feature = feature.unsqueeze(-1).unsqueeze(-1)
    similarity = (feature * unfolded).sum(dim=1, keepdim=True)
    # similarity的尺寸为torch.Size([12, 32, 192, 640, 3, 3])
    # eps = torch.zeros(similarity.shape).to(similarity.device) + 1e-9
    affinity = torch.clamp(2 - 2 * similarity, min=1e-9).sqrt()
    # affinity的尺寸为torch.Size([12, 32, 192, 640, 3, 3])
    return affinity
def create_feature_guide_loss(smooth_loss_guide_mask, feature, disp, edge_map, kernel_size=3,
                              enhan_factor=1, SSIM_enhan_factor=None):
    # disp的尺寸为torch.Size([12, 1, 192, 640])
    # smooth_loss_guide_mask尺寸为torch.Size([12, 1, 192, 640])
    # feature的尺寸为torch.Size([12, 32, 192, 640])
    # SSIM_enhan_factor的尺寸为torch.Size([12, 1, 192, 640])
    # edge_map尺寸为torch.Size([12, 1, 192, 640])
    disp = disp.squeeze(1)
    not_grad_disp = disp.detach()
    smooth_loss_guide_mask = smooth_loss_guide_mask.squeeze(1)
    SSIM_enhan_factor = SSIM_enhan_factor.squeeze(1)
    batch_size, height, width = smooth_loss_guide_mask.shape
    device = torch.device("cuda")
    current_smooth_loss_guide_mask_indexs = torch.nonzero(smooth_loss_guide_mask)
    # current_smooth_loss_guide_mask_indexs的尺寸为??x3
    # SSIM_enhan_factor_values = SSIM_enhan_factor[current_smooth_loss_guide_mask_indexs[:, 0],
    # current_smooth_loss_guide_mask_indexs[:, 1], current_smooth_loss_guide_mask_indexs[:, 2]]
    # SSIM_enhan_factor_values的尺寸为torch.Size([???])
    if current_smooth_loss_guide_mask_indexs.shape[0] == 0:
        return 0
    else:
        center_xy = current_smooth_loss_guide_mask_indexs[:, 1:]
        # center_xy的尺寸为torch.Size([11704, 2])
        batch_indices_1 = current_smooth_loss_guide_mask_indexs[:, 0]
        # batch_indices的尺寸为torch.Size([11704])
        center_disp_value = disp[batch_indices_1, center_xy[:, 0], center_xy[:, 1]]
        # center_disp_value的尺寸为torch.Size([11704])
        feature = F.normalize(feature, dim=1)
        center_feature_value = feature[batch_indices_1, :, center_xy[:, 0], center_xy[:, 1]]
        # center_feature_value的尺寸为torch.Size([11704,32])
        offsets = torch.tensor([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1],
                                [1, -1], [1, 0], [1, 1]], device=torch.device("cuda"))
        # offsets = torch.tensor([[-1, 0], [0, -1], [0, 1], [1, 0]], device=torch.device("cuda"))
        expanded_index_matrix = center_xy[:, None, :] + offsets
        # expanded_index_matrix的尺寸为torch.Size([11704, 4, 2])
        x_indices = expanded_index_matrix[..., 0]
        y_indices = expanded_index_matrix[..., 1]
        # x_indices的尺寸为torch.Size([11704, 4])
        batch_indices_2 = batch_indices_1[:, None]
        current_disp_value = not_grad_disp[batch_indices_2, x_indices, y_indices]
        # current_disp_value的尺寸为torch.Size([11704, 4])
        current_feature_value = feature[batch_indices_2, :, x_indices, y_indices]
        # current_feature_value的尺寸为torch.Size([11704, 4, 32])
        current_disp_value_min, min_dix = torch.min(current_disp_value, 1)
        # current_disp_value_min和min_dix的尺寸为torch.Size([11704])
        row_idx = torch.arange(current_feature_value.shape[0])
        current_feature_value_min = current_feature_value[row_idx, min_dix]
        # current_feature_value_min的尺寸为torch.Size([11704,32])
        current_disp_value_min = current_disp_value_min.unsqueeze(1)
        center_disp_value = center_disp_value.unsqueeze(1)
        current_01_value = torch.where(current_disp_value_min < center_disp_value, torch.tensor(1, device=device),
                                       torch.tensor(0, device=device))
        # current_01_value的尺寸为torch.Size([11704, 1])
        similarity = (center_feature_value * current_feature_value_min).sum(dim=1, keepdim=True)
        # similarity的尺寸为torch.Size([11704, 1])
        # eps = torch.zeros(similarity.shape).to(similarity.device) + 1e-9
        affinity = torch.clamp(2 - 2 * similarity, min=1e-9).sqrt()
        # affinity的尺寸为torch.Size([11704, 1])
        affinity = affinity * current_01_value
        loss = affinity
        loss = loss.mean() * 0.1
        return loss
def create_smooth_guide_L1loss(smooth_loss_guide_mask, img, disp, enhan_factor=1, SSIM_enhan_factor=None):
    # disp的尺寸为torch.Size([12, 1, 192, 640])
    # smooth_loss_guide_mask尺寸为torch.Size([12, 1, 192, 640])
    # img的尺寸为torch.Size([12, 3, 192, 640])
    # SSIM_enhan_factor的尺寸为torch.Size([12, 1, 192, 640])
    disp = disp.squeeze(1)
    not_grad_disp = disp.detach()
    smooth_loss_guide_mask = smooth_loss_guide_mask.squeeze(1)
    SSIM_enhan_factor = SSIM_enhan_factor.squeeze(1)
    batch_size, height, width = smooth_loss_guide_mask.shape
    device = torch.device("cuda")
    current_smooth_loss_guide_mask_indexs = torch.nonzero(smooth_loss_guide_mask)
    # current_smooth_loss_guide_mask_indexs的尺寸为??x3
    SSIM_enhan_factor_values = SSIM_enhan_factor[current_smooth_loss_guide_mask_indexs[:, 0],
    current_smooth_loss_guide_mask_indexs[:, 1], current_smooth_loss_guide_mask_indexs[:, 2]]
    # SSIM_enhan_factor_values的尺寸为torch.Size([???])
    if current_smooth_loss_guide_mask_indexs.shape[0] == 0:
        return 0
    else:
        center_xy = current_smooth_loss_guide_mask_indexs[:, 1:]
        # center_xy的尺寸为torch.Size([11704, 2])
        batch_indices_1 = current_smooth_loss_guide_mask_indexs[:, 0]
        # batch_indices的尺寸为torch.Size([11704])
        center_disp_value = disp[batch_indices_1, center_xy[:, 0], center_xy[:, 1]]
        # center_disp_value的尺寸为torch.Size([11704])
        # offsets = torch.tensor([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1],
        #                         [1, -1], [1, 0], [1, 1]], device=torch.device("cuda"))
        offsets = torch.tensor([[-1, 0], [0, -1], [0, 1], [1, 0]], device=torch.device("cuda"))
        expanded_index_matrix = center_xy[:, None, :] + offsets
        # expanded_index_matrix的尺寸为torch.Size([11704, 4, 2])
        x_indices = expanded_index_matrix[..., 0]
        y_indices = expanded_index_matrix[..., 1]
        # x_indices的尺寸为torch.Size([11704, 4])
        batch_indices_2 = batch_indices_1[:, None]
        current_disp_value = not_grad_disp[batch_indices_2, x_indices, y_indices]
        # current_disp_value的尺寸为torch.Size([11704, 4])
        center_img_value = img[batch_indices_1, :, center_xy[:, 0], center_xy[:, 1]]
        # center_img_value的尺寸为torch.Size([11704, 3])
        current_img_value = img[batch_indices_2, :,  x_indices, y_indices]
        # current_img_value的尺寸为torch.Size([11704, 4, 3])
        center_disp_value = center_disp_value.unsqueeze(1)
        grad_disp = torch.abs(center_disp_value - current_disp_value)
        # grad_disp的尺寸为torch.Size([11704, 4])
        grad_img = torch.mean(torch.abs(center_img_value.unsqueeze(1) - current_img_value), 2, keepdim=True).squeeze(2)
        # grad_img的尺寸为torch.Size([11704, 4])
        grad_disp *= torch.exp(-grad_img)
        # grad_img的尺寸为torch.Size([11704, 4])
        current_01_value = torch.where(current_disp_value < center_disp_value, torch.tensor(1, device=device),
                                       torch.tensor(0, device=device))
        # current_01_value的尺寸为torch.Size([11704, 4])
        current_01_value_sum = torch.sum(current_01_value, 1).unsqueeze(1)
        final_value = current_01_value * grad_disp / (current_01_value_sum + 1e-7)
        # final_value = current_01_value * grad_disp
        # SSIM_enhan_factor_values = SSIM_enhan_factor_values.unsqueeze(1)
        # final_value = final_value * SSIM_enhan_factor_values
        l1_loss = final_value.sum()
        l1_loss = l1_loss / (batch_size * (height-1) * width) * enhan_factor
        return l1_loss

def get_smooth_loss(disp, img, mask3_Threshold=None, edge_mask=None, start_smooth_guide=False, enhan_factor=1,
                    Res_guide_mask1=None, SSIM_num=None):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)
    if start_smooth_guide:
        # 先计算平滑损失图
        batch_size, _, height, width = disp.shape
        device = torch.device("cuda")
        middle_grad_disp = torch.zeros(batch_size, 1, height, width, device=device)
        middle_grad_disp[:, :, 1:(height-1), 1:(width-1)] = grad_disp_x[:, :, 1:(height-1), 1:(width-1)]\
                                                            + grad_disp_y[:, :, 1:(height-1), 1:(width-1)]
        middle_grad_disp = middle_grad_disp.detach()
        # middle_grad_disp = middle_grad_disp / torch.max(middle_grad_disp)
        # save_image(middle_grad_disp, "D:\pycharm_file\MonoViT-main\middle_grad_disp.png")
        smooth_loss_guide_mask = torch.where(middle_grad_disp > mask3_Threshold, torch.tensor(1, device=device),
                                             torch.tensor(0, device=device))
        # smooth_loss_guide_mask_2 = 1 - (middle_grad_disp / torch.max(torch.max(middle_grad_disp, 3, keepdim=True)[0], 2, keepdim=True)[0])
        # smooth_loss_guide_mask_2 = 1 - torch.clip(middle_grad_disp, min=0.0, max=0.15)
        smooth_loss_guide_mask_2 = torch.exp(middle_grad_disp * (-0.75))
        middle_smooth_loss_guide_mask = torch.ones_like(smooth_loss_guide_mask)
        smooth_loss_guide_mask_2 =  smooth_loss_guide_mask * smooth_loss_guide_mask_2 + (1 - smooth_loss_guide_mask) * middle_smooth_loss_guide_mask
        # edge_mask和smooth_loss_guide_mask的尺寸都为torch.Size([2, 1, 192, 640])
        # save_image(smooth_loss_guide_mask.to(torch.float64), "D:\pycharm_file\MonoViT-main\smooth_loss_guide_mask_1.png")
        # if Res_guide_mask1 != None:
        #     # save_image(Res_guide_mask1.to(torch.float64), "D:\pycharm_file\MonoViT-main\Res_guide_mask1.png")
        #     # Res_guide_mask1_mask = smooth_loss_guide_mask * Res_guide_mask1
        #     # middle_Res_guide_mask = torch.zeros(batch_size, 1, height, width, device=device)
        #     # middle_Res_guide_mask[:, :, 1:(height - 1), 1:(width - 1)] = Res_guide_mask1_mask[:, :, 1:(height - 1),
        #     #                                                              1:(width - 1)]
        #     # save_image(middle_Res_guide_mask.to(torch.float64), "D:\pycharm_file\MonoViT-main\middle_Res_guide_mask.png")
        #     Res_smooth_guide_L1loss = create_smooth_guide_L1loss(Res_guide_mask1, img, disp, enhan_factor,
        #                                                          SSIM_num)
        # # print(smooth_loss_guide_mask[:,:,120:136,130:136])
        # smooth_loss_guide_mask = smooth_loss_guide_mask * (1 - edge_mask)
        # # smooth_loss_guide_mask尺寸为torch.Size([12, 1, 192, 640])
        # # save_image(edge_mask, "D:\pycharm_file\MonoViT-main\edge_mask.png")
        # # save_image(smooth_loss_guide_mask, "D:\pycharm_file\MonoViT-main\smooth_loss_guide_mask_2.png")
        # smooth_guide_L1loss = create_smooth_guide_L1loss(smooth_loss_guide_mask, img, disp, enhan_factor,
        #                                                  SSIM_num)
        # if Res_guide_mask1 != None:
        #     return grad_disp_x.mean() + grad_disp_y.mean() + Res_smooth_guide_L1loss + smooth_guide_L1loss, Res_guide_mask1_mask
        # else:
        #     return grad_disp_x.mean() + grad_disp_y.mean() + smooth_guide_L1loss
        if Res_guide_mask1 != None:
            return grad_disp_x.mean() + grad_disp_y.mean(), smooth_loss_guide_mask_2
        else:
            return grad_disp_x.mean() + grad_disp_y.mean(), smooth_loss_guide_mask_2
    elif Res_guide_mask1 != None:
        batch_size, _, height, width = disp.shape
        device = torch.device("cuda")
        smooth_loss_guide_mask = get_smooth_loss_guide_mask(disp, img, mask3_Threshold)
        Res_guide_mask1_mask = smooth_loss_guide_mask * Res_guide_mask1
        middle_Res_guide_mask = torch.zeros(batch_size, 1, height, width, device=device)
        middle_Res_guide_mask[:, :, 1:(height - 1), 1:(width - 1)] = Res_guide_mask1_mask[:, :, 1:(height - 1),
                                                                     1:(width - 1)]
        # Res_smooth_guide_L1loss = create_smooth_guide_L1loss(Res_guide_mask1, img, disp, enhan_factor,
        #                                                      SSIM_num)
        return grad_disp_x.mean() + grad_disp_y.mean(), Res_guide_mask1_mask
    else:
        return grad_disp_x.mean() + grad_disp_y.mean()

# def get_smooth_loss(disp, img, mask3_Threshold=None, edge_mask=None, start_smooth_guide=False, enhan_factor=1,
#                     Res_guide_mask1=None):
#     """Computes the smoothness loss for a disparity image
#     The color image is used for edge-aware smoothness
#     """
#     grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
#     grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])
#
#     grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
#     grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)
#
#     grad_disp_x *= torch.exp(-grad_img_x)
#     grad_disp_y *= torch.exp(-grad_img_y)
#     if start_smooth_guide:
#         # grad_disp_x尺寸为torch.Size([12, 1, 192, 639])
#         # grad_disp_y尺寸为torch.Size([12, 1, 191, 640]
#         # Res_guide_mask1尺寸为12, 1, 192, 640
#         # 先计算平滑损失图
#         batch_size, _, height, width = disp.shape
#         device = torch.device("cuda")
#         middle_grad_disp = torch.zeros(batch_size, 1, height, width, device=device)
#         middle_grad_disp[:, :, 1:(height-1), 1:(width-1)] = grad_disp_x[:, :, 1:(height-1), 1:(width-1)]\
#                                                             + grad_disp_y[:, :, 1:(height-1), 1:(width-1)]
#         middle_grad_disp = middle_grad_disp / torch.max(middle_grad_disp)
#         smooth_loss_guide_mask = torch.where(middle_grad_disp > mask3_Threshold, torch.tensor(1, device=device),
#                                              torch.tensor(0, device=device))
#         # edge_mask和smooth_loss_guide_mask的尺寸都为torch.Size([2, 1, 192, 640])
#         # Save_picture_path = "D:\pycharm_file\Lite-Mono-main\smooth_loss_guide_mask_test.png"
#         # save_image(smooth_loss_guide_mask.to(torch.float64), Save_picture_path)
#         smooth_loss_guide_mask = smooth_loss_guide_mask * (1 - edge_mask)
#         # smooth_loss_guide_mask尺寸为torch.Size([12, 1, 192, 640])
#         # Save_picture_path = "D:\pycharm_file\Lite-Mono-main\edge_mask_test.png"
#         # save_image(edge_mask, Save_picture_path)
#         # cv2.waitKey(100000)
#         if Res_guide_mask1 != None:
#             middle_Res_guide_mask = torch.zeros(batch_size, 1, height, width, device=device)
#             middle_Res_guide_mask[:, :, 1:(height - 1), 1:(width - 1)] = Res_guide_mask1[:, :, 1:(height - 1),
#                                                                          1:(width - 1)]
#             # final_smooth_loss_guide_mask = (1 - smooth_loss_guide_mask) * middle_Res_guide_mask + smooth_loss_guide_mask
#             # Res_smooth_guide_L1loss = create_smooth_guide_L1loss(final_smooth_loss_guide_mask, img, disp,
#             #                                                      enhan_factor)
#             Res_smooth_guide_L1loss = create_smooth_guide_L1loss(middle_Res_guide_mask, img, disp, enhan_factor)
#         # else:
#         #     final_smooth_loss_guide_mask = smooth_loss_guide_mask
#         # smooth_guide_L1loss = create_smooth_guide_L1loss(final_smooth_loss_guide_mask, img, disp,
#         #                                                  middle_enhan_factor_matrix)
#         smooth_guide_L1loss = create_smooth_guide_L1loss(smooth_loss_guide_mask, img, disp, enhan_factor)
#         if Res_guide_mask1 != None:
#             return grad_disp_x.mean() + grad_disp_y.mean() + Res_smooth_guide_L1loss + smooth_guide_L1loss
#         else:
#             return grad_disp_x.mean() + grad_disp_y.mean() + smooth_guide_L1loss
#     elif Res_guide_mask1 != None:
#         batch_size, _, height, width = disp.shape
#         device = torch.device("cuda")
#         middle_Res_guide_mask = torch.zeros(batch_size, 1, height, width, device=device)
#         middle_Res_guide_mask[:, :, 1:(height - 1), 1:(width - 1)] = Res_guide_mask1[:, :, 1:(height - 1), 1:(width - 1)]
#         # smooth_guide_L1loss = create_smooth_guide_L1loss(middle_Res_guide_mask, img, disp)
#         Res_smooth_guide_L1loss = create_smooth_guide_L1loss(middle_Res_guide_mask, img, disp, enhan_factor)
#         return grad_disp_x.mean() + grad_disp_y.mean() + Res_smooth_guide_L1loss
#     else:
#         return grad_disp_x.mean() + grad_disp_y.mean()

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


