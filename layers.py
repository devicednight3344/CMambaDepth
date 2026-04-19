# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE .convs["fusion_{}".format(file.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from mamba_ssm.utils.generation import InferenceParams
from mamba_ssm.modules.mamba_simple import Mamba
import math
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from torch.utils.checkpoint import checkpoint
from timm.models.layers import DropPath
from functools import partial
from timm.models.helpers import named_apply
from timm.models.layers import trunc_normal_tf_

def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

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


def right_from_parameters(axisangle, translation0, translation1):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t0 = translation0.clone()
    t1 = translation1.clone()

    t1 *= -1
    t = t0 + t1

    T = get_translation_matrix(t)

    M = torch.matmul(R, T)

    return M


def left_from_parameters(translation):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    t = translation.clone()

    t *= -1

    T = get_translation_matrix(t)

    return T


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
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

class ConvBlock1x1(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock1x1, self).__init__()

        self.conv = Conv1x1(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

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

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class Conv1x1(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels):
        super(Conv1x1, self).__init__()

        self.conv = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1, stride=1)

    def forward(self, x):
        out = self.conv(x)
        return out

class DetailGuide(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(DetailGuide, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

    def forward(self, height_re, width_re, dxy):

        points_all = []
        for i in range(self.batch_size):
            meshgrid = np.meshgrid(range(int(dxy[i,0]), int(dxy[i,0]+self.width)), range(int(dxy[i,1]), int(dxy[i,1]+self.height)), indexing='xy')
            id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
            id_coords = nn.Parameter(torch.from_numpy(id_coords),
                                          requires_grad=False)  #[2,192,640]

            ones = nn.Parameter(torch.ones(1, 1, self.height * self.width),
                                     requires_grad=False).cuda()   #[1,1,122880]

            pix_coords = nn.Parameter(torch.unsqueeze(torch.stack(
                [id_coords[0].view(-1), id_coords[1].view(-1)], 0), 0), requires_grad=False).cuda()    #[1,2,122880]
            # self.pix_coords = self.pix_coords.repeat(self.batch_size, 1, 1)  #[12,2,122880]

            points_all.append(pix_coords)

        points_all = torch.cat(points_all, 0)   #[12,2,122880]
        points_all = points_all.view(self.batch_size, 2, self.height, self.width)
        points_all = points_all.permute(0, 2, 3, 1) #[12,192,640,2]
        points_all[..., 0] *= (self.width * 1.0 / width_re)
        points_all[..., 1] *= (self.height * 1.0 / height_re)
        points_all[..., 0] /= self.width - 1
        points_all[..., 1] /= self.height - 1
        points_all = (points_all - 0.5) * 2

        return points_all


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        # self.ones = nn.Parameter(torch.ones(1, 1, self.height * self.width),
        #                              requires_grad=False)   #[1,1,122880]

        # meshgrid = np.meshgrid(range(dx, dx+640), range(dy, dy+192), indexing='xy')
        # self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        # self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
        #                               requires_grad=False)  #[2,192,640]

        # self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
        #                          requires_grad=False)   #[12,1,122880]

        # self.pix_coords = torch.unsqueeze(torch.stack(
        #     [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)    #[1,2,122880]
        # self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)  #[12,2,122880]
        # self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
        #                                requires_grad=False)   #[12,3,122880]

    def forward(self, depth, inv_K, dxy):

        cam_points_all = []
        for i in range(self.batch_size):
            meshgrid = np.meshgrid(range(int(dxy[i,0]), int(dxy[i,0]+self.width)), range(int(dxy[i,1]), int(dxy[i,1]+self.height)), indexing='xy')
            id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
            id_coords = nn.Parameter(torch.from_numpy(id_coords),
                                          requires_grad=False)  #[2,192,640]

            ones = nn.Parameter(torch.ones(1, 1, self.height * self.width),
                                     requires_grad=False).cuda()   #[1,1,122880]

            pix_coords = torch.unsqueeze(torch.stack(
                [id_coords[0].view(-1), id_coords[1].view(-1)], 0), 0).cuda()    #[1,2,122880]
            # self.pix_coords = self.pix_coords.repeat(self.batch_size, 1, 1)  #[12,2,122880]
            pix_coords = nn.Parameter(torch.cat([pix_coords, ones], 1),
                                           requires_grad=False).cuda()   #[1,3,122880]


            cam_points = torch.matmul(inv_K[i, :3, :3], pix_coords)   #[1,3,122880]
            cam_points = depth[i,0,:,:].view(1, 1, -1) * cam_points   #[1,3,122880]
            cam_points = torch.cat([cam_points, ones], 1)   #[1,4,122880]
            cam_points_all.append(cam_points)

        cam_points_all = torch.cat(cam_points_all, 0)

        return cam_points_all


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T, dxy):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        dxy = dxy.unsqueeze(1).unsqueeze(2).expand(-1, self.height, self.width, -1)
        pix_coords -= dxy
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


class Project3D_poseconsis(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D_poseconsis, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, T):
        # P = torch.matmul(K, T)[:, :3, :] #P:[12,3,4]   T:[12,4,4]    points:[12,4,1]

        # cam_points = torch.matmul(P, points)

        cam1 = torch.matmul(T, points)  #[12,4,1]

        return cam1


def updown_sample(x, scale_fac):
    """Upsample input tensor by a factor of scale_fac
    """
    return F.interpolate(x, scale_factor=scale_fac, mode="nearest")


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")

def downsample(x):
    """Downsample input tensor by a factor of 1/2
    """
    return F.interpolate(x, scale_factor=1.0/2, mode="nearest")

# def get_smooth_loss(disp, img):
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
#
#     return grad_disp_x.mean() + grad_disp_y.mean()

def create_smooth_guide_L1loss(Res_guide_mask, img, disp, enhan_factor=1, SSIM_enhan_factor=None):
    # disp的尺寸为torch.Size([12, 1, 192, 640])
    # Res_guide_mask尺寸为torch.Size([12, 1, 192, 640])
    # img的尺寸为torch.Size([12, 3, 192, 640])
    # SSIM_enhan_factor的尺寸为torch.Size([12, 1, 192, 640])
    disp = disp.squeeze(1)
    not_grad_disp = disp.detach()
    Res_guide_mask = Res_guide_mask.squeeze(1)
    SSIM_enhan_factor = SSIM_enhan_factor.squeeze(1)
    batch_size, height, width = Res_guide_mask.shape
    device = torch.device("cuda")
    current_Res_guide_mask_indexs = torch.nonzero(Res_guide_mask)
    # current_Res_guide_mask_indexs的尺寸为??x3
    SSIM_enhan_factor_values = SSIM_enhan_factor[current_Res_guide_mask_indexs[:, 0],
    current_Res_guide_mask_indexs[:, 1], current_Res_guide_mask_indexs[:, 2]]
    # SSIM_enhan_factor_values的尺寸为torch.Size([???])
    if current_Res_guide_mask_indexs.shape[0] == 0:
        return 0
    else:
        center_xy = current_Res_guide_mask_indexs[:, 1:]
        # center_xy的尺寸为torch.Size([11704, 2])
        batch_indices_1 = current_Res_guide_mask_indexs[:, 0]
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
        # current_01_value_sum的尺寸为torch.Size([11704, 1])
        # current_01_value_sum_mask = torch.where(current_01_value_sum < 2, 0, 1)
        # final_value = current_01_value * grad_disp * current_01_value_sum_mask
        final_value = current_01_value * grad_disp / (current_01_value_sum + 1e-7)
        SSIM_enhan_factor_values = SSIM_enhan_factor_values.unsqueeze(1)
        final_value = final_value * SSIM_enhan_factor_values
        l1_loss = final_value.sum()
        l1_loss = l1_loss / (batch_size * (height - 1) * width) * enhan_factor
        return l1_loss

def get_smooth_loss(disp, img, start_smooth_guide=False, enhan_factor=1,
                    Res_guide_mask=None, SSIM_num=None, Res_Res_guide_mask1=None):
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
        if Res_guide_mask != None:
            middle_Res_guide_mask = torch.zeros(batch_size, 1, height, width, device=device)
            middle_Res_guide_mask[:, :, 1:(height - 1), 1:(width - 1)] = Res_guide_mask[:, :, 1:(height - 1),
                                                                         1:(width - 1)]
            Res_smooth_guide_L1loss = create_smooth_guide_L1loss(middle_Res_guide_mask, img, disp, enhan_factor,
                                                                 SSIM_num)
        middle_grad_disp = middle_grad_disp.detach()
        smooth_loss_guide_mask = torch.exp(middle_grad_disp * (-0.40))
        if Res_guide_mask != None:
            return grad_disp_x.mean() + grad_disp_y.mean() + Res_smooth_guide_L1loss, smooth_loss_guide_mask
        else:
            return grad_disp_x.mean() + grad_disp_y.mean(), smooth_loss_guide_mask
    elif Res_guide_mask != None:
        batch_size, _, height, width = disp.shape
        device = torch.device("cuda")
        middle_Res_guide_mask = torch.zeros(batch_size, 1, height, width, device=device)
        middle_Res_guide_mask[:, :, 1:(height - 1), 1:(width - 1)] = Res_guide_mask[:, :, 1:(height - 1),
                                                                     1:(width - 1)]
        Res_smooth_guide_L1loss = create_smooth_guide_L1loss(middle_Res_guide_mask, img, disp, enhan_factor,
                                                             SSIM_num)
        return grad_disp_x.mean() + grad_disp_y.mean() + Res_smooth_guide_L1loss, Res_guide_mask
    else:
        return grad_disp_x.mean() + grad_disp_y.mean()

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

def transform_tensor(x,b,h,w,scale_factor):

    B, C1, C2 = x.shape # torch.Size([5280, 18, 16])

    x = x.view(b, h, w, C1, C2)  # torch.Size([11, 12, 40, 18, 16])
    x = x.reshape(b, h, w, C1 * C2) # torch.Size([11, 12, 40, 18*16])
    x = x.permute(0, 3, 1, 2).contiguous()  # (11, 18*16, 12, 40)

    x = F.interpolate(x, scale_factor=(scale_factor, scale_factor), mode='nearest')  # (11, 18*16, 24, 80)

    x = x.permute(0, 2, 3, 1).contiguous()  # (11, 24, 80, 18*16)
    b, h, w, _ = x.shape

    x = x.view(b * h * w, C1 * C2)  # (21120, 18*16)

    x = x.view(-1, C1, C2)  # (B1 * 2*B2 * 2*B3, 16, 16)

    return x

# MambaModule类
class MambaModule(nn.Module):
    def __init__(self, input_channels, hide_channel, d_model, mamba_sqe, mamba_sqe_list, mamba_channel_list,
                 linear_in_channel, out_channel, out_channels, out_sqe_num, out_mamba_sqe_num, d_model_origin, downsample_size,
                 d_conv=1, layer_idx=0, index=None):
        super(MambaModule, self).__init__()
        self.convs = nn.ModuleDict()
        self.d_model = d_model
        self.mamba_sqe = mamba_sqe
        self.mamba_sqe_list = mamba_sqe_list
        self.mamba_channel_list = mamba_channel_list
        self.downsample_size = downsample_size
        self.index = index
        self.ori_d_model = input_channels // mamba_sqe

        self.input_Batch_norm = nn.BatchNorm2d(input_channels)
        if downsample_size != 0:
            for i in range(downsample_size):
                if i == 0:
                    self.convs["downconv_{}_0".format(i)] = nn.Sequential(
                        nn.Conv2d(input_channels, hide_channel, 3, 2, 1, groups=mamba_sqe, bias=False),
                        nn.BatchNorm2d(hide_channel),
                        nn.SiLU(inplace=True))
                else:
                    self.convs["downconv_{}_0".format(i)] = nn.Sequential(
                        nn.Conv2d(hide_channel, hide_channel, 3, 2, 1, groups=mamba_sqe, bias=False),
                        nn.BatchNorm2d(hide_channel),
                        nn.SiLU(inplace=True))
                    # nn.SiLU(inplace=True)
            for idx in range(len(self.mamba_channel_list)):
                self.convs["upconv_{}_1".format(idx)] = nn.Sequential(
                    nn.Conv2d(self.mamba_channel_list[idx], self.mamba_channel_list[idx], 1, bias=False),
                    # nn.Conv2d(self.mamba_channel_list[idx], self.mamba_channel_list[idx], 3, padding=1, bias=False),
                    nn.BatchNorm2d(self.mamba_channel_list[idx]),
                    nn.SiLU(inplace=True))
        else:
            self.convs["conv_0"] = nn.Sequential(
                        nn.Conv2d(input_channels, hide_channel, 3, padding=1, groups=mamba_sqe, bias=False),
                        nn.SiLU(inplace=True))
            for idx in range(len(self.mamba_channel_list)):
                self.convs["conv_{}_1".format(idx)] = ConvBlock(self.mamba_channel_list[idx], self.mamba_channel_list[idx])

        self.convs["embedding"] = nn.Conv2d(hide_channel, d_model * self.mamba_sqe, 3, padding=1,
                                            groups=self.mamba_sqe)
        self.layer_norm = nn.LayerNorm(d_model)
        self.mamba = nn.ModuleDict()
        dt_rank = math.floor(d_model / 2)
        for i in range(len(mamba_sqe_list)):
            self.mamba["{}".format(i)] = Mamba(d_model=d_model, d_conv=d_conv, dt_rank=dt_rank, layer_idx=layer_idx)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.convs["decoder_embedding"] = nn.Conv2d(d_model*out_sqe_num, linear_in_channel, 3, padding=1,
                                                    groups=out_mamba_sqe_num)
        out_channels_sum = 0
        for idx in range(len(self.mamba_channel_list)):
            self.convs["fusion_{}".format(idx)] = nn.Sequential(
                nn.Conv2d(d_model_origin + self.mamba_channel_list[idx],
                          out_channels[idx], 3, padding=1),
                nn.BatchNorm2d(out_channels[idx]),
                nn.SiLU(inplace=True))
            out_channels_sum = out_channels_sum + out_channels[idx]
        self.convs["out_fusion"] = nn.Conv2d(out_channels_sum, out_channel, 1)
        self.silu = nn.SiLU(inplace=True)
    #     self.apply(self._init_weights)
    # def _init_weights(self, m):
    #     if isinstance(m, nn.Conv2d):
    #         trunc_normal_(m.weight, std=.02)
    #         # nn.init.constant_(m.bias, 0)
    #         if m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    def forward(self, features, high_feature, features_identity, expanded_ratio, out_sqe_num):
        inference_params = InferenceParams(max_seqlen=12, max_batch_size=23040, seqlen_offset=1)
        expanded_high_feature = high_feature.repeat(1, expanded_ratio, 1, 1)
        features = features + expanded_high_feature
        if self.index == "d3_0":
            features = torch.cat([features, high_feature], 1)
        else:
            features = torch.cat([high_feature, features], 1)
        features = self.input_Batch_norm(features)
        if self.downsample_size != 0:
            for i in range(self.downsample_size):
                features = self.convs["downconv_{}_0".format(i)](features)
        else:
            features = self.convs["conv_0"](features)
        features = self.convs["embedding"](features)
        b, c, h, w = features.size()
        features = rearrange(features, 'b c h w -> (b h w) c', b=b, c=c, h=h, w=w)
        features = rearrange(features, 'b (s d) -> b s d', s=self.mamba_sqe, d=self.d_model)
        Mamba_feature_identity = features[:,(-out_sqe_num):]
        features = self.layer_norm(features)
        mamba_sqe_chunk_orig = 0
        return_ssm_state = True
        record_out = False
        features_final = []
        for i in range(len(self.mamba_sqe_list)):
            if i == (len(self.mamba_sqe_list) - 1):
                return_ssm_state = False
            if mamba_sqe_chunk_orig == (self.mamba_sqe - out_sqe_num):
                record_out = True
            mamba_sqe_chunk = self.mamba_sqe_list[i]
            if record_out:
                out, ssm_state = self.mamba["{}".format(i)](
                    features[:, mamba_sqe_chunk_orig:mamba_sqe_chunk + mamba_sqe_chunk_orig],
                    inference_params)
                features_final.append(out)
                if return_ssm_state:
                    inference_params.key_value_memory_dict[0][1] = ssm_state
            else:
                _, ssm_state = self.mamba["{}".format(i)](
                    features[:, mamba_sqe_chunk_orig:mamba_sqe_chunk + mamba_sqe_chunk_orig],
                    inference_params)
                inference_params.key_value_memory_dict[0][1] = ssm_state
            mamba_sqe_chunk_orig = mamba_sqe_chunk + mamba_sqe_chunk_orig
        features = torch.cat(features_final, 1)
        features = Mamba_feature_identity + features
        features = self.layer_norm_2(features)
        # out_sqe_num = 2
        B, s, d = features.size()
        features = rearrange(features, 'b s d -> b (s d)', b=B, s=s, d=d)
        # 23040x72
        B, c = features.size()
        features = rearrange(features, '(b h w) c -> b c h w', b=b, c=c, h=h, w=w)
        features = self.convs["decoder_embedding"](features)
        # 12x72x24x80
        middle_features = []
        d_model_orin = 0
        for channel_num in self.mamba_channel_list:
            d_model_next = d_model_orin + channel_num
            middle_features.append(features[:, d_model_orin:d_model_next])
            d_model_orin = d_model_orin + channel_num
        final_features = []
        for idx in range(len(self.mamba_channel_list)):
            input_feature = features_identity[idx]
            middle_feature = middle_features[idx]
            if self.downsample_size != 0:
                _, _, h, w = middle_feature.size()
                h_scale = h * (2 ** self.downsample_size)
                w_scale = w * (2 ** self.downsample_size)
                middle_feature = F.interpolate(middle_feature, [h_scale, w_scale], mode="nearest")
                middle_feature = self.convs["upconv_{}_1".format(idx)](middle_feature)
            else:
                middle_feature = self.convs["conv_{}_1".format(idx)](middle_feature)
            middle_feature = torch.cat([middle_feature, input_feature], 1)
            middle_feature = self.convs["fusion_{}".format(idx)](middle_feature)
            final_features.append(middle_feature)
        features = torch.cat(final_features, 1)
        features = self.convs["out_fusion"](features)
        features = self.silu(features)
        return features

class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        activation_fn=F.gelu,
        dropout=0.0,
        activation_dropout=0.0,
        layernorm_eps=1e-6,
        subln=False,
        subconv=True
        ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = activation_fn
        self.activation_dropout_module = torch.nn.Dropout(activation_dropout)
        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = nn.Conv2d(self.embed_dim, ffn_dim, 1)
        self.fc2 = nn.Conv2d(ffn_dim, self.embed_dim, 1)
        self.ffn_layernorm = nn.LayerNorm(ffn_dim, eps=layernorm_eps) if subln else None
        self.dwconv = nn.Conv2d(ffn_dim, ffn_dim, 3, 1, 1, groups=ffn_dim) if subconv else None

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        if self.ffn_layernorm is not None:
            self.ffn_layernorm.reset_parameters()

    def forward(self, x: torch.Tensor):
        '''
        x: (b c h w)
        '''
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout_module(x)
        residual = x
        if self.dwconv is not None:
            x = self.dwconv(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = x + residual
        x = self.fc2(x)
        x = self.dropout_module(x)
        return x

class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            # d_state="auto", # 20240109
            d_conv=3,
            mamba_sqe=None,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.mamba_sqe = mamba_sqe
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
            # nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            # nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            # self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
            #              **factory_kwargs),
            # self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
            #              **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=2, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=2, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn
        # self.selective_scan = selective_scan_ref

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32

        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        _, C, H, W = x.shape
        L = self.mamba_sqe
        K = 2
        x = rearrange(x, '(b s) c h w -> b s c h w', s=self.mamba_sqe, c=C, h=H, w=W)
        x = rearrange(x, 'b s c h w -> (b h w) c s', s=self.mamba_sqe, c=C, h=H, w=W)
        B = x.shape[0]
        x = x.unsqueeze(1)
        # x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
        #                      dim=1).view(B, 2, -1, L)
        # Bx1xCxL
        # xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)
        xs = torch.cat([x, torch.flip(x, dims=[-1])], dim=1)  # (b, k, d, l)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)
        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        # inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        # wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        # invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        inv_y = torch.flip(out_y[:, 1], dims=[-1]).view(B, 1, -1, L)

        return out_y[:, 0], inv_y[:, 0]

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y1, y2 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2
        # y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = torch.transpose(y, dim0=1, dim1=2).contiguous() # bxsxd
        _, s, d = y.shape
        y = rearrange(y, '(b h w) s d -> b h w s d', s=s, d=d, h=H, w=W)
        y = rearrange(y, 'b h w s d -> (b s) h w d', s=s, d=d, h=H, w=W)
        # y: B, H, W, C
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class MambaModule_2(nn.Module):
    def __init__(self, input_channels, hide_channel, hide_channels, out_channel, out_channels, d_model, mamba_sqe,
                 mamba_channel_list, Anchor, height, downsample_size=None, mamba_num=1, d_conv=3, dt_rank=3):
        super(MambaModule_2, self).__init__()
        self.convs = nn.ModuleDict()
        self.d_model = d_model
        self.mamba_sqe = mamba_sqe
        self.mamba_channel_list = mamba_channel_list
        self.height = height
        self.downsample_size = downsample_size
        self.mamba_num = mamba_num
        self.mambas = nn.ModuleDict()

        for idx in range(5):
            if idx < Anchor:
                for i in range(downsample_size[idx]):
                    if i == 0:
                        self.convs["downconv_{}_{}_0".format(idx, i)] = nn.Sequential(
                            nn.Conv2d(input_channels[idx], hide_channels[idx], 3, 2, 1, bias=False),
                            nn.BatchNorm2d(hide_channels[idx]),
                            nn.SiLU(inplace=True))

                    else:
                        self.convs["downconv_{}_{}_0".format(idx, i)] = nn.Sequential(
                            nn.Conv2d(hide_channels[idx], hide_channels[idx], 3, 2, 1, bias=False),
                            nn.BatchNorm2d(hide_channels[idx]),
                            nn.SiLU(inplace=True))
                self.convs["upconv_{}_1".format(idx)] = nn.Sequential(
                    nn.Conv2d(self.mamba_channel_list[idx], self.mamba_channel_list[idx], 1, bias=False),
                    # nn.Conv2d(self.mamba_channel_list[idx], self.mamba_channel_list[idx], 3, padding=1, bias=False),
                    nn.BatchNorm2d(self.mamba_channel_list[idx]),
                    nn.SiLU(inplace=True))
            elif idx > Anchor:
                self.convs["upconv_{}_0".format(idx)] = nn.Sequential(
                    nn.Conv2d(input_channels[idx], hide_channels[idx], 1, bias=False),
                    nn.BatchNorm2d(hide_channels[idx]),
                    nn.SiLU(inplace=True))
                for i in range(downsample_size[idx]):
                    self.convs["downconv_{}_{}_1".format(idx, i)] = nn.Sequential(
                        nn.Conv2d(self.mamba_channel_list[idx], self.mamba_channel_list[idx], 3, 2, 1, bias=False),
                        nn.BatchNorm2d(self.mamba_channel_list[idx]),
                        nn.SiLU(inplace=True))
            else:
                self.convs["conv_{}_0".format(idx)] = ConvBlock(input_channels[idx], hide_channels[idx])
                self.convs["conv_{}_1".format(idx)] = ConvBlock(self.mamba_channel_list[idx],
                                                                self.mamba_channel_list[idx])
            # self.convs["fusion_{}".format(idx)] = ConvBlock(input_channels[idx] + self.mamba_channel_list[idx],
            #                                                 out_channels[idx])
            # self.convs["fusion_{}".format(idx)] = ConvBlock1x1(input_channels[idx] + self.mamba_channel_list[idx],
            #                                                     out_channels[idx])
            self.convs["fusion_{}".format(idx)] = nn.Sequential(nn.Conv2d(input_channels[idx] + self.mamba_channel_list[idx],
                                                                          out_channels[idx],3,padding=1),
                                                                nn.SiLU(inplace=True))
        self.silu = nn.SiLU(inplace=True)
        self.convs["embedding"] = nn.Conv2d(hide_channel, d_model * self.mamba_sqe, 3, padding=1, groups=self.mamba_sqe)
        for j in range(self.mamba_num):
            self.mambas["layer_norm_{}".format(j)] = nn.LayerNorm(d_model)
            drp = [x.item() for x in torch.linspace(0, 0.15, self.mamba_num)]
            self.mambas["SS2D_{}".format(j)] = SS2D(d_model=d_model, mamba_sqe=mamba_sqe, d_conv=d_conv,
                                                    dt_rank=dt_rank, dropout=drp[j])
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.convs["decoder_embedding"] = nn.Conv2d(d_model*self.mamba_sqe, out_channel, 3, padding=1,
                                                    groups=self.mamba_sqe)

    def forward(self, input_features):
        middle_features = []
        for idx in range(5):
            middle_feature = input_features[idx]
            _, _, h, w = middle_feature.size()
            if h > self.height:
                for i in range(self.downsample_size[idx]):
                    # h_scale = h // (2 ** (i + 1))
                    # w_scale = w // (2 ** (i + 1))
                    # middle_feature = F.interpolate(middle_feature, [h_scale, w_scale], mode="nearest")
                    middle_feature = self.convs["downconv_{}_{}_0".format(idx, i)](middle_feature)
                middle_features.append(middle_feature)
            elif h < self.height:
                h_scale = h * (2 ** self.downsample_size[idx])
                w_scale = w * (2 ** self.downsample_size[idx])
                middle_feature = F.interpolate(middle_feature, [h_scale, w_scale], mode="nearest")
                middle_feature = self.convs["upconv_{}_0".format(idx)](middle_feature)
                middle_features.append(middle_feature)
            else:
                middle_features.append(self.convs["conv_{}_0".format(idx)](middle_feature))
        feature = torch.cat(middle_features, 1)
        feature = self.convs["embedding"](feature)
        b, c, h, w = feature.size()
        feature = rearrange(feature, 'b (s d) h w -> (b s) d h w', b=b, s=self.mamba_sqe, d=self.d_model, h=h, w=w)
        feature = rearrange(feature, 'b d h w -> b h w d', d=self.d_model, h=h, w=w)
        for j in range(self.mamba_num):
            feature_identity = feature
            feature = self.mambas["layer_norm_{}".format(j)](feature)
            # features:23040x8x18
            # with autocast():  # 自动选择合适的精度（float16 或 float32）
            feature = self.mambas["SS2D_{}".format(j)](feature)
            feature = feature_identity + feature
        feature = self.layer_norm_2(feature)
        B, h, w, c = feature.size()
        feature = rearrange(feature, 'b h w d -> b d h w', b=B, d=self.d_model, h=h, w=w)
        feature = rearrange(feature, '(b s) d h w -> b (s d) h w', b=b, s=self.mamba_sqe, d=self.d_model, h=h, w=w)
        feature = self.convs["decoder_embedding"](feature)
        middle_features = []
        d_model_orin = 0
        for channel_num in self.mamba_channel_list:
            d_model_next = d_model_orin + channel_num
            middle_features.append(feature[:, d_model_orin:d_model_next])
            d_model_orin = d_model_orin + channel_num

        final_features = []
        for idx in range(5):
            input_feature = input_features[idx]
            middle_feature = middle_features[idx]
            _, _, h, w = input_feature.size()
            if h > self.height:
                _, _, h, w = middle_feature.size()
                h_scale = h * (2 ** self.downsample_size[idx])
                w_scale = w * (2 ** self.downsample_size[idx])
                middle_feature = F.interpolate(middle_feature, [h_scale, w_scale], mode="nearest")
                middle_feature = self.convs["upconv_{}_1".format(idx)](middle_feature)
            elif h < self.height:
                for i in range(self.downsample_size[idx]):
                    # h_scale = h // (2 ** (i + 1))
                    # w_scale = w // (2 ** (i + 1))
                    # middle_feature = F.interpolate(middle_feature, [h_scale, w_scale], mode="nearest")
                    middle_feature = self.convs["downconv_{}_{}_1".format(idx, i)](middle_feature)
            else:
                middle_feature = self.convs["conv_{}_1".format(idx)](middle_feature)
            middle_feature = torch.cat([middle_feature, input_feature], 1)
            middle_feature = self.convs["fusion_{}".format(idx)](middle_feature)
            final_features.append(middle_feature)

        return final_features





