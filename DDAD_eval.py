# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
from __future__ import absolute_import, division, print_function
import os

os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402

import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks
from tqdm import tqdm


cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

splits_dir = "splits"

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def validate(encoder, depth_decoder, opt):
    val_dataset = datasets.DDADDataset(4, opt.frame_ids, is_train=False, width=opt.width,
                                        height=opt.height)
    val_loader = DataLoader(
        val_dataset, opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True, drop_last=False)
    errors = []
    ratios = []

    MIN_VAL = 1e-3
    MAX_VAL = 200

    with torch.no_grad():
        iterator = tqdm(val_loader)  # , desc=f"Epoch: {self.epoch + 1}/{self.opt.
        i = 0
        for data in iterator:  # self.val_loader: #tqdm(self.val_loader, desc=f"Epoch: {self.epoch + 1}/{self.opt.num_epochs}. Loop: Validation"):
            input_color = data[("color_MiS", 0, 0)].cuda()
            output = depth_decoder(encoder(input_color), opt.use_channel_mamba, opt.use_channel_mamba_2,
                                   opt.use_HAM)
            pred_disp, _ = disp_to_depth(output[("disp", 0)], 0.1, 200)

            # pred_disp = pred_disp # (b, 1, h, w)
            # vis_depth(pred_disp[0][0].cpu().numpy(), "pred")
            gt_depth = data['depth'].cpu().numpy()
            gt_height, gt_width = gt_depth.shape[-2:]
            pred_disp = torch.nn.functional.interpolate(pred_disp, size=(gt_height, gt_width), mode='bilinear',
                                                        align_corners=True).cpu()[:, 0].numpy()
            # pred_disp的尺寸为(12, 1216, 1936)
            for b_idx in range(pred_disp.shape[0]):
                pred_ = 1. / pred_disp[b_idx]  # 1/pred_disp[b_idx]
                gt_ = gt_depth[b_idx]
                # vis_depth(gt_, "gt")
                # img = Image.fromarray(np.uint8(255 * data[("color", 0, 0)][0].permute(1,2,0))) # no opencv required
                # img.save("./debugout/file.png")

                mask = np.logical_and(gt_ > 0.0, gt_ < 200)

                pred_ = pred_[mask]
                gt_ = gt_[mask]

                pred_ *= opt.pred_depth_scale_factor
                if not opt.disable_median_scaling:
                    ratio = np.median(gt_) / np.median(pred_)
                    ratios.append(ratio)
                    pred_ *= ratio

                pred_[pred_ < 0.0] = 0.0
                pred_[pred_ > 200] = 200
                errors.append(compute_errors(gt_, pred_))
                # print(errors[0])
                # exit(0)
            i += 1
        # errors列表的最终长度为3850
        # np.array(errors).shape = (3850, 7)
        mean_errors = np.array(errors).mean(0)
        # mean_errors的尺寸为(7,)
    return mean_errors


def evaluate(opt):
    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    # decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "mamba_depth.pth")

    encoder_dict = torch.load(encoder_path)

    # encoder = networks.ResnetEncoder(opt.num_layers, False)
    encoder = networks.hrnet18(False)

    depth_decoder = networks.DepthDecoder_MSF(encoder.num_ch_enc, [0], num_output_channels=1,
                                              use_channel_mamba=opt.use_channel_mamba,
                                              use_channel_mamba_2=opt.use_channel_mamba_2,
                                              use_HAM=opt.use_HAM, dataset=opt.dataset)

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(decoder_path))

    encoder.cuda()
    encoder.eval()
    depth_decoder.cuda()
    depth_decoder.eval()

    eval_res = validate(encoder, depth_decoder, opt)
    print(eval_res)

    errors = eval_res
    print("\n  " + ("{:>8} | " * 7).format("abs_rel",
                                           "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*errors.tolist()) + "\\\\")


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
# CUDA_VISIBLE_DEVICES=0 python DDAD_eval.py --eval_mono --height 384 --width 640 --scales 0 --use_HAM --use_channel_mamba --use_channel_mamba_2 --num_workers 16 --dataset ddad --load_weights_folder models/my_train_2025_0522_ddad/models/weights_19