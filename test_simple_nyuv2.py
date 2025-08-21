from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets
import h5py
import networks
from layers import disp_to_depth
import cv2
import heapq
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing function for Lite-Mono models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', required=True)

    parser.add_argument('--load_weights_folder', type=str,
                        help='path of a pretrained model to use',
                        )

    parser.add_argument('--test',
                        action='store_true',
                        help='if set, read images from a .txt file',
                        )

    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    parser.add_argument("--use_channel_mamba",
                        action='store_true')
    parser.add_argument("--use_channel_mamba_2",
                        action='store_true')
    parser.add_argument("--use_HAM",
                        action='store_true')
    parser.add_argument("--dataset",
                             type=str,
                             help="dataset to train on",
                             default="kitti",
                             choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test", "kitti_stereo", "ddad"])
    return parser.parse_args()

def h5_loader(path):
    h5f = h5py.File(path, "r")
    rgb = np.array(h5f['rgb'])
    rgb = np.transpose(rgb, (1, 2, 0))
    depth = np.array(h5f['depth'])
    norm = np.array(h5f['norm'])
    norm = np.transpose(norm, (1, 2, 0))
    valid_mask = np.array(h5f['mask'])

    return rgb, depth, norm, valid_mask

def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    assert args.load_weights_folder is not None, \
        "You must specify the --load_weights_folder parameter"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("-> Loading model from ", args.load_weights_folder)
    encoder_path = os.path.join(args.load_weights_folder, "encoder.pth")
    if not args.use_channel_mamba:
        decoder_path = os.path.join(args.load_weights_folder, "depth.pth")
    else:
        decoder_path = os.path.join(args.load_weights_folder, "mamba_depth.pth")

    encoder_dict = torch.load(encoder_path)
    encoder = networks.hrnet18(False)
    # encoder = networks.hrnet18_liwei(False)
    depth_decoder = networks.DepthDecoder_MSF(encoder.num_ch_enc, [0], num_output_channels=1,
                                              use_channel_mamba=args.use_channel_mamba,
                                              use_channel_mamba_2=args.use_channel_mamba_2,
                                              use_HAM=args.use_HAM, dataset=args.dataset)
    # depth_decoder = networks.DepthDecoder_MSF_liwei(encoder.num_ch_enc, [0], num_output_channels=1)
    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(decoder_path))

    # extract the height and width of image that this model was trained with
    feed_height = encoder_dict['height']
    feed_width = encoder_dict['width']

    encoder.to(device)
    encoder.eval()

    depth_decoder.to(device)
    depth_decoder.eval()

    paths = [args.image_path]
    loader = h5_loader

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue
            data, gt_depth, _, _ = loader(image_path)
            data = data[44: 471, 40: 601, :]
            input_image = pil.fromarray(data)
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            output_directory = os.path.dirname(args.image_path)
            input_image.save(os.path.join(output_directory,"{}.png".format(output_name)))
            # Load image and preprocess
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features, args.use_channel_mamba, args.use_channel_mamba_2, args.use_HAM)

            disp = outputs[("disp", 0)]

            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)
            # Saving numpy file
            scaled_disp, depth = disp_to_depth(disp, 0.1, 100)

            name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
            np.save(name_dest_npy, scaled_disp.cpu().numpy())

            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            # mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            mapper = cm.ScalarMappable(norm=normalizer, cmap='plasma')  # magma
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)

            name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
            im.save(name_dest_im)

            print("   Processed {:d} of {:d} images - saved predictions to:".format(
                idx + 1, len(paths)))
            print("   - {}".format(name_dest_im))
            print("   - {}".format(name_dest_npy))


    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
# python test_simple_nyuv2.py --use_channel_mamba --use_channel_mamba_2 --use_HAM --load_weights_folder models/mytrain_2025_0320_4_438274337_weights_0/weights_0 --image_path D:\NYUv2_test\nyu_test\00001.h5