# 代码从PPEA-Depth抄的
import dgp
from dgp.datasets import SynchronizedSceneDataset

import torch
import torch.utils.data as data
from torchvision import transforms
import numpy as np
import random
from PIL import Image  # using pillow-simd for increased speed
import PIL.Image as pil

DDAD_TRAIN_VAL_JSON_PATH = '/media/a/b81bd773-44f1-4674-846e-436d8b829731/hyq_DDAD/ddad_train_val/ddad.json'
DATUMS = ['lidar'] + ['CAMERA_%02d' % idx for idx in [1, 5, 6, 7, 8, 9]]


# ddad_train = SynchronizedSceneDataset(
#     DDAD_TRAIN_VAL_JSON_PATH,
#     split='train',
#     datum_names=DATUMS,
#     generate_depth_from_datum='lidar'
# )
# print('Loaded DDAD train split containing {} samples'.format(len(ddad_train)))

class DDADDataset(data.Dataset):
    def __init__(self, num_scales, frame_idxs, is_train, width=640, height=384):
        super(DDADDataset, self).__init__()

        split = 'train' if is_train == True else 'val'

        self.ddad_train_with_context = SynchronizedSceneDataset(
            DDAD_TRAIN_VAL_JSON_PATH,
            split=split,
            datum_names=('lidar', 'CAMERA_01',),
            generate_depth_from_datum='lidar',
            forward_context=1,
            backward_context=1
        )
        # self.num_scales = num_scales
        self.num_scales = 1
        self.frame_idxs = frame_idxs
        self.is_train = is_train
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.05, 0.05)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.05

        self.resize = {}
        self.resize_tensor = {}
        self.width = width
        self.height = height
        self.interp = Image.LANCZOS

        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)
            self.resize_tensor[i] = transforms.Resize((self.height // s, self.width // s))

        self.to_tensor = transforms.ToTensor()

    def preprocess(self, inputs, color_aug, height_re_HiS, width_re_HiS, height_re_LoS, width_re_LoS, dx_HiS, dy_HiS, do_crop_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        self.resize_HiS = transforms.Resize((height_re_HiS, width_re_HiS), interpolation=self.interp)
        self.resize_MiS = transforms.Resize((self.height, self.width), interpolation=self.interp)
        self.resize_LoS = transforms.Resize((height_re_LoS, width_re_LoS), interpolation=self.interp)
        box_HiS = (dx_HiS, dy_HiS, dx_HiS+self.width, dy_HiS+self.height)
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n + "_HiS", im, i)] = self.resize_HiS(inputs[(n, im, i - 1)]).crop(box_HiS)
                    inputs[(n + "_MiS", im, i)] = self.resize_MiS(inputs[(n, im, i - 1)])
                    inputs[(n + "_LoS", im, i)] = self.resize_LoS(inputs[(n, im, i - 1)])
        for k in list(inputs):
            f = inputs[k]
            if "color_HiS" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
            if "color_MiS" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f) #[3,192,640]
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
            if "color_LoS" in k:
                n, im, i = k
                LoS_part = self.to_tensor(f)
                point1 = int(2*width_re_LoS-self.width)
                point2 = int(2*height_re_LoS-self.height)
                Tensor_LoS = torch.zeros(3, self.height, self.width)
                Tensor_LoS[:, 0:height_re_LoS, 0:width_re_LoS] = LoS_part
                Tensor_LoS[:, height_re_LoS:self.height, 0:width_re_LoS] = LoS_part[:, point2:height_re_LoS, 0:width_re_LoS]
                Tensor_LoS[:, 0:height_re_LoS, width_re_LoS:self.width] = LoS_part[:, 0:height_re_LoS, point1:width_re_LoS]
                Tensor_LoS[:, height_re_LoS:self.height, width_re_LoS:self.width] = LoS_part[:, point2:height_re_LoS, point1:width_re_LoS]
                inputs[(n, im, i)] = Tensor_LoS

    def __len__(self):
        return len(self.ddad_train_with_context)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "depth_gt"                              for ground truth depth maps

        <frame_id> is:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """

        samples = self.ddad_train_with_context[index]
        front_cam_images = []
        for sample in samples:
            front_cam_images.append(sample[0]['rgb'])

        front_cam_images = [img.resize((self.width, self.height), Image.BILINEAR) for img in front_cam_images]
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5
        do_crop_aug = self.is_train

        # High-Scale
        ra_HiS = 1.1
        rb_HiS = 2.0
        resize_ratio_HiS = (rb_HiS - ra_HiS) * random.random() + ra_HiS
        if do_crop_aug:
            height_re_HiS = int(self.height * resize_ratio_HiS)
            width_re_HiS = int(self.width * resize_ratio_HiS)
        else:
            height_re_HiS = self.height
            width_re_HiS = self.width

        height_d_HiS = height_re_HiS - self.height
        width_d_HiS = width_re_HiS - self.width
        if do_crop_aug:
            dx_HiS = int(width_d_HiS * random.random())
            dy_HiS = int(height_d_HiS*random.random())
        else:
            dx_HiS = 0
            dy_HiS = 0


        # Middle-Scale
        dx_MiS = 0
        dy_MiS = 0


        # Low-Scale
        ra_LoS = 0.7
        rb_LoS = 0.9
        resize_ratio_LoS = (rb_LoS - ra_LoS) * random.random() + ra_LoS
        height_re_LoS = int(self.height * resize_ratio_LoS)
        width_re_LoS = int(self.width * resize_ratio_LoS)

        dx_LoS = 0
        dy_LoS = 0

        inputs[("dxy_HiS")] = torch.Tensor((dx_HiS, dy_HiS))
        inputs[("dxy_MiS")] = torch.Tensor((dx_MiS, dy_MiS))
        inputs[("dxy_LoS")] = torch.Tensor((dx_LoS, dy_LoS))
        inputs[("resize_HiS")] = torch.Tensor((width_re_HiS, height_re_HiS))
        inputs[("resize_LoS")] = torch.Tensor((width_re_LoS, height_re_LoS))

        inputs[("color", -1, -1)] = front_cam_images[0]
        inputs[("color", 0, -1)] = front_cam_images[1]
        inputs[("color", 1, -1)] = front_cam_images[2]
        if do_flip:
            inputs[("color", -1, -1)] = inputs[("color", -1, -1)].transpose(pil.FLIP_LEFT_RIGHT)
            inputs[("color", 0, -1)] = inputs[("color", 0, -1)].transpose(pil.FLIP_LEFT_RIGHT)
            inputs[("color", 1, -1)] = inputs[("color", 1, -1)].transpose(pil.FLIP_LEFT_RIGHT)
        K = np.zeros((4, 4), np.float32)
        K[:3, :3] = samples[1][0]['intrinsics'].copy()
        K[3][3] = 1

        K[0, :] /= 1936
        K[1, :] /= 1216  # self.height // (2 ** scale)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K_HiS = K.copy()
            K_MiS = K.copy()
            K_LoS = K.copy()

            K_HiS[0, :] *= width_re_HiS // (2 ** scale)
            K_HiS[1, :] *= height_re_HiS // (2 ** scale)

            inv_K_HiS = np.linalg.pinv(K_HiS)
            inputs[("K_HiS", scale)] = torch.from_numpy(K_HiS)
            inputs[("inv_K_HiS", scale)] = torch.from_numpy(inv_K_HiS)

            K_MiS[0, :] *= self.width // (2 ** scale)
            K_MiS[1, :] *= self.height // (2 ** scale)
            inv_K_MiS = np.linalg.pinv(K_MiS)
            inputs[("K_MiS", scale)] = torch.from_numpy(K_MiS)
            inputs[("inv_K_MiS", scale)] = torch.from_numpy(inv_K_MiS)

            K_LoS[0, :] *= width_re_LoS // (2 ** scale)
            K_LoS[1, :] *= height_re_LoS // (2 ** scale)
            inv_K_LoS = np.linalg.pinv(K_LoS)
            inputs[("K_LoS", scale)] = torch.from_numpy(K_LoS)
            inputs[("inv_K_LoS", scale)] = torch.from_numpy(inv_K_LoS)

        if do_color_aug:
            color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)

        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug, height_re_HiS, width_re_HiS, height_re_LoS, width_re_LoS, dx_HiS, dy_HiS, do_crop_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]#del删除的是变量，而不是数据。
        # print(inputs[("color", 0, 0)].shape)
        # Image.fromarray((inputs[("color_aug", 1, 0)]*255.).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)).save('0_0.jpg')
        # Image.fromarray((inputs[("color_aug", 0, 0)]*255.).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)).save('0_1.jpg')
        # Image.fromarray((inputs[("color_aug", -1, 0)]*255.).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)).save('0_2.jpg')
        # # Image.fromarray((inputs[("color_aug", 1, 3)]*255.).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)).save('0_3.jpg')
        # exit(0)

        if not self.is_train:
            inputs[("depth")] = samples[1][0]['depth']

        return inputs


if __name__ == "__main__":
    # ddad_train = SynchronizedSceneDataset(
    #     DDAD_TRAIN_VAL_JSON_PATH,
    #     split='val',
    #     datum_names=DATUMS,
    #     generate_depth_from_datum='lidar',
    #     forward_context=1,
    #     backward_context=1
    # )

    # ddad_train_with_context = SynchronizedSceneDataset(
    #     DDAD_TRAIN_VAL_JSON_PATH,
    #     split='train',
    #     datum_names=DATUMS, #('CAMERA_01',),
    #     generate_depth_from_datum='lidar',
    #     forward_context=1,
    #     backward_context=1
    # )
    # print('Loaded DDAD train split containing {} samples'.format(len(ddad_train)))

    dataset = DDADDataset(4, [0, -1, 1], is_train=False)
    from torch.utils.data.dataloader import DataLoader

    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    for item in loader:
        print(111)
        a = 10

    dataset = DDADDataset(4, [0, -1, 1], is_train=True)
    from torch.utils.data.dataloader import DataLoader

    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    for item in loader:
        print(112)
        a = 10
