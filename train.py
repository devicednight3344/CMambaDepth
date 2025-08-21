# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from trainer import Trainer
from options import MonodepthOptions

options = MonodepthOptions()
opts = options.parse()


if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.train()

# CUDA_VISIBLE_DEVICES=0 python train.py --num_epochs 100 --scales 0 --png --log_dir models --data_path /media/a/b81bd773-44f1-4674-846e-436d8b829731/KITTI_raw_data --num_workers 8 --batch_size 11 --use_channel_mamba --use_channel_mamba_2 --use_HAM --model_name my_train_2025_0410
# CUDA_VISIBLE_DEVICES=0 python train.py --num_epochs 100 --scales 0 --png --log_dir models --dataset ddad --num_workers 16 --batch_size 4  --use_channel_mamba --use_channel_mamba_2 --use_HAM --scheduler_step_size 18 --model_name my_train_2025_0523_ddad --width 640 --height 384
# CUDA_VISIBLE_DEVICES=0 python train.py --num_epochs 100 --scales 0 --png --log_dir models --dataset ddad --num_workers 16 --batch_size 4  --use_channel_mamba --use_channel_mamba_2 --use_HAM --model_name my_train_2025_0523_ddad_2 --width 640 --height 384 --models_to_load encoder mamba_depth pose_encoder pose --load_weights_folder models/my_train_2025_0523_ddad/models/weights_5 --scheduler_step_size 12
