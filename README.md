## KITTI:
||AbsRel|SqRel|RMSE|RMSElog|δ<1.25|δ<1.25^2|δ<1.25^3|Pretrain_model|
|-|-|-|-|-|-|-|-|-|
|Ours-MonoVit|0.096|0.675|4.341|0.173|0.902|0.967|0.984|https://drive.google.com/file/d/1yHlkQfMLiXVoEITWuzpj46kSOuzEiPPZ/view?usp=drive_link|
|Ours|0.094|0.616|4.156|0.169|0.905|0.969|0.985|https://drive.google.com/file/d/1GasbJ1eTVma-1OnEi-PhMmpAf6X9kYbs/view?usp=drive_link|


## DDAD:
||AbsRel|SqRel|RMSE|RMSElog|δ<1.25|δ<1.25^2|δ<1.25^3|Pretrain_model|
|-|-|-|-|-|-|-|-|-|
|Ours|0.140|2.761|13.550|0.229|0.829|0.936|0.972|https://drive.google.com/file/d/1n92P-fp2IdznIzhvHiLsXCjHHhROAmuD/view?usp=drive_link|


## Experimental Environment:
The experiments were conducted on a single NVIDIA RTX 4090 GPU with Python 3.7.12. The required deep learning environment can be found in requirements.txt(https://github.com/devicednight3344/CMambaDepth/blob/dfa49acbf45b20131d12136296ff99ac1844cd10/requirements.txt).


## Training:
```
python train.py --num_epochs 35 --scales 0 --png --log_dir models --data_path /media/a/b81bd773-44f1-4674-846e-436d8b829731/KITTI_raw_data --num_workers 8 --batch_size 11 --use_channel_mamba --use_channel_mamba_2 --use_HAM --model_name CMambaDepth
```


## Testing:
```
python evaluate_depth.py --eval_mono --height 192 --width 640 --scales 0 --data_path /media/a/b81bd773-44f1-4674-846e-436d8b829731/KITTI_raw_data --png --use_channel_mamba --use_channel_mamba_2 --use_HAM --load_weights_folder models/CMambaDepth
```

## Infer a single depth map from a RGB:
```
python test_simple.py --use_channel_mamba --use_channel_mamba_2 --use_HAM --model_name CMambaDepth --image_path img/0000000002.png
```


#### Acknowledgement
 - Thank the authors for their superior works: [monodepth2](https://github.com/nianticlabs/monodepth2), [MonoVit](https://github.com/zxcqlf/monovit), [RA-Depth](https://github.com/hmhemu/RA-Depth), [VMamba](https://github.com/MzeroMiko/VMamba/tree/main), [mamba](https://github.com/state-spaces/mamba).

