# [TCSVT2025] AVLTrack: Dynamic Sparse Learning for Aerial Vision-Language Tracking

This is an official pytorch implementation of the 2025 IEEE Transactions on Circuits and Systems for Video Technology paper:
```
AVLTrack: Dynamic Sparse Learning for Aerial Vision-Language Tracking
(accepted by IEEE Transactions on Circuits and Systems for Video Technology, DOI: 10.1109/TCSVT.2025.3549953)
```

![image](https://github.com/xyl-507/AVLTrack/blob/main/figs/framework.jpg)

The models and raw results can be downloaded from [**[GitHub]**](https://github.com/xyl-507/AVLTrack/releases/tag/model_results) and [**[BaiduYun]**](https://pan.baidu.com/s/1sHjaELBFMh8KBjOwAmh6AQ?pwd=43xv).

The tracking demos are displayed on the [Bilibili](https://www.bilibili.com/video/BV1hT9QYBEKU/).

### Proposed modules
- `DSL` and `MLP` in [backbone](https://github.com/xyl-507/AVLTrack/blob/main/lib/models/ostrack/AbaViT_trans_tksa_multi.py)

## Requirements
- python==3.8.18
- torch==1.13.0
- torchvision==0.14.0
- torchaudio==0.13.0
- timm==0.9.10
- pip install transformers==4.12.5 -i https://pypi.tuna.tsinghua.edu.cn/simple/

## Results (AUC) on WebUAV-3M
|Trackers|Source|Initialize|AUC|Pre.|NPre.|cAUC|mAcc|Param.(M)|FPS|
|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
|***AVLTrack***|Ours|NL+BB|55.0|70.0|58.8|54.1|55.7|23.944|80.5@GPU|
|PRL-Track |IROS’24|BB|46.3|62.3|50.8|45.2|46.7|13.377|174.9@GPU|
|TDA-Track |IROS’24|BB|46.4|62.5|51.8|45.3|46.8|5.661|134.7@GPU|
|SmallTrack|TGRS’23|BB|50.1|66.2|55.2|49.2|50.7|29.394|72.5@GPU|
|Aba-ViTrack|ICCV’23|BB|53.5|68.3|57.0|52.6|54.2|7.979|86.6@GPU|
## Results (Pre) on DTB70-NLP
|Trackers|Source|Initialize|Pre. on DTB70-NLP|
|:----|:----|:----|:----|
|AVLTrack|Ours|NL+BB|86.3|
|MixFormerV2|NeurIPS’23|BB|84.1|
|SeqTrack-B384 |CVPR’23|BB|85.9|
|MixFormer-CvT |TPAMI’24|BB|82.7|
|SmallTrack |TGRS’23|BB|85.8|
|Aba-ViTrack |ICCV’23|BB|85.9|
|TDA-Track |IROS’24|BB|80.2|
|DCPT |ICRA’24|BB|84.0|
|AVTrack |ICML’24|BB|84.3|
|LiteTrack |ICRA’24|BB|82.5|
## Results (AUC) on UAV20L-NLP
|Trackers|Source|Initialize|AUC on UAV20L-NLP|
|:----|:----|:----|:----|
|AVLTrack|Ours|NL+BB|63.9|
|E.T.Track |WACV’23|BB|60.0|
|TaMOs-Swin-B |WACV’24|BB|61.2|
|ACM-BAN |TPAMI’24|BB|56.0|
|HiT-Small |ICCV’23|BB|63.0|
|Aba-ViTrack |ICCV’23|BB|63.5|
|PRL-Track |IROS’24|BB|52.0|
|TDA-Track |IROS’24|BB|50.6|
|SAM-DA-Base |ICARM’24|BB|55.9|
|CGDenoiser |IROS’24|BB|54.7|
|QRDT |TIM’24|BB|55.8|
|DaDiff-GAT |IROS’24|BB|57.3|

It should be noted that the above pretrained model is trained on an Ubuntu 18.04 server with multiple NVIDIA RTX A100 GPUs. For WebUAV-3M, we recommend the official [evaluation toolkit](https://github.com/983632847/WebUAV-3M).

## Set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```
## Data Preparation
Put the tracking datasets in ./data. It should look like this:
   ```
   ${PROJECT_ROOT}
    -- data
        -- lasot
            |-- airplane
            |-- basketball
            |-- bear
            ...
        -- got10k
            |-- test
            |-- train
            |-- val
        -- coco
            |-- annotations
            |-- images
        -- trackingnet
            |-- TRAIN_0
            |-- TRAIN_1
            ...
            |-- TRAIN_11
            |-- TEST
   ```

## Training
Download pre-trained [MAE ViT-Base weights](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) and put it to `$PROJECT_ROOT$/AVLTrack/lib/models/pretrained_models`.

1.Training with one GPU.
```
cd /$PROJECT_ROOT$/AVLTrack
CUDA_VISIBLE_DEVICES=0 python tracking/train.py --script ostrack --config abavit_patch16_224_ep300 --save_dir ./output --mode single --nproc_per_node 1
```

2.Training with multiple GPUs.
```
cd /$PROJECT_ROOT$/AVLTrack
CUDA_VISIBLE_DEVICES=0,1 python tracking/train.py --script ostrack --config abavit_patch16_224_ep300 --save_dir ./output --mode multiple --nproc_per_node 2
```

Before training, please make sure the data path in [***local.py***](./lib/train/admin/local.py) is correct.

## Evaluation
Download the model [AVLTrack](https://pan.baidu.com/s/1sHjaELBFMh8KBjOwAmh6AQ?pwd=43xv), extraction code: `43xv`. Add the model to `$PROJECT_ROOT$/AVLTrack/output/checkpoints/train/`.
```
python tracking/test.py --tracker_name ostrack --tracker_param abavit_patch16_224_ep300 --dataset webuav3m --threads 2 --num_gpus 2
python tracking/analysis_results.py

```

Before evaluation, please make sure the data path in [***local.py***](./lib/test/evaluation/local.py) is correct.

## Test FLOPs, and Speed
```
python tracking/profile_model.py --script ostrack --config levit_256_32x4_ep300
```

# UAV vision-language tracking dataset: DTB70-NLP, UAV20L-NLP, UAVDT-NLP, and VisDrone2019-SOT-test-dev-NLP

Considering that WebUAV-3M is the only UAV vision-language tracking dataset, we additionally construct vision-language 
tracking datasets DTB70-NLP, UAV20L-NLP, UAVDT-NLP, and VisDrone2019-SOT-test-dev-NLP based on the vision-only aerial
dataset DTB70, UAV20L, UAVDT, and VisDrone2019-SOT-test-dev. Note that DTB70-NLP and UAV20L-NLP are non-overlapping with
the training set. Therefore, to better evaluate the model generalization, DTB70-NLP and UAV20L-NLP are only used for testing.

|Dataset|#Video|#Total frame|#Mean frame|#Mean language|
|:----|:----|:----|:----|:----|
|WebUAV-3M|780|3.3 M|710|14.4|
|DTB70-NLP|70|15.8 K|225|12.6|
|UAV20L-NLP|20|58.6 K|2934|11.5|


### Acknowledgement
The code based on the [OSTrack](https://github.com/botaoye/OSTrack),
[All-in-One](https://github.com/983632847/All-in-One), and [A-ViT](https://github.com/NVlabs/A-ViT)
We would like to express our sincere thanks to the contributors.

### Citation:
If you find this work useful for your research, please cite the following papers:
```
@ARTICLE{10220112,
  author={Yuanliang Xue,Bineng Zhong,Guodong Jin,Tao Shen,Lining Tan,Ning Li,Yaozong Zheng},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={AVLTrack: Dynamic Sparse Learning for Aerial Vision-Language Tracking}, 
  year={2025},
  doi={10.1109/TCSVT.2025.3549953}}
```
If you have any questions about this work, please contact with me via xyl_507@outlook.com