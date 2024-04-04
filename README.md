# IndoorDepth
This is a pytorch implementation of **IndoorDepth** for [**Deeper into Self-Supervised Monocular Indoor Depth Estimation**](https://arxiv.org/abs/2312.01283).
The training codes and pretrained models are available here. 
It is worth noting that the training codes only support 3 frames of training. If you need to use 5 frames of images as input for training, you need to modify the content in our trainer.py. (For example, from line 312 to line 323).

## Preparation

#### Installation

Assuming a fresh Anaconda distribution (with python 3.6), you can Install pytorch first by running 

```bash
conda install pytorch=1.5.1 torchvision=0.6.1  cudatoolkit=10.1 -c pytorch
```

Then install other requirements

```bash
pip install -r requirements.txt
```

#### Datasets & Preprocessing 

Please download preprocessed (sampled in 5 frames) [NYU-Depth-V2](https://drive.google.com/file/d/1WoOZOBpOWfmwe7bknWS5PMUCLBPFKTOw/view?usp=sharing) dataset by [Junjie Hu](https://scholar.google.com/citations?user=nuZZKu4AAAAJ&hl=en&oi=sra) and extract it. 

Extract the superpixels and line segments by excuting. The following steps can be skipped if only testing our pretrained models.

```
python preprocess/extract_superpixel.py --data_path data/
pip uninstall opencv_python==4.4.0.46
pip install opencv_python==3.1.0.4
python preprocess/extract_lineseg.py --data_path data/ 
pip uninstall opencv_python==3.1.0.4
pip install opencv_python==4.4.0.46
```

## Try an image 

```
python inference_single_image.py --image_path $IMAGE_PATH --load_weights_folder $MODEL_PATH
```

## Training

```
python train.py --data_path $DATA_PATH --frame_ids 0 -2 2
```

## Evaluation  

We provide pretrained models on NYUv2 datasets. You need to uncompress it and put it into the model folder.

[IndoorDepth model](https://1drv.ms/u/s!AudzvQ6XfIoSkEDosbn0yqRASZZ1?e=q6Vgn4):

|Models      | Abs Rel | Log10 | RMS   | Acc.1 | Acc.2 | Acc.3 |
| ----------- | ------- | ----- | ----- | ----- | ----- | ----- |
|     [ARN](https://github.com/JiawangBian/sc_depth_pl)     | 0.138   | 0.059 | 0.532 | 0.820 | 0.956 | 0.989 |
| MonoIndoor  | 0.134   |   -   | 0.526 | 0.823 | 0.958 | 0.989 |
| MonoIndoor++  | 0.132   |   -   | 0.517 | 0.834 | 0.961 | 0.990 |
| **IndoorDepth (Ours)** | **0.126**   | **0.054** | **0.494** | **0.845** | **0.965** | **0.991** |

#### NYU Depth Estimation

```
python evaluate_nyu_depth.py --data_path $DATA_PATH --load_weights_folder $MODEL_PATH 
```

#### ScanNet Depth Estimation

```
python evaluate_scannet_depth.py --data_path $DATA_PATH --load_weights_folder $MODEL_PATH 
```

Note: to evaluate on ScanNet, one has to download the preprocessed [data](https://onedrive.live.com/?authkey=%21ANXK7icE%2D33VPg0&id=C43E510B25EDDE99%21106&cid=C43E510B25EDDE99) by  P^2Net. 


## Acknowledgements

The project borrows codes from [Monodepth2](https://github.com/nianticlabs/monodepth2) , [P^2Net](https://github.com/svip-lab/Indoor-SfMLearner) and [PLNet](https://github.com/HalleyJiang/PLNet). Many thanks to their authors. 

## Citation

Please cite our papers if you find our work useful in your research.

```
@misc{fan2023deeper,
      title={Deeper into Self-Supervised Monocular Indoor Depth Estimation}, 
      author={Chao Fan and Zhenyu Yin and Yue Li and Feiqing Zhang},
      year={2023},
      eprint={2312.01283},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
