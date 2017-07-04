# Fast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Network

by [Jin Yamanaka](https://github.com/jiny2001), Shigesumi Kuwashima and [Takio Kurita](http://home.hiroshima-u.ac.jp/tkurita/profile-e.html)

## Overview

This is a tensorflow implementation of gFast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Networkh, a deep learning based Single-Image Super-Resolution (SISR) model. We named it DCSCN.

The model structure is like below. We use Deep CNN with Residual Net, Skip Connection and Network in Network. A combination of Deep CNNs and Skip connection layers is used as a feature extractor for image features on both local and global area. Parallelized 1x1 CNNs, like the one called Network in Network, is also used for image reconstruction.

![alt tag](https://raw.githubusercontent.com/jiny2001/dcscn-super-resolution/master/documents/figure1.jpeg)

Sample result is here.

![alt tag](https://raw.githubusercontent.com/jiny2001/dcscn-super-resolution/master/documents/result.png)

Our model, DCSCN is much lighter than other Deep Learning based SISR models. Here is a comparison chart of performance vs computation complexity from our paper.

![alt tag](https://raw.githubusercontent.com/jiny2001/dcscn-super-resolution/master/documents/compare.png)


## requirements

python > 3.5
tensorflow > 1.0, scipy, numpy and pillow


## result of PSNR

The result of default parameter is here. You can have even better PSNR than below with using larger filters or deeper layers with our model.

| DataSet | Bicubic | SRCN | SelfEx | DRCN | VDSR | DCSCN (ours) |
|:-------:|:-------:|:----:|:----:|:----:|:----:|:----:|
|Set5 x2|33.66|36.66|36.49|37.63|37.53|37.60|
|Set14 x2|30.24|32.42|32.22|33.04|33.03|33.04|
|BSD100 x2|29.56|31.36|31.18|31.85|31.90|31.90|

## evaluate

Learned weights for some parameters are included in this GitHub. Execute **evaluate.py** with these args below and you get results in **output** folder. When you want to evaluate with other parameters, try training before then evaluate with same parameters as training have done.


```
# evaluating for set14 dataset
python evaluate.py --test_dataset set14 --dataset yang_bsd_4 --save_results True

# evaluating compact version of our model for set14 dataset
python evaluate.py --test_dataset set14 --dataset yang_bsd_4 --save_results True --filters 32 --min_filters 8 --nin_filters 24 --nin_filters2 8
```

## how to train

You can train with any datasets. Put your image files as a training dataset into the directory under **data** directory, then specify with --dataset arg.

```
# training with yang91 dataset
python train.py --dataset yang91

# training with larger filters and deeper layers
python train.py --dataset yang91 --filters 128 --layers 10
```

## data augmentation

To get a better performance, data augmentation is neede. You can use **augmentation.py** to build a augmented dataset. Augment level 4 means it will add right-left, top-bottom and right-left and top-bottom fillped images to make a 4 times bigger dataset. And there will be **ynag91_4** directory as a augmented datase.

```
# build 4x augmented dataset for yang91 dataset (will add flipped images)
python augmentation.py --dataset yang91 --augment_level 4

# build 8x augmented dataset for yang91 dataset (will add flipped and rotated images)
python augmentation.py --dataset yang91 --augment_level 8

```












