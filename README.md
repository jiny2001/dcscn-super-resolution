
# Fast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Network

by [Jin Yamanaka](https://github.com/jiny2001), Shigesumi Kuwashima and [Takio Kurita](http://home.hiroshima-u.ac.jp/tkurita/profile-e.html)


## Overview (Ver 2.)

This is a tensorflow implementation of ["Fast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Network"](https://arxiv.org/abs/1707.05425), a deep learning based Single-Image Super-Resolution (SISR) model. We named it **DCSCN**. If you are checking the paper's code and results, please check [ver1 branch](https://github.com/jiny2001/dcscn-super-resolution/tree/ver1).

The model structure is like below. We use Deep CNN with Residual Net, Skip Connection and Network in Network. A combination of Deep CNNs and Skip connection layers is used as a feature extractor for image features on both local and global area. Parallelized 1x1 CNNs, like the one called Network in Network, is also used for image reconstruction.

<img src="https://raw.githubusercontent.com/jiny2001/dcscn-super-resolution/master/documents/figure1_v2.png" width="800">

As a ver2, we also implemented these features.

* __Pixel Shuffler__ from ["Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network"
](https://arxiv.org/abs/1609.05158)

* __Transposed-CNN__ (optional) from ["Fully Convolutional Networks for Semantic Segmentation"](https://arxiv.org/abs/1411.4038)

* __Self Ensemble__ from ["Seven ways to improve example-based single image super resolution"](https://arxiv.org/abs/1511.02228)

* __Clipping Normalization__ (Gradient clipping)

* Dynamically load training images (optional)

* Add extra layers / Update default parameters for better PSNR result

## Sample result

<img src="https://raw.githubusercontent.com/jiny2001/dcscn-super-resolution/master/documents/result.png" width="864">

Our model, DCSCN was much lighter than other Deep Learning based SISR models which is [ver 1](https://github.com/jiny2001/dcscn-super-resolution/tree/ver1). As a ver2, we use larger model and recent technics to get a better image restoration performace.


## Requirements

python > 3.5

tensorflow > 1.0, scipy, numpy and pillow


## Result of PSNR

The sample result of default parameter is here. The model is trained by DIV2k with 150,000 48x48 images for one epoch.

| Dataset | Scale | Bicubic | DRCN | VDSR | DCSCN (ver2) | DRRN |
|:-------:|:-------:|:-------:|:----:|:----:|:----:|:----:|
| set5 | x2 | 33.66 | 37.63 | 37.53 | 37.79 | 37.74 |
|  | x3 | 30.39 | 33.82 | 33.66 | 34.06 | 34.03 |
|  | x4 | 28.42 | 31.53 | 31.35 | 31.72 | 31.68 |
| set14 | x2 | 30.24 | 33.04 | 33.03 | 33.25 | 33.23 |
|  | x3 | 27.55 | 29.76 | 29.77 | 29.95 | 29.96 |
|  | x4 | 26.00 | 28.02 | 28.01 | 28.25 | 28.21 |
| bsd100 | x2 | 29.56 | 31.85 | 31.90 | 32.00 | 32.05 |
|  | x3 | 27.21 | 28.80 | 28.82 | 28.89 | 28.95 |
|  | x4 | 25.96 | 27.23 | 27.29 | 27.35 | 27.38 |

Results and model will be uploaded in some days!!

## Evaluate

Learned weights for some parameters are included in this GitHub. Execute **evaluate.py** with these args below and you get results in **output** directory. When you want to evaluate with other parameters, try training first then evaluate with same parameters as training have done. Results will be logged at log.txt, please check.

Some pre-trained models are included.

```
# evaluating set14 dataset
python evaluate.py --test_dataset=set14 --save_results=true

# evaluating set5 dataset with small model
python evaluate.py --test_dataset=set5 --save_results=true --layers=8 --filters=96

# evaluating all(set5,set14,bsd100) dataset
python evaluate.py --test_dataset=all
```

## Apply to your own image

Place your image file in this project directory. And then run "sr.py --file 'your_image_file'" to apply Super Resolution. Results will be generated in **output** directory. Please note you should use same args which you used for training.

If you want to apply this model on your image001.png file, try those.

```
# apply super resolution on image001.jpg (then see results at output directory)
python sr.py --file=your_file.png

# apply super resolution with small model
python sr.py --file=your_file.png --layers=7 --filters=64
```

## How to train

You can train with any datasets. Put your image files as a training dataset into the directory under **data** directory, then specify with --dataset arg. There are some other hyper paramters to train, check [args.py](https://github.com/jiny2001/dcscn-super-resolution/blob/master/helper/args.py) to use other training parameters.

Each training and evaluation result will be added to **log.txt**.

```
# training for x2 with bsd200 dataset
python train.py --dataset=bsd200 --training_images=80000

# training for x2 with small model
python train.py --dataset=bsd200 --layers=8 --filters=96 --training_images=30000

# training for x2 with tiny model for test
python train.py --dataset=set5 --layers=5 --filters=32 --use_nin=false --training_images=10000

# training for x2 with transposed CNN instead of using Pixel Shuffler layer for up-sampling
python train.py --dataset=bsd200 --training_images=80000 --pixel_shuffler=false

# training for x3 scale
python train.py --scale=3
```

Please note loading/converting batch images for each training is a bit heavy process since there will be a lot of iterations. Here are some options. You can use those option to reduce training time significantly.

1. Use "convert_y.py" to convert your dataset images to Y-channel monochrome bitmap.
If your training data is compressed like PNG or jpeg and the image resolution is larger, you must convert it before. Especially for DIV2K dataset, you can save a big time for decompressing and converting image process.
Also in this mode, each input batch image may be flipped horizontally by the probability of 50%.

2. Use "--build_batch True" option for smaller dataset
If your dataset is small enough to store in CPU memory, please use this. It will build a batch images before the training. When you're using HDD(not SSD) and the dataset is not large like (Yang91 + BSD200) augmented by 8 methods, this option can avoid loading/converting process for each batch.
In this case, batch image positions are adjusted and limited to be on the grid with the half of batch_image_size. However, as far as I experimented, that doesn't affect to PSNR performance so much.

# Important parameters

| Parameter arg | Name | Default | Explanation |
|:-------:|:-------:|:----:|:----:|
| layers | Num CNN layers | 12 | Number of layers of feature-extraction CNNs |
| filters | Num of first CNN filters | 196 | Number of filters of the first feature-extraction CNNs |
| min_filters | Num of last CNN filters | 48 | Number of filters of the last feature-extraction CNNs |
| filters_decay_gamma | Decay Gamma | 1.5 | Number of CNN filters are decayed from [filters] to [min_filters] by this gamma on each layers |
| pixel_shuffler | Pixel Shuffler | True | Use Pixel Shuffler as up-sampling layer. If it's False, use transposed CNN as up-sampling layer. |
| self_ensemble | Self Ensemble | 8 | Apply SR for 1-8 flipped/rotate images and then use mean image as result. |
| training_images | Batch images for training epoch | 24000 | This number of batch images are used for training one epoch. I usually use 100,000 batch images for each 10 epochs for each Learning Rate. |
| dropout_rate | Dropout rate | 0.8 | Output nodes should be kept by this probability. Should be 1 >= drop out > 0. If 1, don't use dropout. |
| initializer | Initialize method | he | Initialize method of each weight. Can be one of [uniform, stddev, xavier, he, identity, zero]. |
| activator | Activator function | prelu | Type of activator functions for each CNN. Can be one of [relu, leaky_relu, prelu, sigmoid, tanh] |
| optimizer | Optimizer function | adam | Method of optimizer. Can be one of [gd, momentum, adadelta, adagrad, adam, rmsprop] |
| batch_image_size | Image size for each Batch | 48 | Each training image will be splitted this size. |
| batch_num | Image num for each batch | 20 | Number of batch images for one training step. |
| clipping_norm | value for gradient clipping | 5 | Norm for gradient clipping. If it's <= 0 we don't use gradient clipping. |

Also learning late and other model parameters are still important.

We'll train [ __training_images__ / __batch_num__ ] steps for each epoch. Learning rate will be started from __initial_lr__. After [lr_decay_epoch] epochs has done, learning rate will be decayed by __lr_decay__. And when the current learning rate gets lower than __end_lr__, then training will be finished.

Usually you can control how much iterate the training batch by adjusting __training_images__ and __lr_decay_epoch__. Please check arg.py for other params.

## Data augmentation

To get a better performance, data augmentation is needed. You can use **augmentation.py** to build an augmented dataset. The arg, augment_level = 4, means it will add right-left, top-bottom and right-left and top-bottom fillped images to make 4 times bigger dataset. And there **yang91_4** directory will be generated as an augmented dataset.

To have better model, you should use larger training data like (BSD200 + Yang91) x (8 augment) dataset.

```
# build 4x augmented dataset for yang91 dataset (will add flipped images)
python augmentation.py --dataset=yang91 --augment_level=4

# build 8x augmented dataset for yang91 dataset (will add flipped and rotated images)
python augmentation.py --dataset=yang91 --augment_level=8

# train with augmented data
python train.py --dataset=yang91_4
```

## How to calculate PSNR

RGB image is converted to YCbCr image. And then trained and applied only for Y channel.

For PSNR, by the default, result and original image are cropped by the border size = (2 + scale), rounded to integer and then MSE and PSNR are calculated.


## Visualization

During the training, tensorboard log is available under **tf_log** directory.

<img src="https://raw.githubusercontent.com/jiny2001/dcscn-super-resolution/master/documents/model_v2.png" width="400">

You can check test PSNR transition during training. Also mean / std / histogram of every weights and biases are logged at tensorboard.

<img src="https://raw.githubusercontent.com/jiny2001/dcscn-super-resolution/master/documents/tensorboard_1.png" width="800">
<img src="https://raw.githubusercontent.com/jiny2001/dcscn-super-resolution/master/documents/tensorboard_2.png" width="800">
