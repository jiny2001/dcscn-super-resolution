"""
Paper: "Fast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Network"
Ver: 2

functions for sharing arguments and their default values
"""

import sys
import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# Model
flags.DEFINE_integer("filters", 96, "Number of CNN filters")
flags.DEFINE_integer("min_filters", 32, "Number of the last CNN filters")
flags.DEFINE_integer("nin_filters", 64, "Number of CNN filters in A1 at Reconstruction network")
flags.DEFINE_integer("nin_filters2", 32, "Number of CNN filters in B1 and B2 at Reconstruction net.")
flags.DEFINE_integer("cnn_size", 3, "Size of CNN filters")
flags.DEFINE_integer("reconstruct_layers", 1, "Number of Reconstruct CNN Layers. Should be larger than 1")
flags.DEFINE_integer("reconstruct_filters", 32, "Number of Reconstruct CNN Filters")
flags.DEFINE_integer("layers", 7, "Number of layers of CNNs")
flags.DEFINE_boolean("use_nin", True, "Use Network In Network")
flags.DEFINE_boolean("bicubic_init", True, "make bicubic interpolation values as initial input of x2")
flags.DEFINE_float("dropout_rate", 0.8, "For dropout value for  value. Don't use if it's 1.0.")
flags.DEFINE_string("activator", "prelu", "Activator can be [relu, leaky_relu, prelu, sigmoid, tanh]")
flags.DEFINE_float("filters_decay_gamma", 1.5, "Gamma")
flags.DEFINE_boolean("batch_norm", False, "batch normalization")
flags.DEFINE_boolean("pixel_shuffler", False, "Use Pixel Shuffler insted of using transposed CNN")
flags.DEFINE_integer("self_ensemble", 8, "Number of using self ensemble method. [1 - 8]")


# Training
flags.DEFINE_string("initializer", "he", "Initializer for weights can be [uniform, stddev, xavier, he, identity, zero]")
flags.DEFINE_float("weight_dev", 0.01, "Initial weight stddev (won't be used when you use he or xavier initializer)")
flags.DEFINE_float("l2_decay", 0.001, "l2_decay")
flags.DEFINE_string("optimizer", "adam", "Optimizer can be [gd, momentum, adadelta, adagrad, adam, rmsprop]")
flags.DEFINE_float("beta1", 0.1, "Beta1 for adam optimizer")
flags.DEFINE_float("beta2", 0.1, "Beta2 for adam optimizer")
flags.DEFINE_float("momentum", 0.9, "Momentum for momentum optimizer and rmsprop optimizer")
flags.DEFINE_integer("batch_num", 20, "Number of mini-batch images for training")
flags.DEFINE_integer("batch_image_size", 32, "Image size for mini-batch")
flags.DEFINE_integer("stride_size", 0, "Stride size for mini-batch. If it is 0, use half of batch_image_size")
flags.DEFINE_float("clipping_norm", 5, "Norm for gradient clipping")

# Learning Rate Control for Training
flags.DEFINE_float("initial_lr", 0.002, "Initial learning rate")
flags.DEFINE_float("lr_decay", 0.5, "Learning rate decay rate when it does not reduced during specific epoch")
flags.DEFINE_integer("lr_decay_epoch", 10, "")
flags.DEFINE_float("end_lr", 2e-5, "Training end learning rate (2e-5")

# Dataset or Others
flags.DEFINE_string("test_dataset", "set5", "Directory for test dataset used during training [set5, set14, bsd100, urban100]")
flags.DEFINE_string("evaluate_dataset", "all", "Directory for evaluate dataset [set5, set14, bsd100, urban100, all]")
flags.DEFINE_string("dataset", "yang91", "Training dataset dir. [yang91, general100, bsd200]")
flags.DEFINE_integer("tests", 1, "Number of training tests")

# Image Processing
flags.DEFINE_integer("scale", 2, "Scale factor for Super Resolution (can be 2 or more)")
flags.DEFINE_float("max_value", 255, "For normalize image pixel value")
flags.DEFINE_integer("channels", 1, "Number of image channels used. Use only Y of YCbCr when channels=1.")
flags.DEFINE_boolean("jpeg_mode", False, "Turn on or off jpeg mode when converting from rgb to ycbcr")

# Environment (all directory name should not contain '/' after )
flags.DEFINE_string("checkpoint_dir", "models", "Directory for checkpoints")
flags.DEFINE_string("graph_dir", "graphs", "Directory for graphs")
flags.DEFINE_string("data_dir", "data", "Directory for original images")
flags.DEFINE_string("batch_dir", "batch_data", "Directory for training batch images")
flags.DEFINE_string("output_dir", "output", "Directory for output test images")
flags.DEFINE_string("tf_log_dir", "tf_log", "Directory for tensorboard log")
flags.DEFINE_string("log_filename", "log.txt", "log filename")
flags.DEFINE_string("model_name", "", "model name for save files and tensorboard log")
flags.DEFINE_string("load_model_name", "", "Filename of model loading before start [filename or 'default']")

# Debugging or Logging
flags.DEFINE_boolean("debug", False, "Display each calculated MSE and weight variables")
flags.DEFINE_boolean("initialise_tf_log", True, "Clear all tensorboard log before start")
flags.DEFINE_boolean("save_loss", True, "Save loss")
flags.DEFINE_boolean("save_weights", True, "Save weights and biases")
flags.DEFINE_boolean("save_images", False, "Save CNN weights as images")
flags.DEFINE_integer("save_images_num", 10, "Number of CNN images saved")
flags.DEFINE_boolean("save_meta_data", False, "")


def get():
	print("Python Interpreter version:%s" % sys.version[:3])
	print("tensorflow version:%s" % tf.__version__)
	print("numpy version:%s" % np.__version__)

	# check which library you are using
	# np.show_config()
	return FLAGS
