"""
SR ResNet
"""

import logging
import os
import random
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from helper import loader, utilty as util

#todo: put this in arg
RESAMPLING_METHOD = "bicubic"
MODEL_NAME = "dcscn"



class SuperResolution:
	def __init__(self, flags, model_name=""):

		# Model Parameters
		self.filters = flags.filters
		self.min_filters = flags.min_filters
		self.cnn_size = flags.cnn_size
		self.last_cnn_size = flags.last_cnn_size
		self.last_layers = flags.last_layers
		self.last_filters = flags.last_filters
		self.cnn_stride = 1
		self.layers = 1 + 2 * ((flags.layers - 1) // 2)
		self.self_ensemble = flags.self_ensemble

		self.dropout = flags.dropout
		self.activator = flags.activator
		self.filters_decay_gamma = flags.filters_decay_gamma
		self.batch_norm = flags.batch_norm

		# needed?
		self.nin_filters = flags.nin_filters
		self.nin_filters2 = flags.nin_filters2 if flags.nin_filters2 != 0 else flags.nin_filters // 2

		# Training Parameters
		self.initializer = flags.initializer
		self.weight_dev = flags.weight_dev
		self.l2_decay = flags.l2_decay
		self.optimizer = flags.optimizer
		self.beta1 = flags.beta1
		self.beta2 = flags.beta2
		self.momentum = flags.momentum
		self.batch_num = flags.batch_num
		self.batch_image_size = flags.batch_image_size
		if flags.stride_size == 0:
			self.stride_size = flags.batch_image_size // 2
		else:
			self.stride_size = flags.stride_size
		self.clipping_norm = flags.clipping_norm

		# Learning Rate Control for Training
		self.initial_lr = flags.initial_lr
		self.lr_decay = flags.lr_decay
		self.lr_decay_epoch = flags.lr_decay_epoch

		# Dataset or Others
		self.dataset = flags.dataset
		self.test_dataset = flags.test_dataset
		self.training_data_num = (flags.training_data_num // flags.batch_num) * flags.batch_num

		# Image Processing Parameters
		self.scale = flags.scale
		self.input_max = flags.input_max
		self.output_max = flags.output_max
		self.output_min = flags.output_min
		self.channels = flags.channels
		self.jpeg_mode = flags.jpeg_mode
		self.output_channels = 1  # self.scale * self.scale
		self.resampling_method = RESAMPLING_METHOD

		# Environment (all directory name should not contain '/' after )
		self.checkpoint_dir = flags.checkpoint_dir
		self.tf_log_dir = flags.tf_log_dir
		self.batch_dir = flags.batch_dir

		# Debugging or Logging
		self.debug = flags.debug
		self.save_loss = flags.save_loss
		self.save_weights = flags.save_weights
		self.save_images = flags.save_images
		self.save_images_num = flags.save_images_num
		self.save_meta_data = flags.save_meta_data
		self.log_weight_image_num = 32

		# initialize variables
		self.name = self.get_model_name(model_name)
		self.receptive_fields = 0
		self.complexity = 0
		self.pix_per_input = 1

		# initialize environment
		util.make_dir(self.checkpoint_dir)
		util.make_dir(flags.graph_dir)
		util.make_dir(self.tf_log_dir)
		if flags.initialise_tf_log:
			util.clean_dir(self.tf_log_dir)

		log_filename = util.get_from_ini("local_settings.ini", "log", "filename", "log.txt")
		util.set_logging(log_filename, stream_log_level=logging.INFO, file_log_level=logging.INFO,
		                 tf_log_level=tf.logging.WARN)
		self.init_session()
		self.init_train_step()

		logging.info("\n[%s] -------------------------------------" % MODEL_NAME)
		logging.info(flags.log_header)
		logging.info("%s [%s]" % (util.get_now_date(), self.name))

	def init_session(self):
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = False

		print("Initialised session and graph")
		self.sess = tf.InteractiveSession(config=config, graph=tf.Graph())

	def get_model_name(self, model_name, name_postfix=""):
		if model_name is "":
			name = "%s_L%d_F%d" % (MODEL_NAME, self.layers, self.filters)
			# if self.min_filters != 0:
			# 	name += "to%d" % self.min_filters
			# if self.filters_decay_gamma != 1.0:
			# 	name += "_G%2.2f" % self.filters_decay_gamma
			if self.cnn_size != 3:
				name += "_C%d" % self.cnn_size
			if self.scale != 2:
				name += "_Sc%d" % self.scale
			if self.dropout != 1.0:
				name += "_D%0.2f" % self.dropout
			if self.activator != "relu":
				name += "_%s" % self.activator
			if self.dataset != "yang91":
				name += "_" + self.dataset
			if self.batch_image_size != 32:
				name += "_B%d" % self.batch_image_size
			if self.clipping_norm > 0:
				name += "_cn%.1f" % self.clipping_norm
			# if self.batch_norm:
			# 	name += "_BN"
			if name_postfix is not "":
				name += "_" + name_postfix
		else:
			name = "%s_%s" % (MODEL_NAME, model_name)

		return name

	def load_datasets(self, target, data_dir, batch_dir, batch_image_size, stride_size=0):

		batch_dir += "/scale%d" % self.scale
		print("Loading datasets for [%s]..." % target)
		util.make_dir(batch_dir)

		if stride_size == 0:
			stride_size = batch_image_size // 2

		datasets = loader.DataSets(self.scale, batch_image_size, stride_size, channels=self.channels, jpeg_mode=self.jpeg_mode)

		if not datasets.is_batch_exist(batch_dir):
			datasets.build_batch(data_dir, batch_dir)

		if target == "training":
			datasets.load_batch_image_count(batch_dir)
			self.train = datasets
		else:
			datasets.load_batch(batch_dir)
			self.test = datasets

			for i in range(self.test.input.count):
				self.test.input.images[i] = util.scale(self.test.input.images[i], 0, 255, 0, self.input_max)
				self.test.input.quad_images[i] = util.scale(self.test.input.quad_images[i], 0, 255, self.output_min, self.output_max)
				self.test.true.quad_images[i] = util.scale(self.test.true.quad_images[i], 0, 255, self.output_min, self.output_max)

	def open_datasets(self, target, data_dir, batch_image_size, stride_size=0):

		if stride_size == 0:
			stride_size = batch_image_size // 2

		datasets = loader.DataSets(self.scale, batch_image_size, stride_size, channels=self.channels,
		                           jpeg_mode=self.jpeg_mode)

		datasets.data_dir = data_dir
		datasets.true_filenames = util.get_files_in_directory(data_dir)
		datasets.input.count = len(datasets.true_filenames)

		if target == "training":
			self.train = datasets
		else:
			self.test = datasets

	def build_training_datasets(self, data_dir, batch_dir, batch_image_size, stride_size=0):

		print("Building datasets for [%s]..." % "train")
		util.make_dir(batch_dir)

		if stride_size == 0:
			stride_size = batch_image_size // 2

		self.train = loader.DataSets(self.scale, batch_image_size, stride_size, channels=self.channels,
		                             jpeg_mode=self.jpeg_mode)

		if not self.train.is_batch_exist(batch_dir):
			self.train.build_batch(data_dir, batch_dir)

	def init_epoch_index(self):

		# allocate batch memory
		self.batch_input = self.batch_num * [None]
		self.batch_input_quad = self.batch_num * [None]
		self.batch_true_quad = self.batch_num * [None]

		self.index_in_epoch = 0
		self.training_psnr_sum = 0
		self.training_mse_sum = 0
		self.training_step = 0

	def build_conv(self, name, input_tensor, cnn_size, input_feature_num, output_feature_num, use_bias=True,
	               activator=None, use_batch_norm=False, use_dropout=True):

		with tf.variable_scope(name):
			w = util.weight([cnn_size, cnn_size, input_feature_num, output_feature_num],
			                stddev=self.weight_dev, name="conv_W", initializer=self.initializer)
			if use_bias:
				b = util.bias([output_feature_num], name="conv_B")
			else:
				b = None
			h = self.conv2d(input_tensor, w, self.cnn_stride, bias=b, batch_norm=use_batch_norm, name=name)

			if activator is not None:
				self.build_activator(h, output_feature_num, activator, name=name)
				if use_dropout and self.dropout != 1.0:
					self.H[-1] = tf.nn.dropout(self.H[-1], self.dropout_input, name="dropout")
			else:
				if use_dropout and self.dropout != 1.0:
					h = tf.nn.dropout(h, self.dropout_input, name="dropout")
				self.H.append(h)

			if self.save_weights:
				util.add_summaries("weight", self.name, w, save_stddev=True, save_mean=True)
				util.add_summaries("output", self.name, h, save_stddev=True, save_mean=True)
				if use_bias:
					util.add_summaries("bias", self.name, b, save_stddev=True, save_mean=True)

			if self.save_images and cnn_size > 1 and input_feature_num == 1:
				weight_transposed = tf.transpose(w, [3, 0, 1, 2])
				with tf.name_scope("image"):
					tf.summary.image(self.name, weight_transposed, max_outputs=self.log_weight_image_num)

		if self.receptive_fields == 0:
			self.receptive_fields = cnn_size
		else:
			self.receptive_fields += (cnn_size - 1)

		self.Weights.append(w)
		if use_bias:
			self.Biases.append(b)

		self.features += "%d " % output_feature_num

	def build_transposed_conv(self, name, input_tensor, scale, channels):
		with tf.variable_scope(name):
			w = util.upscale_weight(scale=scale, channels=channels, name="Tconv_W")

			batch_size = tf.shape(input_tensor)[0]
			height = tf.shape(input_tensor)[1] * scale
			width = tf.shape(input_tensor)[2] * scale

			h = tf.nn.conv2d_transpose(input_tensor, w, output_shape=[batch_size, height, width, channels],
			                           strides=[1, scale, scale, 1], name=name)

		self.pix_per_input *= scale * scale
		self.complexity += self.pix_per_input * util.get_upscale_filter_size(scale) * util.get_upscale_filter_size(
			scale) * channels * channels
		self.receptive_fields += 1

		self.Weights.append(w)
		self.H.append(h)
		return h

	def build_input_batch(self, batch_dir):

		for i in range(self.batch_num):
			image_no = random.randrange(self.train.input.count)
			image = loader.load_random_patch(self.train.true_filenames[image_no], self.batch_image_size * self.scale,
			                                 self.batch_image_size * self.scale, self.jpeg_mode)

			while image is None:
				image_no = random.randrange(self.train.input.count)
				image = loader.load_random_patch(self.train.true_filenames[image_no], self.batch_image_size * self.scale,
				                                 self.batch_image_size * self.scale, self.jpeg_mode)

			if random.randrange(2) == 0:
				image = np.fliplr(image)

			# print("[%d,%d-%d,%d,%s]" % (x, y, x + self.batch_image_size * self.scale, y + self.batch_image_size * self.scale,
			#                             self.train.true_filenames[image_no]))

			# util.save_image("output/%d_input.png" % i, util.resize_image_by_pil(true_image, 1 / self.scale))
			# util.save_image("output/%d_true.png" % i, true_image)

			self.batch_input[i] = util.scale(util.resize_image_by_pil(image, 1 / self.scale), 0, 255, 0, self.input_max)
			self.batch_input_quad[i] = util.scale(util.resize_image_by_pil(self.batch_input[i], self.scale), 0, 255, self.output_min, self.output_max)
			self.batch_true_quad[i] = util.scale(image, 0, 255, self.output_min, self.output_max)
			self.index_in_epoch += 1

	def build_graph(self, use_dropout=True):

		self.Weights = []
		self.Biases = []
		self.H = []
		self.features = ""

		input_feature_num = self.channels

		self.x = tf.placeholder(tf.float32, shape=[None, None, None, input_feature_num], name="x")
		self.y = tf.placeholder(tf.float32, shape=[None, None, None, self.output_channels], name="y")
		self.x2 = tf.placeholder(tf.float32, shape=[None, None, None, self.output_channels], name="x2")
		self.dropout_input = tf.placeholder(tf.float32, shape=[], name="dropout_keep_rate")
		self.is_training = tf.placeholder(tf.bool, name="training_phase")

		output_feature_num = self.layers * [0]
		total_output_feature_num = 0
		input_feature_num = self.channels
		input_tensor = self.x

		for i in range(self.layers):
			if self.min_filters != 0:
				if i == 0:
					output_feature_num[i] = self.filters
				else:
					x1 = i / float(self.layers - 1)
					y1 = pow(x1, 1.0 / self.filters_decay_gamma)
					output_feature_num[i] = int((self.filters - self.min_filters) * (1 - y1) + self.min_filters)
			else:
				output_feature_num[i] = self.filters
			total_output_feature_num += output_feature_num[i]

			self.build_conv("conv%d" % i, input_tensor, self.cnn_size,
			                                     input_feature_num,
			                                     output_feature_num[i],
			                                     use_batch_norm=self.batch_norm,
			                                     use_dropout=use_dropout)
			input_feature_num = output_feature_num[i]
			input_tensor = self.H[-1]

		with tf.variable_scope("concat"):
			self.H_concat = tf.concat(self.H, 3, name="H_concat")

		# building reconstruction layers ---

		self.build_conv("A1", self.H_concat, 1, total_output_feature_num, self.nin_filters,
		                                     use_dropout=use_dropout)
		self.receptive_fields -= (self.cnn_size - 1)

		self.build_conv("B1", self.H_concat, 1, total_output_feature_num, self.nin_filters2,
		                                     use_dropout=use_dropout)

		self.H_B2 = self.build_conv("B2", self.H[-1], 3, self.nin_filters2, self.nin_filters2,
		                                     use_dropout=use_dropout)

		#				self.H_concat2 = tf.concat([self.H[0], self.H2[0], self.H2[1]], 3, name="H_concat2")
		h = tf.concat([self.H[-1], self.H[-3]], 3, name="H_concat2")
		upsampling_layer_input_channels = self.nin_filters + self.nin_filters2

		# building upsampling layer
		self.build_transposed_conv("Up", h, self.scale, upsampling_layer_input_channels)

		input_channels = upsampling_layer_input_channels
		for i in range(self.last_layers):
			self.build_conv("L%d" % (i + 1), self.H[-1], self.last_cnn_size, input_channels, self.last_filters,
			                             use_dropout=use_dropout)
			input_channels = self.last_filters

		self.build_conv("F", self.H[-1], self.last_cnn_size, input_channels, self.output_channels)

		self.y_ = self.H[-1] + self.x2
		self.weights = self.Weights

		logging.info("Feature:%s Complexity:%s Receptive Fields:%d" % (
			self.features, "{:,}".format(self.complexity), self.receptive_fields))

	def build_graph_bn(self):

		# building feature extraction layers
		self.Weights = []
		self.Biases = []
		self.H = []
		self.features = ""

		input_feature_num = self.channels
		if self.dropout == 1.0:
			use_dropout = False
		else:
			use_dropout = True

		self.x = tf.placeholder(tf.float32, shape=[None, None, None, input_feature_num], name="x")
		self.y = tf.placeholder(tf.float32, shape=[None, None, None, self.output_channels], name="y")
		self.dropout_input = tf.placeholder(tf.float32, shape=[], name="dropout_keep_rate")
		self.is_training = tf.placeholder(tf.bool, name="training_phase")

		i = 0
		self.build_conv("conv%d" % i, self.x, self.cnn_size, self.channels, self.filters, use_batch_norm=self.batch_norm,
		                activator=self.activator, use_dropout=use_dropout, use_bias=True)

		for i in range(1, self.layers, 2):
			self.build_conv("conv%d" % i, self.H[-1], self.cnn_size, self.filters, self.filters,
			                use_batch_norm=self.batch_norm, activator=self.activator, use_dropout=use_dropout,
			                use_bias=False)

			self.build_conv("conv%d" % (i + 1), self.H[-1], self.cnn_size, self.filters, self.filters,
			                use_batch_norm=self.batch_norm, use_dropout=use_dropout, use_bias=False)
			self.H[-1] = self.H[-1] * 0.1

			self.H[-1] = self.H[-1] + self.H[-1 - 2]

		# build final - residual block layer
		self.build_conv("conv%d" % (self.layers - 1 + 1), self.H[-1], self.cnn_size, self.filters, self.filters,
		                use_batch_norm=False, use_dropout=False, use_bias=False)

		self.H[-1] = self.H[-1] + self.x

		# building upsampling layer
		self.build_upsampling_layer(self.H[-1])

		# final convolutional layer
		self.build_conv("conv%d" % (len(self.Weights)), self.H[-1], 9,
		                self.filters, 1, use_batch_norm=False, use_dropout=False, use_bias=True)

		self.y_ = self.H[-1]

		logging.info("Feature:%s Complexity:%s Receptive Fields:%d" % (
			self.features, "{:,}".format(self.complexity), self.receptive_fields))

	def build_upsampling_layer(self, h):

		if self.scale == 2 or self.scale == 4:
			self.build_conv("up_conv_1", h, self.cnn_size, self.filters, 2 * 2 * self.filters,
			                use_batch_norm=False, use_dropout=False, use_bias=True)

			self.H.append(tf.depth_to_space(self.H[-1], 2))

			self.build_activator(self.H[-1], self.filters, self.activator, "prelu_1")

			if self.scale == 4:
				self.build_conv("up_conv_2", self.H[-1], self.cnn_size, self.filters, 2 * 2 * self.filters,
				                use_batch_norm=False, use_dropout=False, use_bias=True)

				self.H.append(tf.depth_to_space(self.H[-1], 2))

				self.build_activator(self.H[-1], self.filters, self.activator(), "prelu_2")

		if self.scale == 3:
			self.build_conv("up_conv_1", self.H[-1], self.cnn_size, self.filters, 3 * 3 * self.filters,
			                use_batch_norm=False, use_dropout=False, use_bias=True)

			self.H.append(tf.depth_to_space(self.H[-1], 3))

			self.build_activator(self.H[-1], self.filters, self.activator, "prelu_1")

	def build_activator(self, hidden, features, activator=None, leaky_relu_alpha=0.1, name=""):
		if activator == "relu":
			hidden = tf.nn.relu(hidden, name=name + "_relu")
		elif activator == "sigmoid":
			hidden = tf.nn.sigmoid(hidden, name=name + "_sigmoid")
		elif activator == "tanh":
			hidden = tf.nn.tanh(hidden, name=name + "_tanh")
		elif activator == "leaky_relu":
			hidden = tf.maximum(hidden, leaky_relu_alpha * hidden, name=name + "_leaky")
		elif activator == "prelu":
			with tf.variable_scope("prelu"):
				alphas = tf.Variable(tf.constant(0.1, shape=[features]), name=name + "_prelu")
				if self.save_weights:
					util.add_summaries("prelu_alpha", self.name, alphas, save_stddev=False, save_mean=False)
					hidden = tf.nn.relu(hidden) + tf.multiply(alphas, (hidden - tf.abs(hidden))) * 0.5
		else:
			raise NameError('Not implemented activator:%s' % activator)

		self.complexity += self.pix_per_input * features
		self.H.append(hidden)

	def conv2d(self, x, w, stride, bias=None, batch_norm=False, name=""):

		output = tf.nn.conv2d(x, w, strides=[stride, stride, 1, 1], padding="SAME", name=name + "_conv")
		self.complexity += self.pix_per_input * int(w.shape[0] * w.shape[1] * w.shape[2] * w.shape[3])

		if bias is not None:
			output = tf.add(output, bias, name=name + "_add")
			self.complexity += self.pix_per_input * int(bias.shape[0])

		if batch_norm:
			output = tf.layers.batch_normalization(output, training=self.is_training, name='BN')

		return output

	def build_optimizer(self):

		self.lr_input = tf.placeholder(tf.float32, shape=[], name="LearningRate")

		diff = self.y_ - self.y

		mse = tf.reduce_mean(tf.square(diff), name="mse")

		if self.debug:
			mse = tf.Print(mse, [mse], message="MSE: ")

		loss = mse

		if self.l2_decay > 0:
			l2_losses = [tf.nn.l2_loss(w) for w in self.Weights]
			#			l2_losses = [tf.reduce_sum(tf.abs(w)) for w in self.weights]
			l2_loss = self.l2_decay * tf.add_n(l2_losses)
			loss += l2_loss
			if self.save_loss:
				tf.summary.scalar("loss_l2/" + self.name, l2_loss)

		if self.save_loss:
			tf.summary.scalar("loss/" + self.name, loss)

		self.loss = loss
		self.mse = mse

		self.training_optimizer = self.add_optimizer_op(loss, self.lr_input)

		util.print_num_of_total_parameters(output_detail=True)

	def get_psnr_tensor(self, mse):

		with tf.variable_scope('get_PSNR'):
			value = tf.constant(self.output_max - self.output_min, dtype=mse.dtype) / tf.sqrt(mse)
			numerator = tf.log(value)
			denominator = tf.log(tf.constant(10, dtype=mse.dtype))
			return tf.constant(20, dtype=mse.dtype) * numerator / denominator

	def add_optimizer_op(self, loss, lr_input):

		if self.optimizer == "gd":
			optimizer = tf.train.GradientDescentOptimizer(lr_input)
		elif self.optimizer == "adadelta":
			optimizer = tf.train.AdadeltaOptimizer(lr_input)
		elif self.optimizer == "adagrad":
			optimizer = tf.train.AdagradOptimizer(lr_input)
		elif self.optimizer == "adam":
			optimizer = tf.train.AdamOptimizer(lr_input, beta1=self.beta1, beta2=self.beta2)
		elif self.optimizer == "momentum":
			optimizer = tf.train.MomentumOptimizer(lr_input, self.momentum)
		elif self.optimizer == "rmsprop":
			optimizer = tf.train.RMSPropOptimizer(lr_input, momentum=self.momentum)
		else:
			print("Optimizer arg should be one of [gd, adadelta, adagrad, adam, momentum, rmsprop].")
			return None

		if self.clipping_norm > 0:
			trainables = tf.trainable_variables()
			grads = tf.gradients(loss, trainables)
			grads, _ = tf.clip_by_global_norm(grads, clip_norm=self.clipping_norm)
			grad_var_pairs = zip(grads, trainables)

			training_optimizer = optimizer.apply_gradients(grad_var_pairs)
		else:
			training_optimizer = optimizer.minimize(loss)

		return training_optimizer

	def build_summary_saver(self):
		if self.save_loss or self.save_weights or self.save_meta_data:
			self.summary_op = tf.summary.merge_all()
			self.train_writer = tf.summary.FileWriter(self.tf_log_dir + "/train")
			self.test_writer = tf.summary.FileWriter(self.tf_log_dir + "/test", graph=self.sess.graph)

		self.saver = tf.train.Saver(max_to_keep=None)

	def init_all_variables(self):
		self.sess.run(tf.global_variables_initializer())
		print("Model Initialised.")

	def load_model(self, name="", trial=0, output_log=False):

		if name == "" or name == "default":
			filename = self.checkpoint_dir + "/" + self.name + "_" + str(trial) + ".ckpt"
		else:
			filename = self.checkpoint_dir + "/" + name + ".ckpt"

		if not os.path.isfile(filename + ".index"):
			print("Error. [%s] is not exist!" % filename)
			exit(-1)

		print("Model restoring... [ %s ]." % filename)  ##

		self.saver.restore(self.sess, filename)
		if output_log:
			logging.info("Model restored [ %s ]." % filename)
		else:
			print("Model restored [ %s ]." % filename)

	def save_model(self, name="", trial=0, output_log=False):

		if name == "":
			filename = self.checkpoint_dir + "/" + self.name + "_" + str(trial) + ".ckpt"
		else:
			filename = self.checkpoint_dir + "/" + name + ".ckpt"

		self.saver.save(self.sess, filename)
		if output_log:
			logging.info("Model saved [%s]." % filename)
		else:
			print("Model saved [%s]." % filename)

	def train_batch(self):

		feed_dict = {self.x: self.batch_input, self.x2: self.batch_input_quad, self.y: self.batch_true_quad,
		             self.lr_input: self.lr, self.dropout_input: self.dropout, self.is_training: True}
		extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		mse, _, _ = self.sess.run([self.mse, self.training_optimizer, extra_update_ops], feed_dict=feed_dict)

		self.training_mse_sum += mse
		self.training_psnr_sum += util.get_psnr(mse, max_value=self.output_max - self.output_min)
		self.training_step += 1
		self.step += 1

	def evaluate(self, save_meta_data=False, trial=0, logging=True):

		save_meta_data = save_meta_data and self.save_meta_data and (trial == 0)
		feed_dict = {self.x: self.test.input.images,
		             self.y: self.test.true.quad_images,
		             self.x2: self.test.input.quad_images,
		             self.dropout_input: 1.0,
		             self.is_training: False}

		if logging == True and (self.save_loss or self.save_weights or save_meta_data):

			if save_meta_data:
				# profiler = tf.profiler.Profile(self.sess.graph)

				run_metadata = tf.RunMetadata()
				run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
				summary_str, mse = self.sess.run([self.summary_op, self.mse], feed_dict=feed_dict, options=run_options,
				                                 run_metadata=run_metadata)
				self.test_writer.add_run_metadata(run_metadata, "step%d" % self.epochs_completed)

				filename = self.checkpoint_dir + "/" + self.name + "_metadata.txt"
				with open(filename, "w") as out:
					out.write(str(run_metadata))

				# filename = self.checkpoint_dir + "/" + self.name + "_memory.txt"
				# tf.profiler.write_op_log(
				# 	tf.get_default_graph(),
				# 	log_dir=self.checkpoint_dir,
				# 	#op_log=op_log,
				# 	run_meta=run_metadata)

				tf.contrib.tfprof.model_analyzer.print_model_analysis(
					tf.get_default_graph(), run_meta=run_metadata,
					tfprof_options=tf.contrib.tfprof.model_analyzer.PRINT_ALL_TIMING_MEMORY)

			else:
				summary_str, mse = self.sess.run([self.summary_op, self.mse], feed_dict=feed_dict)

			self.train_writer.add_summary(summary_str, self.epochs_completed)
			if self.training_step > 0:
				util.log_scalar_value(self.train_writer, 'training_PSNR', self.training_psnr_sum / self.training_step,
				                      self.epochs_completed)
			util.log_scalar_value(self.train_writer, 'LR', self.lr, self.epochs_completed)
			self.train_writer.flush()

			util.log_scalar_value(self.test_writer, 'PSNR', util.get_psnr(mse), self.epochs_completed)
			self.test_writer.flush()
		else:
			mse = self.sess.run(self.mse, feed_dict=feed_dict)

		return mse

	# def evaluate(self, save_meta_data=False):
	#
	# 	save_meta_data = save_meta_data and self.save_meta_data
	# 	test_index_in_epoch = 0
	# 	total_mse = 0;
	# 	test_count = 0
	#
	# 	while True:
	# 		for i in range(self.batch_num):
	# 			if test_index_in_epoch >= self.test.input.count:
	# 				return total_mse / test_count
	#
	# 			self.batch_input[i] = self.test.input.images[test_index_in_epoch]
	# 			self.batch_input_quad[i] = self.test.input.quad_images[test_index_in_epoch]
	# 			self.batch_true_quad[i] = self.test.true.quad_images[test_index_in_epoch]
	# 			test_index_in_epoch += 1
	#
	# 		feed_dict = {self.x: self.batch_input,
	# 		             self.x2: self.batch_input_quad,
	# 		             self.y: self.batch_true_quad,
	# 		             self.dropout_input: 1.0,
	# 		             self.phase: 0}
	#
	# 		if self.save_loss or self.save_weights or save_meta_data:
	# 			if save_meta_data:
	# 				run_metadata = tf.RunMetadata()
	# 				run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
	# 				summary_str, mse = self.sess.run([self.summary_op, self.mse], feed_dict=feed_dict, options=run_options,
	# 				                                 run_metadata=run_metadata)
	# 				self.test_writer.add_run_metadata(run_metadata, "step%d" % self.epochs_completed)
	# 			else:
	# 				summary_str, mse = self.sess.run([self.summary_op, self.mse], feed_dict=feed_dict)
	#
	# 			self.test_writer.add_summary(summary_str, self.epochs_completed)
	# 			self.test_writer.flush()
	# 		else:
	# 			mse = self.sess.run(self.mse, feed_dict=feed_dict)
	#
	# 		total_mse += mse
	# 		test_count += 1
	# 		save_meta_data = False
	#

	def update_epoch_and_lr(self, mse):
		lr_updated = False

		if self.min_validation_mse < 0 or self.min_validation_mse > mse:
			# update new mse
			self.min_validation_epoch = self.epochs_completed
			self.min_validation_mse = mse
		else:
			if self.epochs_completed > self.min_validation_epoch + self.lr_decay_epoch:
				# set new learning rate
				self.lr_updated_lr.append(self.lr)
				self.lr_updated_epoch.append(self.epochs_completed)
				self.lr_updated_psnr.append(util.get_psnr(mse, max_value=self.output_max - self.output_min))

				self.min_validation_epoch = self.epochs_completed
				self.lr *= self.lr_decay
				lr_updated = True

		psnr = util.get_psnr(mse, max_value=self.output_max - self.output_min)
		self.csv_epochs.append(self.epochs_completed)
		self.csv_psnr.append(psnr)
		self.csv_training_psnr.append(self.training_psnr_sum / self.training_step)

		return lr_updated

	def print_status(self, mse, log=False):

		psnr = util.get_psnr(mse, max_value=self.output_max - self.output_min)

		if self.step == 0:
			logging.info("Initial MSE:%f PSNR:%f" % (mse, psnr))
		else:
			processing_time = (time.time() - self.start_time) / self.step
			line_a = "%s Step:%d MSE:%f PSNR:%f (Training PSNR:%0.3f)" % (
				util.get_now_date(), self.step, mse, psnr, self.training_psnr_sum / self.training_step)
			line_b = "Epoch:%d (Step:%s) LR:%f (%2.3fsec/step) Max PSNR:%0.3f" % (
				self.epochs_completed, "{:,}".format(self.step), self.lr, processing_time,
				util.get_psnr(self.min_validation_mse, max_value=self.output_max - self.output_min))
			if log:
				logging.info(line_a)
				logging.info(line_b)
			else:
				print(line_a)
				print(line_b)

	def print_weight_variables(self):

		for bias in self.Biases:
			util.print_filter_biases(bias)

		for weight in self.Weights:
			util.print_filter_weights(weight)

	def save_graphs(self, checkpoint_dir, trial):

		psnr_graph = np.column_stack((self.csv_epochs, self.csv_psnr, self.csv_training_psnr))
		filename = checkpoint_dir + "/" + self.name + ("_%d.csv" % trial)
		np.savetxt(filename, psnr_graph, delimiter=",")

		filename2 = checkpoint_dir + "/" + self.name + ("_%d.png" % trial)
		plt.plot(self.csv_epochs, self.csv_training_psnr, "b", label='Training PSNR')
		plt.plot(self.csv_epochs, self.csv_psnr, "r", label='Test PSNR')
		plt.vlines(self.csv_epochs[-1], 0, self.csv_psnr[-1], color='0.75')
		plt.hlines(self.csv_psnr[-1], 0, self.csv_epochs[-1], color='0.75')
		plt.ylim(ymin=30)

		if trial == 0:
			plt.legend(loc='lower right')
		plt.savefig(filename2)
		plt.savefig("PSNR.png")

		print("Graph saved [%s / %s]." % (filename, filename2))

	def do(self, input_image):

		h, w = input_image.shape[:2]
		channel = input_image.shape[2] if len(input_image.shape) > 2 else 1

		input_image = util.scale(input_image, 0, 255, 0, self.input_max)

		if self.self_ensemble > 1:
			output = np.zeros([self.scale * h, self.scale * w, 1])

			for i in range(self.self_ensemble):
				image = util.flip(input_image, i)
				quad_image = util.resize_image_by_pil(image, self.scale)
				new_h, new_w = image.shape[:2]
				y = self.sess.run(self.y_, feed_dict={self.x: image.reshape(1, new_h, new_w, channel),
				                                      self.x2: quad_image.reshape(1, self.scale * new_h, self.scale * new_w, channel),
				                                      self.dropout_input: 1.0, self.is_training: False})
				restored = util.flip(y[0], i, invert=True)
				output += restored

			output /= self.self_ensemble
		else:
			y = self.sess.run(self.y_, feed_dict={self.x: input_image.reshape(1, h, w, channel),
			                                      self.dropout_input: 1.0, self.is_training: False})
			output = y[0]

		output = util.scale(output, self.output_min, self.output_max, 0, 255)
		return output

	def do_for_file(self, file_path, output_folder="output"):
		""" do super resolution for file """

		filename, extension = os.path.splitext(file_path)
		output_folder += "/"
		org_image = util.load_image(file_path)
		util.save_image(output_folder + file_path, org_image)

		if len(org_image.shape) >= 3 and org_image.shape[2] == 3 and self.channels == 1:
			input_y_image = util.convert_rgb_to_y(org_image, jpeg_mode=self.jpeg_mode)
			scaled_image = util.resize_image_by_pil(input_y_image, self.scale, resampling_method=self.resampling_method)
			util.save_image(output_folder + filename + "_bicubic_y" + extension, scaled_image)
			output_y_image = self.do(input_y_image)
			util.save_image(output_folder + filename + "_result_y" + extension, output_y_image)

			scaled_ycbcr_image = util.convert_rgb_to_ycbcr(
				util.resize_image_by_pil(org_image, self.scale, self.resampling_method),
				jpeg_mode=self.jpeg_mode)
			image = util.convert_y_and_cbcr_to_rgb(output_y_image, scaled_ycbcr_image[:, :, 1:3], jpeg_mode=self.jpeg_mode)
		else:
			scaled_image = util.resize_image_by_pil(org_image, self.scale, resampling_method=self.resampling_method)
			util.save_image(output_folder + filename + "_bicubic_y" + extension, scaled_image)
			image = self.do(org_image)

		util.save_image(output_folder + filename + "_result" + extension, image)
		return 0

	def do_for_evaluate(self, file_path, output_directory="output", output=True, print_console=False):
		""" do for file with GT and evaluate the PSNR """
		# TODO combine to do_for_file()

		filename, extension = os.path.splitext(file_path)
		output_directory += "/" + self.name + "/"
		util.make_dir(output_directory)
		true_image = util.set_image_alignment(util.load_image(file_path, print_console=False), self.scale)

		if true_image.shape[2] == 3 and self.channels == 1:
			input_y_image = loader.build_input_image(true_image, channels=self.channels, scale=self.scale,
			                                         alignment=self.scale, convert_ycbcr=True, jpeg_mode=self.jpeg_mode)
			# for color images
			if output:
				input_bicubic_y_image = util.resize_image_by_pil(input_y_image, self.scale)

				true_ycbcr_image = util.convert_rgb_to_ycbcr(true_image, jpeg_mode=self.jpeg_mode)

				output_y_image = self.do(input_y_image)
				mse = util.compute_mse(true_ycbcr_image[:, :, 0:1], output_y_image, border_size=self.scale)
				loss_image = util.get_loss_image(true_ycbcr_image[:, :, 0:1], output_y_image, border_size=self.scale)

				output_color_image = util.convert_y_and_cbcr_to_rgb(output_y_image, true_ycbcr_image[:, :, 1:3],
				                                                    jpeg_mode=self.jpeg_mode)

				util.save_image(output_directory + file_path, true_image)
				util.save_image(output_directory + filename + "_input" + extension, input_y_image)
				util.save_image(output_directory + filename + "_input_bicubic" + extension, input_bicubic_y_image)
				util.save_image(output_directory + filename + "_true_y" + extension, true_ycbcr_image[:, :, 0:1])
				util.save_image(output_directory + filename + "_result" + extension, output_y_image)
				util.save_image(output_directory + filename + "_result_c" + extension, output_color_image)
				util.save_image(output_directory + filename + "_loss" + extension, loss_image)
			else:
				true_y_image = util.convert_rgb_to_y(true_image, jpeg_mode=self.jpeg_mode)
				output_y_image = self.do(input_y_image)
				mse = util.compute_mse(true_y_image, output_y_image, border_size=self.scale)

		elif true_image.shape[2] == 1 and self.channels == 1:

			# for monochrome images
			input_image = loader.build_input_image(true_image, channels=self.channels, scale=self.scale, alignment=self.scale)
			output_image = self.do(input_image)
			mse = util.compute_mse(true_image, output_image, border_size=self.scale)
			if output:
				util.save_image(output_directory + file_path, true_image)
				util.save_image(output_directory + filename + "_result" + extension, output_image)

		if print_console:
			print("MSE:%f, PSNR:%f" % (mse, util.get_psnr(mse, max_value=self.output_max - self.output_min)))
		return mse

	def init_train_step(self):
		self.lr = self.initial_lr
		self.csv_epochs = []
		self.csv_psnr = []
		self.csv_training_psnr = []
		self.epochs_completed = 0
		self.min_validation_mse = -1
		self.min_validation_epoch = -1
		self.step = 0

		self.lr_updated_lr = []
		self.lr_updated_epoch = []
		self.lr_updated_psnr = []

		self.start_time = time.time()

	def end_train_step(self):
		self.total_time = time.time() - self.start_time

	def print_steps_completed(self, output_to_logging=False):

		if self.step == 0:
			return

		processing_time = self.total_time / self.step
		h = self.total_time // (60 * 60)
		m = (self.total_time - h * 60 * 60) // 60
		s = (self.total_time - h * 60 * 60 - m * 60)

		status = "Finished at Total Epoch:%d Steps:%s Time:%02d:%02d:%02d (%2.3fsec/step) %d x %d x %d patches" % (
			self.epochs_completed, "{:,}".format(self.step), h, m, s, processing_time,
			self.batch_image_size, self.batch_image_size, self.train.input.count)

		if output_to_logging:
			logging.info(status)
		else:
			print(status)

	def log_model_analysis(self):
		run_metadata = tf.RunMetadata()
		run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

		_, mse = self.sess.run([self.optimizer, self.mse], feed_dict={self.x: self.batch_input,
		                                                              self.y: self.batch_true_quad,
		                                                              self.lr_input: self.lr,
		                                                              self.dropout_input: self.dropout},
		                       options=run_options, run_metadata=run_metadata)

		# tf.contrib.tfprof.model_analyzer.print_model_analysis(
		#   tf.get_default_graph(),
		#   run_meta=run_metadata,
		#   tfprof_options=tf.contrib.tfprof.model_analyzer.PRINT_ALL_TIMING_MEMORY)
		self.first_training = False

	def copy_log_to_archive(self, archive_name):

		archive_directory = self.tf_log_dir + '_' + archive_name
		model_archive_directory = archive_directory + '/' + self.name
		util.make_dir(archive_directory)
		util.delete_dir(model_archive_directory)
		try:
			shutil.copytree(self.tf_log_dir, model_archive_directory)
			print("tensorboard log archived to [%s]." % model_archive_directory)
		except OSError as e:
			print(e)
			print("NG: tensorboard log archived to [%s]." % model_archive_directory)

	def report_updated_history(self):

		updated_history = ''
		for i in range(len(self.lr_updated_lr)):
			updated_history += "LR:%f E:%d PSNR:%f, " % (
				self.lr_updated_lr[i], self.lr_updated_epoch[i], self.lr_updated_psnr[i])
		logging.info(updated_history)

		updated_history = 'Efforts:'
		for i in range(len(self.lr_updated_lr) - 1):
			updated_history += "(%f, E:%d [%f]), " % (
				self.lr_updated_lr[i], self.lr_updated_epoch[i + 1] - self.lr_updated_epoch[i],
				self.lr_updated_psnr[i + 1] - self.lr_updated_psnr[i])
		logging.info(updated_history)
