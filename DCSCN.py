"""
Paper: "Fast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Network"
Ver: 2.0

DCSCN model implementation (Transposed-CNN / Pixel Shuffler version)
See Detail: https://github.com/jiny2001/dcscn-super-resolution/

Please note this model is updated version of the paper.
If you want to check original source code and results of the paper, please see https://github.com/jiny2001/dcscn-super-resolution/tree/ver1.
"""

import logging
import math
import os
import time

import numpy as np
import tensorflow as tf

from helper import loader, tf_graph, utilty as util

BICUBIC_METHOD_STRING = "bicubic"


class SuperResolution(tf_graph.TensorflowGraph):
	def __init__(self, flags, model_name=""):

		super().__init__(flags)

		# Model Parameters
		self.scale = flags.scale
		self.layers = flags.layers
		self.filters = flags.filters
		self.min_filters = min(flags.filters, flags.min_filters)
		self.filters_decay_gamma = flags.filters_decay_gamma
		self.use_nin = flags.use_nin
		self.nin_filters = flags.nin_filters
		self.nin_filters2 = flags.nin_filters2
		self.reconstruct_layers = max(flags.reconstruct_layers, 1)
		self.reconstruct_filters = flags.reconstruct_filters
		self.resampling_method = BICUBIC_METHOD_STRING
		self.pixel_shuffler = flags.pixel_shuffler
		self.self_ensemble = flags.self_ensemble

		# Training Parameters
		self.l2_decay = flags.l2_decay
		self.optimizer = flags.optimizer
		self.beta1 = flags.beta1
		self.beta2 = flags.beta2
		self.epsilon = flags.epsilon
		self.momentum = flags.momentum
		self.batch_num = flags.batch_num
		self.batch_image_size = flags.batch_image_size
		if flags.stride_size == 0:
			self.stride_size = flags.batch_image_size // 2
		else:
			self.stride_size = flags.stride_size
		self.clipping_norm = flags.clipping_norm
		self.use_l1_loss = flags.use_l1_loss

		# Learning Rate Control for Training
		self.initial_lr = flags.initial_lr
		self.lr_decay = flags.lr_decay
		self.lr_decay_epoch = flags.lr_decay_epoch

		# Dataset or Others
		self.training_images = int(math.ceil(flags.training_images / flags.batch_num) * flags.batch_num)
		self.train = None
		self.test = None

		# Image Processing Parameters
		self.max_value = flags.max_value
		self.channels = flags.channels
		self.output_channels = 1
		self.psnr_calc_border_size = flags.psnr_calc_border_size
		if self.psnr_calc_border_size < 0:
			self.psnr_calc_border_size = 2 + self.scale

		# Environment (all directory name should not contain tailing '/'  )
		self.batch_dir = flags.batch_dir

		# initialize variables
		self.name = self.get_model_name(model_name)
		self.total_epochs = 0
		lr = self.initial_lr
		while lr > flags.end_lr:
			self.total_epochs += self.lr_decay_epoch
			lr *= self.lr_decay

		# initialize environment
		util.make_dir(self.checkpoint_dir)
		util.make_dir(flags.graph_dir)
		util.make_dir(self.tf_log_dir)
		if flags.initialize_tf_log:
			util.clean_dir(self.tf_log_dir)
		util.set_logging(flags.log_filename, stream_log_level=logging.INFO, file_log_level=logging.INFO,
		                 tf_log_level=tf.logging.WARN)
		logging.info("\nDCSCN v2-------------------------------------")
		logging.info("%s [%s]" % (util.get_now_date(), self.name))

		self.init_train_step()

	def get_model_name(self, model_name, name_postfix=""):
		if model_name is "":
			name = "dcscn_L%d_F%d" % (self.layers, self.filters)
			if self.min_filters != 0:
				name += "to%d" % self.min_filters
			if self.filters_decay_gamma != 1.5:
				name += "_G%2.2f" % self.filters_decay_gamma
			if self.cnn_size != 3:
				name += "_C%d" % self.cnn_size
			if self.scale != 2:
				name += "_Sc%d" % self.scale
			if self.use_nin:
				name += "_NIN"
				if self.nin_filters != 0:
					name += "_A%d" % self.nin_filters
				if self.nin_filters2 != self.nin_filters // 2:
					name += "_B%d" % self.nin_filters2
			if self.pixel_shuffler:
				name += "_PS"
			if self.max_value != 255.0:
				name += "_M%2.1f" % self.max_value
			if self.activator != "prelu":
				name += "_%s" % self.activator
			if self.batch_norm:
				name += "_BN"
			if self.reconstruct_layers >= 1:
				name += "_R%d" % self.reconstruct_layers
				if self.reconstruct_filters != 1:
					name += "F%d" % self.reconstruct_filters
			if name_postfix is not "":
				name += "_" + name_postfix
		else:
			name = "dcscn_%s" % model_name

		return name

	def load_dynamic_datasets(self, data_dir, batch_image_size):
		""" loads datasets
		Opens image directory as a datasets. Images will be loaded when build_input_batch() is called.
		"""

		self.train = loader.DynamicDataSets(self.scale, batch_image_size, channels=self.channels,
		                                    resampling_method=self.resampling_method)
		self.train.set_data_dir(data_dir)

	def load_datasets(self, data_dir, batch_dir, batch_image_size, stride_size=0):
		""" build input patch images and loads as a datasets
		Opens image directory as a datasets.
		Each images are splitted into patch images and converted to input image. Since loading
		(especially from PNG/JPG) and building input-LR images needs much computation in the
		training phase, building pre-processed images makes training much faster. However, images
		are limited by divided grids.
		"""

		batch_dir += "/scale%d" % self.scale

		self.train = loader.BatchDataSets(self.scale, batch_dir, batch_image_size, stride_size, channels=self.channels,
		                                  resampling_method=self.resampling_method)

		if not self.train.is_batch_exist():
			self.train.build_batch(data_dir)
		else:
			self.train.load_batch_counts()
		self.train.load_all_batch_images()

	def init_epoch_index(self):

		self.batch_input = self.batch_num * [None]
		self.batch_input_bicubic = self.batch_num * [None]
		self.batch_true = self.batch_num * [None]

		self.training_psnr_sum = 0
		self.training_loss_sum = 0
		self.training_step = 0
		self.train.init_batch_index()

	def build_input_batch(self):

		for i in range(self.batch_num):
			self.batch_input[i], self.batch_input_bicubic[i], self.batch_true[i] = self.train.load_batch_image()

	def build_graph(self):

		self.x = tf.placeholder(tf.float32, shape=[None, None, None, self.channels], name="x")
		self.y = tf.placeholder(tf.float32, shape=[None, None, None, self.output_channels], name="y")
		self.x2 = tf.placeholder(tf.float32, shape=[None, None, None, self.output_channels], name="x2")
		self.dropout = tf.placeholder(tf.float32, shape=[], name="dropout_keep_rate")
		self.is_training = tf.placeholder(tf.bool, name="is_training")

		# building feature extraction layers

		output_feature_num = self.filters
		total_output_feature_num = 0
		input_feature_num = self.channels
		input_tensor = self.x

		if self.save_weights:
			with tf.name_scope("X"):
				util.add_summaries("output", self.name, self.x, save_stddev=True, save_mean=True)

		for i in range(self.layers):
			if self.min_filters != 0 and i > 0:
				x1 = i / float(self.layers - 1)
				y1 = pow(x1, 1.0 / self.filters_decay_gamma)
				output_feature_num = int((self.filters - self.min_filters) * (1 - y1) + self.min_filters)

			self.build_conv("CNN%d" % (i + 1), input_tensor, self.cnn_size, input_feature_num,
			                output_feature_num, use_bias=True, activator=self.activator,
			                use_batch_norm=self.batch_norm, dropout_rate=self.dropout_rate)
			input_feature_num = output_feature_num
			input_tensor = self.H[-1]
			total_output_feature_num += output_feature_num

		with tf.variable_scope("Concat"):
			self.H_concat = tf.concat(self.H, 3, name="H_concat")
		self.features += " Total: (%d)" % total_output_feature_num

		# building reconstruction layers ---

		if self.use_nin:
			self.build_conv("A1", self.H_concat, 1, total_output_feature_num, self.nin_filters,
			                dropout_rate=self.dropout_rate, use_bias=True, activator=self.activator)
			self.receptive_fields -= (self.cnn_size - 1)

			self.build_conv("B1", self.H_concat, 1, total_output_feature_num, self.nin_filters2,
			                dropout_rate=self.dropout_rate, use_bias=True, activator=self.activator)

			self.build_conv("B2", self.H[-1], 3, self.nin_filters2, self.nin_filters2,
			                dropout_rate=self.dropout_rate, use_bias=True, activator=self.activator)

			self.H.append(tf.concat([self.H[-1], self.H[-3]], 3, name="Concat2"))
			input_channels = self.nin_filters + self.nin_filters2
		else:
			self.H.append(self.H_concat)
			input_channels = total_output_feature_num

		# building upsampling layer
		if self.pixel_shuffler:
			if self.scale == 4:
				self.build_pixel_shuffler_layer("Up-PS", self.H[-1], 2, input_channels)
				self.build_pixel_shuffler_layer("Up-PS2", self.H[-1], 2, input_channels)
			else:
				self.build_pixel_shuffler_layer("Up-PS", self.H[-1], self.scale, input_channels)
		else:
			self.build_transposed_conv("Up-TCNN", self.H[-1], self.scale, input_channels)

		for i in range(self.reconstruct_layers - 1):
			self.build_conv("R-CNN%d" % (i + 1), self.H[-1], self.cnn_size, input_channels, self.reconstruct_filters,
			                dropout_rate=self.dropout_rate, use_bias=True, activator=self.activator)
			input_channels = self.reconstruct_filters

		self.build_conv("R-CNN%d" % self.reconstruct_layers, self.H[-1], self.cnn_size, input_channels,
		                self.output_channels)

		self.y_ = self.H[-1] + self.x2

		if self.save_weights:
			with tf.name_scope("Y_"):
				util.add_summaries("output", self.name, self.y_, save_stddev=True, save_mean=True)

		logging.info("Feature:%s Complexity:%s Receptive Fields:%d" % (
			self.features, "{:,}".format(self.complexity), self.receptive_fields))

	def build_optimizer(self):
		"""
		Build loss function. We use 6+scale as a border	and we don't calculate MSE on the border.
		"""

		self.lr_input = tf.placeholder(tf.float32, shape=[], name="LearningRate")

		diff = self.y_ - self.y

		if self.use_l1_loss:
			self.mse = tf.reduce_mean(tf.square(diff), name="mse")
			self.loss = tf.reduce_mean(tf.abs(diff), name="loss")
		else:
			self.mse = tf.reduce_mean(tf.square(diff), name="mse")
			self.loss = self.mse

		if self.l2_decay > 0:
			l2_norm_losses = [tf.nn.l2_loss(w) for w in self.Weights]
			# l1_norm_losses = [tf.reduce_sum(tf.abs(w)) for w in self.weights]  # l1 loss
			l2_norm_loss = self.l2_decay * tf.add_n(l2_norm_losses)
			if self.enable_log:
				tf.summary.scalar("L2WeightDecayLoss/" + self.name, l2_norm_loss)

			self.total_loss = self.loss + l2_norm_loss
		else:
			self.total_loss = self.loss

		if self.enable_log:
			tf.summary.scalar("Loss/" + self.name, self.total_loss)

		if self.batch_norm:
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				self.training_optimizer = self.add_optimizer_op(self.total_loss, self.lr_input)
		else:
			self.training_optimizer = self.add_optimizer_op(self.total_loss, self.lr_input)

		util.print_num_of_total_parameters(output_detail=True)

	def get_psnr_tensor(self, mse):

		with tf.variable_scope('get_PSNR'):
			value = tf.constant(self.max_value, dtype=mse.dtype) / tf.sqrt(mse)
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
			optimizer = tf.train.AdamOptimizer(lr_input, beta1=self.beta1, beta2=self.beta2, epsilon=self.epsilon)
		elif self.optimizer == "momentum":
			optimizer = tf.train.MomentumOptimizer(lr_input, self.momentum)
		elif self.optimizer == "rmsprop":
			optimizer = tf.train.RMSPropOptimizer(lr_input, momentum=self.momentum)
		else:
			print("Optimizer arg should be one of [gd, adadelta, adagrad, adam, momentum, rmsprop].")
			return None

		if self.clipping_norm > 0 or self.enable_log:
			trainables = tf.trainable_variables()
			grads = tf.gradients(loss, trainables)
			grads, _ = tf.clip_by_global_norm(grads, clip_norm=self.clipping_norm)
			grad_var_pairs = zip(grads, trainables)
			training_optimizer = optimizer.apply_gradients(grad_var_pairs)

			if self.enable_log:
				for i in range(len(grads)):
					mean = tf.reduce_mean(tf.abs(grads[i]))
					tf.summary.scalar("Grad%02d/mean/%s" % (i, self.name), mean)
					max_grad = tf.reduce_max(tf.abs(grads[i]))
					tf.summary.scalar("Grad%02d/max/%s" % (i, self.name), max_grad)
		else:
			training_optimizer = optimizer.minimize(loss)

		return training_optimizer

	def train_batch(self):

		feed_dict = {self.x: self.batch_input, self.x2: self.batch_input_bicubic, self.y: self.batch_true,
		             self.lr_input: self.lr, self.dropout: self.dropout_rate, self.is_training: 1}

		if self.use_l1_loss:
			_, loss = self.sess.run([self.training_optimizer, self.loss], feed_dict=feed_dict)
			self.training_loss_sum += loss
		else:
			_, mse = self.sess.run([self.training_optimizer, self.mse], feed_dict=feed_dict)
			self.training_loss_sum += mse
			self.training_psnr_sum += util.get_psnr(mse, max_value=self.max_value)

		self.training_step += 1
		self.step += 1

	def log_to_tensorboard(self, test_filename, psnr, save_meta_data=True):

		# todo
		save_meta_data = False

		org_image = util.set_image_alignment(util.load_image(test_filename, print_console=False), self.scale)

		if len(org_image.shape) >= 3 and org_image.shape[2] == 3 and self.channels == 1:
			org_image = util.convert_rgb_to_y(org_image)

		input_image = util.resize_image_by_pil(org_image, 1.0 / self.scale, resampling_method=self.resampling_method)
		bicubic_image = util.resize_image_by_pil(input_image, self.scale, resampling_method=self.resampling_method)

		feed_dict = {self.x: input_image.reshape([1, input_image.shape[0], input_image.shape[1], input_image.shape[2]]),
		             self.x2: bicubic_image.reshape(
			             [1, bicubic_image.shape[0], bicubic_image.shape[1], bicubic_image.shape[2]]),
		             self.y: org_image.reshape([1, org_image.shape[0], org_image.shape[1], org_image.shape[2]]),
		             self.dropout: 1.0,
		             self.is_training: 0}

		if save_meta_data:
			# profiler = tf.profiler.Profile(self.sess.graph)

			run_metadata = tf.RunMetadata()
			run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
			summary_str, _ = self.sess.run([self.summary_op, self.loss], feed_dict=feed_dict, options=run_options,
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
			summary_str, _ = self.sess.run([self.summary_op, self.loss], feed_dict=feed_dict)

		self.train_writer.add_summary(summary_str, self.epochs_completed)
		if not self.use_l1_loss:
			util.log_scalar_value(self.train_writer, 'PSNR', self.training_psnr_sum / self.training_step,
			                      self.epochs_completed)
		util.log_scalar_value(self.train_writer, 'LR', self.lr, self.epochs_completed)
		self.train_writer.flush()

		util.log_scalar_value(self.test_writer, 'PSNR', psnr, self.epochs_completed)
		self.test_writer.flush()

	def update_epoch_and_lr(self):

		self.epochs_completed_in_stage += 1

		if self.epochs_completed_in_stage >= self.lr_decay_epoch:

			# set new learning rate
			self.lr *= self.lr_decay
			self.epochs_completed_in_stage = 0
			return True
		else:
			return False

	def print_status(self, mse, psnr, log=False):

		if self.step == 0:
			logging.info("Initial MSE:%f PSNR:%f" % (mse, psnr))
		else:
			processing_time = (time.time() - self.start_time) / self.step
			if self.use_l1_loss:
				line_a = "%s Step:%s MSE:%f PSNR:%f (Training Loss:%0.3f)" % (
					util.get_now_date(), "{:,}".format(self.step), mse, psnr,
					self.training_loss_sum / self.training_step)
			else:
				line_a = "%s Step:%s MSE:%f PSNR:%f (Training PSNR:%0.3f)" % (
					util.get_now_date(), "{:,}".format(self.step), mse, psnr,
					self.training_psnr_sum / self.training_step)
			estimated = processing_time * (self.total_epochs - self.epochs_completed) * (
				self.training_images // self.batch_num)
			h = estimated // (60 * 60)
			estimated -= h * 60 * 60
			m = estimated // 60
			s = estimated - m * 60
			line_b = "Epoch:%d LR:%f (%2.3fsec/step) Estimated:%d:%d:%d" % (
				self.epochs_completed, self.lr, processing_time, h, m, s)
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

	def evaluate(self, test_filenames):

		total_mse = total_psnr = 0
		if len(test_filenames) == 0:
			return 0, 0

		for filename in test_filenames:
			mse = self.do_for_evaluate(filename, print_console=False)
			total_mse += mse
			total_psnr += util.get_psnr(mse, max_value=self.max_value)

		return total_mse / len(test_filenames), total_psnr / len(test_filenames)

	def do(self, input_image, bicubic_input_image=None):

		h, w = input_image.shape[:2]
		ch = input_image.shape[2] if len(input_image.shape) > 2 else 1

		if self.max_value != 255.0:
			input_image = np.multiply(input_image, self.max_value / 255.0)  # type: np.ndarray

		if bicubic_input_image is None:
			bicubic_input_image = util.resize_image_by_pil(input_image, self.scale,
			                                               resampling_method=self.resampling_method)

		if self.self_ensemble > 1:
			output = np.zeros([self.scale * h, self.scale * w, 1])

			for i in range(self.self_ensemble):
				image = util.flip(input_image, i)
				bicubic_image = util.flip(bicubic_input_image, i)
				y = self.sess.run(self.y_, feed_dict={self.x: image.reshape(1, image.shape[0], image.shape[1], ch),
				                                      self.x2: bicubic_image.reshape(1, self.scale * image.shape[0],
				                                                                     self.scale * image.shape[1],
				                                                                     ch),
				                                      self.dropout: 1.0, self.is_training: 0})
				restored = util.flip(y[0], i, invert=True)
				output += restored

			output /= self.self_ensemble
		else:
			y = self.sess.run(self.y_, feed_dict={self.x: input_image.reshape(1, h, w, ch),
			                                      self.x2: bicubic_input_image.reshape(1, self.scale * h,
			                                                                           self.scale * w, ch),
			                                      self.dropout: 1.0, self.is_training: 0})
			output = y[0]

		if self.max_value != 255.0:
			hr_image = np.multiply(output, 255.0 / self.max_value)
		else:
			hr_image = output

		return hr_image

	def do_for_file(self, file_path, output_folder="output"):

		org_image = util.load_image(file_path)

		filename, extension = os.path.splitext(os.path.basename(file_path))
		output_folder += "/" + self.name + "/"
		util.save_image(output_folder + filename + extension, org_image)

		if len(org_image.shape) >= 3 and org_image.shape[2] == 3 and self.channels == 1:
			input_y_image = util.convert_rgb_to_y(org_image)
			scaled_image = util.resize_image_by_pil(input_y_image, self.scale, resampling_method=self.resampling_method)
			util.save_image(output_folder + filename + "_bicubic_y" + extension, scaled_image)
			output_y_image = self.do(input_y_image)
			util.save_image(output_folder + filename + "_result_y" + extension, output_y_image)

			scaled_ycbcr_image = util.convert_rgb_to_ycbcr(
				util.resize_image_by_pil(org_image, self.scale, self.resampling_method))
			image = util.convert_y_and_cbcr_to_rgb(output_y_image, scaled_ycbcr_image[:, :, 1:3])
		else:
			scaled_image = util.resize_image_by_pil(org_image, self.scale, resampling_method=self.resampling_method)
			util.save_image(output_folder + filename + "_bicubic_y" + extension, scaled_image)
			image = self.do(org_image)

		util.save_image(output_folder + filename + "_result" + extension, image)

	def do_for_evaluate_with_output(self, file_path, output_directory, print_console=False):

		filename, extension = os.path.splitext(file_path)
		output_directory += "/" + self.name + "/"
		util.make_dir(output_directory)

		true_image = util.set_image_alignment(util.load_image(file_path, print_console=False), self.scale)

		if true_image.shape[2] == 3 and self.channels == 1:

			# for color images
			input_y_image = loader.build_input_image(true_image, channels=self.channels, scale=self.scale,
			                                         alignment=self.scale, convert_ycbcr=True)
			input_bicubic_y_image = util.resize_image_by_pil(input_y_image, self.scale,
			                                                 resampling_method=self.resampling_method)

			true_ycbcr_image = util.convert_rgb_to_ycbcr(true_image)

			output_y_image = self.do(input_y_image, input_bicubic_y_image)
			mse = util.compute_mse(true_ycbcr_image[:, :, 0:1], output_y_image,
			                       border_size=self.psnr_calc_border_size)
			loss_image = util.get_loss_image(true_ycbcr_image[:, :, 0:1], output_y_image,
			                                 border_size=self.psnr_calc_border_size)

			output_color_image = util.convert_y_and_cbcr_to_rgb(output_y_image, true_ycbcr_image[:, :, 1:3])

			util.save_image(output_directory + file_path, true_image)
			util.save_image(output_directory + filename + "_input" + extension, input_y_image)
			util.save_image(output_directory + filename + "_input_bicubic" + extension, input_bicubic_y_image)
			util.save_image(output_directory + filename + "_true_y" + extension, true_ycbcr_image[:, :, 0:1])
			util.save_image(output_directory + filename + "_result" + extension, output_y_image)
			util.save_image(output_directory + filename + "_result_c" + extension, output_color_image)
			util.save_image(output_directory + filename + "_loss" + extension, loss_image)

		elif true_image.shape[2] == 1 and self.channels == 1:

			# for monochrome images
			input_image = loader.build_input_image(true_image, channels=self.channels, scale=self.scale,
			                                       alignment=self.scale)
			input_bicubic_y_image = util.resize_image_by_pil(input_image, self.scale,
			                                                 resampling_method=self.resampling_method)
			output_image = self.do(input_image, input_bicubic_y_image)
			mse = util.compute_mse(true_image, output_image, border_size=self.psnr_calc_border_size)
			util.save_image(output_directory + file_path, true_image)
			util.save_image(output_directory + filename + "_result" + extension, output_image)
		else:
			mse = 0

		if print_console:
			print("[%s] MSE:%f, PSNR:%f" % (filename, mse, util.get_psnr(mse)))

		return mse

	def do_for_evaluate(self, file_path, print_console=False):

		true_image = util.set_image_alignment(util.load_image(file_path, print_console=False), self.scale)

		if true_image.shape[2] == 3 and self.channels == 1:

			# for color images
			input_y_image = loader.build_input_image(true_image, channels=self.channels, scale=self.scale,
			                                         alignment=self.scale, convert_ycbcr=True)
			true_y_image = util.convert_rgb_to_y(true_image)
			input_bicubic_y_image = util.resize_image_by_pil(input_y_image, self.scale,
			                                                 resampling_method=self.resampling_method)
			output_y_image = self.do(input_y_image, input_bicubic_y_image)
			mse = util.compute_mse(true_y_image, output_y_image, border_size=self.psnr_calc_border_size)

		elif true_image.shape[2] == 1 and self.channels == 1:

			# for monochrome images
			input_image = loader.build_input_image(true_image, channels=self.channels, scale=self.scale,
			                                       alignment=self.scale)
			input_bicubic_y_image = util.resize_image_by_pil(input_image, self.scale,
			                                                 resampling_method=self.resampling_method)
			output_image = self.do(input_image, input_bicubic_y_image)
			mse = util.compute_mse(true_image, output_image, border_size=self.psnr_calc_border_size)
		else:
			mse = 0

		if print_console:
			print("MSE:%f, PSNR:%f" % (mse, util.get_psnr(mse)))

		return mse

	def init_train_step(self):
		self.lr = self.initial_lr
		self.epochs_completed = 0
		self.epochs_completed_in_stage = 0
		self.min_validation_mse = -1
		self.min_validation_epoch = -1
		self.step = 0

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
			self.batch_image_size, self.batch_image_size, self.training_images)

		if output_to_logging:
			logging.info(status)
		else:
			print(status)

	def log_model_analysis(self):
		run_metadata = tf.RunMetadata()
		run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

		_, loss = self.sess.run([self.optimizer, self.loss], feed_dict={self.x: self.batch_input,
		                                                                self.x2: self.batch_input_bicubic,
		                                                                self.y: self.batch_true,
		                                                                self.lr_input: self.lr,
		                                                                self.dropout: self.dropout_rate},
		                        options=run_options, run_metadata=run_metadata)

		# tf.contrib.tfprof.model_analyzer.print_model_analysis(
		#   tf.get_default_graph(),
		#   run_meta=run_metadata,
		#   tfprof_options=tf.contrib.tfprof.model_analyzer.PRINT_ALL_TIMING_MEMORY)
		self.first_training = False
