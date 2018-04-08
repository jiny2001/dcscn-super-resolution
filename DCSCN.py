"""
Paper: "Fast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Network"
Ver: 2.0

DCSCN model implementation (Transposed-CNN version)
"""

import logging
import os
import random
import time

import numpy as np
import tensorflow as tf

from helper import loader, tf_graph, utilty as util

BICUBIC_METHOD_STRING = "bicubic"


class SuperResolution(tf_graph.TensorflowGraph):
	def __init__(self, flags, model_name=""):

		super().__init__(flags)

		# Model Parameters
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
		self.data_num = max(1,(flags.data_num // flags.batch_num)) * flags.batch_num

		# Image Processing Parameters
		self.scale = flags.scale
		self.max_value = flags.max_value
		self.channels = flags.channels
		self.jpeg_mode = flags.jpeg_mode
		self.output_channels = 1

		# Environment (all directory name should not contain '/' after )
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
		if flags.initialise_tf_log:
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
			if self.filters_decay_gamma != 1.0:
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
			if self.dropout_rate != 1.0:
				name += "_D%0.2f" % self.dropout_rate
			if self.max_value != 255.0:
				name += "_M%2.1f" % self.max_value
			if self.activator != "relu":
				name += "_%s" % self.activator
			if self.dataset != "yang91":
				name += "_" + self.dataset
			if self.batch_image_size != 32:
				name += "_B%d" % self.batch_image_size
			if self.clipping_norm > 0:
				name += "_CN%.1f" % self.clipping_norm
			if self.batch_norm:
				name += "_BN"
			if self.reconstruct_layers > 1:
				name += "_R%d" % self.reconstruct_layers
				if self.reconstruct_filters != 1:
					name += "F%d" % self.reconstruct_filters
			if name_postfix is not "":
				name += "_" + name_postfix
		else:
			name = "dcscn_%s" % model_name

		return name

	def open_datasets(self, target, data_dir, batch_image_size, stride_size=0):

		# todo stride_size may not be used?
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

	def load_datasets(self, target, data_dir, batch_dir, batch_image_size, stride_size=0):

		batch_dir += "/scale%d" % self.scale
		print("Loading datasets for [%s]..." % target)
		util.make_dir(batch_dir)

		if stride_size == 0:
			stride_size = batch_image_size // 2

		datasets = loader.DataSets(self.scale, batch_image_size, stride_size, channels=self.channels,
		                           jpeg_mode=self.jpeg_mode, max_value=self.max_value,
		                           resampling_method=self.resampling_method)

		if not datasets.is_batch_exist(batch_dir):
			datasets.build_batch(data_dir, batch_dir)

		if target == "training":
			datasets.load_batch_image_count(batch_dir)
			self.train = datasets
		else:
			datasets.load_batch(batch_dir)
			self.test = datasets

	def build_training_datasets(self, data_dir, batch_dir, batch_image_size, stride_size=0):

		print("Building datasets for [%s]..." % "train")
		util.make_dir(batch_dir)

		if stride_size == 0:
			stride_size = batch_image_size // 2

		self.train = loader.DataSets(self.scale, batch_image_size, stride_size, channels=self.channels,
		                             jpeg_mode=self.jpeg_mode, max_value=self.max_value,
		                             resampling_method=self.resampling_method)

		if not self.train.is_batch_exist(batch_dir):
			self.train.build_batch(data_dir, batch_dir)

	def init_epoch_index(self):

		self.batch_input = self.batch_num * [None]
		self.batch_input_quad = self.batch_num * [None]
		self.batch_true_quad = self.batch_num * [None]

		self.batch_index = random.sample(range(0, self.train.input.count), self.train.input.count)
		self.index_in_epoch = 0
		self.training_psnr_sum = 0
		self.training_mse_sum = 0
		self.training_step = 0

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

			# util.save_image("output/%d_input.png" % i, util.resize_image_by_pil(true_image, 1 / self.scale))
			# util.save_image("output/%d_true.png" % i, true_image)

			# todo add scaling function
			self.batch_input[i] = util.resize_image_by_pil(image, 1 / self.scale)
			self.batch_input_quad[i] = util.resize_image_by_pil(self.batch_input[i], self.scale)
			self.batch_true_quad[i] = image
			self.index_in_epoch += 1


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
			input_channels = total_output_feature_num

		# building upsampling layer
		if self.pixel_shuffler:
			if self.scale == 4:
				self.build_pixel_shuffler_layer("Up-PS", self.H[-1], 2, input_channels)
				self.build_pixel_shuffler_layer("Up-PS2", self.H[-1], 2, input_channels)
			else:
				self.build_pixel_shuffler_layer("Up-PS", self.H[-1], self.scale, input_channels)
		else:
			self.build_transposed_conv("Up-TC", self.H[-1], self.scale, input_channels)

		for i in range(self.reconstruct_layers - 1):
			self.build_conv("R-CNN%d" % (i + 1), self.H[-1], self.cnn_size, input_channels, self.reconstruct_filters,
			                dropout_rate=self.dropout_rate, use_bias=True, activator=self.activator)
			input_channels = self.reconstruct_filters

		self.build_conv("R-CNN%d" % self.reconstruct_layers, self.H[-1], self.cnn_size, input_channels, self.output_channels)

		self.y_ = self.H[-1] + self.x2

		logging.info("Feature:%s Complexity:%s Receptive Fields:%d" % (
			self.features, "{:,}".format(self.complexity), self.receptive_fields))

	def build_optimizer(self):

		self.lr_input = tf.placeholder(tf.float32, shape=[], name="LearningRate")
		diff = self.y_ - self.y

		self.mse = tf.reduce_mean(tf.square(diff), name="mse")
		loss = self.mse

		if self.l2_decay > 0:
			l2_losses = [tf.nn.l2_loss(w) for w in self.Weights]
			# l1_losses = [tf.reduce_sum(tf.abs(w)) for w in self.weights]  # l1 loss

			l2_loss = self.l2_decay * tf.add_n(l2_losses)
			loss += l2_loss
			if self.save_loss:
				tf.summary.scalar("loss_l2/" + self.name, l2_loss)

		if self.save_loss:
			tf.summary.scalar("loss/" + self.name, loss)

		self.loss = loss

		if self.batch_norm:
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				self.training_optimizer = self.add_optimizer_op(loss, self.lr_input)
		else:
			self.training_optimizer = self.add_optimizer_op(loss, self.lr_input)

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

	def train_batch(self):
		# todo rename quad
		feed_dict = {self.x: self.batch_input, self.x2: self.batch_input_quad, self.y: self.batch_true_quad,
		             self.lr_input: self.lr, self.dropout: self.dropout_rate, self.is_training: 1}

		_, mse = self.sess.run([self.training_optimizer, self.mse], feed_dict=feed_dict)

		self.training_mse_sum += mse
		self.training_psnr_sum += util.get_psnr(mse, max_value=self.max_value)
		self.training_step += 1
		self.step += 1

	def evaluate_test_batch(self, save_meta_data=False, trial=0, log_profile=True):

		save_meta_data = save_meta_data and self.save_meta_data and (trial == 0)
		feed_dict = {self.x: self.test.input.images,
		             self.x2: self.test.input.quad_images,
		             self.y: self.test.true.quad_images,
		             self.dropout: 1.0,
		             self.is_training: 0}

		if log_profile and (self.save_loss or self.save_weights or save_meta_data):

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
			util.log_scalar_value(self.train_writer, 'training_PSNR', self.training_psnr_sum / self.training_step,
			                      self.epochs_completed)
			util.log_scalar_value(self.train_writer, 'LR', self.lr, self.epochs_completed)
			self.train_writer.flush()

			util.log_scalar_value(self.test_writer, 'PSNR', util.get_psnr(mse), self.epochs_completed)
			self.test_writer.flush()
		else:
			mse = self.sess.run(self.mse, feed_dict=feed_dict)

		return mse

	def update_epoch_and_lr(self, mse):

		self.epochs_completed_in_stage += 1

		if self.epochs_completed_in_stage >= self.lr_decay_epoch:

			# set new learning rate
			self.lr *= self.lr_decay
			self.epochs_completed_in_stage = 0
			return True
		else:
			return False

	def print_status(self, mse, log=False):

		psnr = util.get_psnr(mse, max_value=self.max_value)

		if self.step == 0:
			logging.info("Initial MSE:%f PSNR:%f" % (mse, psnr))
		else:
			processing_time = (time.time() - self.start_time) / self.step
			line_a = "%s Step:%s MSE:%f PSNR:%f (Training PSNR:%0.3f)" % (
				util.get_now_date(), "{:,}".format(self.step), mse, psnr, self.training_psnr_sum / self.training_step)
			estimated = processing_time * (self.total_epochs - self.epochs_completed) * (self.data_num // self.batch_num)
			h = estimated // (60 * 60)
			estimated -= h * 60 *60
			m = estimated // 60
			s = estimated - m * 60
			line_b = "Epoch:%d LR:%f (%2.3fsec/step) Estimated:%d:%d:%d" % ( self.epochs_completed, self.lr, processing_time,
			                                                                 h, m , s)
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
			                                      self.x2: bicubic_input_image.reshape(1, self.scale * h, self.scale * w, ch),
			                                      self.dropout: 1.0, self.is_training: 0})
			output = y[0]

		if self.max_value != 255.0:
			quad_image = np.multiply(output, 255.0 / self.max_value)
		else:
			quad_image = output

		return quad_image

	def do_for_file(self, file_path, output_folder="output"):

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

	def do_for_evaluate(self, file_path, output_directory="output", output=True, print_console=False):

		filename, extension = os.path.splitext(file_path)
		output_directory += "/" + self.name + "/"
		util.make_dir(output_directory)
		true_image = util.set_image_alignment(util.load_image(file_path, print_console=False), self.scale)

		if true_image.shape[2] == 3 and self.channels == 1:
			input_y_image = loader.build_input_image(true_image, channels=self.channels, scale=self.scale,
			                                         alignment=self.scale, convert_ycbcr=True, jpeg_mode=self.jpeg_mode)
			# for color images
			if output:
				input_bicubic_y_image = util.resize_image_by_pil(input_y_image, self.scale,
				                                                 resampling_method=self.resampling_method)

				true_ycbcr_image = util.convert_rgb_to_ycbcr(true_image, jpeg_mode=self.jpeg_mode)

				output_y_image = self.do(input_y_image, input_bicubic_y_image)
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
				input_bicubic_y_image = util.resize_image_by_pil(input_y_image, self.scale,
				                                                 resampling_method=self.resampling_method)
				output_y_image = self.do(input_y_image, input_bicubic_y_image)
				mse = util.compute_mse(true_y_image, output_y_image, border_size=self.scale)

		elif true_image.shape[2] == 1 and self.channels == 1:

			# for monochrome images
			input_image = loader.build_input_image(true_image, channels=self.channels, scale=self.scale, alignment=self.scale)
			input_bicubic_y_image = util.resize_image_by_pil(input_image, self.scale,
			                                                 resampling_method=self.resampling_method)
			output_image = self.do(input_image, input_bicubic_y_image)
			mse = util.compute_mse(true_image, output_image, border_size=self.scale)
			if output:
				util.save_image(output_directory + file_path, true_image)
				util.save_image(output_directory + filename + "_result" + extension, output_image)
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
			self.batch_image_size, self.batch_image_size, self.train.input.count)

		if output_to_logging:
			logging.info(status)
		else:
			print(status)

	def log_model_analysis(self):
		run_metadata = tf.RunMetadata()
		run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

		_, mse = self.sess.run([self.optimizer, self.mse], feed_dict={self.x: self.batch_input,
		                                                              self.x2: self.batch_input_quad,
		                                                              self.y: self.batch_true_quad,
		                                                              self.lr_input: self.lr,
		                                                              self.dropout: self.dropout_rate},
		                       options=run_options, run_metadata=run_metadata)

		# tf.contrib.tfprof.model_analyzer.print_model_analysis(
		#   tf.get_default_graph(),
		#   run_meta=run_metadata,
		#   tfprof_options=tf.contrib.tfprof.model_analyzer.PRINT_ALL_TIMING_MEMORY)
		self.first_training = False

