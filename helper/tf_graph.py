"""
Paper: "Fast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Network"
Ver: 2

functions for building tensorflow graph
"""

import logging
import os
import shutil

import tensorflow as tf

from helper import utilty as util


class TensorflowGraph(tf.Graph):

    def __init__(self, flags):
        # inherit tf.Graph so some builtin methods can be used
        super().__init__()

        self.name = ""

        # graph settings
        self.dropout_rate = flags.dropout_rate
        self.activator = flags.activator
        self.batch_norm = flags.batch_norm
        self.cnn_size = flags.cnn_size
        self.cnn_stride = 1
        self.initializer = flags.initializer
        self.weight_dev = flags.weight_dev

        # graph placeholders / objects
        self.is_training = None
        self.dropout = False
        self.saver = None
        self.summary_op = None
        self.train_writer = None
        self.test_writer = None

        # Debugging or Logging
        self.enable_log = flags.enable_log
        self.save_weights = flags.save_weights and flags.enable_log
        self.save_images = flags.save_images and flags.enable_log
        self.save_images_num = flags.save_images_num
        self.save_meta_data = flags.save_meta_data and flags.enable_log
        self.log_weight_image_num = 32

        # Environment (all directory name should not contain '/' after )
        self.checkpoint_dir = flags.checkpoint_dir
        self.tf_log_dir = flags.tf_log_dir

        # status / attributes
        self.Weights = []
        self.Biases = []
        self.features = ""
        self.H = []
        self.receptive_fields = 0
        self.complexity = 0
        self.pix_per_input = 1

        self.init_session(flags.gpu_device_id)

    def init_session(self, device_id=0):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  ## just for use the necesary memory of GPU
        config.gpu_options.visible_device_list = str(device_id)  ## this values depends of numbers of GPUs

        print("Session and graph initialized.")
        self.sess = tf.InteractiveSession(config=config, graph=self)

    def init_all_variables(self):
        self.sess.run(tf.global_variables_initializer())
        print("Model initialized.")

    def build_activator(self, input_tensor, features: int, activator="", leaky_relu_alpha=0.1, base_name=""):
        features = int(features)
        if activator is None or "":
            return
        elif activator == "relu":
            output = tf.nn.relu(input_tensor, name=base_name + "_relu")
        elif activator == "sigmoid":
            output = tf.nn.sigmoid(input_tensor, name=base_name + "_sigmoid")
        elif activator == "tanh":
            output = tf.nn.tanh(input_tensor, name=base_name + "_tanh")
        elif activator == "leaky_relu":
            output = tf.maximum(input_tensor, leaky_relu_alpha * input_tensor, name=base_name + "_leaky")
        elif activator == "prelu":
            with tf.variable_scope("prelu"):
                alphas = tf.Variable(tf.constant(0.1, shape=[features]), name=base_name + "_prelu")
                if self.save_weights:
                    util.add_summaries("prelu_alpha", self.name, alphas, save_stddev=False, save_mean=False)
                output = tf.nn.relu(input_tensor) + tf.multiply(alphas, (input_tensor - tf.abs(input_tensor))) * 0.5
        elif activator == "selu":
            output = tf.nn.selu(input_tensor, name=base_name + "_selu")
        else:
            raise NameError('Not implemented activator:%s' % activator)

        self.complexity += (self.pix_per_input * features)

        return output

    def conv2d(self, input_tensor, w, stride, bias=None, use_batch_norm=False, name=""):
        output = tf.nn.conv2d(input_tensor, w, strides=[1, stride, stride, 1], padding="SAME", name=name + "_conv")
        self.complexity += self.pix_per_input * int(w.shape[0] * w.shape[1] * w.shape[2] * w.shape[3])

        if bias is not None:
            output = tf.add(output, bias, name=name + "_add")
            self.complexity += self.pix_per_input * int(bias.shape[0])

        if use_batch_norm:
            output = tf.layers.batch_normalization(output, training=self.is_training, name='BN')

        return output

    def build_conv(self, name, input_tensor, cnn_size, input_feature_num, output_feature_num, use_bias=False,
                   activator=None, use_batch_norm=False, dropout_rate=1.0):
        with tf.variable_scope(name):
            w = util.weight([cnn_size, cnn_size, input_feature_num, output_feature_num],
                            stddev=self.weight_dev, name="conv_W", initializer=self.initializer)

            b = util.bias([output_feature_num], name="conv_B") if use_bias else None
            h = self.conv2d(input_tensor, w, self.cnn_stride, bias=b, use_batch_norm=use_batch_norm, name=name)

            if activator is not None:
                h = self.build_activator(h, output_feature_num, activator, base_name=name)

            if dropout_rate < 1.0:
                h = tf.nn.dropout(h, self.dropout, name="dropout")

            self.H.append(h)

            if self.save_weights:
                util.add_summaries("weight", self.name, w, save_stddev=True, save_mean=True)
                util.add_summaries("output", self.name, h, save_stddev=True, save_mean=True)
                if use_bias:
                    util.add_summaries("bias", self.name, b, save_stddev=True, save_mean=True)

            if self.save_images and cnn_size > 1:
                util.log_cnn_weights_as_images(self.name, w, max_outputs=self.save_images_num)

        if self.receptive_fields == 0:
            self.receptive_fields = cnn_size
        else:
            self.receptive_fields += (cnn_size - 1)
        self.features += "%d " % output_feature_num

        self.Weights.append(w)
        if use_bias:
            self.Biases.append(b)

        return h

    def depthwise_separable_conv2d(self, input_tensor, w, stride, channel_multiplier=1, bias=None, use_batch_norm=False, name=""):
        # w format is filter_height, filter_width, in_channels, out_channels
        depthwise_filter = util.weight([int(w.shape[0]), int(w.shape[1]), int(w.shape[2]), channel_multiplier],
                                        stddev=self.weight_dev, name="depthwise_W", initializer=self.initializer)
        pointwise_filter = util.weight([1, 1, channel_multiplier * int(w.shape[2]), int(w.shape[3])],
                                        stddev=self.weight_dev, name="pointwise_W", initializer=self.initializer)
        output = tf.nn.separable_conv2d(input_tensor, \
            depthwise_filter, \
            pointwise_filter, \
            strides=[1, stride, stride, 1], \
            padding="SAME", \
            name=name + "_conv")
        self.complexity += (self.pix_per_input * int(w.shape[0] * w.shape[1] * w.shape[2] * channel_multiplier) + \
                            self.pix_per_input * int(w.shape[2] * w.shape[3]))

        if bias is not None:
            output = tf.add(output, bias, name=name + "_add")
            self.complexity += self.pix_per_input * int(bias.shape[0])

        if use_batch_norm:
            output = tf.layers.batch_normalization(output, training=self.is_training, name='BN')

        return output

     # adding the use of depthwise separable convolutions
    def build_depthwise_separable_conv(self, name, input_tensor, cnn_size, input_feature_num, output_feature_num, use_bias=False,
                    activator=None, use_batch_norm=False, dropout_rate=1.0):
        with tf.variable_scope(name):
            w = util.weight([cnn_size, cnn_size, input_feature_num, output_feature_num],
                            stddev=self.weight_dev, name="conv_W", initializer=self.initializer)

            b = util.bias([output_feature_num], name="conv_B") if use_bias else None
            h = self.depthwise_separable_conv2d(input_tensor, w, self.cnn_stride, bias=b, use_batch_norm=use_batch_norm, name=name)
            
            if activator is not None:
                h = self.build_activator(h, output_feature_num, activator, base_name=name)

            if dropout_rate < 1.0:
                h = tf.nn.dropout(h, self.dropout, name="dropout")

            self.H.append(h)

            if self.save_weights:
                util.add_summaries("weight", self.name, w, save_stddev=True, save_mean=True)
                util.add_summaries("output", self.name, h, save_stddev=True, save_mean=True)
                if use_bias:
                    util.add_summaries("bias", self.name, b, save_stddev=True, save_mean=True)

            if self.save_images and cnn_size > 1:
                util.log_cnn_weights_as_images(self.name, w, max_outputs=self.save_images_num)

        if self.receptive_fields == 0:
            self.receptive_fields = cnn_size
        else:
            self.receptive_fields += (cnn_size - 1)
        self.features += "%d " % output_feature_num

        self.Weights.append(w)
        if use_bias:
            self.Biases.append(b)

        return h


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

    def build_pixel_shuffler_layer(self, name, h, scale, input_filters, output_filters, activator=None, depthwise_separable=False):
        with tf.variable_scope(name):
            if (depthwise_separable):
                self.build_depthwise_separable_conv(name + "_CNN", h, self.cnn_size, input_filters, scale * scale * output_filters,
                            use_batch_norm=False,
                            use_bias=True)
            else:
                self.build_conv(name + "_CNN", h, self.cnn_size, input_filters, scale * scale * output_filters,
                            use_batch_norm=False,
                            use_bias=True)
            self.H.append(tf.depth_to_space(self.H[-1], scale))
            self.build_activator(self.H[-1], output_filters, activator, base_name=name)

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

    def load_model(self, name="", trial=0, output_log=False):
        if name == "" or name == "default":
            name = self.name

        if trial > 0:
            filename = self.checkpoint_dir + "/" + name + "_" + str(trial) + ".ckpt"
        else:
            filename = self.checkpoint_dir + "/" + name + ".ckpt"

        if not os.path.isfile(filename + ".index"):
            print("Error. [%s] is not exist!" % filename)
            exit(-1)

        self.saver.restore(self.sess, filename)
        if output_log:
            logging.info("Model restored [ %s ]." % filename)
        else:
            print("Model restored [ %s ]." % filename)

    def save_model(self, name="", trial=0, output_log=False):
        if name == "" or name == "default":
            name = self.name

        if trial > 0:
            filename = self.checkpoint_dir + "/" + name + "_" + str(trial) + ".ckpt"
        else:
            filename = self.checkpoint_dir + "/" + name + ".ckpt"

        self.saver.save(self.sess, filename)

        if output_log:
            logging.info("Model saved [%s]." % filename)
        else:
            print("Model saved [%s]." % filename)

    def build_summary_saver(self, with_saver=True):
        if self.enable_log:
            self.summary_op = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(self.tf_log_dir + "/train")
            self.test_writer = tf.summary.FileWriter(self.tf_log_dir + "/test", graph=self.sess.graph)

        if (with_saver):
            self.saver = tf.train.Saver(max_to_keep=None)
