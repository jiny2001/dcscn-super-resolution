"""
Paper: "Fast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Network"

functions for loading/converting data
"""

import configparser
import os

import numpy as np

from helper import utilty as util

INPUT_IMAGE_DIR = "input"
INTERPOLATED_IMAGE_DIR = "interpolated"
TRUE_IMAGE_DIR = "true"


def convert_to_multi_channel_image(multi_channel_image, image, scale):
	height = multi_channel_image.shape[0]
	width = multi_channel_image.shape[1]

	for y in range(height):
		for x in range(width):
			for y2 in range(scale):
				for x2 in range(scale):
					multi_channel_image[y, x, y2 * scale + x2] = image[y * scale + y2, x * scale + x2, 0]


def convert_from_multi_channel_image(image, multi_channel_image, scale):
	height = multi_channel_image.shape[0]
	width = multi_channel_image.shape[1]

	for y in range(height):
		for x in range(width):
			for y2 in range(scale):
				for x2 in range(scale):
					image[y * scale + y2, x * scale + x2, 0] = multi_channel_image[y, x, y2 * scale + x2]


def load_input_image(filename, width=0, height=0, channels=1, scale=1, alignment=0, convert_ycbcr=True,
                     jpeg_mode=False, print_console=True):
	image = util.load_image(filename, print_console=print_console)
	return build_input_image(image, width, height, channels, scale, alignment, convert_ycbcr, jpeg_mode)


def build_input_image(image, width=0, height=0, channels=1, scale=1, alignment=0, convert_ycbcr=True, jpeg_mode=False):
	"""
	build input image from file.
	crop, adjust the image alignment for the scale factor, resize, convert color space.
	"""

	if width != 0 and height != 0:
		if image.shape[0] != height or image.shape[1] != width:
			x = (image.shape[1] - width) // 2
			y = (image.shape[0] - height) // 2
			image = image[y: y + height, x: x + width, :]

	if image.shape[2] >= 4:
		image = image[:, :, 0:3]

	if alignment > 1:
		image = util.set_image_alignment(image, alignment)

	if scale != 1:
		image = util.resize_image_by_pil(image, 1.0 / scale)

	if channels == 1 and image.shape[2] == 3:
		if convert_ycbcr:
			image = util.convert_rgb_to_y(image, jpeg_mode=jpeg_mode)
	else:
		if convert_ycbcr:
			image = util.convert_rgb_to_ycbcr(image, jpeg_mode=jpeg_mode)

	return image


class DataSet:
	def __init__(self, batch_image_size, channels=1, scale=1, max_value=255.0, alignment=0, jpeg_mode=False):

		self.batch_image_size = batch_image_size
		self.max_value = max_value
		self.channels = channels
		self.scale = scale
		self.max_value = max_value
		self.alignment = alignment
		self.jpeg_mode = jpeg_mode

		self.count = 0
		self.images = None
		self.quad_images = None

	def load_test_image(self, filename):

		image = load_input_image(filename, channels=self.channels, scale=1, alignment=self.alignment,
		                         jpeg_mode=self.jpeg_mode, print_console=False)
		if self.max_value != 255.0:
			image = np.multiply(image, self.max_value / 255.0)

		return image

	def load_input_image(self, filename, rescale=False, resampling_method="bicubic"):

		image = load_input_image(filename, channels=self.channels, scale=self.scale, alignment=self.alignment,
		                         jpeg_mode=self.jpeg_mode, print_console=True)
		if self.max_value != 255.0:
			image = np.multiply(image, self.max_value / 255.0)

		if rescale:
			rescaled_image = util.resize_image_by_pil(image, self.scale, resampling_method=resampling_method)
			return image, rescaled_image
		else:
			return image

	def load_batch_images(self, batch_dir, input_batch, count):

		print("Loading %d batch images from %s for [%s]" % (count, batch_dir, "input" if input_batch else "true"))

		self.count = count
		if input_batch:
			self.images = np.zeros(shape=[count, self.batch_image_size, self.batch_image_size, 1])  # type: np.ndarray
		else:
			self.images = None
		self.quad_images = np.zeros(
			shape=[count, self.batch_image_size, self.batch_image_size, self.scale * self.scale])  # type: np.ndarray

		for i in range(count):
			if input_batch:
				self.images[i] = util.load_image(batch_dir + "/" + INPUT_IMAGE_DIR + "/%06d.bmp" % i, print_console=False)
				quad_image = util.load_image(batch_dir + "/" + INTERPOLATED_IMAGE_DIR + "/%06d.bmp" % i, print_console=False)
			else:
				quad_image = util.load_image(batch_dir + "/" + TRUE_IMAGE_DIR + "/%06d.bmp" % i, print_console=False)

			convert_to_multi_channel_image(self.quad_images[i], quad_image, self.scale)

			if i % 1000 == 0:
				print('.', end='', flush=True)

		print("Finished")


class DataSets:
	def __init__(self, scale, batch_image_size, stride_size, channels=1,
	             jpeg_mode=False, max_value=255.0, resampling_method="nearest"):

		self.scale = scale
		self.batch_image_size = batch_image_size
		self.stride = stride_size
		self.channels = channels
		self.jpeg_mode = jpeg_mode
		self.max_value = max_value
		self.resampling_method = resampling_method

		self.input = DataSet(batch_image_size, channels=channels, scale=scale, alignment=scale, jpeg_mode=jpeg_mode,
		                     max_value=max_value)
		self.true = DataSet(batch_image_size, channels=channels, scale=scale, alignment=scale, jpeg_mode=jpeg_mode,
		                    max_value=max_value)

	def build_batch(self, data_dir, batch_dir):
		""" load from input files. Then save batch images on file to reduce memory consumption. """

		print("Building batch images for %s..." % batch_dir)
		filenames = util.get_files_in_directory(data_dir)
		images_count = 0

		util.make_dir(batch_dir)
		util.clean_dir(batch_dir)
		util.make_dir(batch_dir + "/" + INPUT_IMAGE_DIR)
		util.make_dir(batch_dir + "/" + INTERPOLATED_IMAGE_DIR)
		util.make_dir(batch_dir + "/" + TRUE_IMAGE_DIR)

		for filename in filenames:
			output_window_size = self.batch_image_size * self.scale
			output_window_stride = self.stride * self.scale
			input_image, input_bicubic_image = self.input.load_input_image(filename, rescale=True,
			                                                               resampling_method=self.resampling_method)
			test_image = self.true.load_test_image(filename)

			# split into batch images
			input_batch_images = util.get_split_images(input_image, self.batch_image_size, stride=self.stride)
			input_bicubic_batch_images = util.get_split_images(input_bicubic_image, output_window_size,
			                                                   stride=output_window_stride)
			input_count = input_batch_images.shape[0]

			test_batch_images = util.get_split_images(test_image, output_window_size, stride=output_window_stride)

			for i in range(input_count):
				# util.save_image_data(batch_dir + "/" + INPUT_IMAGE_DIR + "/%06d.npy" % images_count, input_batch_images[i])
				# util.save_image_data(batch_dir + "/" + INTERPOLATED_IMAGE_DIR + "/%06d.npy" % images_count,
				#                 input_bicubic_batch_images[i])
				# util.save_image_data(batch_dir + "/" + TRUE_IMAGE_DIR + "/%06d.npy" % images_count, test_batch_images[i])
				util.save_image(batch_dir + "/" + INPUT_IMAGE_DIR + "/%06d.bmp" % images_count, input_batch_images[i])
				util.save_image(batch_dir + "/" + INTERPOLATED_IMAGE_DIR + "/%06d.bmp" % images_count,
				                input_bicubic_batch_images[i])
				util.save_image(batch_dir + "/" + TRUE_IMAGE_DIR + "/%06d.bmp" % images_count, test_batch_images[i])

				images_count += 1

		print("%d mini-batch images are built(saved)." % images_count)

		config = configparser.ConfigParser()
		config.add_section("batch")
		config.set("batch", "count", str(images_count))
		config.set("batch", "scale", str(self.scale))
		config.set("batch", "batch_image_size", str(self.batch_image_size))
		config.set("batch", "stride", str(self.stride))
		config.set("batch", "channels", str(self.channels))
		config.set("batch", "jpeg_mode", str(self.jpeg_mode))
		config.set("batch", "max_value", str(self.max_value))

		with open(batch_dir + "/batch_images.ini", "w") as configfile:
			config.write(configfile)

	def load_batch(self, batch_dir):
		""" load already built batch images. """

		config = configparser.ConfigParser()
		config.read(batch_dir + "/batch_images.ini")
		count = config.getint("batch", "count")

		self.input.load_batch_images(batch_dir, True, count)
		self.true.load_batch_images(batch_dir, False, count)

	def is_batch_exist(self, batch_dir):
		if not os.path.isdir(batch_dir):
			return False

		config = configparser.ConfigParser()
		try:
			with open(batch_dir + "/batch_images.ini") as f:
				config.read_file(f)

			if config.getint("batch", "count") <= 0:
				return False

			if config.getint("batch", "scale") != self.scale:
				return False
			if config.getint("batch", "batch_image_size") != self.batch_image_size:
				return False
			if config.getint("batch", "stride") != self.stride:
				return False
			if config.getint("batch", "channels") != self.channels:
				return False
			if config.getboolean("batch", "jpeg_mode") != self.jpeg_mode:
				return False
			if config.getint("batch", "max_value") != self.max_value:
				return False

			return True

		except IOError:
			return False
