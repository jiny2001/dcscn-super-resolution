"""
Paper: "Fast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Network"
Ver: 2

functions for loading/converting data
"""

import configparser
import logging
import os
import random

import numpy as np

from helper import utilty as util

INPUT_IMAGE_DIR = "input"
INTERPOLATED_IMAGE_DIR = "interpolated"
TRUE_IMAGE_DIR = "true"


def build_image_set(file_path, channels=1, scale=1, convert_ycbcr=True, resampling_method="bicubic",
                    print_console=True):
	true_image = util.set_image_alignment(util.load_image(file_path, print_console=print_console), scale)

	if channels == 1 and true_image.shape[2] == 3:
		true_image = util.convert_rgb_to_y(true_image)

	input_image = build_input_image(true_image, channels=channels, scale=scale, alignment=scale,
	                                convert_ycbcr=convert_ycbcr)
	input_interpolated_image = util.resize_image_by_pil(input_image, scale, resampling_method=resampling_method)

	return true_image, input_image, input_interpolated_image


def load_input_image(filename, width=0, height=0, channels=1, scale=1, alignment=0, convert_ycbcr=True,
                     print_console=True):
	image = util.load_image(filename, print_console=print_console)
	return build_input_image(image, width, height, channels, scale, alignment, convert_ycbcr)


def build_input_image(image, width=0, height=0, channels=1, scale=1, alignment=0, convert_ycbcr=True):
	"""
	build input image from file.
	crop, adjust the image alignment for the scale factor, resize, convert color space.
	"""

	if width != 0 and height != 0:
		if image.shape[0] != height or image.shape[1] != width:
			x = (image.shape[1] - width) // 2
			y = (image.shape[0] - height) // 2
			image = image[y: y + height, x: x + width, :]

	if alignment > 1:
		image = util.set_image_alignment(image, alignment)

	if scale != 1:
		image = util.resize_image_by_pil(image, 1.0 / scale)

	if channels == 1 and image.shape[2] == 3:
		if convert_ycbcr:
			image = util.convert_rgb_to_y(image)
	else:
		if convert_ycbcr:
			image = util.convert_rgb_to_ycbcr(image)

	return image


def load_input_batch_image(batch_dir, image_number):
	return util.load_image(batch_dir + "/" + INPUT_IMAGE_DIR + "/%06d.bmp" % image_number, print_console=False)


def load_interpolated_batch_image(batch_dir, image_number):
	return util.load_image(batch_dir + "/" + INTERPOLATED_IMAGE_DIR + "/%06d.bmp" % image_number, print_console=False)


def load_true_batch_image(batch_dir, image_number):
	return util.load_image(batch_dir + "/" + TRUE_IMAGE_DIR + "/%06d.bmp" % image_number, print_console=False)


def save_input_batch_image(batch_dir, image_number, image):
	return util.save_image(batch_dir + "/" + INPUT_IMAGE_DIR + "/%06d.bmp" % image_number, image)


def save_interpolated_batch_image(batch_dir, image_number, image):
	return util.save_image(batch_dir + "/" + INTERPOLATED_IMAGE_DIR + "/%06d.bmp" % image_number, image)


def save_true_batch_image(batch_dir, image_number, image):
	return util.save_image(batch_dir + "/" + TRUE_IMAGE_DIR + "/%06d.bmp" % image_number, image)


def get_batch_count(batch_dir):
	if not os.path.isdir(batch_dir):
		return 0

	config = configparser.ConfigParser()
	try:
		with open(batch_dir + "/batch_images.ini") as f:
			config.read_file(f)
		return config.getint("batch", "count")

	except IOError:
		return 0


class DataSet:
	def __init__(self, batch_image_size, channels=1, scale=1, max_value=255.0, alignment=0):

		self.batch_image_size = batch_image_size
		self.max_value = max_value
		self.channels = channels
		self.scale = scale
		self.max_value = max_value
		self.alignment = alignment

		self.count = 0
		self.images = None
		self.hr_images = None

	def release_images(self):

		if hasattr(self, 'images'):
			del self.images
		self.images = None

		if hasattr(self, 'hr_images'):
			del self.hr_images
		self.hr_images = None

	def load_input_image(self, filename, rescale=False, model=None, resampling_method="bicubic"):

		image = load_input_image(filename, channels=self.channels, scale=self.scale, alignment=self.alignment,
		                         print_console=True)
		if self.max_value != 255.0:
			image = np.multiply(image, self.max_value / 255.0)

		if rescale:
			if model is not None:
				rescaled_image = model.do(image)
			else:
				rescaled_image = util.resize_image_by_pil(image, self.scale, resampling_method=resampling_method)

			return image, rescaled_image
		else:
			return image

	def load_batch_images(self, batch_dir, is_input, count):

		self.release_images()

		if is_input:
			print("Allocate memories for %d * %d^2 + %d * %d^2" % (
				count, self.batch_image_size, count, self.batch_image_size * self.scale))
		else:
			print("Allocate memories for %d * %d^2" % (
				count, self.batch_image_size * self.scale))
		print("Loading %d batch images from %s for [%s]..." % (count, batch_dir, "input" if is_input else "true"))

		self.count = count
		if is_input:
			self.images = np.zeros(shape=[count, self.batch_image_size, self.batch_image_size, 1])  # type: np.ndarray
		else:
			self.images = None
		self.hr_images = np.zeros(
			shape=[count, self.batch_image_size * self.scale, self.batch_image_size * self.scale,
			       1])  # type: np.ndarray

		for i in range(count):
			if is_input:
				self.images[i] = load_input_batch_image(batch_dir, i)
				self.hr_images[i] = load_interpolated_batch_image(batch_dir, i)
			else:
				self.hr_images[i] = load_true_batch_image(batch_dir, i)

			if i % 1000 == 0:
				print('.', end='', flush=True)

		print("Finished")


class BatchDataSets:
	def __init__(self, scale, batch_dir, batch_image_size, stride_size=0, channels=1, resampling_method="bicubic"):

		self.scale = scale
		self.batch_image_size = batch_image_size
		if stride_size == 0:
			self.stride = batch_image_size // 2
		else:
			self.stride = stride_size
		self.channels = channels
		self.resampling_method = resampling_method
		self.count = 0
		self.batch_dir = batch_dir
		self.batch_index = None

	def build_batch(self, data_dir, batch_dir):
		""" Build batch images and. """

		print("Building batch images for %s..." % batch_dir)
		util.make_dir(batch_dir)
		filenames = util.get_files_in_directory(data_dir)
		images_count = 0

		util.make_dir(batch_dir)
		util.clean_dir(batch_dir)
		util.make_dir(batch_dir + "/" + INPUT_IMAGE_DIR)
		util.make_dir(batch_dir + "/" + INTERPOLATED_IMAGE_DIR)
		util.make_dir(batch_dir + "/" + TRUE_IMAGE_DIR)

		processed_images = 0
		for filename in filenames:
			output_window_size = self.batch_image_size * self.scale
			output_window_stride = self.stride * self.scale

			true_image, input_image, input_interpolated_image = build_image_set(filename, channels=self.channels,
			                                                                    scale=self.scale, print_console=False)

			# split into batch images
			input_batch_images = util.get_split_images(input_image, self.batch_image_size, stride=self.stride)
			input_interpolated_batch_images = util.get_split_images(input_interpolated_image, output_window_size,
			                                                        stride=output_window_stride)
			if input_batch_images is None or input_interpolated_batch_images is None:
				continue
			input_count = input_batch_images.shape[0]

			true_batch_images = util.get_split_images(true_image, output_window_size, stride=output_window_stride)

			for i in range(input_count):
				save_input_batch_image(batch_dir, images_count, input_batch_images[i])
				save_interpolated_batch_image(batch_dir, images_count, input_interpolated_batch_images[i])
				save_true_batch_image(batch_dir, images_count, true_batch_images[i])
				images_count += 1
			processed_images +=1
			if processed_images % 10 == 0:
				print('.', end='', flush=True)

		print("Finished")
		self.count = images_count

		print("%d mini-batch images are built(saved)." % images_count)

		config = configparser.ConfigParser()
		config.add_section("batch")
		config.set("batch", "count", str(images_count))
		config.set("batch", "scale", str(self.scale))
		config.set("batch", "batch_image_size", str(self.batch_image_size))
		config.set("batch", "stride", str(self.stride))
		config.set("batch", "channels", str(self.channels))

		with open(batch_dir + "/batch_images.ini", "w") as configfile:
			config.write(configfile)

	def load_batch_counts(self, batch_dir):
		""" load already built batch images. """

		config = configparser.ConfigParser()
		config.read(batch_dir + "/batch_images.ini")
		count = config.getint("batch", "count")

		self.count = count

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

			return True

		except IOError:
			return False

	def init_batch_index(self):
		self.batch_index = random.sample(range(0, self.count), self.count)

	def load_batch_image(self, index):

		index = index % self.count
		image_number = self.batch_index[index]

		input = load_input_batch_image(self.batch_dir, image_number)
		input_interpolated = load_interpolated_batch_image(self.batch_dir, image_number)
		true = load_true_batch_image(self.batch_dir, image_number)

		return input, input_interpolated, true

class DynamicDataSets:
	def __init__(self, scale, batch_image_size, channels=1, resampling_method="bicubic"):

		self.scale = scale
		self.batch_image_size = batch_image_size
		self.channels = channels
		self.resampling_method = resampling_method

		self.filenames = []
		self.count = 0
		self.batch_index = None

	def set_data_dir(self, data_dir):
		self.filenames = util.get_files_in_directory(data_dir)
		self.count = len(self.filenames)
		if self.count <= 0:
			logging.error("Data Directory is empty.")
			exit(-1)

	def init_batch_index(self):
		self.batch_index = random.sample(range(0, self.count), self.count)

	def load_batch_image(self, index):

		image = None
		image_no = index % self.count
		while image is None:
			image = self.load_random_patch(self.filenames[image_no])
			image_no = random.randrange(self.count)

		if random.randrange(2) == 0:
			image = np.fliplr(image)

		input_image = util.resize_image_by_pil(image, 1 / self.scale)
		input_bicubic_image = util.resize_image_by_pil(input_image, self.scale)
		return input_image, input_bicubic_image, image

	def load_random_patch(self, filename):

		image = util.load_image(filename, print_console=False)
		height, width = image.shape[0:2]

		load_batch_size = self.batch_image_size * self.scale
		if height < load_batch_size or width < load_batch_size:
			return None

		y = random.randrange(height - load_batch_size)
		x = random.randrange(width - load_batch_size)
		image = image[y:y + load_batch_size, x:x + load_batch_size, :]
		image = build_input_image(image, channels=self.channels, convert_ycbcr=True)

		return image
