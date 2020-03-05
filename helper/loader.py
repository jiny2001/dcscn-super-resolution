"""
Paper: "Fast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Network"
Ver: 2

functions for loading/converting data
"""
import configparser
import logging
import os
import random
import time
import math
from multiprocessing import Process, Queue
from helper import utilty as util
import numpy as np
from scipy import misc

INPUT_IMAGE_DIR = "input"
INTERPOLATED_IMAGE_DIR = "interpolated"
TRUE_IMAGE_DIR = "true"

def loadall(q1,count, threads, thread, batch_dir):
    inputfiles = os.listdir(batch_dir+'/'+INPUT_IMAGE_DIR)[math.floor(count/threads)*thread:math.floor(count/threads)*(thread+1)]
    interpolatedfiles = os.listdir(batch_dir+'/'+INTERPOLATED_IMAGE_DIR)[math.floor(count/threads)*thread:math.floor(count/threads)*(thread+1)]
    truefiles = os.listdir(batch_dir+'/'+TRUE_IMAGE_DIR)[math.floor(count/threads)*thread:math.floor(count/threads)*(thread+1)]
    
    load_input_batch_image(q1,inputfiles,batch_dir)
    load_interpolated_batch_image(q1,interpolatedfiles,batch_dir)
    load_true_batch_image(q1,truefiles,batch_dir)
    
def build_batch(data_dir, thread, threads):
    """ Build batch images and. """

    filenames = util.get_files_in_directory(data_dir)
    images_count = 0
    processed_images = int(len(filenames)/threads*(thread))
    for filename in filenames[int(len(filenames)/threads*thread):int(len(filenames)/threads*(thread+1))]:
        output_window_size = BatchDataSets.batch_image_size * BatchDataSets.scale
        output_window_stride = BatchDataSets.stride * BatchDataSets.scale

        input_image, input_interpolated_image, true_image = \
            build_image_set(filename, channels=BatchDataSets.channels, resampling_method=BatchDataSets.resampling_method,
                            scale=BatchDataSets.scale, print_console=False)

        # split into batch images
        input_batch_images = util.get_split_images(input_image, BatchDataSets.batch_image_size, stride=BatchDataSets.stride)
        input_interpolated_batch_images = util.get_split_images(input_interpolated_image, output_window_size,
                                                                stride=output_window_stride)

        if input_batch_images is None or input_interpolated_batch_images is None:
            # if the original image size * scale is less than batch image size
            continue
        input_count = input_batch_images.shape[0]

        true_batch_images = util.get_split_images(true_image, output_window_size, stride=output_window_stride)

        for i in range(input_count):
            BatchDataSets.save_input_batch_image(thread*1000000+images_count, input_batch_images[i])
            BatchDataSets.save_interpolated_batch_image(thread*1000000+images_count, input_interpolated_batch_images[i])
            BatchDataSets.save_true_batch_image(thread*1000000+images_count, true_batch_images[i])
            images_count += 1
        processed_images += 1
        if processed_images % 10 == 0:
            print('.', end='', flush=True)

    print("Finished")
    BatchDataSets.count = images_count

    print("%d mini-batch images are built(saved)." % images_count)

    config = configparser.ConfigParser()
    config.add_section("batch")
    config.set("batch", "count", str(images_count))
    config.set("batch", "scale", str(BatchDataSets.scale))
    config.set("batch", "batch_image_size", str(BatchDataSets.batch_image_size))
    config.set("batch", "stride", str(BatchDataSets.stride))
    config.set("batch", "channels", str(BatchDataSets.channels))

    with open(BatchDataSets.batch_dir + "/batch_images.ini", "w") as configfile:
        config.write(configfile)

def load_input_batch_image(q, files, batch_dir):
    for image_number in files:
        image = misc.imread(batch_dir + "/" + INPUT_IMAGE_DIR + "/" +image_number)
        image = image.reshape(image.shape[0], image.shape[1], 1)
        q.put(image)

def load_interpolated_batch_image(q, files,batch_dir):
    for image_number in files:
        image = misc.imread(batch_dir + "/" + INTERPOLATED_IMAGE_DIR + "/" +image_number)
        image = image.reshape(image.shape[0], image.shape[1], 1)
        q.put(image)

def load_true_batch_image(q, files,batch_dir):
    for image_number in files:
        image = misc.imread(batch_dir + "/" + TRUE_IMAGE_DIR + "/" + image_number)
        image = image.reshape(image.shape[0], image.shape[1], 1)
        q.put(image)

def build_image_set(file_path, channels=1, scale=1, convert_ycbcr=True, resampling_method="bicubic",
                    print_console=True):
    true_image = util.set_image_alignment(util.load_image(file_path, print_console=print_console), scale)

    if channels == 1 and true_image.shape[2] == 3 and convert_ycbcr:
        true_image = util.convert_rgb_to_y(true_image)

    input_image = util.resize_image_by_pil(true_image, 1.0 / scale, resampling_method=resampling_method)
    input_interpolated_image = util.resize_image_by_pil(input_image, scale, resampling_method=resampling_method)

    return input_image, input_interpolated_image, true_image


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

    if channels == 1 and image.shape[2] == 3:
        if convert_ycbcr:
            image = util.convert_rgb_to_y(image)
    else:
        if convert_ycbcr:
            image = util.convert_rgb_to_ycbcr(image)

    if scale != 1:
        image = util.resize_image_by_pil(image, 1.0 / scale)

    return image


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

    def build_batch_threaded(self, data_dir, batch_dir, threads):
        threadlist = []
        for i in range(threads):
            threadlist.append(Process(target=self.build_batch, args=(data_dir,i,threads,)))
            threadlist[i].start()

        for i in range(threads):
            if(threads > i):
                threadlist[i].join()

        self.count = len(os.listdir(batch_dir))

        
    def build_batch(self, data_dir, thread, threads):
        """ Build batch images and. """

        print("Building batch images for %s..." % self.batch_dir)
        filenames = util.get_files_in_directory(data_dir)
        images_count = 0
        processed_images = int(len(filenames)/threads*(thread))
        for filename in filenames[int(len(filenames)/threads*thread):int(len(filenames)/threads*(thread+1))]:
            output_window_size = self.batch_image_size * self.scale
            output_window_stride = self.stride * self.scale

            input_image, input_interpolated_image, true_image = \
                build_image_set(filename, channels=self.channels, resampling_method=self.resampling_method,
                                scale=self.scale, print_console=False)

            # split into batch images
            input_batch_images = util.get_split_images(input_image, self.batch_image_size, stride=self.stride)
            input_interpolated_batch_images = util.get_split_images(input_interpolated_image, output_window_size,
                                                                    stride=output_window_stride)

            if input_batch_images is None or input_interpolated_batch_images is None:
                # if the original image size * scale is less than batch image size
                continue
            input_count = input_batch_images.shape[0]

            true_batch_images = util.get_split_images(true_image, output_window_size, stride=output_window_stride)

            for i in range(input_count):
                self.save_input_batch_image(thread*1000000+images_count, input_batch_images[i])
                self.save_interpolated_batch_image(thread*1000000+images_count, input_interpolated_batch_images[i])
                self.save_true_batch_image(thread*1000000+images_count, true_batch_images[i])
                images_count += 1
            processed_images += 1
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

        with open(self.batch_dir + "/batch_images.ini", "w") as configfile:
            config.write(configfile)

    def load_batch_counts(self):
        """ load already built batch images. """

        if not os.path.isdir(self.batch_dir):
            self.count = 0
            return

        config = configparser.ConfigParser()
        try:
            with open(self.batch_dir + "/batch_images.ini") as f:
                config.read_file(f)
            self.count = config.getint("batch", "count")

        except IOError:
            self.count = 0
            return

    def load_all_batch_images(self, threads):

        self.count = len(os.listdir(self.batch_dir+"/input"))
        print("Allocating memory for all batch images.")
        self.input_images = np.zeros(shape=[self.count, self.batch_image_size, self.batch_image_size, 1],
                                     dtype=np.uint8)  # type: np.ndarray
        self.input_interpolated_images = np.zeros(
            shape=[self.count, self.batch_image_size * self.scale, self.batch_image_size * self.scale, 1],
            dtype=np.uint8)  # type: np.ndarray
        self.true_images = np.zeros(
            shape=[self.count, self.batch_image_size * self.scale, self.batch_image_size * self.scale, 1],
            dtype=np.uint8)  # type: np.ndarray

        batch_dir = self.batch_dir
        queuelist = []
        threadlist = []
        
        print("Loading all batch images.")
        for i in range(threads):
            queuelist.append(Queue())
            threadlist.append(Process(target=loadall, args=(queuelist[i],self.count,threads,i,batch_dir,)))
            threadlist[i].start()
        count = 0
        
        #recieve input images
        for u in range(0,math.floor(self.count/threads)):
            for i in range(threads):
                self.input_images[count] = queuelist[i].get()
                count+=1
            if(u%10000 == 0):
                print('.', end='', flush=True)
        count = 0
        print("\n")


        #recieve interpolated images
        for u in range(0,math.floor(self.count/threads)):
            for i in range(threads):
                self.input_interpolated_images[count] = queuelist[i].get()
                count+=1
            if(u%10000 == 0):
                print('.', end='', flush=True)
        count = 0
        print("\n")

        #recieve true images
        for u in range(0,math.floor(self.count/threads)):
            for i in range(threads):
                self.true_images[count] = queuelist[i].get()
                count+=1
            if(u%10000 == 0):
                print('.', end='', flush=True)
        count = 0
        print("\n")

        #shut down threads
        for i in range(threads):
            threadlist[i].join()
        
        print("\n")
        print("Load finished.")

    def release_batch_images(self):

        if hasattr(self, 'input_images'):
            del self.input_images
        self.input_images = None

        if hasattr(self, 'input_interpolated_images'):
            del self.input_interpolated_images
        self.input_interpolated_images = None

        if hasattr(self, 'true_images'):
            del self.true_images
        self.true_images = None

    def is_batch_exist(self):
        if not os.path.isdir(self.batch_dir):
            return False

        config = configparser.ConfigParser()
        try:
            with open(self.batch_dir + "/batch_images.ini") as f:
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
        self.index = 0

    def get_next_image_no(self):

        if self.index >= self.count:
            self.init_batch_index()

        image_no = self.batch_index[self.index]
        self.index += 1
        return image_no

    def load_batch_image_from_disk(self, image_number):

        image_number = image_number % self.count

        input_image = self.load_input_batch_image(image_number)
        input_interpolated = self.load_interpolated_batch_image(image_number)
        true = self.load_true_batch_image(image_number)

        return input_image, input_interpolated, true

    def load_batch_image(self, max_value):

        number = self.get_next_image_no()
        if max_value == 255:
            return self.input_images[number], self.input_interpolated_images[number], self.true_images[number]
        else:
            scale = max_value / 255.0
            return np.multiply(self.input_images[number], scale), \
                np.multiply(self.input_interpolated_images[number], scale), \
                np.multiply(self.true_images[number], scale)

    def save_input_batch_image(self, image_number, image):
        return util.save_image(self.batch_dir + "/" + INPUT_IMAGE_DIR + "/%d.bmp" % image_number, image)

    def save_interpolated_batch_image(self, image_number, image):
        return util.save_image(self.batch_dir + "/" + INTERPOLATED_IMAGE_DIR + "/%d.bmp" % image_number, image)

    def save_true_batch_image(self, image_number, image):
        return util.save_image(self.batch_dir + "/" + TRUE_IMAGE_DIR + "/%d.bmp" % image_number, image)


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
        self.index = 0

    def get_next_image_no(self):

        if self.index >= self.count:
            self.init_batch_index()

        image_no = self.batch_index[self.index]
        self.index += 1
        return image_no

    def load_batch_image(self, max_value):

        """ index won't be used. """

        image = None
        while image is None:
            image = self.load_random_patch(self.filenames[self.get_next_image_no()])

        if random.randrange(2) == 0:
            image = np.fliplr(image)

        input_image = util.resize_image_by_pil(image, 1 / self.scale)
        input_bicubic_image = util.resize_image_by_pil(input_image, self.scale)

        if max_value != 255:
            scale = max_value / 255.0
            input_image = np.multiply(input_image, scale)
            input_bicubic_image = np.multiply(input_bicubic_image, scale)
            image = np.multiply(image, scale)

        return input_image, input_bicubic_image, image

    def load_random_patch(self, filename):

        image = util.load_image(filename, print_console=False)
        height, width = image.shape[0:2]

        load_batch_size = self.batch_image_size * self.scale

        if height < load_batch_size or width < load_batch_size:
            print("Error: %s should have more than %d x %d size." % (filename, load_batch_size, load_batch_size))
            return None

        if height == load_batch_size:
            y = 0
        else:
            y = random.randrange(height - load_batch_size)

        if width == load_batch_size:
            x = 0
        else:
            x = random.randrange(width - load_batch_size)
        image = image[y:y + load_batch_size, x:x + load_batch_size, :]
        image = build_input_image(image, channels=self.channels, convert_ycbcr=True)

        return image
