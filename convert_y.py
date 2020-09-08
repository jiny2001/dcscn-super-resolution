"""
Paper: "Fast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Network"
Author: Jin Yamanaka
Github: https://github.com/jiny2001/dcscn-image-super-resolution

Convert RGB(A)-(PNG or Jpeg) Image to Y-BMP images

Put your images under data/[your dataset name]/ and specify [your dataset name] for --dataset.


"""
import os

import tensorflow.compat.v1 as tf

from helper import args, utilty as util

FLAGS = args.get()


def main(not_parsed_args):
    if len(not_parsed_args) > 1:
        print("Unknown args:%s" % not_parsed_args)
        exit()

    print("Building Y channel data...")

    training_filenames = util.get_files_in_directory(FLAGS.data_dir + "/" + FLAGS.dataset + "/")
    target_dir = FLAGS.data_dir + "/" + FLAGS.dataset + "_y/"
    util.make_dir(target_dir)

    for file_path in training_filenames:
        org_image = util.load_image(file_path)
        if org_image.shape[2] == 3:
            org_image = util.convert_rgb_to_y(org_image)

        filename = os.path.basename(file_path)
        filename, extension = os.path.splitext(filename)

        new_filename = target_dir + filename
        util.save_image(new_filename + ".bmp", org_image)


if __name__ == '__main__':
    tf.app.run()
