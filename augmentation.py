"""
Paper:

DCSCN: functions for data augmentation
"""
import os

import numpy as np
import tensorflow.compat.v1 as tf

from helper import args, utilty as util

args.flags.DEFINE_integer("augment_level", 4, "Augmentation level. 4:+LR/UD/LR-UD flipped, 7:+rotated")

FLAGS = args.get()


def main(not_parsed_args):
    if len(not_parsed_args) > 1:
        print("Unknown args:%s" % not_parsed_args)
        exit()

    print("Building x%d augmented data." % FLAGS.augment_level)

    training_filenames = util.get_files_in_directory(FLAGS.data_dir + "/" + FLAGS.dataset + "/")
    target_dir = FLAGS.data_dir + "/" + FLAGS.dataset + ("_%d/" % FLAGS.augment_level)
    util.make_dir(target_dir)

    for file_path in training_filenames:
        org_image = util.load_image(file_path)

        filename = os.path.basename(file_path)
        filename, extension = os.path.splitext(filename)

        new_filename = target_dir + filename
        util.save_image(new_filename + extension, org_image)

        if FLAGS.augment_level >= 2:
            ud_image = np.flipud(org_image)
            util.save_image(new_filename + "_v" + extension, ud_image)
        if FLAGS.augment_level >= 3:
            lr_image = np.fliplr(org_image)
            util.save_image(new_filename + "_h" + extension, lr_image)
        if FLAGS.augment_level >= 4:
            lr_image = np.fliplr(org_image)
            lrud_image = np.flipud(lr_image)
            util.save_image(new_filename + "_hv" + extension, lrud_image)

        if FLAGS.augment_level >= 5:
            rotated_image1 = np.rot90(org_image)
            util.save_image(new_filename + "_r1" + extension, rotated_image1)
        if FLAGS.augment_level >= 6:
            rotated_image2 = np.rot90(org_image, -1)
            util.save_image(new_filename + "_r2" + extension, rotated_image2)

        if FLAGS.augment_level >= 7:
            rotated_image1 = np.rot90(org_image)
            ud_image = np.flipud(rotated_image1)
            util.save_image(new_filename + "_r1_v" + extension, ud_image)
        if FLAGS.augment_level >= 8:
            rotated_image2 = np.rot90(org_image, -1)
            ud_image = np.flipud(rotated_image2)
            util.save_image(new_filename + "_r2_v" + extension, ud_image)


if __name__ == '__main__':
    tf.app.run()
