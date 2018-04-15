"""
Paper: "Fast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Network"
Author: Jin Yamanaka
Github: https://github.com/jiny2001/dcscn-image-super-resolution
Ver: 2.0

Apply Super Resolution for image file: Python 3.6.1, tensorflow 1.2.0
"""

import tensorflow as tf

import DCSCN
from helper import args

args.flags.DEFINE_string("file", "image.jpg", "Target filename")
FLAGS = args.get()


def main(_):
	model = DCSCN.SuperResolution(FLAGS, model_name=FLAGS.model_name)
	model.build_graph()
	model.build_optimizer()
	model.build_summary_saver()

	model.init_all_variables()
	model.load_model()

	model.do_for_file(FLAGS.file, FLAGS.output_dir)


if __name__ == '__main__':
	tf.app.run()
