"""
Paper: "Fast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Network"

functions for evaluating results
"""

import logging

import tensorflow as tf

import DCSCN
from helper import args, utilty as util

args.flags.DEFINE_boolean("save_results", False, "Save result, bicubic and loss images")

FLAGS = args.get()


def main(not_parsed_args):
	if len(not_parsed_args) > 1:
		print("Unknown args:%s" % not_parsed_args)
		exit()

	# modifying process/build options for faster processing
	if FLAGS.load_model_name == "":
		FLAGS.load_model_name = "default"
	FLAGS.save_loss = False
	FLAGS.save_weights = False
	FLAGS.save_images = False

	logging.info("evaluate model performance")
	test_filenames = util.get_files_in_directory(FLAGS.data_dir + "/" + FLAGS.test_dataset)

	model = DCSCN.SuperResolution(FLAGS, model_name=FLAGS.model_name)
	model.build_graph()
	model.build_summary_saver()
	model.init_all_variables(load_model_name=FLAGS.load_model_name)

	total_psnr = total_mse = 0

	for filename in test_filenames:
		mse = model.do_super_resolution_for_evaluate(filename, output_folder=FLAGS.output_dir, output=FLAGS.save_results)
		total_mse += mse
		total_psnr += util.get_psnr(mse)

	logging.info("\n=== Average [%s] MSE:%f, PSNR:%f ===" % (
		FLAGS.test_dataset, total_mse / len(test_filenames), total_psnr / len(test_filenames)))


if __name__ == '__main__':
	tf.app.run()
