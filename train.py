"""
Paper: "Fast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Network"
Author: Jin Yamanaka
Github: https://github.com/jiny2001/dcscn-image-super-resolution
Ver: 2.0

Testing Environment: Python 3.6.1, tensorflow >= 1.3.0
"""

import logging
import sys

import tensorflow as tf

import DCSCN
from helper import args, utilty as util

FLAGS = args.get()


def main(not_parsed_args):
	if len(not_parsed_args) > 1:
		print("Unknown args:%s" % not_parsed_args)
		exit()

	model = DCSCN.SuperResolution(FLAGS, model_name=FLAGS.model_name)

	model.load_datasets("training", FLAGS.data_dir + "/" + FLAGS.dataset, FLAGS.batch_dir + "/" + FLAGS.dataset,
	                    FLAGS.batch_image_size, FLAGS.stride_size)
	model.load_datasets("test", FLAGS.data_dir + "/" + FLAGS.test_dataset, FLAGS.batch_dir + "/" + FLAGS.test_dataset,
	                    FLAGS.batch_image_size, FLAGS.stride_size)

	model.build_graph()
	model.build_optimizer()
	model.build_summary_saver()
	logging.info("\n" + str(sys.argv))
	logging.info("Test Data:" + FLAGS.test_dataset + " Training Data:" + FLAGS.dataset)

	final_mse = final_psnr = 0
	test_filenames = util.get_files_in_directory(FLAGS.data_dir + "/" + FLAGS.test_dataset)

	for i in range(FLAGS.tests):

		train(model, FLAGS, i)

		total_psnr = total_mse = 0
		for filename in test_filenames:
			mse = model.do_for_evaluate(filename, FLAGS.output_dir, output=i is (FLAGS.tests - 1), print_console=False)
			total_mse += mse
			total_psnr += util.get_psnr(mse, max_value=FLAGS.max_value)

		logging.info("\nTrial(%d) %s" % (i, util.get_now_date()))
		model.print_steps_completed(output_to_logging=True)
		logging.info("MSE:%f, PSNR:%f\n" % (total_mse / len(test_filenames), total_psnr / len(test_filenames)))

		final_mse += total_mse
		final_psnr += total_psnr

	logging.info("=== summary [%d] %s [%s] ===" % (FLAGS.tests, model.name, util.get_now_date()))
	util.print_num_of_total_parameters(output_to_logging=True)
	n = len(test_filenames) * FLAGS.tests
	logging.info("\n=== Final Average [%s] MSE:%f, PSNR:%f ===" % (FLAGS.test_dataset, final_mse / n, final_psnr / n))

	model.copy_log_to_archive("archive")


def train(model, flags, trial, load_model_name=""):

	model.init_all_variables()
	if load_model_name != "":
		model.load_model(load_model_name, output_log=True)

	model.init_train_step()
	model.init_epoch_index()
	save_meta_data = True

	while model.lr > flags.end_lr:

		model.build_input_batch(flags.batch_dir + "/" + flags.dataset + "/scale%d" % flags.scale)
		model.train_batch()

		if model.index_in_epoch >= model.train.input.count:
			model.epochs_completed += 1
			mse = model.evaluate_test_batch(save_meta_data, trial)
			save_meta_data = model.update_epoch_and_lr(mse)
			model.print_status(mse)
			model.init_epoch_index()

	model.end_train_step()
	model.save_model(trial=trial, output_log=True)

	if FLAGS.evaluate_dataset == "":
		mse = model.evaluate_test_batch()
		model.print_status(mse)
	else:
		if FLAGS.evaluate_dataset == "all":
			test_list = ['set5', 'set14', 'bsd100']
		else:
			test_list = [FLAGS.evaluate_dataset]

		for test_data in test_list:
			test(model, test_data)


def test(model, test_data):
	test_filenames = util.get_files_in_directory(FLAGS.data_dir + "/" + test_data)
	total_psnr = total_mse = 0

	for filename in test_filenames:
		mse = model.do_for_evaluate(filename, output_directory=FLAGS.output_dir, output=True)
		total_mse += mse
		total_psnr += util.get_psnr(mse, max_value=FLAGS.max_value)

	logging.info("\n=== Average [%s] MSE:%f, PSNR:%f ===" % (
		test_data, total_mse / len(test_filenames), total_psnr / len(test_filenames)))


if __name__ == '__main__':
	tf.app.run()
