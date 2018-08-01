"""
Paper: "Fast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Network"
Ver: 2.0

Functions for evaluating model performance

Put your images under data/[your dataset name]/ and specify [your dataset name] for --test_dataset.
This script will create LR images from your test dataset and evaluate the model's performance.

--save_results=True: will provide generated HR images and bi-cubic HR images.
see output/[model_name]/data/[your test data]/ for checking result images.

Also you must put same model args as you trained.

For ex, if you trained like below,
> python train.py --scale=3

Then you must run evaluate.py like below.
> python evaluate.py --scale=3 --file=your_image_file_path


If you trained like below,
> python train.py --dataset=bsd200 --layers=8 --filters=96 --training_images=30000

Then you must run evaluate.py like below.
> python evaluate.py --layers=8 --filters=96 --file=your_image_file_path
"""

import logging
import tensorflow as tf

import DCSCN
from helper import args, utilty as util

args.flags.DEFINE_boolean("save_results", True, "Save result, bicubic and loss images")

FLAGS = args.get()


def main(not_parsed_args):
	if len(not_parsed_args) > 1:
		print("Unknown args:%s" % not_parsed_args)
		exit()

	model = DCSCN.SuperResolution(FLAGS, model_name=FLAGS.model_name)
	model.build_graph()
	model.build_summary_saver()
	model.init_all_variables()

	if FLAGS.test_dataset == "all":
		test_list = ['set5', 'set14', 'bsd100']
	else:
		test_list = [FLAGS.test_dataset]

	for i in range(FLAGS.tests):
		model.load_model(FLAGS.load_model_name, trial=i, output_log=True if FLAGS.tests > 1 else False)
		for test_data in test_list:
			test(model, test_data)


def test(model, test_data):
	test_filenames = util.get_files_in_directory(FLAGS.data_dir + "/" + test_data)
	total_psnr = total_mse = 0

	for filename in test_filenames:
		if FLAGS.save_results:
			mse = model.do_for_evaluate_with_output(filename, output_directory=FLAGS.output_dir, print_console=True)
		else:
			mse = model.do_for_evaluate(filename, print_console=False)
		total_mse += mse
		total_psnr += util.get_psnr(mse, max_value=FLAGS.max_value)

	logging.info("\n=== Average [%s] MSE:%f, PSNR:%f ===" % (
		test_data, total_mse / len(test_filenames), total_psnr / len(test_filenames)))


if __name__ == '__main__':
	tf.app.run()
