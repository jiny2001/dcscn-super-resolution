"""
Paper: "Fast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Network"
Ver: 2.0

Script to test inference speed when loading a saved model
"""

import logging

import tensorflow as tf

import DCSCN
from helper import args, utilty as util

args.flags.DEFINE_boolean("save_results", True, "Save result, bicubic and loss images.")
args.flags.DEFINE_boolean("compute_bicubic", False, "Compute bicubic performance.")

FLAGS = args.get()


def main(not_parsed_args):
    if len(not_parsed_args) > 1:
        print("Unknown args:%s" % not_parsed_args)
        exit()

    model = DCSCN.SuperResolution(FLAGS, model_name=FLAGS.model_name)
    model.load_graph()
    model.build_summary_saver(with_saver=False) # no need because we are not saving any variables
    model.init_all_variables()

    if FLAGS.test_dataset == "all":
        test_list = ['set5', 'set14', 'bsd100']
    else:
        test_list = [FLAGS.test_dataset]

    # FLAGS.tests refer to the number of training sets to be used
    for i in range(FLAGS.tests):

        if FLAGS.compute_bicubic:
            for test_data in test_list:
                evaluate_bicubic(model, test_data)

        for test_data in test_list:
            evaluate_model(model, test_data)


def evaluate_bicubic(model, test_data):
    test_filenames = util.get_files_in_directory(FLAGS.data_dir + "/" + test_data)
    total_psnr = total_ssim = 0

    for filename in test_filenames:
        psnr, ssim = model.evaluate_bicubic(filename, print_console=False)
        total_psnr += psnr
        total_ssim += ssim

    logging.info("Bicubic Average [%s] PSNR:%f, SSIM:%f" % (
        test_data, total_psnr / len(test_filenames), total_ssim / len(test_filenames)))


def evaluate_model(model, test_data):
    test_filenames = util.get_files_in_directory(FLAGS.data_dir + "/" + test_data)
    total_psnr = total_ssim = total_time = 0

    for filename in test_filenames:
        psnr, ssim, elapsed_time = model.do_for_evaluate(filename, output_directory=FLAGS.output_dir,
                                                        print_console=False, save_output_images=FLAGS.save_results)
        total_psnr += psnr
        total_ssim += ssim
        total_time += elapsed_time

    logging.info("Model Average [%s] PSNR:%f, SSIM:%f, Elapsed Time:%f" % (
        test_data, total_psnr / len(test_filenames), total_ssim / len(test_filenames), total_time / len(test_filenames)))


if __name__ == '__main__':
    tf.app.run()
