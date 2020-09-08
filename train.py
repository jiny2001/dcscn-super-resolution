"""
Paper: "Fast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Network"
URL: https://arxiv.org/abs/1707.05425
Author: Jin Yamanaka
Github: https://github.com/jiny2001/dcscn-image-super-resolution


Testing Environment: Python 3.6.1, tensorflow 1.2.0
"""

import logging
import sys

import tensorflow.compat.v1 as tf

import DCSCN
from helper import args, utilty as util

tf.disable_eager_execution()
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

    final_psnr = final_ssim = 0
    test_filenames = util.get_files_in_directory(FLAGS.data_dir + "/" + FLAGS.test_dataset)

    for i in range(FLAGS.tests):

        train(model, FLAGS, i)

        total_psnr = total_ssim = 0
        for filename in test_filenames:
            psnr, ssim = model.do_for_evaluate(filename, FLAGS.output_dir, output=i is (FLAGS.tests - 1))
            total_psnr += psnr
            total_ssim += ssim

        logging.info("\nTrial(%d) %s" % (i, util.get_now_date()))
        model.print_steps_completed(output_to_logging=True)
        logging.info("PSNR:%f, SSIM:%f\n" % (total_psnr / len(test_filenames), total_ssim / len(test_filenames)))

        final_psnr += total_psnr
        final_ssim += total_ssim

    logging.info("=== summary [%d] %s [%s] ===" % (FLAGS.tests, model.name, util.get_now_date()))
    util.print_num_of_total_parameters(output_to_logging=True)
    n = len(test_filenames) * FLAGS.tests
    logging.info("\n=== Average [%s] PSNR:%f, SSIM:%f ===" % (FLAGS.test_dataset, final_psnr / n, final_ssim / n))


def train(model, flags, trial):
    model.init_all_variables(load_model_name=flags.load_model_name)
    model.init_train_step()
    model.init_epoch_index()
    model.print_status(model.evaluate())
    epochs_completed = 0
    min_mse = -1

    while model.lr > flags.end_lr:

        model.build_input_batch(FLAGS.batch_dir + "/" + FLAGS.dataset)
        model.train_batch()

        if epochs_completed < model.epochs_completed:
            mse = model.evaluate()
            model.update_epoch_and_lr(mse)
            model.print_status(mse)
            epochs_completed = model.epochs_completed

            if min_mse == -1 or min_mse > mse:
                model.save_model(flags.checkpoint_dir, trial)
                min_mse = mse

    model.end_train_step()
    model.save_model(flags.checkpoint_dir, trial)
    model.save_graphs(flags.graph_dir, trial)

    mse = model.evaluate()
    model.print_status(mse)

    if flags.debug:
        model.print_weight_variables()


if __name__ == '__main__':
    tf.app.run()
