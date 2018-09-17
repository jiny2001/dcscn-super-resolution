"""
Paper: "Fast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Network"
Author: Jin Yamanaka
Github: https://github.com/jiny2001/dcscn-image-super-resolution
Ver: 2.0

DCSCN training functions.
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

    if FLAGS.build_batch:
        model.load_datasets(FLAGS.data_dir + "/" + FLAGS.dataset, FLAGS.batch_dir + "/" + FLAGS.dataset,
                            FLAGS.batch_image_size, FLAGS.stride_size)
    else:
        model.load_dynamic_datasets(FLAGS.data_dir + "/" + FLAGS.dataset, FLAGS.batch_image_size)
    model.build_graph()
    model.build_optimizer()
    model.build_summary_saver()

    logging.info("\n" + str(sys.argv))
    logging.info("Test Data:" + FLAGS.test_dataset + " Training Data:" + FLAGS.dataset)
    util.print_num_of_total_parameters(output_to_logging=True)

    total_psnr = total_ssim = 0

    for i in range(FLAGS.tests):
        psnr, ssim = train(model, FLAGS, i)
        total_psnr += psnr
        total_ssim += ssim

        logging.info("\nTrial(%d) %s" % (i, util.get_now_date()))
        model.print_steps_completed(output_to_logging=True)
        logging.info("PSNR:%f, SSIM:%f\n" % (psnr, ssim))

    if FLAGS.tests > 1:
        logging.info("\n=== Final Average [%s] PSNR:%f, SSIM:%f ===" % (
            FLAGS.test_dataset, total_psnr / FLAGS.tests, total_ssim / FLAGS.tests))

    model.copy_log_to_archive("archive")


def train(model, flags, trial):
    test_filenames = util.get_files_in_directory(flags.data_dir + "/" + flags.test_dataset)
    if len(test_filenames) <= 0:
        print("Can't load images from [%s]" % (flags.data_dir + "/" + flags.test_dataset))
        exit()

    model.init_all_variables()
    if flags.load_model_name != "":
        model.load_model(flags.load_model_name, output_log=True)

    model.init_train_step()
    model.init_epoch_index()
    model_updated = True

    psnr, ssim = model.evaluate(test_filenames)
    model.print_status(psnr, ssim, log=True)
    model.log_to_tensorboard(test_filenames[0], psnr, save_meta_data=True)

    while model.lr > flags.end_lr:

        model.build_input_batch()
        model.train_batch()

        if model.training_step * model.batch_num >= model.training_images:

            # one training epoch finished
            model.epochs_completed += 1
            psnr, ssim = model.evaluate(test_filenames)
            model.print_status(psnr, ssim, log=model_updated)
            model.log_to_tensorboard(test_filenames[0], psnr, save_meta_data=model_updated)
            model.save_model(trial=trial, output_log=False)

            model_updated = model.update_epoch_and_lr()
            model.init_epoch_index()

    model.end_train_step()

    # save last generation anyway
    model.save_model(trial=trial, output_log=True)

    # outputs result
    evaluate_model(model, flags.test_dataset)

    if FLAGS.do_benchmark:
        for test_data in ['set5', 'set14', 'bsd100']:
            if test_data != flags.test_dataset:
                evaluate_model(model, test_data)

    return psnr, ssim


def evaluate_model(model, test_data):
    test_filenames = util.get_files_in_directory(FLAGS.data_dir + "/" + test_data)
    total_psnr = total_ssim = 0

    for filename in test_filenames:
        psnr, ssim = model.do_for_evaluate_with_output(filename, output_directory=FLAGS.output_dir, print_console=False)
        total_psnr += psnr
        total_ssim += ssim

    logging.info("Model Average [%s] PSNR:%f, SSIM:%f" % (
        test_data, total_psnr / len(test_filenames), total_ssim / len(test_filenames)))


if __name__ == '__main__':
    tf.app.run()
