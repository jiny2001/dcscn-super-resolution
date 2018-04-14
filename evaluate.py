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

    model = DCSCN.SuperResolution(FLAGS, model_name=FLAGS.model_name)
    model.build_graph()
    model.build_summary_saver()
    model.init_all_variables()

    logging.info("evaluate model performance")

    if FLAGS.test_dataset == "all":
        test_list = ['set5', 'set14', 'bsd100']
    else:
        test_list = [FLAGS.test_dataset]

    for i in range(FLAGS.tests):
        model.load_model(FLAGS.load_model_name, i, True if FLAGS.tests > 1 else False)
        for test_data in test_list:
            test(model, test_data)


def test(model, test_data):
    test_filenames = util.get_files_in_directory(FLAGS.data_dir + "/" + test_data)
    total_psnr = total_mse = 0

    for filename in test_filenames:
        mse = model.do_for_evaluate(filename, output_directory=FLAGS.output_dir, output=FLAGS.save_results)
        total_mse += mse
        total_psnr += util.get_psnr(mse, max_value=FLAGS.max_value)

    logging.info("\n=== Average [%s] MSE:%f, PSNR:%f ===" % (
        test_data, total_mse / len(test_filenames), total_psnr / len(test_filenames)))


if __name__ == '__main__':
    tf.app.run()
