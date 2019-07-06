"""
Paper: "Fast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Network"
Ver: 2.0

Functions for evaluating model performance when loading from tflite file
"""

import logging
import os
import time

import numpy as np
import tensorflow as tf

import DCSCN
from helper import args, loader, utilty as util

args.flags.DEFINE_boolean("save_results", True, "Save result, bicubic and loss images.")
args.flags.DEFINE_boolean("compute_bicubic", False, "Compute bicubic performance.")

FLAGS = args.get()


def main(not_parsed_args):
    if len(not_parsed_args) > 1:
        print("Unknown args:%s" % not_parsed_args)
        exit()

    load_and_evaluate_tflite_graph(FLAGS.output_dir, FLAGS.data_dir, FLAGS.test_dataset)

def load_and_evaluate_tflite_graph(output_dir, data_dir, test_data, model_path=os.path.join(os.getcwd(),'model_to_freeze/converted_model.tflite')):
    # https://stackoverflow.com/questions/50443411/how-to-load-a-tflite-model-in-script
    # https://www.tensorflow.org/lite/convert/python_api#tensorflow_lite_python_interpreter_
    output_directory = output_dir
    output_directory += "/" + "tflite" + "/"
    util.make_dir(output_directory)

    test_filepaths = util.get_files_in_directory(data_dir + "/" + test_data)
    total_psnr = total_ssim = total_time = 0

    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_path)
    # interpreter = tf.contrib.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for file_path in test_filepaths:
        # split filename from extension
        filename, extension = os.path.splitext(file_path)

        # prepare true image
        true_image = util.set_image_alignment(util.load_image(file_path, print_console=False), FLAGS.scale)
        
        # start the timer
        if true_image.shape[2] == 3 and FLAGS.channels == 1:
            # prepare input and ground truth images
            input_y_image = loader.build_input_image(true_image, channels=FLAGS.channels, scale=FLAGS.scale,
                                                    alignment=FLAGS.scale, convert_ycbcr=True)
            input_bicubic_y_image = util.resize_image_by_pil(input_y_image, FLAGS.scale,
                                                            resampling_method=FLAGS.resampling_method)
            true_ycbcr_image = util.convert_rgb_to_ycbcr(true_image)

            # pass inputs through the model (need to recast and reshape inputs)
            input_y_image_reshaped = input_y_image.astype('float32')
            input_y_image_reshaped = input_y_image_reshaped.reshape(1, input_y_image.shape[0], input_y_image.shape[1], FLAGS.channels)
            
            input_bicubic_y_image_reshaped = input_bicubic_y_image.astype('float32')
            input_bicubic_y_image_reshaped = input_bicubic_y_image_reshaped.reshape(1, 
                input_bicubic_y_image.shape[0], input_bicubic_y_image.shape[1], FLAGS.channels)
            
            interpreter.set_tensor(input_details[0]['index'], input_y_image_reshaped) # pass x
            interpreter.set_tensor(input_details[1]['index'], input_bicubic_y_image_reshaped) # pass x2
            
            start = time.time()
            interpreter.invoke()
            end = time.time()
            
            output_y_image = interpreter.get_tensor(output_details[0]['index']) # get y
            # resize the output into an image
            output_y_image = output_y_image.reshape(output_y_image.shape[1], output_y_image.shape[2], FLAGS.channels)

            # calculate psnr and ssim for the output
            psnr, ssim = util.compute_psnr_and_ssim(true_ycbcr_image[:, :, 0:1], output_y_image,
                                                    border_size=FLAGS.psnr_calc_border_size)

            # get the loss image
            loss_image = util.get_loss_image(true_ycbcr_image[:, :, 0:1], output_y_image,
                                        border_size=FLAGS.psnr_calc_border_size)

            # get output color image
            output_color_image = util.convert_y_and_cbcr_to_rgb(output_y_image,
                                                            true_ycbcr_image[:, :, 1:3])

            # save all images
            util.save_image(output_directory + file_path, true_image)
            util.save_image(output_directory + filename + "_input" + extension, input_y_image)
            util.save_image(output_directory + filename + "_input_bicubic" + extension, input_bicubic_y_image)
            util.save_image(output_directory + filename + "_true_y" + extension, true_ycbcr_image[:, :, 0:1])
            util.save_image(output_directory + filename + "_result" + extension, output_y_image)
            util.save_image(output_directory + filename + "_result_c" + extension, output_color_image)
            util.save_image(output_directory + filename + "_loss" + extension, loss_image)
        elapsed_time = end - start
        total_psnr += psnr
        total_ssim += ssim
        total_time += elapsed_time
    testSize = len(test_filepaths)
    print("Model Average [%s] PSNR:%f, SSIM:%f, Elapsed Time:%f" % (
        test_data, total_psnr / testSize, total_ssim / testSize, total_time / testSize))


if __name__ == '__main__':
    tf.app.run()
