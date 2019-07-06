import os, argparse

import tensorflow as tfs

dir = os.path.dirname(os.path.realpath(__file__))


def convert_graph(model_dir):
    # Converting a GraphDef from file.
    converter = tf.lite.TFLiteConverter.from_frozen_graph(
        model_dir + '/frozen_model_optimized.pb', 
        ['x','x2'], ['output'], input_shapes={"x":[1,64,64,1],"x2":[1,256,256,1]})
    converter.target_ops = [tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    open(model_dir + "/converted_model.tflite", "wb").write(tflite_model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="", help="Model folder to export")
    args = parser.parse_args()

    convert_graph(args.model_dir)