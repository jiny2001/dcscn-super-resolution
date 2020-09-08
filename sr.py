"""
Paper: "Fast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Network"
Author: Jin Yamanaka
Github: https://github.com/jiny2001/dcscn-image-super-resolution
Ver: 2.0

Apply Super Resolution for image file.

--file=[your image filename]: will generate HR images.
see output/[model_name]/ for checking result images.

Also you must put same model args as you trained.

For ex, if you trained like below,
> python train.py --scale=3

Then you must run sr.py like below.
> python sr.py --scale=3 --file=your_image_file_path


If you trained like below,
> python train.py --dataset=bsd200 --layers=8 --filters=96 --training_images=30000

Then you must run sr.py like below.
> python sr.py --layers=8 --filters=96 --file=your_image_file_path

"""

import tensorflow.compat.v1 as tf

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
