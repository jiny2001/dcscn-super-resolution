import tensorflow as tf

from pprint import pprint

sess=tf.Session()    

# checkpoint_dir = r"./models/dcscn_L7_F32to8_G1.20_Sc4_NIN_A24_B8_PS_R1F32.ckpt"
meta_path = r"./models/dcscn_L7_F32to8_G1.20_Sc4_NIN_A24_B8_PS_R1F32.ckpt.meta"

saver = tf.train.import_meta_graph(meta_path)

saver.restore(sess,tf.train.latest_checkpoint('./models/'))

pprint([n.name for n in tf.get_default_graph().as_graph_def().node])


# #First let's load meta graph and restore weights
# saver = tf.train.import_meta_graph('my_test_model-1000.meta')
# saver.restore(sess,tf.train.latest_checkpoint('./'))

# # Add ops to save and restore all the variables.
# # saver = tf.train.Saver()

# # Later, launch the model, use the saver to restore variables from disk, and
# # do some work with the model.
# with tf.Session() as sess:
#     # Restore variables from disk.
#     sess.restore(sess, checkpoint_dir)

# print([n.name for n in tf.get_default_graph().as_graph_def().node])
