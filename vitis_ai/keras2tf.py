import os
import sys
import shutil
import tensorflow as tf
from tensorflow.keras import  backend
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model


tf.compat.v1.disable_eager_execution()
#========================================================================================================================================

CHKPT_MODEL_DIR = './build/checkpoints'
backend.set_learning_phase(0)

weights='./model_model/ldfgnet.hdf5'

model=tf.keras.models.load_model(weights)

model.summary()
#Check the input and output name
print ("\n TF input node name:")
print(model.inputs)
print ("\n TF output node name:")
print(model.outputs)

#========================================================================================================================================

# set up tensorflow saver object
#saver = tf.train.Saver()
saver = tf.compat.v1.train.Saver()

# fetch the tensorflow session using the Keras backend
#sess = backend.get_session()
sess = tf.compat.v1.keras.backend.get_session()

graph_def = sess.graph.as_graph_def()


layer_names=[layer.name for layer in model.layers]
#print(layer_names)
output_names=[layer.name for layer in model.outputs]
#print(output_names)


#==================================================================================================================================

save_path = saver.save(sess, os.path.join(CHKPT_MODEL_DIR, "float_model.ckpt"))
tf.train.write_graph(graph_def, CHKPT_MODEL_DIR + "/", "infer_graph.pb", as_text=False)
#tf.io.write_graph(graph_def, CHKPT_MODEL_DIR + "/", "infer_graph.pb", as_text=False)

#========================================================================================================================================

