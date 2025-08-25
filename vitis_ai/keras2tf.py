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

#weights='./saved_model/kv_dense121_87.2_128_16hw.hdf5'  #120
#weights='./saved_model/kv_dense121_90.5_128_16cn.hdf5'  #120
#weights='./saved_model/kv_vgg16_85.5_128_16.hdf5'
#weights='./saved_model/kv_mobv1_87.6_128.hdf5'
#weights='./saved_model/kvf_mobv1_92_128.hdf5'
#weights='./saved_model/kv_resnet50_88.3_128_16.hdf5'  #48
#weights='./saved_model/kv_mobv2_92.3_128_16.hdf5'
#weights='./saved_model/kv_xception_93.2_128_16.hdf5'
###weights='./saved_model/kv_xception_97.1_200.hdf5'  #32
#weights='./saved_model/kvftype7_97.4 and 98kvf_488noattn_200.hdf5'
#weights='./saved_model/kv_ccsn_extn_962_128_16_adam.hdf5'
#weights='./saved_model/kv_ccsn_extn_96.7_128_8_add.hdf5'

#weights='./saved_model/tf15_vgg_94.hdf5'
#weights='./saved_model/tf15_resnet_91.1.hdf5'       #48
#weights='./saved_model/tf15_mob_93.8.hdf5'
#weights='./saved_model/tf15_mobv2_93.7.hdf5'
#weights='./saved_model/tf15_dense_94.6.hdf5'  #120
#weights='./saved_model/tf15_xcept_95.4.hdf5'   #32
#weights='./saved_model/tf15_ccsn_96.4_add.hdf5'
#weights='./saved_model/new_test_mobv2xcep_dense_97.3_kvf.hdf5'
#weights='./saved_model/cc_mobv1.hdf5'
#weights='./saved_model/cc_mobv2.hdf5'
#weights='./saved_model/cc_hybrid.hdf5'
#weights='./saved_model/hybrid_test_4.hdf5'
#weights='./saved_model/ntest1.hdf5'
weights='./saved_model/tf15_ccsn_96.4_add.hdf5'

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

