#!/bin/bash

##########################################################################
###################### COMPILATION FOR ULZCU104 #########################
##########################################################################
#OUT_NODE=dense/BiasAdd
#OUT_NODE=activation_37/Softmax
DIR=./build/compile/test1_4_2
NAME=qcomp
OUT_NODE=activation/Softmax
ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json
#ARCH=/Ultra96.json
NAME=qcomp


compile() {
      vai_c_tensorflow \
            --frozen_pb	./build/quantize/quantize_eval_model.pb \
            --arch		$ARCH \
            --output_dir	$DIR \
            --net_name		$NAME \
    	      --options   "{'mode':'debug', 'dump':'all', 'enable_profiling':'true'}"  ##"{'mode':'debug'}"     ####--options    "{'mode':'normal'}" 
}

rm -rf $DIR
mkdir -p $DIR

# convert keras model to tensor flow graph

run_quant() {
 
  # log the quantizer version being used
  vai_q_tensorflow --version
  
  # quantize
  vai_q_tensorflow quantize \
    		--input_frozen_graph	./build/freeze/frozen_graph.pb \
		--input_fn		graph_input_fn.calib_input \
		--output_dir		./build/quantize \
	  	--input_nodes		input_1 \
		--output_nodes		$OUT_NODE \
		--input_shapes		?,128,128,3 \
		--calib_iter		10 \
		--weight_bit		8 \
		--activation_bit	8 \
    		--gpu			0,1
}


keras2tf() 
{
  python keras2tf.py 
}


eval_quant() 
{
  python qeval.py 
}



freeze() 
{
  freeze_graph \
    --input_graph	./build/checkpoints/infer_graph.pb \
    --input_meta_graph	./build/checkpoints/float_model.ckpt.meta \
    --input_checkpoint	./build/checkpoints/float_model.ckpt \
    --output_graph	./build/freeze/frozen_graph.pb \
    --output_node_names	$OUT_NODE \
    --input_binary	true
}


####################################################################################

echo "-----------------------------------------"
echo "CONVERTING KERAS MODEL TO TF CHECKPOINT.."
echo "-----------------------------------------"

rm -rf ./build/logs
mkdir -p ./build/logs

rm -rf ./build/checkpoints
mkdir -p ./build/checkpoints


keras2tf 2>&1 | tee ./build/logs/keras2tf.log


echo "-----------------------------------------"
echo "FINISHED KERAS TO TF CHECKPOINT CONVERSION..."
echo "-----------------------------------------"

######################################################################################

echo "-----------------------------------------"
echo "START FREEZING THE GRAPH.."
echo "-----------------------------------------"

rm -rf ./build/freeze
mkdir -p ./build/freeze

freeze 2>&1 | tee ./build/logs/freeze.log

echo "-----------------------------------------"
echo "FREEZE GRAPH COMPLETED"
echo "-----------------------------------------"

######################################################################################
######################################################################################



echo "-----------------------------------------"
echo "QUANTIZATION STARTED.."
echo "-----------------------------------------"

rm -rf ./build/quantize
mkdir -p ./build/quantize

run_quant 2>&1 | tee ./build/logs/quant.log
 
echo "-----------------------------------------"
echo "QUANTIZATION SUCCESSFULLY COMPLETED"
echo "-----------------------------------------"

##########################################################################

eval_quant 2>&1 | tee ./build/logs/evalquant.log

##########################################################################
###################### COMPILATION FOR ZCU104 ############################
##########################################################################




##########################################################################

echo "-----------------------------------------"
echo "COMPILE STARTED.."
echo "-----------------------------------------"

compile 2>&1 | tee build/logs/compile.log
 
echo "-----------------------------------------"
echo "COMPILE COMPLETED"
echo "-----------------------------------------"


