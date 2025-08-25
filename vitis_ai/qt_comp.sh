#!/bin/bash

##########################################################################
###################### COMPILATION FOR ULZCU104 #########################
##########################################################################

DIR=./build/compile/test
NAME=ldfgnet
OUT_NODE=activation/Softmax
ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json



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
###################### COMPILATION FOR ZCU104 ############################
##########################################################################

compile() {
      vai_c_tensorflow \
            --frozen_pb	./build/quantize/quantize_eval_model.pb \
            --arch		$ARCH \
            --output_dir	$DIR \
            --net_name		$NAME \
    	    --options   "{'mode':'debug', 'dump':'all', 'enable_profiling':'true'}"  ##"{'mode':'debug'}"     ####--options    "{'mode':'normal'}" 
}


##########################################################################

echo "-----------------------------------------"
echo "COMPILE STARTED.."
echo "-----------------------------------------"

compile 2>&1 | tee build/logs/compile.log
 
echo "-----------------------------------------"
echo "COMPILE COMPLETED"
echo "-----------------------------------------"


