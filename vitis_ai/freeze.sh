#!/bin/bash

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
echo "START FREEZING THE GRAPH.."
echo "-----------------------------------------"

rm -rf ./build/freeze
mkdir -p ./build/freeze

freeze 2>&1 | tee ./build/logs/freeze.log

echo "-----------------------------------------"
echo "FREEZE GRAPH COMPLETED"
echo "-----------------------------------------"
