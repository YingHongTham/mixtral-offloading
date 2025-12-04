#!/bin/bash

## actually turns out only need the quantized version...

#mkdir -p /ephemeral/huggingface_cache/mistralai/Mixtral-8x7B-v0.1
sudo mkdir -p /ephemeral/huggingface_cache/lavawolfiee/Mixtral-8x7B-Instruct-v0.1-offloading-demo

#python script_download_model.py
sudo ./script_download_quantized_model.py

softlink_path="Mixtral-8x7B-Instruct-v0.1-offloading-demo"
if [ ! -e $softlink_path ]; then
	ln -s /ephemeral/huggingface_cache/lavawolfiee/Mixtral-8x7B-Instruct-v0.1-offloading-demo $softlink_path
fi
