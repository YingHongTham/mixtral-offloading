#!/bin/bash

## run two instances of inferencing, see how it affects runtime, offloading times etc

sudo /opt/nvidia/nsight-systems/2024.6.2/bin/nsys profile -o nsysreps/demo2_len16_offload4_instance_01.nsys-rep --trace=cuda,nvtx ./notebooks/demo_2.py &
sudo /opt/nvidia/nsight-systems/2024.6.2/bin/nsys profile -o nsysreps/demo2_len16_offload4_instance_02.nsys-rep --trace=cuda,nvtx ./notebooks/demo_2.py &
