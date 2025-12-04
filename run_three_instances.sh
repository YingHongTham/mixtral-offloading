#!/bin/bash

## run four instances of inferencing, see how it affects runtime, offloading times etc

mkdir -p nsysreps/run_three_instances
sudo /opt/nvidia/nsight-systems/2024.6.2/bin/nsys profile -o nsysreps/run_three_instances/demo2_len16_offload4_instance_01.nsys-rep --trace=cuda,nvtx ./notebooks/demo_2.py &
sudo /opt/nvidia/nsight-systems/2024.6.2/bin/nsys profile -o nsysreps/run_three_instances/demo2_len16_offload4_instance_02.nsys-rep --trace=cuda,nvtx ./notebooks/demo_2.py &
sudo /opt/nvidia/nsight-systems/2024.6.2/bin/nsys profile -o nsysreps/run_three_instances/demo2_len16_offload4_instance_03.nsys-rep --trace=cuda,nvtx ./notebooks/demo_2.py &
