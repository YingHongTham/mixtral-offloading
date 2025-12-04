#!/bin/bash

run_name=$1
outdir="nsysreps/${run_name}_metrics"
out_report="$outdir/${run_name}_report.nsys-rep"
mkdir -p $outdir
sudo /opt/nvidia/nsight-systems/2024.6.2/bin/nsys profile \
	-o $out_report \
	--cuda-graph-trace=node \
	--trace=cuda,nvtx,osrt \
	./notebooks/demo_2.py

RESULT=$?
if [ $RESULT -ne 0 ]; then
	exit 1
fi

nsys stats --report cuda_gpu_trace --input $out_report --format csv --output $outdir/
nsys recipe nvtx_gpu_proj_trace --input $out_report --csv --output $outdir/nvtx_gpu_proj_trace
