#!/usr/bin/bash

## name of the form run_???
run_name=$1
outdir="nsysreps/${run_name}_metrics"
out_report="${outdir}/${run_name}_report.nsys-rep"
mkdir -p ${outdir}

if [ ! -f run_scripts/${run_name}.sh ]; then
	echo "run_scripts/${run_name}.sh not found"
fi

## non-docker only-virtual-env version needed sudo, and bin different path
# sudo /opt/nvidia/nsight-systems/2024.6.2/bin/nsys profile \
/usr/local/cuda/bin/nsys profile \
	--force-overwrite true \
	-o $out_report \
	--trace-fork-before-exec=true \
	--cuda-graph-trace=node \
	--trace=cuda,nvtx,osrt \
	bash run_scripts/${run_name}.sh
#python mixtral_offloading_generate.py


RESULT=$?
if [ $RESULT -ne 0 ]; then
	exit 1
fi

## extract
nsys stats --force-export=true --report cuda_gpu_trace --input $out_report --format csv --output $outdir/
nsys recipe nvtx_gpu_proj_trace --force-overwrite --input $out_report --csv --output $outdir/nvtx_gpu_proj_trace
nsys recipe nccl_gpu_overlap_trace --force-overwrite --input $out_report --output $outdir/nccl_gpu_overlap_trace
nsys recipe nccl_gpu_time_util_map --force-overwrite --input $out_report --output $outdir/nccl_gpu_time_util_map
