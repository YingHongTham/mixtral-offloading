cd /workspace/mixtral-offloading
python mixtral_offloading_generate.py \
	--offload_per_layer 4 \
	--max_new_tokens 32 \
	--device 0 \
	--prompts_filepath run_scripts/run_001_prompts.txt


