#!/home/ubuntu/myenv01/bin/python

import numpy
import os, sys

sys.path.append("/home/ubuntu/mixtral-offloading")
import torch
import torch.cuda.nvtx as nvtx
from torch.nn import functional as F
from hqq.core.quantize import BaseQuantizeConfig
from huggingface_hub import snapshot_download
#from IPython.display import clear_output
from tqdm.auto import trange
from transformers import AutoConfig, AutoTokenizer
from transformers.utils import logging as hf_logging

from src.build_model import OffloadConfig, QuantConfig, build_model

#################################################
cache_path = "/ephemeral/huggingface_cache/"

model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
quantized_model_name = "lavawolfiee/Mixtral-8x7B-Instruct-v0.1-offloading-demo"
state_path = "Mixtral-8x7B-Instruct-v0.1-offloading-demo"
quantized_model_path = os.path.join(cache_path, quantized_model_name)
model_path = os.path.join(cache_path, model_name)
#model_path = os.path.join(cache_path, quantized_model_name)

import time
_time = time.time()
config = AutoConfig.from_pretrained(quantized_model_path)
endtime = time.time()
print(f"time: {endtime - _time}")

device = torch.device("cuda:0")

###############################################################
#++# some configs for experimenting

##### Change this to 5 if you have only 12 GB of GPU VRAM #####
offload_per_layer = 6
# offload_per_layer = 5

#max_new_tokens=512 ## slow and huge profile output...
max_new_tokens=16

###############################################################

num_experts = config.num_local_experts

offload_config = OffloadConfig(
    main_size=config.num_hidden_layers * (num_experts - offload_per_layer),
    offload_size=config.num_hidden_layers * offload_per_layer,
    buffer_size=4,
    offload_per_layer=offload_per_layer,
)


attn_config = BaseQuantizeConfig(
    nbits=4,
    group_size=64,
    quant_zero=True,
    quant_scale=True,
)
attn_config["scale_quant_params"]["group_size"] = 256


ffn_config = BaseQuantizeConfig(
    nbits=2,
    group_size=16,
    quant_zero=True,
    quant_scale=True,
)
quant_config = QuantConfig(ffn_config=ffn_config, attn_config=attn_config)

#from transformers import AutoModelForCausalLM
#_time = time.time()
#model = AutoModelForCausalLM.from_pretrained(model_path, dtype="auto", device_map="auto")
#endtime = time.time()
#print(f"time: {endtime - _time}")


import src
from src.build_model import make_and_load_expert_wrapper

#from importlib import reload
#src = reload(src)
#make_and_load_expert_wrapper = src.build_model.make_and_load_expert_wrapper

#import hqq
#hqq = reload(hqq)
#BaseQuantizeConfig = hqq.core.quantize.BaseQuantizeConfig

#model_config = AutoConfig.from_pretrained(model_path)
#make_and_load_expert_wrapper(config=model_config, quant_config=quant_config,states_dir=model_path,expert_uid=(0,0),device=device)

_time = time.time()
model = build_model(
    device=device,
    quant_config=quant_config,
    offload_config=offload_config,
    state_path=state_path,
    #state_path=model_path,
)
endtime = time.time()
print(f"load time: {endtime - _time}")


#################################################
#++#

prompts = [
    "Imagine that you are a doctor, and a patient comes with the following symptoms: fever, strong headache and fatigue",
    "imbalance imbalance imbalance imbalance imbalance imbalance imbalance imbalance imbalance imbalance " + \
    "imbalance imbalance imbalance imbalance imbalance imbalance imbalance imbalance imbalance imbalance " + \
    "imbalance imbalance imbalance imbalance imbalance imbalance imbalance imbalance imbalance imbalance",
]

#################################################
#++#

from transformers import TextStreamer


tokenizer = AutoTokenizer.from_pretrained(quantized_model_name)
#tokenizer = AutoTokenizer.from_pretrained(model_name)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
past_key_values = None
sequence = None


#################################################
#++# block up all additional VRAM on device (by blocks of some size)

##shape = (2**10,2**10) ## not ok, seems it needs some last minute memory for matmul..
#shape = (2**10,2**15)
#max_num_blocks = int(48 * 2**30 / (shape[0] * shape[1]))
#xx = []
#for i in range(max_num_blocks * 2):
#    try:
#        xx.append(torch.randn(shape, device=device))
#    except Exception as e:
#        print(e)
#        break

print(*torch.cuda.memory_summary().split('\n'),sep='\n')

#################################################
#++# generate

#seq_len = 0

torch.cuda.cudart().cudaProfilerStart()
for i,prompt in enumerate(prompts):
    print(f"Prompt: {prompt}")
#
    user_entry = dict(role="user", content=prompt)
    input_ids = tokenizer.apply_chat_template([user_entry], return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids)
    print("Mixtral: ", end="")
    nvtx.range_push(f"prompt_{i:03}")
    result = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        streamer=streamer,
        do_sample=True,
        temperature=0.9,
        top_p=0.9,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_hidden_states=True,
    )
    nvtx.range_pop()
    print("\n")

torch.cuda.cudart().cudaProfilerStop()

#################################################

#https://cseweb.ucsd.edu/classes/wi15/cse262-a/static/cuda-5.5-doc/pdf/CUDA_Profiler_Users_Guide.pdf
#Page 32, Example 2
#
#COMPUTE_PROFILE_CSV
#
#gpustarttimestamp
#gridsize3d
#threadblocksize
#dynsmemperblock
#stasmemperblock
#regperthread
#memtransfersize
#memtransferdir
#streamid
#countermodeaggregate
#active_warps
#active_cycles
#
#
#-p --nvtx-capture range@domain, range, range@* none Specify NVTX range and domain to trigger the profiling session. This option is applicable only when used along with --capture-range=nvtx.

## profiling with nsys profile:
# with offload_per_layer = 4
# sudo /opt/nvidia/nsight-systems/2024.6.2/bin/nsys profile -o demo_2.nsys-rep --trace=nvtx ./notebooks/demo_2.py
# sudo /opt/nvidia/nsight-systems/2024.6.2/bin/nsys profile -o demo_2_v2.nsys-rep --trace=cuda ./notebooks/demo_2.py
# with offload_per_layer = 0
# sudo /opt/nvidia/nsight-systems/2024.6.2/bin/nsys profile -o demo_2_v3.nsys-rep --trace=cuda,nvtx ./notebooks/demo_2.py

## redo with max_new_tokens=16 (originally above 512)
# with offload_per_layer = 0,2,4,6
# sudo /opt/nvidia/nsight-systems/2024.6.2/bin/nsys profile -o demo2_len16_offload0.nsys-rep --trace=cuda,nvtx ./notebooks/demo_2.py
# sudo /opt/nvidia/nsight-systems/2024.6.2/bin/nsys profile -o demo2_len16_offload2.nsys-rep --trace=cuda,nvtx ./notebooks/demo_2.py
# sudo /opt/nvidia/nsight-systems/2024.6.2/bin/nsys profile -o demo2_len16_offload4.nsys-rep --trace=cuda,nvtx ./notebooks/demo_2.py
# sudo /opt/nvidia/nsight-systems/2024.6.2/bin/nsys profile -o demo2_len16_offload6.nsys-rep --trace=cuda,nvtx ./notebooks/demo_2.py

