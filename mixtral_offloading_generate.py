import torch
import torch.cuda.nvtx as nvtx
from torch.nn import functional as F
import numpy
import os, sys
import time

sys.path.append("/workspace/mixtral-offloading")

from hqq.core.quantize import BaseQuantizeConfig
from huggingface_hub import snapshot_download
from tqdm.auto import trange
from transformers import AutoConfig, AutoTokenizer
from transformers.utils import logging as hf_logging

from src.build_model import OffloadConfig, QuantConfig, build_model

#################################################

cache_path = "/ephemeral/huggingface_cache/"

model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
quantized_model_name = "lavawolfiee/Mixtral-8x7B-Instruct-v0.1-offloading-demo"
#state_path = "Mixtral-8x7B-Instruct-v0.1-offloading-demo"
quantized_model_path = os.path.join(cache_path, quantized_model_name)
model_path = os.path.join(cache_path, model_name)

_time = time.time()
config = AutoConfig.from_pretrained(quantized_model_path)
endtime = time.time()
print(f"load config time: {endtime - _time}")

###############################################################
#++# some configs for experimenting

import argparse

parser = argparse.ArgumentParser(prog='mixtral_offloading_generate')

parser.add_argument('--offload_per_layer', type=int, default=4)
parser.add_argument('--max_new_tokens', type=int, default=16)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--prompts_filepath', type=str, required=True)

args = parser.parse_args()

offload_per_layer = args.offload_per_layer
max_new_tokens = args.max_new_tokens
#device = torch.device(f"cuda:{args.device}")
device = torch.device('cuda:0')

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

_time = time.time()
model = build_model(
    device=device,
    quant_config=quant_config,
    offload_config=offload_config,
    state_path=quantized_model_path,
)
endtime = time.time()
print(f"load time: {endtime - _time}")


#################################################
#++#

with open(args.prompts_filepath) as f:
    prompts = [line.rstrip() for line in f]

#################################################
#++#

from transformers import TextStreamer
tokenizer = AutoTokenizer.from_pretrained(quantized_model_name)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
past_key_values = None
sequence = None

#################################################
#++# generate

torch.cuda.cudart().cudaProfilerStart()
for i,prompt in enumerate(prompts):
    print(f"Prompt: {prompt}")
#
    user_entry = dict(role="user", content=prompt)
    input_ids = tokenizer.apply_chat_template([user_entry], return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids)
    print("Mixtral: ", end="")
    nvtx.range_push(f"prompt_{i:03}")
    #breakpoint()
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

