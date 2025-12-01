# fix numpy in colab
import numpy
#from IPython.display import clear_output

## fix triton in colab
#!export LC_ALL="en_US.UTF-8"
#!export LD_LIBRARY_PATH="/usr/lib64-nvidia"
#!export LIBRARY_PATH="/usr/local/cuda/lib64/stubs"
#!ldconfig /usr/lib64-nvidia
#
#!git clone https://github.com/dvmazur/mixtral-offloading.git --quiet
#!cd mixtral-offloading && pip install -q -r requirements.txt
#!huggingface-cli download lavawolfiee/Mixtral-8x7B-Instruct-v0.1-offloading-demo --quiet --local-dir Mixtral-8x7B-Instruct-v0.1-offloading-demo
#
#clear_output()

#################################################

import os, sys

sys.path.append("/home/ubuntu/mixtral-offloading")
import torch
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

##### Change this to 5 if you have only 12 GB of GPU VRAM #####
offload_per_layer = 4
# offload_per_layer = 5
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

from transformers import TextStreamer


tokenizer = AutoTokenizer.from_pretrained(quantized_model_name)
#tokenizer = AutoTokenizer.from_pretrained(model_name)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
past_key_values = None
sequence = None

seq_len = 0
while True:
    print("User: ", end="")
    user_input = input()
    print("\n")

    user_entry = dict(role="user", content=user_input)
    input_ids = tokenizer.apply_chat_template([user_entry], return_tensors="pt").to(device)

    if past_key_values is None:
        attention_mask = torch.ones_like(input_ids)
    else:
        seq_len = input_ids.size(1) + past_key_values[0][0][0].size(1)
        attention_mask = torch.ones([1, seq_len - 1], dtype=torch.int, device=device)

    print("Mixtral: ", end="")
    result = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        streamer=streamer,
        do_sample=True,
        temperature=0.9,
        top_p=0.9,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_hidden_states=True,
    )
    print("\n")

    sequence = result["sequences"]
    past_key_values = result["past_key_values"]

#################################################
#################################################

