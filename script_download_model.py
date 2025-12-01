#!/home/ubuntu/myenv01/bin/python
from huggingface_hub import hf_hub_download

filenames = [
    "README.md",
    "config.json",
    "generation_config.json",
    "model-00001-of-00019.safetensors",
    "model-00002-of-00019.safetensors",
    "model-00003-of-00019.safetensors",
    "model-00004-of-00019.safetensors",
    "model-00005-of-00019.safetensors",
    "model-00006-of-00019.safetensors",
    "model-00007-of-00019.safetensors",
    "model-00008-of-00019.safetensors",
    "model-00009-of-00019.safetensors",
    "model-00010-of-00019.safetensors",
    "model-00011-of-00019.safetensors",
    "model-00012-of-00019.safetensors",
    "model-00013-of-00019.safetensors",
    "model-00014-of-00019.safetensors",
    "model-00015-of-00019.safetensors",
    "model-00016-of-00019.safetensors",
    "model-00017-of-00019.safetensors",
    "model-00018-of-00019.safetensors",
    "model-00019-of-00019.safetensors",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
]

for filename in filenames:
    hf_hub_download(repo_id="mistralai/Mixtral-8x7B-v0.1", filename=filename, local_dir="/ephemeral/mixtral/mistralai/Mixtral-8x7B-v0.1")
