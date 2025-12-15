# Mixtral offloading

forked from https://github.com/dvmazur/mixtral-offloading

To distinguish tokens, modify torch library
(here generation use model.generate(..))
look for sample method in the GenerationMixin class in:
	~/venv/lib/python3.10/site-packages/transformers/generation/utils.py 
added nvtx range push/pop in the main while loop

Experiments:
1. different offload expert numbers
observation: more offload, slower

2. congestion - several instances of the model running inference
(currently max at 3, with offload=4)
observation: also slowdown

3. batch size
(this would require more loading? cos likely that every expert is needed at each layer...)

4. topic no change vs many changes...
so far not conclusive, but it does seem that the repeating "imbalance imbalance ..." is faster
but it could be that it is the second prompt
also the first prompt was mostly on one topic as well

TODO
stream line metric collecting, summarizing
currently only look at NSight GUI, very slow
have already added the post-processing to extract csv of kernel durations,
including the prompt and token ranges
so can compare total memcpy durations dtoh/htod vs compute kernels...


==========================================================
### temporarily just copied from parent README

# Setup

if want to store docker images in another folder (e.g. /ephemeral):
	# mkdir /ephemeral/docker
	# sudo vim /etc/docker/daemon.json
-add data-root option to config (see ~/project/docker_daemon_copy.json):
	"runtimes": {
		..
	},
	"data-root": "/ephemeral/docker"
then restart the daemon:
	# systemctl restart docker

manually download image (or just do it automatically with docker run)
	$ docker pull nvcr.io/nvidia/pytorch:24.10-py3

run and execute docker container
	$ cd ~/project/setup
	$ bash docker_run.sh
	$ bash docker_exec.sh
	<in docker>
	# bash setup_docker_environment.sh ## for vim configs
	# bash install_vllm.sh
	# python -m ensurepip --default-pip ## somehow pip not installed...
	# bash install_mixtral_offload.sh ## note the numpy versioning issue


## Note

model weights are cached in /ephemeral, by setting, in vllm_generate.py:
	os.environ["VLLM_CACHE_ROOT"] = "/ephemeral/"
	os.environ["HF_HUB_CACHE"] = "/ephemeral/"

## to track individual prompts more easily, modify transformer library directly,
add nvtx ranges around the generation:
/usr/local/lib/python3.xx/dist-packages/transformers/generation/utils.py(2672)sample()
venv/lib/python3.xx/site-packages/transformers/generation/utils.py(2672)sample()
in particular around the while loop around line 2852 (search auto-regressive generation),
add a token_counter and nvtx.mark:

	## YH change
	token_counter = 0
	# auto-regressive generation
	while True:
		nvtx.mark(f"token_{token_counter:03}") ## YH change
		...

