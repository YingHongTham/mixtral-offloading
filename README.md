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
