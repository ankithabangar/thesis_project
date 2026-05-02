"""
Patches qaeval/generation/model.py for transformers 4.x compatibility.
Run once at Docker build time (RUN python3 /tmp/patch_qaeval.py).
"""
import glob, os, sys

candidates = glob.glob(
    "/usr/local/lib/python3.9/site-packages/qaeval/generation/model.py"
)
if not candidates:
    print("ERROR: qaeval generation model.py not found", file=sys.stderr)
    sys.exit(1)

path = candidates[0]
src = open(path).read()
original = src

# 1. Remove generation_mode kwarg (removed in transformers 4.x)
src = src.replace(
    "generation_mode=GenerationMode.GREEDY_SEARCH,\n                ", ""
)

# 2. Fix use_fast=False for ElectraTokenizerFast (rejects list inputs)
src = src.replace(
    'AutoTokenizer.from_pretrained(answering_model_path)',
    'AutoTokenizer.from_pretrained(answering_model_path, use_fast=False)'
)

# 3. past_key_values rename (decoder_cached_states → past_key_values)
src = src.replace("decoder_cached_states", "past_key_values")

# 4. ModelOutput iteration yields keys in transformers 4.x — use .values()
src = src.replace(
    "for v in model_outputs",
    "for v in model_outputs.values()"
)

# 5. Rewrite _cache_dict_to_list helper to handle tuple-of-tuples format
old_cache_to_list = '''\
    def _cache_dict_to_list(self, cache):
        if cache is None:
            return None
        cache_list = []
        for layer_index in sorted(cache.keys()):
            layer_cache = cache[layer_index]
            cache_list.append(tuple([layer_cache[key] for key in sorted(layer_cache.keys())]))
        return tuple(cache_list)'''
new_cache_to_list = '''\
    def _cache_dict_to_list(self, cache):
        if cache is None:
            return None
        if isinstance(cache, (tuple, list)):
            return cache
        cache_list = []
        for layer_index in sorted(cache.keys()):
            layer_cache = cache[layer_index]
            cache_list.append(tuple([layer_cache[key] for key in sorted(layer_cache.keys())]))
        return tuple(cache_list)'''
src = src.replace(old_cache_to_list, new_cache_to_list)

# 6. Rewrite _cache_list_to_dict helper to handle tuple-of-tuples format
old_cache_to_dict = '''\
    def _cache_list_to_dict(self, cache):
        if cache is None:
            return None
        cache_dict = {}
        for layer_index, layer_cache in enumerate(cache):
            cache_dict[layer_index] = {}
            for key_index, value in enumerate(layer_cache):
                cache_dict[layer_index][key_index] = value
        return cache_dict'''
new_cache_to_dict = '''\
    def _cache_list_to_dict(self, cache):
        if cache is None:
            return None
        if isinstance(cache, (tuple, list)):
            return cache
        cache_dict = {}
        for layer_index, layer_cache in enumerate(cache):
            cache_dict[layer_index] = {}
            for key_index, value in enumerate(layer_cache):
                cache_dict[layer_index][key_index] = value
        return cache_dict'''
src = src.replace(old_cache_to_dict, new_cache_to_dict)

# 7. Rewrite take_step to avoid positional-embedding IndexError with use_cache.
#    Instead of incremental decoding, accumulate the full sequence each step.
old_take_step = '''\
    def take_step(
        self,
        last_predictions: torch.Tensor,
        state: Dict[str, torch.Tensor],
        step: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        input_ids = last_predictions.unsqueeze(1)
        decoder_past_key_values = self._cache_list_to_dict(state.get("past_key_values"))
        encoder_outputs = state.get("encoder_outputs")
        if encoder_outputs is not None:
            encoder_outputs = ((encoder_outputs,),)
        output = self._generation_model(
            input_ids=input_ids,
            encoder_outputs=encoder_outputs,
            past_key_values=decoder_past_key_values,
            use_cache=True,
        )
        logits = output.logits
        log_probs = torch.nn.functional.log_softmax(logits[:, -1, :], dim=-1)
        state["past_key_values"] = self._cache_dict_to_list(output.past_key_values)
        if encoder_outputs is None:
            state["encoder_outputs"] = output.encoder_last_hidden_state
        return log_probs, state'''
new_take_step = '''\
    def take_step(
        self,
        last_predictions: torch.Tensor,
        state: Dict[str, torch.Tensor],
        step: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Accumulate decoder tokens to avoid positional-embedding IndexError
        # that occurs with incremental use_cache decoding in transformers 4.x.
        prev_ids = state.get("decoder_input_ids")
        new_token = last_predictions.unsqueeze(1)
        if prev_ids is None:
            decoder_input_ids = new_token
        else:
            decoder_input_ids = torch.cat([prev_ids, new_token], dim=1)
        state["decoder_input_ids"] = decoder_input_ids

        encoder_hidden = state.get("encoder_outputs")
        if encoder_hidden is not None:
            from transformers.modeling_outputs import BaseModelOutput
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)
        else:
            encoder_outputs = None

        output = self._generation_model(
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            use_cache=False,
        )
        logits = output.logits
        log_probs = torch.nn.functional.log_softmax(logits[:, -1, :], dim=-1)
        if encoder_hidden is None:
            state["encoder_outputs"] = output.encoder_last_hidden_state
        return log_probs, state'''
src = src.replace(old_take_step, new_take_step)

if src == original:
    print("WARNING: no patches applied — source may have already been patched or changed")
else:
    open(path, "w").write(src)
    print(f"Patched {path} successfully")
