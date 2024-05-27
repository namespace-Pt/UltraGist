import os
import torch
import numpy as np
import torch.distributed as dist
from transformers.utils import logging
from transformers import AutoTokenizer
from itertools import cycle
from typing import List

logger = logging.get_logger(__name__)


class Memory(torch.nn.Module):
    def __init__(
        self, 
        model_config, 
        k_seq_dim:int=2, 
        v_seq_dim:int=2, 
    ):
        """Setup necessary attributes."""
        super().__init__()

        self.model_config = model_config

        # initialize necessary parameters
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.num_layers = model_config.num_hidden_layers
        self.max_position_embeddings = model_config.max_position_embeddings
        self.rng = np.random.default_rng(42)

        self.ultragist_window = model_config.ultragist_window
        self.ultragist_stride = model_config.ultragist_stride
        self.ultragist_attn = model_config.ultragist_attn
        self.ultragist_ratio = model_config.ultragist_ratio
        self.ultragist_ratio_mix = model_config.ultragist_ratio_mix
        self.ultragist_param = model_config.ultragist_param
        self.ultragist_sink_size = model_config.ultragist_sink_size
        self.ultragist_attend_prev = model_config.ultragist_attend_prev

        self.ultragist_tokens = torch.zeros(1, dtype=torch.long) + model_config.vocab_size

        self._post_validation()
        self.reset()

    def _post_validation(self, verbose=True):
        assert self.ultragist_window >= self.ultragist_stride, f"Make sure the ultragist_window {self.ultragist_window} >= ultragist_stride {self.ultragist_stride}!"
        for ratio in self.ultragist_ratio:
            assert ratio >= 0, f"Make sure all ultragist ratios are greater than or equal to 0, found {self.ultragist_ratio}!"
        assert self.ultragist_attn in ["segmentation", "step-expansion", "full-coverage"], f"ultragist_attn {self.ultragist_attn} not implemented!"
        assert self.ultragist_ratio_mix in ["instance-random", "step-random", "sequence", "join"] or "adapt-" in self.ultragist_ratio_mix, f"ultragist_ratio_mix {self.ultragist_ratio_mix} not implemented!"
        if self.ultragist_ratio_mix == "join":
            # create another stream for moving gpu tensor to cpu
            # self.stream = torch.cuda.Stream()
            pass

        self._cpu = torch.device("cpu")

        if verbose:
            info = f"applying ultragist on {self.ultragist_param} (the ultragist embedding is initialized from {'bos' if self.model_config.ultragist_embed_init == 'bos' else 'eos'} embedding), with window size {self.ultragist_window}, stride {self.ultragist_stride}, {self.ultragist_attn} attention{' (attending to previous ultragists)' if self.ultragist_attend_prev else ' (no attending to previous ultragists)'}, sink size {self.ultragist_sink_size}, condensing ratio {self.ultragist_ratio} (mixed by {self.ultragist_ratio_mix})..."
            logger.info(info)

    def set(self, verbose=True, **kwargs):
        if "ultragist_ratio_mix" in kwargs and kwargs["ultragist_ratio_mix"] == "join" and self.ultragist_ratio_mix != "join":
            raise ValueError(f"You cannot switch ultragist_ratio_mix from non-join strategy to join!")
        if self.ultragist_ratio_mix == "join" and "ultragist_ratio" in kwargs and sorted(kwargs["ultragist_ratio"]) != sorted(self.ultragist_ratio):
            raise ValueError(f"You cannot change ultragist_ratio given ultragist_ratio_mix=join!")
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._post_validation(verbose=verbose)

    def reset(self):
        """Initialize attributes for a new sequence."""
        # the cursor pointing to the start of the current window
        self._start_idx = 0
        # the cursor pointing to the end of the current window
        self._end_idx = 0
        # the ultragist sizes of all strides
        self._total_ultragist_sizes = []
        # the ultragist ratios of all strides
        self._main_ultragist_sizes = []
        # the loss per batch
        self._batch_loss = None
        # the valid token number per batch
        self._valid_token_num = None
        # the step index for processing the input_ids
        self._step_idx = 0

        # used in set_compression_ratio
        self._ratio = None
        self._ultragist_ratio_iter = None

        self.all_input_ids = torch.tensor([], dtype=torch.long)
        self.all_attention_mask = torch.tensor([], dtype=torch.long)
        if hasattr(self, "all_labels"):
            del self.all_labels

        # the raw activations of recent tokens
        self.raw_activations = [(None, None) for _ in range(self.num_layers)]
        # the attention sink activations
        self.sink_activations = [(None, None) for _ in range(self.num_layers)]

        # the ultragist activations
        if self.ultragist_ratio_mix == "join":
            self.l1_to_ln_ultragist_activations = [
                [(None, None) for _ in range(self.num_layers)]
                for _ in self.ultragist_ratio
            ]
        else:
            self.l1_to_ln_ultragist_activations = [
                [(None, None) for _ in range(self.num_layers)]
            ]

    def rewind(self, size=None, trim=False):
        """
        Rewind raw activations that have not been condensed yet.

        Args:
            trim: if true, the input_ids corresponding to the raw activations are trimmed.
        """
        raw_memory_size = self.get_memory_size()[1]
        if size is None:
            size = raw_memory_size
        assert size <= raw_memory_size, f"Make sure the rewind size ({size}) is smaller or equal to the raw memory size ({raw_memory_size})!"

        if size > 0:
            self._end_idx -= size
            for layer_idx, (key, value) in enumerate(self.raw_activations):
                key = slice_tensor(key, end=-size, dim=self.k_seq_dim)
                value = slice_tensor(value, end=-size, dim=self.v_seq_dim)
                self.raw_activations[layer_idx] = (key, value)

            if trim:
                self.all_input_ids = self.all_input_ids[:, :-size]
                self.all_attention_mask = self.all_attention_mask[:, :-size]
                if hasattr(self, "all_labels"):
                    self.all_labels = self.all_labels[:, :-size]

    @property
    def finish(self):
        is_finish = self._end_idx == self.all_sequence_length

        # print(f"{dist.get_rank()} Finish: {self._end_idx}, {self.all_sequence_length}")
        # if is_finish and hasattr(self, "stream"):
        #     self.stream.synchronize()
        return is_finish
    
    def get_memory_size(self):
        ultragist_memory_size = 0
        raw_memory_size = 0
        sink_memory_size = 0
        if self.l1_to_ln_ultragist_activations[0][0][0] is not None:
            ultragist_memory_size += self.l1_to_ln_ultragist_activations[0][0][0].shape[self.k_seq_dim]
        if self.raw_activations[0][0] is not None:
            raw_memory_size += self.raw_activations[0][0].shape[self.k_seq_dim]
        if self.sink_activations[0][0] is not None:
            sink_memory_size += self.sink_activations[0][0].shape[self.k_seq_dim]
        return ultragist_memory_size, raw_memory_size, sink_memory_size
    
    def get_memory(self, ultragist_sizes=None, total_ultragist_size=None, raw_size_to_cache=None, window_size=None):
        """
        Get the compressed kv cache for generating next tokens.
        """
        past_key_values = []
        for layer_idx in range(self.num_layers):
            sink_key, sink_value = self.sink_activations[layer_idx]
            ultragist_key, ultragist_value = self.l1_to_ln_ultragist_activations[0][layer_idx]
            raw_key, raw_value = self.raw_activations[layer_idx]

            key = cat_tensor([
                sink_key, ultragist_key, raw_key,
            ], dim=self.k_seq_dim)
            value = cat_tensor([
                sink_value, ultragist_value, raw_value,
            ], dim=self.v_seq_dim)

            if ultragist_sizes is not None:
                layer_past_key_values = (key, value, ultragist_sizes, total_ultragist_size, raw_size_to_cache, window_size)
            else:
                layer_past_key_values = (key, value)

            past_key_values.append(layer_past_key_values)
        return past_key_values

    def prepare(self, input_ids, attention_mask, labels):
        """
        Prepare inputs for the model. These inputs belong to the same sequence.
        """
        assert input_ids.shape[0] == 1, "Make sure the batch size is 1!"
        assert attention_mask is None or (attention_mask == 1).all(), "Make sure there is no padding!"

        if not hasattr(self, "_device"):
            self._device = input_ids.device

        # accumulate input_ids and attention_mask
        self.all_input_ids = torch.cat([self.all_input_ids, input_ids.cpu()], dim=1)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        self.all_attention_mask = torch.cat([self.all_attention_mask, attention_mask.cpu()], dim=1)
        self.all_sequence_length = self.all_input_ids.shape[1]

        if labels is not None:
            # rotate labels in advance so that the loss of the last token is not ignored in every window
            labels = torch.cat([labels[:, 1:].cpu(), torch.tensor([-100]).expand(labels.shape[0], 1)], dim=1)
            if not hasattr(self, "all_labels"):
                self.all_labels = labels
            else:
                self.all_labels = torch.cat([self.all_labels, labels], dim=1)
            assert self.all_input_ids.shape[1] == self.all_labels.shape[1], f"Found inconsistent all_input_ids {self.all_input_ids.shape} and all_labels {self.all_labels.shape}!"

    def set_compression_ratio(self, start_idx, end_idx):
        """Choose a condensing ratio from self.ultragist_ratio"""
        def filter_ratio(ratios, stride):
            valid_ratios = []
            for ratio in ratios:
                # stride must be bigger than condensing ratio because we there must be at least one ultragist
                if stride < ratio:
                    continue
                # the stride must be evenly divisible by condensing ratio
                if ratio > 0 and (stride % ratio) != 0:
                    continue
                # when training, ratio=0 is valid if previous windows contain ultragist or later windows contain ultragist
                if ratio == 0 and self.training:
                    previous_has_zero = -1 in self._main_ultragist_sizes
                    following_has_nonzero = (start_idx + stride + self.ultragist_window) <= self.all_sequence_length
                    if previous_has_zero or (not following_has_nonzero):
                        continue
                valid_ratios.append(ratio)
            assert len(valid_ratios), f"Cannot find valid condensing ratio (among {ratios}) for stride {stride}!"
            return valid_ratios

        def get_max_length(ratios):
            max_lengths = []
            for condensing_ratio in ratios:
                if condensing_ratio > 0:
                    max_lengths.append((self.max_position_embeddings - self.ultragist_window) * condensing_ratio + self.ultragist_window)
                else:
                    max_lengths.append(self.max_position_embeddings)
            return max_lengths

        if len(self.ultragist_ratio) == 1:
            return [self.ultragist_ratio[0]]

        ratio_mix = self.ultragist_ratio_mix

        ultragist_ratio = filter_ratio(self.ultragist_ratio, self.ultragist_stride)

        if ratio_mix == "instance-random":
            if self._ratio is None:
                ultragist_ratio = self.rng.choice(ultragist_ratio, size=1).tolist()
                self._ratio = ultragist_ratio
            else:
                ultragist_ratio = self._ratio

        elif ratio_mix == "step-random":
            ultragist_ratio = self.rng.choice(ultragist_ratio, size=1).tolist()
        
        elif ratio_mix == "sequence":
            if self._ultragist_ratio_iter is None:
                self._ultragist_ratio_iter = cycle(ultragist_ratio)
            ultragist_ratio = [next(self._ultragist_ratio_iter)]
        
        elif ratio_mix == "join":
            ultragist_ratio = ultragist_ratio
        
        elif "adapt" in ratio_mix:
            if self._ratio is None:
                future_length = int(ratio_mix.split("-")[1])
                sequence_length = self.all_input_ids.shape[1] + future_length
                max_lengths = get_max_length(ultragist_ratio)
                # ascendingly sort the max lengths
                valid_max_lengths_and_indices = [x for x in enumerate(max_lengths) if x[1] >= sequence_length]
                if len(valid_max_lengths_and_indices):
                    minimum_length_index = min(valid_max_lengths_and_indices, key=lambda x: x[1])[0]
                    # use the minimal possible length for this sequence (the smallest fold ratio)
                    ultragist_ratio = [ultragist_ratio[minimum_length_index]]
                else:
                    ultragist_ratio = [max(ultragist_ratio)]
                    # logger.warning(f"Failed to find valid fold window and size for sequence length {sequence_length}, as the maximum theoretical length is {max(max_lengths)}. Fall back to use the maximum one: {ultragist_ratio}.")
                self._ratio = ultragist_ratio
            else:
                ultragist_ratio = self._ratio

        return ultragist_ratio

    def step(self):
        """
        Yield one window with the following logic:

        The window size is L, the stride is S.
        The window moves over S tokens at a time. The raw activations passed by the window are condensed according to a condensing_ratio.
        The ultragists are added if and only if the raw activations fulfill the window.
        In the future, we may switch window size to decrease cache size of raw activations.
        """
        # the starting position of the current window w.r.t. the start of the current input sequence
        start_idx = self._start_idx
        # the end position of the current window w.r.t. the start of the current input sequence
        end_idx = start_idx + self.ultragist_window

        # indicates if the current window is completely filled by raw activations and new tokens
        # we only append ultragist tokens for full windows
        if end_idx > self.all_sequence_length:
            # the input is shorter than the initial window size
            end_idx = self.all_sequence_length
            is_full_window = False
        else:
            is_full_window = True

        # NOTE: in training, the entire sequence is input to the model at once
        # In the last window, we do not need to append ultragists because they will not be used at all
        if self.training and end_idx == self.all_sequence_length:
            is_full_window = False

        # the real window size (remaining_size + new_token_size)
        window_size = end_idx - start_idx

        if is_full_window:
            ultragist_stride = self.ultragist_stride
            # a list of condensing ratios
            compression_ratios = self.set_compression_ratio(start_idx=start_idx, end_idx=end_idx)

            ultragist_sizes = []
            for condensing_ratio in compression_ratios:
                if condensing_ratio > 0:
                    # the stride must be evenly divisible by condensing_ratio
                    ultragist_sizes.append(ultragist_stride // condensing_ratio)
                else:
                    # the raw activations are used as ultragist activations
                    ultragist_sizes.append(-1)
            # forward start_idx and end_idx
            next_start_idx = start_idx + ultragist_stride
            # how many raw activations to save
            raw_size_to_cache = end_idx - next_start_idx

        else:
            # no stride because the sequence has finished
            next_start_idx = start_idx
            # cache all recent raw activations to be used in the next window
            raw_size_to_cache = window_size
            ultragist_sizes = [0]
            compression_ratios = [0]

        total_ultragist_size = sum(s for s in ultragist_sizes if s >= 0)

        past_key_values = self.get_memory(
            ultragist_sizes,
            total_ultragist_size,
            raw_size_to_cache,
            window_size
        )

        # streamingly add new input_ids
        input_ids = self.all_input_ids[:, self._end_idx: end_idx].to(self._device)
        attention_mask = self.all_attention_mask[:, self._end_idx: end_idx].to(self._device)
        if hasattr(self, "all_labels"):
            labels = self.all_labels[:, self._end_idx: end_idx].to(self._device)
        else:
            labels = None
        batch_size = input_ids.shape[0]

        # append ultragists if necessary
        if is_full_window:
            if total_ultragist_size > 0:
                input_ids = torch.cat([input_ids, self.ultragist_tokens.expand(batch_size, total_ultragist_size).to(input_ids.device, dtype=input_ids.dtype)], dim=1)
                # NOTE: prepend ultragist_memory_size 1 to attention_mask because we have past_key_values
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones(batch_size, total_ultragist_size)], dim=1)
                if labels is not None:
                    labels = torch.cat([labels, labels.new_zeros(batch_size, total_ultragist_size) - 100], dim=1)

        # prepend 1 to attention mask for previous memory
        first_key = past_key_values[0][0]
        memory_size = first_key.shape[self.k_seq_dim] if first_key is not None else 0
        if memory_size > 0:
            attention_mask = torch.cat([attention_mask.new_ones(batch_size, memory_size), attention_mask], dim=1)

        # involked in self.output()
        self._total_ultragist_sizes.append(total_ultragist_size)
        # involked in self.set_compression_ratio
        self._main_ultragist_sizes.append(ultragist_sizes[0])

        # update end_idx
        self._start_idx = next_start_idx
        self._end_idx = end_idx
        self._step_idx += 1

        # print("****************************************")
        # if is_full_window:
        #     print(f"stride:               {ultragist_stride}")
        #     print(f"compression ratios:   {compression_ratios}")
        #     print(f"ultragist_sizes:      {ultragist_sizes}")
        # print(f"input_ids:          {input_ids.shape}")
        # print(f"start_idx:          {start_idx}")
        # print(f"next_start_idx:     {next_start_idx}")
        # print(f"end_idx:            {end_idx}")
        # x = input()
        # if x == "s":
        #     return
        
        return input_ids, attention_mask, past_key_values, labels

    def update_memory(self, past_key_values):
        """
        Accumulate ultragist activations and raw activations.
        """
        for layer_idx, (key, value, ultragist_sizes, total_ultragist_size, raw_size_to_cache, window_size) in enumerate(past_key_values):
            # NOTE: the past_key_values are incrementally returned (only the new keys and values are returned)

            # key/value: (num_layer, 2, batch_size, num_head, new_seq_len, head_dim)
            # ultragist_size: how many ultragist activations are in key and value
            # raw_size_to_cache: how many raw activations should be kept

            previous_raw_key, previous_raw_value = self.raw_activations[layer_idx]

            if self._step_idx == 1:
                # save the sink activations
                # NOTE: we do not slice the key/value activations, which may cause duplication when ultragist_ratio=-1 for the first window, but it's okay
                self.sink_activations[layer_idx] = [
                    slice_tensor(key, end=self.ultragist_sink_size, dim=self.k_seq_dim),
                    slice_tensor(value, end=self.ultragist_sink_size, dim=self.v_seq_dim),
                ]

            if ultragist_sizes == [0]:
                # this means the current input does not fulfill a window
                # thus, the key and value are all raw activations, and we accumulate them until the window is fulfilled
                assert raw_size_to_cache == window_size
                raw_key = cat_tensor([
                    previous_raw_key,
                    key
                ], dim=self.k_seq_dim)
                raw_value = cat_tensor([
                    previous_raw_value, 
                    value
                ], dim=self.v_seq_dim)
                self.raw_activations[layer_idx] = (raw_key, raw_value)

            else:
                for ultragist_size_idx, ultragist_size in enumerate(ultragist_sizes):
                    # NOTE: use the correct previous_ultragist_key and value!
                    previous_ultragist_key, previous_ultragist_value = self.l1_to_ln_ultragist_activations[ultragist_size_idx][layer_idx]
                    
                    # if ultragist_size_idx == 0:
                    #     ctx_manager = nullcontext()
                    # else:
                    #     ctx_manager = torch.cuda.stream(self.stream)                    
                    # FIXME: only the first iteration works...
                    # with ctx_manager:

                    ultragist_key, ultragist_value, raw_key, raw_value = self._extract_ultragist_and_raw_memory(key, value, previous_ultragist_key, previous_ultragist_value, previous_raw_key, previous_raw_value, raw_size_to_cache, total_ultragist_size, ultragist_sizes, ultragist_size_idx)

                    self.l1_to_ln_ultragist_activations[ultragist_size_idx][layer_idx] = (ultragist_key, ultragist_value)
                    if ultragist_size_idx == 0:
                        self.raw_activations[layer_idx] = (raw_key, raw_value)
                        
                    # if ultragist_size_idx != 0:
                    #     print(self.stream.query())

    def update_loss(self, batch_loss, valid_token_num):
        """
        Accumulate loss for later perplexity computation and backward pass; past_key_values according to cache_method.
        """
        # print(f"process {dist.get_rank()}: valid_token_num: {valid_token_num}; loss {batch_loss}")
        if self._batch_loss is None:
            # NOTE: multiply valid_token_num because batch_loss is divided by it in advance
            self._batch_loss = batch_loss * valid_token_num
            self._valid_token_num = valid_token_num
        else:
            # NOTE: avoid in-place operations, otherwise there will be gradient errors in training
            self._batch_loss = self._batch_loss + batch_loss * valid_token_num
            self._valid_token_num = self._valid_token_num + valid_token_num

    def output(self, model_outputs):
        """
        Override loss with accumulated loss.
        """
        # override loss
        if self._batch_loss is not None:
            # here the batch_loss is the summation of all token losses in each element
            loss = self._batch_loss.sum() / self._valid_token_num.sum()

            # NOTE: prevent nan
            batch_loss = self._batch_loss / self._valid_token_num
            if (self._valid_token_num == 0).any():
                batch_loss = batch_loss.masked_fill(self._valid_token_num == 0, 0.)

            # NOTE: we must use dict to override values, otherwise trainer cannot find loss
            model_outputs["loss"] = loss
            model_outputs["batch_loss"] = batch_loss
            model_outputs["valid_token_num"] = self._valid_token_num

        # override last_hidden_states (used in generation)
        ultragist_size = self._total_ultragist_sizes[-1]
        # remove logits corresponding to ultragist tokens
        if ultragist_size > 0:
            model_outputs["logits"] = model_outputs["logits"][:, :-ultragist_size]

        return model_outputs

    def _extract_ultragist_and_raw_memory(self, key, value, previous_ultragist_key, previous_ultragist_value, previous_raw_key, previous_raw_value, raw_size_to_cache, total_ultragist_size, ultragist_sizes, ultragist_size_idx):
        """Extract ultragist and raw memory from the returned key and value. The raw memory is computed only if the ultragist_size_idx == 0."""
        ultragist_size = ultragist_sizes[ultragist_size_idx]
        # NOTE: ignore -1
        previous_ultragist_size = sum(x for x in ultragist_sizes[:ultragist_size_idx] if x > 0)

        if previous_ultragist_key is not None:
            target_device = previous_ultragist_key.device
        else:
            if ultragist_size_idx == 0:
                target_device = self._device
            else:
                target_device = self._cpu

        if ultragist_size == -1:
            actual_ultragist_size = self.ultragist_window - raw_size_to_cache

            # the raw activations are used as ultragist activations
            concat_raw_key = cat_tensor([
                previous_raw_key, 
                key
            ], dim=self.k_seq_dim)
            concat_raw_value = cat_tensor([
                previous_raw_value, 
                value
            ], dim=self.v_seq_dim)

            ultragist_key = cat_tensor([
                previous_ultragist_key,
                slice_tensor(concat_raw_key, end=actual_ultragist_size, dim=self.k_seq_dim).to(target_device, non_blocking=True)
            ], dim=self.k_seq_dim)
            ultragist_value = cat_tensor([
                previous_ultragist_value,
                slice_tensor(concat_raw_value, end=actual_ultragist_size, dim=self.v_seq_dim).to(target_device, non_blocking=True)
            ], dim=self.v_seq_dim)

            if ultragist_size_idx == 0:
                raw_key = slice_tensor(concat_raw_key, start=actual_ultragist_size, end=self.ultragist_window, dim=self.k_seq_dim)
                raw_value = slice_tensor(concat_raw_value, start=actual_ultragist_size, end=self.ultragist_window, dim=self.v_seq_dim)

        else:
            # [-ultragist_size:] activations are from ultragists, need to be accumulated
            # [-raw_cache_size-ultragist_size:-ultragist_size] raw activations will be cached; if they are shorter than raw_cache_size, part of the previous raw activations will also be kept
            
            ultragist_start_idx = - total_ultragist_size + previous_ultragist_size
            ultragist_end_idx = ultragist_start_idx + ultragist_size

            # NOTE: avoid end=0 for slicing
            if ultragist_end_idx == 0:
                ultragist_end_idx = None
            
            ultragist_key = cat_tensor([
                previous_ultragist_key,
                slice_tensor(key, start=ultragist_start_idx, end=ultragist_end_idx, dim=self.k_seq_dim).to(target_device, non_blocking=True)
            ], dim=self.k_seq_dim)
            ultragist_value = cat_tensor([
                previous_ultragist_value,
                slice_tensor(value, start=ultragist_start_idx, end=ultragist_end_idx, dim=self.v_seq_dim).to(target_device, non_blocking=True)
            ], dim=self.v_seq_dim)

            # the raw activations are only updated once
            if ultragist_size_idx == 0:
                if key.shape[self.k_seq_dim] < raw_size_to_cache + ultragist_size:
                    concat_raw_key = cat_tensor([
                        previous_raw_key, 
                        key
                    ], dim=self.k_seq_dim)
                    concat_raw_value = cat_tensor([
                        previous_raw_value, 
                        value
                    ], dim=self.v_seq_dim)
                    raw_key = slice_tensor(concat_raw_key, start=self.ultragist_window - raw_size_to_cache, end=self.ultragist_window, dim=self.k_seq_dim)
                    raw_value = slice_tensor(concat_raw_value, start=self.ultragist_window - raw_size_to_cache, end=self.ultragist_window, dim=self.v_seq_dim)
                else:
                    # becomes None when raw_size_to_cache = 0
                    raw_key = slice_tensor(key, start=ultragist_start_idx - raw_size_to_cache, end=ultragist_start_idx, dim=self.k_seq_dim)
                    raw_value = slice_tensor(value, start=ultragist_start_idx - raw_size_to_cache, end=ultragist_start_idx, dim=self.v_seq_dim)

        if ultragist_size_idx == 0:
            return ultragist_key, ultragist_value, raw_key, raw_value
        else:
            # NOTE: only l1 ultragist activations are kept on GPU
            return ultragist_key.detach().to(target_device, non_blocking=True), ultragist_value.detach().to(target_device, non_blocking=True), None, None
            # return ultragist_key, ultragist_value, None, None


def slice_tensor(x, start=None, end=None, dim=2):
    if x is None:
        return None
    if end == 0:
        return None
    if start == x.shape[dim]:
        return None
    if start == end:
        return None
    if dim == 2:
        if start is None and end is not None:
            return x[:, :, :end, ...]
        elif start is not None and end is None:
            return x[:, :, start:, ...]
        elif start is not None and end is not None:
            return x[:, :, start:end, ...]
    elif dim == 1:
        if start is None and end is not None:
            return x[:, :end, ...]
        elif start is not None and end is None:
            return x[:, start:, ...]
        elif start is not None and end is not None:
            return x[:, start:end, ...]
    else:
        raise NotImplementedError

def cat_tensor(list_of_tensors, dim=-1):
    list_of_tensors = [t for t in list_of_tensors if t is not None]
    if len(list_of_tensors) > 1:
        result = torch.cat(list_of_tensors, dim=dim)
    elif len(list_of_tensors) == 1:
        result = list_of_tensors[0]
    else:
        result = None
    return result

def slice_activations(activations, start=None, end=None, k_seq_dim=2, v_seq_dim=2):
    new_activations = []
    for key, value in activations:
        new_key = slice_tensor(key, start=start, end=end, dim=k_seq_dim)
        new_value = slice_tensor(value, start=start, end=end, dim=v_seq_dim)
        new_activations.append([new_key, new_value])
    return new_activations

def cat_activations(list_of_activations, k_seq_dim=2, v_seq_dim=2):
    assert all(len(x) == len(list_of_activations[0]) for x in list_of_activations), f"Make sure all activations have the same number of layers! Found {[len(x) for x in list_of_activations]}."

    new_activations = []
    for layer_idx in range(len(list_of_activations[0])):
        keys = [x[layer_idx][0] for x in list_of_activations]
        values = [x[layer_idx][1] for x in list_of_activations]

        new_key = cat_tensor(keys, dim=k_seq_dim)
        new_value = cat_tensor(values, dim=v_seq_dim)
        new_activations.append([new_key, new_value])
    return new_activations

def interleave_activations(main_activations, augment_activations, main_spans, augment_spans, k_seq_dim=2, v_seq_dim=2, device=torch.device("cuda")):
    """ Interleave main_activations and augment_activations according to main_span and augment_span.

    Args:
        main_span: a list of tuples (start_idx, end_idx). when start_idx and end_idx is None, the augment_activations will be plugged in.
        augment_span: a list of tuples (start_idx, end_idx)
    """
    assert len(main_activations) == len(augment_activations) , f"Make sure main and augment activations have the same number of layers! Found {len(main_activations)} and {len(augment_activations)}!"
    assert sum(x[0] is None and x[1] is None for x in main_spans) == len(augment_spans), f"Make sure the number of slots for augmentation (start_idx=None and end_idx=None in main_spans) matches the number of augmentations. Found {sum(x for x in main_spans if x[0] is None and x[1] is None)} slots but {len(augment_spans)} augmentations!"

    new_activations = []
    for layer_idx in range(len(main_activations)):
        main_key, main_value = main_activations[layer_idx]
        augment_key, augment_value = augment_activations[layer_idx]

        sliced_keys = []
        sliced_values = []

        augment_idx = 0
        for start, end in main_spans:
            if start is None and end is None:
                # this means the augment key/value should be plugged in
                augment_start, augment_end = augment_spans[augment_idx]
                sliced_key = slice_tensor(
                    augment_key, 
                    start=augment_start, 
                    end=augment_end,
                    dim=k_seq_dim
                ).to(device)
                sliced_value = slice_tensor(
                    augment_value, 
                    start=augment_start, 
                    end=augment_end,
                    dim=v_seq_dim
                ).to(device)

            else:
                sliced_key = slice_tensor(
                    main_key, 
                    start=start, 
                    end=end,
                    dim=k_seq_dim
                )
                sliced_value = slice_tensor(
                    main_value, 
                    start=start, 
                    end=end,
                    dim=v_seq_dim
                )

            sliced_keys.append(sliced_key)
            sliced_values.append(sliced_value)

        new_key = cat_tensor(sliced_keys, dim=k_seq_dim)
        new_value = cat_tensor(sliced_values, dim=v_seq_dim)
        new_activations.append([new_key, new_value])

    return new_activations

def softmax(x:np.ndarray, axis=-1, temperature=1):
    if isinstance(x, list):
        x = np.array(x)
    x = x / temperature
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

def l1_norm(x):
    sum_x = sum(x)
    x = [y/sum_x for y in x]
    return x