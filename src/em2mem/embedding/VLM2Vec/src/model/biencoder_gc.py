from typing import List, Union, Callable, Any, Dict
from contextlib import nullcontext
from itertools import repeat
from collections import UserDict
import logging

import torch
from torch import nn, Tensor
from torch.cuda.amp import GradScaler, autocast

from src.grad_cache.context_managers import RandContext
from src.model.biencoder import BiEncoder
from utils import dist_utils
logger = logging.getLogger(__name__)


def is_binary_tensor(tensor):
    unique_elements = torch.unique(tensor)
    return torch.equal(unique_elements, torch.tensor([0, 1], dtype=tensor.dtype).to(unique_elements.device))


class BiEncoderGradCache(nn.Module):
    """
    Gradient Cache class. Implements input chunking, first graph-less forward pass, Gradient Cache creation, second
    forward & backward gradient computation. Optimizer step is not included. Native torch automatic mixed precision is
    supported. User needs to handle gradient unscaling and scaler update after a gradeitn cache step.
    """
    def __init__(
            self,
            models: List[nn.Module],
            chunk_sizes: Union[int, List[int]],
            loss_fns,
            split_input_fn: Callable[[Any, int], Any] = None,
            get_rep_fn: Callable[..., Tensor] = None,
            fp16_or_bf16: bool = False,
            dtype=torch.float32,
            scaler: GradScaler = None,
    ):
        """
        Initialize the Gradient Cache class instance.
        :param models: A list of all encoder models to be updated by the current cache.
        :param chunk_sizes: An integer indicating chunk size. Or a list of integers of chunk size for each model.
        :param loss_fns: A dict of loss functions that takes arbitrary numbers of representation tensors and
        arbitrary numbers of keyword arguments as input. It should not in any case modify the input tensors' relations
        in the autograd graph, which are later relied upon to create the gradient cache.
        :param split_input_fn: An optional function that split generic model input into chunks. If not provided, this
        class will try its best to split the inputs of supported types. See `split_inputs` function.
        :param get_rep_fn: An optional function that takes generic model output and return representation tensors. If
        not provided, the generic output is assumed to be the representation tensor.
        :param fp16_or_bf16: If True, run mixed precision training, which requires scaler to also be set.
        :param scaler: A GradScaler object for automatic mixed precision training.
        """
        super(BiEncoderGradCache, self).__init__()
        self.models = models
        self.q_encoder = models[0]
        self.k_encoder = models[1]

        if isinstance(chunk_sizes, int):
            self.chunk_sizes = [chunk_sizes for _ in range(len(models))]
        else:
            self.chunk_sizes = chunk_sizes

        self.split_input_fn = split_input_fn
        self.get_rep_fn = get_rep_fn
        self.loss_fns = loss_fns

        self.fp16_or_bf16 = fp16_or_bf16
        self.dtype = dtype
        self.scaler = scaler

        self._get_input_tensors_strict = False

    def __call__(self, *args, **kwargs):
        """
        Call the cache_step function.
        :return: Current step loss.
        """
        return self.cache_step(*args, **kwargs)

    def split_inputs(self, model_input, chunk_size: int) -> List:
        """
        Split input into chunks. Will call user provided `split_input_fn` if specified. Otherwise,
        it can handle input types of tensor, list of tensors and dictionary of tensors.
        :param model_input: Generic pytorch input.
        :param chunk_size:  Size of each chunk.
        :return: A list of chunked pytorch input.
        """
        # delegate splitting to user provided function
        if self.split_input_fn is not None:
            return self.split_input_fn(model_input, chunk_size)

        if isinstance(model_input, (dict, UserDict)) and all(isinstance(x, Tensor) for x in model_input.values()):
            keys = list(model_input.keys())
            chunked_tensors = [model_input[k].split(chunk_size, dim=0) for k in keys]
            return [dict(zip(kk, tt)) for kk, tt in zip(repeat(keys), zip(*chunked_tensors))]

        elif isinstance(model_input, list) and all(isinstance(x, Tensor) for x in model_input):
            chunked_x = [t.split(chunk_size, dim=0) for t in model_input]
            return [list(s) for s in zip(*chunked_x)]

        elif isinstance(model_input, Tensor):
            return list(model_input.split(chunk_size, dim=0))

        elif isinstance(model_input, tuple) and list(map(type, model_input)) == [list, dict]:
            args_chunks = self.split_inputs(model_input[0], chunk_size)
            kwargs_chunks = self.split_inputs(model_input[1], chunk_size)
            return list(zip(args_chunks, kwargs_chunks))

        else:
            raise NotImplementedError(f'Model input split not implemented for type {type(model_input)}')

    def get_input_tensors(self, model_input) -> List[Tensor]:
        """
        Recursively go through model input and grab all tensors, which are then used to record current device random
        states. This method will do its best to parse types of Tensor, tuple, list, dict and UserDict. Other types will
        be ignored unless self._get_input_tensors_strict is set to True, in which case an exception will be raised.
        :param model_input: input to model
        :return: all torch tensors in model_input
        """
        if isinstance(model_input, Tensor):
            return [model_input]

        elif isinstance(model_input, (list, tuple)):
            return sum((self.get_input_tensors(x) for x in model_input), [])

        elif isinstance(model_input, (dict, UserDict)):
            return sum((self.get_input_tensors(x) for x in model_input.values()), [])

        elif self._get_input_tensors_strict:
            raise NotImplementedError(f'get_input_tensors not implemented for type {type(model_input)}')

        else:
            return []

    def model_call(self, model: nn.Module, model_input):
        """
        Literally call the model's __call__ method.
        :param model: model to be called
        :param model_input: input to the model call
        :return: model output
        """
        with autocast('cuda', dtype=self.dtype) if self.fp16_or_bf16 else nullcontext():
            if isinstance(model_input, Tensor):
                return model(model_input)
            elif isinstance(model_input, list):
                return model(*model_input)
            elif isinstance(model_input, (dict, UserDict)):
                return model(**model_input)
            elif isinstance(model_input, tuple) and list(map(type, model_input)) == [list, dict]:
                model_args, model_kwargs = model_input
                return model(*model_args, **model_kwargs)
            elif isinstance(model_input, tuple):
                return model(*model_input)
            else:
                raise NotImplementedError

    def get_reps(self, model_out) -> Tensor:
        """
        Return representation tensor from generic model output
        :param model_out: generic model output
        :return: a single tensor corresponding to the model representation output
        """
        if self.get_rep_fn is not None:
            return self.get_rep_fn(model_out)
        else:
            return model_out

    def compute_loss(self, loss_mapping=None, *reps: Tensor, **loss_kwargs) -> Tensor:
        """
        Compute the loss based on the representation tensors. The tensors should be ordered same as the list of models
        registered in this GradCache class instance.
        :param reps: Representations for computing the loss.
            reps[0]: query vector, shape=[B,H]
            reps[1]: doc vector, shape=[B*num_neg,H]
        :param loss_kwargs: Keyword arguments input to the loss function.
        :return: the loss tensor.
        """
        if loss_mapping is None:
            loss_fn = self.loss_fns["distributed_inbatch_contrastive"]
            loss, loss_details = loss_fn(*reps, **loss_kwargs)
        else:
            # print('start to compute loss')
            bsz, hdim = reps[0].shape
            loss, loss_details = 0.0, {}
            preds = torch.zeros(bsz * dist_utils.get_world_size(), dtype=torch.long, device=reps[0].device)
            labels = torch.zeros(bsz * dist_utils.get_world_size(), dtype=torch.long, device=reps[0].device)
            for loss_name, data_idxs in loss_mapping.items():
                # print("get loss_name, data_indxs", loss_name, data_idxs)
                data_idxs = torch.tensor(data_idxs).to(reps[0].device)
                q = reps[0].index_select(0, index=data_idxs)
                if len(reps[1].shape) == 1 or is_binary_tensor(reps[1]):
                    # in cases d is one-hot label for classification loss
                    d = reps[1]
                else:
                    d = reps[1].view(bsz, -1, hdim).index_select(0, index=data_idxs)
                    d = d.view(-1, hdim)
                # print_rank(f"loss_name={loss_name}, q.shape={q.shape}, d.shape={d.shape}")
                _loss, _loss_details = self.loss_fns[loss_name](q, d, **loss_kwargs)
                loss += _loss
                # print("finish loss fns")
                if "labels" in _loss_details:
                    # since we compute losses per group/loss-type (stored in loss_mapping), so the data is reordered by group and we need to gather preds/labels
                    if torch.distributed.is_initialized():
                        data_idxs = data_idxs + bsz * dist_utils.get_rank()
                        # print('start to gather data index')
                        data_idxs = dist_utils.dist_gather(data_idxs)
                    # print('finish gather the data index')
                    # TODO, this might not work correctly for classification loss
                    preds.index_copy_(0, data_idxs, _loss_details["preds"])
                    labels.index_copy_(0, data_idxs, _loss_details["labels"])
                    loss_details["preds"] = preds
                    loss_details["labels"] = labels
                    # print('finish loss', data_idxs)
        # print('finish to compute_loss')
        return loss, loss_details

    def forward_no_grad(
            self,
            model: nn.Module,
            model_inputs,
    ) -> [Tensor, List[RandContext]]:
        """
        The first forward pass without gradient computation.
        :param model: Encoder model.
        :param model_inputs: Model input already broken into chunks. A tuple of two lists (ids, masks)
        :return: A tuple of a) representations and b) recorded random states.
        """
        rnd_states = []
        model_reps = []

        with torch.no_grad():
            for x in zip(*model_inputs):
                rnd_states.append(RandContext(*self.get_input_tensors(x)))
                y = self.model_call(model, x)
                model_reps.append(self.get_reps(y))

        # concatenate all sub-batch representations
        model_reps = torch.cat(model_reps, dim=0)
        return model_reps, rnd_states

    def build_cache(self, deepspeed=None, loss_mapping=None, *reps: Tensor, **loss_kwargs) -> [List[Tensor], Tensor]:
        """
        Compute the gradient cache
        :param reps: Computed representations from all encoder models
        :param loss_kwargs: Extra keyword arguments to the loss function
        :return: A tuple of a) gradient cache for each encoder model, and b) loss tensor
        """
        new_reps = []
        for r in reps:
            if isinstance(r, torch.Tensor) and r.ndim == 2:
                new_reps.append(r.detach().requires_grad_())
            elif isinstance(r, list):
                new_reps.append(torch.cat(r, dim=0))
        # reps = [r.detach().requires_grad_() for r in reps]
        reps = tuple(new_reps)
        with autocast(dtype=self.dtype) if self.fp16_or_bf16 else nullcontext():
            loss, loss_details = self.compute_loss(loss_mapping, *reps, **loss_kwargs)

        if deepspeed is None:
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
        else:
            deepspeed.backward(loss)

        cache = [r.grad for r in reps if len(r.shape) > 1 and not is_binary_tensor(r[0])]

        return cache, loss.detach(), loss_details

    def forward_backward(
            self,
            model: nn.Module,
            model_inputs,
            cached_gradients: List[Tensor],
            random_states: List[RandContext],
            no_sync_except_last: bool = False,
            deepspeed = None,
    ):
        """
        Run the second forward and the backward pass to compute gradient for a model.
        :param model: Encoder model.
        :param model_inputs: Chunked input to the encoder model.
        :param cached_gradients: Chunked gradient cache tensor for each input.
        :param random_states: Each input's device random state during the first forward.
        :param no_sync_except_last: If True, under distributed setup, only trigger gradient reduction across processes
        for the last sub-batch's forward-backward pass.
        """
        if no_sync_except_last and deepspeed is None:
            sync_contexts = [model.no_sync for _ in range(len(model_inputs) - 1)] + [nullcontext]
        else:
            sync_contexts = [nullcontext for _ in range(len(model_inputs))]

        for x, state, gradient, sync_context in zip(model_inputs, random_states, cached_gradients, sync_contexts):
            with sync_context():
                with state:
                    y = self.model_call(model, x)
                reps = self.get_reps(y)

                surrogate = torch.dot(reps.flatten(), gradient.flatten())
                if deepspeed is None:
                    surrogate.backward()
                else:
                    deepspeed.backward(surrogate)

    def cache_step(
            self,
            inputs,
            masks,
            no_sync_except_last: bool = False,
            deepspeed: object = None,
            loss_mapping = None,
            **loss_kwargs
    ) -> Tensor:
        """
        Run a cached step to compute gradient over the inputs.
        :param model_inputs: Input to each encoder model. Should be in similar order as the class's model.
        :param no_sync_except_last: If True, under distributed setup, for each model, only trigger gradient reduction
        across processes for the last sub-batch's forward-backward pass.
        :param loss_kwargs: Additional keyword arguments to the loss function.
        :return: The current's loss.
        """
        all_reps = []
        all_rnd_states = []

        inputs = [self.split_inputs(x, chunk_size) for x, chunk_size in zip(inputs, self.chunk_sizes)]
        masks = [self.split_inputs(x, chunk_size) for x, chunk_size in zip(masks, self.chunk_sizes)]

        for model, input, mask in zip(self.models, inputs, masks):
            if len(input[0].shape) == 1 or is_binary_tensor(input[0]):
                # input is label
                all_reps.append(input)
                all_rnd_states.append(input)
            else:
                model_reps, rnd_states = self.forward_no_grad(model, model_inputs=(input, mask))
                all_reps.append(model_reps)
                all_rnd_states.append(rnd_states)

        # print('start to build cache')
        cache, loss, loss_details = self.build_cache(deepspeed, loss_mapping, *all_reps, **loss_kwargs)
        cache = [c.split(chunk_size) for c, chunk_size in zip(cache, self.chunk_sizes)]

        for model, input, mask, model_cache, rnd_states in zip(self.models, inputs, masks, cache, all_rnd_states):
            self.forward_backward(model, model_inputs=list(zip(input, mask)),
                                  cached_gradients=model_cache, random_states=rnd_states,
                                  no_sync_except_last=no_sync_except_last,
                                  deepspeed=deepspeed,
                                  )

        # print('finish forward backward')
        log_stats = BiEncoder._report_train_metrics(q=all_reps[0], k=all_reps[1],
                                                    preds=loss_details["preds"], labels=loss_details["labels"],
                                                    loss_details=loss_details)
        return loss, log_stats
