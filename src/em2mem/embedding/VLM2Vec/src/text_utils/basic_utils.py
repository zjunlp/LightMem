import argparse
from contextlib import nullcontext

# import deepspeed
import collections
import json
import os
import re

import torch
from time import time
from src.text_utils.logging import get_logger
from contextlib import contextmanager
from timeit import default_timer

logger = get_logger(__name__)
########################################################################################################
## text_utils


@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def save_args_to_json(args, output_json_path):
    serializable_args = {}
    for k, v in vars(args).items():
        try:
            v = json.dumps(v)
            serializable_args[k] = v
        except Exception as e:
            continue
    with open(output_json_path, 'w') as arg_json:
        json.dump(serializable_args, arg_json)


def load_args_from_json(output_json_path):
    if os.path.isdir(output_json_path):
        output_json_path += 'train_args.json'
    with open(output_json_path, 'r') as arg_json:
        kwargs = json.load(arg_json)
    _kwargs = {}
    for k, v in kwargs.items():
        if v == 'null':
            v = None
        elif v == 'true' or v == 'false':
            v = True if v == 'true' else False
        else:
            try:
                v = eval(v)
            except ValueError:
                pass
        _kwargs[k] = v
    args = argparse.Namespace(**_kwargs)
    return args

def tensor_norm(input, input_mask=None):
    if input_mask is not None:
        _norm = torch.linalg.norm((input * input_mask.unsqueeze(-1)), dim=1)
        _norm = torch.masked_select(_norm, input_mask.bool().reshape(-1))
    else:
        _norm = torch.linalg.norm(input, dim=1, ord=2)
    return _norm.mean()


class print_time():
    def __init__(self, task):
        self.task = task

    def __enter__(self):
        print_master(self.task)
        self.t = time()

    def __exit__(self, type, value, traceback):
        print_master(f'{self.task} took {time()-self.t:.02f}s')


def print_rank(message):
    """If distributed is initialized, print the rank."""
    if torch.distributed.is_initialized():
        logger.info(f'rank{torch.distributed.get_rank()}: ' + message)
    else:
        logger.info(message)


def print_master(message):
    """If distributed is initialized print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            logger.info(message)
    else:
        logger.info(message)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def calc_gradient_norm(model, return_param_norm=False, return_details=True, is_deepspeed=False):
    '''
    return_param_norm: if True it returns the norm of parameters, otherwise grad
    No effect for DeepSpeed as it handles parameters differently
    '''
    total_norm = 0.0
    n_parameter = 0
    group_norm = collections.defaultdict(float)
    group_norm['total'] = 0.0
    for n, p in model.named_parameters():
        # with deepspeed.zero.GatheredParameters(p, modifier_rank=None) if is_deepspeed else nullcontext():
        with nullcontext():
            if p.requires_grad and p.grad is not None:
                if return_param_norm:
                    param_norm = p.detach().data.norm(p=2).item()
                else:
                    param_norm = p.grad.detach().data.norm(p=2).item()
                # param_norm = p.grad.detach().data.norm(p=float('inf'))
                total_norm += param_norm ** 2
                n_parameter += torch.numel(p.grad)
                module_name = 'q_encoder'
                # only work for BERT/mistral
                if return_details:
                    if 'embed' in n:
                        part_name = 'embeddings'
                        group_norm[f'{module_name}-{part_name}'] += param_norm
                    elif 'addon_layer' in n:
                        part_name = 'addon_layer'
                        group_norm[f'{module_name}-{part_name}'] += param_norm
                    elif 'layer' in n:
                        part_name = re.search('layers.\d+|layer.\d+', n)
                        if part_name:
                            part_name = part_name.group(0)
                        else:
                            part_name = 'unknown_group'
                    # will include a lot of stats if the model is large
                    group_norm[f'{module_name}-{part_name}'] += param_norm
                    if "model" in n:
                        part_name = n[n.rfind("model")+6:]
                    part_name = part_name.replace('module.', '').replace('.dense', '').replace('.weight', '').replace('.bias', '').replace('.pytorch', '').replace('.default', '')
                    group_norm[f'{part_name}'] += param_norm

    group_norm['total'] = total_norm ** 0.5
    return group_norm


def get_gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2).item() if p.grad is not None else 0.0
        total_norm += param_norm ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def count_parameters(model):
    total_num = sum(p.numel() for p in model.parameters())
    grad_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'#Total parameters: {total_num}')
    print(f'#Parameters require gradient: {grad_num}')
