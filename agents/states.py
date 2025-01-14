import random
import numpy as np
from util import Dict
import torch
from isp_blocks import ISPBlocks


def torch_to_numpy(list_of_tensors):
    return [tensor.cpu().numpy() for tensor in list_of_tensors]


def tensor_list_to_record(list_of_tensors):
    return [tensor.detach().clone().cpu() for tensor in list_of_tensors]


def numpy_to_torch(list_of_arrays):
    return [torch.tensor(array) for array in list_of_arrays]


'''
in new settings:
state will have ->
- original input
- current step
- the landscape of steps (how many rows? depends on ISP pipeline)
- stop signal, when to stop, max iter
- past param (or just current param -- to be operated on)
- last time reward value?

state can be determined on ->
- isp pipeline
- 

each time ->
img -> para, retouch, reward, delta, selected id 

we can be selective on model inp state
but this state_supp dict must have all
-> we could unified state_supp Dict
-> different state to inp_state converter function
'''


def get_init_state_supp(img, isp: ISPBlocks, max_step=None):
    max_step = isp.num_blocks if max_step is None else max_step
    filter_params_list, _ = isp.get_random_filter_params()  # list of param tensor
    ret = Dict(
        original_img=img.to('cpu'),
        step=0,
        max_step=max_step,
        params=filter_params_list,
        last_reward=None,
        stopped=0
    )
    return ret


def states_supp_to_inp(states_supp, device="cuda"):
    """
    states_supp: list, a batch of state_supp dict
    return: tensor of shape [batch_size, N], N dim includes step, filter_params
    """
    stops = torch.tensor([d.stopped for d in states_supp]).view(-1, 1)
    steps = torch.tensor([d.step for d in states_supp]).view(-1, 1)
    params = torch.stack([torch.cat([i.view(-1) for i in d.params])
                          for d in states_supp])
    return torch.cat([stops, steps, params], dim=1).to(device)


def states_supp_to_inp_value(states_supp, device="cuda"):
    stops = torch.tensor([d.stopped for d in states_supp]).view(-1, 1)
    steps = torch.tensor([d.step for d in states_supp]).view(-1, 1)
    return torch.cat([stops, steps], dim=1).to(device)


def states_supp_to_isp_inp(states_supp):
    im_list = [d.original_img.detach().clone().to("cuda") for d in states_supp]
    return torch.stack(im_list, 0)

def states_supp_to_records(states_supp):
    for idx, state_supp in enumerate(states_supp):
        states_supp[idx].original_img = states_supp[idx].original_img.detach().clone().cpu()
        states_supp[idx].params = tensor_list_to_record(states_supp[idx].params)
        if states_supp[idx].last_reward is not None:
            states_supp[idx].last_reward = states_supp[idx].last_reward.detach().cpu()
    return states_supp


def get_updated_states_supp(states_supp, params_batch, stopped=None, last_rewards=None, device='cpu'):
    assert len(states_supp) == len(params_batch)
    for idx, state_supp in enumerate(states_supp):
        states_supp[idx].step = state_supp.step + 1
        states_supp[idx].last_reward = last_rewards[idx].to(device) if last_rewards is not None else None
        states_supp[idx].params = params_batch[idx]
        states_supp[idx].stopped = stopped[idx].item()
    return states_supp


def state_inp_func_to_dim(state_inp_func, isp: ISPBlocks):
    dim = 3
    if state_inp_func.__name__ == "states_supp_to_inp":
        dim = dim + 1 + 1 + sum(param.numel() for param in isp.get_random_filter_params()[0])
    elif state_inp_func.__name__ == "states_supp_to_inp_value":
        dim = dim + 1 + 1
    else:
        raise NotImplementedError("state_supp_to_inp function not supported")
    return dim


def isp_to_action_dim(isp: ISPBlocks):
    return sum(param.numel() for param in isp.get_random_filter_params()[0])