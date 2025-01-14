import os
import cv2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import random_tensor
from isp.filters import *


class ISPBlocks():
    '''
    bridging between isp blocks and agent, offering unified interface
    ISP blocks: white box each step & black box as a whole
    differential or not

    all isp block must obey the same format ->
        single interface for blackbox, multiple for whitebox,
        torch model for diff, anything for non-diff
    an adapter can be written here to convert own interface to current format

    also, format of ISP param must also be unified
    -> just use torch tensor or have a converter here
    -> diff para: para scheme should be same - blackbox - whitebox
    -> unified states: state in replay mem & agent inp

    '''
    def __init__(self, cfg, is_blackbox=False, is_differential=True, device='cuda'):
        self.filters = []
        self.cfg = cfg
        self.is_blackbox = is_blackbox
        self.is_diff = is_differential
        self.num_blocks = 0
        self.device = device
        if not is_blackbox:
            self.init_filters(self.cfg.filters)

    def get_whitebox_filters(self):
        assert not self.is_blackbox
        return self.filters

    def get_random_filter_params(self, device='cpu'):
        # TODO!! actually to initiate param, is random okay? or zero if filter not used yet
        # TODO!! this affects state input of agent
        param_list = []
        param_dict = {}
        for filter in self.filters:
            assert isinstance(filter, Filter)
            para_size = filter.num_filter_parameters
            para_range_l = filter.range_l
            para_range_r = filter.range_r
            cur_para = random_tensor(para_range_l, para_range_r, para_size)
            if filter.init_params is not None and not self.cfg.params_random_init:
                cur_para = torch.tensor(filter.init_params, device=device)
            param_list.append((cur_para.unsqueeze(0)).to(device))
            param_dict[filter.__class__.__name__] = cur_para.tolist()
        return param_list, param_dict

    def init_filters(self, filter_list):
        self.filters = []
        for func in filter_list:
            filter = func(self.cfg, predict=True).to(self.device)
            self.__setattr__(filter.get_short_name(), filter)
            self.filters.append(filter)
        self.num_blocks = len(self.filters)

    def _run_filters_as_blackbox(self, img, params: list, change_param_dist=True):
        assert len(params) == self.num_blocks
        for idx, para in enumerate(params):
            # todo putting the change param distribution thing here may be good
            # since it may also be used when constructing dataset
            # img, _, _ = self.filters[idx](img, specified_parameter=para)
            if change_param_dist:
                img, _, _ = self.filters[idx](img, specified_parameter=self.filters[idx].change_param_dist(para))
            else:
                img, _, _ = self.filters[idx](img, specified_parameter=para)
        return img

    def run(self, img, params: list, change_param_dist=True):
        if self.is_blackbox:
            return self._run_filters_as_blackbox(img, params, change_param_dist)
        return self.filters

if __name__ == '__main__':
    isp = ISPBlocks(cfg=None)
    if isp.is_diff:
        isp.run()