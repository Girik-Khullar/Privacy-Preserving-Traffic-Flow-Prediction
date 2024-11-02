#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import torch
from torch import nn
import torch.nn.functional as F

class GRU(nn.Module):
    def __init__(self, input_num, hidden_num, output_num):
        super(GRU, self).__init__()
        self.hidden_size = hidden_num
        # 这里设置了 batch_first=True, 所以应该
        # 针对时间序列预测问题，相当于将时间步（seq_len）设置为 1。
        self.GRU_layer = nn.GRU(input_size=input_num, hidden_size=hidden_num, batch_first=True)
        self.output_linear = nn.Linear(hidden_num, output_num)
        self.hidden = None

    def forward(self, x):
        # h_n of shape (num_layers * num_directions, batch, hidden_size)
        # 这里不用显式地传入隐层状态 self.hidden
        x = x.view(x.shape[0], -1, x.shape[1])
        x = x.float()
        x, self.hidden = self.GRU_layer(x)
        x = self.output_linear(x)
        return x, self.hidden
