#!/usr/bin/env Python
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, activation='tanh'):
        super(CustomLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation

        self.i2h = nn.Linear(input_size, 4 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size)

        if activation == 'relu':
            self.activation_func = F.relu
        elif activation == 'leaky_relu':
            self.activation_func = F.leaky_relu
        else:  # default to 'tanh'
            self.activation_func = torch.tanh

    def forward(self, input, hx):
        h, c = hx
        gates = self.i2h(input) + self.h2h(h)

        PP1 = self.i2h(input)
        pp2 = self.h2h(h)
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)

        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        cell_gate = self.activation_func(cell_gate)
        output_gate = torch.sigmoid(output_gate)

        c_next = (forget_gate * c) + (input_gate * cell_gate)
        h_next = output_gate * self.activation_func(c_next)

        return h_next, c_next
