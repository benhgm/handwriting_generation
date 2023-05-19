"""
Modules for performing meta learning.

Reference source code:
https://github.com/fmu2/PyTorch-MAML/blob/master/models/modules.py
"""
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn.utils.stateless import functional_call

from collections import OrderedDict

__all__ = ['Module', 'get_child_dict', 'Linear']

def bivariate_gaussian_nll(pi, mu, sigma, rho, eos, x, x_mask):
    # gaussian part
    x1 = x[:, :, 1].unsqueeze(-1)
    x2 = x[:, :, 2].unsqueeze(-1)

    mu1 = mu[:, :, :, 0]
    mu2 = mu[:, :, :, 1]

    sigma1 = sigma[:, :, :, 0]
    sigma2 = sigma[:, :, :, 1]

    # end of stroke part
    x3 = x[:, :, 0].unsqueeze(-1)
    eos_loss = (x3 * eos + (1 - x3) * (1 - eos)).log().squeeze(-1)

    Z = (torch.pow((x1 - mu1), 2) / torch.pow(sigma1.exp(), 2)) + \
        (torch.pow((x2 - mu2), 2) / torch.pow(sigma2.exp(), 2)) - \
        ((2 * rho * (x1 - mu1) * (x2 - mu2)) / (sigma1 + sigma2).exp())
    
    pi_term = - torch.log((torch.tensor(2 * torch.pi)))
    mog_lik1 = pi_term - sigma1 - sigma2 - 0.5*((1 - rho ** 2).log())
    mog_lik2 = Z/(2 * (1 - rho ** 2))
    gaussian = ((pi.log() + (mog_lik1 - mog_lik2)).exp().sum(dim=-1) + 1e-20).log()
    
    return -((eos_loss*x_mask).sum() + (gaussian*x_mask).sum())

def mixture_of_bivariate_normal_sample(pi, mu, sigma, rho, eps=1e-6, bias=0):
    batch_size = mu.shape[0]
    ndims = pi.dim()
    
    # Sample mixture using mixture probabilities
    categorical = Categorical(pi)
    mixture_idx = categorical.sample()
    # mixture_idx = pi.multinomial(1).squeeze(1)

    # Index the correct mixture component
    mu, sigma, rho = [
        x[torch.arange(mixture_idx.shape[0]), mixture_idx]
        for x in [mu, sigma, rho]
    ]

    # Calculate biased variances
    sigma = sigma - bias

    # Sample from bivariate normal distribution
    mu1 = mu[:, 0]
    mu2 = mu[:, 1]

    sigma1 = sigma[:, 0]
    sigma2 = sigma[:, 1]

    v1 = sigma1.exp() ** 2
    v2 = sigma2.exp() ** 2
    c = rho * sigma1.exp() * sigma2.exp()
    cov1 = torch.stack([v1, c], dim=1)
    cov2 = torch.stack([c, v2], dim=1)
    cov = torch.stack([cov1, cov2], dim=-1)

    norm = MultivariateNormal(mu, cov)
    sample = norm.sample()

    return sample


def get_child_dict(params, key=None):
    """
    Constructs parameter dictionary for a network module

    Args:
        params (dict): parent dictionary of named parameters
        key (str, optional): a key that specifie the root of the child dicitionary
    """
    if params is None:
        return None
    if key is None or (isinstance(key, str) and key == ''):
        return params
    
    key_re = re.compile(r'^{re.escape(key)}\.(.+)')
    if not any(filter(key_re.match, params.keys())):
        key_re = re.compile(r'^module\.{re.escape(key)}\.(.+)')

    child_dict = OrderedDict(
        (key_re.sub(r'\1', k), value) for (k, value) in params.items() if key_re.match(k) is not None
    )
    return child_dict


class Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.efficient = False
        self.first_pass = True
    
    def go_efficient(self, mode=True):
        """
        Switch on / off gradient checkpointing
        """
        self.efficient = mode
        for m in self.children():
            if isinstance(m, Module):
                m.go_efficient(mode)

    def is_first_pass(self, mode=True):
        """
        Tracks the progress of forward and backward pass when gradient checkpointing is enabled
        """
        self.first_pass = mode
        for m in self.children():
            if isinstance(m, Module):
                m.is_first_pass(mode)

class Linear(nn.Linear, Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)
    
    def forward(self, x, params=None, episode=None):
        if params is None:
            x = super(Linear, self).forward(x)
        else:
            weight, bias = params.get('weight'), params.get('bias')
            if weight is None:
                weight = self.weight
            if bias is None:
                bias = self.bias
            x = F.linear(x, weight, bias)
        return x
    
class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()

        self.lstm_cell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size, bias=bias)
    
    def forward(self, x, hidden_states, params=None, episode=None):
        if params is None:
            x, hidden_states = self.lstm_cell(x, hidden_states)
        else:
            weight_ih = params.get('weight_ih')
            weight_hh = params.get('weight_hh')
            bias_ih = params.get('bias_ih')
            bias_hh = params.get('bias_hh')
            
            if weight_ih is None:
                weight_ih = self.lstm_cell.weight_ih
            if weight_hh is None:
                weight_hh = self.lstm_cell.weight_hh
            if bias_ih is None:
                bias_ih = self.lstm_cell.bias_ih
            if bias_hh is None:
                bias_hh = self.lstm_cell.bias_hh

            new_params = OrderedDict()
            new_params = {
                "weight_ih": weight_ih,
                "weight_hh": weight_hh,
                "bias_ih": bias_ih,
                "bias_hh": bias_hh
            }

            x, hidden_states = functional_call(self.lstm_cell, new_params, (x, hidden_states))
        
        return x, hidden_states

class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True):
        super().__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=bias, batch_first=batch_first)
    
    def forward(self, x, hidden_states, params=None, episode=None):
        if params is None:
            x, hidden_states = self.lstm(x, hidden_states)
        else:
            weight_ih = params.get('weight_ih_l0')
            weight_hh = params.get('weight_hh_l0')
            bias_ih = params.get('bias_ih_l0')
            bias_hh = params.get('bias_hh_l0')
            
            if weight_ih is None:
                weight_ih = self.lstm.weight_ih_l0
            if weight_hh is None:
                weight_hh = self.lstm.weight_hh_l0
            if bias_ih is None:
                bias_ih = self.lstm.bias_ih_l0
            if bias_hh is None:
                bias_hh = self.lstm.bias_hh_l0
            
            new_params = {
                "weight_ih": weight_ih,
                "weight_hh": weight_hh,
                "bias_ih": bias_ih,
                "bias_hh": bias_hh
            }

            x, hidden_states = functional_call(nn.LSTMCell, new_params, (x, (hidden_states)))
        
        return x, hidden_states



if __name__ == "__main__":
    import torch

    x = torch.randn((1, 3))
    lstm = LSTMCell(3, 400, 1)

    for n, p in lstm.named_parameters():
        print(n)