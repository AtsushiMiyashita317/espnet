#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Positionwise feed forward layer definition."""

import torch


class Reparametrize(torch.nn.Module):
    """Positionwise feed forward layer.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim, scale=1e-1, offset=-1):
        """Construct an PositionwiseFeedForward object."""
        super(Reparametrize, self).__init__()
        self.scale = scale
        self.offset = offset
        self.linear_mu = torch.nn.Linear(idim, idim)
        self.linear_ln_var = torch.nn.Linear(idim, idim)
        
    def forward(self, x, grad_rate=1.0):
        """Forward function."""
        mu = self.linear_mu(x)
        ln_var = self.linear_ln_var(x)*self.scale+self.offset
        x = mu + torch.randn_like(mu)*ln_var.mul(0.5).exp()
        return x, grad_rate*mu+(1-grad_rate)*mu.detach(), grad_rate*ln_var+(1-grad_rate)*ln_var.detach()
    