#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Length regulator related modules."""

import math
from typeguard import check_argument_types
from typing import Tuple

import torch
import gw

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class Upsampling(torch.nn.Module):
    def __init__(
        self,
        idim: int,
        odim: int,
        n_layers: int = 4,
        n_chans: int = 384,
        kernel_size: int = 4,
        bias: bool = True,
    ):
        """Initilize duration predictor module.

        Args:
            idim (int): Input dimension.
            n_layers (int): Number of convolutional layers.
            n_chans (int): Number of channels of convolutional layers.
            kernel_size (int): Kernel size of convolutional layers.
            dropout_rate (float): Dropout rate.

        """
        assert check_argument_types()
        super().__init__()
        self.conv = torch.nn.ModuleList()
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [
                torch.nn.Sequential(
                    torch.nn.ConvTranspose1d(
                        in_chans,
                        n_chans,
                        kernel_size,
                        stride=2,
                        padding=(kernel_size - 2) // 2,
                        bias=bias,
                    ),
                    torch.nn.ReLU(),
                    LayerNorm(n_chans, dim=1),
                )
            ]
        self.linear = torch.nn.Linear(n_chans, odim)
        
    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor): Batch of masks indicating padded part (B, Tmax).

        Returns:
            Tensor: Batch of predicted sequences (B, Tmax, odim).

        """
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)

        xs = self.linear(xs.transpose(1, 2))  # (B, Tmax, odim)

        return xs


class LengthRegulator(torch.nn.Module):
    """Length regulator module for feed-forward Transformer.

    This is a module of length regulator described in
    `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    The length regulator expands char or
    phoneme-level embedding features to frame-level by repeating each
    feature based on the corresponding predicted durations.

    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf

    """

    def __init__(self, window_size=16, n_iter=256, sr=4):
        """Initilize length regulator module.

        Args:
            pad_value (float, optional): Value used for padding.

        """
        super().__init__()
        self.window_size = window_size
        self.n_iter = n_iter
        self.sr = sr

    def forward(self, xs, ds, is_inference, grad_stop=False):
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of sequences of char or phoneme embeddings (B, Tmax, D).
            ds (LongTensor): Batch of durations of each frame (B, Tmax, n_iter).
            alpha (float, optional): Alpha value to control speed of speech.

        Returns:
            Tensor: replicated input tensor based on durations (B, Tmax, D).

        """
        # ys = gw.stgw.stgw(ds, xs, window_size=self.window_size, n_iter=self.n_iter)
        ds = ds.div(ds.size(-1)*4)
        f = gw.gw_ode(ds[:,:,0], m=1)
        for i in range(1, ds.size(-1)):
            f = gw.gw_ode(ds[:,:,i], f=f)
        ys = gw.cubic_interpolation(xs.transpose(-1,-2), f).transpose(-1,-2)
        return ys, f
    
    
    
