#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Length regulator related modules."""

import math

import torch
import gw

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask


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

    def __init__(self):
        """Initilize length regulator module.

        Args:
            pad_value (float, optional): Value used for padding.

        """
        super().__init__()

    def integrate(self, ws: torch.Tensor, lengths: torch.Tensor):
        masks = make_pad_mask(lengths).to(ws.device)
        ws = torch.nn.functional.pad(ws, [0,0,1,0])[...,:-1,:]
        ws = ws.masked_fill(masks.unsqueeze(-1), 0.0)
        ws = ws - ws.sum(-2, keepdim=True) / lengths.unsqueeze(-1).unsqueeze(-1)
        ws = ws.cumsum(-2)
        ws = ws.masked_fill(masks.unsqueeze(-1), 0.0)
        ws = ws.div(ws.size(-1)*4)
        return ws
    
    def culculate_function(self, ws: torch.Tensor):
        f = gw.gw_ode(ws[:,:,0])
        for i in range(1, ws.size(-1)):
            f = gw.gw_ode(ws[:,:,i], f=f)
        return f
    
    def warp(self, xs: torch.Tensor, f:torch.Tensor):
        ys = gw.cubic_interpolation(xs, f)
        return ys

    def map(self, x_length: int, f:torch.Tensor):
        m = torch.eye(x_length, device=f.device, dtype=torch.float).unsqueeze(0)
        m = self.warp(m, f).transpose(-1, -2)
        return m
        
    def forward(self, xs: torch.Tensor, ws: torch.Tensor, x_lengths: torch.Tensor, y_lengths: torch.Tensor, plot=None):
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of sequences of char or phoneme embeddings (B, Tmax, D).
            ds (LongTensor): Batch of durations of each frame (B, Tmax, n_iter).
            alpha (float, optional): Alpha value to control speed of speech.

        Returns:
            Tensor: replicated input tensor based on durations (B, Tmax, D).

        """
        ws = ws.transpose(-1, -2)
        ws = self.integrate(ws, y_lengths)
        f = self.culculate_function(ws)
        f = f * x_lengths.unsqueeze(-1) / y_lengths.unsqueeze(-1)
        ys = self.warp(xs, f)
        return ys, f
    
