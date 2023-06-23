#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Length regulator related modules."""

import logging

import torch
import gw


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

    def forward(self, xs, ds, grad_stop=False):
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of sequences of char or phoneme embeddings (B, Tmax, D).
            ds (LongTensor): Batch of durations of each frame (B, Tmax).
            alpha (float, optional): Alpha value to control speed of speech.

        Returns:
            Tensor: replicated input tensor based on durations (B, Tmax, D).

        """
        # ys = gw.stgw.stgw(ds, xs, window_size=self.window_size, n_iter=self.n_iter)
        map = self.map(ds)
        ys = self.warp(xs, map.detach() if grad_stop else map)
        return ys, map
    
    def map(self, ds):
        if ds.ndim == 3:
            ds = torch.cat([ds[...,:1] + 0j, ds[...,1::2] + 1j*ds[...,2::2]], dim=-1)
            ds = gw.GW.spectrogram_to_signal(ds)
        return gw.GW.map(ds,sr=self.sr,pad=16)
    
    def warp(self, xs, map):
        return map@xs
