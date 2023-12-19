#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Length regulator related modules."""

import math

import torch
import gw


class ContinuousConv1d(torch.nn.Module):
    def __init__(self, kernel_size, n_fft, in_channel, out_channel):
        self.kernel_size = kernel_size
        self.n_fft = n_fft
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel = torch.nn.Parameter(torch.zeros(out_channel, in_channel, n_fft))
        self.bias = torch.nn.Parameter(torch.zeros(out_channel))
        self.reset_parameters()
    
    def forward(self, x:torch.Tensor, index:torch.Tensor) -> torch.Tensor:
        """continuous 1d convolution

        Args:
            x (torch.Tensor, (b, t1, f1)): Input signal
            index (torch.Tensor, (b, t2)): continuous index
            
        Returns:
            torch.Tensor, (b, t2, f2): Output signal
        """
        # (b, t2)
        i = index.floor().long()
        f = index - i
        # (n//2+1,)
        shift = torch.arange(self.n_fft//2+1, device=x.device)
        # (b, t2, n//2+1)
        shift = shift.mul(2j*torch.pi*f.unsqueeze(-1)/self.n_fft).exp()
        # (f2, f1, n//2+1,)
        kernel = torch.fft.rfft(self.kernel)
        # (b, t2, f2, f1, n//2+1)
        kernel = kernel*shift.unsqueeze(-2).unsqueeze(-2)
        # (b, t2, f2, f1, k)
        kernel = torch.fft.irfft(kernel, n=self.kernel_size)
        # (b, t2, f2, f1*k)
        kernel = kernel.flatten(-2)
        
        size = [x.size(0), x.size(1), x.size(2), self.kernel_size]
        x = torch.nn.functional.pad(x, [0, 0, self.kernel_size//2, self.kernel_size//2])
        stride = [x.stride(0), x.stride(1), x.stride(2), x.stride(1)]
        # (b, t1, f1, k)
        x = torch.as_strided(x, size=size, stride=stride)
        # (b, 1)
        b = torch.arange(i.size(0), device=i.device, dtype=i.dtype).unsqueeze(-1)
        # (b, t2, f1*k, 1)
        x = x[b, i].flatten(-2).unsqueeze(-1)
        
        # (b, t2, f2, 1)
        x = kernel@x
        # (b, t2, f2)
        return x.squeeze(-1) + self.bias
    
    def reset_parameters(self):
        k = 1/math.sqrt(self.kernel_size*self.in_channel)
        torch.nn.init.uniform_(self.kernel, -k, k)
        torch.nn.init.uniform_(self.bias, -k, k)



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
        
    def forward(self, xs, ds, *args):
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
        f = gw.gw_ode(ds[:,:,0])
        for i in range(1, ds.size(-1)):
            f = gw.gw_ode(ds[:,:,i], f=f)
        ys = gw.cubic_interpolation(xs.transpose(-1,-2), f).transpose(-1,-2)
        return ys, f
    
