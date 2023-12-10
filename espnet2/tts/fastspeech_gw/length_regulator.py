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
    
        
class LambdaGW(torch.nn.Module):
    def __init__(
        self, 
        idim, 
        ldim, 
        n_layers, 
        n_composite,
        kernel_size,
        n_chans,
        
    ) -> None:
        super().__init__()
        self.idim = idim
        self.ldim = ldim
        self.n_layers = n_layers
        self.n_composite = n_composite
        
        conv = []
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            conv += [
                torch.nn.Conv1d(
                    in_chans,
                    n_chans,
                    kernel_size,
                    stride=1,
                    padding=(kernel_size - 1) // 2,
                ),
                torch.nn.ReLU(),
                LayerNorm(n_chans, dim=1),
            ]
        self.conv = torch.nn.Sequential(*conv)
        linear = []
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            linear += [
                torch.nn.Linear(in_chans,n_chans),
                torch.nn.ReLU(),
                LayerNorm(n_chans, dim=-1),
            ]
        linear += [torch.nn.Linear(n_chans, ldim*n_layers*n_composite)]
        self.linear_lf = torch.nn.Sequential(*linear)
        self.linear_kf = torch.nn.Linear(n_chans, ldim)
        self.linear_vf = torch.nn.Linear(n_chans, idim)
        
        # self.linear_kw = torch.nn.Linear(idim, ldim)
        # self.linear_vw = torch.nn.Linear(idim, ldim)
        # self.linear_lw = torch.nn.Linear(ldim, 1)
        # self.linear_ly = torch.nn.Linear(ldim, idim)
        
        self.norm_s = LayerNorm(ldim)
        self.emb_s = torch.nn.Linear(idim, ldim)
        self.linear_s = torch.nn.ModuleList()
        for idx in range(n_layers):
            odim = 1 if idx == n_layers-1 else ldim
            self.linear_s += [
                torch.nn.Sequential(
                    torch.nn.Linear(ldim, n_chans),
                    torch.nn.ReLU(),
                    LayerNorm(n_chans),
                    torch.nn.Linear(n_chans, odim),
                )
            ]
    
    def positional_encoding(self, t:torch.Tensor, d:int) -> torch.Tensor:
        """Positional Encoding

        Args:
            t (torch.Tensor, (b,*)): Position
            d (int): dimension

        Returns:
            torch.Tensor, (b,*,ldim): Positional encoding
        """
        div_term = torch.exp(
            torch.arange(0, d, 2, dtype=torch.float32, device=t.device)
            * -(math.log(2500.0) / d)
        )
        pe1 = torch.sin(t.unsqueeze(-1)*div_term*4)
        pe2 = torch.cos(t.unsqueeze(-1)*div_term*4)
        pe = torch.cat([pe1,pe2], dim=-1)/math.sqrt(d/2)
        return pe
        
    def forward_one_lambda(self, l:torch.Tensor, x:torch.Tensor) -> torch.Tensor:
        """Forward one lambda

        Args:
            l (torch.Tensor, (b, ldim+1, *)): Lamda
            x (torch.Tensor, (b, time2, ldim)): Input

        Returns:
            torch.Tensor, (b, time2, *): Output
        """
        b = self.norm_b.forward(l[:,:1])*math.sqrt(2/self.ldim)
        w = self.norm_w.forward(l[:,1:])*math.sqrt(2/self.ldim)
        return b+x@w
        
    def forward_gw_signal(self, l:torch.Tensor, t:torch.Tensor) -> torch.Tensor:
        """Forward GW signal

        Args:
            l (torch.Tensor, (b, ldim, ldim, n_layers)): Lamda
            t (torch.Tensor, (b, time2)): Time index
            
        Returns:
            torch.Tensor, (b, time2): GW signal
        """
        # (b, time2, idim)
        s = self.positional_encoding(t, self.idim)
        # (b, time2, ldim)
        s = self.emb_s.forward(s)
        
        for i, m in enumerate(self.linear_s):
            # (b, time2, ldim)
            res = s
            s = self.norm_s.forward(s)
            s = res + s@l[:,:,:,i]
            s = m.forward(s)
        
        # (b, time2)
        s = s.squeeze(-1).div(self.n_composite)
        return s
    
    def forward_one_composite(self, l:torch.Tensor, f:torch.Tensor) -> torch.Tensor:
        """Forward GW composite

        Args:
            l (torch.Tensor, (b, ldim+1, ldim, n_layers)): Lamda
            f (torch.Tensor, (b, time2)): Initial warping

        Returns:
            torch.Tensor, (b, time2): New warping function
        """
        k1 = self.forward_gw_signal(l, f)
        k2 = self.forward_gw_signal(l, f+k1/2)
        k3 = self.forward_gw_signal(l, f+k2/2)
        k4 = self.forward_gw_signal(l, f+k3)
        f = f + (k1+2*k2+2*k3+k4)/6
        return f
            
    def forward_function(self, x:torch.Tensor, ilens:torch.Tensor, olens:torch.Tensor) -> torch.Tensor:
        """Forward function

        Args:
            x (torch.Tensor, (b,time1,idim)): Input
            ilens (torch.Tensor, (b,)): Input length
            olens (torch.Tensor, (b,)): Output length

        Returns:
            torch.Tensor, (b,time2): Warping function
        """
        # (time1,)
        t = torch.arange(x.size(1), device=x.device)
        # (time1, idim)
        pe = self.positional_encoding(t, self.idim)
        x = x + pe
        x = self.conv.forward(x.transpose(-1,-2)).transpose(-1,-2)
        
        # (batch, time1, ldim)
        k = self.linear_kf.forward(x)
        # (batch, time1, idim)
        v = self.linear_vf.forward(x)
        
        # (b, t, 1)
        mask = make_pad_mask(ilens).to(device=x.device).unsqueeze(-1)
        min_value = torch.finfo(k.dtype).min
        k = k.masked_fill(mask, min_value)
        v = v.masked_fill(mask, 0.0)
        
        scores = torch.softmax(k, dim=-2)
        
        # (batch, ldim, idim)
        l = scores.transpose(-1,-2)@v
        # (batch, ldim, ldim*n_layers*n_composite)
        l = self.linear_lf.forward(l)
        # (batch, ldim, ldim, n_layers, n_composite)
        l = l.unflatten(-1, (self.ldim, self.n_layers, self.n_composite))
        
        max_len = olens.max().item()
        # (b,time2)
        f = torch.arange(max_len, dtype=torch.float32, device=x.device)*(ilens/olens).unsqueeze(-1)
        for i in range(self.n_composite):
            f = self.forward_one_composite(l[:,:,:,:,i], f)
            
        return f
        
    def forward_warp(self, x:torch.Tensor, f:torch.Tensor, ilens:torch.Tensor, olens:torch.Tensor) -> torch.Tensor:
        """Forward warp

        Args:
            x (torch.Tensor, (b,time1,idim)): Input
            f (torch.Tensor, (b, time2)): Warping function
            ilens (torch.Tensor, (b,)): Input length
            olens (torch.Tensor, (b,)): Output length
            
        Returns:
            torch.Tensor, (b,time2,idim): Output
        """
        # (batch, time1, ldim)
        k = self.linear_kw.forward(x)
        # (batch, time1, idim)
        v = self.linear_vw.forward(x)
        # (b, t, 1)
        mask = make_pad_mask(ilens).to(device=x.device).unsqueeze(-1)
        min_value = torch.finfo(k.dtype).min
        k = k.masked_fill(mask, min_value)
        v = v.masked_fill(mask, 0.0)
        
        # (batch, time2, time1)
        t = torch.arange(x.size(1), device=x.device) - f.unsqueeze(-1)
        # (batch, time2, time1, ldim)
        pe = self.positional_encoding(t, self.ldim)
        
        # (batch, time1, ldim)
        scores = torch.softmax(k, dim=-2)
        
        # (batch, idim, ldim)
        lc = v.transpose(-1,-2)@scores
        # (batch, time2, idim, ldim)
        lp = v.transpose(-1,-2).unsqueeze(1)@pe
        l = lp + lc.unsqueeze(1)
        # (batch, time2, ldim)
        y = self.linear_lw.forward(l).squeeze(-1)
        y = self.act.forward(y)
        y = self.linear_ly.forward(y)
        mask = make_pad_mask(olens).to(device=x.device).unsqueeze(-1)
        y = y.masked_fill(mask, 0.0)
        
        return y
    
    def forward(self, x:torch.Tensor, ilens:torch.Tensor, olens:torch.Tensor) -> torch.Tensor:
        """GW

        Args:
            x (torch.Tensor, (b, time1, idim)): Input
            ilens (torch.Tensor, (b,)): Input length
            olens (torch.Tensor, (b,)): Output length

        Returns:
            torch.Tensor, (b, time2 idim): Output
        """
        f = self.forward_function(x, ilens, olens)
        # y = self.forward_warp(x, f, ilens, olens)
        y = gw.cubic_interpolation(x.transpose(-1,-2), f).transpose(-1,-2)
        
        return y, f*(olens/ilens).unsqueeze(-1)
