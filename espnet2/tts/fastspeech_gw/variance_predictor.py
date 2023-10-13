#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Duration predictor related modules."""

import torch
from gw.utils import LPF
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention, EdenAttention
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.decoder import Decoder


class VariancePredictor(torch.nn.Module):
    """Variance predictor module.

    This is a module of variacne predictor described in `FastSpeech 2:
    Fast and High-Quality End-to-End Text to Speech`_.

    .. _`FastSpeech 2: Fast and High-Quality End-to-End Text to Speech`:
        https://arxiv.org/abs/2006.04558

    """

    def __init__(
        self,
        idim: int,
        odim: int = 1,
        n_layers: int = 2,
        n_chans: int = 384,
        kernel_size: int = 3,
        bias: bool = True,
        dropout_rate: float = 0.5,
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
                    torch.nn.Conv1d(
                        in_chans,
                        n_chans,
                        kernel_size,
                        stride=1,
                        padding=(kernel_size - 1) // 2,
                        bias=bias,
                    ),
                    torch.nn.ReLU(),
                    LayerNorm(n_chans, dim=1),
                    torch.nn.Dropout(dropout_rate),
                )
            ]
        self.linear = torch.nn.Linear(n_chans, odim)

    def forward(self, xs: torch.Tensor, x_masks: torch.Tensor = None) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor): Batch of masks indicating padded part (B, Tmax).

        Returns:
            Tensor: Batch of predicted sequences (B, Tmax, 1).

        """
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)

        xs = self.linear(xs.transpose(1, 2))  # (B, Tmax, odim)

        if x_masks is not None:
            xs = xs.masked_fill(x_masks, 0.0)

        return xs


class AlignmentModule(torch.nn.Module):
    """Alignment module.
    """

    def __init__(
        self, 
        tdim, 
        fdim,
        odim=1, 
        n_layers=2, 
        n_chans=384, 
        kernel_size=3,
        dropout_rate=0.1, 
    ):
        """Initilize duration predictor module.

        Args:
            idim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.

        """
        assert check_argument_types()
        super(AlignmentModule, self).__init__()
        
        self.conv_feat = torch.nn.ModuleList()
        for idx in range(n_layers):
            in_chans = fdim if idx == 0 else n_chans
            self.conv_feat += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        in_chans,
                        n_chans,
                        kernel_size,
                        stride=1,
                        padding=(kernel_size - 1) // 2,
                    ),
                    torch.nn.ReLU(),
                    LayerNorm(n_chans, dim=1),
                    torch.nn.Dropout(dropout_rate),
                )
            ]
        self.linear_feat = torch.nn.Linear(n_chans, tdim)
        
        self.conv_out = torch.nn.ModuleList()
        for idx in range(n_layers):
            in_chans = tdim if idx == 0 else n_chans
            self.conv_out += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        in_chans,
                        n_chans,
                        kernel_size,
                        stride=1,
                        padding=(kernel_size - 1) // 2,
                    ),
                    torch.nn.ReLU(),
                    LayerNorm(n_chans, dim=1),
                    torch.nn.Dropout(dropout_rate),
                )
            ]
        self.linear_out = torch.nn.Linear(n_chans, odim)

    def forward(self, texts, feats, masks):
        """Calculate forward propagation.

        Args:
            texts (Tensor): Batch of input texts (B, Ttext).
            feats (Tensor): Batch of target feature (B, Tfeat, fdim).
            text_masks (ByteTensor, optional): Batch of masks indicating padded part (B, Ttext, 1).
            feat_masks (ByteTensor, optional): Batch of masks indicating padded part (B, Tfeat, 1).

        Returns:
            Tensor: Batch of predicted durations in log domain (B, Tfeat).

        """
        xs = feats.transpose(1, -1)  # (B, C, Tfeat)
        for f in self.conv_feat:
            xs = f(xs)  # (B, C, Tfeat)

        xs = texts + self.linear_feat(xs.transpose(1, -1))  # (B, Tfeat, tdim)
        
        xs = xs.transpose(1, -1)  # (B, C, Tfeat)
        for f in self.conv_out:
            xs = f(xs)  # (B, C, Tfeat)

        xs = self.linear_out(xs.transpose(1, -1))  # (B, Tfeat, odim)

        xs = xs.masked_fill(masks.unsqueeze(-1), 0.0)  # (B, Tfeat, odim)

        return xs


# class AlignmentModule(torch.nn.Module):
#     """Alignment module.
#     """

#     def __init__(
#         self, 
#         tgt_dim,
#         src_dim,
#         odim,
#         attention_dim=256,
#         attention_heads=4,
#         linear_units=2048,
#         num_blocks=2,
#         dropout_rate=0.0,
#         positional_dropout_rate=0.0,
#         attention_dropout_rate=0.0,
#         input_layer=None,
#         pos_enc_class=None,
#         normalize_before=True,
#         concat_after=False
#     ):
#         """Initilize duration predictor module.

#         Args:
#             idim (int): Input dimension.
#             n_layers (int, optional): Number of convolutional layers.
#             n_chans (int, optional): Number of channels of convolutional layers.
#             kernel_size (int, optional): Kernel size of convolutional layers.
#             dropout_rate (float, optional): Dropout rate.

#         """
#         assert check_argument_types()
#         super(AlignmentModule, self).__init__()
        
#         self.encoder = Encoder(
#             idim=src_dim,
#             attention_dim=attention_dim,
#             attention_heads=attention_heads,
#             linear_units=linear_units,
#             num_blocks=num_blocks,
#             dropout_rate=dropout_rate,
#             positional_dropout_rate=positional_dropout_rate,
#             attention_dropout_rate=attention_dropout_rate,
#             input_layer=input_layer,
#             pos_enc_class=pos_enc_class,
#             normalize_before=normalize_before,
#             concat_after=concat_after,
#         )
        
#         self.decoder = Decoder(
#             odim=tgt_dim,
#             attention_dim=attention_dim,
#             attention_heads=attention_heads,
#             linear_units=linear_units,
#             num_blocks=num_blocks,
#             dropout_rate=dropout_rate,
#             positional_dropout_rate=positional_dropout_rate,
#             self_attention_dropout_rate=attention_dropout_rate,
#             src_attention_dropout_rate=attention_dropout_rate,
#             input_layer=None,
#             pos_enc_class=pos_enc_class,
#             normalize_before=normalize_before,
#             concat_after=concat_after,
#         )
        
#         self.linear = torch.nn.Linear(tgt_dim, odim)
        
        
#     def forward(self, tgt, tgt_mask, src, src_mask):
#         """Calculate forward propagation.

#         Args:
#             texts (Tensor): Batch of input texts (B, Ttext).
#             feats (Tensor): Batch of target feature (B, Tfeat, fdim).
#             text_masks (ByteTensor, optional): Batch of masks indicating padded part (B, Ttext, 1).
#             feat_masks (ByteTensor, optional): Batch of masks indicating padded part (B, Tfeat, 1).

#         Returns:
#             Tensor: Batch of predicted durations in log domain (B, Tfeat).

#         """
#         memory, memory_mask = self.encoder.forward(src, src_mask)
#         output, _ = self.decoder.forward(tgt, tgt_mask, memory, memory_mask)
#         return self.linear(output)
