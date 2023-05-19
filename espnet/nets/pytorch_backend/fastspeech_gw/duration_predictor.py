#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Duration predictor related modules."""

import torch
from gw.utils import LPF

from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet2.tts.fastspeech2.variance_predictor import VariancePredictor
from espnet2.layers.stft import Stft


class DurationPredictor(torch.nn.Module):
    """Duration predictor module.

    This is a module of duration predictor described
    in `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    The duration predictor predicts a duration of each frame in log domain
    from the hidden embeddings of encoder.

    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf

    Note:
        The calculation domain of outputs is different
        between in `forward` and in `inference`. In `forward`,
        the outputs are calculated in log domain but in `inference`,
        those are calculated in linear domain.

    """

    def __init__(
        self, 
        idim,
        predictor_type,
        # for variance predictor
        vp_n_layers=None, 
        vp_n_chans=None, 
        vp_kernel_size=None, 
        vp_dropout_rate=None, 
        # for transformer encoder
        te_attention_dim=None,
        te_attention_heads=None,
        te_linear_units=None,
        te_num_blocks=None,
        te_input_layer=None,
        te_dropout_rate=None,
        te_positional_dropout_rate=None,
        te_attention_dropout_rate=None,
        te_pos_enc_class=None,
        te_normalize_before=None,
        te_concat_after=None,
        te_positionwise_layer_type=None,
        te_positionwise_conv_kernel_size=None,
        # post process
        use_lpf=False,
        lpf_window_size=64,
        scale=1e-1,
        cumsum=False
    ):
        """Initilize duration predictor module.

        Args:
            idim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
            offset (float, optional): Offset value to avoid nan in log domain.

        """
        super(DurationPredictor, self).__init__()
        self.predictor_type = predictor_type
        if predictor_type == 'variance_predictor':
            self.predictor = VariancePredictor(
                idim,
                n_layers=vp_n_layers,
                n_chans=vp_n_chans,
                kernel_size=vp_kernel_size,
                dropout_rate=vp_dropout_rate
            )
        elif predictor_type == 'transformer_encoder':
            self.predictor = Encoder(
                idim=0,
                attention_dim=te_attention_dim,
                attention_heads=te_attention_heads,
                linear_units=te_linear_units,
                num_blocks=te_num_blocks,
                input_layer=te_input_layer,
                dropout_rate=te_dropout_rate,
                positional_dropout_rate=te_positional_dropout_rate,
                attention_dropout_rate=te_attention_dropout_rate,
                pos_enc_class=te_pos_enc_class,
                normalize_before=te_normalize_before,
                concat_after=te_concat_after,
                positionwise_layer_type=te_positionwise_layer_type,
                positionwise_conv_kernel_size=te_positionwise_conv_kernel_size,
            )
            self.linear = torch.nn.Linear(te_attention_dim, 1)
        else:
            raise ValueError()
        self.use_lpf = use_lpf
        if use_lpf:
            self.lpf = LPF(lpf_window_size)
        self.scale = scale
        self.cumsum = cumsum

    def _forward(self, xs, masks):
        if self.predictor_type == 'variance_predictor':
            xs = self.predictor(xs, masks).squeeze(-1)
        elif self.predictor_type == 'transformer_encoder':
            xs, masks = self.predictor(xs, masks.transpose(-2,-1))
            xs = self.linear(xs).squeeze(-1)  # (B, Tmax)
        if self.use_lpf:
            xs = self.lpf(xs)
        xs = xs*self.scale
        if self.cumsum:
            xs = xs.cumsum(-1)

        return xs

    def forward(self, xs, x_masks=None):
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor, optional):
                Batch of masks indicating padded part (B, Tmax).

        Returns:
            Tensor: Batch of predicted durations in log domain (B, Tmax).

        """
        return self._forward(xs, x_masks)

    def inference(self, xs, x_masks=None):
        """Inference duration.

        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor, optional):
                Batch of masks indicating padded part (B, Tmax).

        Returns:
            LongTensor: Batch of predicted durations in linear domain (B, Tmax).

        """
        return self._forward(xs, x_masks)
    
class DurationPredictorLoss(torch.nn.Module):
    """Loss function module for duration predictor.

    The loss value is Calculated in log domain to make it Gaussian.

    """

    def __init__(
        self, 
        reduction="mean",
        prior='linear',
        lam=1e-6,
    ):
        """Initilize duration predictor loss module.

        Args:
            offset (float, optional): Offset value to avoid nan in log domain.
            reduction (str): Reduction type in loss calculation.

        """
        super(DurationPredictorLoss, self).__init__()
        self.reduction = reduction
        self.prior = prior
        self.lam = lam

    def forward(self, outputs, masks):
        """Calculate forward propagation.

        Args:
            outputs (Tensor): Batch of prediction durations in log domain (B, T)
            targets (LongTensor): Always None.

        Returns:
            Tensor: loss value.

        """
        fs = torch.fft.rfft(outputs,dim=-1).abs().square()
        c = torch.linspace(0,1,fs.size(-1),device=fs.device).square()
        return self.lam*torch.mean(fs@c)

