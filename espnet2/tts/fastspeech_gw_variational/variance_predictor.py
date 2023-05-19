#!/usr/bin/env python3

# Copyright 2020 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Variance predictor related modules."""

import torch
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class VariationalVariancePredictor(torch.nn.Module):
    """Variance predictor module.

    This is a module of variacne predictor described in `FastSpeech 2:
    Fast and High-Quality End-to-End Text to Speech`_.

    .. _`FastSpeech 2: Fast and High-Quality End-to-End Text to Speech`:
        https://arxiv.org/abs/2006.04558

    """

    def __init__(
        self,
        idim: int,
        tdim: int,
        odim: int,
        n_layers: int = 2,
        n_chans: int = 384,
        n_latent: int = 192,
        kernel_size: int = 3,
        bias: bool = True,
        dropout_rate: float = 0.5,
    ):
        """Initilize duration predictor module.

        Args:
            idim (int): Input dimension.
            tdim (int): Number of different tokens.
            n_layers (int): Number of convolutional layers.
            n_chans (int): Number of channels of convolutional layers.
            kernel_size (int): Kernel size of convolutional layers.
            dropout_rate (float): Dropout rate.

        """
        assert check_argument_types()
        super().__init__()
        
        self.encoder1 = torch.nn.ModuleList()
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.encoder1 += [
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
        
        self.decoder1 = torch.nn.ModuleList()
        for idx in range(n_layers):
            in_chans = n_latent if idx == 0 else n_chans
            # self.decoder1 += [
            #     torch.nn.Sequential(
            #         torch.nn.Conv1d(
            #             in_chans,
            #             n_chans,
            #             kernel_size,
            #             stride=1,
            #             padding=(kernel_size - 1) // 2,
            #             bias=bias,
            #         ),
            #         torch.nn.ReLU(),
            #         LayerNorm(n_chans, dim=1),
            #         torch.nn.Dropout(dropout_rate),
            #     )
            # ]
            self.decoder1 += [
                torch.nn.Sequential(
                    torch.nn.Linear(
                        in_chans,
                        n_chans,
                        bias=bias,
                    ),
                    torch.nn.ReLU(),
                    LayerNorm(n_chans, dim=2),
                    torch.nn.Dropout(dropout_rate),
                )
            ]
            
        self.encoder2 = torch.nn.ModuleList()
        for idx in range(n_layers):
            in_chans = n_latent if idx == 0 else n_chans
            self.encoder2 += [
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
            
        self.decoder2 = torch.nn.ModuleList()
        for idx in range(n_layers):
            in_chans = odim if idx == 0 else n_chans
            self.decoder2 += [
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
        
        self.encoder1_linear = torch.nn.Linear(n_chans, 2*n_latent)
        self.decoder1_linear = torch.nn.Linear(n_chans, tdim)
        self.encoder2_linear = torch.nn.Linear(n_chans, 2*odim)
        self.decoder2_linear = torch.nn.Linear(n_chans, 2*n_latent)

    def forward(self, xs: torch.Tensor, x_masks: torch.Tensor = None) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor): Batch of masks indicating padded part (B, Tmax).

        Returns:
            Tensor: Batch of predicted sequences (B, Tmax, 1).

        """
        
        hs = xs  # (B, Tmax, idim)
        hs = hs.transpose(1, 2)  # (B, idim, Tmax)
        for f in self.encoder1:
            hs = f(hs)  # (B, C, Tmax)
        hs = self.encoder1_linear(hs.transpose(1, 2))  # (B, Tmax, Z)
        
        mu_z = hs[:,:,0::2]
        ln_var_z = hs[:,:,1::2]
        zs = mu_z + torch.randn_like(mu_z)*torch.exp(0.5*ln_var_z)
        
        if x_masks is not None:
            zs = zs.masked_fill(x_masks, 0.0)
        
        hs = zs
        # hs = hs.transpose(1, 2)  # (B, Z, Tmax)
        for f in self.decoder1:
            hs = f(hs)  # (B, C, Tmax)
        # ys = self.decoder1_linear(hs.transpose(1, 2))  # (B, Tmax, idim)
        ys = self.decoder1_linear(hs)  # (B, Tmax, idim)

        if x_masks is not None:
            ys = ys.masked_fill(x_masks, 0.0)
            
        hs = zs
        hs = hs.transpose(1, 2)  # (B, Z, Tmax)
        for f in self.encoder2:
            hs = f(hs)  # (B, C, Tmax)
        hs = self.encoder2_linear(hs.transpose(1, 2))  # (B, Tmax, odim)
        
        mu_v = hs[:,:,0::2]
        ln_var_v = hs[:,:,1::2]
        vs = mu_v + torch.randn_like(mu_v)*torch.exp(0.5*ln_var_v)

        if x_masks is not None:
            vs = vs.masked_fill(x_masks, 0.0)
            
        hs = vs
        hs = hs.transpose(1, 2)  # (B, odim, Tmax)
        for f in self.decoder2:
            hs = f(hs)  # (B, C, Tmax)
        hs = self.decoder2_linear(hs.transpose(1, 2))  # (B, Tmax, Z)
        
        mu_w = hs[:,:,0::2]
        ln_var_w = hs[:,:,1::2]
    
        return dict(
            output=vs, 
            reconstruct=ys, 
            mu_z=mu_z, 
            ln_var_z=ln_var_z,
            mu_v=mu_v,
            ln_var_v=ln_var_v,
            mu_w=mu_w,
            ln_var_w=ln_var_w
        )
    
class VAELoss(torch.nn.Module):
    """Loss function module for variance predictor.
    """

    def __init__(self, reduction="mean", mu=0, log_var=0):
        """Initilize duration predictor loss module.

        Args:
            reduction (str): Reduction type in loss calculation.

        """
        super(VAELoss, self).__init__()
        self.recons = torch.nn.CrossEntropyLoss(reduction='none')
        self.reduction = reduction
        self.register_buffer('mu', torch.tensor(mu))
        self.register_buffer('log_var', torch.tensor(log_var))
        
    def _reconstruction_loss(self, pred, target, masks):
        recons = self.recons(pred.transpose(-2,-1), target)
        if masks is not None:
            recons = recons.masked_select(masks)
        if self.reduction == 'sum':
            recons = recons.sum()
        elif self.reduction == 'mean':
            recons = recons.mean()
        return recons
    
    def _kl_divergence_loss(self, mu_q, log_var_q, mu_p, log_var_p, masks=None):
        kl_loss = 0.5 * (log_var_p - log_var_q + torch.exp(log_var_q - log_var_p) + torch.square(mu_q - mu_p)*torch.exp(-log_var_p) - 1)
    
        if masks is not None:
            kl_loss = kl_loss.masked_select(masks.unsqueeze(-1))
        kl_loss = kl_loss.sum(-1)
        if self.reduction == 'sum':
            kl_loss = kl_loss.sum()
        elif self.reduction == 'mean':
            kl_loss = kl_loss.mean()
        return kl_loss
        

    def forward(self, input, reconstruct, mu_z, ln_var_z, mu_v, ln_var_v, mu_w, ln_var_w, masks, **kwargs):
        """Calculate forward propagation.

        Args:
            outputs (Tensor): Batch of prediction durations in log domain (B, T)
            targets (LongTensor): Always None.

        Returns:
            Tensor: loss value.

        """
        rc_loss = self._reconstruction_loss(reconstruct, input, masks=masks)
        kl_loss_z = self._kl_divergence_loss(mu_z, ln_var_z, mu_w, ln_var_w, masks=masks)
        kl_loss_v = self._kl_divergence_loss(mu_v, ln_var_v, self.mu, self.log_var,masks=masks)
        return rc_loss, kl_loss_z, kl_loss_v


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
        tdim=None,
        odim=None,
        n_layers: int = 2,
        n_chans: int = 384,
        n_latent: int = None,
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
        self.linear = torch.nn.Linear(n_chans, 1)

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

        xs = self.linear(xs.transpose(1, 2))  # (B, Tmax, 1)

        if x_masks is not None:
            xs = xs.masked_fill(x_masks, 0.0)

        return dict(output=xs)
