#!/usr/bin/env python3

# Copyright 2020 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Variance predictor related modules."""

import torch
from typeguard import check_argument_types
import logging

from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class Sampling(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, zs: torch.Tensor) -> torch.Tensor:
        if zs.size(-1) == 2:
            mu = zs.select(-1, 0)
            ln_var = zs.select(-1, 1)
        elif zs.size(-1)%2 == 0:
            mu, ln_var = zs.split(-1, zs.size(-1)//2)
        else:
            raise RuntimeError()
                
        zs = mu + ln_var.mul(0.5).exp()*torch.randn_like(mu)
            
        return dict(zs=zs, mu=mu, ln_var=ln_var)


class FFTwrapper(torch.nn.Module):
    def __init__(
            self,
            nets: torch.nn.Module,
            arg_ref = None,
            ret_ref = None,
            filter: str = 'linear'
        ) -> None:
        super().__init__()
        self.nets = nets
        self.arg_ref = arg_ref
        self.ret_ref = ret_ref
        self.filter = filter
        
    def rfft(self, inputs: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        b = lens.size(0)
        ys = [None]*b
        xs = inputs if inputs.ndim==3 else inputs.unsqueeze(-1)
        for i in range(lens.size(0)):
            n: int = lens[i].item()
            tmp = torch.fft.rfft(xs[i], n=n, dim=-2, norm='forward')
            if n%2 == 0:
                tmp = torch.cat([tmp.real, tmp[...,1:-1,:].imag], dim=-2)
            else:
                tmp = torch.cat([tmp.real, tmp[...,1:,:].imag], dim=-2)
            ys[i] = torch.nn.functional.pad(tmp, [0, 0, 0, xs.size(1)-n])
        ys = torch.stack(ys, dim=0)
        outputs = ys if inputs.ndim==3 else ys.squeeze(-1)
        return outputs
        
    def irfft(self, inputs: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        b = lens.size(0)
        xs = [None]*b
        ys = inputs if inputs.ndim==3 else inputs.unsqueeze(-1)
        for i in range(lens.size(0)):
            n: int = lens[i].item()
            if n%2 == 0:
                tmp = torch.complex(ys.select(0, i).narrow(-2, 0, n//2+1),
                    torch.nn.functional.pad(
                        ys.select(0, i).narrow(-2, n//2+1, n//2-1),
                        [0, 0, 1, 1]
                    )
                )
            else:
                tmp = torch.complex(
                    ys.select(0, i).narrow(-2, 0, n//2+1),
                    torch.nn.functional.pad(
                        ys.select(0, i).narrow(-2, n//2+1, n//2),
                        [0, 0, 1, 0]
                    )
                )
            if self.filter == 'linear':
                f = torch.linspace(
                    0, 16, n//2+1, device=inputs.device
                ).square().add(1.0).sqrt()
                tmp = tmp/f.unsqueeze(-1)
            tmp = torch.fft.irfft(tmp, n=n, dim=-2, norm='forward')
            xs[i] = torch.nn.functional.pad(tmp, [0, 0, 0, ys.size(1)-n])
        xs = torch.stack(xs, dim=0)
        outputs = xs if inputs.ndim==3 else xs.squeeze(-1)
        return outputs
    
    def forward(self, lens, **kwargs):
        kwargs[self.arg_ref] = self.rfft(kwargs[self.arg_ref], lens)
        rets = self.nets(**kwargs)
        if self.ret_ref is None:
            rets = self.irfft(rets, lens)
        else:
            rets[self.ret_ref] = self.irfft(rets[self.ret_ref], lens)
        return rets
 
    
class KLDivergenceLoss(torch.nn.Module):
    """Loss function module for variance predictor.
    """

    def __init__(self, reduction="mean", mu=0, log_var=0):
        """Initilize duration predictor loss module.

        Args:
            reduction (str): Reduction type in loss calculation.

        """
        super(KLDivergenceLoss, self).__init__()
        self.reduction = reduction
        self.register_buffer('mu', torch.tensor(mu))
        self.register_buffer('log_var', torch.tensor(log_var))
        
    def forward(self, mu_q, log_var_q, mu_p=None, log_var_p=None, masks=None):
        if mu_p is None:
            mu_p = self.mu
        if log_var_p is None:
            log_var_p = self.log_var
        kl_loss = 0.5 * (
            log_var_p - log_var_q 
            + torch.exp(log_var_q - log_var_p) 
            + torch.square(mu_q - mu_p)*torch.exp(-log_var_p) 
            - 1
        )
        # kl_loss = kl_loss.sum(-1)
        if masks is not None:
            kl_loss = kl_loss.masked_select(masks)
        if self.reduction == 'sum':
            kl_loss = kl_loss.sum()
        elif self.reduction == 'mean':
            kl_loss = kl_loss.mean()
        return kl_loss
