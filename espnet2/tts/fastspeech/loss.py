# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Fastspeech2 related loss module for ESPnet2."""

from typing import Tuple

import torch
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.fastspeech.duration_predictor import (  # noqa: H301
    DurationPredictorLoss,
)
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask


class VAELoss(torch.nn.Module):
    """Loss function module for feed-forward Transformer."""

    def __init__(self, use_masking=True, use_weighted_masking=False):
        """Initialize feed-forward Transformer loss module.

        Args:
            use_masking (bool):
                Whether to apply masking for padded part in loss calculation.
            use_weighted_masking (bool):
                Whether to weighted masking in loss calculation.

        """
        super(VAELoss, self).__init__()
        assert (use_masking != use_weighted_masking) or not use_masking
        self.use_masking = use_masking
        self.use_weighted_masking = use_weighted_masking

        # define criterions
        self.duration_criterion = DurationPredictorLoss(reduction='none')
        
    def kl_divergence(self, mu_q, log_var_q, mu_p, log_var_p):
        kl_loss = 0.5 * (
            log_var_p - log_var_q 
            + torch.exp(log_var_q - log_var_p) 
            + torch.square(mu_q - mu_p)*torch.exp(-log_var_p) 
            - 1
        )
        kl_loss = kl_loss.sum(-1)
        return kl_loss
    
    def nll_laplace(self, mu, ln_var, target):
        nll_loss = torch.abs(mu-target).mul(ln_var.mul(-0.5).exp()) + ln_var.mul(0.5)
        nll_loss = nll_loss.sum(-1)
        return nll_loss

    def forward(
            self, 
            mu, 
            ln_var, 
            d_outs, 
            ys, 
            ds, 
            ilens, 
            olens,
            stats1,
            stats2,
        ):
        """Calculate forward propagation.

        Args:
            after_outs (Tensor): Batch of outputs after postnets (B, Lmax, odim).
            before_outs (Tensor): Batch of outputs before postnets (B, Lmax, odim).
            d_outs (Tensor): Batch of outputs of duration predictor (B, Tmax).
            ys (Tensor): Batch of target features (B, Lmax, odim).
            ds (Tensor): Batch of durations (B, Tmax).
            ilens (LongTensor): Batch of the lengths of each input (B,).
            olens (LongTensor): Batch of the lengths of each target (B,).

        Returns:
            Tensor: L1 loss value.
            Tensor: Duration predictor loss value.

        """
        
        # calculate loss
        n = ys.size(-1)
        l1_loss = self.nll_laplace(mu, ln_var, ys)/n
        duration_loss = self.duration_criterion(d_outs, ds)
        kl_losses1 = [self.kl_divergence(*s)/n for s in stats1]
        kl_losses2 = [self.kl_divergence(*s)/n for s in stats2]

        # apply mask to remove padded part
        if self.use_masking:
            in_masks = make_non_pad_mask(ilens).to(ys.device)
            duration_loss = duration_loss.masked_select(in_masks).mean()
            for i in range(len(kl_losses1)):
                kl_losses1[i] = kl_losses1[i].masked_select(in_masks).mean()
            out_masks = make_non_pad_mask(olens).to(ys.device)
            l1_loss = l1_loss.masked_select(out_masks).mean()
            for i in range(len(kl_losses2)):
                kl_losses2[i] = kl_losses2[i].masked_select(out_masks).mean()
        
        # make weighted mask and apply it
        if self.use_weighted_masking:
            out_masks = make_non_pad_mask(olens).to(ys.device)
            out_weights = out_masks.float() / out_masks.sum(dim=1, keepdim=True).float()
            out_weights /= ys.size(0)
            in_masks = make_non_pad_mask(ilens).to(ys.device)
            in_weights = in_masks.float() / in_masks.sum(dim=1, keepdim=True).float()
            in_weights /= ds.size(0)

            # apply weight
            l1_loss = l1_loss.mul(out_weights).masked_select(out_masks).sum()
            for i in range(len(kl_losses2)):
                kl_losses2 = kl_losses2.mul(out_weights).masked_select(out_masks).sum()
            duration_loss = (
                duration_loss.mul(in_weights).masked_select(in_masks).sum()
            )
            for i in range(len(kl_losses2)):
                kl_losses1 = (
                    kl_losses1.mul(in_weights).masked_select(in_masks).sum()
                )
            
        return l1_loss, duration_loss, kl_losses1+kl_losses2

