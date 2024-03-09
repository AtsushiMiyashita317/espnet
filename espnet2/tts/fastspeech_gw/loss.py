# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Fastspeech2 related loss module for ESPnet2."""

from typing import Tuple

import torch
from typeguard import check_argument_types
import gw

from espnet.nets.pytorch_backend.fastspeech_gw.duration_predictor import (  # noqa: H301
    DurationPredictorLoss,
)
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask, make_pad_mask
from espnet2.tts.fastspeech_gw.variational import KLDivergenceLoss
from espnet.nets.pytorch_backend.fastspeech.length_regulator import LengthRegulator
from espnet2.tts.fastspeech_gw.length_regulator import LengthRegulator as GW


class DurationPredictorgwLoss(torch.nn.Module):
    def __init__(self, lr_before=True) -> None:
        super().__init__()
        self.lr = LengthRegulator()
        self.gw = GW(sr=1)
        self.l1 = torch.nn.L1Loss(reduction='none')
        self.lr_before = lr_before
        
    def forward(self, d_outs, ds, ilens, olens):
        xs = torch.arange(ds.size(-1), device=ds.device, dtype=torch.float).unsqueeze(0).expand(ds.size()).unsqueeze(-1)
        ys = self.lr.forward(xs, ds)
        xs = gw.utils.interpolate(xs, ilens, olens, mode='nearest')
        y_outs = gw.cubic_interpolation(xs.transpose(-1,-2), d_outs).transpose(-1,-2)
        loss = self.l1.forward(y_outs, ys).squeeze(-1)
        masks = make_non_pad_mask(olens).to(ds.device)
        return loss.masked_select(masks).mean()  


class FastSpeechGWLoss(torch.nn.Module):
    """Loss function module for FastSpeech2."""

    def __init__(
        self, 
        use_masking: bool = True, 
        use_weighted_masking: bool = False,
        lr_mode: str = 'after',
        duration_predictor_prior: str = 'linear',
        duration_predictor_lam: float = 1.0,
    ):
        """Initialize feed-forward Transformer loss module.

        Args:
            use_masking (bool): Whether to apply masking for padded part in loss
                calculation.
            use_weighted_masking (bool): Whether to weighted masking in loss
                calculation.

        """
        assert check_argument_types()
        super().__init__()

        assert (use_masking != use_weighted_masking) or not use_masking
        self.use_masking = use_masking
        self.use_weighted_masking = use_weighted_masking
        self.lr_mode = lr_mode

        # define criterions
        reduction = "none" if self.use_weighted_masking else "mean"
        self.l1_criterion = torch.nn.L1Loss(reduction=reduction)
        self.mse_criterion = torch.nn.MSELoss(reduction=reduction)
        self.duration_criterion = DurationPredictorLoss(
            reduction=reduction,
            prior=duration_predictor_prior,
            lam=duration_predictor_lam,
        )

    def forward(
        self,
        after_outs: torch.Tensor,
        before_outs: torch.Tensor,
        d_outs: torch.Tensor,
        p_outs: torch.Tensor,
        e_outs: torch.Tensor,
        ys: torch.Tensor,
        ds: torch.Tensor,
        ps: torch.Tensor,
        es: torch.Tensor,
        ilens: torch.Tensor,
        olens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate forward propagation.

        Args:
            after_outs (Tensor): Batch of outputs after postnets (B, T_feats, odim).
            before_outs (Tensor): Batch of outputs before postnets (B, T_feats, odim).
            d_outs (LongTensor): Batch of outputs of duration predictor (B, T_text).
            p_outs (Tensor): Batch of outputs of pitch predictor (B, T_text, 1).
            e_outs (Tensor): Batch of outputs of energy predictor (B, T_text, 1).
            ys (Tensor): Batch of target features (B, T_feats, odim).
            ds (LongTensor): Always None.
            ps (Tensor): Batch of target token-averaged pitch (B, T_text, 1).
            es (Tensor): Batch of target token-averaged energy (B, T_text, 1).
            ilens (LongTensor): Batch of the lengths of each input (B,).
            olens (LongTensor): Batch of the lengths of each target (B,).

        Returns:
            Tensor: L1 loss value.
            Tensor: Duration predictor loss value.
            Tensor: Pitch predictor loss value.
            Tensor: Energy predictor loss value.

        """
        # apply mask to remove padded part
        if self.use_masking:
            out_masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            before_outs = before_outs.masked_select(out_masks)
            if after_outs is not None:
                after_outs = after_outs.masked_select(out_masks)
            ys = ys.masked_select(out_masks)
            if self.lr_mode == 'after':
                duration_masks = make_non_pad_mask(olens).to(ys.device)
            elif self.lr_mode == 'before':
                duration_masks = make_non_pad_mask(ilens).to(ys.device)
            d_outs = d_outs.masked_select(duration_masks)
            # ds = ds.masked_select(duration_masks)
            pitch_masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            p_outs = p_outs.masked_select(pitch_masks)
            e_outs = e_outs.masked_select(pitch_masks)
            ps = ps.masked_select(pitch_masks)
            es = es.masked_select(pitch_masks)

        # calculate loss
        l1_loss = self.l1_criterion(before_outs, ys)
        if after_outs is not None:
            l1_loss += self.l1_criterion(after_outs, ys)
        duration_loss = self.duration_criterion(d_outs, duration_masks)
        pitch_loss = self.mse_criterion(p_outs, ps)
        energy_loss = self.mse_criterion(e_outs, es)

        # make weighted mask and apply it
        if self.use_weighted_masking:
            out_masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            out_weights = out_masks.float() / out_masks.sum(dim=1, keepdim=True).float()
            out_weights /= ys.size(0) * ys.size(2)
            duration_masks = make_non_pad_mask(ilens).to(ys.device)
            duration_weights = (
                duration_masks.float() / duration_masks.sum(dim=1, keepdim=True).float()
            )
            duration_weights /= ys.size(0)

            # apply weight
            l1_loss = l1_loss.mul(out_weights).masked_select(out_masks).sum()
            duration_loss = (
                duration_loss.mul(duration_weights).masked_select(duration_masks).sum()
            )
            pitch_masks = duration_masks.unsqueeze(-1)
            pitch_weights = duration_weights.unsqueeze(-1)
            pitch_loss = pitch_loss.mul(pitch_weights).masked_select(pitch_masks).sum()
            energy_loss = (
                energy_loss.mul(pitch_weights).masked_select(pitch_masks).sum()
            )

        return l1_loss, duration_loss, pitch_loss, energy_loss

class VariationalFastSpeechGWLoss(torch.nn.Module):
    """Loss function module for FastSpeech2."""

    def __init__(
        self, 
        use_masking: bool = True, 
        use_weighted_masking: bool = False,
        lr_mode: str = 'after',
        lr_n_fft : int = 80
    ):
        """Initialize feed-forward Transformer loss module.

        Args:
            use_masking (bool): Whether to apply masking for padded part in loss
                calculation.
            use_weighted_masking (bool): Whether to weighted masking in loss
                calculation.

        """
        assert check_argument_types()
        super().__init__()

        assert use_masking
        assert not use_weighted_masking
        self.use_masking = use_masking
        self.use_weighted_masking = use_weighted_masking
        self.lr_mode = lr_mode
        self.hop_length = lr_n_fft//4
        self.processed_mbins = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.grad_rate_begin = 120.0
        self.grad_rate_end = 240.0

        # define criterions
        reduction = "none" if self.use_weighted_masking else "mean"
        self.l1_criterion = torch.nn.L1Loss(reduction=reduction)
        self.mse_criterion = torch.nn.MSELoss(reduction=reduction)
        self.duration_criterion = KLDivergenceLoss(reduction=reduction)

    def forward(
        self,
        after_outs: torch.Tensor,
        before_outs: torch.Tensor,
        d_outs: torch.Tensor,
        p_outs: torch.Tensor,
        e_outs: torch.Tensor,
        ys: torch.Tensor,
        ds: torch.Tensor,
        ps: torch.Tensor,
        es: torch.Tensor,
        ilens: torch.Tensor,
        olens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate forward propagation.

        Args:
            after_outs (Tensor): Batch of outputs after postnets (B, T_feats, odim).
            before_outs (Tensor): Batch of outputs before postnets (B, T_feats, odim).
            d_outs (LongTensor): Batch of outputs of duration predictor (B, T_text).
            p_outs (Tensor): Batch of outputs of pitch predictor (B, T_text, 1).
            e_outs (Tensor): Batch of outputs of energy predictor (B, T_text, 1).
            ys (Tensor): Batch of target features (B, T_feats, odim).
            ds (LongTensor): Always None.
            ps (Tensor): Batch of target token-averaged pitch (B, T_text, 1).
            es (Tensor): Batch of target token-averaged energy (B, T_text, 1).
            ilens (LongTensor): Batch of the lengths of each input (B,).
            olens (LongTensor): Batch of the lengths of each target (B,).

        Returns:
            Tensor: L1 loss value.
            Tensor: Duration predictor loss value.
            Tensor: Pitch predictor loss value.
            Tensor: Energy predictor loss value.

        """
        # apply mask to remove padded part
        out_masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
        before_outs = before_outs.masked_select(out_masks)
        if after_outs is not None:
            after_outs = after_outs.masked_select(out_masks)
        ys = ys.masked_select(out_masks)
        pitch_masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
        p_outs = p_outs.masked_select(pitch_masks)
        e_outs = e_outs.masked_select(pitch_masks)
        ps = ps.masked_select(pitch_masks)
        es = es.masked_select(pitch_masks)

        # calculate loss
        l1_loss = self.l1_criterion(before_outs, ys)
        if after_outs is not None:
            l1_loss += self.l1_criterion(after_outs, ys)
        t = self.grad_rate()
        duration_loss = self.duration_criterion(ds*t + ds.detach()*(1-t), d_outs, pitch_masks)
        self.count_iteration(olens.sum().item())
        pitch_loss = self.mse_criterion(p_outs, ps)
        energy_loss = self.mse_criterion(e_outs, es)

        return l1_loss, duration_loss, pitch_loss, energy_loss
    
    def count_iteration(self, bins):
        if self.training:
            self.processed_mbins += bins/1000000
    
    def grad_rate(self):
        b = self.processed_mbins.item()
        return max(0, min(1, (b-self.grad_rate_begin)/(self.grad_rate_end-self.grad_rate_begin)))