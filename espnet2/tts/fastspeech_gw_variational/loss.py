# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Fastspeech2 related loss module for ESPnet2."""

from typing import Dict

import torch
from typeguard import check_argument_types

from espnet2.tts.fastspeech_gw_variational.variance_predictor import VAELoss
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask


class FastSpeechGWLoss(torch.nn.Module):
    """Loss function module for FastSpeech2."""

    def __init__(
        self, 
        lr_mode: str = 'after',
        duration_predictor_variational: bool = True,
        pitch_predictor_variational: bool = False,
        energy_predictor_variational: bool = False,
        l1_lambda: float = 1.0,
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

        self.lr_mode = lr_mode
        self.duration_predictor_variational = duration_predictor_variational
        self.pitch_predictor_variational = pitch_predictor_variational
        self.energy_predictor_variational = energy_predictor_variational
        self.l1_lambda = l1_lambda
        
        # define criterions
        reduction = "mean"
        self.l1_criterion = torch.nn.L1Loss(reduction=reduction)
        self.mse_criterion = torch.nn.MSELoss(reduction=reduction)
        self.vae_criterion = VAELoss(reduction=reduction)

    def forward(
        self,
        after_outs: torch.Tensor,
        before_outs: torch.Tensor,
        d_outs: Dict[str, torch.Tensor],
        p_outs: Dict[str, torch.Tensor],
        e_outs: Dict[str, torch.Tensor],
        xs: torch.Tensor,
        ys: torch.Tensor,
        ps: torch.Tensor,
        es: torch.Tensor,
        ilens: torch.Tensor,
        olens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Calculate forward propagation.

        Args:
            after_outs (Tensor): Batch of outputs after postnets (B, T_feats, odim).
            before_outs (Tensor): Batch of outputs before postnets (B, T_feats, odim).
            d_outs (LongTensor): Batch of outputs of duration predictor (B, T_text).
            p_outs (Tensor): Batch of outputs of pitch predictor (B, T_text, 1).
            e_outs (Tensor): Batch of outputs of energy predictor (B, T_text, 1).
            ys (Tensor): Batch of target features (B, T_feats, odim).
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
        odim = before_outs.size(-1) if after_outs is None else after_outs.size(-1)
        # apply mask to remove padded part        
        out_masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
        before_outs = before_outs.masked_select(out_masks)
        if after_outs is not None:
            after_outs = after_outs.masked_select(out_masks)
        ys = ys.masked_select(out_masks)
        if self.lr_mode == 'after':
            duration_masks = make_non_pad_mask(olens).to(ys.device)
        elif self.lr_mode == 'before':
            duration_masks = make_non_pad_mask(ilens).to(ys.device)
        pitch_masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)

        # calculate loss
        l1_loss = self.l1_criterion(before_outs, ys)
        if after_outs is not None:
            l1_loss += self.l1_criterion(after_outs, ys)
        if self.duration_predictor_variational:
            rc_loss, kl_loss_z, kl_loss_v = self.vae_criterion(xs, **d_outs, masks=duration_masks)
            duration_loss = dict(
                duration_reconstruction=rc_loss/odim/self.l1_lambda,
                duration_kl_divergence=kl_loss_v/odim/self.l1_lambda,
                duration_z_kl_divergence=kl_loss_z/odim/self.l1_lambda,
            )
        else:
            duration_loss = dict()
        if self.pitch_predictor_variational:
            mse_loss = self.mse_criterion(p_outs['output'], ps)
            rc_loss, kl_loss_z, kl_loss_v  = self.vae_criterion(xs, **p_outs, masks=pitch_masks)
            pitch_loss = dict(
                pitch_loss=mse_loss,
                pitch_reconstruction=rc_loss/odim/self.l1_lambda,
                pitch_kl_divergence=kl_loss_v/odim/self.l1_lambda,
                pitch_z_kl_divergence=kl_loss_z/odim/self.l1_lambda
            )
        else:
            pitch_loss = dict(pitch_loss=self.mse_criterion(p_outs['output'], ps))
        if self.energy_predictor_variational:
            mse_loss = self.mse_criterion(e_outs[0], es)
            rc_loss, kl_loss_z, kl_loss_v = self.vae_criterion(xs, **e_outs, masks=pitch_masks)
            energy_loss = dict(
                energy_loss=mse_loss,
                energy_reconstruction=rc_loss/odim/self.l1_lambda,
                energy_kl_divergence=kl_loss_v/odim/self.l1_lambda,
                energy_z_kl_divergence=kl_loss_z/odim/self.l1_lambda
            )
        else:
            energy_loss = dict(energy_loss=self.mse_criterion(e_outs['output'], es))
       
        
        return dict(
            l1_loss=l1_loss,
            **duration_loss,
            **pitch_loss, 
            **energy_loss, 
        )
