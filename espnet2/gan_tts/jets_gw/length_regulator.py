# Copyright 2022 Dan Lim
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging

import torch
import gw

from espnet2.tts.fastspeech_gw.length_regulator import LengthRegulator as GW
from espnet2.gan_tts.jets_gw.variance_predictor import VariationalVariancePredictor
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask


class LengthRegulator(torch.nn.Module):
    def __init__(self, variance_predictor:VariationalVariancePredictor):
        super().__init__()
        self.variance_predictor = variance_predictor
        self.length_regulator = GW()

    def forward(
        self, 
        xs : torch.Tensor, 
        ys : torch.Tensor, 
        text_lengths : torch.Tensor, 
        feats_lengths : torch.Tensor,
        ds : torch.Tensor = None
    ):
        masks = make_pad_mask(feats_lengths).to(xs.device)
        xs = gw.utils.interpolate(xs, text_lengths, feats_lengths, mode='nearest')
        pz, _ = self.variance_predictor.inference(xs, masks)
        
        pz = torch.nn.functional.pad(pz, [0,0,1,0])[...,:-1,:]
        pz = pz.masked_fill(masks.unsqueeze(-1), 0.0)
        pz = pz - pz.sum(-2, keepdim=True)/feats_lengths.unsqueeze(-1).unsqueeze(-1)
        pz = pz.cumsum(-2)
        pz = pz.masked_fill(masks.unsqueeze(-1), 0.0)
        
        qz, p, q = self.variance_predictor.forward(xs, ys, masks)
        
        qz = torch.nn.functional.pad(qz, [0,0,1,0])[...,:-1,:]
        qz = qz.masked_fill(masks.unsqueeze(-1), 0.0)
        qz = qz - qz.sum(-2, keepdim=True)/feats_lengths.unsqueeze(-1).unsqueeze(-1)
        qz = qz.cumsum(-2)
        qz = qz.masked_fill(masks.unsqueeze(-1), 0.0)
        
        xs, func = self.length_regulator(xs, qz, pz, ds, text_lengths, feats_lengths)  # (B, T_feats, adim)
        
        return xs, func, p, q
    
    def inference(
        self, 
        xs : torch.Tensor, 
        text_lengths : torch.Tensor, 
        feats_lengths : torch.Tensor
    ):
        masks = make_pad_mask(feats_lengths).to(xs.device)
        xs = gw.utils.interpolate(xs, text_lengths, feats_lengths, mode='nearest')
        pz, _ = self.variance_predictor.inference(xs, masks)
        
        pz = torch.nn.functional.pad(pz, [0,0,1,0])[...,:-1,:]
        pz = pz.masked_fill(masks.unsqueeze(-1), 0.0)
        pz = pz - pz.sum(-2, keepdim=True)/feats_lengths.unsqueeze(-1).unsqueeze(-1)
        pz = pz.cumsum(-2)
        pz = pz.masked_fill(masks.unsqueeze(-1), 0.0)
        
        xs, func = self.length_regulator(xs, pz)  # (B, T_feats, adim)
        
        return xs, func, pz
    
    def sampling(
        self, 
        n : int,
        xs : torch.Tensor, 
        ys : torch.Tensor, 
        text_lengths : torch.Tensor, 
        feats_lengths : torch.Tensor,
    ):
        masks = make_pad_mask(feats_lengths).to(xs.device)
        xs_ = gw.utils.interpolate(xs, text_lengths, feats_lengths, mode='nearest')
        qz, pz = self.variance_predictor.sampling(n, xs_, ys, masks)

        pz = torch.nn.functional.pad(pz, [0,0,1,0])[...,:-1,:]
        pz = pz.masked_fill(masks.unsqueeze(-1), 0.0)
        pz = pz - pz.sum(-2, keepdim=True)/feats_lengths.unsqueeze(-1).unsqueeze(-1)
        pz = pz.cumsum(-2)
        pz = pz.masked_fill(masks.unsqueeze(-1), 0.0)
        pz = pz*torch.unsqueeze(text_lengths/feats_lengths, -1)
            
        qz = torch.nn.functional.pad(qz, [0,0,1,0])[...,:-1,:]
        qz = qz.masked_fill(masks.unsqueeze(-1), 0.0)
        qz = qz - qz.sum(-2, keepdim=True)/feats_lengths.unsqueeze(-1).unsqueeze(-1)
        qz = qz.cumsum(-2)
        qz = qz.masked_fill(masks.unsqueeze(-1), 0.0)
        qz = qz*torch.unsqueeze(text_lengths/feats_lengths, -1)
        
        m = torch.eye(xs.size(-2), device=xs.device).unsqueeze(0)
        pm, _ = self.length_regulator.forward(m, pz)
        qm, _ = self.length_regulator.forward(m, qz)
        return pm, qm