from collections import defaultdict
from typing import Dict, List
import logging

import torch
import gw

from espnet2.gan_tts.jets.alignments import AlignmentModule
from espnet2.gan_tts.hifigan.loss import MelSpectrogramLoss
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.tts.abs_tts import AbsTTS
from espnet.nets.pytorch_backend.rnn.attentions import (
    AttAdd,
    AttCov,
    AttCovLoc,
    AttDot,
    AttForward,
    AttForwardTA,
    AttLoc,
    AttLoc2D,
    AttLocRec,
    AttMultiHeadAdd,
    AttMultiHeadDot,
    AttMultiHeadLoc,
    AttMultiHeadMultiResLoc,
    NoAtt,
)
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet2.tts.fastspeech_gw.length_regulator import LengthRegulator
from espnet.nets.pytorch_backend.nets_utils import pad_list

@torch.no_grad()
def calculate_all_attentions(
    model: AbsESPnetModel, batch: Dict[str, torch.Tensor]
) -> Dict[str, List[torch.Tensor]]:
    """Derive the outputs from the all attention layers

    Args:
        model:
        batch: same as forward
    Returns:
        return_dict: A dict of a list of tensor.
        key_names x batch x (D1, D2, ...)

    """
    bs = len(next(iter(batch.values())))
    assert all(len(v) == bs for v in batch.values()), {
        k: v.shape for k, v in batch.items()
    }

    # 1. Register forward_hook fn to save the output from specific layers
    outputs = {}
    handles = {}
    for name, modu in model.named_modules():
        def hook(module, input, output, name=name):
            if isinstance(module, MultiHeadedAttention):
                # NOTE(kamo): MultiHeadedAttention doesn't return attention weight
                # attn: (B, Head, Tout, Tin)
                outputs[name] = module.attn.detach().cpu()
            elif isinstance(module, AttLoc2D):
                c, w = output
                # w: previous concate attentions
                # w: (B, nprev, Tin)
                att_w = w[:, -1].detach().cpu()
                outputs.setdefault(name, []).append(att_w)
            elif isinstance(module, (AttCov, AttCovLoc)):
                c, w = output
                assert isinstance(w, list), type(w)
                # w: list of previous attentions
                # w: nprev x (B, Tin)
                att_w = w[-1].detach().cpu()
                outputs.setdefault(name, []).append(att_w)
            elif isinstance(module, AttLocRec):
                # w: (B, Tin)
                c, (w, (att_h, att_c)) = output
                att_w = w.detach().cpu()
                outputs.setdefault(name, []).append(att_w)
            elif isinstance(
                module,
                (
                    AttMultiHeadDot,
                    AttMultiHeadAdd,
                    AttMultiHeadLoc,
                    AttMultiHeadMultiResLoc,
                ),
            ):
                c, w = output
                # w: nhead x (B, Tin)
                assert isinstance(w, list), type(w)
                att_w = [_w.detach().cpu() for _w in w]
                outputs.setdefault(name, []).append(att_w)
            elif isinstance(
                module,
                (
                    AttAdd,
                    AttDot,
                    AttForward,
                    AttForwardTA,
                    AttLoc,
                    NoAtt,
                ),
            ):
                c, w = output
                att_w = w.detach().cpu()
                outputs.setdefault(name, []).append(att_w)
            elif isinstance(module, AlignmentModule):
                w = output
                att_w = torch.exp(w).detach().cpu()
                outputs.setdefault(name, []).append(att_w)
            elif isinstance(module, (LengthRegulator)):
                xs, _, ps, ds, ilens, olens = input
                _, fq = output
                _, fp = module.forward(xs.detach(), ps)
                
                fq = fq*torch.unsqueeze(ilens/olens, -1)
                fp = fp*torch.unsqueeze(ilens/olens, -1)
                
                i = torch.arange(ds.size(-1), device=ds.device)
                repeat = [torch.repeat_interleave(i, d, dim=0) for d in ds]
                f = pad_list(repeat, 0.0)
                
                m = torch.eye(ds.size(-1), device=fp.device).unsqueeze(0)
                mp = gw.cubic_interpolation(m, fp.detach()).transpose(-1,-2).detach().cpu()
                mq = gw.cubic_interpolation(m, fq.detach()).transpose(-1,-2).detach().cpu()
                mfa = gw.cubic_interpolation(m, f.to(device=m.device, dtype=m.dtype)).transpose(-1,-2).detach().cpu()
                outputs[name] = [mp, mq, mfa]
                    
        handle = modu.register_forward_hook(hook)
        handles[name] = handle

    # 2. Just forward one by one sample.
    # Batch-mode can't be used to keep requirements small for each models.
    keys = []
    for k in batch:
        if not (k.endswith("_lengths") or k in ["utt_id"]):
            keys.append(k)

    return_dict = defaultdict(list)
    for ibatch in range(bs):
        # *: (B, L, ...) -> (1, L2, ...)
        _sample = {
            k: batch[k][ibatch, None, : batch[k + "_lengths"][ibatch]]
            if k + "_lengths" in batch
            else batch[k][ibatch, None]
            for k in keys
        }

        # *_lengths: (B,) -> (1,)
        _sample.update(
            {
                k + "_lengths": batch[k + "_lengths"][ibatch, None]
                for k in keys
                if k + "_lengths" in batch
            }
        )

        if "utt_id" in batch:
            _sample["utt_id"] = batch["utt_id"]

        model(**_sample)

        # Derive the attention results
        for name, output in outputs.items():
            if isinstance(output, list):
                if isinstance(output[0], list):
                    # output: nhead x (Tout, Tin)
                    output = torch.stack(
                        [
                            # Tout x (1, Tin) -> (Tout, Tin)
                            torch.cat([o[idx] for o in output], dim=0)
                            for idx in range(len(output[0]))
                        ],
                        dim=0,
                    )
                else:
                    # Tout x (1, Tin) -> (Tout, Tin)
                    output = torch.cat(output, dim=0)
            else:
                # output: (1, NHead, Tout, Tin) -> (NHead, Tout, Tin)
                output = output.squeeze(0)
            # output: (Tout, Tin) or (NHead, Tout, Tin)
            return_dict[name].append(output)
        outputs.clear()

    # 3. Remove all hooks
    for _, handle in handles.items():
        handle.remove()

    return dict(return_dict)


@torch.no_grad()
def calculate_feats(
    model: AbsESPnetModel, batch: Dict[str, torch.Tensor]
) -> Dict[str, List[torch.Tensor]]:
    """Derive the outputs from the all attention layers

    Args:
        model:
        batch: same as forward
    Returns:
        return_dict: A dict of a list of tensor.
        key_names x batch x (D1, D2, ...)

    """
    bs = len(next(iter(batch.values())))
    assert all(len(v) == bs for v in batch.values()), {
        k: v.shape for k, v in batch.items()
    }

    outputs = {}
    handles = {}
    for name, modu in model.named_modules():

        def hook(module, input, output, name=name):
            if isinstance(module, AbsTTS):
                if type(output) is dict:
                    if 'feats' in output:
                        outputs[name] = output['feats']
            elif isinstance(module, MelSpectrogramLoss):
                y_hat, y, *_ = input
                mel_hat, _ = module.wav_to_mel(y_hat.squeeze(1))
                mel, _ = module.wav_to_mel(y.squeeze(1))
                outputs[name] = torch.stack([mel, mel_hat], dim=1)
        handle = modu.register_forward_hook(hook)
        handles[name] = handle

    
    # 2. Just forward one by one sample.
    # Batch-mode can't be used to keep requirements small for each models.
    keys = []
    for k in batch:
        if not (k.endswith("_lengths") or k in ["utt_id"]):
            keys.append(k)

    return_list = []
    for ibatch in range(bs):
        # *: (B, L, ...) -> (1, L2, ...)
        _sample = {
            k: batch[k][ibatch, None, : batch[k + "_lengths"][ibatch]]
            if k + "_lengths" in batch
            else batch[k][ibatch, None]
            for k in keys
        }

        # *_lengths: (B,) -> (1,)
        _sample.update(
            {
                k + "_lengths": batch[k + "_lengths"][ibatch, None]
                for k in keys
                if k + "_lengths" in batch
            }
        )

        if "utt_id" in batch:
            _sample["utt_id"] = batch["utt_id"]
            
        model(**_sample)

        for name, output in outputs.items():
            return_list.append(output.squeeze(0))
        outputs.clear()
        
    # 3. Remove all hooks
    for _, handle in handles.items():
        handle.remove()

    return return_list

@torch.no_grad()
def calculate_alignments(
    model: AbsESPnetModel, batch: Dict[str, torch.Tensor]
) -> Dict[str, List[torch.Tensor]]:
    """Derive the outputs from the all attention layers

    Args:
        model:
        batch: same as forward
    Returns:
        return_dict: A dict of a list of tensor.
        key_names x batch x (D1, D2, ...)

    """
    bs = len(next(iter(batch.values())))
    assert all(len(v) == bs for v in batch.values()), {
        k: v.shape for k, v in batch.items()
    }

    # 1. Register forward_hook fn to save the output from specific layers
    outputs = {}
    handles = {}
    for name, modu in model.named_modules():
        def hook(module, input, output, name=name):
            if isinstance(module, (LengthRegulator)):
                xs, _, ps, ds, ilens, olens = input
                _, fq = output
                _, fp = module.forward(xs.detach(), ps)
                
                i = torch.arange(ds.size(-1), device=ds.device)
                repeat = [torch.repeat_interleave(i, d, dim=0) for d in ds]
                f = pad_list(repeat, 0.0)*torch.unsqueeze(olens/ilens, -1)
                
                m = torch.eye(xs.size(-2), device=fp.device).unsqueeze(0)
                mp = gw.cubic_interpolation(m, fp.detach()).transpose(-1,-2).detach().cpu()
                mq = gw.cubic_interpolation(m, fq.detach()).transpose(-1,-2).detach().cpu()
                mfa = gw.cubic_interpolation(m, f.to(device=m.device)).transpose(-1,-2).detach().cpu()
                outputs[name] = [mp, mq, mfa]
                    
        handle = modu.register_forward_hook(hook)
        handles[name] = handle

    # 2. Just forward one by one sample.
    # Batch-mode can't be used to keep requirements small for each models.
    keys = []
    for k in batch:
        if not (k.endswith("_lengths") or k in ["utt_id"]):
            keys.append(k)

    return_dict = defaultdict(list)
    for ibatch in range(bs):
        # *: (B, L, ...) -> (1, L2, ...)
        _sample = {
            k: batch[k][ibatch, None, : batch[k + "_lengths"][ibatch]]
            if k + "_lengths" in batch
            else batch[k][ibatch, None]
            for k in keys
        }

        # *_lengths: (B,) -> (1,)
        _sample.update(
            {
                k + "_lengths": batch[k + "_lengths"][ibatch, None]
                for k in keys
                if k + "_lengths" in batch
            }
        )

        if "utt_id" in batch:
            _sample["utt_id"] = batch["utt_id"]

        model(**_sample)

        # Derive the attention results
        for name, output in outputs.items():
            if isinstance(output, list):
                if isinstance(output[0], list):
                    # output: nhead x (Tout, Tin)
                    output = torch.stack(
                        [
                            # Tout x (1, Tin) -> (Tout, Tin)
                            torch.cat([o[idx] for o in output], dim=0)
                            for idx in range(len(output[0]))
                        ],
                        dim=0,
                    )
                else:
                    # Tout x (1, Tin) -> (Tout, Tin)
                    output = torch.cat(output, dim=0)
            else:
                # output: (1, NHead, Tout, Tin) -> (NHead, Tout, Tin)
                output = output.squeeze(0)
            # output: (Tout, Tin) or (NHead, Tout, Tin)
            return_dict[name].append(output)
        outputs.clear()

    # 3. Remove all hooks
    for _, handle in handles.items():
        handle.remove()

    return dict(return_dict)

