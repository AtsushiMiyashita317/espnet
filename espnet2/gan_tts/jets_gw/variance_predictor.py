import torch

from espnet2.tts.fastspeech_gw.variance_predictor import VariancePredictor, AlignmentModule

class VariationalVariancePredictor(torch.nn.Module):
    def __init__(
        self,
        xdim,
        ydim,
        hdim,
        odim,
        n_layers,
        n_chans,
        kernel_size,
    ):
        super().__init__()
        # define duration predictor
        self.prior_encoder = VariancePredictor(
            idim=xdim,
            odim=hdim*2,
            n_layers=n_layers,
            n_chans=n_chans,
            kernel_size=kernel_size,
            dropout_rate=0.0   
        )
            
        self.prior_decoder = VariancePredictor(
            idim=hdim,
            odim=odim*2,
            n_layers=n_layers,
            n_chans=n_chans,
            kernel_size=kernel_size,
            dropout_rate=0.0   
        )
            
        self.posterior_encoder = AlignmentModule(
            tdim=xdim,
            fdim=ydim,
            odim=hdim*2,
            n_layers=n_layers,
            n_chans=n_chans,
            kernel_size=kernel_size,
            dropout_rate=0.0   
        )
            
        self.posterior_decoder = AlignmentModule(
            tdim=hdim,
            fdim=ydim,
            odim=odim*2,
            n_layers=n_layers,
            n_chans=n_chans,
            kernel_size=kernel_size,
            dropout_rate=0.0   
        )
        
    
    def forward(self, xs:torch.Tensor, ys:torch.Tensor, masks:torch.Tensor):
        _ = self.prior_encoder.forward(xs, masks) # (B, T_text, adim)
        mu1, ln_var1 = _.chunk(2, -1)  # (B, T_text, adim)
        hs = self.posterior_encoder.forward(xs, ys, masks) # (B, T_text, adim)
        mu2, ln_var2 = hs.chunk(2, -1)  # (B, T_text, adim)
        hs = mu2 + torch.randn_like(mu2)*ln_var2.mul(0.5).exp()
        
        _ = self.prior_decoder.forward(hs, masks)  # (B, T_text, n_iter)
        mu3, ln_var3 = _.chunk(2, -1)  # (B, T_text, n_iter)
        hs = self.posterior_decoder.forward(hs, ys, masks)  # (B, T_text, n_iter)
        mu4, ln_var4 = hs.chunk(2, -1)  # (B, T_text, n_iter)
        zs = mu4 + torch.randn_like(mu4)*ln_var4.mul(0.5).exp()
        
        p = torch.cat([mu1, mu3, ln_var1, ln_var3], dim=-1)
        q = torch.cat([mu2, mu4, ln_var2, ln_var4], dim=-1)
        
        return zs, p, q
        
    
    def inference(self, xs, masks):
        hs = self.prior_encoder.forward(xs, masks) # (B, T_text, adim)
        mu1, ln_var1 = hs.chunk(2, -1)  # (B, T_text, adim)
        hs = mu1 + torch.randn_like(mu1)*ln_var1.mul(0.5).exp()
        
        hs = self.prior_decoder.forward(hs, masks)  # (B, T_text, n_iter)
        mu3, ln_var3 = hs.chunk(2, -1)  # (B, T_text, n_iter)
        zs = mu3 + torch.randn_like(mu3)*ln_var3.mul(0.5).exp()
        
        p = torch.cat([mu1, mu3, ln_var1, ln_var3], dim=-1)
        
        return zs, p
    
    def sampling(self, n:int, xs:torch.Tensor, ys:torch.Tensor, masks:torch.Tensor):
        assert xs.size(0) == 1
        xs = xs.expand((xs.size(0)*n,) + xs.size()[1:])
        ys = ys.expand((ys.size(0)*n,) + ys.size()[1:])
        masks = masks.expand((masks.size(0)*n,) + masks.size()[1:])
        qz,*_ = self.forward(xs, ys, masks)
        pz,*_ = self.inference(xs, masks)
        return qz, pz
        