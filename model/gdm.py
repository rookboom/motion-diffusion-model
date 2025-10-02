
import torch.nn as nn
import torch

from gatr.nets.axial_gatr import AxialGATr
from gatr.layers.attention.config import SelfAttentionConfig
from gatr.layers.mlp.config import MLPConfig
from model.mdm import PositionalEncoding, TimestepEmbedder, EmbedAction

from model.rotation2xyz import Rotation2xyz

class GDM(nn.Module):
    def __init__(self, 
                num_actions: int,
                num_heads: int, 
                hidden_mv_channels: int, 
                hidden_s_channels: int,
                dataset: str = 'uestc',
                cond_mode: str = 'action',
                num_blocks: int = 20,
                latent_dim: int = 256,
                pose_rep: str = 'rot6d',
                translation: bool = True,
                glob: bool = True,
                glob_rot: bool = True,
                dropout=0.1,
                cond_mask_prob: float = 0.,
                pos_embed_max_len: int = 5000):
        super().__init__()
        self.num_actions = num_actions  
        self.pose_rep = pose_rep
        self.translation = translation
        self.glob = glob
        self.glob_rot = glob_rot
        self.cond_mode = cond_mode
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.cond_mask_prob = cond_mask_prob
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout, max_len=pos_embed_max_len)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        self.embed_action = EmbedAction(self.num_actions, self.latent_dim)
        self.model = AxialGATr(
            in_mv_channels=3, 
            out_mv_channels=3, 
            hidden_mv_channels=hidden_mv_channels, 
            in_s_channels=self.latent_dim, 
            out_s_channels=0, 
            hidden_s_channels=hidden_s_channels,
            num_blocks=num_blocks,
            attention=SelfAttentionConfig(num_heads=num_heads),
            pos_encodings=(True, False),
            mlp=MLPConfig())

        self.rot2xyz = Rotation2xyz(device='cpu', dataset=dataset)            

    def forward(self, x, timesteps, y=None):
            """
            x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
            timesteps: [batch_size] (int)
            """
            time_emb = self.embed_timestep(timesteps)  
            force_mask = y.get('uncond', False)

            if 'action' in self.cond_mode:
                action_emb = self.embed_action(y['action'])
                emb = time_emb + self.mask_cond(action_emb, force_mask=force_mask)
            else: 
                # unconstrained
                emb = time_emb

            # Repeat the emb for all joints and frames
            emb = emb.squeeze(0).unsqueeze(1).unsqueeze(2).repeat(1, x.shape[3], x.shape[1], 1)  # [batch_size, max_frames, njoints, latent_dim]
            mv = rot6d_to_multivectors(x)  # [batch_size, max_frames, njoints, 3, 16]
            outputs_mv, _ = self.model(mv, emb)  # [batch_size, max_frames, njoints, 16]
            
            out = multivectors_to_rot6d(outputs_mv)
            return out

    def mask_cond(self, cond, force_mask=False):
        bs = cond.shape[-2]
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(1, bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond            

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]
    
    def _apply(self, fn):
        super()._apply(fn)
        self.rot2xyz.smpl_model._apply(fn)


    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.rot2xyz.smpl_model.train(*args, **kwargs)    


def rot6d_to_multivectors(x: torch.Tensor) -> torch.Tensor:
    # Converts 6dof to 3 multi-vectors representing oriented planes
    # This has shown to be more effective than embedding rotations as bivectors.
    # See [gatr issues](https://github.com/Qualcomm-AI-research/geometric-algebra-transformer/issues/9).
    # nfeats is expected to be 6 (rot6d)
    # permute from bjft to btjf
    x = x.permute(0, 3, 1, 2)  # [batch_size, max_frames, njoints, nfeats]
    # reshape 6dof to 2x3
    x = x.reshape(*x.shape[:-1], 2, 3)
    third_vec = torch.linalg.cross(x[..., [0], :], x[..., [1], :])
    x = torch.cat((x, third_vec), dim=-2)

    batch_shape = x.shape[:-1]
    multivector = torch.zeros(*batch_shape, 16, dtype=x.dtype, device=x.device)

    # Embedding a plane through origin into vectors
    multivector[..., 2:5] = x[..., :]
    return multivector

def multivectors_to_rot6d(mv: torch.Tensor) -> torch.Tensor:
    # Inverse of rot6d_to_multivectors
    # mv is expected to have shape [..., 16]
    # Extract the 3 vectors from the multivector
    x = mv[..., 2:5]  # [..., 3, 3]
    # Return only the first two vectors as 6dof
    x = x[..., :2, :]  # [..., 2, 3]
    x = x.reshape(*x.shape[:-2], 6)  # [..., 6]
    # Permute back to bjft
    x = x.permute(0, 2, 3, 1)  # [batch_size, njoints, nfeats, max_frames]
    return x  # [batch_size, njoints, nfeats, max_frames]