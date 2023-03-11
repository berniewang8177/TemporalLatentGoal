# basic utils

import einops
# deep learning stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
# project-specific
# embeddings
from .utils import DiscreteEmbedding as emb
# components
from .networks import ConvLayer


# 1 vector feature at sentence level
CLIP1 = 'CLIP-origin' 
# features of a tokenized sentence at word level
CLIP2 = 'CLIP-modified' 
BERT = 'BERT'

class LanguageFrontend(nn.Module):
    """
    Extract language featurers with CLIP / BERT during DATA GENERATION.

    LAVA uses 1 vector embedding from CLIP or mean-pool BERT.
    VALA uses sequence of embeddings from BERT last layer.

    Arguments
    ----------
    ?
    ?
    """

    def __init__(
        self, 
        a,
        b,
        feature_type = 'CLIP-origin'
    ):
        super().__init__()
        self.feature_type = feature_type
    def forward(self, x):
        if self.feature_type == BERT or self.feature_type == CLIP2:
            pass
        else:
            pass



class HiveFormerVisionFrontend(nn.Module):
    """
    Extract multiview visual featurers with during TRAINING for LAVA.

    Raw visual feature is first extracted with U-net encoder.
    We apply patch tokenization on the visual input and add embeddings.

    Arguments
    ----------
    feat_layers:
        number of layers of feature encoders
    d_model:
        dimension of the transformer
    
    ?
    """

    def __init__(
        self, 
        feat_layers,
        d_model,
        cameras,
        max_horizons,
        max_patches,
        
    ):
        super().__init__()

        # embeddings
        self.camera_emb = emb(cameras, d_model)
        self.time_emb = emb(max_horizons, d_model)
        self.patch_emb = emb(max_patches, d_model)
        # 19 = 16 + 3 (encoded rgb + 3 from pointcloud)
        self.visual_embedding = nn.Linear(19, d_model)

        self.camera_norm = nn.LayerNorm(d_model)
        self.time_norm = nn.LayerNorm(d_model)
        self.patch_norm = nn.LayerNorm(d_model)
        self.visual_norm = nn.LayerNorm(d_model)
        
        # Input RGB + Point Cloud Preprocess (SiameseNet)
        self.rgb_preprocess = ConvLayer(
            4,  # 3 channel + 1 (attn map)
            8,
            kernel_size=(3, 3),
            stride_size=(1, 1),
            apply_norm=False,
        )
        self.to_feat = ConvLayer(
            8,
            16,
            kernel_size=(1, 1),
            stride_size=(1, 1),
            apply_norm=False,
        )

        # feature encoding
        self.feature_encoder = nn.ModuleList()
        for i in range(feat_layers):
            self.feature_encoder.append(
                ConvLayer(
                    in_channels=16,
                    out_channels=16,
                    kernel_size=(3, 3),
                    stride_size=(2, 2),
                    residual=True,
                )
            )
    
    def forward(self, 
        rgb,
        pcd,
        padding_mask,
        ):
        """
        For an image-like input, first tokenized it.
        Finally, add embedding to each visual tokens and return.

        Arguments
        ----------
        rgb:
            rgb observations
        pcd:
            point cloud observations
        padding:
            padding mask created by dataset.py for padded demo
        """
        # processing encoding feature
        B, T, N = rgb.shape[:3]

        rgb_obs_ = einops.rearrange(rgb, "b t n ch h w -> (b t n) ch h w")
        rgb_obs_ = self.rgb_preprocess(rgb_obs_)

        x = self.to_feat(rgb_obs_)

        # encoding features with U-net encoder
        enc_feat = []
        for l in self.feature_encoder:
            x, res = l(x)
            # res = einops.rearrange(res, "(b t n) c h w -> b t n c h w", n=N, t=T)
            # res = res[padding_mask]
            # res = einops.rearrange(res, "bpad n c h w -> (bpad n) c h w")
            # residual are collected and will be "added" layer-by-layer in reverse order when doing prediction
            enc_feat.append(res)
        
        # collect the last layer feature from Unet encoder, used by Unet decoder
        unet_last_feature = x
        x = einops.rearrange(x, "(b t n) c h w -> b t n c h w", n=N, t=T)

        # postponed random masking implemented in HiveFromer
        
        # in line 673-675, 687, 689
        # https://github.com/guhur/hiveformer/blob/main/network.py
        # there's a padding and concatenation happening
        # I don't have it here.

        # Add extra channels with Point Clouds
        pcd = einops.rearrange(pcd, "b t n c h w -> (b t n) c h w")
        pcd = F.avg_pool2d(pcd, 16)
        pcd = einops.rearrange(pcd, "(b t n) c h w -> b t n c h w", b=B, t=T, n=N)
        x = torch.cat([x, pcd], dim = 3)

        return self.tokenize_and_embed(x), unet_last_feature, enc_feat, N
    
    def tokenize_and_embed(self,
        x
        ):
        """
        For an image-like input, first tokenized it.
        Finally, add embedding to each visual tokens and return.

        Arguments
        ----------
        x:
            visual observation (gripper pose, rgb, point cloud info)
        """
        B, T, N, C, H, W = x.shape

        timesteps = torch.arange(T).type_as(x).unsqueeze(0).long()
        time_emb = self.time_emb(timesteps)

        time_emb = self.time_norm(time_emb).squeeze(0)
        time_emb = einops.repeat(time_emb, "t d -> b t n h w d", b=B, n=N, h=H, w=W)

        # patch_id = torch.arange(H * W).type_as(x).unsqueeze(0).long()
        # patch_emb = self.patch_emb(patch_id)
        # patch_emb = self.patch_norm(patch_emb).squeeze(0)
        # patch_emb = einops.repeat(
        #     patch_emb, "(h w) d -> b t n h w d", b=B, n=N, t=T, h=H, w=W
        # )

        # cam_id = torch.arange(N).type_as(x).unsqueeze(0).long()
        # cam_emb = self.camera_emb(cam_id)
        # cam_emb = self.camera_norm(cam_emb).squeeze(0)
        # cam_emb = einops.repeat(cam_emb, "n d -> b t n h w d", b=B, h=H, w=W, t=T)

        # encoding history
        xe = einops.rearrange(x, "b t n c h w -> b t n h w c")
        # emb dim 19 to 64
        xe = self.visual_embedding(xe)
        xe = self.visual_norm(xe)
        xe += time_emb # + patch_emb  + cam_emb 

        xe = einops.rearrange(xe, "b t n h w c -> b t n (h w) c")

        return xe