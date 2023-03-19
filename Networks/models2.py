# basic utils

import einops
# RLbench if needed


# deep learning stuff
import torch
import torch.nn as nn

# project-specific
# embeddings
from .utils import DiscreteEmbedding as emb
# components
from .networks import ConvLayer
from .transformer.transformer import TransformerEncoder
# masking
from .utils import get_causal_mask, get_present_features

class TemporalTransformer(nn.Module):
    """
    This is an interface for a temporal transformer encoder takes causal mask.
    It could become a multiway transformer if input is multimodal.

    Arguments
    ----------
    num_layers: int
        number of transformer layers
    nhead: int
        number of head
    d_ffn: int
        feedforward network dimension
    d_model: int
        hidden dim of transformer
    kdim:
        not used
    vdim: 
        not used
    dropout: float
        drop out probability
    activation: function
        activation function
    normailze_before: bool
        layer norm before self-attention or after
    expert_counts: int
        number of parallal feedforward netowkr per layer (> 1, multiway transformer)
    """

    def __init__(
        self, 
        num_layers,
        nhead,
        d_ffn,
        d_model,
        kdim=None,
        vdim=None,
        dropout=0.0,
        activation=nn.ReLU,
        normalize_before=True,
        expert_counts = 1,
        cameras = 3,
    ):
        super().__init__()
        
        self.expert_counts = expert_counts
        self.d_model = d_model
        self.cameras = cameras
        self.encoder = TransformerEncoder( 
                    num_layers = num_layers,
                    nhead = nhead,
                    d_ffn = d_ffn,
                    d_model= d_model,
                    kdim=None,
                    vdim=None,
                    dropout=dropout,
                    activation=activation,
                    normalize_before=normalize_before,
                    expert_counts = expert_counts,
                    views = self.cameras)
    
    def forward(self, x, causal_mask, padding_mask):
        """
        This is a temporal transformer encoder takes causal mask.
        It could become a multiway transformer if input is multimodal.

        Arguments
        ----------
        x: 
            padded input.
        causal_mask:
            maskout future for autregressive workflow
        padding_mask:
            restrict attention to unpadded area
        """
        out, attn = self.encoder(
                src = x,
                src_mask = causal_mask,
                src_key_padding_mask = padding_mask,
        )
        return out

class Models(nn.Module):
    """
    This is a general pipeline wrapper to chain all trainable components together.

    Arguments
    ----------
    frontend: nn.Module
        a visual feature extractor. (Unet-encoder)
    cross_atten1: nn.Module
        a cross-attention transformer that fused 2 modailties together
        used by language (eos feature) attends to vision
    cross_atten2: nn.Module
        a cross-attention transformer that fused 2 modailties together
        used by vision attends to language (tokens features)
    policy: nn.Module
        a temporal transformer acts as a policy.
        It's output will modified residuals comes from Unet-encoder and thus affect Unet decoding.
    backend: nn.Module
        an NN predicting rotations, positions and gripper state
    depth: int
        depth of the Unet-decoder (should be similar to the Unet encoder depth)
    fusion: nn.Module
        additive fusion to fuse language goal and vision observation
    """

    def __init__(
        self, 
        frontend,
        cross_atten1,
        cross_atten2,
        policy,
        backend,
        depth,
        fusion,
    ):
        super().__init__()
        
        self.frontend = frontend
        self.cross_atten1 = cross_atten1
        self.cross_atten2 = cross_atten2
        self.policy = policy
        self.backend = backend
        self.fusion = fusion
        self.trans_decoder = nn.ModuleList()
        self.depth = depth
        for i in range(depth):
            self.trans_decoder.extend(
                [
                    nn.Sequential(
                        ConvLayer(
                            in_channels= 32 ,
                            out_channels= 16,
                            kernel_size=(3, 3),
                            stride_size=(1, 1),
                        ),
                        nn.Upsample(
                            scale_factor=2, mode="bilinear", align_corners=True
                        ),
                    )
                ]
            )
        if self.policy != None and self.cross_atten2 is not None:
            # create modality embedding for vision and language latent goal
            number_modalities = 2
            self.modality_emb = emb(number_modalities, self.policy.d_model)
            self.mod_id = torch.arange(0, 2).long()
            # view emb == camera emb
            views = self.policy.expert_counts // number_modalities
            self.view_emb = emb(views, self.policy.d_model)
            self.view_id = torch.arange(0, views).long()
            self.emb_norm = nn.LayerNorm(self.policy.d_model)

        self._init_params()

    def forward(self, 
        rgb,
        pcd,
        padding_mask_vision,
        instructions,
        padding_mask_lang,
        ):
        """
        chain front-end, vision_language, policy, and backend in a role

        Arguments
        ----------
        rgb:
            rgb observations
        pcd:
            point cloud observations
        padding_mask_vision:
            padding mask created by dataset.py for padded visual demo
        instructions:
            a tuple = (tokens_feature, esoses_features)
            the language instructions features needs to follow 
        padding_mask_lang:
            padding mask created by lang_feature.py for padded language instructions
            it is only used by VALA
        """
        tokens, eoses = instructions
        # obtain visual features in patch. Vectorize each patch and add embeddings
        # last layer feature and residuals from U-net encoder are alway returned
        visual_tokens, residuals, views = self.frontend(rgb, pcd)
        # reverse the order of residuals (for unet decoding)
        residuals.reverse()
        _, _ch, H,W = residuals[0].shape
        
        if self.cross_atten2 is None:
            # LAVA forward
            features = self.LAVA_forward( eoses, visual_tokens, padding_mask_vision)
            features = einops.rearrange(
                features,
                '(B views) T (H W ch) -> B T views ch H W', views = views, H = H, W = W)
            decoded_features = self.unet_decode(features, residuals, padding_mask_vision)
            # prediction and rotations and position separately
            lang_features = None
            vision_features = features
        else:
            # VALA forward
            # vision mask is inflated so that 1st dim: Bxviews
            lang_goal, inflate_pad_mask = self.VALA_forward( tokens, padding_mask_lang, visual_tokens.clone(), padding_mask_vision)
            visual_obs = einops.rearrange(visual_tokens, 'B T views dim -> (B views) T dim')
            features = self.policy_forward(lang_goal, visual_obs, inflate_pad_mask)
            if self.fusion is not None:
                vision_features = einops.rearrange( features, '(B N) T (H W ch) -> B T N ch H W', N = views, H = H, W = W )
                lang_features = None
            else:
                features = einops.rearrange( features, '(B N) (T M) (H W ch) -> B T M N ch H W', N = views, M = 2, H = H, W = W )
                vision_features = features[:,:,1]
                lang_features = features[:,:,0]
            decoded_features = self.unet_decode(vision_features, residuals, padding_mask_vision)
        
        # prediction and rotations and position separately
        vision_features = einops.rearrange(
                vision_features,
                'B T views ch H W -> B T views (H W ch)', views = views, H = H, W = W)
        
        predictions = self.backend(
            views, pcd, lang_features, vision_features, decoded_features, padding_mask_vision, instructions)

        return predictions

    def LAVA_forward(self, eoses, visual_tokens, padding_mask_vision):
        """Language-Attends-Vision-to-Act (LAVA): language queries visual tokens
        
        Arguments
        ----------
        eoses:
            CLIP eos features
        visual_tokens:
            visual features from Unet-encoder last layer
        padding_mask_vision:
            mask to determine unpadded timesteps
        """
        B, T, views, dim = visual_tokens.shape
        # collapse batch and view together
        visual_tokens = einops.rearrange(visual_tokens, "B T views dim -> (B views) T dim")
        # inflate eoses to match new batch dim and horizon dim
        eoses = einops.repeat(
            eoses[:, None, :], 
            "B dummy dim -> (B views) (T dummy) dim", views = views, T = T)
        # cross attention
        return self.cross_atten1(visual_tokens, padding_mask_vision, eoses, None, views)

    def VALA_forward(self, tokens, padding_mask_lang, visual_tokens, padding_mask_vision):
        """Vision-Attends-Language-to-Act (VALA): visual tokens queries language tokens
        
        Arguments
        ----------
        tokens:
            token-level features, from wave2vec or CLIP or something else
        padding_mask_lang:
            denotes padded token
        visual_tokens:
            visual features from Unet-encoder last layer
        padding_mask_vision:
            mask to determine unpadded timesteps
        """
        temporal_features, inflate_vision_padding = self.cross_atten2(visual_tokens, padding_mask_vision, tokens, padding_mask_lang)
        return temporal_features, inflate_vision_padding

    def policy_forward(self, lang_goal, vision, vision_padding ):
        """forward the temporal transformer policy

        Arguments
        ----------
        lang_goal:
            temporally learned language goal
        vision:
            visual observation sequence
        vision_padding:
            denotes unpadd visual tokens
        """
        if self.fusion is not None:
            joint_features = self.fusion(lang_goal, vision)
            policy_padding = ~vision_padding
        else:
            joint_features = einops.rearrange(
                torch.cat([lang_goal, vision], dim = 1),
                "B (M T) dim -> B (T M) dim",M = 2)
            policy_padding = einops.repeat(
                ~vision_padding, # reverse the unpad mask to pad mask
                "B_N T -> B_N (T M)", M = 2
            )
        causal_mask = get_causal_mask(joint_features)
        out = self.policy(
            x = joint_features,
            causal_mask = causal_mask, padding_mask = policy_padding)
        return out

    def unet_decode(self, decode_feat, residuals, padding_mask_vision):
        """During decoding, the feature map of unet decoder is affected by the 
        multi-view FiLM generater. Particularly, the residuals from encoder 
        during decoding is affected by FiLM' generated scale and bias
        
        Arguments
        ----------
        decode_feat:
            the unet feature from unet encoder that is need to be decoded by unet decoder
        residuals:
            the residuals (feature map) from unet encoder 
        padding_mask_vision:
            if true, it means the timestep is not padded, false is padded.
            Use to select unpad visual feature
        """
        b, t, n, ch, h, w =  decode_feat.shape
        decode_feat = einops.rearrange(
                decode_feat[padding_mask_vision],
                'unpad n ch h w -> (unpad n) ch h w')

        for l, unet_decode_layer in enumerate(self.trans_decoder):
            if l > 0:
                residual = residuals[l]
                residual = einops.rearrange(
                    residual, 
                    "(b t n) c h w -> b t n c h w", b=b, t=t, n=n)[padding_mask_vision]
                residual = einops.rearrange(residual, "pad_batch n c h w -> (pad_batch n) c h w")
                try:
                    decode_feat = torch.cat([decode_feat, residual  ], dim =1)
                except:
                    assert False, f"strange dim: {decode_feat.shape}, {residual.shape}"
            decode_feat = unet_decode_layer(decode_feat)
            
        return einops.rearrange( decode_feat, "(pad_b n) ch h w -> pad_b n ch h w", n = n)

    

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p)

        