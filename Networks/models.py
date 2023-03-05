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
    film: nn.Module
        a multi-view multi-layer FiLM generator that generates
        scales and biases to modify Unet's decoding by conditioning on policy's output
    backend: nn.Module
        an NN predicting rotations, positions and gripper state
    depth: int
        depth of the Unet-decoder (should be similar to the Unet encoder depth)
    """

    def __init__(
        self, 
        frontend,
        cross_atten1,
        cross_atten2,
        policy,
        film,
        unet_cross,
        backend,
        depth,
    ):
        super().__init__()
        
        self.frontend = frontend
        self.cross_atten1 = cross_atten1
        self.cross_atten2 = cross_atten2
        self.policy = policy
        self.film = film
        self.unet_cross = unet_cross
        self.backend = backend

        self.trans_decoder = nn.ModuleList()
        self.depth = depth
        for i in range(depth):
            if i == 0:
                self.trans_decoder.extend(
                    [
                        nn.Sequential(
                            ConvLayer(
                                in_channels=16 if unet_cross is None else 64,
                                out_channels=16 if unet_cross is None else 64,
                                kernel_size=(3, 3),
                                stride_size=(1, 1),
                            ),
                            nn.Upsample(
                                scale_factor=2, mode="bilinear", align_corners=True
                            ),
                        )
                    ]
                )
            elif i == depth - 1:
                self.trans_decoder.extend(
                    [
                        nn.Sequential(
                            ConvLayer(
                                in_channels= 32 if unet_cross is None else 64 + 16,
                                out_channels=16,
                                kernel_size=(3, 3),
                                stride_size=(1, 1),
                            ),
                            nn.Upsample(
                                scale_factor=2, mode="bilinear", align_corners=True
                            ),
                        )
                    ]
                )
            else:
                self.trans_decoder.extend(
                    [
                        nn.Sequential(
                            ConvLayer(
                                in_channels= 32 if unet_cross is None else 64 + 16,
                                out_channels= 16 if unet_cross is None else 64,
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
        visual_tokens, unet_last_feature, residuals, views = self.frontend(rgb, pcd, padding_mask_vision)

        # LAVA: language vector (eos features as query) attends to visual tokens (keys and values)
        cross_feature, new_padding_mask = self.cross_atten1(
            visual_tokens, padding_mask_vision, eoses, padding_mask_lang)

        # VALA: visual queries attends to language tokens (keys and values)
        if self.cross_atten2 is not None:
            cross_feature2, _useless_mask = self.cross_atten2(
                cross_feature, new_padding_mask, tokens, padding_mask_lang, views)
            lang_attend_vision = einops.rearrange(cross_feature, "B (T N) dim -> B T N dim", N = views)
            B,T,N,dim = lang_attend_vision.shape
            vision_attend_language = einops.rearrange(cross_feature2, "B (T N) dim -> B T N dim", N = views)
            cross_feature = torch.cat([vision_attend_language, lang_attend_vision], dim = 2)
            cross_feature = einops.rearrange(cross_feature, "B T N dim -> B (T N) dim")
            new_padding_mask = new_padding_mask.repeat_interleave(2, dim = 1)
            # add view and modality embedding
            mod_embs = self.modality_emb(self.mod_id.to(cross_feature.device))[None,None,:]
            mod_embs = einops.repeat(mod_embs, 'b t M d -> (B b) (T t M view) d',B = B, T = T, view = views )
            if self.policy.expert_counts == 6:
                view_embs = self.view_emb(self.view_id.to(cross_feature.device))[None, None,:] 
                view_embs = einops.repeat(view_embs, "b t view d -> (B b) (T t M view) d", B = B, T = T, M = 2)
                cross_feature = self.emb_norm( cross_feature + mod_embs + view_embs)
            else:
                cross_feature = self.emb_norm( cross_feature + mod_embs)
        
        # decision backbone
        causal_mask = get_causal_mask(cross_feature)
        # original padding is used to choose unpadded one, reverse it.
        policy_padding_mask = ~new_padding_mask 
        # temporal transformer policy
        out  = self.policy(
                x = cross_feature,
                causal_mask = causal_mask, padding_mask = policy_padding_mask)
        if self.cross_atten2 is not None:
            # M is the number of modalities (2)
            features = einops.rearrange( out, 'B (T M N) d -> B T M N d',  M = 2, N = views)
            # select visual features instead of language
            vision_features = features[:,:,1]
            lang_features = features[:,:,0]
        else:
            vision_features = einops.rearrange( out, 'B (T N) d -> B T N d',  N = views)
            lang_features = None
        B,T,views,d = vision_features.shape

        # FiLM generator, used to generate params to modify Unet-decoding
        if self.film is not None:
            scales, biases = self.get_film(vision_features)
        
        # Prepare U-net decoder's input 
        residuals.reverse()
        decode_feat0 = unet_last_feature 
        decode_feat0 = einops.rearrange(decode_feat0, '(b t n) ch h w -> b t n ch h w', b = B, t = T, n = views)
        
        # unet decoding with Film's scaling and biasing
        if self.film is not None:
            decode_features = self.unet_decode_film(decode_feat0, residuals, scales, biases, padding_mask_vision)

        else:
            decode_features = self.unet_decode_crossatten(views, decode_feat0, vision_features, residuals, padding_mask_vision)
            decode_features = decode_features[padding_mask_vision]
        # prediction and rotations and position separately
        predictions = self.backend(
            views, pcd, lang_features, vision_features, decode_features, padding_mask_vision, instructions)

        return predictions
    
    def get_film(self, vision_features):
        """get scales and biases from FiLM layers based on visual input"""
        scales, biases = self.film(vision_features)
        # from dim: views x layers x batch x horizons x channel
        # to dim:   layers x batch x horizons x views x channel
        scales = einops.rearrange(scales, 'v L b h d -> L b h v d')
        biases = einops.rearrange(biases, 'v L b h d -> L b h v d')
        return scales, biases
    
    def unet_decode_film(self, decode_feat, residuals, scales, biases, padding_mask_vision):
        """During decoding, the feature map of unet decoder is affected by the 
        multi-view FiLM generater. Particularly, the residuals from encoder 
        during decoding is affected by FiLM' generated scale and bias
        
        Arguments
        ----------
        decode_feat:
            the unet feature from unet encoder that is need to be decoded by unet decoder
        residuals:
            the residuals (feature map) from unet encoder 
        scales:
            the channel-wise scaling of the residuals
        biases:
            the channel-wise biasing of the residuals
        padding_mask_vision:
            if true, it means the timestep is not padded, false is padded.
            Use to select unpad visual feature
        """
        b, t, n, ch, h, w =  decode_feat.shape
        decode_feat = einops.rearrange(
                decode_feat[padding_mask_vision],
                'unpad n ch h w -> (unpad n) ch h w')

        for l, unet_decode_layer in enumerate(self.trans_decoder):
            residual = residuals[l]
            scale = einops.rearrange( scales[l][padding_mask_vision], 'unpad view dim -> (unpad view) dim' )
            bias = einops.rearrange( biases[l][padding_mask_vision],  'unpad view dim -> (unpad view) dim')
            if l == 0:
                # directly modifying the input froom unet-encoder without residuals
                decode_feat = scale[:,:, None,None]*decode_feat + bias[:,:, None,None] 
            else:
                residual = einops.rearrange(
                    residual, 
                    "(b t n) c h w -> b t n c h w", b=b, t=t, n=n)[padding_mask_vision]
                residual = einops.rearrange(residual, "pad_batch n c h w -> (pad_batch n) c h w")
                # residual is modified by film's scale and bias
                residual = scale[:,:, None,None]*residual + bias[:,:, None,None] 
                decode_feat = torch.cat([decode_feat, residual  ], dim =1)
            decode_feat = unet_decode_layer(decode_feat)
        return einops.rearrange( decode_feat, "(pad_b n) ch h w -> pad_b n ch h w", n = n)
    
    def unet_decode_crossatten(self, views, decode_feat, vision_features, residuals, padding_mask_vision):
        """During decoding, the feature map of unet decoder cross-attend
        to view-specific visual features from transformer to include history.
        
        Arguments
        ----------
        views:
            number of views == cameras
        decode_feat:
            the unet feature from unet encoder that is need to be decoded by unet decoder
        vision_features:
            the feature learned by the temporal transformer that are associated with visual tokens
        residuals:
            the residuals (feature map) from unet encoder 
        padding_mask_vision:
            if true, it means the timestep is not padded, false is padded.
            Use to select unpad visual feature
        """
        b, t, n, c, h, w = decode_feat.shape
  
        for l, unet_decode_layer in enumerate(self.trans_decoder):
            decode_feat = self.unet_cross(l, views, decode_feat, None, vision_features, padding_mask_vision )
            if l != 0:
                residual = residuals[l]
                decode_feat = torch.cat([decode_feat, residual  ], dim =1)

            decode_feat = unet_decode_layer(decode_feat)
 
            decode_feat = einops.rearrange(decode_feat, '(b t n) c h w -> b t n c h w', b = b, t = t, n = n)
            
        return decode_feat

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p)

        