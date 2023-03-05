# basic utils
import einops
# deep learning stuff
import torch
import torch.nn as nn
# project-specific
from Networks.transformer.transformer import CrossModalLayer
from Networks.utils import get_causal_mask

class UnetCrossAtten(nn.Module):
    """
    When Unet decoder is decoding and upsampling, depth-wise feature on the feature map
    will attends to historical-language-driven features from transformer.
    

    Arguments
    ----------
    num_layers : int
        Number of transformer layers to include.
    nhead : int
        Number of attention heads.
    d_ffn : int
        Hidden size of self-attention Feed Forward layer.
    d_model : int
        The dimension of the input embedding.
    kdim : int
        Dimension for key (Optional).
    vdim : int
        Dimension for value (Optional).
    dropout : float
        Dropout for the encoder (Optional).
    causal_self: bool, optional
        If true, apply a causal self-attn before cross-attn like a vanilla transformer decoder
        Not used in Unet cross attention
    """

    def __init__(
        self, 
        num_layers,
        nhead,
        d_ffn,
        d_model=None,
        kdim=None,
        vdim=None,
        dropout=0.0,
        activation=nn.ReLU,
        normalize_before=False,
        causal_self = False
    ):
        super().__init__()
        self.vison_project = nn.Linear(16, d_model)
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [
                CrossModalLayer(
                    d_ffn=d_ffn,
                    nhead=nhead,
                    d_model=d_model,
                    kdim=kdim,
                    vdim=vdim,
                    dropout=dropout,
                    activation=activation,
                    normalize_before=normalize_before,
                    causal_self = causal_self,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, 
        layer,
        views,
        tgt, 
        multi_view_attn_mask, 
        ref,
        ref_padding = None):
        """
        Learned a new feature map for Unet by querying transformer features.
        
        Arguments
        ----------
        layer:
            the layer index we're working with
        tgt:
            depth-wise feature from Unet-decoder feature map
        multi_view_attn_mask:
            not used
        ref:
            multi-view features from transformer.
        ref_padding:
            not used.
        """

        # Batch x padded_horizon x views x channel x height x width
        b, t, n, ch, h, w = tgt.shape
        tgt = einops.rearrange(tgt, 'b t n ch h w ->  (b n) (t h w) ch')
        if layer == 0:
            tgt = self.vison_project(tgt)
        
        # inflate batch dimension of padding mask
        ref = einops.rearrange(ref, 'b t n dim -> (b n) t dim' )
        key_padding_mask = ~einops.repeat(ref_padding, 'b t -> (b n) t', n = n )
        # inflate the target sequence len L to include Unet feature map size 8x8
        causal_mask = einops.repeat( 
            get_causal_mask(ref), 
            'L S -> (L h w) S', h = h, w = w )
        features, self_attns_map, cross_attns_map = self.layers[layer](
            tgt,
            ref,
            tgt_mask=causal_mask,  
            ref_mask=None,  
            tgt_key_padding_mask=None,
            ref_key_padding_mask=key_padding_mask) 
        # (batch x views) x padded_horizon x height x width x channel
        # -> batch x padded_horizon x views x channels x h x w 
        features = einops.rearrange(
            features, 
            '(b n) (t h w) ch -> (b t n) ch h w', 
            b = b, n = n, t = t, h = h, w = w)
        
        return features
