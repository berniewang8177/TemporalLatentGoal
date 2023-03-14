# basic utils
import einops
# deep learning stuff
import torch
import torch.nn as nn
# project-specific
from Networks.transformer.transformer import CrossrEncoder
from Networks.utils import get_causal_mask

class VALA(nn.Module):
    """
    This is an implementation of Vision-temporally-Attends-to-Language-to-Act (VALA).
    VALA use temporal visual queries attends to language to learn temporal latent goal.

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
        It is used for VALA (Vision-Attends-Language-to-Act) to learn temporal features before cross-attn.
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
        causal_self = False,
        lang_emb = 512
    ):
        super().__init__()

        # clip has emb dim 512
        lang_dim  = 512 if lang_emb == 'CLIP' else 300
        if lang_dim != d_model:
            self.proj_instr_encoder = nn.Linear(lang_dim, d_model)
        else:
            self.proj_instr_encoder = None

        self.vala = CrossrEncoder(
            num_layers,
            nhead,
            d_ffn,
            d_model=d_model,
            kdim=None,
            vdim=None,
            dropout=0.0,
            activation=activation,
            normalize_before=normalize_before,
            causal_self = causal_self
        )
        self.causal_self = causal_self
        self.causal_mask = None
    
    def forward(self, vision, vision_padding, language, language_padding):
        """
        Learned some temporal latent goals by querying language instruction with visual query.
        
        Arguments
        ----------
        vision:
            visual inputs supports keys,values for cross-modal attention.
        vision_padding:
            a multi-view padding mask to avoid attention on padded multi-view vision tokens.
        language:
            language supports queries for cross-modal attention.
        language_padding:
            a padding mask to avoid attention on padded language tokens.
        """
        B,T,views,d = vision.shape
        tgt = vision
        # make sure visual input have 3 dim (Batch x N) x (T) x dim 
        # where N is number of cameras
        assert len(tgt.shape) == 4
        tgt = einops.rearrange(tgt, "B T N dim -> (B N) T dim")
        inflate_vision_padding = einops.repeat(vision_padding, "B T -> (B N) T", N = views)

        ref = language
        B,tokens,dim = ref.shape
        # language vector have 3 dim Batch x T x dim where T = number of bpe tokens
        assert len(ref.shape) == 3
        if self.proj_instr_encoder is not None:
            ref = einops.repeat(ref, 'B pad_tokens dim -> (B N pad_tokens) dim', N = views)
            ref = self.proj_instr_encoder(ref)
            ref = einops.rearrange(
                ref, 
                '(B N pad_token) dim -> (B N) pad_token dim', B = B, N = views, pad_token = tokens)
        else:
            ref = einops.repeat(ref, 'B pad_tokens dim -> (B N) pad_tokens dim', N = views)
        inflate_language_padding = einops.repeat(language_padding, 'B tokens -> (B N) tokens', N = views )
        
        # only used for vision-attends-language
        if ref.requires_grad:
            # during training, horizon is fix
            if self.causal_self and self.causal_mask == None:
                length = vision_padding.shape[1] 
                self.causal_mask = get_causal_mask(tgt, length)
        else:
            # during evaluation, causal mask needs to be re-made since horizon isn't fixed
            length = vision_padding.shape[1] 
            self.causal_mask = get_causal_mask(tgt, length)
        # we apply self-attention with cauasl mask on multi-view visual tokens
        # to learn temporal visual features. Collapse Batch and view dim for efficent training.
        temporal_features, self_attns_map, cross_attns_map = self.vala(
            tgt,
            ref,
            tgt_mask=self.causal_mask,
            ref_mask=None,
            tgt_key_padding_mask=None,
            ref_key_padding_mask= inflate_language_padding) 
 
        # the final step
        # return the feature and also an inflated vision padding mask W.R.T the # of cameras
        return temporal_features, inflate_vision_padding