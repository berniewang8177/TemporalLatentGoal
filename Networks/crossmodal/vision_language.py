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
        self.proj_instr_encoder = nn.Linear(lang_dim, d_model)
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
    
    def forward(self, vision, vision_padding, language, language_padding, N):
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
        N:
            number of views (cameras)
        """

        tgt = vision
        # make sure visual input have 4 dim Batch x (TxN) x dim 
        # where N is number of cameras
        assert len(tgt.shape) == 3
        B,_,dim = tgt.shape

        ref = language
        B,tokens,dim = ref.shape
        # make sure language vector have 3 dim Batch x T x dim 
        # where T = number of bpe tokens
        assert len(ref.shape) == 3
        ref = einops.rearrange(ref, 'B pad_tokens dim -> (B pad_tokens) dim')
        ref = self.proj_instr_encoder(ref)
        ref = einops.rearrange(ref, '(B pad_token) dim -> B pad_token dim', B = B)

        # only used for vision-attends-language
        if ref.requires_grad:
            # during training, horizon is fix
            if self.causal_self and self.causal_mask == None:
                length = vision_padding.shape[1] // N
                self.causal_mask = get_causal_mask(tgt, length)
                # views within same timestep could attends to each other
                self.causal_mask = self.causal_mask.repeat_interleave(N,dim = 1).repeat_interleave(N,dim = 0)
        else:
            length = vision_padding.shape[1] // N
            self.causal_mask = get_causal_mask(tgt, length)
            # views within same timestep could attends to each other
            self.causal_mask = self.causal_mask.repeat_interleave(N,dim = 1).repeat_interleave(N,dim = 0)
        # we apply self-attention with cauasl mask on multi-view visual tokens
        # to learn temporal visual features. Collapse time and view dim for efficent training.
        temporal_features, self_attns_map, cross_attns_map = self.vala(
            tgt,
            ref,
            tgt_mask=self.causal_mask,
            ref_mask=None,
            tgt_key_padding_mask=None,
            ref_key_padding_mask=language_padding) 

        # the final step
        # return the feature and also an inflated vision padding mask W.R.T the # of cameras
        return temporal_features, vision_padding