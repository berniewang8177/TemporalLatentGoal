# basic utils
import einops
# deep learning stuff
import torch
import torch.nn as nn
# project-specific
from Networks.transformer.transformer import CrossrEncoder

class LAVA(nn.Module):
    """
    This is an re-implementation of Language-Attends-to-Vision-to-Act (LAVA).
    LAVA use language queries attends to vision for learning a new observation space.
    
    Ref: Interactive Language: Talking to Robots in Real Time

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
    ):
        super().__init__()

        # clip has emb dim 512
        lang_dim = 512 # 
        self.proj_instr_encoder = nn.Linear(lang_dim, d_model)

        self.lava = CrossrEncoder(
            num_layers,
            nhead,
            d_ffn,
            d_model=d_model,
            kdim=None,
            vdim=None,
            dropout=dropout,
            activation=activation,
            normalize_before=normalize_before,
            causal_self = causal_self
        )

    def forward(self, 
        vision, 
        vision_padding, 
        language,
        language_padding = None):
        """
        Learned a new observation space by querying visual observation with langauge query.
        
        Arguments
        ----------
        vision:
            visual inputs supports keys,values for cross-modal attention.
        vision_padding:
            a padding mask to avoid attention on padded visual tokens.
            Not used.
        language:
            language supports queries for cross-modal attention.
        language_padding:
            not used.
        """

        ref = vision
        # make sure visual input have 5 dim B x T x cameras x (number of tokens) x dim 
        assert len(ref.shape) == 5
        B,T,N,tokens,dim = ref.shape
        # group (batch and views) for efficient attention
        ref = einops.rearrange(ref, 'b t n tokens dim -> (b t n) tokens dim')

        tgt = language
        # reduce language from clip dim to model dim
        tgt = self.proj_instr_encoder(tgt)
        tgt = einops.repeat(tgt, 'B d -> (B T N) d', T = T, N = N)
        tgt = tgt[:,None,:]
        # tgt has been inflated, so does the padding mask (not used by cross-attn transformer)
        multi_view_mask = vision_padding.repeat_interleave(N, dim = 1)

        # we do cross-attn between a language vector and batchxviewxhorizon visual tokens
        features, self_attns_map, cross_attns_map = self.lava(
            tgt,
            ref,
            tgt_mask=None,  # no causal self-attn, so no need for this
            ref_mask=None,  # no padding since horizon dim has been part of Batch dim
            tgt_key_padding_mask=None,
            ref_key_padding_mask=None) 
        # (B T views) dummy d --> B (T views) d
        lava_features = einops.rearrange(
            features, '(B T N) dummy dim -> B (T N dummy) dim', B = B, T = T, N = N )
        
        return lava_features, multi_view_mask 
