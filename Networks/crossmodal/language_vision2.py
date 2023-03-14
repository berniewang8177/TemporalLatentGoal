# basic utils
import einops
# deep learning stuff
import torch
import torch.nn as nn
# project-specific
from Networks.utils import DiscreteEmbedding as emb
from Networks.transformer.transformer import CrossrEncoder
from Networks.utils import get_causal_mask

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
        max_horizons = 6
    ):
        super().__init__()

        # clip has emb dim 512
        lang_dim = 512 # 
        self.causal_mask = None
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

        self.time_emb = emb(max_horizons, d_model)
        self.time_norm = nn.LayerNorm(d_model)

    def forward(self, 
        vision, 
        vision_padding, 
        language,
        language_padding = None,
        views = 3):
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
        views:
            number of views (cameras)
        """

        ref = vision
        # make sure visual input have 3 dim (B x views) x T x dim 
        assert len(ref.shape) == 3
        b_views, T, dim = ref.shape
        # reverse map since now it is used by transformer
        inflate_vision_padding = ~einops.repeat(vision_padding, 'B tokens -> (B views) tokens', views = views )

        tgt = language
        # make sure language input have 3 dim (B x views) x T x dim 
        assert len(ref.shape) == 3
        timesteps = torch.arange(T).type_as(ref).unsqueeze(0).long()
        time_embeddings = self.time_emb(timesteps)
        time_embeddings = self.time_norm(time_embeddings)
        time_embeddings =  einops.repeat(time_embeddings, "dummy t d -> (dummy b_views) t d", b_views = b_views)
        tgt = tgt + time_embeddings
        # only used for vision-attends-language
        if ref.requires_grad:
            # during training, horizon is fix
            if self.causal_mask == None:
                length = vision_padding.shape[1] 
                self.causal_mask = get_causal_mask(tgt, length)

        else:
            # during evaluation, causal mask needs to be re-made since horizon isn't fixed
            length = vision_padding.shape[1] 
            self.causal_mask = get_causal_mask(tgt, length)

        # we do cross-attn between a language vector and batchxviewxhorizon visual tokens
        features, self_attns_map, cross_attns_map = self.lava(
            tgt,
            ref,
            tgt_mask=self.causal_mask,  # causal self-attn
            ref_mask=None,  # no padding since horizon dim has been part of Batch dim
            tgt_key_padding_mask=None,
            ref_key_padding_mask=inflate_vision_padding) 
        
        return features
