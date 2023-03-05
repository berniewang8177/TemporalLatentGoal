# basic utils
from typing import Optional
# deep learning stuff
import torch
import torch.nn as nn
# project-specific
# components
from .multiway import MultiwayNetwork
from .components import MultiheadAttention
from .components import PositionalwiseFeedForward

class TransformerEncoderLayer(nn.Module):
    """This is an implementation of self-attention encoder layer.
    Arguments
    ----------
    d_ffn: int, optional
        The dimension of the feedforward network model hidden layer.
    nhead: int
        The number of heads in the multi-head attention models (default=8).
    d_model: int
        The number of expected features in the encoder/decoder inputs (default=512).
    kdim: int, optional
        Dimension of the key.
    vdim: int, optional
        Dimension of the value.
    dropout: int, optional
        The dropout value.
    activation: torch.nn.Module, optional
        The activation function for Feed-Forward Netowrk layer,
        e.g., relu or gelu or swish.
    normalize_before: bool, optional
        Whether normalization should be applied before or after MHA or FFN in Transformer layers.
        Defaults to True as this was shown to lead to better performance and training stability.
    expert_counts: int, optional
        number of parallal modality experts we need per layer. 
        Default = 1, which is a standard uni-way transformer. > 1, then it becomes multiway
    """

    def __init__(
        self,
        d_ffn,
        nhead,
        d_model,
        kdim=None,
        vdim=None,
        dropout=0.0,
        activation=nn.ReLU,
        normalize_before=True,
        expert_counts = 1,
        views = 3,
    ):
        super().__init__()
        
        self.self_att = MultiheadAttention(
            nhead=nhead,
            d_model=d_model,
            dropout=dropout,
            kdim=kdim,
            vdim=vdim,
        )
        self.expert_counts = expert_counts
        self.experts = MultiwayNetwork(
            expert_counts,
            d_model=d_model,
            d_ffn=d_ffn,
            dropout=dropout,
            activation=activation,
            views = views,
        )

        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.normalize_before = normalize_before

    def forward(
        self,
        src,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        Arguments
        ----------
        src : torch.Tensor
            The sequence to the encoder layer.
        src_mask : torch.Tensor
            The mask for the src query for each example in the batch.
            This is a causal mask in this project because we're a temporal transformer.
        src_key_padding_mask : torch.Tensor, optional
            The mask for the src keys for each example in the batch.
        """

        if self.normalize_before:
            src1 = self.norm1(src)
        else:
            src1 = src

        output, self_attn = self.self_att(
            src1,
            src1,
            src1,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )

        # add & norm
        src = src + self.dropout1(output)
        if not self.normalize_before:
            src = self.norm1(src)

        if self.normalize_before:
            src1 = self.norm2(src)
        else:
            src1 = src

        output = self.experts(src1)

        # add & norm
        output = src + self.dropout2(output)
        if not self.normalize_before:
            output = self.norm2(output)
        return output, self_attn


class TransformerEncoder(nn.Module):
    """This class implements the transformer encoder.
    Arguments
    ---------
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
        Dropout for the encoder (Optional).]
    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> net = TransformerEncoder(1, 8, 512, d_model=512)
    >>> output, _ = net(x)
    >>> output.shape
    torch.Size([8, 60, 512])
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
        expert_counts = 1,
        views = 3,
    ):

        super().__init__()

        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_ffn=d_ffn,
                    nhead=nhead,
                    d_model=d_model,
                    kdim=kdim,
                    vdim=vdim,
                    dropout=dropout,
                    activation=activation,
                    normalize_before=normalize_before,
                    expert_counts = expert_counts,
                    views = views
                )
                for i in range(num_layers)
            ]
        )
        
    def forward(
        self,
        src,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos_embs: Optional[torch.Tensor] = None,
    ):
        """
        Arguments
        ----------
        src : tensor
            The sequence to the encoder layer (required).
        src_mask : tensor
            The mask for the src sequence (optional).
            It is a causal mask in our case because we're using a temporal transformer.
        src_key_padding_mask : tensor
            The mask for the src keys per batch (optional).
        """
        output = src
        attention_lst = []
        for i, enc_layer in enumerate(self.layers):
            
            output, attention = enc_layer(
                output,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
            )

            attention_lst.append(attention)

        output = self.norm(output)
        return output, attention_lst

class CrossModalLayer(nn.Module):
    """This is an implementation of cross-modal encoder layer.
    It could be LAVA (Language-Attend-Vison-to-Act) where a query comes from language
    It could be LAV (Location-aware-Vison) where a query comes from location.
    It could be VALA (Vision-Attend-Language-to-Act) where queries comes from vision.
    Arguments
    ----------
    d_ffn: int, optional
        The dimension of the feedforward network model hidden layer.
    nhead: int
        The number of heads in the multi-head attention models (default=8).
    d_model: int
        The number of expected features in the encoder/decoder inputs (default=512).
    kdim: int, optional
        Dimension of the key.
    vdim: int, optional
        Dimension of the value.
    dropout: int, optional
        The dropout value.
    activation: torch.nn.Module, optional
        The activation function for Feed-Forward Netowrk layer,
        e.g., relu or gelu or swish.
    normalize_before: bool, optional
        Whether normalization should be applied before or after MHA or FFN in Transformer layers.
        Defaults to True as this was shown to lead to better performance and training stability.
    causal_self: bool, optional
        If true, apply a causal self-attn before cross-attn like a vanilla transformer decoder
        It is used for VALA (Vision-Attends-Language-to-Act) to learn temporal features before cross-attn.
    """

    def __init__(
        self,
        d_ffn,
        nhead,
        d_model,
        kdim=None,
        vdim=None,
        dropout=0.0,
        activation=nn.ReLU,
        normalize_before=True,
        causal_self = False,
    ):
        super().__init__()
        self.causal_self = causal_self
        if causal_self:
            # used by causal self-atten for VALA (causal vision attends language)
            self.self_att = MultiheadAttention(
                nhead=nhead,
                d_model=d_model,
                dropout=dropout,
                kdim=kdim,
                vdim=vdim,
            )
            self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
            self.dropout1 = nn.Dropout(dropout)

        self.cross_att = MultiheadAttention(
            nhead=nhead,
            d_model=d_model,
            dropout=dropout,
            kdim=kdim,
            vdim=vdim,
        )
        self.pos_ffn = PositionalwiseFeedForward(
            d_ffn=d_ffn,
            input_size=d_model,
            dropout=dropout,
            activation=activation,
            )
        
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm3 = nn.LayerNorm(d_model, eps=1e-6)
        
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.normalize_before = normalize_before

    def forward(
        self,
        tgt,
        ref,
        tgt_mask=None,
        ref_mask=None,
        tgt_key_padding_mask=None,
        ref_key_padding_mask=None,
    ):
        """transformer forward pass.
        For Location-aware Vision/Language-Attends-Vision, no causal self-attn. Cross-attn only.
        For Vision-Attends-Language, causal self-attn and then cross-attend
        Arguments
        ----------
        tgt : torch.Tensor
            Apply causal self-attn on tgt if needed.
            Queries used by cross-attn comes from this.
        ref : torch.Tensor
            The reference used by cross-attn (analogous to encoded state from "encoder")
            Keys and values used by cross-attn comes from this.
        tgt_mask : torch.Tensor
            Used if we need a causal self-attn on tgt before applying cross attn
        ref_mask : torch.Tensor, optional
            Not used. None type
        tgt_key_padding_mask: tensor
            The mask for the tgt keys per batch (optional).
        ref_key_padding_mask: tensor
            The mask for the ref keys per batch (optional).
        """
        # we only apply causal self-attn if we're doing Vision-Attends-Language
        self_attn = None
        if self.causal_self:
            if self.normalize_before:
                tgt1 = self.norm1(tgt)
            else:
                tgt1 = tgt

            # self-attention over the target sequence
            tgt2, self_attn = self.self_att(
                query=tgt1,
                key=tgt1,
                value=tgt1,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                pos_embs=None, # not used
            )

            # add & norm
            tgt = tgt + self.dropout1(tgt2)
            if not self.normalize_before:
                tgt = self.norm1(tgt)

        # cross-attn between tgt and ref
        # for VALA: tgt is vision and ref is language
        # for Location-aware-Vision: tgt is location, ref is vision
        # for LAVA: tgt is langauge and ref is vision

        if self.normalize_before:
            tgt1 = self.norm2(tgt)
        else:
            tgt1 = tgt

        # multi-head attention over the target sequence and encoder states
        tgt2, cross_attention = self.cross_att(
            query=tgt1,
            key=ref,
            value=ref,
            attn_mask=ref_mask,
            key_padding_mask=ref_key_padding_mask,
            pos_embs= None,
        )

        # add & norm
        tgt = tgt + self.dropout2(tgt2)
        if not self.normalize_before:
            tgt = self.norm2(tgt)

        if self.normalize_before:
            tgt1 = self.norm3(tgt)
        else:
            tgt1 = tgt

        tgt2 = self.pos_ffn(tgt1)

        # add & norm
        tgt = tgt + self.dropout3(tgt2)
        if not self.normalize_before:
            tgt = self.norm3(tgt)

        return tgt, self_attn, cross_attention

class CrossrEncoder(nn.Module):
    """This class implements the cross-modal encoder where
    query VS. key-value comes from different modalities.
    Arguments
    ---------
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
    -------
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
        self.causal_self = causal_self
      
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

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
        
    def forward(
        self,
        tgt,
        ref,
        tgt_mask=None,
        ref_mask=None,
        tgt_key_padding_mask=None,
        ref_key_padding_mask=None,
    ):
        """
        Arguments
        ----------
        tgt: tensor
            The input of the layer that gives queries
        ref: tensor
            The input of the layer that gives keys and values
        tgt_mask : tensor
            The mask for the tgt sequence (optional).
            Useful during causal self-attention
        ref_mask : tensor
            The mask used for the ref sequence (optional).
            Not used
        tgt_key_padding_mask : tensor
            The mask used for the tgt keys per batch (optional).
            Useful during self-attention
        ref_key_padding_mask : tensor
            The mask used for the ref keys per batch (optional).
        """
        output = tgt
        self_attns, cross_attns = [], []
        for dec_layer in self.layers:
            output, self_attn, cross_attn = dec_layer(
                output,
                ref,
                tgt_mask=tgt_mask,
                ref_mask=ref_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                ref_key_padding_mask=ref_key_padding_mask,
            )
            self_attns.append(self_attn)
            cross_attns.append(cross_attn)
        output = self.norm(output)

        return output, self_attns, cross_attns