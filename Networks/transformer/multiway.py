# basic utils
import einops
# deep learning stuff
import torch
import torch.nn as nn
# project-specific
# components
from .components import PositionalwiseFeedForward


#reference: https://github.com/microsoft/torchscale/blob/main/torchscale/component/multiway_network.py
# I modify the original usages so that now it creates multiple copies of FFNs


class MultiwayNetwork(nn.Module):
    """This is an implementation of self-attention encoder layer.
    Arguments
    ----------
    expert_counts: int,
        number of parallal FFNs we need per layer.
        Usually, expert_counts == number of modalities we have within the input.
    d_model: int
        The number of expected features in the encoder/decoder inputs (default=512).
    dd_ffn: int
        The number of expected features in the FFNs
    kdim: int, optional
        Dimension of the key.
    dropout: int, optional
        The dropout value.
    activation: torch.nn.Module, optional
        The activation function for Feed-Forward Netowrk layer,
        e.g., relu or gelu or swish.
    normalize_before: bool, optional
        Whether normalization should be applied before or after MHA or FFN in Transformer layers.
        Defaults to True as this was shown to lead to better performance and training stability.
    -------
    """
    def __init__(self,
        expert_counts,
        d_model,
        d_ffn,
        dropout,
        activation,
        views = 3,
        ):
        super().__init__()

        self.d_model = d_model
        self.d_ffn = d_ffn
        self.expert_counts = expert_counts
        if self.expert_counts == 2:
            self.views = views
        else:
            # every experts takes a view and also a modality if self.expert_counts = 6
            self.views = 1
        # creates many parallal modality experts for each modality
        expert_list = [
            PositionalwiseFeedForward(
                d_ffn=d_ffn,
                input_size=d_model,
                dropout=dropout,
                activation=activation,
                )
            for  idx in range(self.expert_counts) 
        ]
        self.experts = nn.ModuleList(expert_list)


    def forward(self, x):
        """forward pass of modality experts.
        For a multimodal seqeuntial inputs,
        1. decompose into a list of uni-modal sequences
        2. each uni-modal sequence is sent to its own modality expert (FFN)
        3. outputs are gathered and restores to the orignal input dim

        Arguments
        ----------
        x: torch sequence,
            a multimodal sequence
        """
        batch_size, length, hidden_size = x.shape
        # batch x (T*experts*views) x dim --> batch x experts x T x v x dim
        x = einops.rearrange(
            x, 
            "B (T experts V) d -> B experts (T V) d",
            experts = self.expert_counts, V=self.views
            )
        # modality chunk is a list of tensors. 
        # Each tensor is a uni-modal tensor: batch x 1 x T x dim
        modality_chunk = x.chunk( self.expert_counts, dim = 1)
        outputs = []
        for i, expert in enumerate(self.experts):
            # apply each modality expert to each chunk of data W.R.T a modality
            tmp = expert(  modality_chunk[i].squeeze(1 )).unsqueeze(1)
            outputs.append(tmp)
        outputs = torch.cat(outputs, dim = 1)
        
        outputs = einops.rearrange(
            outputs, 
            "B experts (T V) d -> B (T experts V) d",
            experts = self.expert_counts, V=self.views
            )
        return outputs