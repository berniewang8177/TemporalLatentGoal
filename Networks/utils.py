# basic utils


# RLbench if needed


# deep learning stuff
import torch
import torch.nn as nn

##############  Others #################
def get_present_features(
    features, 
    padding_mask, 
    views
    ):
    """
    We get the present features by removing history and padding
    Since each view is represented as 1 vector, present feature
    is "view" number of vectors.

    Arguments
    ---------
    features: torch.tensor
        padded features learned by transformer
    padding_mask:torch.tensor
        mask indicates padding
    views: int
        number of cameras we have
    """
    padding_mask =padding_mask.long().detach().cpu().numpy()
    present_features = []
    present_features_idx = []
    for i in range(padding_mask.shape[0]):
        high = padding_mask[i].nonzero()[0].min() 
        low = high-views
        present_features.append(features[i:i+1, low:high,:])
        present_features_idx.append( low + i*padding_mask.shape[0])
    return torch.cat(present_features, dim = 0)
##############  Embedding #################

class DiscreteEmbedding(nn.Module):
    """This class provides discrete embedding space to support:
    timestep embedding, patch embedding, modalities embedding, camera embedding.
    
    Arguments
    ---------
    size: int
        The size of embedding table.
    d_model: int
        The dim of expected features.
    """

    def __init__(self, size, d_model,):
        super().__init__()
        self.emb = nn.Embedding(size, d_model)
        self.d_model = d_model

    def forward(self, x):
        """ Processes the input tensor x and returns an output tensor."""
        return self.emb(x.long()) 


############## Masking ###################
# https://github.com/speechbrain/speechbrain/tree/develop/speechbrain/lobes/models/transformer/Transformer.py

def get_causal_mask(padded_input, seq_len = None):
    """Creates a binary mask for autoregressive prediction.
    Arguments
    ---------
    padded_input: torch.Tensor
        Padded input tensor.
    Example
    -------
    tensor([[0., -inf, -inf],
            [0., 0., -inf],
            [0., 0., 0.]])
    """
    if seq_len is None:
        seq_len = padded_input.shape[1]
    mask = (
        torch.triu(torch.ones((seq_len, seq_len), device=padded_input.device))
        == 1
    ).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask.detach().to(padded_input.device)

def get_padding_mask(padded_input, pad_idx):
    """Creates a binary mask to prevent attention to padded locations.
    It is modified to adapt to both vision and language input
    Arguments
    ----------
    padded_input: int
        Padded input.
    pad_idx:
        idx for padding element.
    Example
    -------
    >>> a = torch.LongTensor([[1,1,0], [2,3,0], [4,5,0]])
    >>> get_key_padding_mask(a, pad_idx=0)
    tensor([[False, False,  True],
            [False, False,  True],
            [False, False,  True]])
    """
    if len(padded_input.shape) == 4:
        bz, time, ch1, ch2 = padded_input.shape
        padded_input = padded_input.reshape(bz, time, ch1 * ch2)

    key_padded_mask = padded_input.eq(pad_idx).to(padded_input.device)

    # if the input is more than 2d, mask the locations where they are silence
    # across all channels
    if len(padded_input.shape) > 2:
        key_padded_mask = key_padded_mask.float().prod(dim=-1).bool()
        return key_padded_mask.detach()

    return key_padded_mask.detach()