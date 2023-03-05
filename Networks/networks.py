from typing import Tuple, Union
import einops
# deep learning stuff
import torch
import torch.nn as nn
# project-specific
# embeddings
from .utils import DiscreteEmbedding as emb

class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride_size,
        apply_norm=True,
        apply_activation=True,
        residual=False,
    ):
        super().__init__()
        self._residual = residual

        padding_size = (
            kernel_size // 2
            if isinstance(kernel_size, int)
            else (kernel_size[0] // 2, kernel_size[1] // 2)
        )

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride_size,
            padding_size,
            padding_mode="replicate",
        )

        if apply_norm:
            self.norm = nn.GroupNorm(1, out_channels, affine=True)

        if apply_activation:
            self.activation = nn.LeakyReLU(0.02)

    def forward(
        self, ft: torch.Tensor
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        out = self.conv(ft)
        res = out.clone()

        if hasattr(self, "norm"):
            out = self.norm(out)

        if hasattr(self, "activation"):
            out = self.activation(out)
            res = self.activation(res)

        if self._residual:
            return out, res
        else:
            return out

class FiLMGenerator(nn.Module):
    def __init__(
            self,
            hidden,
            channel,
            depth,
        ):
        """
        For a Bxdxv tensors where B is batch, d is dim, v is number of views,
        we reduce it to Bxd and output Film scaling and bias.
        
        Arguments
        ----------
        hidden:
            the dimension of input features
        channel:
            the channel which scaling and bias wants to modify during decoding
        depth:
            depth of the Unet decoder we want to influence
        """
        super().__init__()
        self.hidden = hidden
        half_hidden = hidden//2

        # self.view_reduce = torch.nn.Conv1d(
        #         in_channels = views, 
        #         out_channels = 1, 
        #         kernel_size = 1 )
        self.depth = depth
        self.layer_emb = emb(depth, hidden)
        self.norm = nn.LayerNorm(hidden, eps=1e-6)
        
        self.layers = nn.Sequential(
            nn.Linear(hidden,half_hidden ),
            nn.ReLU(),
            nn.Linear(half_hidden, half_hidden ),
            nn.ReLU(),
        )
        self.scale = nn.Linear(half_hidden, channel)
        self.bias = nn.Linear(half_hidden, channel)
    
    def forward(self, x):
        """Add layer emb and predicts channel wise scaling and bias"""
        B,T,dim = x.shape
        x = einops.rearrange(x, 'b t dim -> (b t) dim')

        scales = []
        biases = []
        for d in range(self.depth):
            features = self.layers( 
                self.norm(
                    x + self.layer_emb(torch.tensor(d).long().to(x.device) )
                ) 
            )
            scales.append( 
                einops.rearrange(self.scale(features),'(b t) dim -> b t dim', b = B, t = T)  )
            biases.append( 
                einops.rearrange(self.bias(features),'(b t) dim -> b t dim', b = B, t = T)  )

        return torch.stack(scales), torch.stack(biases)

class FiLMGeneratorOnce(nn.Module):
    def __init__(
            self,
            hidden,
            channel,
            depth,
        ):
        """
        Ggenerate scales and biases but for all Unet-decoder at once
        
        Arguments
        ----------
        hidden:
            the dimension of input features
        channel:
            the channel which scaling and bias wants to modify during decoding
        depth:
            depth of the Unet decoder we want to influence
        """
        super().__init__()
        self.hidden = hidden
        
        self.depth = depth
        new_hidden = self.depth * channel * 2

        self.layers = nn.Sequential(
            nn.Linear(hidden, new_hidden ),
            nn.ReLU(),
            nn.Linear(new_hidden, new_hidden ),
            nn.ReLU(),
        )
        self.scale = nn.Linear(new_hidden, self.depth * channel)
        self.bias = nn.Linear(new_hidden, self.depth * channel)

    
    def forward(self, x):
        """predicts channel wise scaling and bias for all layers at once"""
        B,T,dim = x.shape
        x1 = einops.rearrange(x, 'b t dim -> (b t) dim')

        x2 = self.layers(x1 )
        scales = self.scale(x2)
        biases = self.bias(x2)

        scales_reshape  = einops.rearrange(
            scales, 
            '(b t) (depth channel) -> depth b t channel', b = B, t = T, depth = self.depth )  
        biases_reshape = einops.rearrange(
            biases,
            '(b t) (depth channel) -> depth b t channel', b = B, t = T, depth = self.depth) 

        return scales_reshape, biases_reshape

class MultiViewFiLM(nn.Module):
    def __init__(
            self,
            hidden,
            channel,
            views,
            depth,
            film_once
        ):
        """
        For a Bxdxv tensors where B is batch, d is dim, v is number of views,
        we reduce it to Bxd and output Film scaling and bias.
        
        Arguments
        ----------
        hidden:
            the dimension of input features
        channel:
            the channel which scaling and bias wants to modify during decoding
        views:
            number of views to modify during Unet-decoding. Views == Cameras
        depth:
            depth of the Unet decoder we want to influence
        film_once:
            whether generate scales/biases for all unet decoder layers once
        """
        super().__init__()
        self.hidden = hidden
        self.films = nn.ModuleList()
        self.film_once = film_once
 
        for v in range(views):
            if film_once:
                self.films.append(FiLMGeneratorOnce(hidden, channel, depth))
            else:
                self.films.append(FiLMGenerator(hidden, channel, depth))
    def forward(self, z):
        """Reduce x dim and predicts scaling and bias for each view
        
        Arguments
        ----------
        z:
            a conditioning variable has dim: BxVxdim where
            B is batch size, V is number of views, dim is feature size
        """
        B, horizons, views,_ = z.shape
        # uni_view_features = self.view_reduce(x)
        scales = []
        biases = []
        z_views = z.chunk( views, dim = 2)
        for v in range(views):
            # z dim: B x T x view x dim
            z_view = einops.rearrange(z_views[v], 'B T dummy dim -> B T (dummy dim)')
            # the clone here is to avoid "views to be modified inplace" error
            scale, bias = self.films[v](z_view.clone())
            # scale and bias dim: layers x B x T x dim
            scales.append( scale ) 
            biases.append( bias )
        scales = torch.stack(scales )
        biases = torch.stack(biases)

        # scales/biases dim: views x layers x B x T x dim
        return scales, biases