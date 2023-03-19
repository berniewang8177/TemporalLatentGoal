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
            film_first,
            hidden,
            channel,
            depth,
        ):
        """
        Ggenerate scales and biases but for all Unet-decoder at once
        
        Arguments
        ----------
        film_first:
            whether modify 1st feature map or feature map of all layers
        hidden:
            the dimension of input features
        channel:
            the channel which scaling and bias wants to modify during decoding
        depth:
            depth of the Unet decoder we want to influence
        """
        super().__init__()
        self.film_first = film_first
        self.hidden = hidden
        self.depth = depth
        new_hidden = self.depth * channel * 2
        # new_hidden = 1 * channel * 2 if film_first else self.depth * channel * 2

        self.layers = nn.Sequential(
            nn.Linear(hidden, new_hidden ),
            nn.ReLU(),
            nn.Linear(new_hidden, new_hidden ),
            nn.ReLU(),
        )
        
        if film_first:
            self.scale = nn.Linear(new_hidden, channel)
            self.bias = nn.Linear(new_hidden,  channel)
        else:
            self.scale = nn.Linear(new_hidden, self.depth * channel)
            self.bias = nn.Linear(new_hidden, self.depth * channel)

    def forward(self, x):
        """predicts channel wise scaling and bias for all layers at once"""
        B,T,dim = x.shape
        x1 = einops.rearrange(x, 'b t dim -> (b t) dim')

        x2 = self.layers(x1 )
        scales = self.scale(x2)
        biases = self.bias(x2)
    
        d = 1 if self.film_first else self.depth
        scales_reshape  = einops.rearrange(
            scales, 
            '(b t) (depth channel) -> depth b t channel', b = B, t = T, depth = d )  
        biases_reshape = einops.rearrange(
            biases,
            '(b t) (depth channel) -> depth b t channel', b = B, t = T, depth = d) 

        return scales_reshape, biases_reshape

class MultiViewFiLM(nn.Module):
    def __init__(
            self,
            hidden,
            channel,
            views,
            depth,
            film_once,
            film_first,
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
        film_first:
            whether modify 1st feature map or feature map of all layers
        """
        super().__init__()
        self.hidden = hidden
        self.films = nn.ModuleList()
        self.film_once = film_once
        self.film_first = film_first
        for v in range(views):
            if film_once:
                self.films.append(FiLMGeneratorOnce(film_first, hidden, channel, depth))
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

class AdditiveFusion(nn.Module):
    """The class implements and additive fusion
    Arguments
    ----------
    d_ffn: int
        Hidden layer size.
    input_shape : tuple, optional
        Expected shape of the input. Alternatively use ``input_size``.
    input_size : int, optional
        Expected size of the input. Alternatively use ``input_shape``.
    dropout: float, optional
        Dropout rate.
    activation: torch.nn.Module, optional
        activation functions to be applied (Recommendation: ReLU, GELU).
    """

    def __init__(
        self,
        d_ffn,
        input_shape=None,
        input_size=None,
        dropout=0.0,
        activation=nn.ReLU,
    ):
        super().__init__()

        if input_shape is None and input_size is None:
            raise ValueError("Expected one of input_shape or input_size")

        if input_size is None:
            input_size = input_shape[-1]

        self.linear1 = nn.Linear(input_size, d_ffn)
        self.linear2 = nn.Linear(input_size, d_ffn)
        self.activation = activation()
        self.linear3 = nn.Linear(d_ffn, input_size)

    def forward(self, x, y):
        """Applies Additive fusion to the input tensor x and y"""
        # give a tensor of shap (time, batch, fea)
        x = x.permute(1, 0, 2)
        y = y.permute(1, 0, 2)
        x_new = self.linear1(x)
        y_new = self.linear2(y)
        fused = self.activation( x_new + y_new )
        project_fused = self.linear3(fused)

        # reshape the output back to (batch, time, fea)
        final = project_fused.permute(1, 0, 2)

        return final