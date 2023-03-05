# basic utils
import einops
from einops.layers.torch import Rearrange
from typing import Tuple, Literal, Union, List
# deep learning stuff
import torch
import torch.nn as nn
# project-specific
# components
from .networks import ConvLayer
from .utils import DiscreteEmbedding as emb

class PredictionHead(nn.Module):
    """
    This network predicts actions = [gripper_position, rotation, gripper_state]

    Arguments
    ----------
    views:
        number of cameras
    max_horizons:
        max number timesteps per demo
    hidden:
        hidden dim
    position_offset:
        if True, we use an lstm to predict the 3d position offset out of point cloud.
    lang_offset:
        if True, we use the language features to predict offset
    """

    def __init__(
        self, 
        views,
        max_horizons,
        hidden,
        position_offset,
        lang_offset = False,
        offset_emb = False,
    ):
        super().__init__()
        self.pred_rot_stat = RotationStateNet(views, hidden)
        self.pred_position = PositionNet(views, hidden, max_horizons, position_offset, offset_emb)
        self.lang_offset = lang_offset
    def forward(self, 
        N, 
        pcd, 
        lang_features,
        vision_features,
        unet_feature, 
        padding_mask_vision, 
        instructions,
        ):
        """
        predicts actions by first predicting gripper rotation and state
        Lastly, predicting gripper position using different features
        Arguments
        ----------
        N:
            is this number of channel or cameras?
        pcd:
            point cloud observation.
        lang_features:
            language features learned by transformer
        vision_features:
            vision features learned by transformer
        unet_features:
            features learned by transformer and decoded by Unet-decoder
        padding_mask_vision:
            padding mask used by visial observation
        instructions:
            language features 
        """
        rotation, gripper_state = self.pred_rot_stat(vision_features, padding_mask_vision)
        features = lang_features if self.lang_offset else vision_features
        position = self.pred_position(features, unet_feature, pcd, instructions, N, padding_mask_vision)

        return {"rotation":rotation,'gripper':gripper_state, 'position':position }

class RotationStateNet(nn.Module):
    """
    This network predicts gripper rotation and gripper_state

    Arguments
    ----------
    views:
        number of cameras
    
    hidden:
        hidden dim
    """

    def __init__(
        self, 
        views,
        hidden,
    ):
        super().__init__()
        self.views = views
        self.view_reduce = torch.nn.Conv1d(
                in_channels = views, 
                out_channels = 1, 
                kernel_size = 1 )
        self.view_emb = emb(views, hidden)
        self.view_norm = nn.LayerNorm(hidden, eps=1e-6)

        half_hidden = hidden // 2
        self.layers = nn.Sequential(
            nn.Linear(hidden,half_hidden ),
            nn.ReLU(),
            nn.Linear(half_hidden, half_hidden ),
            nn.ReLU(),
        )
        self.pred_head = nn.Linear(half_hidden, 5)
        
    
    def normalise_quat(self, x: torch.Tensor):
        return x / x.square().sum(dim=-1).sqrt().unsqueeze(-1)
    
    def forward(self, x, padding):
        # add view emb
        views = torch.arange(0, self.views).long().to(x.device)
        view_embeddings = self.view_emb(views)
        x += self.view_norm(view_embeddings)
        x = x[padding]
        # dim: unpad_timesteps x views x dim 
        x = self.view_reduce(x)
 
        x = einops.rearrange(x, "unpad_t reduced_view  dim -> (unpad_t reduced_view)  dim")
        x = self.layers(x)
        x = self.pred_head(x)

        rotation = x[:, :-1]
        rotation = self.normalise_quat(rotation)
        gripper_state = torch.sigmoid(x[:, -1:])
        return rotation, gripper_state


class PositionNet(nn.Module):
    """
    This network predicts gripper position = [coordinates, offset]

    Arguments
    ----------
    views:
        number of cameras we have.
    hidden:
        dim of hidden space
    max_horizons:
        max length of a demo
    position_offset:
        if True, we use an lstm to predict the 3d position offset out of point cloud
    offset_emb:
        if True, we will add muti-view /time embeddings to features 
        before predicting position offset.
    """

    def __init__(
        self, 
        views,
        hidden,
        max_horizons,
        position_offset,
        offset_emb = False
    ):
        super().__init__()
        self.views = views
        # for 3d coordiantes
        self.maps_to_coord = ConvLayer(
            in_channels=16,
            out_channels=1,
            kernel_size=(1, 1),
            stride_size=(1, 1),
            apply_norm=False,
            apply_activation=False,
        )
        self.offset_emb = offset_emb
        self.position_offset = position_offset
        if position_offset:
            # for 3d offset prediction
            self.view_reduce = torch.nn.Conv1d(
                    in_channels = views+1 if self.offset_emb else views,
                    out_channels = 1, 
                    kernel_size = 1 )
            if self.offset_emb:
                self.view_emb = emb(views, hidden)
                self.view_norm = nn.LayerNorm(hidden, eps=1e-6)
                self.time_emb = emb(max_horizons, hidden)
                self.time_norm = nn.LayerNorm(hidden, eps=1e-6)
            self.lstm = nn.LSTM(
                input_size = hidden, 
                hidden_size = hidden, 
                num_layers = 3, 
                batch_first = True)
            half_hidden = hidden // 2
            self.offset = nn.Sequential(
                nn.Linear(hidden,half_hidden ),
                nn.ReLU(),
                nn.Linear(half_hidden, half_hidden ),
                nn.ReLU(),
                nn.Linear(half_hidden, 3)
            )
    
    def forward(self, features, unet_feature, pcd, instructions, N, padding_mask_vision):
        """
        This network predicts gripper rotation and gripper_state

        Arguments
        ----------
        features:
            transformer output features
        unet_feature:
            Unet encoder features decoded by Unet-decoder
        pcd:
            point cloud obervation
        N:
            number of views == cameras
        instructions:
            instructions features
        padding_mask_vision:
            padding mask to mask visial paddings
        
        """
        # predicting coordinates
        pcd = pcd[padding_mask_vision]
        
        unet_feature = einops.rearrange(unet_feature, 'pad_b n ch h w -> (pad_b n) ch h w')
        xt = self.maps_to_coord(unet_feature)
        xt = einops.rearrange(xt, "(b n) ch h w -> b (n ch h w)", n=N, ch=1)

        xt = torch.softmax(xt / 0.1, dim=1)
        attn_map = einops.rearrange(
            xt, "b (n ch h w) -> b n ch h w", n=N, ch=1, h=128, w=128
        )
        coordinates = einops.reduce(pcd * attn_map, "b n ch h w -> b ch", "sum")

        offsets = 0
        if self.position_offset:
            # predicting offset
            B,T,ch,dim = features.shape
            if self.offset_emb:
                # add view emb
                views = torch.arange(0, self.views).long().to(features.device)
                view_embeddings = self.view_emb(views)
                features += self.view_norm(view_embeddings)
                # cat with timestep emb
                times = torch.arange(0, T).long().to(features.device)
                # dim: T x dim
                time_embeddings = self.time_emb(times)
                # dim: B x T x 1 x dim (create a dummy dim to cat with multi-view dim)
                time_embeddings = time_embeddings[:, None, :].repeat(B,1,1,1)
                features = torch.cat([features, time_embeddings], dim = 2)
            features = einops.rearrange(features, 'B T ch dim -> (B T) ch dim')
            reduced_features = self.view_reduce(features)
            # get read for lstm and removes dummy channel ch after views reduction
            reduced_features = einops.rearrange(reduced_features, '(B T) ch dim -> B T (ch dim)', B = B, T = T)
            lstm_features, _ = self.lstm(reduced_features)
            offsets = self.offset(lstm_features[padding_mask_vision])

        return offsets + coordinates
