# basic utils


# RLbench if needed


# deep learning stuff
import torch
import torch.nn as nn
import torch.optim as optim

# project-specific
# architectures
from .frontend import HiveFormerVisionFrontend
from .crossmodal.language_vision import LAVA 
from .crossmodal.vision_language import VALA
from .crossmodal.unet_cross_atten import UnetCrossAtten
from .models import TemporalTransformer, Models
from .networks import MultiViewFiLM
from .backend import PredictionHead
# masking
from .utils import get_causal_mask, get_padding_mask


class Agent:
    """
    An agent maps observations to actions, by following language instruction.

    Arguments
    ----------
    args:
        it has all hyper-param we need
    """

    def __init__(
        self, 
        args
    ):
        super().__init__()

        self.device = args.device
        
        # Decision-making pipeline
        self.frontend = HiveFormerVisionFrontend(
            feat_layers = args.depth,
            d_model = args.hidden_dim,
            cameras = len(args.cameras),
            max_horizons = args.max_episode_length,
            max_patches = 64).to(self.device)

        self.cross_atten = args.name
        
        # google's Language-Attends-Vision-to-Act
        self.lang_vision = LAVA(
            num_layers = args.cross_layers,
            nhead = 1,
            d_ffn =  args.hidden_dim * 4,
            d_model=  args.hidden_dim,
            kdim=None,
            vdim=None,
            dropout=0.0,
            activation=nn.ReLU,
            normalize_before=False,
            causal_self = False, # True when using VALA
            ).to(self.device)
        self.vision_lang = None
        if self.cross_atten == 'VALA':
            # use ours Vision-temporally-Attends-Language-to-Act
            self.vision_lang = VALA(
                num_layers = args.cross_layers,
                nhead = 1,
                d_ffn = args.hidden_dim * 4,
                d_model= args.hidden_dim,
                kdim=None,
                vdim=None,
                dropout=0.0,
                activation=nn.ReLU,
                normalize_before=False,
                causal_self = True, # True when using VALA
                lang_emb = args.lang_emb # determine dimension to reduce
            ).to(self.device)
        # temporla transformer policy, generate multi-vew conditioning vectors
        self.policy = TemporalTransformer( 
            num_layers = args.policy_layers,
            nhead = 1,
            d_ffn = args.hidden_dim * 4,
            d_model= args.hidden_dim,
            kdim=None,
            vdim=None,
            dropout=0.0,
            activation=nn.ReLU,
            normalize_before=True,
            expert_counts = args.expert_counts,
            cameras = len(args.cameras),
            ).to(self.device)
        
        # modify unet decoding by FiLM or cross-attentino
        if args.cross_decode:
            self.film = None
            self.unet_cross = UnetCrossAtten(
                    num_layers = args.depth,
                    nhead = 1,
                    d_ffn = args.hidden_dim * 4,
                    d_model= args.hidden_dim,
                    kdim=None,
                    vdim=None,
                    dropout=0.0,
                    activation=nn.ReLU,
                    normalize_before=False,
                    causal_self = False # True when using VALA
                ).to(self.device)
        else:
            self.unet_cross = None
            self.film =MultiViewFiLM(
            hidden = args.hidden_dim, 
            channel = 16,
            views = len(args.cameras),
            depth = args.depth,
            film_once = args.film_once,
            film_first = args.film_first
            ).to(self.device)

        self.backend = PredictionHead(
            views = len(args.cameras),
            max_horizons = args.max_episode_length,
            hidden = args.hidden_dim,
            position_offset = args.position_offset,
            lang_offset = args.lang_offset,
            offset_emb = args.offset_emb
            ).to(self.device)

        self.model = Models(
            frontend = self.frontend,
            cross_atten1 = self.lang_vision,
            cross_atten2 = self.vision_lang,
            policy = self.policy,
            no_film = args.no_film,
            film = self.film,
            unet_cross = self.unet_cross,
            backend = self.backend,
            depth = args.depth,
            ).to(self.device)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr = args.lr)

    def get_model(self):
        """
        Used for checkpointing. 
        
        Returns model.state_dict and optmizer
        """
        return self.model.state_dict(), self.optimizer
    
    def act(self, step_idx, rgbs, pcds, instructions, lang_mask):
        """
        Used for evaluation. Takes observations, output 1 action.
        
        Arguments
        ----------
        step_idx:
            current step id, uesd to calcualte horizon length
        rgbs:
            rgb images
        pcds:
            point clouds
        instructions:
            a tuple contains token and eos features
        lang_mask:
            a padding mask for language (true is padded token)
            

        """
        device = rgbs.device
        horizon_len = step_idx+1
        # Note that true denotes unpad tokens
        visual_mask = torch.tensor([True] * horizon_len).unsqueeze(0).to(device)
        pred = self.model(rgbs, pcds, visual_mask, instructions, lang_mask)

        return pred
