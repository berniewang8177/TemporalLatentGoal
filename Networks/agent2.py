# basic utils


# RLbench if needed


# deep learning stuff
import torch
import torch.nn as nn
import torch.optim as optim

# project-specific
# architectures
from .frontend import UnetFrontend
from .crossmodal.language_vision2 import LAVA 
from .crossmodal.vision_language2 import VALA
from .models2 import TemporalTransformer, Models
from .backend2 import PredictionHead
from .networks import AdditiveFusion, OracleGoal
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
        self.frontend = UnetFrontend(
            feat_layers = args.depth,
            d_model = args.hidden_dim,
            cameras = len(args.cameras),
            max_horizons = args.max_episode_length).to(self.device)

        self.cross_atten = args.name
        
        # google's Language-Attends-Vision-to-Act
        self.lang_vision = LAVA(
            num_layers = args.cross_layers * 2,
            nhead = 1,
            d_ffn =  args.dim_feedforward * 4,
            d_model=  args.dim_feedforward,
            kdim=None,
            vdim=None,
            dropout=0.0,
            activation=nn.ReLU,
            normalize_before=False,
            causal_self = False, # True when using VALA
            max_horizons = args.max_episode_length,
            ).to(self.device)
        self.vision_lang = None
        self.policy = None
        if self.cross_atten == 'VALA':
            if args.oracle_goal:
                self.goal_emb = OracleGoal(32)

            else:
                self.goal_emb = None
                # use ours Vision-temporally-Attends-Language-to-Act
                self.vision_lang = VALA(
                    num_layers = args.cross_layers,
                    nhead = 1,
                    d_ffn = args.dim_feedforward * 4,
                    d_model= args.dim_feedforward,
                    kdim=None,
                    vdim=None,
                    dropout=0.0,
                    activation=nn.ReLU,
                    normalize_before=False,
                    causal_self = True, # True when using VALA
                    lang_emb = args.lang_emb # determine dimension to reduce
                ).to(self.device)
            if args.modality_fusion:
                self.fusion = AdditiveFusion(
                    d_ffn = args.dim_feedforward*2,
                    input_size = args.dim_feedforward )
            else:
                self.fusion = None
            # temporla transformer policy, only used by VALA
            self.policy = TemporalTransformer( 
                num_layers = args.policy_layers,
                nhead = 1,
                d_ffn = args.dim_feedforward * 4,
                d_model= args.dim_feedforward,
                kdim=None,
                vdim=None,
                dropout=0.0,
                activation=nn.ReLU,
                normalize_before=True,
                expert_counts = args.expert_counts,
                cameras = len(args.cameras),
                ).to(self.device)
            
        self.backend = PredictionHead(
            views = len(args.cameras),
            max_horizons = args.max_episode_length,
            hidden = 512,
            position_offset = args.position_offset,
            lang_offset = args.lang_offset,
            offset_emb = args.offset_emb
            ).to(self.device)

        self.model =  Models(
            frontend = self.frontend,
            cross_atten1 = self.lang_vision,
            cross_atten2 = self.vision_lang,
            policy = self.policy,
            backend = self.backend,
            depth = args.depth,
            fusion = self.fusion,
            goal_emb = self.goal_emb
            ).to(self.device)
        
        self.optimizer = optim.AdamW( self.model.parameters(), lr = args.lr)
        # self.optimizer = Lion(self.model.parameters(), lr = args.lr)
        lr_ratio = 0.25
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda steps: min((steps+1)/args.warmup, 1) if steps < args.warmup else lr_ratio + (1-lr_ratio) * ( 0.9970 ** (steps-args.warmup) )
        )
        # self.scheduler = optim.lr_scheduler.LambdaLR(
        #     self.optimizer,
        #     lambda steps: min((steps+1)/args.warmup, 1) 
        #     )

    def get_model(self):
        """
        Used for checkpointing. 
        
        Returns model.state_dict and optmizer
        """
        return self.model.state_dict(), self.optimizer
    
    def act(self, step_idx, rgbs, pcds, instructions, lang_mask, variation = -1):
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
        variation:
            the variation of the test task
        """
        device = rgbs.device
        horizon_len = step_idx+1
        # Note that true denotes unpad tokens
        visual_mask = torch.tensor([True] * horizon_len).unsqueeze(0).to(device)
        variation = [ variation ]

        pred = self.model(rgbs, pcds, visual_mask, instructions, lang_mask, variation)

        return pred
