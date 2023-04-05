# basic utils
import tap
from typing import List, Tuple, Optional
from pathlib import Path
import wandb
import os
# os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:2"
# os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
# RLbench


# deep learning stuff
import torch
# import torch._dynamo as dynamo
# dynamo.config.verbose=True
# dynamo.config.suppress_errors = True
# torch.use_deterministic_algorithms(True, warn_only=True)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
import torch.nn as nn
# for debugging
torch.autograd.set_detect_anomaly(True)
# project-specific
from Networks.agent2 import Agent
# train
from Train.arguments2 import Arguments
from Utils.utils import set_seed, count_parameters
from Utils.utils import get_data_loader
from Train.training import training
from Utils.utils import LossAndMetrics


if __name__ == "__main__":

    args = Arguments().parse_args()

    # creat the dataset path
    args.variations = args.variations.split()
    args.val_variations = args.val_variations.split()
    args.tasks = args.tasks.split()
    if args.name == 'LAVA' and args.lang_emb != 'CLIP':
        assert False, f"LAVA should use CLIP, not wave2vec"
    # creating wandb setup
    project_name = "Latent_Goal-oracle-multitasking"
    group_name = args.tasks[0] + '-variation-' + str(args.variations)
    name = args.lang_emb + "_" + args.name + "_pos_offset_" + str(args.position_offset) + f"_Oracle_{args.oracle_goal}" + f"_Film_" + "lr_" + str(args.lr)

    # log for wandb
    logger = dict()
    if args.log_to_wandb:
        wandb.init(
            name=name,
            group=group_name,
            project= project_name,
            config=args
        )

    # fix training seed
    set_seed(args.seed)

    # agent
    agent = Agent(args)
    print( f"In total, {count_parameters(agent.model)} M # of params")
    # load model if needed
    if args.load_model:
        load_name = os.path.join(args.save_path, args.load_name)
        agent.model = torch.load(load_name, map_location=torch.device(args.device))
        print("Load model sucess")
    params, optimizer = agent.get_model()
    
    # get metric
    loss_and_metrics = LossAndMetrics()

    # get data loaders (train and validation)
    loader = get_data_loader(args)

    # training mode set
    agent.model.train()

    training(
        args = args, 
        agent = agent, 
        optimizer = optimizer, 
        loader = loader, 
        metrics = loss_and_metrics)

    # DONE!


    



    