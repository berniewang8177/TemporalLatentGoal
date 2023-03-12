# basic utils
import tap
from typing import List, Tuple, Optional
from pathlib import Path
import wandb
import os
# os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:2"
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
# RLbench


# deep learning stuff
import torch
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
# for debugging
torch.autograd.set_detect_anomaly(True)
# project-specific
from Networks.agent import Agent
# train
from Train.arguments import Arguments
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
    project_name = "Latent_Goal-seen-validation"
    group_name = args.tasks[0] + '-variation-' + str(args.variations)
    if args.position_offset:
        name = args.name + "-lang_offset" if args.lang_offset else args.name + "-vision_offset"
        name = name + "_emb" if args.offset_emb else name + "_no_emb"
    else:
        name = args.name + "-no_offset"
    name += f'-{args.expert_counts}-experts'
    if args.cross_decode:
        name += '-cross_decode'
    else:
        if args.film_first:
            name = name + '-film_first_res_intact' if args.film_once else name + '-film_layer_wise'
        else:
            name = name + '-film_all_res_intact' if args.film_once else name + '-film_layer_wise'
    name = args.lang_emb + "-" + name

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


    



    