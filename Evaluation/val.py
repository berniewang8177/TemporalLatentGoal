# basic utils
from tqdm import tqdm, trange

# RLbench


# deep learning stuff
import torch
import torch.nn as nn

# project-specific
from Utils import Checkpointer, LossAndMetrics
from Evaluation.val import validation
# some loss/metric?


def validation(
    args,
    agent,
    val_loaders = None,
    max_iters = 5,
    metrics = None,
    logger = None
):
    """ Validate an agent:
        1. get inputs (rgb, point cloud, instr) and target action (gripper)
        2. model forward 
        3. prediction
        Arguments
        ----------
        args:   
            A dictionary of configuration/hyper-param.
        agent:  
            contains a policy needs to be trained.
        val_loaders:
            validation loaders for different tasks
            feed data to policy with keyframes + demo augmentation
        max_iters:
            max number of validation interaction per task
        metrics:
            compute performance and loss
        logger:
            a dictionary of performance scores. Used by wandb
        ------- 
    """ 
    iter_loader = iter(val_loader)
    device = agent.device
    agent.model.eval()
    for val_id, val_loader in enumerate(val_loaders):
        for i, sample in enumerate(val_loader):
            if i == max_iters:
                break
           

            # fetch samples rgb, point cloud, gripper(not used?), outputs
            
            # forward

            # loss compute


    # log write (validation)
