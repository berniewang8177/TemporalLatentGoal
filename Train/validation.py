# basic utils
from tqdm import tqdm, trange
import wandb
import numpy as np
import gc
import torch

def validating(
    agent,
    val_idx,
    loader,
    metrics = None,
):
    """ validating a behavior cloning policy by:
        1. get inputs (rgb, point cloud, instr) and target action (gripper)
        2. model forward 
        3. prediction
        Arguments
        ----------
        agent:  
            contains a policy needs to be validated.
        val_idx:
            the index indicating the variation of the task
        loader:
            a loader for validation
        metrics:
            compute performance and loss
        ------- 
    """
    device = agent.device
    val_log_losses = None
    for s_id, sample in enumerate(loader):
        # fetch samples rgb, point cloud, gripper(not used?), outputs
        rgbs = sample["rgbs"].to(device)
        pcds = sample["pcds"].to(device)
        gripper = sample["gripper"].to(device)
        outputs = sample["action"].to(device)
        padding_mask = sample["padding_mask"].to(device)
        if sample["tokens"] is not None:
            tokens = sample["tokens"].to(device)
        else:
            tokens = None
        eoses = sample["eoses"].to(device)
        instr = (tokens, eoses)
        if sample["instr_mask"] is not None:
            instr_mask = sample["instr_mask"].to(device)
        else:
            instr_mask = None

        # forward
        pred = agent.model(
            rgbs,
            pcds,
            padding_mask,
            instr,
            instr_mask ,
            variation = val_idx
        )
        # loss compute
        val_losses = metrics.compute_loss(pred, sample)
        total = sum(list(val_losses.values()))  # type: ignore
        if val_log_losses is None:
            val_log_losses = { k: [val_losses[k].detach().cpu().numpy(),] for k in val_losses}
            val_log_losses['total'] = [total.detach().cpu().numpy(),]
        else:
            val_log_losses['total'].append(total.detach().cpu().numpy())
            for k in val_losses:
                val_log_losses[k].append( val_losses[k].detach().cpu().numpy() )
    
        del sample
        del pred
        gc.collect()
        torch.cuda.empty_cache()
    
    return_logs = { f"val-{val_idx}-" + k: np.mean(val_log_losses[k]) for k in val_log_losses}
    return return_logs[f"val-{val_idx}-total"], return_logs