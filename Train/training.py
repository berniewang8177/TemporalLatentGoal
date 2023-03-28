# basic utils
from tqdm import tqdm, trange
import wandb
import gc
import copy
import os
from PIL import Image
import numpy as np
import einops
# deep learning stuff
import torch
# project-specific
from Train.validation import validating

def check_training(batch_rgbs):
    B,T,view, channel, H,W = batch_rgbs.shape
    # only take 1 sample, 1 view, remove attention map at channel dim
    _rgbs = batch_rgbs[0,:,0,:3,:,:] * 255

    _rgbs = _rgbs.astype(np.uint8)
    for i, rgb in enumerate(_rgbs):
        rgb = einops.rearrange(rgb, "channel H W -> H W channel")
        rgb_img = Image.fromarray(rgb )
        img_path = f'./{i}.jpg'
        rgb_img.save(img_path)

def save_model(args, path, train_idx, val_idx, model):
    """save the model
    
    Arguments
    ----------
    path:   
        path to save the model
    val_idx:
        the variation index during validations
    model: 
        save its parameters 
    """
    best_model = copy.deepcopy(model)
    if args.no_film:
        film = 'no_film'
        if args.modality_fusion == False:
            film = 'film'
    else:
        film = 'film_once' if args.film_once else 'film_layer_wise'
        if args.cross_decode:
            film = 'cross_decode'
    name = f'{args.lang_emb}_{args.name}_train_{train_idx}_variation_{val_idx}_{film}.pth'
    final_path = os.path.join(path, name)
    if os.path.exists(final_path):
        print("Delete one!", end = '\t')
        os.remove(final_path)
    print("Saved one!")
    torch.save(best_model, final_path)

def training(
    args,
    agent,
    optimizer,
    loader,
    checkpointer = None,
    metrics = None,
    logger = None
):
    """ Train a behavior cloning policy by:
        1. get inputs (rgb, point cloud, instr) and target action (gripper)
        2. model forward 
        3. prediction
        Arguments
        ----------
        args:   
            A dictionary of configuration/hyper-param.
        agent:  
            contains a policy needs to be trained.
        optimizer:
            optimizer for policy
        loader:
            a tuple of (train_loader, val_loader)
        checkpointer
            checkpoint policy if val has improvement.
        metrics:
            compute performance and loss
        logger:
            a dictionary of performance scores. Used by wandb
        ------- 
    """
    train_loader, val_loaders = loader

    iter_loader = iter(train_loader)
    device = agent.device
    val_interval = args.train_iters // args.val_number 
    best_loss = 1000
    with trange(args.train_iters) as tbar:
        for step_id in tbar:
            try:
                sample = next(iter_loader)
            except StopIteration:
                iter_loader = iter(train_loader)
                sample = next(iter_loader)
            agent.model.train()
            # fetch samples rgb, point cloud, gripper(not used?), outputs
            rgbs = sample["rgbs"].to(device)
            # check_training(sample['clean_rgbs'].cpu().numpy())
            # assert False, f"{rgbs.shape}"
            pcds = sample["pcds"].to(device)
            gripper = sample["gripper"].to(device)
            outputs = sample["action"].to(device)
            padding_mask = sample["padding_mask"].to(device)
            variation_nums = sample['variation'].tolist()

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

            if step_id % args.accumulate_grad_batches == 0:
                optimizer.zero_grad()

            # forward
            pred = agent.model(
                rgbs,
                pcds,
                padding_mask,
                instr,
                instr_mask ,
                variation = variation_nums
            )
            # loss compute
            train_losses = metrics.compute_loss(pred, sample)
            
            train_losses["total"] = sum(list(train_losses.values()))  # type: ignore
            train_losses["total"].backward()
            # backward
            if step_id % args.accumulate_grad_batches == args.accumulate_grad_batches - 1:
                # grandient clip
                torch.nn.utils.clip_grad_norm_(agent.model.parameters(), .25)
                optimizer.step()
            
            # move this inside the if state before
            if agent.scheduler is not None:
                agent.scheduler.step()

            del sample
            del pred
            gc.collect()
            torch.cuda.empty_cache()

            if step_id % val_interval == val_interval -1:
                with torch.no_grad():
                    agent.model.eval()
                    # loop through a dictionary of val loaders
                    for val_idx in val_loaders:
                        val_loader = val_loaders[val_idx]
                        val_loss, val_logs = validating(agent, val_idx, val_loader, metrics)
                        if args.save_model and val_loss < best_loss:
                            best_losss = val_loss
                            save_model(args, args.save_path, args.variations[0], val_idx, agent.model)
                        if args.log_to_wandb:
                            wandb.log(val_logs)
            if args.log_to_wandb:
                train_log_losses = { "train " + k :train_losses[k].detach().cpu().numpy() for k in train_losses}
                train_log_losses['lr'] = agent.scheduler.get_last_lr()[0]
                # training log
                wandb.log(train_log_losses)
    

