import random
from time import time
from typing import Tuple, Optional
from copy import deepcopy
from pathlib import Path
from PIL import Image
import torch
import numpy as np
import tap
import os
import einops
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
# deep learning stuff
import torch
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
# for debugging
torch.autograd.set_detect_anomaly(True)

# project-specific
from Test.arguments import Arguments
from Networks.agent2 import Agent
from Preprocess.data_utils import RLBenchEnv
from Utils.utils import (
    set_seed,
    load_episodes,
    get_max_episode_length,
    load_instructions)
# we don't have actioner like HiveFormer but choose the Agent class instead 

if __name__ == "__main__":
    args = Arguments().parse_args()
    args.tasks = args.tasks.split() # create a list of task
    # fix training seed
    set_seed(args.seed)

    agent = Agent(args)
    # load model
    if args.load_model:
        load_name = os.path.join(args.save_path, args.load_name)
        agent.model = torch.load(load_name, map_location=torch.device(args.device))
        print("Load model sucess")
    else:
        assert False, f"please load the model for the evaluation !"
    
    # load RLBench environment
    env = RLBenchEnv(
        data_path= args.dataset_val[0], # place holder, we don't actually need it 
        apply_rgb=True,
        apply_pc=True,
        headless=args.headless,
        apply_cameras= args.cameras,
    )

    # load instruction including: lang_feat, eos_feat, lang_pad, lang_num
    instructions = load_instructions(
        args,
        args.dataset_val[0],
        args.var_num,) # it should be int)
    
    max_eps_dict = load_episodes(args.episodes_json_path)["max_episode_length"]
    
    for task_str in args.tasks:

        success_rate, sucess_rgbs_episode, failed_rgbs_episode = env.evaluate(
            task_str = args.tasks[0],
            max_episodes=max_eps_dict[task_str],
            variation= args.var_num,
            num_demos=args.num_episodes,
            agent=agent,
            instructions = instructions,
        )

        print("Testing Success Rate {}: {:.04f}".format(task_str, success_rate))
        print( "Sucess trials", len(sucess_rgbs_episode), "  failed trials: ", len(failed_rgbs_episode)  )
        # store what we have
        for idx, rgbs in enumerate( sucess_rgbs_episode ):
            rgbs = (rgbs[:,0,:,:,:] ).astype(np.uint8)
            for i, rgb in enumerate(rgbs):
                rgb_img = Image.fromarray(rgb)
                name = args.load_name.replace(".pth", "")
                suffix = str(time())[:2]
                img_path = f'/home/ubuntu/workspace/imgs/sucess-{idx}-{name}-{i}-{suffix}.jpg'
                rgb_img.save(img_path)
        for idx, rgbs in enumerate( failed_rgbs_episode ):
            rgbs = (rgbs[:,0,:,:,:] ).astype(np.uint8)
            for i, rgb in enumerate(rgbs):
                rgb_img = Image.fromarray(rgb)
                name = args.load_name.replace(".pth", "")
                suffix = str(time())[:2]
                img_path = f'/home/ubuntu/workspace/imgs/failed-{idx}-{name}-{i}-{suffix}.jpg'
                rgb_img.save(img_path)
            break
        print('Done')
