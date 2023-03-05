import random
from typing import Tuple, Optional
from copy import deepcopy
from pathlib import Path
import torch
import numpy as np
import tap
from filelock import FileLock
import os
# project-specific
from Test.arguments import Arguments
from Networks.agent import Agent
# from Preprocess.data_utils import RLBenchEnv
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
        data_path= None,
        apply_rgb=True,
        apply_pc=True,
        headless=args.headless,
        apply_cameras=("left_shoulder", "right_shoulder", "wrist"),
    )

    # load instruction including: lang_feat, eos_feat, lang_pad, lang_num
    instructions = load_instructions(
        args,
        args.dataset_val[0],
        args.var_num,) # it should be int)
    
    max_eps_dict = load_episodes(args.episodes_json_path)["max_episode_length"]

    for task_str in args.tasks:
        for variation in args.variations:
            success_rate = env.evaluate(
                task_str = args.tasks,
                max_episodes=max_eps_dict[task_str],
                variation=variation,
                num_demos=args.num_episodes,
                demos=None,
                offset=args.offset,
                agent=agent,
                instructions = instructions,
                log_dir=log_dir / task_str if args.save_img else None,
                max_tries=args.max_tries,
                save_attn=args.attention,
            )

            print("Testing Success Rate {}: {:.04f}".format(task_str, success_rate))
