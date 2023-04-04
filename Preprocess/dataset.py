# basic utils
import glob
import os
import itertools
from typing import Dict, List
import json
import einops
import numpy as np
import pickle
# RLbench if needed
# deep learning stuff
import torch
# project-specific
from .data_utils import RLBenchEnv, get_observation 
from Utils.utils import get_attn_indices_from_demo

class Dataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.low_dim = args.low_dim
        # load RLBench environment
        self.env = RLBenchEnv(
            data_path=args.data_dir,
            apply_rgb=True,
            apply_pc=True,
            apply_cameras=args.cameras,
        )

        with open("/home/ubuntu/workspace/TemporalLatentGoal/Preprocess/episodes.json") as fid:
            episodes = json.load(fid)
        self.max_eps_dict = episodes["max_episode_length"]
        self.variable_lengths = set(episodes["variable_length"])

        for task_str in self.args.tasks:
            if task_str in self.max_eps_dict:
                continue
            _, state_ls, _ = get_observation(task_str, self.args.offset, 0, self.env)
            self.max_eps_dict[task_str] = len(state_ls) - 1
            raise ValueError(
                f"Guessing that the size of {task_str} is {len(state_ls) - 1}"
            )

        broken = set(episodes["broken"])
        tasks = [t for t in self.args.tasks if t not in broken]
        # instead of process all, process specific variations
        # variations = range(self.args.offset, self.args.max_variations)
        variations = self.args.specific_vars
        self.items = []
        for task_str, variation in itertools.product(tasks, variations):
            episodes_dir = self.args.data_dir + '/' + task_str + '/' + f"variation{variation}" + '/' + "episodes"
            # get abs path of each episode dir, retrieve the episode index
            episodes = [ (task_str, variation, int(name.split('/')[-1][7:])) for name in glob.glob(episodes_dir + '/episode*') ]
            self.items += episodes

        self.num_items = len(self.items)

    def __len__(self) -> int:
        return self.num_items

    def __getitem__(self, index: int) -> None:
        task, variation, episode = self.items[index]
        taskvar_dir = self.args.output + '/' + f"{task}+{variation}"
        if os.path.exists(taskvar_dir) == False:
            os.mkdir(taskvar_dir)
        try:
            demo, state_ls, action_ls = get_observation(
                task, self.low_dim, variation, episode, self.env
            )
        except (FileNotFoundError, RuntimeError, IndexError) as e:
            print(e)
            return
        if self.low_dim == False:
            state_ls = einops.rearrange(
                state_ls,
                "t 1 (m n ch) h w -> t n m ch h w",
                ch=3,
                n=len(self.args.cameras),
                m=2,
            )
        else:
            assert False, f"{ len(state_ls) }"

        frame_ids = list(range(len(state_ls) - 1))
        num_frames = len(frame_ids)
        attn_indices = get_attn_indices_from_demo(task, demo, self.args.cameras)

        if (task in self.variable_lengths and num_frames > self.max_eps_dict[task]) or (
            task not in self.variable_lengths and num_frames != self.max_eps_dict[task]
        ):
            print(f"ERROR ({task}, {variation}, {episode})")
            print(f"\t {len(frame_ids)} != {self.max_eps_dict[task]}")
            return

        state_dict: List = [[] for _ in range(5)]
        print("Demo {}".format(episode))

        horizon = len(frame_ids)
        state_dict[0].extend(frame_ids)
        state_dict[1].extend(state_ls[:-1])
        state_dict[2].extend(action_ls[1:])
        # make sure number of attn_indices match number of state_dict
        if len(attn_indices) > horizon:
            attn_indices = attn_indices[:horizon]
        if self.low_dim == False:
            state_dict[3].extend(attn_indices) 
            state_dict[4].extend(action_ls[:-1])  # gripper pos
        print("Success !")
        print("Gonna save at", taskvar_dir + '/' + f"low_dim_ep{episode}.npy")
        try:
            # np.save(taskvar_dir + '/' + f"ep{episode}.npy", state_dict)  # type: ignore
            with open(f"{taskvar_dir}/ep{episode}.pkl", 'wb') as f:
                pickle.dump(state_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        except Exception as e: 
            print(e)
            
            print("---"*5,'\n')
            print("problem:", taskvar_dir + '/' + f"ep{episode}.npy")
            for idx in range(5):
                print("Check len:", len( state_dict[idx] ),  np.array(state_dict[idx][0]).shape  )
                if len(np.array(state_dict[idx][0]).shape ) < 2:
                    print(state_dict[idx])
            print("---"*5, '\n')
            assert False