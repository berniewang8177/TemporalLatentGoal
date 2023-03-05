# basic utils
import os
import numpy as np
import random
from typing import List, Dict, Tuple,  TypedDict, Union, Literal
import pickle
import json
import itertools

# RLbench 
# from rlbench.demo import Demo

# deep learning stuff
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data._utils.collate import default_collate
from torch.utils.data import DataLoader

# project-specific
from .dataset import RLBenchDataset

# Camera = Literal["wrist", "left_shoulder", "right_shoulder", "overhead", "front"]
# Instructions = Dict[str, Dict[int, torch.Tensor]]

class Demo(object):

    def __init__(self, observations, random_seed=None):
        self._observations = observations
        self.random_seed = random_seed

    def __len__(self):
        return len(self._observations)

    def __getitem__(self, i):
        return self._observations[i]

    def restore_state(self):
        np.random.set_state(self.random_seed)

############################# Task Meta #############################

def load_episodes(episodes_json_path) :
    with open(episodes_json_path) as fid:
        return json.load(fid)

def get_max_episode_length(tasks,  episodes_json_path) -> int:
    max_episode_length = 0
    max_eps_dict = load_episodes(episodes_json_path)["max_episode_length"]

    for task in tasks:
        if max_eps_dict[task] > max_episode_length:
            max_episode_length = max_eps_dict[task]

    return max_episode_length

############################# Instruction related #############################
def load_instructions(
    args,
    path,
    var_num,
    ):
    """
    load the language features and its padding mask

    Arguments:
    ----------
    args:
        has everything
    path: str
        it has the path to the task data (training or validation)
    var_num: int
        which variation should I retrieve
    """
    numbers = 0

    if path is not None:
        instru_path = os.path.join( path, args.tasks[0] + f'+{var_num}')
        print("get instruction path:", instru_path)
        if args.name == 'LAVA':
            eos_path = os.path.join(instru_path, 'language', 'eos_features.pkl')
            instructions_path = None
            padding_path = None
        else:
            # VALA case
            if args.lang_emb == 'CLIP':
                instructions_path = os.path.join(instru_path, 'language', 'token_features.pkl')
                padding_path = os.path.join(instru_path, 'language', 'padding_mask.pkl')
            else:
                instructions_path = os.path.join(instru_path, 'language', 'w2v_features.pkl')
                padding_path = os.path.join(instru_path, 'language', 'w2v_mask.pkl')
            eos_path = os.path.join(instru_path, 'language', 'eos_features.pkl')
            

        if instructions_path != None:
            with open(instructions_path, "rb") as fid:
                instr = pickle.load(fid).numpy()
                numbers = len(instr)
        else:
            instr = None

        if eos_path != None:
            with open(eos_path, "rb") as fid:
                eos = pickle.load(fid).numpy()
                numbers = len(eos)
        else:
            eos = None

        if padding_path != None:
            with open(padding_path, "rb") as fid:
                pad = pickle.load(fid).numpy()
        else:
            pad = None
        return_data = [instr, eos, pad, numbers]
        
        for i, data in enumerate(return_data):
            if data is None:
                return_data[i] = [None] * numbers

        return return_data

    return None, None, None, None
############################# Demo usage #############################
def obs_to_attn(obs, camera: str) -> Tuple[int, int]:
    """
    use extrinsic to transformer coordinates to camera coordiantes 
    and transformer to image coordiate with intrinsic 
    """
    extrinsics_44 = torch.from_numpy(obs.misc[f"{camera}_camera_extrinsics"]).float()
    extrinsics_44 = torch.linalg.inv(extrinsics_44)
    intrinsics_33 = torch.from_numpy(obs.misc[f"{camera}_camera_intrinsics"]).float()
    intrinsics_34 = F.pad(intrinsics_33, (0, 1, 0, 0))
    gripper_pos_3 = torch.from_numpy(obs.gripper_pose[:3]).float()
    gripper_pos_41 = F.pad(gripper_pos_3, (0, 1), value=1).unsqueeze(1)
    points_cam_41 = extrinsics_44 @ gripper_pos_41

    proj_31 = intrinsics_34 @ points_cam_41
    proj_3 = proj_31.float().squeeze(1)
    u = int((proj_3[0] / proj_3[2]).round())
    v = int((proj_3[1] / proj_3[2]).round())

    return u, v

def get_attn_indices_from_demo(
    task_str: str, demo: Demo, cameras: Tuple[str, ...]
) -> List[Dict[str, Tuple[int, int]]]:
    frames = keypoint_discovery(demo)

    # HACK tower3
    if task_str == "tower3":
        frames = [k for i, k in enumerate(frames) if i % 6 in set([1, 4])]

    # HACK tower4
    elif task_str == "tower4":
        frames = frames[6:]

    frames.insert(0, 0)
    return [{cam: obs_to_attn(demo[f], cam) for cam in cameras} for f in frames]

# Identify way-point in each RLBench Demo
def _is_stopped(demo, i, obs, stopped_buffer):
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = i < (len(demo) - 2) and (
        obs.gripper_open == demo[i + 1].gripper_open
        and obs.gripper_open == demo[i - 1].gripper_open
        and demo[i - 2].gripper_open == demo[i - 1].gripper_open
    )
    small_delta = np.allclose(obs.joint_velocities, 0, atol=0.1)
    stopped = (
        stopped_buffer <= 0
        and small_delta
        and (not next_is_not_final)
        and gripper_state_no_change
    )
    return stopped

def keypoint_discovery(demo: Demo) -> List[int]:
    """
    Discover keyframes within demo. Return keyframe indices as a list.
    """
    episode_keypoints = []
    prev_gripper_open = demo[0].gripper_open
    stopped_buffer = 0
    for i, obs in enumerate(demo):
        stopped = _is_stopped(demo, i, obs, stopped_buffer)
        stopped_buffer = 4 if stopped else stopped_buffer - 1
        # If change in gripper, or end of episode.
        last = i == (len(demo) - 1)
        if i != 0 and (obs.gripper_open != prev_gripper_open or last or stopped):
            episode_keypoints.append(i)
        prev_gripper_open = obs.gripper_open
    if (
        len(episode_keypoints) > 1
        and (episode_keypoints[-1] - 1) == episode_keypoints[-2]
    ):
        episode_keypoints.pop(-2)

    # HACK for tower3 task
    return episode_keypoints

################################# training usage #################################
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 10**6

def set_seed(seed: int = 1):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # below option set seed for current gpu
    torch.cuda.manual_seed(seed)
    # below option set seed for ALL GPU
    # torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed for training set as {seed}")

def norm_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor / torch.linalg.norm(tensor, ord=2, dim=-1, keepdim=True)

class Sample(TypedDict):
    frame_id: torch.Tensor
    task: Union[List[str], str]
    variation: Union[List[int], int]
    rgbs: torch.Tensor
    pcds: torch.Tensor
    action: torch.Tensor
    padding_mask: torch.Tensor
    instr: torch.Tensor
    gripper: torch.Tensor

def compute_rotation_metrics(
    pred: torch.Tensor,
    true: torch.Tensor,
    reduction: str = "mean",
) -> Dict[str, torch.Tensor]:
    pred = norm_tensor(pred)
    acc = (pred - true).abs().max(1).values < 0.05
    acc = acc.to(pred.dtype)

    if reduction == "mean":
        acc = acc.mean()
    return {"rotation": acc}

def compute_rotation_loss(logit: torch.Tensor, rot: torch.Tensor):
    dtype = logit.dtype
    rot_ = -rot.clone()
    loss = F.mse_loss(logit, rot, reduction="none").to(dtype)
    loss = loss.mean(1)

    loss_ = F.mse_loss(logit, rot_, reduction="none").to(dtype)
    loss_ = loss_.mean(1)

    select_mask = (loss < loss_).float()

    sym_loss = 4 * (select_mask * loss + (1 - select_mask) * loss_)

    return {"rotation": sym_loss.mean()}

class LossAndMetrics:
    def __init__(
        self,
    ):
        # we don't predict task id like HiveFormer
        pass

    def compute_loss(
        self, pred: Dict[str, torch.Tensor], sample: Sample
    ) -> Dict[str, torch.Tensor]:
        """use for training"""
        device = pred["position"].device
        padding_mask = sample["padding_mask"].to(device)
        outputs = sample["action"].to(device)[padding_mask]

        losses = {}
        losses["position"] = F.mse_loss(pred["position"], outputs[:, :3]) * 3

        losses.update(compute_rotation_loss(pred["rotation"], outputs[:, 3:7]))
        losses["gripper"] = F.mse_loss(pred["gripper"], outputs[:, 7:8])

        return losses

    def compute_metrics(
        self, pred: Dict[str, torch.Tensor], sample: Sample
    ) -> Dict[str, torch.Tensor]:
        """use for validation"""
        device = pred["position"].device
        dtype = pred["position"].dtype
        padding_mask = sample["padding_mask"].to(device)
        outputs = sample["action"].to(device)[padding_mask]

        metrics = {}

        acc = ((pred["position"] - outputs[:, :3]) ** 2).sum(1).sqrt() < 0.01
        metrics["position"] = acc.to(dtype).mean()

        pred_gripper = (pred["gripper"] > 0.5).squeeze(-1)
        true_gripper = outputs[:, 7].bool()
        acc = pred_gripper == true_gripper
        metrics["gripper"] = acc.to(dtype).mean()

        metrics.update(compute_rotation_metrics(pred["rotation"], outputs[:, 3:7]))

        return metrics

def collate_fn(batch: List[Dict]):
    keys = batch[0].keys()
    return {
        key: default_collate([item[key] for item in batch])
        if batch[0][key] is not None
        else None
        for key in keys
    }

def get_data_loader(args):

    # get instruction features and padding (if needed)
    # i hard code so that I will take the first element (1st path)
    lang_feat, eos_feat, lang_pad, lang_num  = load_instructions( 
        args,
        path = args.dataset[0],
        var_num = args.variations[0] )

    max_episode_length = get_max_episode_length(args.tasks, args.episodes_json_path)

    dataset = RLBenchDataset(
        root=args.dataset,
        tasks = args.tasks,
        taskvar=args.variations[0],
        name=args.name,
        instructions=(lang_feat, eos_feat, lang_pad, lang_num ),
        max_episode_length=max_episode_length,
        max_episodes_per_taskvar=args.max_episodes_per_taskvar,
        cache_size=args.cache_size,
        num_iters=args.train_iters,
        cameras=args.cameras,  # type: ignore
    )
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loaders = {}
    if args.dataset_val[0] != None:
        for var_index in args.val_variations:
            lang_feat, eos_feat, lang_pad, lang_num = load_instructions( 
                args,
                path = args.dataset_val[0],
                var_num = var_index )

            val_dataset = RLBenchDataset(
            root= args.dataset_val,
            tasks = args.tasks,
            taskvar= var_index,
            name=args.name,
            instructions=(lang_feat, eos_feat, lang_pad, lang_num ),
            max_episode_length=max_episode_length,
            max_episodes_per_taskvar=args.max_episodes_per_taskvar,
            cache_size=args.cache_size,
            num_iters=args.train_iters,
            cameras=args.cameras,  # type: ignore
            )
            val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=args.val_batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                collate_fn=collate_fn,
            )
            val_loaders[var_index] = val_loader
    return (train_loader, val_loaders)

class Checkpointer:
    pass