# basic utils
import os
import numpy as np
import random
from typing import Tuple, Dict, List
import einops
# RLbench 
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.environment import Environment
from rlbench.backend.observation import Observation
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.demo import Demo
from pyrep.const import RenderMode

# deep learning stuff
import torch
import torch.nn as nn

# project-specific
from Utils.utils import obs_to_attn, keypoint_discovery
from Test.utils import Mover

############################# Dataset Preprocess Usage #############################
def transform(obs_dict, scale_size=(0.75, 1.25), augmentation=False):
    apply_depth = len(obs_dict.get("depth", [])) > 0
    apply_pc = len(obs_dict["pc"]) > 0
    num_cams = len(obs_dict["rgb"])

    obs_rgb = []
    obs_depth = []
    obs_pc = []
    for i in range(num_cams):
        rgb = torch.tensor(obs_dict["rgb"][i]).float().permute(2, 0, 1)
        depth = (
            torch.tensor(obs_dict["depth"][i]).float().permute(2, 0, 1)
            if apply_depth
            else None
        )
        pc = (
            torch.tensor(obs_dict["pc"][i]).float().permute(2, 0, 1) if apply_pc else None
        )

        if augmentation:
            raise NotImplementedError()  # Deprecated

        # normalise to [-1, 1]
        rgb = rgb / 255.0
        rgb = 2 * (rgb - 0.5)

        obs_rgb += [rgb.float()]
        if depth is not None:
            obs_depth += [depth.float()]
        if pc is not None:
            obs_pc += [pc.float()]
    obs = obs_rgb + obs_depth + obs_pc
    return torch.cat(obs, dim=0)
############################# RLBenchEnv #############################

def task_file_to_task_class(task_file):
    import importlib

    name = task_file.replace(".py", "")
    class_name = "".join([w[0].upper() + w[1:] for w in name.split("_")])
    mod = importlib.import_module("rlbench.tasks.%s" % name)
    mod = importlib.reload(mod)
    task_class = getattr(mod, class_name)
    return task_class

class RLBenchEnv:
    def __init__(
        self,
        data_path,
        apply_rgb=False,
        apply_depth=False,
        apply_pc=False,
        headless=False,
        apply_cameras=("left_shoulder", "right_shoulder", "wrist", "front"),
    ):

        # setup required inputs
        self.data_path = data_path
        self.apply_rgb = apply_rgb
        self.apply_depth = apply_depth
        self.apply_pc = apply_pc
        self.apply_cameras = apply_cameras

        # setup RLBench environments
        self.obs_config = self.create_obs_config(
            apply_rgb, apply_depth, apply_pc, apply_cameras
        )
        self.action_mode = MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(),
            gripper_action_mode=Discrete(),
        )
        self.env = Environment(
            self.action_mode, str(data_path), self.obs_config, headless=headless
        )

    def get_obs_action(self, obs):
        """
        Fetch the desired state and action based on the provided demo.
            :param obs: incoming obs
            :return: required observation and action list
        """

        # fetch state
        state_dict = {"rgb": [], "depth": [], "pc": []}
        for cam in self.apply_cameras:
            if self.apply_rgb:
                rgb = getattr(obs, "{}_rgb".format(cam))
                state_dict["rgb"] += [rgb]

            if self.apply_depth:
                depth = getattr(obs, "{}_depth".format(cam))
                state_dict["depth"] += [depth]

            if self.apply_pc:
                pc = getattr(obs, "{}_point_cloud".format(cam))
                state_dict["pc"] += [pc]

        # fetch action
        action = np.concatenate([obs.gripper_pose, [obs.gripper_open]])
        return state_dict, torch.from_numpy(action).float()

    def get_rgb_pcd_gripper_from_obs(
        self, obs: Observation
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return rgb, pcd, and gripper from a given observation
        :param obs: an Observation from the env
        :return: rgb, pcd, gripper
        """
        state_dict, gripper = self.get_obs_action(obs)
        state = transform(state_dict, augmentation=False)
        state = einops.rearrange(
            state,
            "(m n ch) h w -> n m ch h w",
            ch=3,
            n=len(self.apply_cameras),
            m=2,
        )
        rgb = state[:, 0].unsqueeze(0)  # 1, N, C, H, W
        pcd = state[:, 1].unsqueeze(0)  # 1, N, C, H, W
        gripper = gripper.unsqueeze(0)  # 1, D

        attns = torch.Tensor([])
        for cam in self.apply_cameras:
            u, v = obs_to_attn(obs, cam)
            attn = torch.zeros((1, 1, 1, 128, 128))
            if not (u < 0 or u > 127 or v < 0 or v > 127):
                attn[0, 0, 0, v, u] = 1
            attns = torch.cat([attns, attn], 1)
        rgb = torch.cat([rgb, attns], 2)

        return rgb, pcd, gripper

    def get_obs_action_from_demo(self, demo: Demo):
        """
        Fetch the desired state and action based on the provided demo.
            :param demo: fetch each demo and save key-point observations
            :param normalise_rgb: normalise rgb to (-1, 1)
            :return: a list of obs and action
        """
        key_frame = keypoint_discovery(demo)
        key_frame.insert(0, 0)
        state_ls = []
        action_ls = []
        for f in key_frame:
            state, action = self.get_obs_action(demo._observations[f])
            state = transform(state, augmentation=False)
            state_ls.append(state.unsqueeze(0))
            action_ls.append(action.unsqueeze(0))
        return state_ls, action_ls

    def get_demo(self, task_name, variation, episode_index):
        """
        Fetch a demo from the saved environment.
            :param task_name: fetch task name
            :param variation: fetch variation id
            :param episode_index: fetch episode index: 0 ~ 99
            :return: desired demo
        """
        demos = self.env.get_demos(
            task_name=task_name,
            variation_number=variation,
            amount=1,
            from_episode_number=episode_index,
            random_selection=False,
        )
        return demos


    def create_obs_config(
        self, apply_rgb, apply_depth, apply_pc, apply_cameras, **kwargs
    ):
        """
        Set up observation config for RLBench environment.
            :param apply_rgb: Applying RGB as inputs.
            :param apply_depth: Applying Depth as inputs.
            :param apply_pc: Applying Point Cloud as inputs.
            :param apply_cameras: Desired cameras.
            :return: observation config
        """
        unused_cams = CameraConfig()
        unused_cams.set_all(False)
        used_cams = CameraConfig(
            rgb=apply_rgb,
            point_cloud=apply_pc,
            depth=apply_depth,
            mask=False,
            render_mode=RenderMode.OPENGL,
            **kwargs,
        )

        camera_names = apply_cameras
        kwargs = {}
        for n in camera_names:
            kwargs[n] = used_cams

        obs_config = ObservationConfig(
            front_camera=kwargs.get("front", unused_cams),
            left_shoulder_camera=kwargs.get("left_shoulder", unused_cams),
            right_shoulder_camera=kwargs.get("right_shoulder", unused_cams),
            wrist_camera=kwargs.get("wrist", unused_cams),
            overhead_camera=kwargs.get("overhead", unused_cams),
            joint_forces=False,
            joint_positions=False,
            joint_velocities=True,
            task_low_dim_state=False,
            gripper_touch_forces=False,
            gripper_pose=True,
            gripper_open=True,
            gripper_matrix=True,
            gripper_joint_positions=True,
        )

        return obs_config
    
    def evaluate(
        self,
        task_str: str,
        max_episodes: int,
        variation: int,
        num_demos: int,
        log_dir: Optional[Path],
        agent,
        instructions,
        offset: int = 0,
        max_tries: int = 1,
        save_attn: bool = False,
    ):
        """
        Evaluate the policy network on the desired demo or test environments
            :param task_type: type of task to evaluate
            :param max_episodes: maximum episodes to finish a task
            :param num_demos: number of test demos for evaluation
            :param model: the policy network
            :param demos: whether to use the saved demos
            :return: success rate
        """
        lang_feat, eos_feat, lang_pad, lang_num = instructions
        lang_idx = 0
        instruction = (lang_feat, eos_feat) 

        self.env.launch()
        # task_str: push_buttons.py
        task_type = task_file_to_task_class(task_str)
        task = self.env.get_task(task_type)
        task.set_variation(variation)  # type: ignore

        device = agent.device

        success_rate = 0.0

        if demos is None:
            fetch_list = [i for i in range(num_demos)]
        else:
            fetch_list = demos

        with torch.no_grad():
            for demo_id, demo in enumerate(tqdm(fetch_list)):

                images = []
                rgbs = torch.Tensor([]).to(device)
                pcds = torch.Tensor([]).to(device)
                grippers = torch.Tensor([]).to(device)

                # reset a new demo or a defined demo in the demo list
                _, obs = task.reset()

                images.append(
                    {cam: getattr(obs, f"{cam}_rgb") for cam in self.apply_cameras}
                )
                move = Mover(task, max_tries=max_tries)
                reward = None

                for step_id in range(max_episodes):
                    # fetch the current observation, and predict one action
                    rgb, pcd, gripper = self.get_rgb_pcd_gripper_from_obs(obs)

                    rgb = rgb.to(device)
                    pcd = pcd.to(device)
                    gripper = gripper.to(device)

                    rgbs = torch.cat([rgbs, rgb.unsqueeze(1)], dim=1)
                    pcds = torch.cat([pcds, pcd.unsqueeze(1)], dim=1)
                    grippers = torch.cat([grippers, gripper.unsqueeze(1)], dim=1)

                    output = agend.act( step_id, rgbs, pcds, instructions[lang_idx:lang_idx+1], lang_pad[lang_idx:lang_idx+1])
                    action = output["action"]

                    if action is None:
                        break

                    # update the observation based on the predicted action
                    try:
                        action_np = action[-1].detach().cpu().numpy()

                        obs, reward, terminate, step_images = move(action_np)

                        images += step_images

                        if reward == 1:
                            success_rate += 1 / num_demos
                            break

                        if terminate:
                            print("The episode has terminated!")

                    except (IKError, ConfigurationPathError, InvalidActionError) as e:
                        print(task_type, demo, step_id, success_rate, e)
                        reward = 0
                        break

                print(
                    task_str,
                    "Reward",
                    reward,
                    "Step",
                    demo_id,
                )

        self.env.shutdown()
        return success_rate

def get_observation(task_str: str, variation: int, episode: int, env: RLBenchEnv):
    demos = env.get_demo(task_str, variation, episode)
    demo = demos[0]

    key_frame = keypoint_discovery(demo)
    # HACK for tower3
    if task_str == "tower3":
        key_frame = [k for i, k in enumerate(key_frame) if i % 6 in set([1, 4])]
    # HACK tower4
    elif task_str == "tower4":
        key_frame = key_frame[6:]
    key_frame.insert(0, 0)

    state_ls = []
    action_ls = []
    for f in key_frame:
        state, action = env.get_obs_action(demo._observations[f])
        state = transform(state)
        state_ls.append(state.unsqueeze(0))
        action_ls.append(action.unsqueeze(0))

    return demo, state_ls, action_ls
