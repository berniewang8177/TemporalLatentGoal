import numpy as np
import pathlib
import einops
import torch
import pickle
import torch.nn.functional as F

class Mover:
    def __init__(self, task, disabled: bool = False, max_tries: int = 1):
        self._task = task
        self._last_action: Optional[np.ndarray] = None
        self._step_id = 0
        self._max_tries = max_tries
        self._disabled = disabled

    def __call__(self, action: np.ndarray):
        if self._disabled:
            return self._task.step(action)

        target = action.copy()
        if self._last_action is not None:
            action[7] = self._last_action[7].copy()

        images = []
        try_id = 0
        obs = None
        terminate = None
        reward = 0

        for try_id in range(self._max_tries):
            feedback = self._task.step(action)
            obs, reward, terminate = feedback # we don't return other_obs
            # if other_obs == []:
            #     other_obs = [obs]
            # for o in other_obs:
            #     images.append(
            #         {
            #             k.split("_")[0]: getattr(o, k)
            #             for k in o.__dict__.keys()
            #             if "_rgb" in k and getattr(o, k) is not None
            #         }
            #     )

            pos = obs.gripper_pose[:3]
            rot = obs.gripper_pose[3:7]
            gripper = obs.gripper_open
            dist_pos = np.sqrt(np.square(target[:3] - pos).sum())
            dist_rot = np.sqrt(np.square(target[3:7] - rot).sum())
            criteria = (dist_pos < 5e-2,)

            if all(criteria) or reward == 1:
                
                break

            print(
                f"Too far away (pos: {dist_pos:.3f}, rot: {dist_rot:.3f}, step: {self._step_id})... Retrying..."
            )

        # we execute the gripper action after re-tries
        action = target
        if (
            not reward
            and self._last_action is not None
            and action[7] != self._last_action[7]
        ):
            # obs, reward, terminate, other_obs = self._task.step(action)
            obs, reward, terminate = self._task.step(action)
            # if other_obs == []:
            #     other_obs = [obs]
            # for o in other_obs:
            #     images.append(
            #         {
            #             k.split("_")[0]: getattr(o, k)
            #             for k in o.__dict__.keys()
            #             if "_rgb" in k and getattr(o, k) is not None
            #         }
            #     )

        if try_id == self._max_tries:
            print(f"Failure after {self._max_tries} tries")

        self._step_id += 1
        self._last_action = action.copy()
        print("terminate", terminate)
        return obs, reward, terminate , images

class NearestNeighbor:
    def __init__(self, args, ref_data_paths, goals):
        """Init the reference dataset for retrieval
        
        ref_data_paths: reference data location
        goals (Optional): goals for each dataset
        
        """
        self.args = args
        self.paths = ref_data_paths
        self.goals = goals
        self.ref_data = []
        self.ref_action = []

        episodes = list(pathlib.Path(self.paths).glob('low_dim_ep*'))

        for episode_path in episodes:
            with open(episode_path, 'rb') as f:
                episode = pickle.load(f)

                frame_id = episode[0]
                state = episode[1][0]
                action = episode[2][0]
            self.ref_data.append(state)
            self.ref_action.append(action)
        self.size = len(self.ref_data)

        self.ref_state = torch.cat(self.ref_data)
        self.ref_action =  torch.cat(self.ref_action)
        assert False, f"{self.ref_state.shape} {self.ref_action.shape}"

    def get_action(self, state):
        assert False, f"{state.shape}"
        states = einops.repeat(state, "B d -> (B repeat) d", repeat = self.size)
        ref_states = torch.tensor(self.ref_data)
        states = torch.tensor(states)
        index = F.cosine_similarity(ref_states,states).argmax().item()
        return self.ref_action[index]