import tap
from typing import List, Tuple, Optional
from pathlib import Path

class Arguments(tap.Tap):
    accumulate_grad_batches: int = 2
    cameras: Tuple[str, ...] = ("wrist", "left_shoulder", "right_shoulder")
    checkpoint: Optional[Path] = None
    checkpoint_period: int = 10
    dataset: List[Path] = ["/home/yiqiw2/experiment/language_rl/train_datasets/"]
    dataset_val: List[Path] = ["/home/yiqiw2/experiment/language_rl/val_datasets/"]
    device: str = "cuda:0"
    xp: Path = Path(__file__).parent / "xp"
    name: str = "LAVA"
    num_workers: int = 5
    max_tries: int = 10
    max_episodes_per_taskvar: int = 100
    instructions: Optional[Path] = None
    cache_size: int = 100
    lang_emb: str = 'CLIP'
    seed: int = 2

    tasks: str = "push_buttons" # # if multi-tasks, then "task_a task_b task_c "
    variations: str =  '1 ' # if variations 1 2 3 then "1 2 3"
    val_variations: str = '7 '
    episodes_json_path: str = "/home/yiqiw2/experiment/language_rl/TemporalLatentGoal/Preprocess/episodes.json"
    val_number: int = 10 # frequency of validation
    
    # Train
    batch_size: int = 32
    lr: float = 0.001
    
    val_batch_size: int = 1 # given the fact that the dataset has only 10 episodes
    train_iters: int = 100
    jitter: bool = False
    load_model: bool = False
    load_name: str = ''
    save_model: bool = False # whether saviing the best model
    save_path: str = '/home/yiqiw2/experiment/language_rl/saved_model'

    # tests
    headless: bool = True
    output: Path = Path(__file__).parent / "records.txt"

    # model
    depth: int = 5
    dim_feedforward: int = 512
    hidden_dim: int = 32 # used for visual tokens
    instr_size: int = 512
    mask_obs_prob: float = 0.0
    num_layers: int = 1
    cross_layers: int = 3 # will be double for LAVA since it doesn't have policy
    policy_layers: int = 3
    expert_counts: int = 1 # default 1, if using VALA, then 6 = 2 modalities x 3 views
    modality_fusion: bool = True # whether we fuse lang_goal and vision or not
    position_offset: bool = True
    lang_offset: bool = False
    offset_emb: bool = False # add multi-view and time embedding before making a prediction
    no_film: bool = True # use for debuggging
    max_episode_length: int = 10

    oracle_goal: bool = False # we manually provide sub-goal per step

    log_to_wandb: bool = False