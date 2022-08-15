from dataclasses import dataclass
from torch.utils.data import Dataset
from typing import Optional

from rlprompt.modules import BaseModule
from rlprompt.trainers import Trainer


def make_trainer(module: BaseModule,
                 train_dataset: Optional[Dataset],
                 eval_dataset: Optional[Dataset],
                 config: "DictConfig") -> Trainer:
    return Trainer(module, train_dataset, config.train_batch_size,
                   config.train_shuffle, config.train_drop_last, 
                   config.num_train_epochs, config.max_train_steps, 
                   config.do_eval, eval_dataset, config.eval_batch_size, 
                   config.eval_steps, config.do_save, config.save_dir, 
                   config.save_steps, config.learning_rate, 
                   config.gradient_clip, config.gradient_clip_norm, 
                   config.checkpoint_path, config.random_seed,
                   config.report_to_wandb, config.project_name, 
                   config.run_name)


@dataclass
class TrainerConfig:
    # Train params
    train_batch_size: int = 16
    train_shuffle: bool = True
    train_drop_last: bool = True
    num_train_epochs: int = 1
    max_train_steps: int = -1
    # Eval params
    do_eval: bool = True
    eval_batch_size: int = 16
    eval_steps: int = -1
    # Save params
    do_save: bool = True
    save_dir: str = './outputs'
    save_steps: int = -1
    # Optimizer params
    learning_rate: float = 1e-4
    gradient_clip: bool = True
    gradient_clip_norm: float = 5.0
    # Checkpoint params
    checkpoint_path: Optional[str] = None
    # Random seed
    random_seed: Optional[int] = None
    # Wandb reporting
    report_to_wandb: bool = True
    project_name: Optional[str] = 'rl-prompt'
    run_name: Optional[str] = None