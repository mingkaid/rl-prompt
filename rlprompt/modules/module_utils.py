import torch
# import numpy as np
# import sys
# if sys.version_info >= (3, 8):
#     from typing import Union, List, Dict, Any, TypedDict, NamedTuple, Callable
# else:
#     from typing import Union, List, Dict, Any, NamedTuple, Callable
#     from typing_extensions import TypedDict
from typing import Callable
from enum import Enum


class ForwardMode(Enum):
    # MLE = "MLE"
    # PG = "PG"
    SQL_ON = "SQL_ON"
    SQL_OFF_GT = "SQL_OFF_GT"
    # SQL_OFF_RB = "SQL_OFF_RB"
    # SQL_OFF_BEHAVIOR = "SQL_OFF_BEHAVIOR"
    INFER = "INFER"


def get_reward_shaping_func(
    old_min: float,
    old_max: float,
    new_min: float,
    new_max: float
) -> Callable[[torch.Tensor], torch.Tensor]:
    def _shaping_func(reward: torch.Tensor) -> torch.Tensor:
        percentile = (reward - old_min) / (old_max - old_min)
        return percentile * (new_max - new_min) + new_min

    return _shaping_func

