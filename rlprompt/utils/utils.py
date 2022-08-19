"""
Miscellaneous Utility Functions
"""
import click
import warnings
from typing import Dict, Any, Optional, List


def add_prefix_to_dict_keys_inplace(
        d: Dict[str, Any],
        prefix: str,
        keys_to_exclude: Optional[List[str]] = None,
) -> None:

    # https://stackoverflow.com/questions/4406501/change-the-name-of-a-key-in-dictionary
    keys = list(d.keys())
    for key in keys:
        if keys_to_exclude is not None and key in keys_to_exclude:
            continue

        new_key = f"{prefix}{key}"
        d[new_key] = d.pop(key)

def colorful_print(string: str, *args, **kwargs) -> None:
    print(click.style(string, *args, **kwargs))

def colorful_warning(string: str, *args, **kwargs) -> None:
    warnings.warn(click.style(string, *args, **kwargs))

def unionize_dicts(dicts: List[Dict]) -> Dict:
    union_dict: Dict = {}
    for d in dicts:
        for k, v in d.items():
            if k in union_dict.keys():
                raise KeyError
            union_dict[k] = v

    return union_dict