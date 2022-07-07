import torch
import numpy as np
from typing import Any, Callable, Union, List, Dict


def nested_detach_and_clone(obj: Any, to_cpu: bool = False, to_numpy: bool = False):
    if to_cpu is False and to_numpy is True:
        raise ValueError("Numpy has to be on CPU")

    def _operation(X: torch.Tensor) -> Union[torch.Tensor, np.ndarray]:
        # https://stackoverflow.com/questions/55266154/pytorch-preferred-way-to-copy-a-tensor
        _X = X.detach().clone()
        if to_cpu is True:
            _X = _X.cpu()

        if to_numpy is True:
            _X = _X.numpy()

        return _X

    return nested_tensor_operation(obj=obj, tensor_operation=_operation)


def nested_to_cuda(obj: Any):

    def _operation(X: torch.Tensor) -> torch.Tensor:
        return X.cuda()

    return nested_tensor_operation(obj=obj, tensor_operation=_operation)


def nested_tensor_operation(obj: Any, tensor_operation: Callable[[torch.Tensor], Any]) -> Any:
    """Nested Application of `detach().clone()`.

       This function will remove gradients and reference.
    """
    if isinstance(obj, (list, tuple)):
        return [
            nested_tensor_operation(
                obj=_obj,
                tensor_operation=tensor_operation)
            for _obj in obj]

    if isinstance(obj, dict):
        new_dict_obj = {}
        for key, val in obj.items():
            if not isinstance(key, str):
                raise NotImplementedError

            new_dict_obj[key] = nested_tensor_operation(
                obj=val,
                tensor_operation=tensor_operation)

        return new_dict_obj

    if isinstance(obj, torch.Tensor):
        return tensor_operation(obj)

    if obj is None:
        return obj

    if isinstance(obj, bool):
        # Special handling, since `bool` is subclass of `int
        # https://stackoverflow.com/questions/37888620/comparing-boolean-and-int-using-isinstance
        return obj

    if isinstance(obj, (int, float, str)):
        return obj

    raise TypeError(f"Unrecognized type {type(obj)}")


def unionize_dicts(dicts: List[Dict]) -> Dict:
    union_dict: Dict = {}
    for d in dicts:
        for k, v in d.items():
            if k in union_dict.keys():
                raise KeyError
            union_dict[k] = v

    return union_dict
