"""This File Cannot Import Other Files"""

import os
import sys


def set_cuda_visible_devices(gpu_id: int):
    if is_module_imported("torch") or is_module_imported("tensorflow"):
        raise ValueError

    # https://stackoverflow.com/questions/37893755/tensorflow-set-cuda-visible-devices-within-jupyter
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
    print(f"Set GPU ID to {gpu_id}")


def is_module_imported(module_name: str) -> bool:
    # https://stackoverflow.com/questions/30483246/how-to-check-if-a-python-module-has-been-imported
    return module_name in sys.modules
