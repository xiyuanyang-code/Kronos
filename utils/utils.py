from datetime import datetime
from typing import Optional


def reseed_everything(seed: Optional[int]):
    """
    Set random seeds for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (Optional[int]): The seed value to use. If None, the function does nothing.

    This function sets the random seed for the Python `random` module, NumPy, and PyTorch (both CPU and CUDA).
    It also sets the `PYTHONHASHSEED` environment variable and configures PyTorch's cuDNN backend for deterministic results.
    """
    import random
    import os
    import numpy as np
    import torch
    from torch.backends import cudnn

    # seed everything
    if seed is None:
        return

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True


def generate_timestamp():
    # Get the current datetime object
    now = datetime.now()
    format = now.strftime("%Y-%m-%d-%H")
    return format
