"""
.. _multiprocessing:

fl_sim.utils.multiprocessing
----------------------------

This module provides managers for multiprocessing using torch.multiprocessing.

"""

import torch
from torch.multiprocessing import Manager

__all__ = [
    "ProcessManager",
]

