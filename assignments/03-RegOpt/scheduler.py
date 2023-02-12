from typing import List
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """Customed LR Scheduler, modified from COSINEANNEALINGWARMRESTARTS"""

    def __init__(self, optimizer, T, eta_min=0, last_epoch=-1):
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        """
        self.T = T
        self.eta_min = eta_min
        self.T_cur = last_epoch
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Return lr for the weight"""

        return [
            self.eta_min
            + (base_lr - self.eta_min) * (1 + np.cos(np.pi * self.T_cur / self.T)) / 2
            for base_lr in self.base_lrs
        ]

    def step(self):
        # Update the current epoch
        epoch = self.last_epoch + 1
        self.T_cur = self.T_cur + 1

        # If T_cur overpass T, reset T_cur
        if self.T_cur >= self.T:
            self.T_cur = self.T_cur - self.T

        # Update last epoch
        self.last_epoch = epoch
