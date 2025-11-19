# scheduler.py
import math
from torch.optim.lr_scheduler import _LRScheduler


class CosineWithRestartsLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, t0_steps, t_mult=1.2, min_lr_ratio=0.1, warmup_mode="linear"):
        self.warm = int(warmup_steps)
        self.t0 = int(t0_steps)
        self.t_mult = float(t_mult)
        self.min_ratio = float(min_lr_ratio)
        self.mode = warmup_mode
        super().__init__(optimizer)  # initializes base_lrs and last_epoch = -1

    def get_lr(self):
        s = max(0, self.last_epoch)  # step count compatible with state_dict
        # warmup
        if s < self.warm:
            p = s / max(1, self.warm)
            if self.mode == "linear":
                f = p
            elif self.mode == "cosine":
                f = 0.5 * (1 - math.cos(math.pi * p))
            else:
                f = p
            f = max(0.01, f)
            return [base * f for base in self.base_lrs]

        s2 = s - self.warm
        # compute current cosine cycle
        cyc_len = self.t0
        start = 0
        while start + cyc_len <= s2:
            start += cyc_len
            cyc_len = int(cyc_len * self.t_mult)
        pos = (s2 - start) / max(1, cyc_len)

        lrs = []
        for base in self.base_lrs:
            min_lr = base * self.min_ratio
            lr = min_lr + (base - min_lr) * 0.5 * (1 + math.cos(math.pi * pos))
            lrs.append(lr)
        return lrs

    def step(self, epoch=None):
        super().step(epoch)
