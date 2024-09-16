import random

import numpy as np
import torch


def fix_random_seed(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_tri_stage_schedule(
    optimizer, base_lr: float, min_lr: float, warmup_steps: int, hold_steps: int, decay_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    def tri_stage_schedule(current_step: int) -> float:
        if current_step < warmup_steps:
            return (min_lr + (base_lr - min_lr) * float(current_step) / float(warmup_steps)) / base_lr
        elif current_step < warmup_steps + hold_steps:
            return 1.0
        else:
            progress = float(current_step - (warmup_steps + hold_steps)) / float(decay_steps)
            return (min_lr + (base_lr - min_lr) * (1 - progress)) / base_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, tri_stage_schedule)


def compute_syllable_purity(p_xy: np.ndarray):
    return np.sum(np.max(p_xy, axis=0))


def compute_cluster_purity(p_xy: np.ndarray):
    return np.sum(np.max(p_xy, axis=1))


def compute_mutual_info(p_xy: np.ndarray):
    n_syllables, n_clusters = p_xy.shape

    p_syllable = np.sum(p_xy, axis=1)
    p_cluster = np.sum(p_xy, axis=0)

    mi = 0
    for i in range(n_syllables):
        for j in range(n_clusters):
            if p_xy[i, j] != 0:
                mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_syllable[i] * p_cluster[j]))

    return mi
