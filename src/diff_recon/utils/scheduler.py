import numpy as np
from typing import List


def exponential_scheduler(v_init: float, v_final: float, max_steps: int, delay_steps: int = 0, delay_mult: float = 1.0):

    def scheduler(step: int):
        if step <= 0:
            return v_init
        if step >= max_steps:
            return v_final

        if delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = delay_mult + (1 - delay_mult) * np.sin(0.5 * np.pi * np.clip(step / delay_steps, 0, 1))
        else:
            delay_rate = 1.0

        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(v_init) * (1 - t) + np.log(v_final) * t)
        return delay_rate * log_lerp

    return scheduler


def step_scheduler(v_list: List[float], step_list: List[int]):
    assert len(v_list) == len(step_list) + 1 or len(v_list) == len(step_list)

    def scheduler(step: int):
        for i, s in enumerate(step_list):
            if step < s:
                return v_list[i]
        return v_list[-1]

    return scheduler


def exponential_step_scheduler(v_init: float, v_final: float, max_steps: int, n_stage: int, delay_steps: int = 0, delay_mult: float = 1.0):
    exp_scheduler_ = exponential_scheduler(v_init, v_final, max_steps, delay_steps, delay_mult)
    step_list = [int(max_steps * i / n_stage) for i in range(n_stage + 1)]
    v_list = [exp_scheduler_(step) for step in step_list]
    return step_scheduler(v_list, step_list)
