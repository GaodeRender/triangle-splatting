import os
import multiprocessing
from typing import Callable, List, Tuple, Any


def run_exp(func_list: List[Callable], num_workers: int = 1, multi_process: bool = True, start_method: str = "spawn", delay: float = 0):
    """
    This function has to be run in the main process.
    """
    if num_workers <= 0:
        multi_process = False
    
    if not multi_process:
        for func in func_list:
            func()
    else:
        ctx = multiprocessing.get_context(start_method)

        batch_list = []
        for i in range(0, len(func_list), num_workers):
            batch_list.append(func_list[i : i + num_workers])

        for batch in batch_list:
            processes = []
            for func in batch:
                p = ctx.Process(target=func)
                p.start()
                processes.append(p)
                if delay > 0:
                    os.system(f"sleep {delay}")  # stagger each process
            for p in processes:
                p.join()


def run_exp_with_args(
    func: Callable, args_list: List[Tuple[Any]], num_workers: int = 1, multi_process: bool = True, start_method: str = "spawn", delay: float = 0
):
    """
    This function has to be run in the main process.
    """
    if num_workers <= 0:
        multi_process = False
    
    if not multi_process:
        for args in args_list:
            func(*args)
    else:
        ctx = multiprocessing.get_context(start_method)

        batch_list = []
        for i in range(0, len(args_list), num_workers):
            batch_list.append(args_list[i : i + num_workers])

        for batch in batch_list:
            processes = []
            for args in batch:
                p = ctx.Process(target=func, args=args)
                p.start()
                processes.append(p)
                if delay > 0:
                    os.system(f"sleep {delay}")  # stagger each process

            for p in processes:
                p.join()
