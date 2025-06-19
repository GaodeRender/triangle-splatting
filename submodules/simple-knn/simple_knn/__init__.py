import torch

from . import _C


def distCUDA2(points: torch.Tensor) -> torch.Tensor:
    """
    return the mean squared distance of each point to its 3 nearest neighbor
    args:
        points: torch.Tensor with shape (N, 3)
    """
    return _C.distCUDA2(points)


def nearestNeighbor(points: torch.Tensor, batch_size: int = 1) -> torch.Tensor:
    """
    return the nearest neighbor index of each point, ignore points in the same batch
    args:
        points: torch.Tensor with shape (N, 3) and N % batch_size == 0
        batch_size: int
    """
    return _C.nearestNeighbor(points, batch_size)