import torch
import torch.nn.functional as F

from simple_knn import distCUDA2


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x / (1 - x))


def to_numpy(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return tuple(arg.detach().cpu().numpy() for arg in args)


def cat_tensor_dict(d1: dict[str, torch.Tensor], d2: dict[str, torch.Tensor], dim=0):
    for k, v in d2.items():
        d1[k] = torch.cat((d1[k], v), dim=dim) if k in d1 else v


def filter_tensor_dict(d: dict[str, torch.Tensor], indices: torch.Tensor):
    for k, v in d.items():
        d[k] = v[indices]


def get_first_unique_indices(t: torch.Tensor, dim=0) -> torch.Tensor:
    _, idx, counts = torch.unique(t, dim=dim, sorted=True, return_inverse=True, return_counts=True)
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0], device=t.device), cum_sum[:-1]))
    first_indicies = ind_sorted[cum_sum]
    return first_indicies


def inter_point_distance(pc: torch.Tensor) -> torch.Tensor:
    assert pc.dim() == 2 and pc.size(1) == 3
    return distCUDA2(pc).clamp_(min=1e-10).sqrt()


def get_inside_mask(points: torch.Tensor, bbox: tuple[float, ...]) -> torch.Tensor:
    if bbox is not None:
        if len(bbox) == 4:
            x_min, y_min, x_max, y_max = bbox
            mask = (points[:, 0] >= x_min) & (points[:, 0] <= x_max) & (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
        elif len(bbox) == 6:
            x_min, y_min, z_min, x_max, y_max, z_max = bbox
            mask = (
                (points[:, 0] >= x_min)
                & (points[:, 0] <= x_max)
                & (points[:, 1] >= y_min)
                & (points[:, 1] <= y_max)
                & (points[:, 2] >= z_min)
                & (points[:, 2] <= z_max)
            )
        else:
            raise ValueError(f"bbox must be of length 4 or 6, but got {len(bbox)}")
    else:
        mask = torch.ones_like(points[:, 0], dtype=torch.bool)
    return mask


def argsort(keys: torch.Tensor, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
    indices = torch.argsort(keys)
    return tuple(arg[indices] for arg in args)


def get_color_tensor(color: str) -> torch.Tensor:
    if color == "black":
        return torch.tensor([0, 0, 0], dtype=torch.float32)
    elif color == "white":
        return torch.tensor([1, 1, 1], dtype=torch.float32)
    elif color == "random":
        return torch.rand(3)
    else:
        raise ValueError(f"Unknown background color: {color}")


def build_Rmat(q: torch.Tensor) -> torch.Tensor:
    q = F.normalize(q, dim=1)
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R = torch.zeros((q.size(0), 3, 3)).to(q.device)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def grid_sampling(xyz: torch.Tensor, *attrs: torch.Tensor, grid_size: float = 0.0) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """
    :param xyz: tensor with shape (N, 3), where N is the number of points
    :param attrs: tuple of tensors with shape (N, D)
    :param grid_size: float
    :return: (M, 3) or ((M, 3), ...), where M is the number of sampled points
    """
    if grid_size == 0.0:
        return xyz if len(attrs) == 0 else (xyz, *attrs)

    grid_coords = torch.round(xyz / grid_size).int()
    if len(attrs) == 0:
        grid_coords_unique = torch.unique(grid_coords, dim=0)
        sampled_xyz = grid_coords_unique.float() * grid_size
        return sampled_xyz
    else:
        grid_coords_unique, inverse_indices = torch.unique(grid_coords, return_inverse=True, dim=0)
        sampled_xyz = grid_coords_unique.float() * grid_size
        sampled_attrs = []
        for attr in attrs:
            sampled_attr = torch.zeros((sampled_xyz.shape[0], attr.shape[1])).float().to(attr.device)
            sampled_attr.scatter_reduce_(0, inverse_indices.unsqueeze(1).expand(-1, attr.shape[1]), attr, "mean")
            sampled_attrs.append(sampled_attr)
        return sampled_xyz, *sampled_attrs


def grid_size_search(xyz: torch.Tensor, n_sample: int, tolerance: float = 0.1, max_retry: int = 10) -> float:
    """
    Run a binary search to find the grid size that can sample n_sample points
    (within an error margin) from the input point cloud.

    :param xyz: (N, 3)
    :param n_sample: int
    :return: float
    """
    if n_sample is None or n_sample >= xyz.shape[0]:
        return 0.0

    min_grid_size = 0
    max_grid_size = (xyz.max(dim=0).values - xyz.min(dim=0).values).max().item()
    n_sample_error_tolerance = tolerance * n_sample
    n_sample_min = n_sample - n_sample_error_tolerance
    n_sample_max = n_sample + n_sample_error_tolerance

    grid_size = max_grid_size / n_sample ** (1 / 3)

    for _ in range(max_retry):
        n = grid_sampling(xyz, grid_size=grid_size).shape[0]
        if n_sample_min <= n <= n_sample_max:
            return grid_size
        elif n < n_sample_min:
            max_grid_size = grid_size
            grid_size = (min_grid_size + max_grid_size) / 2
        else:
            min_grid_size = grid_size
            grid_size = (min_grid_size + max_grid_size) / 2
    return grid_size


if __name__ == "__main__":
    xyz = torch.rand(10_000, 3) * 1000
    shs = torch.rand(10_000, 10)
    grid_size = 1
    sampled_xyz, sampled_shs = grid_sampling(xyz, shs, grid_size=grid_size)
    # grid_size_search(xyz, 1_000_000)
