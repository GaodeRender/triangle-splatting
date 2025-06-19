import numpy as np
from plyfile import PlyData, PlyElement
from pathlib import Path
import os
from scipy.spatial import KDTree
from copy import deepcopy
# import open3d as o3d


def argsortMorton(xyz: np.ndarray, octree_level=20) -> np.ndarray:
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    z_min, z_max = z.min(), z.max()
    x_index = np.floor((x - x_min) / (x_max - x_min + 1e-5) * (2**octree_level)).astype(int)
    y_index = np.floor((y - y_min) / (y_max - y_min + 1e-5) * (2**octree_level)).astype(int)
    z_index = np.floor((z - z_min) / (z_max - z_min + 1e-5) * (2**octree_level)).astype(int)
    morton = np.zeros_like(x_index)
    for i in range(octree_level):
        morton |= (x_index & (1 << i)) << (2 * i + 2)
        morton |= (y_index & (1 << i)) << (2 * i + 1)
        morton |= (z_index & (1 << i)) << (2 * i + 0)
    return np.argsort(morton)


class RawGaussian:
    def __init__(
        self,
        xyz: np.ndarray = None,
        rot: np.ndarray = None,
        scale: np.ndarray = None,
        opacity: np.ndarray = None,
        shs: np.ndarray = None,
        *,
        ply_path: str = None,
    ) -> None:
        self.xyz = xyz
        self.rot = rot
        self.scale = scale
        self.opacity = opacity
        self.shs = shs
        self.normals = None

        self.ply_path = None
        self.contained_idx = None

        if ply_path is not None:
            self.loadPLY(ply_path)


    def printStats(self):
        banner = "=" * 20 + " Raw Gaussian Stats " + "=" * 20
        print(banner)

        print(f"Number of points: {len(self)}")
        print(f"Number of SHs: {self.shs.shape[1] // 3}")
        print(f"x range: {self.xyz[:, 0].min():>8.1f} - {self.xyz[:, 0].max():>8.1f}")
        print(f"y range: {self.xyz[:, 1].min():>8.1f} - {self.xyz[:, 1].max():>8.1f}")
        print(f"z range: {self.xyz[:, 2].min():>8.1f} - {self.xyz[:, 2].max():>8.1f}")
        print(f"z mean: {self.xyz[:, 2].mean():>8.1f}")
        print(f"z median: {np.median(self.xyz[:, 2]):>8.1f}")

        print("=" * len(banner))

    def shDegree(self):
        # assert self.shs.shape[1] in list(map(lambda x: ((x + 1) ** 2) * 3, [0, 1, 2, 3]))
        return int(np.sqrt(self.shs.shape[1] / 3) - 1)

    def sortByXYZ(self):
        idx = np.lexsort((self.xyz[:, 2], self.xyz[:, 1], self.xyz[:, 0]))
        self.xyz = self.xyz[idx]
        self.rot = self.rot[idx] if self.rot is not None else None
        self.scale = self.scale[idx] if self.scale is not None else None
        self.opacity = self.opacity[idx] if self.opacity is not None else None
        self.shs = self.shs[idx] if self.shs is not None else None

    def sortByMorton(self):
        idx = argsortMorton(self.xyz)
        self.xyz = self.xyz[idx]
        self.rot = self.rot[idx] if self.rot is not None else None
        self.scale = self.scale[idx] if self.scale is not None else None
        self.opacity = self.opacity[idx] if self.opacity is not None else None
        self.shs = self.shs[idx] if self.shs is not None else None

    def __len__(self):
        return len(self.xyz) if self.xyz is not None else 0

    def __getitem__(self, idx):
        return RawGaussian(
            self.xyz[idx] if self.xyz is not None else None,
            self.rot[idx] if self.rot is not None else None,
            self.scale[idx] if self.scale is not None else None,
            self.opacity[idx] if self.opacity is not None else None,
            self.shs[idx] if self.shs is not None else None,
        )

    def __iadd__(self, other):
        if len(other) == 0:
            return self

        self.xyz = np.concatenate((self.xyz, other.xyz)) if self.xyz is not None else other.xyz
        self.rot = np.concatenate((self.rot, other.rot)) if self.rot is not None else other.rot
        self.scale = np.concatenate((self.scale, other.scale)) if self.scale is not None else other.scale
        self.opacity = np.concatenate((self.opacity, other.opacity)) if self.opacity is not None else other.opacity
        self.shs = np.concatenate((self.shs, other.shs)) if self.shs is not None else other.shs

        self.contained_idx = np.concatenate((self.contained_idx, other.contained_idx)) if self.contained_idx is not None else other.contained_idx
        return self

    def resetContainedIdx(self):
        self.contained_idx = np.ones(len(self), dtype=bool)

    def __isub__(self, other):
        if len(other) == 0:
            return self

        kdTree = KDTree(other.xyz)
        distance, idx = kdTree.query(self.xyz)
        self.contained_idx &= distance > 1e-5
        self.reduce()
        return self

    def __sub__(self, other):
        diff = deepcopy(self)
        diff -= other
        return diff

    def reduce(self):
        if np.all(self.contained_idx):
            return RawGaussian()

        removed_gaussian = RawGaussian(
            self.xyz[~self.contained_idx],
            self.rot[~self.contained_idx],
            self.scale[~self.contained_idx],
            self.opacity[~self.contained_idx],
            self.shs[~self.contained_idx],
        )

        self.xyz = self.xyz[self.contained_idx]
        self.rot = self.rot[self.contained_idx]
        self.scale = self.scale[self.contained_idx]
        self.opacity = self.opacity[self.contained_idx]
        self.shs = self.shs[self.contained_idx]

        self.contained_idx = np.ones(len(self), dtype=bool)
        return removed_gaussian

    def replace(self, indexes, other):
        # if other length is not equal to removed_gaussian, raise error
        if len(indexes) != len(other):
            raise ValueError(
                "Length of removed_gaussian is not equal to other, length of removed_gaussian is {}, length of other is {}".format(
                    len(indexes), len(other)
                )
            )

        self.xyz[indexes] = other.xyz
        self.rot[indexes] = other.rot
        self.scale[indexes] = other.scale
        self.opacity[indexes] = other.opacity
        self.shs[indexes] = other.shs

    def loadPLY(self, path):
        if not os.path.exists(path):
            print(f"[Warning] File {path} does not exist! From loadPLY function in RawGaussian class.")
            return

        plydata = PlyData.read(path)

        xyz = np.stack([np.asarray(plydata.elements[0][i]) for i in ["x", "y", "z"]], axis=1).astype(np.float32)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)
        scales = np.stack([np.asarray(plydata.elements[0][f"scale_{i}"]) for i in range(3)], axis=1).astype(np.float32)
        rots = np.stack([np.asarray(plydata.elements[0][f"rot_{i}"]) for i in range(4)], axis=1).astype(np.float32)
        features_dc = np.stack([np.asarray(plydata.elements[0][f"f_dc_{i}"]) for i in range(3)], axis=1)
        try:
            normals = np.stack([np.asarray(plydata.elements[0][i]) for i in ["nx", "ny", "nz"]], axis=1).astype(np.float32)
        except Exception as e:
            normals = np.zeros_like(xyz)
            # pcd = o3d.geometry.PointCloud()
            # positions = np.asarray(xyz, dtype=np.float64)
            # pcd.points = o3d.utility.Vector3dVector(positions)
            # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            # normals = np.array(pcd.normals)

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        if len(extra_f_names) > 0:
            features_extra = np.stack([np.asarray(plydata.elements[0][name]) for name in extra_f_names], axis=1)
            # features_extra = (
            #     features_extra.reshape((features_extra.shape[0], 3, -1))
            #     .transpose(0, 2, 1)
            #     .reshape(features_extra.shape[0], -1)
            # )

            shs = np.concatenate([features_dc, features_extra], axis=1).astype(np.float32)
        else:
            shs = features_dc

        # rots = rots / np.linalg.norm(rots, axis=-1, keepdims=True)
        # scales = np.exp(scales)
        # opacities = 1/(1 + np.exp(- opacities))  # sigmoid

        self.xyz = xyz
        self.rot = rots
        self.scale = scales
        self.opacity = opacities
        self.shs = shs
        self.normals = normals

        self.ply_path = path
        self.contained_idx = np.ones(len(self), dtype=bool)

        assert len(self.xyz) == len(self.rot) == len(self.scale) == len(self.opacity) == len(self.shs)
        assert len(extra_f_names) in list(map(lambda x: ((x + 1) ** 2 - 1) * 3, [0, 1, 2, 3]))
        
        return self

    def savePLY(self, path, save_empty=False, save_extra=False):
        if not save_empty and len(self) == 0:
            return

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        construct_list_of_attributes = (
            ["x", "y", "z", "nx", "ny", "nz", "opacity"]
            + [f"scale_{i}" for i in range(3)]
            + [f"rot_{i}" for i in range(4)]
            + [f"f_dc_{i}" for i in range(3)]
        )

        f_dc, f_rest = self.shs[:, :3], self.shs[:, 3:]
        # assert f_rest.shape[1] in list(map(lambda x: ((x + 1) ** 2 - 1) * 3, [0, 1, 2, 3]))
        if save_extra:
            construct_list_of_attributes += [f"f_rest_{i}" for i in range(f_rest.shape[1])]

        dtype_full = [(attribute, "f4") for attribute in construct_list_of_attributes]
        elements = np.empty(len(self), dtype=dtype_full)

        normals = self.normals if self.normals else np.zeros_like(self.xyz)
        if save_extra:
            attributes = np.concatenate((self.xyz, normals, self.opacity, self.scale, self.rot, f_dc, f_rest), axis=1)
        else:
            attributes = np.concatenate((self.xyz, normals, self.opacity, self.scale, self.rot, f_dc), axis=1)
        elements[:] = list(map(tuple, attributes))

        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)
