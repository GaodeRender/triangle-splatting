import numpy as np
from plyfile import PlyData, PlyElement
from pathlib import Path
import os
from scipy.spatial import KDTree
from copy import deepcopy
import trimesh

from ..utils.sh_utils import SH2RGB, RGB2SH


class RawTriangle:
    def __init__(
        self,
        vertex: np.ndarray = None,
        opacity: np.ndarray = None,
        shs: np.ndarray = None,
        *,
        ply_path: str = None,
        glb_path: str = None,
    ) -> None:
        self.vertex = vertex
        self.opacity = opacity
        self.shs = shs

        if ply_path is not None:
            self.loadPLY(ply_path)
        if glb_path is not None:
            self.loadGLB(glb_path)

        self.contained_idx = np.ones(len(self), dtype=bool)

    @property
    def center(self):
        return self.vertex.mean(axis=1)

    def printStats(self):
        banner = "=" * 20 + " RawTriangle Stats " + "=" * 20
        print(banner)

        print(f"Number of points: {len(self)}")
        print(f"Number of SHs: {self.shs.shape[1] // 3}")
        print(f"x range: {self.vertex[..., 0].min():>8.1f} - {self.vertex[..., 0].max():>8.1f}")
        print(f"y range: {self.vertex[..., 1].min():>8.1f} - {self.vertex[..., 1].max():>8.1f}")
        print(f"z range: {self.vertex[..., 2].min():>8.1f} - {self.vertex[..., 2].max():>8.1f}")
        print(f"z mean: {self.vertex[..., 2].mean():>8.1f}")
        print(f"z median: {np.median(self.vertex[..., 2]):>8.1f}")

        print("=" * len(banner))

    def shDegree(self):
        # assert self.shs.shape[1] in list(map(lambda x: ((x + 1) ** 2) * 3, [0, 1, 2, 3]))
        return int(np.sqrt(self.shs.shape[1] / 3) - 1)

    def __len__(self):
        return len(self.vertex) if self.vertex is not None else 0

    def __getitem__(self, idx):
        return RawTriangle(
            self.vertex[idx] if self.vertex is not None else None,
            self.opacity[idx] if self.opacity is not None else None,
            self.shs[idx] if self.shs is not None else None,
        )

    def __iadd__(self, other):
        if len(other) == 0:
            return self

        self.vertex = np.concatenate((self.vertex, other.vertex)) if self.vertex is not None else other.vertex
        self.opacity = np.concatenate((self.opacity, other.opacity)) if self.opacity is not None else other.opacity
        self.shs = np.concatenate((self.shs, other.shs)) if self.shs is not None else other.shs

        self.contained_idx = np.concatenate((self.contained_idx, other.contained_idx)) if self.contained_idx is not None else other.contained_idx
        return self

    def resetContainedIdx(self):
        self.contained_idx = np.ones(len(self), dtype=bool)

    def __isub__(self, other):
        if len(other) == 0:
            return self

        kdTree = KDTree(other.center)
        distance, idx = kdTree.query(self.center)
        self.contained_idx &= distance > 1e-5
        self.reduce()
        return self

    def __sub__(self, other):
        diff = deepcopy(self)
        diff -= other
        return diff

    def reduce(self):
        if np.all(self.contained_idx):
            return RawTriangle()

        removed_triangle = RawTriangle(
            self.vertex[~self.contained_idx],
            self.opacity[~self.contained_idx],
            self.shs[~self.contained_idx],
        )

        self.vertex = self.vertex[self.contained_idx]
        self.opacity = self.opacity[self.contained_idx]
        self.shs = self.shs[self.contained_idx]
        self.contained_idx = np.ones(len(self), dtype=bool)

        return removed_triangle

    def replace(self, indices, other):
        # if other length is not equal to removed_triangle, raise error
        if len(indices) != len(other):
            raise ValueError(
                "Length of removed_triangle is not equal to other, length of removed_triangle is {}, length of other is {}".format(
                    len(indices), len(other)
                )
            )

        self.vertex[indices] = other.vertex
        self.opacity[indices] = other.opacity
        self.shs[indices] = other.shs

    def loadPLY(self, path):
        if not os.path.exists(path):
            print(f"[Warning] File {path} does not exist! From loadPLY function in RawTriangle class.")
            return

        self.ply_path = path

        try:
            plydata = PlyData.read(path)
        except Exception as e:
            print(f"Error reading {path}: {e}")
            return

        vertex_properties = ["x1", "y1", "z1", "x2", "y2", "z2", "x3", "y3", "z3"]
        vertex = np.stack([np.asarray(plydata.elements[0][i]) for i in vertex_properties], axis=1).astype(np.float32).reshape(-1, 3, 3)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)
        features_dc = np.stack([np.asarray(plydata.elements[0][f"f_dc_{i}"]) for i in range(3)], axis=1)

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        if len(extra_f_names) > 0:
            features_extra = np.stack([np.asarray(plydata.elements[0][name]) for name in extra_f_names], axis=1)
            shs = np.concatenate([features_dc, features_extra], axis=1).astype(np.float32)
        else:
            shs = features_dc

        self.vertex = vertex
        self.opacity = opacities
        self.shs = shs

        assert len(self.vertex) == len(self.opacity) == len(self.shs)
        assert len(extra_f_names) in list(map(lambda x: ((x + 1) ** 2 - 1) * 3, [0, 1, 2, 3]))
        return self

    def savePLY(self, path, save_empty=False, save_extra=False):
        if not save_empty and len(self) == 0:
            return

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        construct_list_of_attributes = ["x1", "y1", "z1", "x2", "y2", "z2", "x3", "y3", "z3", "opacity"] + [f"f_dc_{i}" for i in range(3)]

        f_dc, f_rest = self.shs[:, :3], self.shs[:, 3:]
        # assert f_rest.shape[1] in list(map(lambda x: ((x + 1) ** 2 - 1) * 3, [0, 1, 2, 3]))
        if save_extra:
            construct_list_of_attributes += [f"f_rest_{i}" for i in range(f_rest.shape[1])]

        dtype_full = [(attribute, "f4") for attribute in construct_list_of_attributes]
        elements = np.empty(len(self), dtype=dtype_full)

        if save_extra:
            attributes = np.concatenate((self.vertex.reshape(-1, 9), self.opacity, f_dc, f_rest), axis=1)
        else:
            attributes = np.concatenate((self.vertex.reshape(-1, 9), self.opacity, f_dc), axis=1)
        elements[:] = list(map(tuple, attributes))

        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def saveGLB(self, path, save_empty=False, save_back=True, process=False):
        if not save_empty and len(self) == 0:
            return

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        triangles = self.vertex
        color = np.clip(SH2RGB(self.shs[:, :3]), 0, 1)
        opacity = 1 / (1 + np.exp(-self.opacity))
        rgba = np.concatenate([color, opacity], axis=1)
        faces = np.arange(len(triangles) * 3).reshape(-1, 3)

        if save_back:
            faces_back = faces[:, ::-1]
            faces = np.concatenate([faces, faces_back], axis=0)
            rgba = np.concatenate([rgba, rgba], axis=0)

        mesh = trimesh.Trimesh(
            vertices=triangles.reshape(-1, 3),
            faces=faces,
            face_colors=rgba,
            # vertex_colors=np.repeat(rgba, 3, axis=0),
            process=process,
        )
        mesh.export(path)

    def loadGLB(self, path):
        if not os.path.exists(path):
            print(f"[Warning] File {path} does not exist! From loadGLB function in RawTriangle class.")

        self.glb_path = path

        mesh = trimesh.load(path).geometry["geometry_0"]
        triangles = np.array(mesh.vertices.reshape(-1, 3, 3))
        rgba = np.array(mesh.visual.face_colors)[: len(triangles)] / 255

        eps = 1e-5
        self.vertex = triangles
        self.opacity = -np.log(1 / np.clip(rgba[:, 3:], eps, 1 - eps) - 1)
        self.shs = RGB2SH(rgba[:, :3])
        return self
