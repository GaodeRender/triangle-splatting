import numpy as np
from plyfile import PlyData, PlyElement
from pathlib import Path
# import open3d as o3d

class PointCloud:
    def __init__(self, points: np.array = None, colors: np.array = None, normals: np.array = None):
        self.points = points
        self.colors = colors
        self.normals = normals

        if points is not None and normals is None:
            self.normals = np.zeros_like(points)

    def fetchPly(self, ply_path):
        plydata = PlyData.read(ply_path)
        vertices = plydata["vertex"]
        positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
        try:
            colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
        except:
            colors = np.random.rand(positions.shape[0], positions.shape[1])
        try:
            normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
        except:
            normals = np.zeros_like(positions)
            # pcd = o3d.geometry.PointCloud()
            # positions = np.asarray(positions, dtype=np.float64)
            # pcd.points = o3d.utility.Vector3dVector(positions)
            # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            # normals = np.array(pcd.normals)

        self.points = positions
        self.colors = colors
        self.normals = normals
        return self

    def storePly(self, ply_path):
        Path(ply_path).parent.mkdir(parents=True, exist_ok=True)

        xyz = self.points
        rgb = self.colors
        normals = self.normals
        # Define the dtype for the structured array
        dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]

        elements = np.empty(xyz.shape[0], dtype=dtype)
        attributes = np.concatenate((xyz, normals, rgb), axis=1)
        elements[:] = list(map(tuple, attributes))

        # Create the PlyData object and write to file
        vertex_element = PlyElement.describe(elements, "vertex")
        ply_data = PlyData([vertex_element])
        ply_data.write(ply_path)
    
    def __len__(self):
        return self.points.shape[0] if self.points is not None else 0

    def __iadd__(self, other):
        if len(other) == 0:
            return self
        
        self.points = np.concatenate((self.points, other.points)) if self.points is not None else other.points
        self.colors = np.concatenate((self.colors, other.colors)) if self.colors is not None else other.colors
        self.normals = np.concatenate((self.normals, other.normals)) if self.normals is not None else other.normals
        return self