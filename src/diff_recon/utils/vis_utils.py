from shapely.geometry import Polygon
import trimesh
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from pathlib import Path

from simple_knn import distCUDA2


def plot_camera(proj_polygon: Polygon, tile_polygon: Polygon, cam_center: np.ndarray = None, name: str = "cam_polygon") -> None:
    fig = plt.figure()

    proj_x, proj_y = proj_polygon.exterior.xy
    tile_x, tile_y = tile_polygon.exterior.xy

    proj_x, proj_y = -np.array(proj_x), -np.array(proj_y)
    tile_x, tile_y = -np.array(tile_x), -np.array(tile_y)
    cam_center = -cam_center

    plt.plot(proj_x, proj_y, label="Camera", color="orange")
    plt.plot(tile_x, tile_y, label="Tile", color="blue")

    if cam_center is not None:
        plt.scatter(cam_center[0], cam_center[1], color="red", label="Camera Center")
        # connect camera center to the projection polygon edges
        for i in range(len(proj_x)):
            plt.plot([cam_center[0], proj_x[i]], [cam_center[1], proj_y[i]], color="orange", linestyle="--")

    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend()
    plt.title(name)
    plt.savefig(f"outputs/{name}.png")


def camera_to_mesh(w2cs: np.ndarray, fovxs: np.ndarray, fovys: np.ndarray, name: str = "camera_mesh") -> None:
    """
    Visualize camera as a mesh.
    Args:
        w2cs: (N, 4, 4) camera extrinsics matrix
        fovxs: (N,) horizontal field of view in radians
        fovys: (N,) vertical field of view in radians
        name: name of the output mesh file
    """
    c2ws = np.linalg.inv(w2cs)
    centers = c2ws[:, :3, 3]
    mean_distance = distCUDA2(torch.tensor(centers).float().cuda()).clamp_(min=1e-10).sqrt().mean().cpu().numpy()
    print(f"Mean distance between camera centers: {mean_distance:.3f} m")
    cam_size = mean_distance * 0.2

    tan_fovx = np.tan(fovxs / 2)
    tan_fovy = np.tan(fovys / 2)

    right = c2ws[:, :3, 0]
    up = -c2ws[:, :3, 1]
    forward = c2ws[:, :3, 2]

    v0 = centers
    v1 = centers + cam_size * (right * tan_fovx[:, None] + up * tan_fovy[:, None] + forward)
    v2 = centers + cam_size * (right * tan_fovx[:, None] - up * tan_fovy[:, None] + forward)
    v3 = centers + cam_size * (-right * tan_fovx[:, None] - up * tan_fovy[:, None] + forward)
    v4 = centers + cam_size * (-right * tan_fovx[:, None] + up * tan_fovy[:, None] + forward)
    vertices = np.stack([v0, v1, v2, v3, v4], axis=1)  # (N, 5, 3)
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1], [1, 2, 3], [1, 3, 4]])  # (6, 3)
    faces = np.tile(faces[None, :, :], (len(c2ws), 1, 1))  # (N, 6, 3)
    faces += np.arange(len(c2ws))[:, None, None] * 5  # shift faces for each camera
    face_colors = np.array(
        [
            [0.0, 0.0, 1.0], # blue for right surface
            [0.5, 0.5, 0.5], # gray for down surface
            [0.5, 0.5, 0.5], # gray for left surface
            [1.0, 0.0, 0.0], # red for up surface
            [0.5, 0.5, 0.5], # gray for front surface
            [0.5, 0.5, 0.5], # gray for front surface
        ]
    )  # (6, 3)
    face_colors = np.tile(face_colors[None, :, :], (len(c2ws), 1, 1))  # (N, 6, 3)
    mesh = trimesh.Trimesh(vertices=vertices.reshape(-1, 3), faces=faces.reshape(-1, 3), face_colors=face_colors.reshape(-1, 3), process=False)
    mesh.export(f"outputs/{name}.ply")


def save_image_tensor(image: torch.Tensor, path: str) -> None:
    """
    Args:
        image: tensor with shape (3, H, W)
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    image = image.detach().cpu().numpy().transpose(1, 2, 0) * 255
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).astype(np.uint8)
    cv2.imwrite(path, image)


def depth_to_image(depth: np.ndarray) -> np.ndarray:
    """
    Convert a depth array to a grayscale image.
    Args:
        depth: array with shape (H, W)
    Returns:
        image: array with shape (H, W)
    """
    image = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    return image


def normal_to_image(normal: np.ndarray) -> np.ndarray:
    """
    Convert a normal array to a grayscale image.
    Args:
        normal: array with shape (H, W, 3)
    Returns:
        image: array with shape (H, W)
    """
    normal = (normal + 1) / 2
    image = cv2.normalize(normal, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return image
