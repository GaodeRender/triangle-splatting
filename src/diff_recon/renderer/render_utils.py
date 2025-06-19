import torch
import numpy as np
import cv2
from tqdm import tqdm

from ..utils.camera import Camera
from ..models.VanillaGS_model import VanillaGSModel


def image_tensor_to_cv(image: torch.Tensor) -> np.ndarray:
    image = image.detach().cpu().numpy().transpose(1, 2, 0)[..., ::-1]
    image = np.ascontiguousarray(image * 255).astype(np.uint8)
    return image


@torch.no_grad()
def render_BEV_image(model: VanillaGSModel, save_path: str = None, img_size=(2160, 1440), center=None) -> torch.Tensor:
    R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    FoVx, FoVy = 0.610, 0.414
    elevation = 1200

    center = model.get_xyz.mean(dim=0).detach().cpu().numpy() if center is None else center
    cam_pos = center + np.array([0, 0, elevation])
    T = R.T @ -cam_pos

    cam = Camera(R, T, FoVx, FoVy, img_size[0], img_size[1]).to(model.device)
    image = model.forward(cam, False, "black")["render"]

    if save_path is not None:
        image = image_tensor_to_cv(image)
        cv2.imwrite(save_path, image)

    return image


def pos_target_to_RT(pos: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = target - pos
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    R = np.array([x, y, z]).T
    T = R.T @ -pos
    return R, T


@torch.no_grad()
def render_tour(model: VanillaGSModel, save_path: str, img_size=(2160, 1440)):
    outer_radius_x = 800
    outer_radius_y = 400
    inner_radius_x = 200
    inner_radius_y = 100
    fps = 30
    duration = 10
    num_cams = fps * duration
    FoVx, FoVy = 0.610, 0.414
    elevation = 400

    center = model.get_xyz.mean(dim=0).detach().cpu().numpy()
    theta = np.linspace(0, 2 * np.pi, num_cams, endpoint=False)
    coord = np.array([np.cos(theta), np.sin(theta), np.zeros_like(theta)]).T
    outer_radius = np.array([outer_radius_x, outer_radius_y, 0])
    inner_radius = np.array([inner_radius_x, inner_radius_y, 0])
    cam_pos = coord * outer_radius + center + np.array([0, 0, elevation])
    target_pos = coord * inner_radius + center

    video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, img_size)
    for i in tqdm(range(num_cams)):
        R, T = pos_target_to_RT(cam_pos[i], target_pos[i])
        cam = Camera(R, T, FoVx, FoVy, img_size[0], img_size[1]).to(model.device)
        image = model.forward(cam, False, "black")["render"]
        image = image_tensor_to_cv(image)
        video.write(image)
    video.release()


@torch.no_grad()
def render_tour_compare(model1: VanillaGSModel, model2: VanillaGSModel, save_path: str, img_size=(2160, 1440), name1="model1", name2="model2"):
    outer_radius_x = 800
    outer_radius_y = 400
    inner_radius_x = 200
    inner_radius_y = 100
    fps = 30
    duration = 10
    num_cams = fps * duration
    FoVx, FoVy = 0.610, 0.414
    elevation = 400

    center = model1.get_xyz.mean(dim=0).detach().cpu().numpy()
    theta = np.linspace(0, 2 * np.pi, num_cams, endpoint=False)
    coord = np.array([np.cos(theta), np.sin(theta), np.zeros_like(theta)]).T
    outer_radius = np.array([outer_radius_x, outer_radius_y, 0])
    inner_radius = np.array([inner_radius_x, inner_radius_y, 0])
    cam_pos = coord * outer_radius + center + np.array([0, 0, elevation])
    target_pos = coord * inner_radius + center

    video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, img_size)
    for i in tqdm(range(num_cams)):
        R, T = pos_target_to_RT(cam_pos[i], target_pos[i])
        cam = Camera(R, T, FoVx, FoVy, img_size[0], img_size[1]).to(model1.device)
        image1 = model1.forward(cam, False, "black")["render"]
        image2 = model2.forward(cam, False, "black")["render"]

        image = torch.cat([image1[..., : img_size[0] // 2], image2[..., img_size[0] // 2 :]], dim=2)
        image = image_tensor_to_cv(image)

        # write name on image
        cv2.putText(image, name1, (img_size[0] // 20, img_size[1] // 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, name2, (img_size[0] // 2 + img_size[0] // 20, img_size[1] // 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # draw separation line
        cv2.line(image, (img_size[0] // 2, 0), (img_size[0] // 2, img_size[1]), (255, 255, 255), 2)
        video.write(image)
    video.release()
