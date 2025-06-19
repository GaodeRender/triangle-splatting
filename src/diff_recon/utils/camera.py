import numpy as np
import torch
import math


def getWorld2ViewMatrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    return Rt.astype(np.float32)


def getProjectionMatrix(znear, zfar, fovX, fovY) -> np.ndarray:
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def qvec2rotmat(q: np.ndarray) -> np.ndarray:
    # quaternion in wxyz format
    return np.array(
        [
            [1 - 2 * q[2] ** 2 - 2 * q[3] ** 2, 2 * q[1] * q[2] - 2 * q[0] * q[3], 2 * q[3] * q[1] + 2 * q[0] * q[2]],
            [2 * q[1] * q[2] + 2 * q[0] * q[3], 1 - 2 * q[1] ** 2 - 2 * q[3] ** 2, 2 * q[2] * q[3] - 2 * q[0] * q[1]],
            [2 * q[3] * q[1] - 2 * q[0] * q[2], 2 * q[2] * q[3] + 2 * q[0] * q[1], 1 - 2 * q[1] ** 2 - 2 * q[2] ** 2],
        ]
    )


def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    # return quaternion in wxyz format
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


class Camera(torch.nn.Module):
    def __init__(
        self,
        R: np.ndarray, # camera to world rotation matrix (3x3)
        T: np.ndarray,
        FoVx: float,
        FoVy: float = None,
        image_width: int = None,
        image_height: int = None,
        gt_image: np.ndarray = None,
        gt_alpha_mask: np.ndarray = None,
        image_name: str = None,
        camera_id: int = None,
        uid: int = None,
    ):
        super().__init__()

        self.uid = uid
        self.camera_id = camera_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.device = torch.device("cpu")

        if gt_image is None and (image_width is None or image_height is None):
            raise ValueError("Either image or image_width and image_height must be provided")

        if gt_image is None and FoVy is None:
            raise ValueError("Either image or FoVy must be provided")

        self.gt_image = torch.tensor(gt_image).float().clamp(0.0, 1.0) if gt_image is not None else None
        self.alpha_mask = torch.tensor(gt_alpha_mask).float() if gt_alpha_mask is not None else None
        self.image_width = image_width if image_width is not None else self.gt_image.shape[2]
        self.image_height = image_height if image_height is not None else self.gt_image.shape[1]
        if self.FoVy is None:
            self.FoVy = math.atan(math.tan(self.FoVx / 2) * (self.image_height / self.image_width)) * 2

        self.zfar = 1000.0
        self.znear = 1.0

        self.world_view_transform = torch.tensor(getWorld2ViewMatrix(R, T)).transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.tan_fovx = math.tan(self.FoVx / 2)
        self.tan_fovy = math.tan(self.FoVy / 2)

    def pin_memory(self):
        self.gt_image = self.gt_image.pin_memory() if self.gt_image is not None else None
        self.alpha_mask = self.alpha_mask.pin_memory() if self.alpha_mask is not None else None
        self.world_view_transform = self.world_view_transform.pin_memory()
        self.projection_matrix = self.projection_matrix.pin_memory()
        self.full_proj_transform = self.full_proj_transform.pin_memory()
        self.camera_center = self.camera_center.pin_memory()
        return self

    def to(self, device: torch.device):
        self.gt_image = self.gt_image.to(device) if self.gt_image is not None else None
        self.alpha_mask = self.alpha_mask.to(device) if self.alpha_mask is not None else None
        self.world_view_transform = self.world_view_transform.to(device)
        self.projection_matrix = self.projection_matrix.to(device)
        self.full_proj_transform = self.full_proj_transform.to(device)
        self.camera_center = self.camera_center.to(device)
        self.device = device
        return self
