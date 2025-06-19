import os
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

from .colmap_loader import read_points3D_binary, CameraInfo, readColmapCameras
from .Base_dataset import BaseDatasetFactory
from ..models.point_cloud import PointCloud
from ..models.raw_gaussian import RawGaussian
from ..utils.config import Config
from ..utils.logger import Logger
from ..utils.file_handler import LocalHandler, BaseFileHandler
from ..utils.camera import Camera, getWorld2ViewMatrix
from ..utils.sh_utils import SH2RGB


def solve_target_res(target_res: int | list[int] | None, orig_w: int, orig_h: int) -> tuple[int, int]:
    w, h = orig_w, orig_h

    if target_res is None:
        if w >= h and w > 1600:
            w, h = 1600, 1600 * orig_h // orig_w
        elif w < h and h > 1600:
            w, h = 1600 * orig_w // orig_h, 1600
    elif isinstance(target_res, int):
        if target_res <= 0:
            target_res = 1
        w, h = orig_w // target_res, orig_h // target_res
    elif isinstance(target_res, list):
        w, h = target_res
    else:
        raise ValueError("target_res must be either an int of scale ratio or a list of [width, height]")

    return w, h


class ColmapDataset(Dataset):
    def __init__(
        self,
        file_handler: BaseFileHandler,
        cam_infos: list[CameraInfo],
        target_res: int | list[int] = None,
        background: str = None,
        use_alpha_mask: bool = True,
    ):
        super().__init__()
        self.file_handler = file_handler
        self.cam_infos = cam_infos
        self.target_res = target_res
        self.use_alpha_mask = use_alpha_mask

        if background is None:
            self.bg_color = None
        elif background == "white":
            self.bg_color = np.array([1, 1, 1])
        elif background == "black":
            self.bg_color = np.array([0, 0, 0])
        else:
            raise ValueError("dataset background must be either 'white', 'black' or None")

    def _get_image(self, image_path: str) -> np.ndarray:
        """
        Load, resize and normalize the image
        :param image_path: Path to the image
        :param img_size: Target size (W, H)
        :return: Normalized image as a numpy array of shape (3, H, W) or (4, H, W)
        """
        image = Image.open(self.file_handler.getFilePath(image_path))
        img_size = solve_target_res(self.target_res, image.width, image.height)
        image = image.resize(img_size, Image.Resampling.BILINEAR)
        image_array = np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        image.close()
        return image_array

    def __len__(self):
        return len(self.cam_infos)

    def __getitem__(self, idx) -> Camera:
        cam_info: CameraInfo = self.cam_infos[idx]

        gt_image_array = self._get_image(cam_info.image_path)

        if gt_image_array.shape[0] == 4:
            gt_alpha_mask = gt_image_array[3]
            gt_image_array = gt_image_array[:3]
            if self.bg_color is not None:
                gt_image_array = gt_image_array * gt_alpha_mask + self.bg_color.reshape(3, 1, 1) * (1 - gt_alpha_mask)
        else:
            gt_alpha_mask = None

        camera = Camera(
            R=cam_info.R,
            T=cam_info.T,
            FoVx=cam_info.FovX,
            FoVy=cam_info.FovY,
            gt_image=gt_image_array,
            gt_alpha_mask=gt_alpha_mask if self.use_alpha_mask else None,
            image_name=cam_info.image_name,
            camera_id=cam_info.camera_id,
            uid=idx,
        )
        return camera


def getCameraExtent(cam_infos: list) -> float:
    cam_centers = []
    for cam in cam_infos:
        W2C = getWorld2ViewMatrix(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3])
    cam_centers = np.stack(cam_centers, axis=0)

    center = cam_centers.mean(axis=0, keepdims=True)
    extent = np.linalg.norm(cam_centers - center, axis=1).max() * 1.1

    return extent


class ColmapDatasetFactory(BaseDatasetFactory):
    def __init__(self, config: Config = None, logger: Logger = None):
        super().__init__(config, logger)
        train_target_res = config.train_target_res
        test_target_res = config.test_target_res
        hold_test_set = config.hold_test_set
        background = config.background
        use_alpha_mask = config.use_alpha_mask

        self._file_handler = self._get_file_handler()

        train_cam_infos, test_cam_infos = self.getCameraInfos()
        if not hold_test_set:
            train_cam_infos += test_cam_infos
            self._logger.info(f"hold_test_set not set, will merge test set into train set")
        self._logger.info(f"Train set size: {len(train_cam_infos)}, Test set size: {len(test_cam_infos)}")

        self.cameras_extent = getCameraExtent(train_cam_infos)
        self._logger.info(f"Camera extent: {self.cameras_extent:.2f}")

        self._train_dataset = ColmapDataset(self._file_handler, train_cam_infos, train_target_res, background, use_alpha_mask)
        self._test_dataset = ColmapDataset(self._file_handler, test_cam_infos, test_target_res, background, use_alpha_mask)

        train_cam = self._train_dataset[0]
        test_cam = self._test_dataset[0]
        self._logger.info(
            f"Train resolution: {train_cam.image_width}x{train_cam.image_height}, Test resolution: {test_cam.image_width}x{test_cam.image_height}"
        )
        self._logger.info(f"Using alpha mask: {train_cam.alpha_mask is not None}.")

    def _get_file_handler(self) -> BaseFileHandler:
        if self._config.local_dir is None:
            raise ValueError("local_dir must be set in the config")
        dataset_path = os.path.join(self._config.local_dir, self._config.scene_id) if self._config.scene_id else self._config.local_dir
        return LocalHandler(dataset_path)

    def _getCameraInfos(self) -> tuple[list[CameraInfo], list[CameraInfo]]:
        fs = self._file_handler
        images_bin_path = "sparse/0/images.bin"
        images_txt_path = "sparse/0/images.txt"
        cameras_bin_path = "sparse/0/cameras.bin"
        cameras_txt_path = "sparse/0/cameras.txt"
        images_folder = "images"

        if fs.hasFile(images_bin_path):
            images_path = fs.getFilePath(images_bin_path)
            self._logger.info(f"Fetching extrinsics data from {images_bin_path}.")
        elif fs.hasFile(images_txt_path):
            images_path = fs.getFilePath(images_txt_path)
            self._logger.info(f"Fetching extrinsics data from {images_txt_path}.")
        else:
            raise FileNotFoundError(f"Cannot find {images_bin_path} or {images_txt_path}")

        if fs.hasFile(cameras_bin_path):
            cameras_path = fs.getFilePath(cameras_bin_path)
            self._logger.info(f"Fetching intrinsics data from {cameras_bin_path}.")
        elif fs.hasFile(cameras_txt_path):
            cameras_path = fs.getFilePath(cameras_txt_path)
            self._logger.info(f"Fetching intrinsics data from {cameras_txt_path}.")
        else:
            raise FileNotFoundError(f"Cannot find {cameras_bin_path} or {cameras_txt_path}")

        cam_infos = readColmapCameras(images_path, cameras_path, images_folder)
        cam_infos = sorted(cam_infos, key=lambda x: x.image_name)

        hold_interval = self._config.hold_interval if self._config.hold_interval is not None else 8
        train_cam_infos = [cam for i, cam in enumerate(cam_infos) if i % hold_interval != 0]
        test_cam_infos = [cam for i, cam in enumerate(cam_infos) if i % hold_interval == 0]
        return train_cam_infos, test_cam_infos

    def _getPointCloud(self) -> PointCloud:
        fs = self._file_handler
        pcd_path = self._config.pcd_path

        if pcd_path is None:
            return PointCloud()

        pcd_path = fs.getFilePath(pcd_path)
        self._logger.info(f"Fetching point cloud data from {pcd_path}.")

        if pcd_path.endswith(".bin"):
            xyz, rgb, _ = read_points3D_binary(pcd_path)
            pcd = PointCloud(xyz, rgb)
        elif pcd_path.endswith(".ply"):
            try:
                raw_gs = RawGaussian(ply_path=pcd_path)
                pcd = PointCloud()
                pcd.points = raw_gs.xyz
                pcd.colors = SH2RGB(raw_gs.shs[:, :3])
                pcd.normals = raw_gs.normals
            except Exception as e:
                pcd = PointCloud().fetchPly(pcd_path)
        else:
            raise ValueError(f"Unsupported point cloud file format: {pcd_path.split('.')[-1]}")

        return pcd

    def getCameraInfos(self) -> tuple[list[CameraInfo], list[CameraInfo]]:
        if not hasattr(self, "_cam_infos") or self._cam_infos is None:
            self._cam_infos = self._getCameraInfos()
        return self._cam_infos

    def getPointCloud(self) -> PointCloud:
        if not hasattr(self, "_point_cloud") or self._point_cloud is None:
            self._point_cloud = self._getPointCloud()
        return self._point_cloud

    def getSceneInfo(self) -> dict:
        return None
