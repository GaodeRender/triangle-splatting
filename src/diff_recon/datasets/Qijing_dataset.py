import numpy as np
import json
import math
from shapely.geometry import Polygon

from .colmap_loader import CameraInfo, readColmapCameras
from .Colmap_dataset import ColmapDatasetFactory
from ..models.raw_gaussian import RawGaussian
from ..utils.file_handler import OSSHandler


class QijingDatasetFactory(ColmapDatasetFactory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_file_handler(self) -> OSSHandler:
        oss_path = self._config.oss_path
        scene_id = self._config.scene_id
        local_dir = self._config.local_dir
        skip_exist = self._config.skip_exist

        return OSSHandler(f"{oss_path}/{scene_id}", f"{local_dir}/{scene_id}", get_skip_exist=skip_exist)

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

        train_cam_infos, test_cam_infos = self._split_train_test_views(cam_infos)
        return train_cam_infos, test_cam_infos

    def _split_train_test_views(self, cam_infos: list[CameraInfo]) -> tuple[list[CameraInfo], list[CameraInfo]]:
        """
        select views covered by the tile for testing
        """
        n_test = self._config.n_test
        scene_info = self.getSceneInfo()
        x_min, y_min, x_max, y_max = scene_info["bbox_xyz"]
        ground_z = scene_info["ground_z"]
        tile_polygon = Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])

        inside_idx = []
        for i, cam_info in enumerate(cam_infos):
            R, T = cam_info.R, cam_info.T
            cam_center = R @ (-T)
            nx = math.tan(cam_info.FovX / 2)
            ny = math.tan(cam_info.FovY / 2)
            four_corners_ray = np.array([[nx, ny, 1], [nx, -ny, 1], [-nx, -ny, 1], [-nx, ny, 1]])
            four_corners_ray = four_corners_ray / np.linalg.norm(four_corners_ray, axis=1, keepdims=True)
            four_corners_ray_world = R @ four_corners_ray.T
            t = (ground_z - cam_center[2]) / four_corners_ray_world[2]
            intersection = cam_center[:, None] + t * four_corners_ray_world

            proj_polygon = Polygon(intersection[:2].T)
            if tile_polygon.contains(proj_polygon):
                inside_idx.append(i)

        test_idx = inside_idx[: len(inside_idx) // n_test * n_test : len(inside_idx) // n_test] if len(inside_idx) > n_test else inside_idx
        # test_idx = sorted(np.random.choice(inside_idx, n_test, replace=False).tolist()) if len(inside_idx) > n_test else inside_idx
        train_cam_infos = [c for i, c in enumerate(cam_infos) if i not in test_idx]
        test_cam_infos = [c for i, c in enumerate(cam_infos) if i in test_idx]
        return train_cam_infos, test_cam_infos

    def _getSceneInfo(self) -> dict:
        fs = self._file_handler
        scene_info_path = "tile_bbox.json"

        if fs.hasFile(scene_info_path):
            with open(fs.getFilePath(scene_info_path), "r") as f:
                scene_info = json.load(f)
            return scene_info
        else:
            return None

    def getSceneInfo(self) -> dict:
        if not hasattr(self, "_scene_info") or self._scene_info is None:
            self._scene_info = self._getSceneInfo()
        return self._scene_info

    def _getGTGaussian(self) -> RawGaussian:
        fs = self._file_handler
        scene_id = self._config.scene_id
        gt_gaussian_path = f"models_z18/{scene_id}.ply"

        if fs.hasFile(gt_gaussian_path):
            gt_gaussian = RawGaussian(ply_path=fs.getFilePath(gt_gaussian_path))
            return gt_gaussian
        else:
            return None

    def getGTGaussian(self) -> RawGaussian:
        if not hasattr(self, "_gt_gaussian") or self._gt_gaussian is None:
            self._gt_gaussian = self._getGTGaussian()
        return self._gt_gaussian
