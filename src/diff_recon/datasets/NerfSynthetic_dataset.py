import numpy as np
import json
from pathlib import Path

from .colmap_loader import CameraInfo
from .Colmap_dataset import ColmapDatasetFactory


class NerfSyntheticDatasetFactory(ColmapDatasetFactory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _readCamerasFromTransforms(self, transforms_file: str):
        cam_infos = []
        with open(self._file_handler.getFilePath(transforms_file)) as json_file:
            contents = json.load(json_file)
            FovX = contents["camera_angle_x"]
            frames = contents["frames"]
            for idx, frame in enumerate(frames):
                c2w = np.array(frame["transform_matrix"])  # NeRF 'transform_matrix' is a camera-to-world transform
                c2w[:3, 1:3] *= -1  # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)

                w2c = np.linalg.inv(c2w)
                R = np.transpose(w2c[:3, :3])
                T = w2c[:3, 3]

                cam_infos.append(
                    CameraInfo(
                        camera_id=idx,
                        R=R,
                        T=T,
                        FovY=None,
                        FovX=FovX,
                        image_path=frame["file_path"] + ".png",
                        image_name=Path(frame["file_path"]).stem,
                        width=None,
                        height=None,
                    )
                )

        return cam_infos

    def _getCameraInfos(self) -> tuple[list[CameraInfo], list[CameraInfo]]:
        self._logger.info("Fetching intrinsic and extrinsic data from transforms_train.json and transforms_test.json.")
        train_cam_infos = self._readCamerasFromTransforms("transforms_train.json")
        test_cam_infos = self._readCamerasFromTransforms("transforms_test.json")
        return train_cam_infos, test_cam_infos
