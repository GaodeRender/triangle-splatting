from .Colmap_dataset import ColmapDatasetFactory
from .colmap_loader import CameraInfo, readColmapCameras


class MatrixCityDatasetFactory(ColmapDatasetFactory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _getCameraInfos(self) -> tuple[list[CameraInfo], list[CameraInfo]]:
        fs = self._file_handler
        cam_infos_all = []
        for root_dir in ["train/block_all", "test/block_all_test"]:
            images_txt_path = f"{root_dir}/sparse/images.txt"
            cameras_txt_path = f"{root_dir}/sparse/cameras.txt"
            images_folder = f"{root_dir}/input"

            if fs.hasFile(images_txt_path):
                images_path = fs.getFilePath(images_txt_path)
                self._logger.info(f"Fetching extrinsics data from {images_txt_path}.")
            else:
                raise FileNotFoundError(f"Cannot find {images_txt_path}")

            if fs.hasFile(cameras_txt_path):
                cameras_path = fs.getFilePath(cameras_txt_path)
                self._logger.info(f"Fetching intrinsics data from {cameras_txt_path}.")
            else:
                raise FileNotFoundError(f"Cannot find {cameras_txt_path}")

            cam_infos = readColmapCameras(images_path, cameras_path, images_folder)
            cam_infos = sorted(cam_infos, key=lambda x: x.image_name)
            cam_infos_all.append(cam_infos)
        return tuple(cam_infos_all)
