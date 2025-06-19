from time import time
import math
import numpy as np
import viser
from viser import ClientHandle
import trimesh
import os
import argparse
import cv2
from scipy.spatial.transform import Rotation as Rot

from src.diff_recon import loadConfig, VanillaTSModel, VanillaGSModel, BaseDatasetFactory, VanillaTSTrainer
from src.diff_recon.utils.camera import Camera, rotmat2qvec, qvec2rotmat
from src.diff_recon.utils.vis_utils import depth_to_image, normal_to_image


class VisClient:
    def __init__(self, client: ClientHandle, dataset: BaseDatasetFactory, model_dict: dict, output_dir: str = "outputs"):
        self._client_handle: ClientHandle = client
        self._dataset = dataset
        self._model_dict = model_dict
        self._mesh_dict = {}
        self._curr_image = None
        self._output_dir = output_dir

        client.scene.enable_default_lights(True)
        self._update = True
        self._terminate = False

        self._register_gui()
        self._set_test_camera()

    def _display_mesh(self, name: str = None, mesh: trimesh.Trimesh = None, refresh: bool = False):
        if name is not None:
            if name not in self._mesh_dict or refresh:
                if name.endswith("_simple"):
                    self._mesh_dict[name] = self._client_handle.scene.add_mesh_simple(
                        name, mesh.vertices, mesh.faces, color=(128, 128, 128), wireframe=True, visible=False
                    )
                else:
                    self._mesh_dict[name] = self._client_handle.scene.add_mesh_trimesh(name, mesh, visible=False)
        for k, v in self._mesh_dict.items():
            v.visible = False if k != name else True

    def _register_gui(self):
        client: ClientHandle = self._client_handle

        @client.camera.on_update
        async def _(_):
            if not self._test_view_ckbx.value:
                self._update = True

        self._test_view_slider = client.gui.add_slider("Test View", 0, self._dataset.getTestDatasetSize() - 1, 1, 0)

        @self._test_view_slider.on_update
        async def _(_):
            if self._test_view_ckbx.value:
                self._set_test_camera()
                self._update = True

        self._test_view_ckbx = client.gui.add_checkbox("Test View", False)

        @self._test_view_ckbx.on_update
        async def _(_):
            self._set_test_camera()
            self._update = True

        self._render_mode_dropdown = client.gui.add_dropdown("Render Mode", ["image", "image_gt", "depth", "normal"], "image")

        @self._render_mode_dropdown.on_update
        async def _(_):
            if self._render_mode_dropdown.value == "image_gt":
                self._test_view_ckbx.value = True
                self._set_test_camera()
                self._test_view_ckbx.disabled = True
            else:
                self._test_view_ckbx.disabled = False
            self._update = True

        self._background_dropdown = client.gui.add_dropdown("Background", ["black", "white"], "black")

        @self._background_dropdown.on_update
        async def _(_):
            self._update = True

        self._resolution_slider = client.gui.add_slider("Resolution", 384, 4096, 1, 1024)

        @self._resolution_slider.on_update
        async def _(_):
            self._update = True

        self._back_culling_ckbx = client.gui.add_checkbox("Back Culling", False)

        @self._back_culling_ckbx.on_update
        async def _(_):
            self._update = True

        self._sh_degree_slider = client.gui.add_slider("SH Degree", 0, 3, 1, 0)

        @self._sh_degree_slider.on_update
        async def _(_):
            self._update = True

        self._gamma_slider = client.gui.add_slider("Gamma", 1.0, 50.0, 0.1, 1.0)

        @self._gamma_slider.on_update
        async def _(_):
            self._update = True

        key_list = list(self._model_dict.keys())
        self._model_dropdown = client.gui.add_dropdown("Model", key_list, key_list[0])

        @self._model_dropdown.on_update
        async def _(_):
            with client.atomic():
                name: str = self._model_dropdown.value
                if name.endswith("_mesh"):
                    self._display_mesh(f"{name}_simple" if self._simple_mesh_ckbx.value else name, self._model_dict[name])
                    client.scene.set_background_image(np.zeros((1, 1, 3), dtype=np.uint8), "jpeg", 85)
                    self._render_mode_dropdown.value = "image"
                    self._render_mode_dropdown.disabled = True
                    self._simple_mesh_ckbx.disabled = False
                else:
                    self._display_mesh(None)
                    self._render_mode_dropdown.disabled = False
                    self._simple_mesh_ckbx.disabled = True
            self._update = True

        self._simple_mesh_ckbx = client.gui.add_checkbox("Wireframe", False, disabled=True)

        @self._simple_mesh_ckbx.on_update
        async def _(_):
            with client.atomic():
                name: str = self._model_dropdown.value
                if name.endswith("_mesh"):
                    self._display_mesh(f"{name}_simple" if self._simple_mesh_ckbx.value else name, self._model_dict[name])

        self._snapshot_btn = client.gui.add_button("Snapshot")

        @self._snapshot_btn.on_click
        async def _(_):
            if self._curr_image is not None:
                file_name = os.path.join(self._output_dir, f"{self._model_dropdown.value}_{self._render_mode_dropdown.value}.jpg")
                cv2.imwrite(file_name, cv2.cvtColor(self._curr_image * 255, cv2.COLOR_RGB2BGR).astype(np.uint8))
                print(f"[INFO] Snapshot saved to {file_name}")

    def _get_camera(self) -> Camera:
        client: ClientHandle = self._client_handle
        fovy = client.camera.fov
        # aspect = self._aspect
        aspect = client.camera.aspect
        resolution = max(self._resolution_slider.value, 384)

        fovx = 2 * math.atan(math.tan(fovy / 2) * aspect)
        qvec = client.camera.wxyz  # from world to camera
        tvec = client.camera.position
        R = qvec2rotmat(qvec)
        T = R.T @ -tvec

        camera = Camera(
            R=R,
            T=T,
            FoVx=fovx,
            FoVy=fovy,
            image_width=int(resolution),
            image_height=int(resolution / aspect),
        )
        return camera

    def _set_camera(self, camera: Camera):
        client: ClientHandle = self._client_handle
        with client.atomic():
            client.camera.wxyz = rotmat2qvec(camera.R)
            client.camera.position = camera.camera_center
            client.camera.fov = camera.FoVy
            client.camera.far = camera.zfar
            client.camera.near = camera.znear
            self._aspect = camera.image_width / camera.image_height

    def _set_test_camera(self):
        self._set_camera(self._dataset.getTestData(self._test_view_slider.value))

    def update_image(self):
        if self._model_dropdown.value.endswith("_mesh"):
            self._update = False
            return

        model: VanillaTSModel | VanillaGSModel = self._model_dict[self._model_dropdown.value]
        camera = self._get_camera().to(model.device)

        back_culling = self._back_culling_ckbx.value
        sh_degree = self._sh_degree_slider.value
        gamma = self._gamma_slider.value
        render_mode = self._render_mode_dropdown.value
        background = self._background_dropdown.value

        start_time = time()
        if render_mode == "image_gt":
            image = self._dataset.getTestData(self._test_view_slider.value).gt_image.permute(1, 2, 0).detach().cpu().numpy()
        elif render_mode == "image":
            output = model.forward(
                camera,
                background,
                is_training=False,
                back_culling=back_culling,
                sh_degree=sh_degree,
                gamma=gamma,
            )
            image = output["render"].permute(1, 2, 0).detach().cpu().numpy()
        elif render_mode == "depth":
            output = model.forward(
                camera,
                background,
                is_training=True,
                back_culling=back_culling,
                sh_degree=sh_degree,
                gamma=gamma,
            )
            image = output["depth"].detach().cpu().numpy()
            image = depth_to_image(image)
        elif render_mode == "normal":
            output = model.forward(
                camera,
                background,
                is_training=True,
                back_culling=back_culling,
                sh_degree=sh_degree,
                gamma=gamma,
            )
            image = output["normal"].permute(1, 2, 0).detach().cpu().numpy()
            image = image / np.linalg.norm(image, axis=-1, keepdims=True)
            image = normal_to_image(image)
        else:
            raise ValueError(f"Unknown render mode: {render_mode}")
        rendered_time = time()
        print(f"[INFO] Rendered {image.shape} {image.dtype}, time: {rendered_time-start_time:.3f}s")

        client: ClientHandle = self._client_handle
        with client.atomic():
            self._curr_image = image
            client.scene.set_background_image(image, "jpeg", 85)
        sent_time = time()
        print(f"[INFO] Sent Image {image.shape} {image.dtype}, time: {sent_time-rendered_time:.3f}s")

        self._update = False

    def need_update(self):
        return self._update

    def terminate(self):
        self._terminate = True

    def need_terminate(self):
        return self._terminate


class VisViewer:
    def __init__(self, dataset: BaseDatasetFactory, model_dict):
        self._server = viser.ViserServer()
        self._dataset = dataset
        self._model_dict = model_dict

        self._client_dict: dict[VisClient] = {}
        self._close_viewer = False

        self._register_gui()

    def _register_gui(self):
        close_viewer_btn = self._server.gui.add_button("Close Viewer")

        @close_viewer_btn.on_click
        async def _(_):
            self._close_viewer = True

        @self._server.on_client_connect
        async def _(client: ClientHandle):
            self._client_dict[client.client_id] = VisClient(client, self._dataset, self._model_dict)

        @self._server.on_client_disconnect
        async def _(client: ClientHandle):
            self._client_dict[client.client_id].terminate()

    def _step(self):
        try:
            for client_id in list(self._client_dict.keys()):
                client = self._client_dict[client_id]
                if client.need_terminate():
                    del self._client_dict[client_id]
                    continue
                if client.need_update():
                    client.update_image()
        except Exception as e:
            print(f"[ERROR] {e}")

    def run(self):
        while not self._close_viewer:
            self._step()
        self._server.stop()


def load_mesh(file_name: str) -> trimesh.Trimesh:
    file_type = file_name.split(".")[-1]
    if file_type == "ply":
        mesh = trimesh.load_mesh(file_name, process=False)
    elif file_type == "glb":
        mesh = trimesh.load_mesh(file_name, process=False)
    elif file_type == "obj":
        mesh = trimesh.load_mesh(file_name, process=False)
        R = Rot.from_euler("x", 90, degrees=True)
        mesh.vertices = R.apply(mesh.vertices)
    else:
        raise ValueError(f"Unknown file type: {file_type}")
    return mesh


def run_VisViewer(config_path: str, dataset_path: str, scene_id: str):
    config = loadConfig(config_path)
    if dataset_path:
        config.dataset.local_dir = dataset_path
    config.dataset.scene_id = scene_id
    config.dataset.train_target_res = 1
    config.dataset.test_target_res = 1
    trainer = VanillaTSTrainer(config, exp_name=scene_id)

    ply_files = os.listdir(os.path.join(trainer.output_dir, "point_cloud")) if os.path.exists(os.path.join(trainer.output_dir, "point_cloud")) else []
    glb_files = os.listdir(os.path.join(trainer.output_dir, "glb")) if os.path.exists(os.path.join(trainer.output_dir, "glb")) else []
    gs_files = (
        os.listdir(os.path.join(trainer.output_dir, "gs_point_cloud")) if os.path.exists(os.path.join(trainer.output_dir, "gs_point_cloud")) else []
    )

    model_dict = {}
    for ply_file in ply_files:
        name = ply_file.split(".")[0]
        model_dict[name] = VanillaTSModel(config.model).loadPLY(os.path.join(trainer.output_dir, "point_cloud", ply_file))
    for glb_file in glb_files:
        name = glb_file.split(".")[0]
        model_dict[f"{name}_mesh"] = load_mesh(os.path.join(trainer.output_dir, "glb", glb_file))
    for gs_file in gs_files:
        name = gs_file.split(".")[0]
        model_dict[f"{name}_gs"] = VanillaGSModel(config.model).loadPLY(os.path.join(trainer.output_dir, "gs_point_cloud", gs_file))

    viewer = VisViewer(trainer.dataset, model_dict)
    viewer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Viser Viewer")
    parser.add_argument("--config", type=str, help="Path to the config file")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset", default="")
    parser.add_argument("--scene", type=str, help="Scene ID")
    args = parser.parse_args()

    run_VisViewer(args.config, args.dataset_path, args.scene)
