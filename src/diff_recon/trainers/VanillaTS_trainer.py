import numpy as np
import torch
from tqdm import tqdm

from .Base_trainer import BaseTrainer
from .trainer_utils import *
from ..datasets.Colmap_dataset import ColmapDatasetFactory
from ..models.VanillaTS_model import VanillaTSModel
from ..utils.timer import Timer
from ..utils.config import Config
from ..utils.camera import Camera


class VanillaTSTrainer(BaseTrainer):
    def __init__(self, config: str | Config = None, exp_name: str = None, device: torch.device | int = None, log_file: bool = True) -> None:
        super().__init__(config, exp_name, device, log_file)

        # Initialize model
        self.model = VanillaTSModel(self.config.model, logger=self.logger, device=self.device)
        self.model.setup_color_affine(self.dataset.getTrainDatasetSize())
        self.model.setup_scene_info(self.dataset.getSceneInfo())

        # Initialize loss module
        self.ssimLoss = ssimLoss.to(self.device)
        self.lpips = lpips.to(self.device)
        if self.config.trainer.w_dog > 0:
            self.dogLoss = dogLoss.to(self.device)
        if self.config.trainer.w_smoothness > 0:
            self.smoothnessLoss = smoothnessLoss.to(self.device)
        if self.config.trainer.geometry_loss and self.config.trainer.geometry_loss.w_geometry > 0:
            self.depthNormalLoss = DepthNormalLoss(scale_factor=self.config.trainer.geometry_loss.scale_factor)
        self._nearest_indices_cache = None

        # For logging and profiling
        test_img_count = self.dataset.getTestDatasetSize()
        eval_save_img_count = self.config.trainer.eval_save_img_count if self.config.trainer.eval_save_img_count is not None else 3
        if test_img_count < eval_save_img_count:
            self._save_img_idx = list(range(test_img_count))
        else:
            self._save_img_idx = sorted(np.random.choice(test_img_count, eval_save_img_count, replace=False).tolist())
        self._tb_gt_recorded = False

    def _get_loss(self, iteration: int, render_pkg: dict, cam: Camera) -> torch.Tensor:
        image = render_pkg["render"]
        scaling = render_pkg["scaling"]
        opacity = render_pkg["opacity"]
        vertex = render_pkg["vertex"]
        gt_image = render_pkg["gt_image"]
        gt_mask = render_pkg["gt_mask"]
        depth = render_pkg["depth"]
        normal = render_pkg["normal"]

        config = self.config.trainer
        w_ssim = config.w_ssim
        w_dog = config.w_dog
        w_smoothness = config.w_smoothness
        w_s_reg = config.w_scaling_reg
        w_a_reg = config.w_affine_reg

        v_start_iter = config.vertex_reg.start_iter
        v_interval_iter = config.vertex_reg.interval_iter
        w_v_reg = config.vertex_reg.w_vertex_reg if iteration > v_start_iter else 0

        g_start_iter = config.geometry_loss.start_iter
        w_geometry = config.geometry_loss.w_geometry if iteration > g_start_iter else 0

        w_o_reg_quad = config.w_opacity_reg.quad_reg
        w_o_reg_linear = config.w_opacity_reg.linear_reg
        o_quad_start_iter = config.w_opacity_reg.quad_start_iter
        o_linear_start_iter = config.w_opacity_reg.linear_start_iter

        w_L1 = 1.0 - w_ssim - w_dog - w_smoothness
        assert w_L1 >= 0

        if gt_mask is not None:
            gt_image = gt_image * gt_mask
            image = image * gt_mask

        # pixelwise loss
        l1_loss = L1(image, gt_image) if w_L1 > 0 else 0
        ssim_loss = self.ssimLoss(image, gt_image) if w_ssim > 0 else 0
        dog_loss = self.dogLoss(image, gt_image) if w_dog > 0 else 0
        smoothness_loss = self.smoothnessLoss(image, gt_image) if w_smoothness > 0 else 0
        geometry_loss = self.depthNormalLoss(depth, normal, cam.tan_fovx, cam.tan_fovy) if w_geometry > 0 else 0

        # regularization loss
        scaling_reg = scaling.mean()

        if iteration <= o_quad_start_iter:
            opacity_reg = 0
            w_o_reg = 0
        elif iteration <= o_linear_start_iter:
            opacity_reg = (0.25 - (opacity - 0.5) ** 2).mean()
            w_o_reg = w_o_reg_quad
        else:
            opacity_reg = (1 - opacity).mean()
            w_o_reg = w_o_reg_linear

        if "render_original" in render_pkg:
            image_original = render_pkg["render_original"]
            if gt_mask is not None:
                image_original = image_original * gt_mask
            affine_reg = L1(image, image_original)
        else:
            affine_reg = 0

        if w_v_reg > 0 and ((iteration - 1) % v_interval_iter == 0 or self._nearest_indices_cache is None):
            self._nearest_indices_cache = nearest_neighbor(vertex.view(-1, 3), 3)
        vertex_reg = nearest_dist2(vertex.view(-1, 3), self._nearest_indices_cache).mean() if w_v_reg > 0 else 0

        img_loss = w_L1 * l1_loss + w_ssim * ssim_loss + w_dog * dog_loss + w_smoothness * smoothness_loss + w_geometry * geometry_loss
        reg_loss = w_s_reg * scaling_reg + w_o_reg * opacity_reg + w_a_reg * affine_reg + w_v_reg * vertex_reg

        loss = img_loss + reg_loss
        render_pkg["geometry_loss"] = geometry_loss
        render_pkg["vertex_loss"] = vertex_reg
        return loss

    def _optimize(self, iteration: int):
        self.model.update_learning_rate(iteration)
        self.model.optimizer.step()
        self.model.optimizer.zero_grad(set_to_none=True)

    def _log_stats(self, log_pkg: dict):
        iteration = log_pkg["iteration"]
        loss = log_pkg["loss"]
        geometry_loss = log_pkg["geometry_loss"]
        vertex_loss = log_pkg["vertex_loss"]
        triangle_count = log_pkg["triangle_count"]
        gamma = log_pkg["gamma"]
        sh_degree = log_pkg["sh_degree"]
        time_elapsed = log_pkg["time_elapsed"]

        self.logger.info(f"[ITER {iteration}] Loss: {loss:.5f}, Triangle Count: {triangle_count}, Gamma: {gamma:.5f}, SH Degree: {sh_degree}")

        self.logger.add_scalar("Loss", loss, iteration)
        self.logger.add_scalar("Geometry Loss", geometry_loss, iteration)
        self.logger.add_scalar("Vertex Loss", vertex_loss, iteration)
        self.logger.add_scalar("Triangle Count", triangle_count, iteration)
        self.logger.add_scalar("Training Time (min)", time_elapsed / 60, iteration)

    @torch.no_grad()
    def _histogram(self, iteration: int):
        opacity = self.model.get_opacity.view(-1)
        scaling = self.model.get_scaling.view(-1)

        # downsample to reduce tensorboard log size
        sample_num = self.config.trainer.histogram_sample_num if self.config.trainer.histogram_sample_num is not None else 10_000
        opacity = opacity[torch.randperm(opacity.shape[0])[:sample_num]]
        scaling = scaling[torch.randperm(scaling.shape[0])[:sample_num]]

        self.logger.add_histogram("Opacity Histogram", opacity, iteration)
        self.logger.add_histogram("Scaling Histogram", scaling, iteration)

    @torch.no_grad()
    def _evaluate(self, iteration: int, use_tensorboard: bool = True):
        self.logger.debug("Evaluation started")

        background = self.config.trainer.eval_background if self.config.trainer.eval_background is not None else "black"
        eval_alpha_mask = self.config.trainer.eval_alpha_mask if self.config.trainer.eval_alpha_mask is not None else True

        psnr_vals = []
        ssim_vals = []
        lpips_vals = []
        for i, test_data in enumerate(self.dataset.getTestDataset()):
            camera = test_data.to(self.device)
            image = self.model.forward(camera, background, False, False)["render"]
            gt_image = camera.gt_image
            psnr_vals.append(psnr(image, gt_image, camera.alpha_mask if eval_alpha_mask else None).item())
            ssim_vals.append(1.0 - self.ssimLoss(image, gt_image).item())
            lpips_vals.append(self.lpips(image.unsqueeze(0).clip(min=0, max=1), gt_image.unsqueeze(0)).item())

            if use_tensorboard and i in self._save_img_idx:
                img_log_idx = self._save_img_idx.index(i)
                self.logger.add_image(f"Pred {img_log_idx}", image, iteration)
                if not self._tb_gt_recorded:
                    self.logger.add_image(f"GT {img_log_idx}", gt_image, 0)

        if use_tensorboard:
            self._tb_gt_recorded = True
            self.logger.add_scalar(f"Average PSNR", np.mean(psnr_vals), iteration)
            self.logger.add_scalar(f"Average SSIM", np.mean(ssim_vals), iteration)
            self.logger.add_scalar(f"Average LPIPS", np.mean(lpips_vals), iteration)

        self.logger.info(
            f"[ITER {iteration}] Evaluation PSNR: {np.mean(psnr_vals):.3f}, SSIM: {np.mean(ssim_vals):.3f}, LPIPS: {np.mean(lpips_vals):.3f}, "
            + f"eval view count: {len(psnr_vals)}, triangle count: {self.model.get_vertex.shape[0]}"
        )

        self.logger.debug("Evaluation finished")

    def _train(self):
        dataset: ColmapDatasetFactory = self.dataset
        config = self.config.trainer

        # Initialize model
        first_iter = 0
        if config.start_checkpoint:
            self.logger.info(f"Initializing Triangles from checkpoint {config.start_checkpoint}.ckpt")
            self.model.load_ckpt(f"{self.output_dir}/ckpt/{config.start_checkpoint}.ckpt")
            first_iter = config.start_checkpoint
        elif config.start_pointcloud:
            self.logger.info(f"Initializing Triangles from ply {config.start_pointcloud}.ply")
            self.model.loadPLY(f"{self.output_dir}/point_cloud/{config.start_pointcloud}.ply")
            first_iter = config.start_pointcloud
        if not self.model.initialized:
            self.logger.info("Initializing Triangles from point cloud")
            self.model.create_from_pcd(dataset.getPointCloud())

        if config.initial_eval:
            self._evaluate(first_iter)

        # Start training
        self.logger.info("Training started")
        timer = Timer("Training")
        progress_bar = tqdm(range(first_iter + 1, config.iterations + 1), desc="Training")

        for iteration in progress_bar:
            timer.log("data loading")
            camera = dataset.nextTrainData().to(self.device)

            timer.log("forward pass")
            background = config.train_background if config.train_background is not None else "random"
            render_pkg = self.model.forward(camera, background)

            timer.log("loss calculation")
            loss = self._get_loss(iteration, render_pkg, camera)

            timer.log("backward pass")
            loss.backward()

            timer.log("optimizer step")
            self._optimize(iteration)

            if config.eval_interval_iter > 0 and iteration % config.eval_interval_iter == 0:
                timer.log("evaluation")
                self._evaluate(iteration)

            timer.log("model update")
            self.model.model_update(iteration, render_pkg)

            if config.log_interval_iter > 0 and iteration % config.log_interval_iter == 0:
                timer.log("logging")
                log_pkg = {
                    "iteration": iteration,
                    "loss": loss.item(),
                    "geometry_loss": render_pkg["geometry_loss"],
                    "vertex_loss": render_pkg["vertex_loss"],
                    "triangle_count": (
                        render_pkg["opacity"].shape[0]
                        if self.model.ste_threshold is None
                        else (render_pkg["opacity"] > self.model.ste_threshold).sum().item()
                    ),
                    "gamma": self.model.gamma,
                    "sh_degree": self.model.active_sh_degree,
                    "time_elapsed": timer.total_duration(),
                }
                self._log_stats(log_pkg)

            if config.histogram_interval_iter > 0 and iteration % config.histogram_interval_iter == 0:
                timer.log("histogram")
                self._histogram(iteration)

            if iteration in config.save_iterations or (config.save_interval_iter > 0 and iteration % config.save_interval_iter == 0):
                timer.log("point cloud saving")
                self.model.savePLY(f"{self.output_dir}/point_cloud/{iteration}.ply")

            if iteration in config.checkpoint_iterations or (config.ckpt_interval_iter > 0 and iteration % config.ckpt_interval_iter == 0):
                timer.log("checkpoint saving")
                self.model.save_ckpt(f"{self.output_dir}/ckpt/{iteration}.ckpt")

            if iteration in config.save_glb_iterations:
                timer.log("GLB saving")
                self.model.saveGLB(f"{self.output_dir}/glb/{iteration}.glb")
                self.model.saveGLB(f"{self.output_dir}/mesh_ply/{iteration}_mesh.ply")

            timer.stop()
            if config.log_interval_iter > 0 and iteration % config.log_interval_iter == 0:
                self.logger.debug(timer.message())

        self.logger.info(timer.message())
        self.logger.info("Training finished")

    def train(self):
        try:
            self._train()
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            del self.dataset
            raise e

    def evaluate(self):
        self._evaluate(0, use_tensorboard=False)
