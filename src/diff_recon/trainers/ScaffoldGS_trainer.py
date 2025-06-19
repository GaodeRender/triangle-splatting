import numpy as np
import torch
from tqdm import tqdm
from typing import Union, Optional

from .Base_trainer import BaseTrainer
from .trainer_utils import L1, ssimLoss, psnr
from ..datasets.Colmap_dataset import ColmapDatasetFactory
from ..models.ScaffoldGS_model import ScaffoldGSModel
from ..utils.timer import Timer
from ..utils.config import Config


class ScaffoldGSTrainer(BaseTrainer):
    def __init__(self, config: Union[str, Config], exp_name: Optional[str] = None, device: Optional[int] = None):
        super().__init__(config, exp_name, device)

        # Initialize model
        self.model = ScaffoldGSModel(self.config.model, logger=self.logger, device=self.device)
        self.model.setup_scene_info(self.dataset.getSceneInfo())

        # For logging and profiling
        test_img_count = self.dataset.getTestDatasetSize()
        eval_save_img_count = self.config.trainer.eval_save_img_count
        self.save_img_idx = sorted(np.random.choice(test_img_count, eval_save_img_count, replace=False).tolist())
        self.record_gt = True

    def _get_loss(self, render_pkg: dict, gt_image: torch.Tensor) -> torch.Tensor:
        image, scaling, opacity = render_pkg["render"], render_pkg["scaling"], render_pkg["gaussian_opacity"]

        w_ssim = self.config.trainer.w_ssim
        w_s_reg = self.config.trainer.w_scaling_reg
        w_o_reg = self.config.trainer.w_opacity_reg

        l1_loss = L1(image, gt_image)
        ssim_loss = ssimLoss(image, gt_image)

        scaling_reg = scaling.mean()
        opacity_reg = (0.25 - (opacity - 0.5).pow(2)).mean()

        loss = (1.0 - w_ssim) * l1_loss + w_ssim * ssim_loss + w_s_reg * scaling_reg + w_o_reg * opacity_reg
        return loss

    def _optimize(self, iteration: int):
        self.model.update_learning_rate(iteration)
        self.model.optimizer.step()
        self.model.optimizer.zero_grad(set_to_none=True)
        self.model.maintain_constraints(iteration)

    def _log_stats(self, log_pkg: dict):
        iteration = log_pkg["iteration"]
        loss = log_pkg["loss"]
        anchor_count = log_pkg["anchor_count"]
        gaussian_count = log_pkg["gaussian_count"]
        time_elapsed = log_pkg["time_elapsed"]

        self.logger.info(f"[ITER {iteration}] Loss: {loss:.5f}, Anchor Count: {anchor_count}, Gaussian Count: {gaussian_count}")

        self.logger.add_scalar("Loss", loss, iteration)
        self.logger.add_scalar("Anchor Count", anchor_count, iteration)
        self.logger.add_scalar("Training Time (min)", time_elapsed / 60, iteration)

    def _evaluate(self, iteration: int, log_name: str = ""):
        self.logger.debug("Evaluation started")

        with torch.no_grad():
            psnr_vals = []
            for i, test_data in enumerate(self.dataset.getTestDataset()):
                camera = test_data.to(self.device)
                image = self.model.forward(camera, False)["render"]
                psnr_val = psnr(image, camera.gt_image)
                psnr_vals.append(psnr_val.item())
                if i in self.save_img_idx:
                    img_log_idx = self.save_img_idx.index(i)
                    self.logger.add_image(f"{log_name} Pred {img_log_idx}", image, iteration)
                    if self.record_gt:
                        self.logger.add_image(f"GT {img_log_idx}", camera.gt_image, 0)
            self.record_gt = False

        self.logger.info(f"[ITER {iteration}] Evaluation Average PSNR: {np.mean(psnr_vals)}, eval count: {len(psnr_vals)}")
        self.logger.add_scalar(f"{log_name} Average PSNR", np.mean(psnr_vals), iteration)

        self.logger.debug("Evaluation finished")

    def _histogram(self, iteration: int):
        with torch.no_grad():
            gaussian_pkg = self.model.generate_gaussians(is_training=True)

            activation_count = gaussian_pkg["offset_selection_mask"].view(-1, self.model.n_offsets).sum(dim=1)
            all_opacity = gaussian_pkg["gaussian_opacity"].view(-1)
            all_scaling = gaussian_pkg["scaling"].view(-1)

            # downsample to reduce tensorboard log size
            sample_num = self.config.trainer.histogram_sample_num
            activation_count = activation_count[torch.randperm(activation_count.shape[0])[:sample_num]]
            all_opacity = all_opacity[torch.randperm(all_opacity.shape[0])[:sample_num]]
            all_scaling = all_scaling[torch.randperm(all_scaling.shape[0])[:sample_num]]

            self.logger.add_histogram("Activation Histogram", activation_count, iteration)
            self.logger.add_histogram("Opacity Histogram", all_opacity, iteration)
            self.logger.add_histogram("Scaling Histogram", all_scaling, iteration)

            gaussian_count = gaussian_pkg["opacity"].shape[0]
            self.logger.add_scalar("Gaussian Count", gaussian_count, iteration)
            self.logger.info(f"[ITER {iteration}] Total Gaussian Count: {gaussian_count}")

    def _train(self):
        dataset: ColmapDatasetFactory = self.dataset
        config = self.config.trainer

        # Initialize gaussian model
        first_iter = 0
        if config.start_checkpoint:
            self.logger.info(f"Initializing Gaussians from checkpoint {config.start_checkpoint}.ckpt")
            self.model.load_ckpt(f"{self.output_dir}/ckpt/{config.start_checkpoint}.ckpt")
            first_iter = config.start_checkpoint
        else:
            self.logger.info("Initializing Gaussians from point cloud")
            self.model.create_from_pcd(dataset.getPointCloud())

        self.logger.info("Training started")
        timer = Timer("Training")
        progress_bar = tqdm(range(first_iter + 1, config.iterations + 1), desc="Training")

        for iteration in progress_bar:
            timer.log("data loading")
            camera = dataset.nextTrainData().to(self.device)

            timer.log("forward pass")
            render_pkg = self.model.forward(camera, True)

            timer.log("loss calculation")
            loss = self._get_loss(render_pkg, camera.gt_image)

            timer.log("backward pass")
            loss.backward()

            timer.log("optimizer step")
            self._optimize(iteration)

            if config.log_interval_iter > 0 and iteration % config.log_interval_iter == 0:
                timer.log("logging")
                log_pkg = {
                    "iteration": iteration,
                    "loss": loss.item(),
                    "anchor_count": self.model.anchor_count,
                    "gaussian_count": render_pkg["viewspace_points"].shape[0],
                    "time_elapsed": timer.total_duration(),
                }
                self._log_stats(log_pkg)

            if config.eval_interval_iter > 0 and iteration % config.eval_interval_iter == 0:
                timer.log("evaluation")
                self._evaluate(iteration)

            if config.histogram_interval_iter > 0 and iteration % config.histogram_interval_iter == 0:
                timer.log("histogram")
                self._histogram(iteration)

            timer.log("statistic update")
            self.model.training_statistic(iteration, render_pkg)

            timer.log("anchor adjustment")
            self.model.anchor_update(iteration)

            if iteration in config.save_iterations:
                timer.log("gaussians saving")
                self.model.savePLY(f"{self.output_dir}/point_cloud/{iteration}.ply")

            if iteration in config.checkpoint_iterations:
                timer.log("checkpoint saving")
                self.model.save_ckpt(f"{self.output_dir}/ckpt/{iteration}.ckpt")

            timer.stop()
            if iteration % config.log_interval_iter == 0:
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

    def _mlp_pretrain_get_loss(self, raw_pkg: dict, gt_pkg: dict) -> torch.Tensor:
        w_offset = 1.0
        w_opacity = 1.0
        w_cov = 1.0
        w_color = 1.0

        offset_loss = L1(raw_pkg["g_offset"], gt_pkg["g_offset"])
        opacity_loss = L1(raw_pkg["g_opacity"], gt_pkg["g_opacity"])
        cov_loss = L1(raw_pkg["g_cov"], gt_pkg["g_cov"])
        color_loss = L1(raw_pkg["g_color"], gt_pkg["g_color"])

        loss = w_offset * offset_loss + w_opacity * opacity_loss + w_cov * cov_loss + w_color * color_loss
        return loss

    def _mlp_pretrain_optimize(self, iteration: int) -> dict:
        self.model.update_learning_rate(iteration)
        self.model.optimizer.step()
        self.model.optimizer.zero_grad(set_to_none=True)

    def _mlp_pretrain_log_stats(self, log_pkg: dict):
        iteration = log_pkg["iteration"]
        loss = log_pkg["loss"]

        self.logger.info(f"[ITER {iteration}] Loss: {loss:.5f}")
        self.logger.add_scalar("Pretrain Loss", loss, iteration)

    def mlp_pretrain(self):
        self.logger.info("Initializing Gaussians from ground truth gaussian")
        gt_pkg = self.model.create_from_gt_gaussian(self.dataset.getGTGaussian())

        self.logger.info("Pretraining started")
        config = self.config.trainer.pretrain
        timer = Timer("Pretraining")
        progress_bar = tqdm(range(1, config.iterations + 1), desc="Pretraining")

        for iteration in progress_bar:
            timer.log("forward pass")
            raw_pkg = self.model.get_raw_output()

            timer.log("loss calculation")
            loss = self._mlp_pretrain_get_loss(raw_pkg, gt_pkg)

            timer.log("backward pass")
            loss.backward()

            timer.log("optimizer step")
            self._mlp_pretrain_optimize(iteration)

            if iteration % config.log_interval_iter == 0:
                timer.log("logging")
                log_pkg = {
                    "iteration": iteration,
                    "loss": loss.item(),
                }
                self._mlp_pretrain_log_stats(log_pkg)

            if config.eval_interval_iter > 0 and iteration % config.eval_interval_iter == 0:
                timer.log("evaluation")
                self._evaluate(iteration, "Pretrain")

            if iteration in config.save_iterations:
                timer.log("gaussians saving")
                self.logger.info(f"[ITER {iteration}] Saving Gaussians")
                self.model.savePLY(f"{self.output_dir}/point_cloud/pt_{iteration}.ply")

            if iteration in config.checkpoint_iterations:
                timer.log("checkpoint saving")
                self.logger.info(f"[ITER {iteration}] Saving Checkpoint")
                self.model.save_ckpt(f"{self.output_dir}/ckpt/pt_{iteration}.ckpt")

            timer.stop()
            if iteration % config.log_interval_iter == 0:
                self.logger.debug(timer.message())

        self.logger.info(timer.message())
        self.logger.info("Pretraining finished")
