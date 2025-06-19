import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from .point_cloud import PointCloud
from .raw_gaussian import RawGaussian
from .model_utils import *
from .Base_model import BaseModel
from ..renderer.gaussian_renderer import GaussianRenderer
from ..utils.camera import Camera
from ..utils.config import Config
from ..utils.logger import Logger
from ..utils.scheduler import exponential_scheduler
from ..utils.sh_utils import RGB2SH


class VanillaGSModel(BaseModel):
    def __init__(self, config: Config = None, logger: Logger = None, device: torch.device | int = None):
        super().__init__(config, logger, device)

        self.max_sh_degree = config.max_sh_degree if config is not None else 0
        self.use_color_affine = config.use_color_affine if config is not None else False

        self.active_sh_degree = 0
        self.gamma = 1.0
        self.optimizer = None
        self.scene_bbox = None
        self.initialized = False

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)

    @property
    def get_rotation(self):
        return F.normalize(self._rotation, dim=1)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        return torch.cat((self._f_dc, self._f_rest), dim=1)

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)

    def setup_color_affine(self, view_count):
        if not self.use_color_affine:
            return

        self._color_affine_weight = nn.Parameter(torch.zeros((view_count, 3, 3)).float().to(self.device), requires_grad=True)
        self._color_affine_weight.data[:, 0, 0] = 1
        self._color_affine_weight.data[:, 1, 1] = 1
        self._color_affine_weight.data[:, 2, 2] = 1
        self._color_affine_bias = nn.Parameter(torch.zeros((view_count, 3)).float().to(self.device), requires_grad=True)

    def setup_scene_info(self, scene_info: dict = None):
        if scene_info is None:
            return

        self.scene_bbox = scene_info["bbox_xyz"]

    def _setup_parameters(self, point_count: int):
        self._xyz = nn.Parameter(torch.empty(point_count, 3).float().to(self.device), requires_grad=True)
        self._scaling = nn.Parameter(torch.empty(point_count, 3).float().to(self.device), requires_grad=True)
        self._rotation = nn.Parameter(torch.empty(point_count, 4).float().to(self.device), requires_grad=True)
        self._opacity = nn.Parameter(torch.empty(point_count, 1).float().to(self.device), requires_grad=True)
        self._f_dc = nn.Parameter(torch.empty(point_count, 1, 3).float().to(self.device), requires_grad=True)
        self._f_rest = nn.Parameter(torch.empty(point_count, (self.max_sh_degree + 1) ** 2 - 1, 3).float().to(self.device), requires_grad=True)

    def _setup_optimizer(self):
        if self.config.optimizer is None:
            return

        l = [
            {"params": [self._xyz], "lr": 0, "name": "xyz"},
            {"params": [self._scaling], "lr": 0, "name": "scaling"},
            {"params": [self._rotation], "lr": 0, "name": "rotation"},
            {"params": [self._opacity], "lr": 0, "name": "opacity"},
            {"params": [self._f_dc], "lr": 0, "name": "f_dc"},
            {"params": [self._f_rest], "lr": 0, "name": "f_rest"},
        ]
        if self.use_color_affine:
            l += [
                {"params": [self._color_affine_weight], "lr": 0, "name": "color_affine_weight"},
                {"params": [self._color_affine_bias], "lr": 0, "name": "color_affine_bias"},
            ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def _setup_lr_scheduler(self):
        args = self.config.optimizer
        if args is None:
            return

        self.lr_schedulers = {
            "xyz": exponential_scheduler(**vars(args.xyz)),
            "scaling": exponential_scheduler(**vars(args.scaling)),
            "rotation": exponential_scheduler(**vars(args.rotation)),
            "opacity": exponential_scheduler(**vars(args.opacity)),
            "f_dc": exponential_scheduler(**vars(args.f_dc)),
            "f_rest": exponential_scheduler(**vars(args.f_rest)),
        }

        if self.use_color_affine:
            self.lr_schedulers.update(
                {
                    "color_affine_weight": exponential_scheduler(**vars(args.color_affine)),
                    "color_affine_bias": exponential_scheduler(**vars(args.color_affine)),
                }
            )

    def _setup_model_update_utils(self):
        args = self.config.model_update
        if args is None:
            return

        if args.desnification is not None:
            self.grad_threshold_scheduler = exponential_scheduler(
                v_init=args.densification.grad_threshold_init,
                v_final=args.densification.grad_threshold_final,
                max_steps=args.densification.end_iter - args.densification.start_iter,
            )
        if args.opacity_pruning is not None:
            self.opacity_pruning_scheduler = exponential_scheduler(
                v_init=args.opacity_pruning.opacity_threshold_init,
                v_final=args.opacity_pruning.opacity_threshold_final,
                max_steps=args.opacity_pruning.end_iter - args.opacity_pruning.start_iter,
            )
        if args.opacity_clipping is not None:
            self.opacity_clipping_scheduler = exponential_scheduler(
                v_init=args.opacity_clipping.opacity_threshold_init,
                v_final=args.opacity_clipping.opacity_threshold_final,
                max_steps=args.opacity_clipping.end_iter - args.opacity_clipping.start_iter,
            )
        if args.scale_clipping is not None:
            self.scale_max_scheduler = exponential_scheduler(
                v_init=args.scale_clipping.scale_max_init,
                v_final=args.scale_clipping.scale_max_final,
                max_steps=args.scale_clipping.end_iter - args.scale_clipping.start_iter,
            )
        if args.gamma_schedule is not None:
            self.gamma_scheduler = exponential_scheduler(
                v_init=args.gamma_schedule.gamma_init,
                v_final=args.gamma_schedule.gamma_final,
                max_steps=args.gamma_schedule.end_iter - args.gamma_schedule.start_iter,
            )

        self.gradient_accum = torch.zeros((self._xyz.shape[0]), device=self.device)
        self.gradient_denom = torch.zeros((self._xyz.shape[0]), device=self.device)
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device=self.device)
        self.contrib_sum = torch.zeros((self._xyz.shape[0]), device=self.device)
        self.contrib_max = torch.zeros((self._xyz.shape[0]), device=self.device)
        self.contrib_denom = torch.zeros((self._xyz.shape[0]), device=self.device)

    def _training_setup(self):
        self._setup_optimizer()
        self._setup_lr_scheduler()
        self._setup_model_update_utils()
        self.initialized = True

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] in self.lr_schedulers:
                param_group["lr"] = self.lr_schedulers[param_group["name"]](iteration)

    def _prune_points_update_states(self, mask: torch.Tensor):
        # update optimizer and parameter after runing _prune_points
        for group in self.optimizer.param_groups:
            if group["name"] in ["color_affine_weight", "color_affine_bias"]:
                continue

            param = group["params"][0]
            new_param = nn.Parameter(param[mask], requires_grad=True)

            stored_state = self.optimizer.state.pop(param, None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                self.optimizer.state[new_param] = stored_state

            group["params"][0] = new_param
            self.__setattr__(f"_{group['name']}", new_param)

    def _prune_points(self, prune_mask: torch.Tensor):
        self.gradient_accum = self.gradient_accum[~prune_mask]
        self.gradient_denom = self.gradient_denom[~prune_mask]
        self.max_radii2D = self.max_radii2D[~prune_mask]
        self.contrib_sum = self.contrib_sum[~prune_mask]
        self.contrib_max = self.contrib_max[~prune_mask]
        self.contrib_denom = self.contrib_denom[~prune_mask]
        self._prune_points_update_states(~prune_mask)

    def _grow_points_update_states(self, tensors_dict: dict[str, torch.Tensor]):
        # update optimizer and parameter after runing _grow_points
        for group in self.optimizer.param_groups:
            if group["name"] in ["color_affine_weight", "color_affine_bias"]:
                continue

            param = group["params"][0]
            extension_tensor = tensors_dict[group["name"]]
            new_param = nn.Parameter(torch.cat((param, extension_tensor), dim=0), requires_grad=True)

            stored_state = self.optimizer.state.pop(param, None)
            if stored_state:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                self.optimizer.state[new_param] = stored_state

            group["params"][0] = new_param
            self.__setattr__(f"_{group['name']}", new_param)

    def _grow_points(self, grow_mask: torch.Tensor, n_split: int, split_scale_threshold: float):
        large_point_mask = torch.max(self.get_scaling, dim=1).values > split_scale_threshold
        clone_mask = grow_mask & ~large_point_mask  # clone small points
        split_mask = grow_mask & large_point_mask  # split large points

        clone_xyz = self._xyz[clone_mask]
        clone_scaling = self._scaling[clone_mask]
        clone_rotation = self._rotation[clone_mask]
        clone_opacity = self._opacity[clone_mask]
        clone_f_dc = self._f_dc[clone_mask]
        clone_f_rest = self._f_rest[clone_mask]

        N = n_split
        stds = self.get_scaling[split_mask].repeat(N, 1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_Rmat(self._rotation[split_mask]).repeat(N, 1, 1)
        offsets = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
        split_xyz = self._xyz[split_mask].repeat(N, 1) + offsets
        split_scaling = torch.log(self.get_scaling[split_mask] / (0.8 * N)).repeat(N, 1)
        split_rotation = self._rotation[split_mask].repeat(N, 1)
        split_opacity = self._opacity[split_mask].repeat(N, 1)
        split_f_dc = self._f_dc[split_mask].repeat(N, 1, 1)
        split_f_rest = self._f_rest[split_mask].repeat(N, 1, 1)

        new_xyz = torch.cat((clone_xyz, split_xyz), dim=0)
        new_scaling = torch.cat((clone_scaling, split_scaling), dim=0)
        new_rotation = torch.cat((clone_rotation, split_rotation), dim=0)
        new_opacity = torch.cat((clone_opacity, split_opacity), dim=0)
        new_f_dc = torch.cat((clone_f_dc, split_f_dc), dim=0)
        new_f_rest = torch.cat((clone_f_rest, split_f_rest), dim=0)

        new_points = {
            "xyz": new_xyz,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "opacity": new_opacity,
            "f_dc": new_f_dc,
            "f_rest": new_f_rest,
        }

        self._prune_points(split_mask)
        self._grow_points_update_states(new_points)

        new_points_count = new_xyz.shape[0]
        self.gradient_accum = F.pad(self.gradient_accum, (0, new_points_count), value=0)
        self.gradient_denom = F.pad(self.gradient_denom, (0, new_points_count), value=0)
        self.max_radii2D = F.pad(self.max_radii2D, (0, new_points_count), value=0)
        self.contrib_sum = F.pad(self.contrib_sum, (0, new_points_count), value=0)
        self.contrib_max = F.pad(self.contrib_max, (0, new_points_count), value=0)
        self.contrib_denom = F.pad(self.contrib_denom, (0, new_points_count), value=0)

    def _reset_opacity_update_states(self, tensor_dict: torch.Tensor):
        # update optimizer and parameter after runing _opacity_reset
        for group in self.optimizer.param_groups:
            if not group["name"] in tensor_dict:
                continue

            param = group["params"][0]
            new_param = nn.Parameter(tensor_dict[group["name"]], requires_grad=True)

            stored_state = self.optimizer.state.pop(param, None)
            if stored_state:
                stored_state["exp_avg"] = torch.zeros_like(new_param)
                stored_state["exp_avg_sq"] = torch.zeros_like(new_param)
                self.optimizer.state[new_param] = stored_state

            group["params"][0] = new_param
            self.__setattr__(f"_{group['name']}", new_param)

    def _clipping_update_states(self, clip_mask: torch.Tensor, clip_value: float, name: str):
        # update optimizer and parameter after runing _scale_clipping or _opacity_clipping
        for group in self.optimizer.param_groups:
            if not group["name"] == name:
                continue

            param = group["params"][0]
            new_param = nn.Parameter(param, requires_grad=True)
            new_param[clip_mask] = clip_value

            stored_state = self.optimizer.state.pop(param, None)
            if stored_state:
                stored_state["exp_avg"][clip_mask] = 0.0
                stored_state["exp_avg_sq"][clip_mask] = 0.0
                self.optimizer.state[new_param] = stored_state

            group["params"][0] = new_param
            self.__setattr__(f"_{name}", new_param)

    def _training_statistic(self, iteration: int, render_pkg: dict[str, torch.Tensor]):
        args = self.config.model_update.statistic
        if args is None or not (args.start_iter < iteration <= args.end_iter):
            return

        viewspace_points = render_pkg["viewspace_points"]
        visible_mask = render_pkg["visible_mask"]
        radii = render_pkg["radii"]
        contrib_sum = render_pkg["contrib_sum"]
        contrib_max = render_pkg["contrib_max"]

        self.gradient_accum[visible_mask] += torch.norm(viewspace_points.grad[visible_mask, :2], dim=-1)
        self.gradient_denom[visible_mask] += 1
        self.contrib_sum[visible_mask] = torch.max(self.contrib_sum[visible_mask], contrib_sum[visible_mask])
        self.contrib_max[visible_mask] = torch.max(self.contrib_max[visible_mask], contrib_max[visible_mask])
        self.contrib_denom[visible_mask] += 1
        self.max_radii2D[visible_mask] = torch.max(self.max_radii2D[visible_mask], radii[visible_mask])

    def _densification(self, iteration: int):
        args = self.config.model_update.densification
        if args is None or not (args.start_iter < iteration <= args.end_iter and iteration % args.interval_iter == 0):
            return

        start_iter = args.start_iter
        min_view_count = args.min_view_count
        split_num = args.split_num
        split_scale_threshold = args.split_scale_threshold

        grad_threshold = self.grad_threshold_scheduler(iteration - start_iter)
        select_mask = self.gradient_denom > min_view_count
        grad_mask = self.gradient_accum > grad_threshold * self.gradient_denom
        grow_mask = select_mask & grad_mask

        self.gradient_accum[select_mask] = 0
        self.gradient_denom[select_mask] = 0
        self._grow_points(grow_mask, split_num, split_scale_threshold)
        self.logger.info(f"[ITER {iteration}, densification] Growing {grow_mask.sum().item()} points, grad threshold: {grad_threshold:.5f}")

    def _opacity_pruning(self, iteration: int):
        args = self.config.model_update.opacity_pruning
        if args is None or not (args.start_iter < iteration <= args.hold_iter and iteration % args.interval_iter == 0):
            return

        start_iter = args.start_iter

        opacity_threshold = self.opacity_pruning_scheduler(iteration - start_iter)
        prune_mask = (self.get_opacity < opacity_threshold).squeeze(-1)

        self._prune_points(prune_mask)
        self.logger.info(f"[ITER {iteration}, opacity pruning] Pruning {prune_mask.sum().item()} points, opacity threshold: {opacity_threshold:.5f}")

    def _opacity_clipping(self, iteration: int):
        args = self.config.model_update.opacity_clipping
        if args is None or not (args.start_iter < iteration <= args.hold_iter and iteration % args.interval_iter == 0):
            return

        opacity_threshold = self.opacity_clipping_scheduler(iteration - args.start_iter)
        clip_mask = (self.get_opacity > opacity_threshold).squeeze(-1)
        clip_count = clip_mask.sum().item()

        if clip_count > 0:
            self._clipping_update_states(clip_mask, 10.0, "opacity")
        self.logger.info(f"[ITER {iteration}, opacity clipping] Clipping {clip_count} points, opacity threshold: {opacity_threshold:.5f}")

    def _scale_pruning(self, iteration: int):
        args = self.config.model_update.scale_pruning
        if args is None or not (args.start_iter < iteration <= args.end_iter and iteration % args.interval_iter == 0):
            return

        radii_threshold = args.radii_threshold
        scale_threshold = args.scale_threshold

        radii_prune_mask = self.max_radii2D > radii_threshold
        scale_prune_mask = self.get_scaling.max(dim=1).values > scale_threshold
        prune_mask = radii_prune_mask | scale_prune_mask

        self._prune_points(prune_mask)
        self.logger.info(
            f"[ITER {iteration}, scale pruning] Pruning {prune_mask.sum().item()} points, "
            + f"{radii_prune_mask.sum().item()} by radii, {scale_prune_mask.sum().item()} by scale, "
            + f"radii threshold: {radii_threshold}, scale threshold: {scale_threshold}"
        )

    def _scale_clipping(self, iteration: int):
        args = self.config.model_update.scale_clipping
        if args is None or not (args.start_iter < iteration <= args.hold_iter and iteration % args.interval_iter == 0):
            return

        scale_max = self.scale_max_scheduler(iteration - args.start_iter)
        log_scale_max = np.log(scale_max)
        clip_mask = self._scaling > log_scale_max
        clip_count = clip_mask.any(-1).sum().item()

        if clip_count > 0:
            self._clipping_update_states(clip_mask, log_scale_max, "scaling")
        self.logger.info(f"[ITER {iteration}, scale clipping] Clipping {clip_count} points, scale max: {scale_max:.5f}")

    def _contribution_pruning(self, iteration: int):
        args = self.config.model_update.contribution_pruning
        if args is None or not (args.start_iter < iteration <= args.end_iter and iteration % args.interval_iter == 0):
            return

        min_view_count = args.min_view_count
        target_point_num = args.target_point_num
        prune_ratio = args.prune_ratio
        max_prune_ratio = args.max_prune_ratio
        contrib_max_ratio = args.contrib_max_ratio
        sparsity_retain_ratio = args.sparsity_retain_ratio
        dowsample_iteration = args.downsample_iteration
        downsample_point_num = args.downsample_point_num

        for iter, point_num in zip(dowsample_iteration, downsample_point_num):
            if iteration > iter:
                target_point_num = point_num
                contrib_max_ratio *= 0.5
                new_sparsity_retain_ratio = sparsity_retain_ratio + (0.8 - sparsity_retain_ratio) * 0.5
                prune_ratio *= (1 - sparsity_retain_ratio) / (1 - new_sparsity_retain_ratio)
                sparsity_retain_ratio = new_sparsity_retain_ratio

        total_point_count = self._xyz.shape[0]
        inside_point_count = get_inside_mask(self._xyz, self.scene_bbox).sum().item()
        select_mask = self.contrib_denom > min_view_count
        select_count = select_mask.sum().item()
        diff_point_count = max(0, inside_point_count - target_point_num * 0.99) * total_point_count / inside_point_count
        prune_count = min(diff_point_count * prune_ratio, select_count * max_prune_ratio)
        contrib_max_prune_count = int(prune_count * contrib_max_ratio)
        contrib_sum_prune_count = int(prune_count * (1 - contrib_max_ratio))

        select_idx = torch.argwhere(select_mask).squeeze(1)
        contrib_max_prune_idx = torch.argsort(self.contrib_max[select_mask])[:contrib_max_prune_count]
        contrib_max_prune_idx = select_idx[contrib_max_prune_idx]
        max_pruned_contrib_max = self.contrib_max[contrib_max_prune_idx[-1]].item() if len(contrib_max_prune_idx) > 0 else 0
        contrib_sum_prune_idx = torch.argsort(self.contrib_sum[select_mask])[:contrib_sum_prune_count]
        contrib_sum_prune_idx = select_idx[contrib_sum_prune_idx]
        max_pruned_contrib_sum = self.contrib_sum[contrib_sum_prune_idx[-1]].item() if len(contrib_sum_prune_idx) > 0 else 0
        prune_idx = torch.cat((contrib_max_prune_idx, contrib_sum_prune_idx)).unique()

        retain_point_count = 0
        if sparsity_retain_ratio > 0:
            dist = inter_point_distance(self._xyz)
            pruned_dist = dist[prune_idx]
            retain_point_count = int(sparsity_retain_ratio * len(prune_idx))
            prune_idx = prune_idx[torch.argsort(pruned_dist, descending=True)[retain_point_count:]]

        prune_mask = torch.zeros_like(select_mask)
        prune_mask[prune_idx] = 1

        self.contrib_sum[select_mask] = 0
        self.contrib_max[select_mask] = 0
        self.contrib_denom[select_mask] = 0
        self._prune_points(prune_mask)
        self.logger.info(
            f"[ITER {iteration}, contribution pruning] Pruning {prune_idx.shape[0]} points, "
            + f"{contrib_max_prune_count} by contrib_max, {contrib_sum_prune_count} by contrib_sum, \n"
            + f"max pruned contrib_max: {max_pruned_contrib_max:.5f}, max pruned contrib_sum: {max_pruned_contrib_sum:.5f}, \n"
            + f"{select_count} points with enough view to prune, {retain_point_count} pruned points are retained by sparsity, \n"
            + f"target inside point count: {target_point_num}, before pruning: {inside_point_count}, after pruning: {inside_point_count - prune_idx.shape[0]}"
        )

    def _opacity_reset(self, iteration: int):
        args = self.config.model_update.opacity_reset
        if args is None or not (args.start_iter < iteration <= args.end_iter and iteration % args.interval_iter == 0):
            return

        reset_value = args.reset_value

        opacity_old = self.get_opacity
        opacity_reset = torch.ones_like(opacity_old) * reset_value
        reset_mask = opacity_old > reset_value

        opacity_new = inverse_sigmoid(torch.min(opacity_old, opacity_reset))
        self._reset_opacity_update_states({"opacity": opacity_new})
        self.logger.info(f"[ITER {iteration}, opacity reset] Reset opacity of {reset_mask.sum().item()} points to {reset_value}")

    def _set_gamma(self, iteration: int):
        args = self.config.model_update.gamma_schedule
        if args is None or not (args.start_iter < iteration <= args.end_iter):
            return

        self.gamma = self.gamma_scheduler(iteration - args.start_iter)

    def _set_sh_degree(self, iteration: int):
        args = self.config.model_update.sh_schedule
        if args is None:
            return

        active_sh_degree = 0
        for iter in args.one_up_iters:
            if iteration > iter:
                active_sh_degree += 1
        self.active_sh_degree = min(active_sh_degree, self.max_sh_degree)

    @torch.no_grad()
    def model_update(self, iteration: int, render_pkg: dict[str, torch.Tensor]):
        if self.config.model_update is None:
            return

        self._training_statistic(iteration, render_pkg)
        self._densification(iteration)
        self._opacity_pruning(iteration)
        self._opacity_clipping(iteration)
        self._scale_pruning(iteration)
        self._scale_clipping(iteration)
        self._contribution_pruning(iteration)
        self._opacity_reset(iteration)
        self._set_gamma(iteration)
        self._set_sh_degree(iteration)

    def forward(
        self,
        camera: Camera,
        background: str,
        is_training: bool = True,
        color_affine: bool = None,
        back_culling: bool = None,
        sh_degree: int = None,
        gamma: float = None,
    ) -> dict[str, torch.Tensor]:
        """
        Render the scene.
        """
        color_affine = color_affine if color_affine is not None else self.use_color_affine
        sh_degree = sh_degree if sh_degree is not None else self.active_sh_degree
        gamma = gamma if gamma is not None else self.gamma

        bg_color = get_color_tensor(background).to(self.device)

        xyz = self.get_xyz
        shs = self.get_features
        opacity = self.get_opacity
        scaling = self.get_scaling
        rot = self.get_rotation

        renderer = GaussianRenderer(
            camera,
            bg_color=bg_color,
            sh_degree=min(sh_degree, self.max_sh_degree),
            gamma=gamma,
            rich_info=is_training,
        )
        output_pkg = renderer.render(xyz, shs, None, opacity, scaling, rot)

        render_pkg = {
            "render": output_pkg["render"],
        }
        if is_training:
            training_info = {
                "gt_image": camera.gt_image,
                "gt_mask": camera.alpha_mask,
                "radii": output_pkg["radii"],
                "viewspace_points": output_pkg["means2D"],
                "contrib_sum": output_pkg["contrib_sum"],
                "contrib_max": output_pkg["contrib_max"],
                "scaling": scaling,
                "opacity": opacity,
                "visible_mask": output_pkg["radii"] > 0,
            }
            render_pkg.update(training_info)

        if self.use_color_affine and color_affine:
            image = render_pkg["render"]
            uid = camera.uid
            image_transformed = (image.permute(1, 2, 0) @ self._color_affine_weight[uid] + self._color_affine_bias[uid]).permute(2, 0, 1)
            render_pkg["render"] = image_transformed.clamp(0, 1)
            render_pkg["render_original"] = image

        return render_pkg

    def savePLY(self, ply_path: str, tile_filtering: bool = True):
        self.logger.info(f"Saving gaussians to {ply_path}")
        gaussian = self.toRawGaussian(tile_filtering)
        gaussian.savePLY(ply_path, save_extra=True)

    def loadPLY(self, ply_path: str) -> "VanillaGSModel":
        self.logger.info(f"Loading gaussians from {ply_path}")
        gaussian = RawGaussian(ply_path=ply_path)
        return self.fromRawGaussian(gaussian)

    @torch.no_grad()
    def toRawGaussian(self, tile_filtering: bool = True) -> RawGaussian:
        xyz = self._xyz
        rot = self._rotation
        scale = self._scaling
        opacity = self._opacity
        shs = self.get_features.view(xyz.shape[0], -1)

        if tile_filtering and self.scene_bbox is not None:
            mask = get_inside_mask(xyz, self.scene_bbox)
            xyz = xyz[mask]
            rot = rot[mask]
            scale = scale[mask]
            opacity = opacity[mask]
            shs = shs[mask]

        return RawGaussian(*to_numpy(xyz, rot, scale, opacity, shs))

    def fromRawGaussian(self, gaussian: RawGaussian) -> "VanillaGSModel":
        ng = gaussian.xyz.shape[0]
        shs = torch.tensor(gaussian.shs).float().to(self.device).view(ng, -1, 3)
        features = torch.zeros((ng, (self.max_sh_degree + 1) ** 2, 3)).float().to(self.device)
        if shs.shape[1] > features.shape[1]:
            features = shs[:, : features.shape[1], :]
        else:
            features[:, : shs.shape[1], :] = shs

        self._xyz = nn.Parameter(torch.tensor(gaussian.xyz).float().to(self.device), requires_grad=True)
        self._scaling = nn.Parameter(torch.tensor(gaussian.scale).float().to(self.device), requires_grad=True)
        self._rotation = nn.Parameter(torch.tensor(gaussian.rot).float().to(self.device), requires_grad=True)
        self._opacity = nn.Parameter(torch.tensor(gaussian.opacity).float().to(self.device), requires_grad=True)
        self._f_dc = nn.Parameter(features[:, :1].contiguous(), requires_grad=True)
        self._f_rest = nn.Parameter(features[:, 1:].contiguous(), requires_grad=True)

        self._training_setup()
        return self

    def save_ckpt(self, ckpt_path: str):
        self.logger.info(f"Saving checkpoint to {ckpt_path}")
        Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)

        save_items = (self.state_dict(), self.optimizer.state_dict(), self.scene_bbox, self.gamma)
        torch.save(save_items, open(ckpt_path, "wb"))

    def load_ckpt(self, ckpt_path: str) -> "VanillaGSModel":
        (params_state_dict, optimizer_state_dict, self.scene_bbox, self.gamma) = torch.load(open(ckpt_path, "rb"))

        point_count = params_state_dict["_xyz"].shape[0]
        self._setup_parameters(point_count)
        self.load_state_dict(params_state_dict)

        self._training_setup()
        self.optimizer.load_state_dict(optimizer_state_dict)
        return self

    def _sample_points(self, xyz: torch.Tensor, shs: torch.Tensor, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        args = self.config.sampling
        sample_method = args.sample_method
        n_sample_inside = args.n_sample_inside
        n_sample_outside = args.n_sample_outside
        grid_size_inside = args.grid_size_inside
        grid_size_outside = args.grid_size_outside

        if name == "inside":
            n_sample = n_sample_inside
            grid_size = grid_size_inside
        elif name == "outside":
            n_sample = n_sample_outside
            grid_size = grid_size_outside
        else:
            raise ValueError(f"Unknown sampling name: {name}")

        self.logger.info(f"Running {name} sampling")
        if sample_method == "random":
            if n_sample > xyz.shape[0] or n_sample <= 0:
                self.logger.warning(f"target sample number {n_sample} is invalid, using all points")
                return xyz, shs
            sample_idx = torch.randperm(xyz.shape[0])[:n_sample]
            sampled_xyz, sampled_shs = xyz[sample_idx], shs[sample_idx]
        elif sample_method == "grid":
            grid_size = grid_size_search(xyz, n_sample) if grid_size is None else grid_size
            sampled_xyz, sampled_shs = grid_sampling(xyz, shs, grid_size=grid_size)
        elif sample_method == "direct":
            sampled_xyz, sampled_shs = xyz, shs
        else:
            raise ValueError(f"Unknown sampling method: {sample_method}")

        self.logger.info(f"Sampled {sampled_xyz.shape[0]} points from {xyz.shape[0]} points using {sample_method} sampling")
        if sample_method == "grid":
            self.logger.info(f"Grid size: {grid_size:.5f}")

        return sampled_xyz, sampled_shs

    def create_from_pcd(self, pcd: PointCloud):
        args = self.config.sampling
        if args is None:
            raise ValueError("Sampling config is not provided")

        init_opacity = args.init_opacity if args.init_opacity is not None else 0.1

        points = torch.tensor(np.asarray(pcd.points)).float().to(self.device)
        shs = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().to(self.device))

        inside_mask = get_inside_mask(points, self.scene_bbox)

        inside_points = points[inside_mask]
        inside_shs = shs[inside_mask]
        inside_points, inside_shs = self._sample_points(inside_points, inside_shs, "inside")

        outside_points = points[~inside_mask]
        outside_shs = shs[~inside_mask]
        outside_points, outside_shs = self._sample_points(outside_points, outside_shs, "outside")

        points = torch.cat((inside_points, outside_points), dim=0)
        shs = torch.cat((inside_shs, outside_shs), dim=0)
        scaling = torch.log(inter_point_distance(points))[..., None].repeat(1, 3)
        rotation = torch.tensor([1, 0, 0, 0]).repeat(points.shape[0], 1).float().to(self.device)
        
        if init_opacity == "random":
            opacities = inverse_sigmoid(torch.rand((points.shape[0], 1)).float().to(self.device))
        else:
            opacities = inverse_sigmoid(torch.ones((points.shape[0], 1)).float().to(self.device) * init_opacity)
        features = torch.zeros((shs.shape[0], (self.max_sh_degree + 1) ** 2, 3)).float().to(self.device)
        features[:, 0, :] = shs

        self.logger.info(f"Number of points at initialisation: {points.shape[0]}")

        self._xyz = nn.Parameter(points, requires_grad=True)
        self._scaling = nn.Parameter(scaling, requires_grad=True)
        self._rotation = nn.Parameter(rotation, requires_grad=True)
        self._opacity = nn.Parameter(opacities, requires_grad=True)
        self._f_dc = nn.Parameter(features[:, :1].contiguous(), requires_grad=True)
        self._f_rest = nn.Parameter(features[:, 1:].contiguous(), requires_grad=True)

        self._training_setup()
