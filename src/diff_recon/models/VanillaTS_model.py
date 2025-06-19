import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import scipy.special
from pathlib import Path
from copy import deepcopy

from .point_cloud import PointCloud
from .raw_triangle import RawTriangle
from .model_utils import *
from .Base_model import BaseModel
from ..renderer.triangle_renderer import TriangleRenderer
from ..utils.camera import Camera
from ..utils.config import Config
from ..utils.logger import Logger
from ..utils.scheduler import exponential_scheduler, exponential_step_scheduler
from ..utils.sh_utils import RGB2SH, SH2RGB


class VanillaTSModel(BaseModel):
    def __init__(self, config: Config = None, logger: Logger = None, device: torch.device | int = None):
        super().__init__(config, logger, device)

        self.max_sh_degree = config.max_sh_degree if config.max_sh_degree is not None else 0
        self.use_color_affine = config.use_color_affine if config.use_color_affine is not None else False
        self.back_culling = config.back_culling if config.back_culling is not None else False
        self.back_culling_prob = config.back_culling_prob if config.back_culling_prob is not None else 1.0
        self.ste_threshold = config.ste_threshold if config.ste_threshold is not None else None
        self.gamma_rescale = config.gamma_rescale if config.gamma_rescale is not None else False
        self.render_up_scale = config.render_up_scale if config.render_up_scale is not None else None
        self.rasterizer_type = config.rasterizer_type if config.rasterizer_type is not None else "3D"

        self.active_sh_degree = 0
        self.gamma = 1.0
        self.optimizer = None
        self.scene_bbox = None
        self.initialized = False

    @property
    def get_vertex(self):
        return self._vertex

    @property
    def get_xyz(self):
        return self._vertex.mean(dim=1)

    # @property
    # def get_scaling(self):
    #     """
    #     calculate the radius of the smallest enclosing circle of each triangle
    #     """
    #     side_1 = self._vertex[:, 2] - self._vertex[:, 1]
    #     side_2 = self._vertex[:, 0] - self._vertex[:, 2]
    #     side_3 = self._vertex[:, 1] - self._vertex[:, 0]
    #     l_side1 = side_1.norm(dim=1)
    #     l_side2 = side_2.norm(dim=1)
    #     l_side3 = side_3.norm(dim=1)
    #     sq_side1 = l_side1**2
    #     sq_side2 = l_side2**2
    #     sq_side3 = l_side3**2
    #     is_obtuse = torch.stack((sq_side1 + sq_side2 < sq_side3, sq_side2 + sq_side3 < sq_side1, sq_side3 + sq_side1 < sq_side2), dim=1).any(dim=1)

    #     area = torch.cross(side_1, side_2, dim=1).norm(dim=1) / 2 + 1e-10
    #     circumradius = l_side1 * l_side2 * l_side3 / (4 * area)
    #     l_side_max = torch.stack((l_side1, l_side2, l_side3), dim=1).max(dim=1).values
    #     circumradius[is_obtuse] = l_side_max[is_obtuse] / 2

    #     return circumradius

    @property
    def get_scaling(self):
        l_side1 = (self._vertex[:, 2] - self._vertex[:, 1]).norm(dim=1)
        l_side2 = (self._vertex[:, 0] - self._vertex[:, 2]).norm(dim=1)
        l_side3 = (self._vertex[:, 1] - self._vertex[:, 0]).norm(dim=1)
        return torch.stack((l_side1, l_side2, l_side3), dim=1).mean(dim=1)

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
        self._vertex = nn.Parameter(torch.empty(point_count, 3, 3).float().to(self.device), requires_grad=True)
        self._opacity = nn.Parameter(torch.empty(point_count, 1).float().to(self.device), requires_grad=True)
        self._f_dc = nn.Parameter(torch.empty(point_count, 1, 3).float().to(self.device), requires_grad=True)
        self._f_rest = nn.Parameter(torch.empty(point_count, (self.max_sh_degree + 1) ** 2 - 1, 3).float().to(self.device), requires_grad=True)

    def _setup_optimizer(self):
        if self.config.optimizer is None:
            return

        l = [
            {"params": [self._vertex], "lr": 0, "name": "vertex"},
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

        v_scheduler_ = exponential_scheduler(**vars(args.vertex))
        if args.vertex_scale_up_iter is not None and args.vertex_scale_up is not None:
            v_scheduler = lambda iteration: v_scheduler_(iteration) * (1.0 if iteration <= args.vertex_scale_up_iter else args.vertex_scale_up)
        else:
            v_scheduler = v_scheduler_

        self.lr_schedulers = {
            "vertex": v_scheduler,
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

        if args.densification is not None:
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
            if args.gamma_schedule.step_scheduler:
                self.gamma_scheduler = exponential_step_scheduler(
                    v_init=args.gamma_schedule.gamma_init,
                    v_final=args.gamma_schedule.gamma_final,
                    max_steps=args.gamma_schedule.end_iter - args.gamma_schedule.start_iter,
                    n_stage=args.gamma_schedule.n_stage,
                )
            else:
                self.gamma_scheduler = exponential_scheduler(
                    v_init=args.gamma_schedule.gamma_init,
                    v_final=args.gamma_schedule.gamma_final,
                    max_steps=args.gamma_schedule.end_iter - args.gamma_schedule.start_iter,
                )

        self.gradient_accum = torch.zeros((self._vertex.shape[0]), device=self.device)
        self.gradient_denom = torch.zeros((self._vertex.shape[0]), device=self.device)
        self.max_radii2D = torch.zeros((self._vertex.shape[0]), device=self.device)
        self.contrib_sum = torch.zeros((self._vertex.shape[0]), device=self.device)
        self.contrib_max = torch.zeros((self._vertex.shape[0]), device=self.device)
        self.contrib_denom = torch.zeros((self._vertex.shape[0]), device=self.device)

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
        large_point_mask = self.get_scaling > split_scale_threshold
        clone_mask = grow_mask & ~large_point_mask  # clone small points
        split_mask = grow_mask & large_point_mask  # split large points

        clone_vertex = self._vertex[clone_mask]
        clone_opacity = self._opacity[clone_mask]
        clone_f_dc = self._f_dc[clone_mask]
        clone_f_rest = self._f_rest[clone_mask]

        N = 2
        split_vertex_ori = self._vertex[split_mask]
        side1 = split_vertex_ori[:, 2] - split_vertex_ori[:, 1]
        side2 = split_vertex_ori[:, 0] - split_vertex_ori[:, 2]
        side3 = split_vertex_ori[:, 1] - split_vertex_ori[:, 0]
        l_side = torch.argmax(torch.stack((side1.norm(dim=1), side2.norm(dim=1), side3.norm(dim=1)), dim=1), dim=1)  # longest side
        l_side_p1 = (l_side + 1) % 3
        l_side_p2 = (l_side + 2) % 3
        r = torch.arange(split_vertex_ori.shape[0], device=self.device)
        l_side_center = (split_vertex_ori[r, l_side_p1] + split_vertex_ori[r, l_side_p2]) / 2  # center point of the longest side
        split_triangle_1 = torch.stack((split_vertex_ori[r, l_side], split_vertex_ori[r, l_side_p1], l_side_center), dim=1)
        split_triangle_2 = torch.stack((split_vertex_ori[r, l_side], l_side_center, split_vertex_ori[r, l_side_p2]), dim=1)
        split_vertex = torch.cat((split_triangle_1, split_triangle_2), dim=0)
        split_opacity = self._opacity[split_mask].repeat(N, 1)
        split_f_dc = self._f_dc[split_mask].repeat(N, 1, 1)
        split_f_rest = self._f_rest[split_mask].repeat(N, 1, 1)

        new_vertex = torch.cat((clone_vertex, split_vertex), dim=0)
        new_opacity = torch.cat((clone_opacity, split_opacity), dim=0)
        new_f_dc = torch.cat((clone_f_dc, split_f_dc), dim=0)
        new_f_rest = torch.cat((clone_f_rest, split_f_rest), dim=0)

        new_points = {
            "vertex": new_vertex,
            "opacity": new_opacity,
            "f_dc": new_f_dc,
            "f_rest": new_f_rest,
        }

        self._prune_points(split_mask)
        self._grow_points_update_states(new_points)

        new_points_count = new_vertex.shape[0]
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

    def _training_statistic(self, iteration: int, render_pkg: dict[str, torch.Tensor] = None):
        args = self.config.model_update.statistic
        if args is None or not (args.start_iter < iteration <= args.end_iter) or render_pkg is None:
            return

        center2D = render_pkg["center2D"]
        visible_mask = render_pkg["visible_mask"]
        radii = render_pkg["radii"]
        contrib_sum = render_pkg["contrib_sum"]
        contrib_max = render_pkg["contrib_max"]

        self.gradient_accum[visible_mask] += torch.norm(center2D.grad[visible_mask, :2], dim=-1)
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
        select_mask = self.gradient_denom >= min_view_count
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
        scale_prune_mask = self.get_scaling > scale_threshold
        prune_mask = radii_prune_mask | scale_prune_mask

        self._prune_points(prune_mask)
        self.logger.info(
            f"[ITER {iteration}, scale pruning] Pruning {prune_mask.sum().item()} points, "
            + f"{radii_prune_mask.sum().item()} by radii, {scale_prune_mask.sum().item()} by scale, "
            + f"radii threshold: {radii_threshold}, scale threshold: {scale_threshold}"
        )

    def _rescale_triangles(self, rescale_ratio: torch.Tensor | float, rescale_mask: torch.Tensor = None):
        """
        rescale triangles by rescale_ratio
        args:
            rescale_ratio: (M, ) tensor or a float, rescale ratio for each triangle
            rescale_mask: (N, ) tensor, mask for triangles to rescale
        return:
            rescaled_vertex: (N, 3, 3) tensor, rescaled vertex
        """
        vertex = self._vertex[rescale_mask] if rescale_mask is not None else self._vertex
        if isinstance(rescale_ratio, torch.Tensor):
            assert rescale_ratio.dim() == 1 and rescale_ratio.size(0) == vertex.size(0)
            rescale_ratio = rescale_ratio.unsqueeze(1).unsqueeze(1)

        t_center = vertex.mean(dim=1, keepdim=True)
        rescaled_vertex = (vertex - t_center) * rescale_ratio + t_center
        return rescaled_vertex

    def _scale_clipping(self, iteration: int):
        args = self.config.model_update.scale_clipping
        if args is None or not (args.start_iter < iteration <= args.hold_iter and iteration % args.interval_iter == 0):
            return

        scale_max = self.scale_max_scheduler(iteration - args.start_iter)
        scaling = self.get_scaling
        clip_mask = scaling > scale_max

        rescale_ratio = scale_max / scaling[clip_mask]
        rescaled_vertex = self._rescale_triangles(rescale_ratio, clip_mask)

        clip_count = clip_mask.sum().item()

        if clip_count > 0:
            self._clipping_update_states(clip_mask, rescaled_vertex, "vertex")
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

        total_point_count = self._vertex.shape[0]
        inside_mask = get_inside_mask(self.get_xyz, self.scene_bbox)
        ste_mask = (
            self.get_opacity > self.ste_threshold if self.ste_threshold is not None else torch.ones_like(self.get_opacity, dtype=torch.bool)
        ).squeeze()
        valid_point_count = (inside_mask & ste_mask).sum().item()
        # inside_point_count = get_inside_mask(self.get_xyz, self.scene_bbox).sum().item()
        select_mask = self.contrib_denom >= min_view_count
        select_count = select_mask.sum().item()
        diff_point_count = max(0, valid_point_count - target_point_num * 0.99) * total_point_count / valid_point_count
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
            dist = inter_point_distance(self.get_xyz)
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
            + f"target inside point count: {target_point_num}, valid point count before pruning: {valid_point_count}"
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
    def model_update(self, iteration: int, render_pkg: dict[str, torch.Tensor] = None):
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
        # back_culling = back_culling if back_culling is not None else self.back_culling
        sh_degree = sh_degree if sh_degree is not None else self.active_sh_degree
        gamma = gamma if gamma is not None else self.gamma
        if back_culling is None:
            if not is_training:
                back_culling = self.back_culling
            elif self.back_culling and torch.rand(1).item() < self.back_culling_prob:
                back_culling = True
            else:
                back_culling = False

        bg_color = get_color_tensor(background).to(self.device)

        vertex = self.get_vertex
        shs = self.get_features
        opacity = self.get_opacity

        # rescale triangles to keep the integration of triangle opacity invariant under different gamma
        if self.gamma_rescale:
            beta = 1 / gamma
            rescale_ratio = 1 / np.sqrt(2**beta * beta * scipy.special.gamma(beta))
            vertex_rescale = self._rescale_triangles(rescale_ratio)

        if self.ste_threshold is not None:
            opacity_ste = ((opacity > self.ste_threshold).float() - opacity).detach() + opacity

        bg_depth = (camera.camera_center.view(1, 1, 3) - vertex).norm(dim=-1).max()

        if self.render_up_scale and self.render_up_scale > 1:
            assert isinstance(self.render_up_scale, int)
            w, h = camera.image_width, camera.image_height
            camera = deepcopy(camera)
            camera.image_width = w * self.render_up_scale
            camera.image_height = h * self.render_up_scale

        renderer = TriangleRenderer(
            camera,
            bg_depth=bg_depth,
            bg_color=bg_color,
            sh_degree=min(sh_degree, self.max_sh_degree),
            gamma=gamma,
            back_culling=back_culling,
            rich_info=is_training,
            rasterizer_type=self.rasterizer_type,
        )
        output_pkg = renderer.render(
            vertex_rescale if self.gamma_rescale else vertex,
            shs,
            None,
            opacity_ste if self.ste_threshold is not None else opacity,
        )

        if self.render_up_scale and self.render_up_scale > 1:
            output_pkg["render"] = F.interpolate(output_pkg["render"].unsqueeze(0), size=(h, w), mode="bilinear").squeeze(0)
            if "radii" in output_pkg:
                output_pkg["radii"] = output_pkg["radii"] // self.render_up_scale
            if "depth" in output_pkg:
                output_pkg["depth"] = F.interpolate(output_pkg["depth"].unsqueeze(0).unsqueeze(0), size=(h, w), mode="bilinear").squeeze(0).squeeze(0)
            if "normal" in output_pkg:
                output_pkg["normal"] = F.interpolate(output_pkg["normal"].unsqueeze(0), size=(h, w), mode="bilinear").squeeze(0)

        render_pkg = {
            "render": output_pkg["render"],
        }
        if is_training:
            training_info = {
                "gt_image": camera.gt_image,
                "gt_mask": camera.alpha_mask,
                "radii": output_pkg["radii"],
                "center2D": output_pkg["center2D"],
                "contrib_sum": output_pkg["contrib_sum"],
                "contrib_max": output_pkg["contrib_max"],
                "depth": output_pkg["depth"],
                "normal": output_pkg["normal"],
                "scaling": self.get_scaling,
                "opacity": opacity,
                "vertex": vertex,
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

    def savePLY(self, ply_path: str, bbox_filtering: bool = True):
        self.logger.info(f"Saving triangles to {ply_path}")
        triangle_model = self.toRawTriangle(bbox_filtering)
        triangle_model.savePLY(ply_path, save_extra=True)

    def loadPLY(self, ply_path: str) -> "VanillaTSModel":
        self.logger.info(f"Loading triangles from {ply_path}")
        triangle_model = RawTriangle(ply_path=ply_path)
        return self.fromRawTriangle(triangle_model)

    @torch.no_grad()
    def toRawTriangle(self, bbox_filtering: bool = True) -> RawTriangle:
        vertex = self._vertex
        opacity = self._opacity
        shs = self.get_features.view(vertex.shape[0], -1)

        if bbox_filtering and self.scene_bbox is not None:
            mask = get_inside_mask(self.get_xyz, self.scene_bbox)
            vertex = vertex[mask]
            opacity = opacity[mask]
            shs = shs[mask]

        if self.ste_threshold is not None:
            ste_mask = torch.sigmoid(opacity).squeeze() > self.ste_threshold
            vertex = vertex[ste_mask]
            opacity = ste_mask[ste_mask].unsqueeze(-1).float() * 10
            shs = shs[ste_mask]
        return RawTriangle(*to_numpy(vertex, opacity, shs))

    def fromRawTriangle(self, triangle_model: RawTriangle) -> "VanillaTSModel":
        np = triangle_model.vertex.shape[0]
        shs = torch.tensor(triangle_model.shs).float().to(self.device).view(np, -1, 3)
        features = torch.zeros((np, (self.max_sh_degree + 1) ** 2, 3)).float().to(self.device)
        if shs.shape[1] > features.shape[1]:
            features = shs[:, : features.shape[1], :]
        else:
            features[:, : shs.shape[1], :] = shs

        self._vertex = nn.Parameter(torch.tensor(triangle_model.vertex).float().to(self.device), requires_grad=True)
        self._opacity = nn.Parameter(torch.tensor(triangle_model.opacity).float().to(self.device), requires_grad=True)
        self._f_dc = nn.Parameter(features[:, :1].contiguous(), requires_grad=True)
        self._f_rest = nn.Parameter(features[:, 1:].contiguous(), requires_grad=True)

        self._training_setup()
        return self

    def saveGLB(self, glb_path: str, bbox_filtering: bool = True, process: bool = False):
        self.logger.info(f"Saving triangles to {glb_path}")
        triangle_model = self.toRawTriangle(bbox_filtering)
        triangle_model.saveGLB(glb_path, save_back=not self.back_culling, process=process)

    def loadGLB(self, glb_path: str) -> "VanillaTSModel":
        self.logger.info(f"Loading triangles from {glb_path}")
        triangle_model = RawTriangle(glb_path=glb_path)
        return self.fromRawTriangle(triangle_model)

    def save_ckpt(self, ckpt_path: str):
        self.logger.info(f"Saving checkpoint to {ckpt_path}")
        Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)

        save_items = (self.state_dict(), self.optimizer.state_dict(), self.scene_bbox, self.gamma)
        torch.save(save_items, open(ckpt_path, "wb"))

    def load_ckpt(self, ckpt_path: str) -> "VanillaTSModel":
        (params_state_dict, optimizer_state_dict, self.scene_bbox, self.gamma) = torch.load(open(ckpt_path, "rb"))

        point_count = params_state_dict["_vertex"].shape[0]
        self._setup_parameters(point_count)
        self.load_state_dict(params_state_dict)

        self._training_setup()
        self.optimizer.load_state_dict(optimizer_state_dict)
        return self

    def _sample_points(
        self,
        points: torch.Tensor,
        shs: torch.Tensor,
        normals: torch.Tensor,
        name: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            if n_sample > points.shape[0] or n_sample <= 0:
                self.logger.warning(f"target sample number {n_sample} is invalid, using all points")
                return points, shs, normals
            sample_idx = torch.randperm(points.shape[0])[:n_sample]
            sampled_points, sampled_shs, sampled_normals = points[sample_idx], shs[sample_idx], normals[sample_idx]
        elif sample_method == "grid":
            grid_size = grid_size_search(points, n_sample) if grid_size is None else grid_size
            sampled_points, sampled_shs, sampled_normals = grid_sampling(points, shs, normals, grid_size=grid_size)
            sampled_normals = sampled_normals / sampled_normals.norm(dim=1, keepdim=True)
        elif sample_method == "direct":
            sampled_points, sampled_shs, sampled_normals = points, shs, normals
        else:
            raise ValueError(f"Unknown sampling method: {sample_method}")

        self.logger.info(f"Sampled {sampled_points.shape[0]} points from {points.shape[0]} points using {sample_method} sampling")
        if sample_method == "grid":
            self.logger.info(f"Grid size: {grid_size:.5f}")

        return sampled_points, sampled_shs, sampled_normals

    def random_pcd(self) -> PointCloud:
        config = self.config.random_init
        if config is None:
            raise ValueError("Random initialization config is not provided")

        bbox_list = config.bbox_list
        point_num_list = config.point_num_list
        normal_list = config.normal_list

        pcd = PointCloud()
        for bbox, point_num, normal in zip(bbox_list, point_num_list, normal_list):
            bbox = np.array(bbox, dtype=np.float32)
            points = np.random.rand(point_num, 3).astype(np.float32) * (bbox[3:] - bbox[:3]) + bbox[:3]
            colors = np.random.rand(point_num, 3).astype(np.float32)
            if normal == "random":
                normals = np.random.randn(point_num, 3).astype(np.float32)
                normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
            else:
                normals = np.tile(np.array(normal, dtype=np.float32), (point_num, 1))
                normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
            pcd += PointCloud(points=points, colors=colors, normals=normals)

        return pcd

    def create_from_pcd(self, pcd: PointCloud):
        if pcd is None or len(pcd) == 0:
            pcd = self.random_pcd()

        args = self.config.sampling
        if args is None:
            raise ValueError("Sampling config is not provided")

        init_opacity = args.init_opacity if args.init_opacity is not None else 0.1
        duplicate_count = args.duplicate_count if args.duplicate_count is not None else 1

        points = torch.tensor(np.asarray(pcd.points)).float().to(self.device)
        shs = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().to(self.device))
        normals = torch.tensor(np.asarray(pcd.normals)).float().to(self.device)
        if not normals.any():
            normals = torch.randn_like(points)
        normals = normals / normals.norm(dim=1, keepdim=True)

        inside_mask = get_inside_mask(points, self.scene_bbox)

        inside_points = points[inside_mask]
        inside_shs = shs[inside_mask]
        inside_normals = normals[inside_mask]
        inside_points, inside_shs, inside_normals = self._sample_points(inside_points, inside_shs, inside_normals, "inside")

        outside_points = points[~inside_mask]
        outside_shs = shs[~inside_mask]
        outside_normals = normals[~inside_mask]
        outside_points, outside_shs, outside_normals = self._sample_points(outside_points, outside_shs, outside_normals, "outside")

        points = torch.cat((inside_points, outside_points), dim=0)
        shs = torch.cat((inside_shs, outside_shs), dim=0)
        normals = torch.cat((inside_normals, outside_normals), dim=0)
        scaling = inter_point_distance(points)[..., None]

        if init_opacity == "random":
            opacities = inverse_sigmoid(torch.rand((points.shape[0], 1)).float().to(self.device))
        else:
            opacities = inverse_sigmoid(torch.ones((points.shape[0], 1)).float().to(self.device) * init_opacity)
        features = torch.zeros((shs.shape[0], (self.max_sh_degree + 1) ** 2, 3)).float().to(self.device)
        features[:, 0, :] = shs

        # repeat points
        if duplicate_count > 1:
            self.logger.info(f"Duplicate points {duplicate_count} times")
            points_random = [points]
            for i in range(duplicate_count - 1):
                random_offset = torch.rand((scaling.shape[0], 3)).float().to(self.device)
                random_offset = (random_offset * 2 - 1) * 0.5 * scaling
                points_random.append(points + random_offset.to(points.device))

            points = torch.cat(points_random, dim=0)
            opacities = opacities.repeat(duplicate_count, 1)
            features = features.repeat(duplicate_count, 1, 1)
            normals = normals.repeat(duplicate_count, 1)
            scaling = inter_point_distance(points)[..., None]

            # pcd_test = PointCloud(points=points.cpu().numpy(), colors=(SH2RGB(features[:, 0, :]).squeeze(1).cpu().numpy() * 255.0).astype(np.uint8), normals=normals.cpu().numpy())
            # pcd_test.storePly("test.ply")

        # make equilateral triangles
        up = torch.tensor([0, 0, 1]).float().to(self.device).repeat(points.shape[0], 1)
        u_dir = torch.cross(up, normals, dim=1)
        u_dir[u_dir.norm(dim=1) < 1e-10] = torch.tensor([1, 0, 0]).float().to(self.device)
        u_dir = u_dir / u_dir.norm(dim=1, keepdim=True)
        v_dir = torch.cross(normals, u_dir, dim=1)
        v_dir[v_dir.norm(dim=1) < 1e-10] = torch.tensor([0, 1, 0]).float().to(self.device)
        v_dir = v_dir / v_dir.norm(dim=1, keepdim=True)

        v1 = points + u_dir * scaling
        v2 = points + (-1 / 2 * u_dir + np.sqrt(3) / 2 * v_dir) * scaling
        v3 = points + (-1 / 2 * u_dir - np.sqrt(3) / 2 * v_dir) * scaling
        vertex = torch.stack((v1, v2, v3), dim=1)

        if self.back_culling:
            vertex_back = torch.stack((v3, v2, v1), dim=1)
            vertex = torch.cat((vertex, vertex_back), dim=0)
            opacities = torch.cat((opacities, opacities), dim=0)
            features = torch.cat((features, features), dim=0)

        self.logger.info(f"Number of points at initialisation: {vertex.shape[0]}")

        self._vertex = nn.Parameter(vertex, requires_grad=True)
        self._opacity = nn.Parameter(opacities, requires_grad=True)
        self._f_dc = nn.Parameter(features[:, :1].contiguous(), requires_grad=True)
        self._f_rest = nn.Parameter(features[:, 1:].contiguous(), requires_grad=True)

        self._training_setup()
