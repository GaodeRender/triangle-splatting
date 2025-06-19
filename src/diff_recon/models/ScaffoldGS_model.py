import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_max
from functools import reduce
import numpy as np
from pathlib import Path

from .point_cloud import PointCloud
from .raw_gaussian import RawGaussian
from .model_utils import *
from .Base_model import BaseModel
from ..renderer.gaussian_renderer import GaussianRenderer
from ..utils.config import Config
from ..utils.logger import Logger
from ..utils.scheduler import exponential_scheduler
from ..utils.camera import Camera
from ..utils.sh_utils import RGB2SH, SH2RGB


class ScaffoldGSModel(BaseModel):
    def __init__(self, config: Config, logger: Logger = None, device: torch.device| int| None = None):
        super().__init__(config, logger, device)

        self.feat_dim = self.config.feat_dim
        self.hidden_dim = self.config.hidden_dim
        self.n_offsets = self.config.n_offsets
        self.voxel_size = self.config.voxel_size

        self.optimizer = None
        self.opacity_threshold = 0.0
        self.scene_bbox = None

        self.mlp_scaling = nn.Sequential(
            nn.Linear(self.feat_dim, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, 6),
        ).to(self.device)

        self.mlp_offset = nn.Sequential(
            nn.Linear(self.feat_dim, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, 3 * self.n_offsets),
            nn.Tanh(),
        ).to(self.device)

        self.mlp_opacity = nn.Sequential(
            nn.Linear(self.feat_dim, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, 1 * self.n_offsets),
            nn.Sigmoid(),
        ).to(self.device)

        self.mlp_cov = nn.Sequential(
            nn.Linear(self.feat_dim, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, 7 * self.n_offsets),
        ).to(self.device)

        self.mlp_color = nn.Sequential(
            nn.Linear(self.feat_dim, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, 3 * self.n_offsets),
            nn.Sigmoid(),
        ).to(self.device)

    @property
    def anchor_count(self):
        return self.anchor.shape[0]

    def _get_anchor_scaling(self, feat):
        scaling = torch.exp(self.mlp_scaling(feat)) * self.voxel_size
        offset_scale = scaling[:, :3].clamp(max=self.config.max_offset_scale)
        scaling_scale = scaling[:, 3:].clamp(max=self.config.max_scaling_scale)
        scaling = torch.cat([offset_scale, scaling_scale], dim=1)

        return scaling

    def _get_gaussian_xyz(self, anchor_mask=None):
        anchor = self.anchor
        feat = self.anchor_feat
        if anchor_mask is not None:
            anchor = anchor[anchor_mask]
            feat = feat[anchor_mask]

        scaling = self._get_anchor_scaling(feat)[:, :3]
        scaling = scaling.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).reshape([-1, 3])

        anchor = anchor.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).reshape([-1, 3])
        g_offset = self.mlp_offset(feat).reshape([-1, 3])
        g_xyz = anchor + scaling * g_offset

        return g_xyz

    def _get_gaussian_xyz_scale_rot(self, anchor_mask=None):
        anchor = self.anchor
        feat = self.anchor_feat
        if anchor_mask is not None:
            anchor = anchor[anchor_mask]
            feat = feat[anchor_mask]

        scaling = self._get_anchor_scaling(feat)
        scaling = scaling.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).reshape([-1, 6])

        anchor = anchor.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).reshape([-1, 3])
        g_offset = self.mlp_offset(feat).reshape([-1, 3])
        g_xyz = anchor + scaling[:, :3] * g_offset

        g_cov = self.mlp_cov(feat).reshape([-1, 7])
        g_scale = scaling[:, 3:] * torch.sigmoid(g_cov[:, :3])
        g_rot = F.normalize(g_cov[:, 3:7], dim=-1)

        return g_xyz, g_scale, g_rot

    def _get_gaussian_opacity(self, anchor_mask=None):
        feat = self.anchor_feat
        if anchor_mask is not None:
            feat = feat[anchor_mask]

        g_opacity = self.mlp_opacity(feat).reshape([-1, 1])

        return g_opacity

    def _get_gaussian_color(self, anchor_mask=None):
        feat = self.anchor_feat
        if anchor_mask is not None:
            feat = feat[anchor_mask]

        g_color = self.mlp_color(feat).reshape([-1, 3])

        return g_color

    def setup_scene_info(self, scene_info: dict = None):
        if scene_info is None:
            return

        self.scene_bbox = scene_info["bbox_xyz"]

    def _setup_parameters(self, anchor_count: int):
        self.anchor = nn.Parameter(torch.empty(anchor_count, 3).float().to(self.device), requires_grad=True)
        # self.offset = nn.Parameter(torch.empty(anchor_count, self.n_offsets, 3).float().to(self.device), requires_grad=True)
        self.anchor_feat = nn.Parameter(torch.empty(anchor_count, self.feat_dim).float().to(self.device), requires_grad=True)
        self.scaling = nn.Parameter(torch.empty(anchor_count, 3).float().to(self.device), requires_grad=False)
        self.rotation = nn.Parameter(torch.empty(anchor_count, 4).float().to(self.device), requires_grad=False)

    def _setup_optimizer(self):
        l = [
            {"params": [self.anchor], "lr": 0, "name": "anchor"},
            # {"params": [self.offset], "lr": 0, "name": "offset"},
            {"params": [self.anchor_feat], "lr": 0, "name": "anchor_feat"},
            {"params": [self.scaling], "lr": 0, "name": "scaling"},
            {"params": [self.rotation], "lr": 0, "name": "rotation"},
            {"params": self.mlp_offset.parameters(), "lr": 0, "name": "mlp_offset"},
            {"params": self.mlp_opacity.parameters(), "lr": 0, "name": "mlp_opacity"},
            {"params": self.mlp_cov.parameters(), "lr": 0, "name": "mlp_cov"},
            {"params": self.mlp_color.parameters(), "lr": 0, "name": "mlp_color"},
            {"params": self.mlp_scaling.parameters(), "lr": 0, "name": "mlp_scaling"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def _setup_lr_scheduler(self):
        o_args = self.config.optimizer

        self.lr_schedulers = {
            "anchor": exponential_scheduler(**vars(o_args.anchor)),
            "anchor_feat": exponential_scheduler(**vars(o_args.ancho_feat)),
            # "scaling": exponential_scheduler(**vars(o_args.scaling)),
            # "rotation": exponential_scheduler(**vars(o_args.rotation)),
            "mlp_offset": exponential_scheduler(**vars(o_args.mlp_offset)),
            "mlp_opacity": exponential_scheduler(**vars(o_args.mlp_opacity)),
            "mlp_cov": exponential_scheduler(**vars(o_args.mlp_cov)),
            "mlp_color": exponential_scheduler(**vars(o_args.mlp_color)),
            "mlp_scaling": exponential_scheduler(**vars(o_args.mlp_scaling)),
        }

    def _setup_anchor_update_utils(self):
        u_args = self.config.anchor_update

        self.grad_threshold_scheduler = exponential_scheduler(
            v_init=u_args.grad_threshold_init,
            v_final=u_args.grad_threshold_final,
            max_steps=u_args.end_iter - u_args.start_iter,
        )
        self.opacity_threshold_scheduler = exponential_scheduler(
            v_init=u_args.opacity_threshold_init,
            v_final=u_args.opacity_threshold_final,
            max_steps=u_args.end_iter - u_args.start_iter,
        )
        self.anchor_update_start_iter = u_args.start_iter
        self.grad_min_view_count = u_args.grad_min_view_count
        self.opacity_min_view_count = u_args.opacity_min_view_count

        self.update_depth = u_args.update_depth
        self.update_init_factor = u_args.update_init_factor
        self.update_hierachy_factor = u_args.update_hierachy_factor

        self.opacity_accum = torch.zeros((self.anchor.shape[0]), dtype=torch.float32).to(self.device)
        self.anchor_denom = torch.zeros((self.anchor.shape[0]), dtype=torch.float32).to(self.device)

        self.offset_gradient_accum = torch.zeros((self.anchor.shape[0] * self.n_offsets), dtype=torch.float32).to(self.device)
        self.offset_denom = torch.zeros((self.anchor.shape[0] * self.n_offsets), dtype=torch.float32).to(self.device)

    def _training_setup(self):
        self._setup_optimizer()
        self._setup_lr_scheduler()
        self._setup_anchor_update_utils()

    def update_learning_rate(self, iteration: int):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] in self.lr_schedulers:
                param_group["lr"] = self.lr_schedulers[param_group["name"]](iteration)

    def maintain_constraints(self, iteration: int):
        self.opacity_threshold = self.opacity_threshold_scheduler(iteration - self.anchor_update_start_iter)

    def training_statistic(self, iteration: int, render_pkg: dict[str, torch.Tensor]):
        u_args = self.config.anchor_update
        if not u_args.start_iter < iteration <= u_args.end_iter:
            return

        viewspace_points = render_pkg["viewspace_points"]
        gaussian_opacity = render_pkg["gaussian_opacity"]
        gaussian_visible_mask = render_pkg["gaussian_visible_mask"]
        offset_selection_mask = render_pkg["offset_selection_mask"]
        anchor_visible_mask = render_pkg["anchor_visible_mask"]

        with torch.no_grad():
            # update opacity stats
            max_opacity = gaussian_opacity.view([-1, self.n_offsets]).max(dim=1).values
            self.opacity_accum[anchor_visible_mask] += max_opacity
            self.anchor_denom[anchor_visible_mask] += 1

            # update gradient stats
            combined_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
            combined_mask[combined_mask.clone()] = offset_selection_mask
            combined_mask[combined_mask.clone()] = gaussian_visible_mask

            grad_norm = torch.norm(viewspace_points.grad[gaussian_visible_mask, :2], dim=-1)
            self.offset_gradient_accum[combined_mask] += grad_norm
            self.offset_denom[combined_mask] += 1

    def _prune_anchor_update_states(self, mask):
        # update optimizer and parameter after runing _prune_anchor
        for group in self.optimizer.param_groups:
            if any(x in group["name"] for x in ["mlp", "conv", "feat_base", "embedding"]):
                continue

            param = group["params"][0]
            new_param = nn.Parameter(param[mask].requires_grad_(True))

            stored_state = self.optimizer.state.pop(param, None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                self.optimizer.state[new_param] = stored_state

            group["params"][0] = new_param
            self.__setattr__(group["name"], new_param)

    def _prune_anchor(self, anchor_mask, opacity, opacity_threshold):
        self.opacity_accum[anchor_mask] = 0
        self.anchor_denom[anchor_mask] = 0

        prune_mask = opacity < opacity_threshold
        prune_mask = prune_mask & anchor_mask

        removed_anchor_count = prune_mask.sum().item()
        if removed_anchor_count > 0:
            self._prune_anchor_update_states(~prune_mask)

            self.anchor_denom = self.anchor_denom[~prune_mask]
            self.opacity_accum = self.opacity_accum[~prune_mask]
            self.offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask].view(-1)
            self.offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask].view(-1)

        return removed_anchor_count

    def _grow_anchor_update_states(self, tensors_dict):
        # update optimizer and parameter after runing _grow_anchor
        for group in self.optimizer.param_groups:
            if any(x in group["name"] for x in ["mlp", "conv", "feat_base", "embedding"]):
                continue

            param = group["params"][0]
            extension_tensor = tensors_dict[group["name"]]
            new_param = nn.Parameter(torch.cat((param, extension_tensor), dim=0).requires_grad_(True))

            stored_state = self.optimizer.state.pop(param, None)
            if stored_state:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                self.optimizer.state[new_param] = stored_state

            group["params"][0] = new_param
            self.__setattr__(group["name"], new_param)

    def _grow_anchor(self, offset_mask, grad, grad_threshold):
        self.offset_denom[offset_mask] = 0
        self.offset_gradient_accum[offset_mask] = 0

        new_anchor_dict = {}
        for i in range(self.update_depth):
            if i > 0 and not "anchor" in new_anchor_dict:
                return 0

            cur_threshold = grad_threshold * ((self.update_hierachy_factor // 2) ** i)
            candidate_mask = grad >= cur_threshold
            candidate_mask = candidate_mask & offset_mask

            rand_mask = torch.rand_like(candidate_mask.float()) > (0.5 ** (i + 1))
            rand_mask = rand_mask.to(self.device)
            candidate_mask = candidate_mask & rand_mask

            size_factor = self.update_init_factor // (self.update_hierachy_factor**i)
            cur_size = self.voxel_size * size_factor

            grid_coords = torch.round(self.anchor / cur_size).int()

            selected_grid_coords = torch.round(self._get_gaussian_xyz()[candidate_mask] / cur_size).int()
            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)

            # split data for reducing peak memory calling
            use_chunk = True
            if use_chunk:
                chunk_size = 40960
                max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
                remove_duplicates_list = []
                for i in range(max_iters):
                    cur_remove_duplicates = (
                        (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i * chunk_size : (i + 1) * chunk_size, :]).all(-1).any(-1).view(-1)
                    )
                    remove_duplicates_list.append(cur_remove_duplicates)

                duplicate_mask = reduce(torch.logical_or, remove_duplicates_list)
            else:
                duplicate_mask = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)

            candidate_anchor = selected_grid_coords_unique[~duplicate_mask] * cur_size

            if candidate_anchor.shape[0] > 0:
                # new_offsets = torch.empty((candidate_anchor.shape[0], self.n_offsets, 3)).normal_(std=0.25).float().to(self.device)
                new_feat = self.anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]
                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][~duplicate_mask]
                new_scaling = torch.ones_like(candidate_anchor).float().to(self.device) * self.config.max_offset_scale
                new_rot = torch.tensor([1, 0, 0, 0]).repeat(candidate_anchor.shape[0], 1).float().to(self.device)

                d = {
                    "anchor": candidate_anchor,
                    # "offset": new_offsets,
                    "anchor_feat": new_feat,
                    "scaling": new_scaling,
                    "rotation": new_rot,
                }
                cat_tensor_dict(new_anchor_dict, d)

        unique_indices = get_first_unique_indices(new_anchor_dict["anchor"], dim=0)
        filter_tensor_dict(new_anchor_dict, unique_indices)
        self._grow_anchor_update_states(new_anchor_dict)

        new_anchor_count = self.anchor.shape[0] - self.anchor_denom.shape[0]
        self.offset_denom = F.pad(self.offset_denom, (0, new_anchor_count * self.n_offsets), value=0)
        self.offset_gradient_accum = F.pad(self.offset_gradient_accum, (0, new_anchor_count * self.n_offsets), value=0)
        self.anchor_denom = F.pad(self.anchor_denom, (0, new_anchor_count), value=0)
        self.opacity_accum = F.pad(self.opacity_accum, (0, new_anchor_count), value=0)
        return new_anchor_count

    def anchor_update(self, iteration: int, grow_anchor: bool = True, prune_anchor: bool = True):
        u_args = self.config.anchor_update
        if not (u_args.start_iter < iteration <= u_args.end_iter and u_args.interval_iter > 0 and iteration % u_args.interval_iter == 0):
            return

        with torch.no_grad():
            if grow_anchor:
                grad_threshold = self.grad_threshold_scheduler(iteration - self.anchor_update_start_iter)
                offset_mask = self.offset_denom > self.grad_min_view_count
                grad = self.offset_gradient_accum / (1e-15 + self.offset_denom)
                added_anchor_count = self._grow_anchor(offset_mask, grad, grad_threshold)
                self.logger.info(f"[ITER {iteration}] grad threshold: {grad_threshold}, added {added_anchor_count} anchors")

            if prune_anchor:
                opacity_threshold = self.opacity_threshold_scheduler(iteration - self.anchor_update_start_iter)
                anchor_mask = self.anchor_denom > self.opacity_min_view_count
                opacity = self.opacity_accum / (1e-15 + self.anchor_denom)
                removed_anchor_count = self._prune_anchor(anchor_mask, opacity, opacity_threshold)
                self.logger.info(f"[ITER {iteration}] opacity threshold: {opacity_threshold}, removed {removed_anchor_count} anchors")

    def prefilter_voxel(self, camera: Camera) -> torch.Tensor:
        with torch.no_grad():
            means3D = self.anchor
            scales = self.scaling
            rotations = self.rotation

            radii_pure = GaussianRenderer(camera).get_radii(means3D=means3D, scales=scales, rotations=rotations)
            visible_mask = radii_pure > 0

        return visible_mask

    def generate_gaussians(
        self,
        visible_mask: torch.Tensor = None,
        is_training: bool = False,
        tile_filtering: bool = False,
    ) -> dict[str, torch.Tensor]:
        g_color = self._get_gaussian_color(visible_mask)
        g_opacity = self._get_gaussian_opacity(visible_mask)
        g_xyz, g_scaling, g_rot = self._get_gaussian_xyz_scale_rot(visible_mask)

        mask = (g_opacity > self.opacity_threshold).squeeze(dim=1)
        if tile_filtering:
            inside_mask = get_inside_mask(g_xyz, self.scene_bbox)
            mask = mask & inside_mask

        gaussian_pkg = {
            "xyz": g_xyz[mask],  # [N, 3]
            "color": g_color[mask],  # [N, 3]
            "opacity": g_opacity[mask],  # [N, 1]
            "scaling": g_scaling[mask],  # [N, 3]
            "rot": g_rot[mask],  # [N, 4]
        }
        if is_training:
            training_info = {
                "gaussian_opacity": g_opacity,  # [N_grid * n_offsets, 1]
                "offset_selection_mask": mask,  # [N_grid * n_offsets]
            }
            gaussian_pkg.update(training_info)

        return gaussian_pkg

    def forward(self, camera: Camera, is_training: bool = False) -> dict[str, torch.Tensor]:
        """
        Render the scene.
        """
        anchor_visible_mask = self.prefilter_voxel(camera)
        gaussian_pkg = self.generate_gaussians(anchor_visible_mask, is_training)

        bg_color = get_color_tensor(self.config.background).to(self.device)

        renderer = GaussianRenderer(camera, bg_color=bg_color)
        output_pkg = renderer.render(
            gaussian_pkg["xyz"],
            None,
            gaussian_pkg["color"],
            gaussian_pkg["opacity"],
            gaussian_pkg["scaling"],
            gaussian_pkg["rot"],
        )

        render_pkg = {
            "render": output_pkg["render"],
        }
        if is_training:
            training_info = {
                "viewspace_points": output_pkg["means2D"],
                "gaussian_visible_mask": output_pkg["radii"] > 0,
                "scaling": gaussian_pkg["scaling"],
                "gaussian_opacity": gaussian_pkg["gaussian_opacity"],
                "offset_selection_mask": gaussian_pkg["offset_selection_mask"],
                "anchor_visible_mask": anchor_visible_mask,
            }
            render_pkg.update(training_info)

        return render_pkg

    def savePLY(self, ply_path, tile_filtering=True):
        self.logger.info(f"Saving gaussians to {ply_path}")

        gaussian_pkg = self.generate_gaussians(tile_filtering=tile_filtering)

        eps = 1e-10
        opacity = inverse_sigmoid(gaussian_pkg["opacity"].clamp(min=eps, max=1 - eps))
        scaling = torch.log(gaussian_pkg["scaling"].clamp(min=eps))
        shs = RGB2SH(gaussian_pkg["color"])

        gaussian = RawGaussian(*to_numpy(gaussian_pkg["xyz"], gaussian_pkg["rot"], scaling, opacity, shs))
        gaussian.savePLY(ply_path)

    def save_ckpt(self, ckpt_path: str):
        self.logger.info(f"Saving checkpoint to {ckpt_path}")
        Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)

        items = (self.state_dict(), self.optimizer.state_dict(), self.voxel_size, self.opacity_threshold, self.scene_bbox)
        torch.save(items, open(ckpt_path, "wb"))

    def load_ckpt(self, ckpt_path: str):
        items = torch.load(open(ckpt_path, "rb"))
        (params_state_dict, optimizer_state_dict, self.voxel_size, self.opacity_threshold, self.scene_bbox) = items

        anchor_count = params_state_dict["anchor"].shape[0]
        self._setup_parameters(anchor_count)
        self.load_state_dict(params_state_dict)

        self._training_setup()
        self.optimizer.load_state_dict(optimizer_state_dict)

    def create_from_pcd(self, pcd: PointCloud):
        points = torch.tensor(pcd.points).float().to(self.device)

        if self.voxel_size <= 0:
            median_dist, _ = torch.kthvalue(inter_point_distance(points), int(points.shape[0] * 0.5))
            self.voxel_size = median_dist.item()
        outside_voxel_size = self.voxel_size * self.config.outside_boundary_ratio
        self.logger.info(f"Initial voxel_size: {self.voxel_size}, outside boundary voxel_size: {outside_voxel_size}")

        inside_mask = get_inside_mask(points, self.scene_bbox)
        points_inside = points[inside_mask]
        points_outside = points[~inside_mask]

        anchor_inside = torch.unique(torch.round(points_inside / self.voxel_size), dim=0) * self.voxel_size
        anchor_outside = torch.unique(torch.round(points_outside / outside_voxel_size), dim=0) * outside_voxel_size
        anchor = torch.cat([anchor_inside, anchor_outside], dim=0)

        # offsets = torch.empty((anchor.shape[0], self.n_offsets, 3)).normal_(std=0.25).float().to(self.device)
        anchor_feat = torch.empty((anchor.shape[0], self.feat_dim)).normal_(std=self.config.feat_init_std).float().to(self.device)
        scaling = torch.ones_like(anchor).float().to(self.device) * self.config.max_offset_scale
        rotation = torch.tensor([1, 0, 0, 0]).repeat(anchor.shape[0], 1).float().to(self.device)

        self.anchor = nn.Parameter(anchor, requires_grad=True)
        # self.offset = nn.Parameter(offsets, requires_grad=True)
        self.anchor_feat = nn.Parameter(anchor_feat, requires_grad=True)
        self.scaling = nn.Parameter(scaling, requires_grad=False)
        self.rotation = nn.Parameter(rotation, requires_grad=False)

        self._training_setup()

    def get_raw_output(self):
        raw_pkg = {
            "anchor": self.anchor,
            # "scaling": self.scaling,
            "scaling": self.mlp_scaling(self.anchor_feat),
            "g_offset": self.mlp_offset(self.anchor_feat).view([-1, self.n_offsets, 3]),
            "g_opacity": self.mlp_opacity(self.anchor_feat).view([-1, self.n_offsets, 1]),
            "g_cov": self.mlp_cov(self.anchor_feat).view([-1, self.n_offsets, 7]),
            "g_color": self.mlp_color(self.anchor_feat).view([-1, self.n_offsets, 3]),
        }
        return raw_pkg

    def gt_gaussian_to_gt_pkg(self, gt_gaussian: RawGaussian):
        voxel_size = self.voxel_size
        n_offsets = self.n_offsets

        xyz = torch.tensor(gt_gaussian.xyz).float().to(self.device)
        opacity = torch.tensor(gt_gaussian.opacity).float().to(self.device)
        scaling = torch.tensor(gt_gaussian.scale).float().to(self.device)
        rot = torch.tensor(gt_gaussian.rot).float().to(self.device)
        shs = torch.tensor(gt_gaussian.shs).float().to(self.device)

        rgb = SH2RGB(shs)
        scaling = torch.exp(scaling)
        opacity = torch.sigmoid(opacity)

        # sort the points by importance
        importance = scaling.prod(dim=1) * opacity.squeeze(dim=1)
        xyz_st, opacity_st, scaling_st, rot_st, rgb_st = argsort(-importance, xyz, opacity, scaling, rot, rgb)

        # split points into voxels
        xyz_grid = torch.round(xyz_st / voxel_size).int()
        xyz_grid_unique, inverse_indices = xyz_grid.unique(return_inverse=True, dim=0)

        anchor = xyz_grid_unique.float() * voxel_size
        anchor_count = anchor.shape[0]

        # sort the points by voxel
        xyz_st, opacity_st, scaling_st, rot_st, rgb_st = argsort(inverse_indices, xyz_st, opacity_st, scaling_st, rot_st, rgb_st)

        # calculate per voxel point counts
        voxel_counts = torch.bincount(inverse_indices)
        max_point_per_voxel = voxel_counts.max().item()
        self.logger.info(f"Max point per voxel: {max_point_per_voxel}")
        if max_point_per_voxel > n_offsets:
            self.logger.warning(f"Some points are discarded because n_offsets: {n_offsets} is less than {max_point_per_voxel}!")

        # calculate per voxel index offsets
        voxel_offsets = torch.cumsum(voxel_counts, dim=0)[:-1]
        voxel_offsets = torch.cat((torch.tensor([0], device=self.device), voxel_offsets))

        # initialize formatted placeholder tensors
        g_offset = torch.zeros(anchor_count, n_offsets, 3).float().to(self.device)
        g_opacity = torch.zeros(anchor_count, n_offsets, 1).float().to(self.device)
        g_cov = torch.zeros(anchor_count, n_offsets, 7).float().to(self.device)
        g_color = torch.zeros(anchor_count, n_offsets, 3).float().to(self.device)

        # fill in the tensors with gt gaussian data
        for i in range(n_offsets):
            mask = (voxel_counts > i).nonzero(as_tuple=True)[0]
            selection_idx = voxel_offsets[mask] + i

            g_offset[mask, i, :] = xyz_st[selection_idx] - anchor[mask]
            g_opacity[mask, i, :] = opacity_st[selection_idx]
            g_cov[mask, i, :3] = scaling_st[selection_idx]
            g_cov[mask, i, 3:] = rot_st[selection_idx]
            g_color[mask, i, :] = rgb_st[selection_idx]

        # normalize the data
        eps = 1e-10
        margin = 0.05

        max_offset_scale = g_offset.abs().max(dim=1, keepdim=True).values * (1 + margin) + eps
        g_offset = g_offset / max_offset_scale

        max_scaling_scale = g_cov[:, :, :3].max(dim=1, keepdim=True).values * (1 + margin) + eps
        g_cov[:, :, :3] = inverse_sigmoid((g_cov[:, :, :3] / max_scaling_scale).clamp(eps, 1 - eps))

        anchor_scale = torch.cat((max_offset_scale, max_scaling_scale), dim=-1).squeeze(dim=1).log()
        g_color = g_color.clamp(0, 1)
        g_opacity = g_opacity.clamp(0, 1)

        gt_pkg = {
            "anchor": anchor,
            "scaling": anchor_scale,
            "g_offset": g_offset,
            "g_opacity": g_opacity,
            "g_cov": g_cov,
            "g_color": g_color,
            # "max_point_per_voxel": voxel_counts.max().item(),
        }
        return gt_pkg

    def create_from_gt_gaussian(self, gt_gaussian: RawGaussian):
        gt_pkg = self.gt_gaussian_to_gt_pkg(gt_gaussian)

        anchor = gt_pkg["anchor"]
        # scaling = gt_pkg["scaling"]
        anchor_feat = torch.empty((anchor.shape[0], self.feat_dim)).normal_(std=self.config.feat_init_std).float().to(self.device)
        # rotation = torch.tensor([1, 0, 0, 0]).repeat(anchor.shape[0], 1).float().to(self.device)

        self.anchor = nn.Parameter(anchor, requires_grad=True)
        self.anchor_feat = nn.Parameter(anchor_feat, requires_grad=True)
        # self.scaling = nn.Parameter(scaling, requires_grad=True)
        # self.rotation = nn.Parameter(rotation, requires_grad=False)

        self._training_setup()
        return gt_pkg
