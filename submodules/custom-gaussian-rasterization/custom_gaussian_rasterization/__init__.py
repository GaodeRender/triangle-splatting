#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
from typing import Callable

from . import _C


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)


def debug_run(func: Callable, *args, debug: bool = False):
    if debug:
        func_name = func.__name__
        cpu_args = cpu_deep_copy_tuple(args)
        try:
            return func(*args)
        except Exception as ex:
            torch.save(cpu_args, f"snapshot_{func_name}.dump")
            print(f"\nAn error occured in {func_name}. Writing snapshot_{func_name}.dump for debugging.")
            raise ex
    else:
        return func(*args)


# def rasterize_gaussians(
#     means3D,
#     means2D,
#     sh,
#     colors_precomp,
#     opacities,
#     scales,
#     rotations,
#     cov3Ds_precomp,
#     raster_settings,
# ):
#     return _RasterizeGaussians.apply(
#         means3D,
#         means2D,
#         sh,
#         colors_precomp,
#         opacities,
#         scales,
#         rotations,
#         cov3Ds_precomp,
#         raster_settings,
#     )


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.gamma,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.rich_info:
            (
                num_rendered,
                color,
                radii,
                depth,
                normal,
                contrib_sum,
                contrib_max,
                geomBuffer,
                binningBuffer,
                imgBuffer,
            ) = debug_run(_C.rasterize_gaussians_rich_info, *args, debug=raster_settings.debug)
        else:
            (
                num_rendered,
                color,
                radii,
                geomBuffer,
                binningBuffer,
                imgBuffer,
            ) = debug_run(_C.rasterize_gaussians, *args, debug=raster_settings.debug)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)

        if raster_settings.rich_info:
            return color, radii, depth, normal, contrib_sum, contrib_max
        else:
            return color, radii

    @staticmethod
    def backward(ctx, *grads_out):
        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        if raster_settings.rich_info:
            grad_out_color, _, grad_depth, grad_normal, _, _ = grads_out
        else:
            grad_out_color, _ = grads_out

        # Restructure args as C++ method expects them
        args = (
            raster_settings.bg,
            means3D,
            radii,
            colors_precomp,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            grad_out_color,
            sh,
            raster_settings.sh_degree,
            raster_settings.gamma,
            raster_settings.campos,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            raster_settings.debug,
        )

        # Compute gradients for relevant tensors by invoking backward method
        (
            grad_means2D,
            grad_colors_precomp,
            grad_opacities,
            grad_means3D,
            grad_cov3Ds_precomp,
            grad_sh,
            grad_scales,
            grad_rotations,
        ) = debug_run(_C.rasterize_gaussians_backward, *args, debug=raster_settings.debug)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )
        return grads


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    gamma: float
    campos: torch.Tensor
    prefiltered: bool
    rich_info: bool
    debug: bool


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings: GaussianRasterizationSettings):
        super().__init__()
        self.raster_settings = raster_settings

    def forward(
        self,
        means3D: torch.Tensor,
        means2D: torch.Tensor,
        opacities: torch.Tensor,
        shs: torch.Tensor = None,
        colors_precomp: torch.Tensor = None,
        scales: torch.Tensor = None,
        rotations: torch.Tensor = None,
        cov3Ds_precomp: torch.Tensor = None,
    ):
        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception("Please provide excatly one of either SHs or precomputed colors!")

        if ((scales is None or rotations is None) and cov3Ds_precomp is None) or (
            (scales is not None or rotations is not None) and cov3Ds_precomp is not None
        ):
            raise Exception("Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!")

        shs = torch.Tensor([]) if shs is None else shs
        colors_precomp = torch.Tensor([]) if colors_precomp is None else colors_precomp
        scales = torch.Tensor([]) if scales is None else scales
        rotations = torch.Tensor([]) if rotations is None else rotations
        cov3Ds_precomp = torch.Tensor([]) if cov3Ds_precomp is None else cov3Ds_precomp

        # Invoke C++/CUDA rasterization routine
        return _RasterizeGaussians.apply(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3Ds_precomp,
            self.raster_settings,
        )

    def in_frustum(self, positions: torch.Tensor) -> torch.Tensor:
        raster_settings = self.raster_settings

        # Mark visible points (based on frustum culling for camera) with a boolean
        with torch.no_grad():
            visible = _C.mark_visible(positions, raster_settings.viewmatrix, raster_settings.projmatrix)

        return visible

    def get_radii(
        self,
        means3D: torch.Tensor,
        scales: torch.Tensor = None,
        rotations: torch.Tensor = None,
        cov3Ds_precomp: torch.Tensor = None,
    ) -> torch.Tensor:
        if ((scales is None or rotations is None) and cov3Ds_precomp is None) or (
            (scales is not None or rotations is not None) and cov3Ds_precomp is not None
        ):
            raise Exception("Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!")

        scales = torch.Tensor([]) if scales is None else scales
        rotations = torch.Tensor([]) if rotations is None else rotations
        cov3Ds_precomp = torch.Tensor([]) if cov3Ds_precomp is None else cov3Ds_precomp

        raster_settings = self.raster_settings

        # Invoke C++/CUDA rasterization routine
        with torch.no_grad():
            radii = _C.rasterize_gaussians_filter(
                means3D,
                scales,
                rotations,
                raster_settings.scale_modifier,
                cov3Ds_precomp,
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
                raster_settings.tanfovx,
                raster_settings.tanfovy,
                raster_settings.image_height,
                raster_settings.image_width,
                raster_settings.prefiltered,
                raster_settings.debug,
            )
        return radii
