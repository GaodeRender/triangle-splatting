import torch

from custom_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from ..utils.camera import Camera


class GaussianRenderer:
    def __init__(
        self,
        cam: Camera,
        bg_color: torch.Tensor = torch.Tensor([0, 0, 0]),
        scaling_modifier: float = 1.0,
        sh_degree: int = 0,
        gamma: float = 1.0,
        rich_info: bool = False,
        debug: bool = False,
    ):
        self.cam = cam
        self.rasterizer = self._getRasterizerFromCamera(cam, bg_color, scaling_modifier, sh_degree, gamma, rich_info, debug)

    def render(
        self,
        xyz: torch.Tensor,
        shs: torch.Tensor,
        color: torch.Tensor,
        opacity: torch.Tensor,
        scaling: torch.Tensor,
        rot: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        means2D = torch.zeros_like(xyz, requires_grad=True)

        output_tuple = self.rasterizer.forward(
            means3D=xyz,
            means2D=means2D,
            shs=shs,
            colors_precomp=color,
            opacities=opacity,
            scales=scaling,
            rotations=rot,
            cov3Ds_precomp=None,
        )

        if self.rasterizer.raster_settings.rich_info:
            rendered_image, radii, depth, normal, contrib_sum, contrib_max = output_tuple
            output_pkg = {
                "render": rendered_image,
                "radii": radii,
                "means2D": means2D,
                "depth": depth,
                "normal": normal,
                "contrib_sum": contrib_sum,
                "contrib_max": contrib_max,
            }
        else:
            rendered_image, radii = output_tuple
            output_pkg = {
                "render": rendered_image,
                "radii": radii,
                "means2D": means2D,
            }
        return output_pkg

    def get_radii(self, means3D: torch.Tensor, scales: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
        return self.rasterizer.get_radii(means3D, scales, rotations)

    def in_frustum(self, positions: torch.Tensor) -> torch.Tensor:
        return self.rasterizer.in_frustum(positions)

    @staticmethod
    def _getRasterizerFromCamera(
        cam: Camera,
        bg_color: torch.Tensor,
        scaling_modifier: float,
        sh_degree: int,
        gamma: float,
        rich_info: bool,
        debug: bool,
    ) -> GaussianRasterizer:
        raster_settings = GaussianRasterizationSettings(
            image_height=int(cam.image_height),
            image_width=int(cam.image_width),
            tanfovx=cam.tan_fovx,
            tanfovy=cam.tan_fovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=cam.world_view_transform,
            projmatrix=cam.full_proj_transform,
            sh_degree=sh_degree,
            gamma=gamma,
            campos=cam.camera_center,
            prefiltered=False,
            rich_info=rich_info,
            debug=debug,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        return rasterizer
