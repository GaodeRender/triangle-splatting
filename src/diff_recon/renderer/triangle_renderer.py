import torch

from diff_triangle_rasterization_2D import (
    TriangleRasterizationSettings as TriangleRasterizationSettings_2D,
    TriangleRasterizer as TriangleRasterizer_2D,
)
from diff_triangle_rasterization_3D import (
    TriangleRasterizationSettings as TriangleRasterizationSettings_3D,
    TriangleRasterizer as TriangleRasterizer_3D,
)

from ..utils.camera import Camera


class TriangleRenderer:
    def __init__(
        self,
        cam: Camera,
        bg_depth: float = 5000.0,
        bg_color: torch.Tensor = torch.Tensor([0, 0, 0]),
        scaling_modifier: float = 1.0,
        sh_degree: int = 0,
        gamma: float = 1.0,
        back_culling: bool = False,
        rich_info: bool = False,
        debug: bool = False,
        rasterizer_type: str = "3D",
    ):
        if rasterizer_type == "2D":
            TriangleRasterizer = TriangleRasterizer_2D
            TriangleRasterizationSettings = TriangleRasterizationSettings_2D
        elif rasterizer_type == "3D":
            TriangleRasterizer = TriangleRasterizer_3D
            TriangleRasterizationSettings = TriangleRasterizationSettings_3D
        else:
            raise ValueError(f"Unknown rasterizer type: {rasterizer_type}. Use '2D' or '3D'.")

        raster_settings = TriangleRasterizationSettings(
            image_height=int(cam.image_height),
            image_width=int(cam.image_width),
            tanfovx=cam.tan_fovx,
            tanfovy=cam.tan_fovy,
            viewmatrix=cam.world_view_transform,
            projmatrix=cam.full_proj_transform,
            campos=cam.camera_center,
            sh_degree=sh_degree,
            gamma=gamma,
            scale_modifier=scaling_modifier,
            background_depth=bg_depth,
            background=bg_color.to(cam.device),
            back_culling=back_culling,
            rich_info=rich_info,
            debug=debug,
        )

        self.cam = cam
        self.rasterizer = TriangleRasterizer(raster_settings=raster_settings)

    def render(
        self,
        vertex: torch.Tensor,
        shs: torch.Tensor,
        color: torch.Tensor,
        opacity: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Create placeholder tensor to store gradients of the 2D (screen-space) center of each triangle.
        center2D = torch.zeros((vertex.shape[0], 2), device=vertex.device, dtype=vertex.dtype, requires_grad=True)

        output_tuple = self.rasterizer.forward(
            vertex=vertex,
            center2D=center2D,
            opacity=opacity,
            shs=shs,
            feature=color,
        )

        if self.rasterizer.raster_settings.rich_info:
            rendered_image, radii, depth, normal, contrib_sum, contrib_max = output_tuple
            output_pkg = {
                "render": rendered_image,
                "radii": radii,
                "center2D": center2D,
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
                "center2D": center2D,
            }
        return output_pkg
