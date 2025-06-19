import torch
import trimesh
import kaolin as kal

from ..utils.camera import Camera


class KaolinRenderer:
    def __init__(
        self,
        cam: Camera,
        bg_color: torch.Tensor = torch.Tensor([0, 0, 0]),
    ):
        self.cam = cam
        self.bg_color = bg_color.to(cam.device)

    def render(
        self,
        vertices: torch.Tensor = None,
        faces: torch.Tensor = None,
        faces_color: torch.Tensor = None,
        mesh_path: str = None,
    ) -> dict[str, torch.Tensor]:
        # Prepare camera parameters
        cam = self.cam
        device = cam.device
        image_width = cam.image_width
        image_height = cam.image_height
        znear = cam.znear
        camera_projection = torch.tensor([1.0 / cam.tan_fovx, 1.0 / cam.tan_fovy, -1]).float().to(device).view(3, 1)
        camera_transform = cam.world_view_transform[:, :3].unsqueeze(0) @ torch.diag(torch.tensor([1, -1, -1])).float().to(device)

        # Load mesh file if mesh_path is provided
        if mesh_path is not None:
            mesh = trimesh.load_mesh(mesh_path)
            vertices = torch.tensor(mesh.vertices.reshape(-1, 3), dtype=torch.float32).to(device)
            faces = torch.tensor(mesh.faces, dtype=torch.int64).to(device)
            faces_color = torch.tensor(mesh.visual.face_colors[: len(faces), :3] / 255, dtype=torch.float32).to(device)
        else:
            if vertices is None or faces is None or faces_color is None:
                raise ValueError("Either mesh_path or vertices, faces, and faces_color must be provided")

        # Kaolin rendering
        face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
            vertices, faces, camera_proj=camera_projection, camera_transform=camera_transform
        )
        face_attributes = [
            faces_color.view(1, -1, 1, 3).repeat(1, 1, 3, 1),
            torch.ones((1, faces.shape[0], 3, 1)).float().to(device),
        ]
        valid_faces = (-face_vertices_camera[:, :, :, -1] > znear).all(dim=-1)
        (image, mask), face_idx = kal.render.mesh.rasterize(
            image_height,
            image_width,
            face_vertices_camera[:, :, :, -1],
            face_vertices_image,
            face_attributes,
            valid_faces=valid_faces,
            backend="nvdiffrast_fwd",
        )

        # apply background color
        image = image * mask + self.bg_color * (1 - mask)
        image = torch.clamp(image, 0.0, 1.0)

        image = image.permute(0, 3, 1, 2).squeeze(0)
        mask = mask.permute(0, 3, 1, 2).squeeze(0)

        return {
            "render": image,
            "mask": mask,
        }
