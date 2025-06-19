import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from simple_knn import nearestNeighbor


class GaussianSmoothing2D(nn.Module):
    def __init__(self, kernel_size: int, sigma: float, normalize: bool = True):
        super().__init__()
        self.kernel_size = kernel_size
        self.kernel = self._gaussian_kernel(kernel_size, sigma, normalize)
        self.kernel_buffer = {}

    @staticmethod
    def _gaussian_kernel(kernel_size: int, sigma: float, normalize: bool):
        x_grid = torch.arange(kernel_size).unsqueeze(0).repeat(kernel_size, 1)
        xy_grid = torch.stack([x_grid, x_grid.T], dim=-1).float()

        mean = (kernel_size - 1) / 2.0
        variance = sigma**2.0

        kernel = torch.exp(-(xy_grid - mean).pow(2).sum(dim=-1) / (2 * variance))
        if normalize:
            kernel = kernel / kernel.sum()
        kernel = kernel.unsqueeze(0).unsqueeze(0).float()

        return kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Tensor of shape (B, C, H, W)
        """
        channels = x.shape[1]
        if channels not in self.kernel_buffer:
            self.kernel_buffer[channels] = self.kernel.repeat(channels, 1, 1, 1).to(x.device)

        kernel = self.kernel_buffer[channels].to(x.device)
        padding = (self.kernel_size - 1) // 2
        return F.conv2d(x, kernel, padding=padding, groups=channels)


class SSIM(nn.Module):
    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.window = GaussianSmoothing2D(window_size, sigma)
        self.C1 = 0.01**2
        self.C2 = 0.03**2

    def forward(self, img1: torch.Tensor, img2: torch.Tensor, keep_batch_dim: bool = False) -> torch.Tensor:
        """
        Args:
            img1 (torch.Tensor): Tensor of shape (B, C, H, W)
            img2 (torch.Tensor): Tensor of shape (B, C, H, W)
            keep_batch_dim (bool): Whether to keep the batch dimension or not.
        """
        mu1 = self.window(img1)
        mu2 = self.window(img2)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = self.window(img1 * img1) - mu1_sq
        sigma2_sq = self.window(img2 * img2) - mu2_sq
        sigma12 = self.window(img1 * img2) - mu1_mu2

        C1 = self.C1
        C2 = self.C2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        ssim_val = ssim_map.mean(dim=(1, 2, 3)) if keep_batch_dim else ssim_map.mean()
        return ssim_val


def normalize_shape(img1: torch.Tensor, img2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if img1.size() != img2.size():
        raise ValueError("Input images must have the same dimensions.")
    if len(img1.size()) == 4:
        pass
    elif len(img1.size()) == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    elif len(img1.size()) == 2:
        img1 = img1.unsqueeze(0).unsqueeze(0)
        img2 = img2.unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError("Input images must have 2, 3, or 4 dimensions.")
    return img1, img2


class SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim = SSIM()

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        img1, img2 = normalize_shape(img1, img2)
        return 1 - self.ssim(img1, img2)


class DoGFilter(nn.Module):
    def __init__(self, sigma: float):
        super().__init__()
        self.sigma1 = sigma
        self.sigma2 = 2 * sigma  # Ensure the 1:2 ratio
        self.kernel_size1 = int(2 * round(3 * self.sigma1) + 1)
        self.kernel_size2 = int(2 * round(3 * self.sigma2) + 1)
        self.filter1 = GaussianSmoothing2D(self.kernel_size1, self.sigma1)
        self.filter2 = GaussianSmoothing2D(self.kernel_size2, self.sigma2)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gaussian1 = self.filter1(x)
        gaussian2 = self.filter2(x)
        dog = gaussian1 - gaussian2
        return dog


class DoGLoss(nn.Module):
    def __init__(self, freq: int = 90, scale_factor: float = 0.5):
        super().__init__()
        self.freq = freq
        self.scale_factor = scale_factor
        self.dog_filter = DoGFilter(0.1 + (100 - freq) * 0.1 if freq >= 50 else 0.1 + freq * 0.1)

    @torch.no_grad()
    def _dog_mask(self, img: torch.Tensor) -> torch.Tensor:
        grayscale = img.mean(dim=1, keepdim=True)
        downsampled = F.interpolate(grayscale, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
        dog_img = self.dog_filter(downsampled)
        upsampled = F.interpolate(dog_img, size=img.shape[-2:], mode="bilinear", align_corners=False)

        normalized = (upsampled - upsampled.min()) / (upsampled.max() - upsampled.min())
        if self.freq >= 50:
            normalized = 1.0 - normalized

        mask = (normalized >= 0.5).float()
        return mask

    def forward(self, img: torch.Tensor, img_gt: torch.Tensor) -> torch.Tensor:
        img, img_gt = normalize_shape(img, img_gt)
        mask = self._dog_mask(img_gt)
        return L1(img * mask, img_gt * mask)


class ScharrFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel_x = torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=torch.float32).view(1, 1, 3, 3) / 32
        self.kernel_y = torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=torch.float32).view(1, 1, 3, 3) / 32
        self.kernel_buffer = {}

    def forward(self, x: torch.Tensor, ret_norm=False) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Tensor of shape (B, C, H, W)
        """
        channels = x.shape[1]
        if channels not in self.kernel_buffer:
            self.kernel_buffer[channels] = (
                self.kernel_x.repeat(channels, 1, 1, 1).to(x.device),
                self.kernel_y.repeat(channels, 1, 1, 1).to(x.device),
            )

        kernel_x, kernel_y = self.kernel_buffer[channels]
        padding = 1
        grad_x = F.conv2d(x, kernel_x, padding=padding, groups=channels)
        grad_y = F.conv2d(x, kernel_y, padding=padding, groups=channels)
        grad = torch.concatenate((grad_x, grad_y), dim=1)

        if ret_norm:
            grad = grad.norm(dim=1, keepdim=True)
        return grad


class SmoothnessLoss(nn.Module):
    def __init__(self, quantile: float = 0.3, scale_factor: float = 0.5):
        super().__init__()
        self.quantile = quantile
        self.scale_factor = scale_factor
        self.scharr = ScharrFilter()

    def _low_grad_mask(self, img: torch.Tensor) -> torch.Tensor:
        downsampled = F.interpolate(img, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
        grad = self.scharr(downsampled, ret_norm=True)
        upsampled = F.interpolate(grad, size=img.shape[-2:], mode="bilinear", align_corners=False)
        grad_threshold = torch.quantile(upsampled, self.quantile)
        mask = (upsampled < grad_threshold).float()
        return mask

    def forward(self, img: torch.Tensor, img_gt: torch.Tensor) -> torch.Tensor:
        img, img_gt = normalize_shape(img, img_gt)
        grad_mask = self._low_grad_mask(img_gt)
        grad_img = self.scharr(img, ret_norm=True)
        loss = (grad_img * grad_mask).mean()
        return loss


class DepthNormalLoss(nn.Module):
    def __init__(self, depth_grad: bool = True, normal_grad: bool = True, scale_factor: float = None, depth_grad_filter_quantile: float = 0.9):
        super().__init__()
        self.depth_grad = depth_grad
        self.normal_grad = normal_grad
        self.scale_factor = scale_factor
        self.depth_grad_filter_quantile = depth_grad_filter_quantile
        self.scharr = ScharrFilter()

    def depth_to_normal(self, depth: torch.Tensor, tan_fovx: float, tan_fovy: float, ret_grad_mask: bool = True) -> torch.Tensor:
        """
        Args:
            depth (torch.Tensor): Tensor of shape (H, W)
        """
        scale_factor = self.scale_factor
        W0, H0 = depth.shape[-1], depth.shape[-2]
        depth = depth.unsqueeze(0).unsqueeze(0)
        if scale_factor is not None and scale_factor != 1:
            depth = F.interpolate(depth, scale_factor=scale_factor, mode="bilinear", align_corners=False)

        depth_grad = self.scharr(depth).squeeze(0)
        Dx, Dy = torch.unbind(depth_grad / depth.squeeze(0), 0)
        W, H = depth.shape[-1], depth.shape[-2]
        x = torch.arange(W, dtype=torch.float32).to(depth.device)
        y = torch.arange(H, dtype=torch.float32).to(depth.device)
        x, y = torch.meshgrid(x, y, indexing="xy")

        nx = W * Dx / (2 * tan_fovx)
        ny = H * Dy / (2 * tan_fovy)
        nz = -(1 + (x - W / 2 + 0.5) * Dx + (y - H / 2 + 0.5) * Dy)
        normal = torch.stack([nx, ny, nz], dim=0)

        if W0 != W or H0 != H:
            normal = F.interpolate(normal.unsqueeze(0), size=(H0, W0), mode="bilinear", align_corners=False).squeeze(0)
        normal = normal / normal.norm(dim=0, keepdim=True)
        if not ret_grad_mask:
            return normal

        grad_norm = depth_grad.norm(dim=0, keepdim=True)
        if W0 != W or H0 != H:
            grad_norm = F.interpolate(grad_norm.unsqueeze(0), size=(H0, W0), mode="bilinear", align_corners=False).squeeze(0)
        grad_threshold = torch.quantile(grad_norm, self.depth_grad_filter_quantile)
        grad_mask = (grad_norm < grad_threshold).float().squeeze(0)
        return normal, grad_mask

    def forward(self, depth: torch.Tensor, normal: torch.Tensor, tan_fovx: float, tan_fovy: float) -> torch.Tensor:
        if not self.depth_grad:
            depth = depth.detach()
        if not self.normal_grad:
            normal = normal.detach()

        depth_normal, grad_mask = self.depth_to_normal(depth, tan_fovx, tan_fovy)
        normal = torch.nn.functional.normalize(normal, p=2, dim=0, eps=1e-8)
        return ((1 - (normal * depth_normal).sum(dim=0)) * grad_mask).mean()


class DiffusionLoss(nn.Module):
    def __init__(self, ckpt_path, clip_ckpt_path, ddim_num_steps=50, ddim_eta=0.0, cfg_scale=1.0, strength=0.5):
        super().__init__()
        from ldm.models.diffusion.ddim import DDIMSampler

        config_path = "./submodules/stablediffusion/configs/stable-diffusion/v2-inference-v.yaml"
        self.model = self._load_model(config_path, ckpt_path, clip_ckpt_path)
        self.sampler = DDIMSampler(self.model)
        self.sampler.make_schedule(ddim_num_steps=ddim_num_steps, ddim_eta=ddim_eta, verbose=False)

        self.ddim_steps = ddim_num_steps
        self.cfg_scale = cfg_scale
        self.strength = strength

    @staticmethod
    def _load_model(config_path: str, ckpt_path: str, clip_ckpt_path:str, verbose: bool = True):
        from omegaconf import OmegaConf
        from ldm.util import instantiate_from_config
        from ldm.models.diffusion.ddpm import LatentDiffusion

        config = OmegaConf.load(config_path)
        config.model.params.cond_stage_config.params.version = clip_ckpt_path
        model: LatentDiffusion = instantiate_from_config(config.model)

        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)

        model.eval()
        return model

    def forward(self, x: torch.Tensor, prompts: list[str], ret_samples=False) -> torch.Tensor:
        batch_size = len(prompts)
        c = self.model.get_learned_conditioning(prompts)
        uc = self.model.get_learned_conditioning(batch_size * [""]) if self.cfg_scale != 1.0 else None

        # encode (scaled latent)
        t_enc = int(self.strength * self.ddim_steps)
        init_latent = self.model.get_first_stage_encoding(self.model.encode_first_stage(x))  # move to latent space
        z_enc = self.sampler.stochastic_encode(init_latent, torch.tensor([t_enc] * batch_size).to(x.device))

        # decode it
        with torch.autocast("cuda"):
            samples = self.sampler.decode(
                z_enc,
                c,
                t_enc,
                unconditional_guidance_scale=self.cfg_scale,
                unconditional_conditioning=uc,
            )

        x_samples = self.model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

        loss = F.mse_loss(x_samples, x)
        return loss if not ret_samples else (loss, x_samples)


def L1(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    return torch.abs((t1 - t2)).mean()


def L2(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    return ((t1 - t2) ** 2).mean()


def psnr(img1: torch.Tensor, img2: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    if mask is None:
        mse = ((img1 - img2) ** 2).mean() + 1e-10
    else:
        mse = (((img1 - img2) ** 2) * mask).sum() / (mask.sum() + 1e-10) + 1e-10
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def nearest_neighbor(pc: torch.Tensor, bs: int = 1) -> torch.Tensor:
    return nearestNeighbor(pc, bs).long()


def nearest_dist2(pc: torch.Tensor, nearest_indices: torch.Tensor) -> torch.Tensor:
    assert pc.dim() == 2 and pc.size(1) == 3 and nearest_indices.dim() == 1 and nearest_indices.size(0) == pc.size(0)
    nearest_point = pc[nearest_indices]
    return ((pc - nearest_point) ** 2).sum(dim=1)


ssimLoss = SSIMLoss()
dogLoss = DoGLoss()
smoothnessLoss = SmoothnessLoss()
lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", reduction="mean", normalize=True)
