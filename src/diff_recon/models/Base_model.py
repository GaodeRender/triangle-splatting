import torch
from torch import nn
import abc

from ..utils.camera import Camera
from ..utils.config import Config
from ..utils.logger import Logger, stdout_logger


class BaseModel(nn.Module):
    def __init__(self, config: Config = None, logger: Logger = None, device: torch.device | int = None):
        super().__init__()
        self.config = config if config is not None else Config()
        self.logger = logger if logger is not None else stdout_logger
        self.device = device if isinstance(device, torch.device) else torch.device(f"cuda:{device}") if device is not None else torch.device("cuda")

    @abc.abstractmethod
    def forward(self, camera: Camera, *args, **kwargs) -> dict:
        """
        Forward pass of the model. Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")
