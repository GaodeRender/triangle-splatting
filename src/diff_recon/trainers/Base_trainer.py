import torch
import numpy as np
import time
import os
import abc

from ..datasets.Base_dataset import BaseDatasetFactory
from ..utils.config import Config, loadConfig
from ..utils.logger import Logger


class BaseTrainer(abc.ABC):
    def __init__(self, config: str | Config = None, exp_name: str = None, device: torch.device | int = None, log_file: bool = True) -> None:
        time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        default_config = Config(trainer=Config(), dataset=Config(), model=Config())
        self.config = config if isinstance(config, Config) else loadConfig(config) if config is not None else default_config
        self.exp_name = time_str if not exp_name else exp_name
        self.device = device if isinstance(device, torch.device) else torch.device(f"cuda:{device}") if device is not None else torch.device("cuda")

        config = self.config.trainer
        output_dir = config.output_dir if config.output_dir is not None else ".output"
        clean_output_dir = config.clean_output_dir if config.clean_output_dir is not None else False
        seed = config.seed if config.seed is not None else 42
        detect_anomaly = config.detect_anomaly if config.detect_anomaly is not None else False
        use_tensorboard = config.use_tensorboard if config.use_tensorboard is not None else True

        self.output_dir = os.path.join(output_dir, self.exp_name)
        if clean_output_dir and log_file:
            os.system(f"rm -rf {self.output_dir}")

        self.logger = Logger(time_str, os.path.join(self.output_dir, "log") if log_file else None, use_tensorboard=use_tensorboard)
        self.logger.info(f"config args: {self.config}")
        if not log_file:
            self.logger.warning("Not creating log file because log_file is set to False")

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.autograd.set_detect_anomaly(detect_anomaly)

        self.dataset = self._load_dataset()
        self.model = None

    @abc.abstractmethod
    def train(self) -> None:
        raise NotImplementedError("Subclasses must implement this method.")

    @abc.abstractmethod
    def evaluate(self) -> None:
        raise NotImplementedError("Subclasses must implement this method.")

    def _load_dataset(self) -> BaseDatasetFactory:
        dataset_type = self.config.dataset.type
        match dataset_type:
            case "Colmap":
                from ..datasets.Colmap_dataset import ColmapDatasetFactory

                dataset = ColmapDatasetFactory(self.config.dataset, self.logger)
            case "NerfSynthetic":
                from ..datasets.NerfSynthetic_dataset import NerfSyntheticDatasetFactory

                dataset = NerfSyntheticDatasetFactory(self.config.dataset, self.logger)
            case "MatrixCity":
                from ..datasets.MatrixCity_dataset import MatrixCityDatasetFactory

                dataset = MatrixCityDatasetFactory(self.config.dataset, self.logger)
            case "Qijing":
                from ..datasets.Qijing_dataset import QijingDatasetFactory

                dataset = QijingDatasetFactory(self.config.dataset, self.logger)
            case _:
                raise ValueError(f"Unknown dataset type: {dataset_type}")

        return dataset
