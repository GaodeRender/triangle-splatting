import abc
from torch.utils.data import Dataset, DataLoader

from ..utils.config import Config
from ..utils.logger import Logger, stdout_logger
from ..utils.camera import Camera
from ..utils.file_handler import BaseFileHandler


def nop(x):
    return x


class BaseDatasetFactory(abc.ABC):
    def __init__(self, config: Config = None, logger: Logger = None):
        self._config = config if config is not None else Config()
        self._logger = logger if logger is not None else stdout_logger

        self._num_workers = config.num_workers if config.num_workers is not None else 1

        self._train_dataset: Dataset = None
        self._test_dataset: Dataset = None
        self._file_handler: BaseFileHandler = None
    
    def getFileHandler(self) -> BaseFileHandler:
        if self._file_handler is None:
            raise ValueError("File handler is not initialized.")
        return self._file_handler
    
    def _getDataLoader(self, dataset: Dataset, shuffle: bool = True):
        if len(dataset) == 0:
            return []

        return iter(
            DataLoader(
                dataset,
                batch_size=None,
                shuffle=shuffle,
                num_workers=self._num_workers,
                pin_memory=True,
                collate_fn=nop,  # can't use lambda x: x because of pickling error when using "spawn" start method
                prefetch_factor=10,
            )
        )

    def getTrainDataset(self) -> DataLoader:
        return self._getDataLoader(self._train_dataset)

    def getTrainDatasetSize(self) -> int:
        return len(self._train_dataset)

    def nextTrainData(self) -> Camera:
        if not hasattr(self, "_train_dataloader"):
            self._train_dataloader = self.getTrainDataset()

        try:
            data = next(self._train_dataloader)
        except StopIteration:
            self._train_dataloader = self.getTrainDataset()
            data = next(self._train_dataloader)
        return data

    def getTestDataset(self) -> DataLoader:
        return self._getDataLoader(self._test_dataset, shuffle=False)

    def getTestDatasetSize(self) -> int:
        return len(self._test_dataset)

    def getTestData(self, idx) -> Camera:
        return self._test_dataset[idx]
