import logging
import os
import sys
from pathlib import Path
from typing import TextIO, NamedTuple, Union, Optional
from multiprocessing import Queue, Process
import traceback
from enum import Enum
from torch.utils.tensorboard import SummaryWriter
import torch


class LogType(Enum):
    LOG = 1
    TB_SCALAR = 2
    TB_IMAGE = 3
    TB_HISTOGRAM = 4


class LogRecord(NamedTuple):
    type: LogType
    tag: str
    value: Union[float, torch.Tensor, logging.LogRecord]
    global_step: int


class ColoredFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(levelname)s: %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def getLogger(name, log_file_dir: Optional[str] = None, stream: Optional[TextIO] = None, level=logging.DEBUG) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    stream_formatter = ColoredFormatter()

    if log_file_dir is not None:
        fileinfo = logging.FileHandler(os.path.join(log_file_dir, f"{name}_outputs.log"))
        fileinfo.setLevel(logging.INFO)
        fileinfo.setFormatter(file_formatter)
        logger.addHandler(fileinfo)

    if stream is not None:
        controlshow = logging.StreamHandler(stream)
        controlshow.setLevel(logging.DEBUG)
        controlshow.setFormatter(stream_formatter)
        logger.addHandler(controlshow)

    return logger


class Logger_MP:
    # handle logging in the background process
    def __init__(self, name, log_file_dir: Optional[str] = None, stream: Optional[TextIO] = sys.stdout, use_tensorboard: bool = True):
        if log_file_dir is not None:
            Path(log_file_dir).mkdir(parents=True, exist_ok=True)

        self.name = name
        self.use_tensorboard = use_tensorboard

        self.log_queue = Queue(-1)
        self.listener = Process(target=self._listener_process, args=(self.log_queue, log_file_dir, stream, use_tensorboard))
        self.listener.start()

    def _listener_process(self, log_queue: Queue, log_file_dir: str, stream: TextIO, use_tensorboard: bool):
        logger = getLogger(self.name, log_file_dir, stream)
        tb_writter = SummaryWriter(log_file_dir) if use_tensorboard and log_file_dir is not None else None

        while True:
            try:
                record = log_queue.get()
                if record is None:  # send None to tell the listener to exit
                    break

                if record.type == LogType.LOG:
                    logger.handle(record.value)
                elif record.type == LogType.TB_SCALAR:
                    if tb_writter is not None:
                        tb_writter.add_scalar(record.tag, record.value, record.global_step)
                elif record.type == LogType.TB_IMAGE:
                    if tb_writter is not None:
                        tb_writter.add_image(record.tag, record.value, record.global_step)
                elif record.type == LogType.TB_HISTOGRAM:
                    if tb_writter is not None:
                        tb_writter.add_histogram(record.tag, record.value, record.global_step)
            except Exception:
                logger.error("Whoops! Problem:")
                traceback.print_exc(file=sys.stderr)

        if tb_writter is not None:
            tb_writter.close()
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.close()

    def add_scalar(self, tag: str, scalar_value: float, global_step: int):
        if self.use_tensorboard:
            log_record = LogRecord(LogType.TB_SCALAR, tag, scalar_value, global_step)
            self.log_queue.put(log_record)

    def add_image(self, tag: str, img_tensor, global_step: int):
        if self.use_tensorboard:
            log_record = LogRecord(LogType.TB_IMAGE, tag, img_tensor, global_step)
            self.log_queue.put(log_record)

    def add_histogram(self, tag: str, values, global_step: int, bins="auto"):
        if self.use_tensorboard:
            log_record = LogRecord(LogType.TB_HISTOGRAM, tag, values, global_step)
            self.log_queue.put(log_record)

    def log(self, level, msg, *args, **kwargs):
        record = logging.LogRecord(name=None, level=level, pathname=None, lineno=None, msg=msg, args=args, exc_info=None, func=None, sinfo=None)
        log_record = LogRecord(LogType.LOG, None, record, None)
        self.log_queue.put(log_record)

    def info(self, message: str):
        self.log(logging.INFO, message)

    def debug(self, message: str):
        self.log(logging.DEBUG, message)

    def warning(self, message: str):
        self.log(logging.WARNING, message)

    def error(self, message: str):
        self.log(logging.ERROR, message)

    def critical(self, message: str):
        self.log(logging.CRITICAL, message)

    def warnOnce(self, message: str):
        if not hasattr(self, "warned_msg"):
            self.warned_msg = set()
        if message not in self.warned_msg:
            self.warning(message)
            self.warned_msg.add(message)

    def __del__(self):
        self.log_queue.put(None)
        self.listener.join()


class Logger:
    # handle logging in the main process
    def __init__(self, name, log_file_dir: Optional[str] = None, stream: Optional[TextIO] = sys.stdout, use_tensorboard: bool = False):
        if log_file_dir is not None:
            Path(log_file_dir).mkdir(parents=True, exist_ok=True)

        self.name = name
        self.logger = getLogger(name, log_file_dir, stream)
        self.tb_writer = SummaryWriter(log_file_dir) if use_tensorboard and log_file_dir is not None else None

    def add_scalar(self, tag: str, scalar_value: float, global_step: int):
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(tag, scalar_value, global_step)

    def add_image(self, tag: str, img_tensor, global_step: int):
        if self.tb_writer is not None:
            self.tb_writer.add_image(tag, img_tensor, global_step)

    def add_histogram(self, tag: str, values, global_step: int, bins="auto"):
        if self.tb_writer is not None:
            self.tb_writer.add_histogram(tag, values, global_step, bins)

    def log(self, level, msg, *args, **kwargs):
        self.logger.log(level, msg, *args, **kwargs)

    def debug(self, message: str):
        self.log(logging.DEBUG, message)

    def info(self, message: str):
        self.log(logging.INFO, message)

    def warning(self, message: str):
        self.log(logging.WARNING, message)

    def error(self, message: str):
        self.log(logging.ERROR, message)

    def critical(self, message: str):
        self.log(logging.CRITICAL, message)

    def warnOnce(self, message: str):
        if not hasattr(self, "warned_msg"):
            self.warned_msg = set()
        if message not in self.warned_msg:
            self.warning(message)
            self.warned_msg.add(message)

    def __del__(self):
        if self.tb_writer is not None:
            self.tb_writer.close()
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.close()


logging.lastResort = logging.NullHandler()
empty_logger = Logger("empty_logger", stream=None)
stdout_logger = Logger("stdout_logger")
