from pathlib import Path
import os
import abc

from .logger import Logger, stdout_logger
from .io_utils import io


class BaseFileHandler(abc.ABC):
    def __init__(self):
        pass
    
    @abc.abstractmethod
    def getFilePath(self, file_path: str = None) -> str:
        raise NotImplementedError
    
    @abc.abstractmethod
    def hasFile(self, file_path: str) -> bool:
        raise NotImplementedError


class LocalHandler(BaseFileHandler):
    def __init__(self, local_root: str, logger: Logger = stdout_logger):
        self._local_root = os.path.abspath(local_root.rstrip("/"))
        self._logger = logger

    def getFilePath(self, file_path: str = None) -> str:
        if not self.hasFile(file_path):
            raise FileNotFoundError(f"File {file_path} not found on {self._local_root}")
        return self._getLocalPath(file_path)

    def hasFile(self, file_path: str) -> bool:
        return os.path.exists(self._getLocalPath(file_path))
    
    def _getLocalPath(self, file_path: str = None) -> str:
        return self._local_root if file_path is None else os.path.join(self._local_root, file_path)

class OSSHandler:
    def __init__(
        self,
        oss_root: str,
        local_root: str,
        logger: Logger = stdout_logger,
        get_skip_exist: bool = None,
        put_skip_exist: bool = None,
        upload_lock: bool = True,
    ):
        oss_root = oss_root.rstrip("/")
        local_root = local_root.rstrip("/")
        Path(local_root).mkdir(parents=True, exist_ok=True)

        self._oss_root = oss_root
        self._local_root = os.path.abspath(local_root)
        self._logger = logger
        self._get_skip_exist = get_skip_exist if get_skip_exist is not None else True
        self._put_skip_exist = put_skip_exist if put_skip_exist is not None else False
        self._upload_lock = upload_lock

    def getFilePath(self, file_path: str, skip_exist: bool = None, is_dir: bool = False) -> str:
        if not self.getFile(file_path, skip_exist, is_dir):
            raise FileNotFoundError(f"File {file_path} not found on {self._oss_root}")
        return self.getLocalPath(file_path)
    
    def hasFile(self, file_path: str) -> bool:
        return self.localExists(file_path) or self.remoteExists(file_path)

    def getFile(self, file_path: str, skip_exist: bool = None, is_dir: bool = False) -> bool:
        """
        Move file from the remote root to the local root.
        Need to pass in is_dir parameter because oss can't tell if an object is a file or a directory.
        """
        skip_exist = skip_exist if skip_exist is not None else self._get_skip_exist
        if skip_exist and self.localExists(file_path):
            return True

        if not self.remoteExists(file_path):
            return False

        if is_dir:
            io.copytree(self.getRemotePath(file_path), self.getLocalPath(file_path))
        else:
            io.safe_copy(self.getRemotePath(file_path), self.getLocalPath(file_path))
        return self.localExists(file_path)

    def getLocalPath(self, file_path: str = None) -> str:
        return self._local_root if file_path is None else f"{self._local_root}/{file_path}"

    def getRemotePath(self, file_path: str = None) -> str:
        return self._oss_root if file_path is None else f"{self._oss_root}/{file_path}"

    def localExists(self, file_path: str) -> bool:
        return os.path.exists(self.getLocalPath(file_path))

    def remoteExists(self, file_path: str) -> bool:
        return io.exists(self.getRemotePath(file_path))

    def localPutFile(self, src_file_path: str, dst_file_path: str, skip_exist: bool = None) -> bool:
        """
        Move file from outside the local root of this handler to the local root.
        :param src_file_path: local file path from outside the local root of this handler
        :param dst_file_path: file path relative to the local root of this handler
        """
        if not os.path.exists(src_file_path):
            return False

        skip_exist = skip_exist if skip_exist is not None else self._put_skip_exist

        if not (skip_exist and self.localExists(dst_file_path)):
            if os.path.isdir(src_file_path):
                io.copytree(src_file_path, self.getLocalPath(dst_file_path))
            else:
                io.copy(src_file_path, self.getLocalPath(dst_file_path))
        return self.localExists(dst_file_path)

    def remotePutFile(self, file_path: str, skip_exist: bool = None) -> bool:
        """
        Move file from the local root to the remote root.
        """
        if self._upload_lock:
            self._logger.warning(f"Upload lock is enabled, skipping remotePutFile: {file_path}")
            return True

        skip_exist = skip_exist if skip_exist is not None else self._put_skip_exist
        if skip_exist and self.remoteExists(file_path):
            return True

        if not self.localExists(file_path):
            return False

        self._logger.info(f"remotePutFile: {file_path}")
        if os.path.isdir(self.getLocalPath(file_path)):
            io.copytree(self.getLocalPath(file_path), self.getRemotePath(file_path))
        else:
            io.safe_copy(self.getLocalPath(file_path), self.getRemotePath(file_path))
        return self.remoteExists(file_path)

    def putFile(self, src_file_path: str, dst_file_path: str, skip_exist: bool = None, upload: bool = False) -> bool:
        """
        :param src_file_path: local file path from outside the local root of this handler.
        :param dst_file_path: file path relative to the local root of this handler
        :param upload: whether to upload the file to the remote root
        """
        if not self.localPutFile(src_file_path, dst_file_path, skip_exist):
            return False
        if upload:
            return self.remotePutFile(dst_file_path, skip_exist)
        return True

    def localCopyFile(self, src_file_path: str, dst_file_path: str, skip_exist: bool = None) -> bool:
        """
        Move file among the local root of this handler.
        :param src_file_path: file path relative to the local root of this handler
        :param dst_file_path: file path relative to the local root of this handler
        """
        self.localPutFile(self.getLocalPath(src_file_path), dst_file_path, skip_exist)

    def remoteCopyFile(self, src_file_path: str, dst_file_path: str, skip_exist: bool = None, is_dir: bool = False) -> bool:
        """
        Move file among the remote root of this handler.
        Need to pass in is_dir parameter because oss can't tell if an object is a file or a directory.
        :param src_file_path: file path relative to the local root of this handler
        :param dst_file_path: file path relative to the local root of this handler
        """
        if self._upload_lock:
            self._logger.warning(f"Upload lock is enabled, skipping remoteCopyFile: {src_file_path} -> {dst_file_path}")
            return True

        skip_exist = skip_exist if skip_exist is not None else self._put_skip_exist
        if skip_exist and self.remoteExists(dst_file_path):
            return True

        if not self.remoteExists(src_file_path):
            return False

        self._logger.info(f"remoteCopyFile: {src_file_path} -> {dst_file_path}")
        if is_dir:
            io.copytree(self.getRemotePath(src_file_path), self.getRemotePath(dst_file_path))
        else:
            io.safe_copy(self.getRemotePath(src_file_path), self.getRemotePath(dst_file_path))
        return self.remoteExists(dst_file_path)

    def remoteListDir(self, dir_path: str) -> list[str]:
        return io.listdir(self.getRemotePath(dir_path))

    def localRemove(self, file_path: str) -> None:
        if self.localExists(file_path):
            io.remove(self.getLocalPath(file_path))

    def remoteRemove(self, file_path: str, is_dir: bool = False) -> None:
        """
        Remove file from the remote root.
        Need to pass in is_dir parameter because oss can't tell if an object is a file or a directory.
        """
        if self._upload_lock:
            self._logger.warning(f"Upload lock is enabled, skipping remoteRemove: {file_path}")
            return

        if self.remoteExists(file_path):
            self._logger.info(f"remoteRemove: {file_path}")
            io.remove(self.getRemotePath(file_path), is_dir=is_dir)

    def localTouchFile(self, file_path: str) -> None:
        Path(self.getLocalPath(file_path)).touch()

    def remoteTouchFile(self, file_path: str) -> None:
        if self._upload_lock:
            self._logger.warning(f"Upload lock is enabled, skipping remoteTouchFile: {file_path}")
            return

        if self.remoteExists(file_path):
            return

        if self.localExists(file_path):
            self.remotePutFile(file_path)
        else:
            self.localTouchFile(file_path)
            self.remotePutFile(file_path)
            self.localRemove(file_path)

    def remoteSetDoneFlag(self, flag_name: str, done_flag: bool = True, target_dir: str = None) -> None:
        if self._upload_lock:
            self._logger.warning(f"Upload lock is enabled, skipping remoteSetDoneFlag: {flag_name}")
            return

        if target_dir is not None:
            flag_name = f"{target_dir}/{flag_name}"

        done_file_path = f"{flag_name}.done"
        fail_file_path = f"{flag_name}.fail"
        self.remoteRemove(done_file_path)
        self.remoteRemove(fail_file_path)

        write_file_path = done_file_path if done_flag else fail_file_path
        self.remoteTouchFile(write_file_path)

    def remoteDone(self, flag_name: str, target_dir: str = None) -> bool:
        if target_dir is not None:
            flag_name = f"{target_dir}/{flag_name}"
        return self.remoteExists(f"{flag_name}.done")

    def remoteFail(self, flag_name: str, target_dir: str = None) -> bool:
        if target_dir is not None:
            flag_name = f"{target_dir}/{flag_name}"
        return self.remoteExists(f"{flag_name}.fail")

    def remoteNoDoneFlag(self, flag_name: str, target_dir: str = None) -> bool:
        if self.remoteDone(flag_name, target_dir):
            self._logger.warning(f"{flag_name}.done exists!")
            return False

        if self.remoteFail(flag_name, target_dir):
            self._logger.warning(f"{flag_name}.fail exists!")
            return False

        return True

    def localClear(self) -> None:
        io.rmtree(self._local_root)
