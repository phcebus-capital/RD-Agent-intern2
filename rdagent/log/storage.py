import json
import pickle
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Literal

from .base import Message, Storage
from .utils import gen_datetime

LOG_LEVEL = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

RUN_METADATA_FILENAME = "MODEL.txt"

# Top-level entries that don't, by themselves, indicate a meaningful run.
# A folder containing only these gets removed at process exit.
_NOISE_TOPLEVEL = frozenset({"debug_tpl", "debug_llm", RUN_METADATA_FILENAME})


class _CompatUnpickler(pickle.Unpickler):
    # Python 3.13 moved pathlib classes into the pathlib._local submodule.
    # Pickles created under 3.13+ store that module path and fail to load under <=3.12.
    def find_class(self, module: str, name: str) -> Any:
        if module == "pathlib._local" and sys.version_info < (3, 13):
            module = "pathlib"
        return super().find_class(module, name)


def _remove_empty_dir(path: Path) -> None:
    """
    Recursively remove empty directories.
    This function will remove the directory if it is empty after removing its subdirectories.
    """
    if path.is_dir():
        sub_dirs = [sub for sub in path.iterdir() if sub.is_dir()]
        for sub in sub_dirs:
            _remove_empty_dir(sub)

        if not any(path.iterdir()):
            try:
                path.rmdir()
            except OSError:
                pass


def _has_run_data(path: Path) -> bool:
    """Return True if the run folder produced any *meaningful* output.

    Folders containing only the model marker, ``debug_tpl/``, and/or ``debug_llm/``
    are treated as crashed-before-real-work and removed by ``FileStorage.cleanup``.
    """
    if not path.is_dir():
        return False
    for entry in path.iterdir():
        if entry.name not in _NOISE_TOPLEVEL:
            return True
    return False


class FileStorage(Storage):
    """
    The info are logginged to the file systems

    TODO: describe the storage format
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def write_run_metadata(self, lines: dict[str, str]) -> Path:
        """Write a small marker file at the storage root recording per-run metadata (model, etc.)."""
        self.path.mkdir(parents=True, exist_ok=True)
        marker = self.path / RUN_METADATA_FILENAME
        body = "\n".join(f"{k}: {v}" for k, v in lines.items()) + "\n"
        marker.write_text(body, encoding="utf-8")
        return marker

    def cleanup(self) -> None:
        """Remove the run folder if it has no logged data; otherwise prune empty subdirs."""
        if not self.path.exists():
            return
        if not _has_run_data(self.path):
            shutil.rmtree(self.path, ignore_errors=True)
            return
        _remove_empty_dir(self.path)

    def log(
        self,
        obj: object,
        tag: str = "",
        timestamp: datetime | None = None,
        save_type: Literal["json", "text", "pkl"] = "pkl",
        **kwargs: Any,
    ) -> str | Path:
        # TODO: We can remove the timestamp after we implement PipeLog
        timestamp = gen_datetime(timestamp)

        cur_p = self.path / tag.replace(".", "/")
        path = cur_p / f"{timestamp.strftime('%Y-%m-%d_%H-%M-%S-%f')}.log"
        cur_p.mkdir(parents=True, exist_ok=True)

        if save_type == "json":
            path = path.with_suffix(".json")
            with path.open("w") as f:
                try:
                    json.dump(obj, f)
                except TypeError:
                    json.dump(json.loads(str(obj)), f)
            return path
        elif save_type == "pkl":
            path = path.with_suffix(".pkl")
            with path.open("wb") as f:
                pickle.dump(obj, f)
            return path
        elif save_type == "text":
            obj = str(obj)
            with path.open("w") as f:
                f.write(obj)
            return path

    log_pattern = re.compile(
        r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) \| "
        r"(?P<level>DEBUG|INFO|WARNING|ERROR|CRITICAL) *\| "
        r"(?P<caller>.+:.+:\d+) - "
    )

    def iter_msg(self, tag: str | None = None, pattern: str | None = None) -> Generator[Message, None, None]:
        msg_l = []

        if pattern:
            pkl_files = pattern
        elif tag:
            pkl_files = f"**/{tag.replace('.','/')}/**/*.pkl"
        else:
            pkl_files = "**/*.pkl"
        for file in self.path.glob(pkl_files):
            if file.name == "debug_llm.pkl":
                continue
            pkl_log_tag = ".".join(file.relative_to(self.path).as_posix().replace("/", ".").split(".")[:-3])
            pid = file.parent.name

            with file.open("rb") as f:
                content = _CompatUnpickler(f).load()

            timestamp = datetime.strptime(file.stem, "%Y-%m-%d_%H-%M-%S-%f").replace(tzinfo=timezone.utc)

            m = Message(tag=pkl_log_tag, level="INFO", timestamp=timestamp, caller="", pid_trace=pid, content=content)

            msg_l.append(m)

        msg_l.sort(key=lambda x: x.timestamp)
        for m in msg_l:
            yield m

    def truncate(self, time: datetime) -> None:
        for file in self.path.glob("**/*.pkl"):
            timestamp = datetime.strptime(file.stem, "%Y-%m-%d_%H-%M-%S-%f").replace(tzinfo=timezone.utc)
            if timestamp > time.replace(tzinfo=timezone.utc):
                file.unlink()

        _remove_empty_dir(self.path)

    def __str__(self) -> str:
        return f"FileStorage({self.path})"
