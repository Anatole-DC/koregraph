from dataclasses import dataclass
from abc import ABC
from pathlib import Path
from typing import Any


@dataclass
class FileStorageInterface(ABC):
    """Abstract class for the file storage system"""

    mode: str = "local"

    def open_file(path: Path) -> Any: ...

    def save_file() -> Path: ...
