"""Collection of utility functions for managing the environment"""

import enum
import os
from pathlib import Path

from dotenv import load_dotenv

_ = load_dotenv()


class ENV:
    """Utility class for managing environment variables"""

    DATA_CSV: Path = Path(os.getenv("DATA_CSV") or "")
    DATA_DIR: Path = Path(os.getenv("DATA_DIR") or "")
    OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR") or "")
    THREADS: int = 20
    CHUNK: int = 10
    SIZE: tuple[int, int] = (500, 500)
    DIMENSIONS: int = 9
    BATCH_SIZE: int = 32


class SUBDIR:
    """Subdirectories for AI and non-AI images"""

    AI: str = "ai"
    NO_AI: str = "no_ai"


class DIMENSIONS(enum.Enum):
    """Enum for dimensions of the image data"""

    R = 0
    G = 1
    B = 2
    H = 3
    S = 4
    V = 5
    EDGES = 6
    DILATE = 7
    ERODE = 8
