"""File IO tools"""

from pathlib import Path

import numpy as np
import zstandard as zstd
from numpy import dtype, ndarray


def get_bytes(file_path: Path) -> bytes:
    """Read bytes from a file.

    Args:
        file_path (Path): The file path to read the bytes from.

    Returns:
        bytes: The bytes read from the file.

    """
    with Path.open(file_path, "rb") as f:
        decomp = zstd.ZstdDecompressor()
        return decomp.decompress(f.read())


def decode_bytes(
    bytes: bytes, shape: tuple[int, ...]
) -> ndarray[tuple[int, ...], dtype[np.uint8]]:
    """Decode bytes into a numpy array.

    Args:
        bytes (bytes): The bytes to decode.
        shape (tuple): The desired shape of the output array.

    Returns:
        ndarray: The decoded array.

    """
    return np.frombuffer(bytes, dtype=np.uint8).reshape(shape)


def read_file(
    file_path: Path, shape: tuple[int, ...] = (9, 500, 500)
) -> ndarray[tuple[int, ...], dtype[np.uint8]]:
    """Read Binary Numpy data from a file.

    Args:
        file_path (Path): The file path to read the data from.
        shape (tuple): The desired shape of the output array.

    Returns:
        ndarray: The loaded array.

    """
    return decode_bytes(get_bytes(file_path), shape)


def encode_bytes(array: ndarray[tuple[int, ...], dtype[np.uint8]]) -> bytes:
    """Encode a numpy array into bytes using Zstandard compression.

    Args:
        array (ndarray): The array to encode.

    Returns:
        bytes: The compressed bytes.

    """
    comp = zstd.ZstdCompressor(level=15)
    return comp.compress(array.tobytes())


def write_bytes(bytes: bytes, file_path: Path) -> int:
    """Write bytes to a file using Zstandard compression.

    Args:
        bytes (bytes): The bytes to write.
        file_path (Path): The file path to write the bytes to.

    Returns:
        int: Return code.

    """
    with Path.open(file_path, "wb") as f:
        return f.write(bytes)


def write_file(data: ndarray[tuple[int, ...], dtype[np.uint8]], file_path: Path) -> int:
    """Write a Binary Numpy array to a file.

    Args:
        data (ndarray): The array to write.
        file_path (Path): The file path to write the data to.

    Returns:
        int: Return code.

    """
    return write_bytes(encode_bytes(data), file_path)
