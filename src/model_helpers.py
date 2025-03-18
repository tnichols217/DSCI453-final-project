"""Collection of functions to help with producing the model Dataset"""

from collections.abc import Generator
from pathlib import Path

import numpy as np
import tensorflow.python.framework.dtypes as tft
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE, DatasetV2
from tensorflow.python.framework.tensor import Tensor, TensorSpec
from tensorflow.python.framework.tensor_conversion import convert_to_tensor_v2

from env_manager import ENV, SUBDIR
from file_manager import read_file


def get_image_list() -> Generator[Path, None, None]:
    """Gets a list of image paths from the output directory.

    Returns:
        Generator[Path]: A generator yielding image file paths.

    """
    return Path(ENV.OUTPUT_DIR).glob("*/*")


def load_one(fp: Path) -> tuple[Tensor, bool]:
    """Loads an image from the specified file path.

    Args:
        fp (Path): The file path to load the image from.

    Returns:
        tf.Tensor: The loaded image tensor.

    """
    image = read_file(fp)
    label = fp.parent.name == SUBDIR.AI
    image = convert_to_tensor_v2(image.astype(np.float32) / 255.0)

    return image, label


def get_data_generator() -> Generator[tuple[Tensor, bool], None, None]:
    """Generator for loading and preparing images in batches.

    Yields:
        tuple[tf.Tensor, bool]: A tuple of image tensor and label.

    """
    gen = get_image_list()
    while n := next(gen, None):
        yield load_one(n)


def create_dataset() -> DatasetV2:
    """Create a TensorFlow dataset

    Returns:
        tf.data.Dataset: Batched and prefetched dataset.

    """
    # Get images, then loads, batches, and preprocesses
    return (
        DatasetV2.from_generator(  # pyright: ignore[reportUnknownMemberType]
            get_data_generator,
            output_signature=(
                TensorSpec(shape=(ENV.DIMENSIONS, *ENV.SIZE), dtype=tft.float32),
                TensorSpec(shape=(), dtype=tft.bool),
            ),
        )
        .batch(ENV.BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )


if __name__ == "__main__":
    # Create the dataset
    dataset = create_dataset()
    print(dataset)
