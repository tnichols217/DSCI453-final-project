"""Collection of functions to help with producing the model Dataset"""

from collections.abc import Generator
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import DatasetV2

from env_manager import ENV, SUBDIR
from file_manager import read_file


def get_image_list() -> Generator[Path, None, None]:
    """Gets a list of image paths from the output directory.

    Returns:
        List[Generator[Path, None, None]]: List of image paths.

    """
    image_paths: Generator[Path, None, None] = Path(ENV.OUTPUT_DIR).glob("*/*")

    return image_paths


def load_one(file: Path) -> tuple[tf.Tensor, bool]:
    """Loads an image from the specified file path.

    Args:
        file (Path): The file path to load the image from.

    Returns:
        tf.Tensor: The loaded image tensor.

    """
    image = read_file(file)
    label = file.parent.name == SUBDIR.AI
    image = tf.convert_to_tensor(image.astype(np.float32) / 255.0)  # pyright: ignore[reportUnknownMemberType]

    return image, label


def create_dataset() -> DatasetV2:
    """Create a TensorFlow dataset

    Returns:
        tf.data.Dataset: Batched and prefetched dataset.

    """
    # Get images, then load and preprocess
    image_paths = get_image_list()
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)  # pyright: ignore[reportUnknownMemberType]
    dataset = dataset.map(load_one, num_parallel_calls=tf.data.AUTOTUNE)  # pyright: ignore[reportUnknownMemberType]
    dataset = dataset.batch(ENV.BATCH_SIZE)  # pyright: ignore[reportUnknownMemberType]

    # Prefetch
    return dataset.prefetch(tf.data.AUTOTUNE)  # pyright: ignore[reportUnknownMemberType]


if __name__ == "__main__":
    # Create the dataset
    dataset = create_dataset()
    print(dataset)
