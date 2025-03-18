"""Collection of functions to help with producing the model Dataset"""

from collections.abc import Generator
from pathlib import Path

import numpy as np
import tensorflow.python.framework.dtypes as tft
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE, DatasetV2
from tensorflow.python.framework.tensor_conversion import convert_to_tensor_v2
from tensorflow.python.types.core import Tensor

from env_manager import ENV, SUBDIR
from file_manager import read_file


def get_image_list() -> Generator[str, None, None]:
    """Gets a list of image paths from the output directory.

    Yields:
        str: Image path.

    """
    image_paths: Generator[Path, None, None] = Path(ENV.OUTPUT_DIR).glob("*/*")

    n = next(image_paths, None)
    while n:
        yield str(n)
        n = next(image_paths, None)


def load_one(file: str) -> tuple[Tensor, bool]:
    """Loads an image from the specified file path.

    Args:
        file (Path): The file path to load the image from.

    Returns:
        tf.Tensor: The loaded image tensor.

    """
    fp = Path(file)
    image = read_file(fp)
    label = fp.parent.name == SUBDIR.AI
    image = convert_to_tensor_v2(image.astype(np.float32) / 255.0)

    return image, label


def create_dataset() -> DatasetV2:
    """Create a TensorFlow dataset

    Returns:
        tf.data.Dataset: Batched and prefetched dataset.

    """
    # Get images, then loads, batches, and preprocesses
    return (DatasetV2  # pyright: ignore[reportUnknownMemberType]
        .from_generator(get_image_list, output_types=tft.string)
        .map(load_one, num_parallel_calls=AUTOTUNE)
        .batch(ENV.BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )


if __name__ == "__main__":
    # Create the dataset
    dataset = create_dataset()
    print(dataset)
