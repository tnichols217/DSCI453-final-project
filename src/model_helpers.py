"""Collection of functions to help with producing the model Dataset"""

import random
from collections.abc import Generator
from pathlib import Path
from typing import Any, Callable, Generator

import keras as k
import numpy as np
from numpy._typing import _16Bit
import tensorflow.python.framework.dtypes as tft
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE, DatasetV2
from tensorflow.python.framework.tensor import Tensor, TensorSpec
from tensorflow.python.framework.tensor_conversion import (
    convert_to_tensor_v2,  # pyright: ignore[reportUnknownVariableType]
)

from env_manager import ENV, SUBDIR
from file_manager import read_file


def get_image_list(seed: int = 1) -> list[Path]:
    """Gets a list of image paths from the output directory.

    Returns:
        Generator[Path]: A generator yielding image file paths.

    """
    files = list(Path(ENV.OUTPUT_DIR).glob("*/*"))
    random.seed(seed)
    random.shuffle(files)
    return files


def get_train_test_images(ratio: float = 0.1) -> tuple[list[Path], list[Path]]:
    """Splits the image list into training and testing sets.

    Args:
        ratio (float): The ratio of the training set size to the total set size.

    Returns:
        tuple[list[Path], list[Path]]: A tuple of training and testing image file paths.

    """
    files = get_image_list()
    test_size = int(len(files) * ratio)
    return files[test_size + 1 :], files[:test_size]


def load_one(fp: Path) -> tuple[Tensor, bool]:
    """Loads an image from the specified file path.

    Args:
        fp (Path): The file path to load the image from.

    Returns:
        tf.Tensor: The loaded image tensor.

    """
    image = read_file(fp)
    label = fp.parent.name == SUBDIR.AI
    image = convert_to_tensor_v2(np.moveaxis(image, 0, -1).astype(np.float32) / 255.0)

    return image, label


def get_data_generator(
    images: list[Path],
) -> Callable[..., Generator[tuple[Tensor, bool], None, None]]:
    """Generator for loading and preparing images in batches.

    Args:
        images (list[Path]): List of image file paths.

    Returns:
        A function that returns a generator yielding image tensors and labels.
        Will Yield:
            tuple[tf.Tensor, bool]: A tuple of image tensor and label.

    """
    def gen_ret() -> Generator[tuple[Tensor, bool], None, None]:
        for n in images:
            yield load_one(n)
    return gen_ret


def create_dataset() -> tuple[DatasetV2, DatasetV2]:
    """Create a TensorFlow dataset

    Returns:
        tf.data.Dataset: Batched and prefetched dataset.

    """
    train, test = get_train_test_images()
    return (
        DatasetV2.from_generator(  # pyright: ignore[reportUnknownMemberType]
            get_data_generator(train),
            output_signature=(
                TensorSpec(shape=(*ENV.SIZE, ENV.DIMENSIONS), dtype=tft.float32),
                TensorSpec(shape=(), dtype=tft.bool),
            ),
        )
        .repeat(count=3)
        .batch(ENV.BATCH_SIZE)
        .prefetch(AUTOTUNE)
    ), (
        DatasetV2.from_generator(  # pyright: ignore[reportUnknownMemberType]
            get_data_generator(test),
            output_signature=(
                TensorSpec(shape=(*ENV.SIZE, ENV.DIMENSIONS), dtype=tft.float32),
                TensorSpec(shape=(), dtype=tft.bool),
            ),
        )
        .batch(ENV.BATCH_SIZE)
    )


def create_model() -> k.Sequential:
    model = k.Sequential(
        [
            k.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=(*ENV.SIZE, ENV.DIMENSIONS)
            ),
            k.layers.MaxPooling2D((2, 2)),
            k.layers.Flatten(),
            k.layers.Dense(1, activation="sigmoid"),  # Binary classification
        ]
    )

    # Compile the model
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    # Create the dataset
    train, test = create_dataset()
    model = create_model()
    model.fit(
        train,
        epochs=5,
        validation_data=test,
    )
    print(model.summary())
