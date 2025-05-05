"""Collection of functions to help with producing the model Dataset"""

import random
from pathlib import Path

import keras as k
from keras.api.callbacks import History
import numpy as np
import pandas as pd
import tensorflow.python.framework.dtypes as tft
import tensorflow.python.ops.script_ops as tpo
from numpy.typing import NDArray
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE, DatasetV2
from tensorflow.python.framework.tensor import Tensor

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


def get_train_test_images(ratio: float = 0.1) -> tuple[list[str], list[str]]:
    """Splits the image list into training and testing sets.

    Args:
        ratio (float): The ratio of the training set size to the total set size.

    Returns:
        tuple[list[Path], list[Path]]: A tuple of training and testing image file paths.

    """
    files = get_image_list()
    test_size = int(len(files) * ratio)
    fstr = [str(i.absolute()) for i in files]
    return fstr[test_size + 1 :], fstr[:test_size]


def load_one(fp: Path | str) -> tuple[NDArray[np.uint8], bool]:
    """Loads an image from the specified file path.

    Args:
        fp (Path): The file path to load the image from.

    Returns:
        tf.Tensor: The loaded image tensor.

    """
    fp = Path(fp)
    image = read_file(fp)
    label = fp.parent.name == SUBDIR.AI
    image = np.moveaxis(image, 0, -1)

    return image, label


def load_one_tensor(fp: Tensor) -> tuple[Tensor, bool]:
    """Loads an image from the specified file path.

    Args:
        fp (Tensor): The file path to load the image from.

    Returns:
        tf.Tensor: The loaded image tensor.

    """
    image, label = tpo.numpy_function(
        func=lambda x: load_one(x.decode("utf-8")), inp=[fp], Tout=(tft.uint8, tft.bool)
    )
    image.set_shape([*ENV.SIZE, ENV.DIMENSIONS])
    label.set_shape([])
    return image, label


def create_dataset(size: int = -1) -> tuple[DatasetV2, DatasetV2]:
    """Create a TensorFlow dataset

    Returns:
        tf.data.Dataset: Batched and prefetched dataset.

    """
    train, test = get_train_test_images()
    return (
        DatasetV2.from_tensor_slices(train)
        .map(load_one_tensor)
        .batch(ENV.BATCH_SIZE)
        .prefetch(AUTOTUNE)
        .take(size)
    ), (
        DatasetV2.from_tensor_slices(test)
        .map(load_one_tensor)
        .batch(ENV.BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )


def create_model(layers: list[k.Layer]) -> k.Sequential:
    """Create a simple convolutional neural network model.

    Returns:
        k.Sequential: The compiled convolutional neural network model.

    """
    model = k.Sequential(layers)

    # Compile the model
    model.compile(  # pyright: ignore[reportUnknownMemberType]
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(
    model: k.Sequential, train: DatasetV2, test: DatasetV2, name: str
) -> k.Sequential:
    """Trains a given tensorflow model, with checkpointing

    Args:
        model: The model to train
        train: The training dataset
        test: The testing dataset
        name: The name to save this model as

    Returns:
        The original model

    """
    history: History = model.fit(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        train,
        epochs=10,
        validation_data=test,
        callbacks=[
            k.callbacks.ModelCheckpoint(
                f"{name}.keras",
                monitor="val_loss",
                verbose=0,
                save_best_only=True,
                save_weights_only=False,
                mode="auto",
                save_freq="epoch",
                initial_value_threshold=None,
            ),
            k.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=1,
                restore_best_weights=True
            )
        ]
    )
    json_file = Path(f"{name}-history.json")
    _ = pd.DataFrame(history.history).to_json(json_file.open("w"))
    model.save(f"{name}-final.keras")  # pyright: ignore[reportUnknownMemberType]
    return model
