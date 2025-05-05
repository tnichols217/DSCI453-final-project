"""Main module for training multiple models"""

from keras.src.utils.summary_utils import count_params

from model_helpers import create_dataset, create_model, train_model
from models import layer_descriptions, layer_generator
import os

if __name__ == "__main__":
    device = int(os.environ["CUDA_VISIBLE_DEVICES"]) or 0
    print(f"Current device: {device}")
    # Create the dataset
    train, test = create_dataset(1000)
    for l in layer_descriptions[device]:
        model = create_model(layer_generator(l))
        _ = train_model(model, train, test, l["name"])
        parameters = count_params(model.trainable_weights)
        print(parameters)
        model.summary()  # pyright: ignore[reportUnknownMemberType]
