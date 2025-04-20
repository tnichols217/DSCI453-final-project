"""Main module for training multiple models"""

from keras.src.utils.summary_utils import count_params

from model_helpers import create_dataset, create_model, train_model
from models import layers

if __name__ == "__main__":
    # Create the dataset
    train, test = create_dataset(1000)
    for i in range(len(layers)):
        model = create_model(layers[i])
        _ = train_model(model, train, test, f"model{i}")
        parameters = count_params(model.trainable_weights)
        print(parameters)
        model.summary()  # pyright: ignore[reportUnknownMemberType]
