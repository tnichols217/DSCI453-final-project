"""Generates a summary for trained models"""

import json
from pathlib import Path

from keras import Sequential
from keras.api.models import load_model
from keras.src.utils.summary_utils import count_params
from tensorflow.python.types.data import DatasetV2

from model_helpers import create_dataset


def generate_report(name: str, test: DatasetV2):
    model: Sequential = load_model(name)
    parameters = count_params(model.trainable_weights)
    eval: list[int] = list(model.evaluate(test))
    return {
        "name": name.split("-final")[0],
        "parameters": parameters,
        "loss": eval[0],
        "accuracy": eval[1],
    }

if __name__ == "__main__":
    # Create the dataset
    files = list(Path.glob(Path(), "*-final.keras"))
    train, test = create_dataset(1000)
    reports = [
        generate_report(str(name), test)
        for name in files
    ]
    with Path("./report.json") as p:
        _ = p.write_text(json.dumps(reports))
