"""Generates a summary for trained models"""

import json
from pathlib import Path

from keras import Sequential
from keras.api.models import load_model
from keras.src.utils.summary_utils import count_params

from model_helpers import create_dataset
from models import layers, names


def generate_report(num: int):
    model: Sequential = load_model(f"model{num}.keras")
    parameters = count_params(model.trainable_weights)
    eval: list[int] = list(model.evaluate(test))
    return {
        "name": names[num],
        "parameters": parameters,
        "loss": eval[0],
        "accuracy": eval[1],
    }

if __name__ == "__main__":
    # Create the dataset
    train, test = create_dataset(1000)
    reports = [
        generate_report(i)
        for i in range(len(layers))
    ]
    with Path("./report.json") as p:
        _ = p.write_text(json.dumps(reports))
