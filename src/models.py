"""File to store the various models to test"""

import keras as k

layers: list[list[k.Layer]] = [
    [
        k.layers.Conv2D(
            32, (3, 3), activation="relu"
        ),
        k.layers.MaxPooling2D((2, 2)),
        k.layers.Flatten(),
        k.layers.Dense(1, activation="sigmoid"),
    ],
    [
        k.layers.Conv2D(
            16, (3, 3), activation="relu"
        ),
        k.layers.MaxPooling2D((2, 2)),
        k.layers.Conv2D(
            16, (3, 3), activation="relu"
        ),
        k.layers.MaxPooling2D((2, 2)),
        k.layers.Flatten(),
        k.layers.Dense(1, activation="sigmoid"),
    ],
    [
        k.layers.Conv2D(
            32, (3, 3), activation="relu"
        ),
        k.layers.MaxPooling2D((2, 2)),
        k.layers.Flatten(),
        k.layers.Dense(5, activation="relu"),
        k.layers.Dense(1, activation="sigmoid"),
    ],
    [
        k.layers.Conv2D(
            32, (5, 5), activation="relu"
        ),
        k.layers.MaxPooling2D((2, 2)),
        k.layers.Flatten(),
        k.layers.Dense(1, activation="sigmoid"),
    ],
]

names = [
    "Control",
    "Double Conv",
    "Extra Dense Layer",
    "Larger Kernel",
]
