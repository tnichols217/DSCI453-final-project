"""File to store the various models to test"""

from typing import TypedDict

from keras import Layer as L
from keras import layers as l


class ConvDescription(TypedDict):
    size: int
    filters: int
    strides: int
    pooling: int


class LayersDescription(TypedDict):
    activation: str
    convs: list[ConvDescription]
    dense: list[int]
    name: str


def layer_generator(ls: LayersDescription) -> list[L]:
    out: list[L] = []
    for c in ls["convs"]:
        out.extend(
            [
                l.Conv2D(
                    filters=c["filters"],
                    kernel_size=(c["size"], c["size"]),
                    activation=ls["activation"],
                    strides=(c["strides"], c["strides"]),
                ),
                l.MaxPooling2D((c["pooling"], c["pooling"])),
            ]
        )
    out.append(l.Flatten())
    out.extend([l.Dense(d, activation=ls["activation"]) for d in ls["dense"]])
    out.append(l.Dense(1, activation="sigmoid"))
    return out


CONTROL: LayersDescription = {
    "activation": "relu",
    "convs": [{"size": 3, "filters": 32, "strides": 1, "pooling": 2}],
    "dense": [],
    "name": "Control",
}

KERNELS: list[LayersDescription] = [
    {
        "activation": "relu",
        "convs": [{"size": i, "filters": 16, "strides": max(i - 2, 1), "pooling": 2}],
        "dense": [],
        "name": f"Kernel: {i}",
    }
    for i in [1, 5, 7, 9, 11]
]

FILTERS: list[LayersDescription] = [
    {
        "activation": "relu",
        "convs": [{"size": 3, "filters": i, "strides": 1, "pooling": 2}],
        "dense": [],
        "name": f"Filter: {i}",
    }
    for i in [2, 4, 8, 16, 64, 128]
]

DENSE: list[LayersDescription] = [
    {
        "activation": "relu",
        "convs": [{"size": 3, "filters": 32, "strides": 1, "pooling": 2}],
        "dense": i,
        "name": f"Dense: {len(i)}",
    }
    for i in [[12], [8, 4], [6, 4, 2], [4, 3, 3, 2]]
]

CONVS: list[list[ConvDescription]] = [
    [
        {"size": 3, "filters": 16, "strides": 1, "pooling": 2},
        {"size": 3, "filters": 16, "strides": 1, "pooling": 2},
    ],
    [
        {"size": 3, "filters": 10, "strides": 1, "pooling": 2},
        {"size": 3, "filters": 10, "strides": 1, "pooling": 2},
        {"size": 3, "filters": 10, "strides": 1, "pooling": 2},
    ],
]

MULTI_CONV: list[LayersDescription] = [
    {"activation": "relu", "convs": i, "dense": [], "name": f"Multi-Conv: {len(i)}"}
    for i in CONVS
]

layer_descriptions: list[list[LayersDescription]] = [
    [
        CONTROL,
        *KERNELS,
        *FILTERS
    ],
    [
        *DENSE,
        *MULTI_CONV,
    ]
]
