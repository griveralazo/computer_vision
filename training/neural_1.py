#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 21:59:00 2020

@author: griveralazo
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

model = keras.Sequential(
    [
        layers.Dense(units=1, input_shape=[1]),
        # layers.Dense(2, activation="relu", name="layer1"),
        # layers.Dense(3, activation="relu", name="layer2"),
        # layers.Dense(4, name="layer3"),
    ]
)

## y = 3 x + 1 
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

model.fit(xs, ys, epochs=500)


print(model.predict([10.0]))


