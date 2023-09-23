import tensorflow as tf
# print(tf.__version__)

import numpy as np
from tensorflow import keras

layer = keras.layers.Dense(units=1, input_shape=[1])
model = keras.Sequential(layer)

model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([ -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([ -3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=500)

# 10 => y = 2 * x - 1 => 2 * 10 - 1 = 19
result = model.predict([10.0])

print(result[0][0]) # [[18.999987]] # expecting 19 # reasons: a) only trained with 6 points only, b) deal with probability
# how to evaluate model?