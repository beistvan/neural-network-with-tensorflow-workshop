"""### Excercise 2

In this exercise you'll try to build a neural network that predicts the price
of a house according to a simple formula. So, imagine if house pricing was as
easy as a house costs 50k + 50k per bedroom, so that a 1 bedroom house costs
100k, a 2 bedroom house costs 150k etc. How would you create a neural network
that learns this relationship so that it would predict a 7 bedroom house as
costing close to 400k etc.
Hint: Your network might work better if you scale the house price down. You
don't have to give the answer 400...it might be better to create something that
predicts the number 4, and then your answer is in the 'hundreds of thousands' etc.
"""

import numpy as np
from tensorflow import keras
from keras.src.optimizers import optimizer

def house_model(y_new):

  xs = []
  ys = []

  for i in range(1, 10):
    xs.append(i) # num of bedrooms
    ys.append([(1 + float(i)) * 50]) # costs of the house

  xs = np.array(xs, dtype=float)
  ys = np.array(ys, dtype=float)

  layer = keras.layers.Dense(units=1, input_shape=[1])
  model = keras.Sequential(layer)

  model.compile(optimizer="sgd", loss="mean_squared_error")

  model.fit(xs, ys, epochs=4500)

  return (model.predict(y_new))

prediction = house_model([7.0])
print(prediction) # should print something near 400$ # result: [[400.00006]] # pretty accurate
