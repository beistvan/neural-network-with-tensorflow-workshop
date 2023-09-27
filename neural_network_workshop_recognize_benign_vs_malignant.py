# -*- coding: utf-8 -*-
"""Workshop.ipynb

### Import the necessary modules
"""

import os
import glob
import zipfile
import random

import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tensorflow.keras.utils import get_file


# to get consistent results after multiple runs
tf.random.set_seed(7)
np.random.seed(7)
random.seed(7)

tf.keras.backend.clear_session()

# 0 for benign, 1 for malignant
class_names = ["benign", "malignant"]

# training parameters
batch_size = 64
optimizer = "rmsprop"
loss = "binary_crossentropy"

"""### Preparing the Dataset

For this workshop, we'll be using only a small part of ISIC archive dataset, the below function downloads and extract the dataset into a new data folder

"""

def download_and_extract_dataset():
  train_url = "https://ews0921.s3.eu-central-1.amazonaws.com/train.zip"
  valid_url = "https://ews0921.s3.eu-central-1.amazonaws.com/valid.zip"
  test_url  = "https://ews0921.s3.eu-central-1.amazonaws.com/test.zip"

  for i, download_link in enumerate([valid_url, train_url, test_url]):
    temp_file = f"temp{i}.zip"
    data_dir = get_file(origin=download_link, fname=os.path.join(os.getcwd(), temp_file))
    print("Extracting", download_link)
    with zipfile.ZipFile(data_dir, "r") as z:
      z.extractall("data")
    # remove the temp file
    os.remove(temp_file)

download_and_extract_dataset()

"""### Generates a metadata CSV file for each set

The below cell generates a metadata CSV file for each set, each row in the CSV file corresponds to a path to an image along with its label (0 or 1)

"""

# preparing data
# generate CSV metadata file to read img paths and labels from it
def generate_csv(folder, label2int):
    folder_name = os.path.basename(folder)
    labels = list(label2int)
    # generate CSV file
    df = pd.DataFrame(columns=["filepath", "label"])
    i = 0
    for label in labels:
        print("Reading", os.path.join(folder, label, "*"))
        for filepath in glob.glob(os.path.join(folder, label, "*")):
            df.loc[i] = [filepath, label2int[label]]
            i += 1
    output_file = f"{folder_name}.csv"
    print("Saving", output_file)
    df.to_csv(output_file)

# generate CSV files for all data portions, labeling nevus and seborrheic keratosis
# as 0 (benign), and melanoma as 1 (malignant)
generate_csv("data/train", {"nevus": 0, "seborrheic_keratosis": 0, "melanoma": 1})
generate_csv("data/valid", {"nevus": 0, "seborrheic_keratosis": 0, "melanoma": 1})
generate_csv("data/test", {"nevus": 0, "seborrheic_keratosis": 0, "melanoma": 1})

"""### Load the data into pandas dataframes and datasets


"""

# loading data
train_metadata_filename = "train.csv"
valid_metadata_filename = "valid.csv"

# load CSV files as DataFrames
df_train = pd.read_csv(train_metadata_filename)
df_valid = pd.read_csv(train_metadata_filename)

n_training_samples = len(df_train)
n_validation_samples = len(df_valid)

print("Number of training samples:", n_training_samples)
print("Number of validation samples:", n_validation_samples)

# load DataFrames as DataSets
train_ds = tf.data.Dataset.from_tensor_slices(( df_train["filepath"], df_train["label"] ))
valid_ds = tf.data.Dataset.from_tensor_slices(( df_valid["filepath"], df_valid["label"] ))

"""### Load and decode the images"""

# preprocess data
def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize and return the image to the desired size.
  img = tf.image.resize(img, [299, 299])

  return img

"""#### Prepare dataset for the training"""

def process_path(filepath, label):
  # load the raw data from the file as a string
  img = tf.io.read_file(filepath)
  img = decode_img(img)
  return img, label


valid_ds = valid_ds.map(process_path)
train_ds = train_ds.map(process_path)

def prepare_for_training(ds, cache=True, batch_size=64, shuffle_buffer_size=1000):
  # check cache
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()

  # shuffle the dataset
  ds.shuffle(buffer_size=shuffle_buffer_size)

  # Repeat forever
  ds = ds.repeat()

  # split to batches
  ds = ds.batch(batch_size)

  # `prefetch` lets the dataset fetch batches in the background while the model is training.
  ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

  return ds

valid_ds = prepare_for_training(valid_ds, batch_size=batch_size, cache="valid-cached-data")
train_ds = prepare_for_training(train_ds, batch_size=batch_size, cache="train-cached-data")

"""### Here is what we did:

* **cache():** Since we're making too many calculations on each set, we used cache() method to save our preprocessed dataset into a local cache file, this will only preprocess it the very first time (in the first epoch during training).

* **shuffle():** To basically shuffle the dataset, so the samples are in random order.

* **repeat():** Every time we iterate over the dataset, it'll keep generating samples for us repeatedly, this will help us during the training.

* **batch():** We batch our dataset into 64 or 32 samples per training step.

* **prefetch():** This will enable us to fetch batches in the background while the model is training.

### Show images
"""

batch = next(iter(valid_ds))

def show_batch(batch):

  plt.figure(figsize=(12,12))

  for n in range(25):
      ax = plt.subplot(5,5,n+1)
      plt.imshow(batch[0][n])
      plt.title(class_names[batch[1][n].numpy()].title())
      plt.axis('off')

show_batch(batch)

"""### Building the Model

Notice before, we resized all images to (299, 299, 3), and that's because of what InceptionV3 architecture expects as input, so we'll be using transfer learning with TensorFlow Hub library to download and load the InceptionV3 architecture along with its ImageNet pre-trained weights
"""

from keras.src.layers.attention.multi_head_attention import activation
# building the model
# InceptionV3 model & pre-trained weights
module_url = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"

model = tf.keras.Sequential(
  [
    hub.KerasLayer(module_url, output_shape=[2048], trainable=False),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
  ]
)

model.build([None, 299, 299, 3])

model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

model.summary()

"""### Training the Model

We now have our dataset and the model, let's get them together
"""

model_name = f"benign-vs-malignant_{batch_size}_{optimizer}"
modelcheckpoint = tf.keras.callbacks.ModelCheckpoint(model_name + "_{val_loss:.3f}.h5", save_best_only=True, verbose=1)

model.fit(
    train_ds,
    validation_data=valid_ds,
    steps_per_epoch=n_training_samples // batch_size,
    validation_steps=n_validation_samples // batch_size,
    verbose=1,
    epochs=25,
    callbacks=[modelcheckpoint]
)

"""### Model Evaluation"""

# evaluation
# load testing set
test_metadata_filename = "test.csv"

df_test = pd.read_csv(test_metadata_filename)

n_testing_samples = len(df_test)

print("Number of testing samples:", n_testing_samples)

test_ds = tf.data.Dataset.from_tensor_slices((df_test["filepath"], df_test["label"]))

def prepare_for_testing(ds, cache=True, shuffle_buffer_size=1000):
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()
  ds = ds.shuffle(buffer_size=shuffle_buffer_size)
  return ds

test_ds = test_ds.map(process_path)

test_ds = prepare_for_testing(test_ds, cache="test-cached-data")

# convert testing set to numpy array to fit in memory (don't do that when testing
# set is too large)
y_test = np.zeros((n_testing_samples,))
X_test = np.zeros((n_testing_samples, 299, 299, 3))
for i, (img, label) in enumerate(test_ds.take(n_testing_samples)):
  # print(img.shape, label.shape)
  X_test[i] = img
  y_test[i] = label.numpy()

print("y_test.shape:", y_test.shape)

# load the weights with the least loss
model.load_weights("benign-vs-malignant_64_rmsprop_0.364.h5")

print("Evaluating the model...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Loss:", loss, "  Accuracy:", accuracy)

"""### Set threshold and check accuracy after"""

from sklearn.metrics import accuracy_score

def get_predictions(threshold=None):
  """
  Returns predictions for binary classification given `threshold`
  For instance, if threshold is 0.3, then it'll output 1 (malignant) for that sample if
  the probability of 1 is 30% or more (instead of 50%)
  """
  y_pred = model.predict(X_test)
  if not threshold:
    threshold = 0.5
  result = np.zeros((n_testing_samples,))
  for i in range(n_testing_samples):
    # test melanoma probability
    if y_pred[i][0] >= threshold:
      result[i] = 1
    # else, it's 0 (benign)
  return result

threshold = 0.23
# get predictions with 23% threshold
# which means if the model is 23% sure or more that is malignant,
# it's assigned as malignant, otherwise it's benign
y_pred = get_predictions(threshold)
accuracy_after = accuracy_score(y_test, y_pred)
print("Accuracy after setting the threshold:", accuracy_after)

"""### Predicting the Class of Images

"""

# a function given a function, it predicts the class of the image
def predict_image_class(img_path, model, threshold=0.5):
  img = tf.keras.preprocessing.image.load_img(img_path, target_size=(299, 299))
  img = tf.keras.preprocessing.image.img_to_array(img)
  img = tf.expand_dims(img, 0) # Create a batch
  img = tf.keras.applications.inception_v3.preprocess_input(img)
  img = tf.image.convert_image_dtype(img, tf.float32)

  predictions = model.predict(img)

  score = predictions.squeeze()
  if score >= threshold:
    print(f"This image is {100 * score:.2f}% malignant.")
  else:
    print(f"This image is {100 * (1 - score):.2f}% benign.")

  plt.imshow(img[0])
  plt.axis('off')
  plt.show()

predict_image_class("data/test/nevus/ISIC_0012092.jpg", model)
predict_image_class("data/test/melanoma/ISIC_0015229.jpg", model)