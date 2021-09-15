import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(input_shape=(100, 100, 3), activation='relu', kernel_size=(5, 5), filters=32),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(activation='relu', kernel_size=(5, 5), filters=64),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(activation='relu', kernel_size=(5, 5), filters=64),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(activation='relu', kernel_size=(5, 5), filters=64),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')

])
model.summary()


ins= model.inputs
outs= model.layers[0].output
feature_map= Model(inputs= ins, outputs= outs)
feature_map.summary()

img=cv2.imread("./chap05/data/cat.jpg")
plt.imshow(img)

img=cv2.resize(img,(100,100))
img.shape
input_img= np.expand_dims(img, axis=0)
print(input_img.shape)
feature=feature_map.predict(input_img)
print(feature.shape)
fig= plt.figure(figsize=(50,50))
for i in range(16):
    ax=fig.add_subplot(8,4,i+1)
    ax.imshow(feature[0,:,:,i])

ins= model.inputs
outs= model.layers[6].output
feature_map= Model(inputs= ins, outputs= outs)
img=cv2.imread("./chap05/data/cat.jpg")
img=cv2.resize(img,(100,100))
input_img= np.expand_dims(img, axis=0)

feature=feature_map.predict(input_img)
fig= plt.figure(figsize=(50,50))
for i in range(48):
    ax=fig.add_subplot(8,8,i+1)
    ax.imshow(feature[0,:,:,i])