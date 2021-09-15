import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalMaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = ResNet50(include_top=True,
                 weights='imagenet',
                 input_tensor=None,
                 input_shape=None,
                 pooling=None,
                 classes=1000)
model.summary()
model.trainable = False
model = Sequential([model,
                    Dense(2,activation='sigmoid')])
model.summary()
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

BATCH_SIZE = 32
image_height = 100
image_width = 100
train_dir = './chap05/data/catanddog/train'
valid_dir = './chap05/data/catanddog/validation'

train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1
)
train_generator = train.flow_from_directory(train_dir,
                                            target_size=(image_height,image_width),
                                            color_mode='rgb',
                                            batch_size=BATCH_SIZE,
                                            seed=1,
                                            shuffle=True,
                                            class_mode='categorical')
valid = ImageDataGenerator(rescale=1.0/255.0)
valid_generator = valid.flow_from_directory(valid_dir,
                                            target_size=(image_height, image_width),
                                            color_mode='rgb',
                                            batch_size=BATCH_SIZE,
                                            seed=7,
                                            shuffle=True,
                                            class_mode='categorical')
history = model.fit(train_generator, epochs=10,
                    validation_data=valid_generator,
                    verbose=2)
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import  font_manager
font_fname = 'C:/Windows/Fonts/malgun.ttf'
font_family = font_manager.FontProperties(fname=font_fname).get_name()

plt.rcParams['font.family'] = font_family
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))

plt.plot(epochs,accuracy,label='훈련 데이터셋')
plt.plot(epochs,val_accuracy,label='검증 데이터셋')
plt.legend()
plt.title('정확도')
plt.figure()

plt.plot(epochs,loss,label='훈련 데이터셋')
plt.plot(epochs,val_loss,label='검증 데이터셋')
plt.legend()
plt.title('오차')

class_names = ['cat','dog']
validation, label_batch = next(iter(valid_generator))
prediction_value = model.predict_classes(validation)
fig = plt.figure(figsize=(12,8))
fig.subplots_adjust(left=0,right=1,bottom=0,hspace=0.05,wspace=0.05)
for i in range(8):
    ax = fig.add_subplot(2,4,i+1,xticks=[],yticks=[])
    ax.imshow(validation[i,:],cmap=plt.cm.gray_r,interpolation='nearest')
    if prediction_value[i] == np.argmax(label_batch[i]):
        ax.text(3,17,class_names[prediction_value[i]],color='yellow',fontsize=14)
    else:
        ax.text(3,17,class_names[prediction_value[i]],color='red',fontsize=14)

import tensorflow_hub as hub
model = tf.keras.Sequential([
    hub.KerasLayer('https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/4',
                   input_shape=(244,244,3),
                   trainable=False),
    tf.keras.layers.Dense(2,activation='softmax')
])
train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1
)
train_generator = train.flow_from_directory(train_dir,
                                            target_size=(image_height,image_width),
                                            color_mode='rgb',batch_size=BATCH_SIZE,
                                            seed=1, shuffle=True,class_mode='categorical')
valid = ImageDataGenerator(rescale=1.0/255.0)
valid_generator = train.flow_from_directory(valid_dir,
                                            target_size=(image_height,image_width),
                                            color_mode='rgb',batch_size=BATCH_SIZE,
                                            seed=7, shuffle=True,class_mode='categorical')
model.compile(loss='binary_crossentropy',
              optimizer='adam',metrics=['accuracy'])

history = model.fit(train_generator,
                    epochs=10,
                    validation_data=valid_generator,
                    verbose=2)
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))

plt.plot(epochs,accuracy,label='훈련 데이터셋')
plt.plot(epochs,val_accuracy,label='검증 데이터셋')
plt.legend()
plt.title('정확도')
plt.figure()
plt.plot(epochs,loss,label='훈련 데이터셋')
plt.plot(epochs,val_loss,label='검증 데이터셋')
plt.legend()
plt.title('오차')

class_names = ['cat','dog']
validation, label_batch = next(iter(valid_generator))
prediction_value = model.predict_classes(validation)
fig = plt.figure(figsize=(12,8))
fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.1,wspace=0.1)
for i in range(8):
    ax = fig.add_subplot(2,4,i+1,xticks=[],yticks=[])
    ax.imshow(validation[i],cmap=plt.cm.gray_r,interpolation='nearest')
    if prediction_value[i] == np.argmax(label_batch[i]):
        ax.text(3,17,class_names[prediction_value[i]],color='yellow',fontsize=14)
    else:
        ax.text(3,17,class_names[prediction_value[i]],color='red',fontsize=14)