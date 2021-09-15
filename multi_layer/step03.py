import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from IPython.display import Image

warnings.filterwarnings('ignore')

SEED = 34

#데이터셋 불러오고 확인
fashiom_mnist = keras.datasets.fashion_mnist
((train_images,train_labels),(test_images,test_labels)) = fashiom_mnist.load_data()
train_images.shape
train_labels.shape
test_images.shape
test_labels.shape
print(min(train_images.reshape(-1)),max(train_images.reshape(-1)))
print(min(train_labels.reshape(-1)),max(train_labels.reshape(-1)))
print(min(test_images.reshape(-1)),max(test_images.reshape(-1)))
print(min(test_labels.reshape(-1)),max(test_labels.reshape(-1)))

#레이블 설정
labels = ["T-shirt/top",  # index 0
        "Trouser",      # index 1
        "Pullover",     # index 2
        "Dress",        # index 3
        "Coat",         # index 4
        "Sandal",       # index 5
        "Shirt",        # index 6
        "Sneaker",      # index 7
        "Bag",          # index 8
        "Ankle boot"]   # index 9
def idx2label(idx):
    return labels[idx]

def show(idx):
    plt.imshow(train_images[idx],cmap='gray')
    plt.title(idx2label(train_labels[idx]))
    plt.show()

show(12)
#0이 아닌 값 출력
train_images[train_images != 0 ][:5]

list(filter(lambda x:x != 0, train_images.reshape(-1)))[:10]

#이미지 값이 가장 큰 idx와 가장 작은 idx 출력
print(train_images.reshape(6000,-1).sum(axis=1).argmax())
print(train_images.reshape(6000,-1).sum(axis=1).argmin())
print(train_images.reshape(6000,-1).sum(axis=1)[5988])
print(train_images.reshape(6000,-1).sum(axis=1)[5852])
show(5988)
show(5852)

#dtype 확인
train_images.dtype

#정수형을 실수형으로 변환
train_images = train_images.astype(np.float64)
test_images = test_images.astype(np.float64)
print(min(train_images.reshape(-1)),max(train_images.reshape(-1)))
print(min(test_images.reshape(-1)),max(test_images.reshape(-1)))

#노멀라이징
def norm(data):
    min_v = data.min()
    max_v = data.max()
    return (data - min_v)/(max_v - min_v)
train_images = norm(train_images)
test_images = norm(test_images)
print(min(train_images.reshape(-1)),max(train_images.reshape(-1)))
print(min(test_images.reshape(-1)),max(test_images.reshape(-1)))

train_images[train_images != 0 ][:10]
test_images[test_images != 0 ][:10]

#train_images 5장 출력
plt.imshow(np.hstack(train_images[:5]),cmap='gray')
plt.imshow(train_images[:5].transpose(1,0,2).reshape(28,-1),cmap='gray')
def filter(label,count):
    imgs = train_images[np.argwhere(train_labels==label)[:count,...,0]].transpose(1,0,2).reshape(28,-1)
    plt.imshow(imgs)
    plt.show()
filter(1,10)

#이미지 4배로 확장 후 4분면 중 한 곳에 이미지 넣기
def expand_4items(img):
    bg = np.zeros(img.shape)
    idx = np.random.randint(0,4)
    slots = [bg,bg,bg,bg]
    slots[idx] = img
    expanded = np.vstack([np.hstack(slots[:2]),
                          np.hstack(slots[2:])])
    return expanded
plt.imshow(expand_4items(train_images[0]))

#train_images, test_images에 저장
train_expand_images = np.array([expand_4items(img) for img in train_images])
test_expand_images = np.array([expand_4items(img) for img in test_images])
train_expand_images.shape
test_expand_images.shape
train_expand_images.min()
train_expand_images.max()

#4배 이미지에 객체를 랜덤 갯수로 4분면에 위치시키는 함수 생성
def expand_4times2(x_data, y_data):
    images = []
    labels = []

    for _ in range(4):
        bg = np.zeros((28, 28))
        obj_count = np.random.randint(0, 5)

        label = np.zeros((10,))  # [0,0,0,0,0,0,0 ...]
        slots = [bg, bg, bg, bg]

        for idx in range(obj_count):
            i = np.random.randint(len(x_data))
            slots[idx] = x_data[i]
            label += tf.keras.utils.to_categorical(y_data[i], 10)

        np.random.shuffle(slots)

    new_img = np.vstack([
        np.hstack(slots[:2]),
        np.hstack(slots[2:])
    ])
    images.append(new_img)
    labels.append((label >= 1).astype(np.int))
    return np.array(images), np.array(labels)

plt.imshow((expand_4times2(train_images,train_labels)[0][0]))

#4배 이미지 트레인 데이터 셋 생성
train_multi_images, train_multi_labels = list(zip(*[expand_4times2(train_images, train_labels) for i in train_images]))
test_multi_images, test_multi_labels = list(zip(*[expand_4times2(test_images, test_labels) for i in test_images]))

train_multi_images = np.array(train_multi_images)[:, 0, :, :].reshape(-1, 56, 56, 1)
train_multi_labels = np.array(train_multi_labels)[:, 0, :]

test_multi_images = np.array(test_multi_images)[:, 0, :, :].reshape(-1, 56, 56, 1)
test_multi_labels = np.array(test_multi_labels)[:, 0, :]
tf.keras.utils.to_categorical(train_labels[0], 10)
plt.bar ([0, 1,2,3,4,5,6,7,8,9], tf.keras.utils.to_categorical(train_labels).sum(axis=0))
plt.bar ([0, 1,2,3,4,5,6,7,8,9], train_multi_labels.sum(axis=0))

#모델링
from keras.layers import Input, Conv2D, MaxPool2D, Dropout, Dense, Flatten, GlobalAvgPool2D
from keras.models import Model
def single_fashion_mnist_model():
    inputs = Input((56,56,1))
    x = Conv2D(16,2,padding='same',activation='relu')(inputs)
    x = MaxPool2D(2)(x)
    x = Dropout(0.3)(x)

    x = Conv2D(32, 2, padding='same', activation='relu')(inputs)
    x = MaxPool2D(2)(x)
    x = Dropout(0.3)(x)

    x = Conv2D(64, 2, padding='same', activation='relu')(inputs)
    x = MaxPool2D(2)(x)
    x = Dropout(0.3)(x)
    x = Flatten()(x)
    x = Dense(10,activation='softmax')(x)
    return Model(inputs,x)
def single_fashion_mnist_model2():
    inputs = Input((56, 56, 1))
    x = Conv2D(16, 2, padding="same", activation="relu")(inputs)
    x = MaxPool2D(2)(x)
    x = Dropout(0.3)(x)

    x = Conv2D(32, 2, padding="same", activation="relu")(x)
    x = MaxPool2D(2)(x)
    x = Dropout(0.3)(x)

    x = Conv2D(64, 2, padding="same", activation="relu")(x)
    x = MaxPool2D(2)(x)
    x = GlobalAvgPool2D()(x)
    x = Dense(10, activation="softmax")(x)
    return Model(inputs, x)
model = single_fashion_mnist_model2()
model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
history = model.fit(train_expand_images.reshape((-1,56,56,1)),
                    tf.keras.utils.to_categorical(train_labels,10),
                    validation_data=(test_expand_images.reshape((-1,56,56,1)),
                                     tf.keras.utils.to_categorical(test_labels,10)),
                    epochs=15,verbose=1)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend(loc='upper left')
plt.show()

#모델에 이미지 넣어보기
res = model.predict(test_expand_images[2].reshape(1,56,56,1))
res.shape
plt.imshow(test_expand_images[2])
plt.bar(np.arange(0, 10), tf.keras.utils.to_categorical(test_labels[2], 10), color="black")
plt.show()
idx2label(test_labels[2])

#멀티 레이어 모델링
def multi_fashin_mnist_model(model):
    model.trainable = False
    x = model.layers[-2].output
    x = Dense(10, activation='sigmoid')(x)
    return Model(model.input, x)
model.summary()

new_model = multi_fashin_mnist_model(model)
new_model.summary()

#모델 학습
new_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
history_2 = new_model.fit(train_multi_images,
                 train_multi_labels,
                 validation_data = (test_multi_images, test_multi_labels),
                 epochs = 15,
                 verbose = 1)

res = new_model.predict(test_multi_images[8].reshape((1,56,56,1)))
res.shape
plt.imshow(test_multi_images[8])
plt.bar(labels,    test_multi_labels[8], color="black")