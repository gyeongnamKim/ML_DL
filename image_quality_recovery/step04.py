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

#데이터셋 불러오기
fasion_mnist = keras.datasets.fashion_mnist
((train_images,train_labels),(test_images,test_labels)) = fasion_mnist.load_data()

#shape, dtype 확인
print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)
print(train_images.dtype)
print(train_labels.dtype)
print(test_images.dtype)
print(test_labels.dtype)

#(28,28) 이미지 출력
train_images[0].shape
plt.imshow(train_images[0])

#0이 아닌 값 출력
print(list(filter(lambda x:x !=0,train_images.reshape(-1)))[:10])

#범위 확인
print(train_images.reshape(-1).min(),train_images.reshape(-1).max())
print(train_labels.reshape(-1).min(),train_labels.reshape(-1).max())
print(test_images.reshape(-1).min(),test_images.reshape(-1).max())
print(test_labels.reshape(-1).min(),test_labels.reshape(-1).max())

#정수형 데이터 실수형으로 변환
train_images = train_images.astype(np.float64)
test_images = test_images.astype(np.float64)
test_images.dtype
train_images.dtype

#데이터 노멀라이즈
def normalize(data):
    data_min = data.min()
    data_max = data.max()
    result = (data - data_min)/(data_max - data_min)
    return result
train_images = normalize(train_images)
test_images = normalize(test_images)

train_images = train_images / 255.0
test_images = test_images / 255.0
#범위 확인
print(train_images.reshape(-1).min(),train_images.reshape(-1).max())
print(train_labels.reshape(-1).min(),train_labels.reshape(-1).max())
print(test_images.reshape(-1).min(),test_images.reshape(-1).max())
print(test_labels.reshape(-1).min(),test_labels.reshape(-1).max())

#0이 아닌 데이터 출력
print(list(filter(lambda x:x !=0 ,train_images.reshape(-1)))[:10])

#흑백 이미지를 컬러이미지의 shape으로 변경
from skimage import color

train_images = np.array([color.gray2rgb(img) for img in train_images])
test_images = np.array([color.gray2rgb(img) for img in test_images])
print(train_images.shape,test_images.shape)

#이미지 5장 (5,28,28,3)으로 출력
plt.imshow(np.hstack(train_images[:5]))
plt.imshow(train_images[:5].transpose((1,0,2,3)).reshape(28,-1,3))

#0~1 랜덤으로 3회 출력
np.random.random()
np.random.random()
np.random.random()
plt.imshow(np.random.random((28,28,3)))

#가우시안 노이즈 함수
noisy = np.random.normal(0.5,0.1,(28,28,3))
noisy[noisy > 1.0] = 1.0
plt.imshow(noisy)

#train_images 와 더한 뒤 출력
noisy_image = train_images[48] + noisy
noisy_image[noisy_image > 1.0] = 1.0
plt.imshow(noisy_image)

#train_images와 test_images에 노이지 적용
train_noisy_images = train_images + np.random.normal(0.5, 0.05, train_images.shape)
train_noisy_images[ train_noisy_images > 1.0] = 1.0
test_noisy_images = test_images + np.random.normal(0.5, 0.05, test_images.shape)
test_noisy_images[ test_noisy_images > 1.0] = 1.0

#이미지 5장 출력
plt.imshow(np.hstack(train_noisy_images[:5]))

#모델링
from keras.layers import Dense, Input, Conv2D, Conv2DTranspose, Flatten, Reshape
from keras.models import Model


inputs = Input(shape=(28, 28, 3))
x = Conv2D(32, 3, 2, activation='relu', padding='same')(inputs)
x = Conv2D(64, 3, 2, activation='relu', padding='same')(x)
x = Flatten()(x)
latent = Dense(10)(x)

x = Dense(7 * 7 * 64)(latent)
x = Reshape((7, 7, 64))(x)
x = Conv2DTranspose(64, 3, 2, activation='relu', padding='same')(x)
x = Conv2DTranspose(32, 3, 2, activation='relu', padding='same')(x)
x = Conv2DTranspose(3, 3, activation='sigmoid', padding='same')(x)
model = Model(inputs, x)

model.summary()

model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])

hist = model.fit(train_noisy_images, train_images, validation_data=(test_noisy_images, test_images), epochs=15, verbose=2)

plt.plot(hist.history['accuracy'], label='accuracy')
plt.plot(hist.history['loss'], label='loss')
plt.plot(hist.history['val_accuracy'], label='val_accuracy')
plt.plot(hist.history['val_loss'], label='val_loss')
plt.legend(loc = 'upper left')
plt.show()

#결과 확인
res = model.predict(test_images[0:1])
res.shape

plt.imshow(np.concatenate([test_images[0],res[0],test_noisy_images[0]],axis=1))

five = model.predict(test_noisy_images[:5])
result = np.concatenate( [test_images[:5], five, test_noisy_images[:5]], axis = 2 ).transpose((1, 0, 2, 3)).reshape((28, -1 ,3))
plt.imshow(result)
plt.show()