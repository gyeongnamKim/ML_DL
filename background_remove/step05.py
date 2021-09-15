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

#pfcn_small 불러오기
pfcn_small = np.load('./background_remove/data/pfcn_small.npz')
train_images = pfcn_small['train_images']
train_mattes = pfcn_small['train_mattes']
test_images = pfcn_small['test_images']
test_mattes = pfcn_small['test_mattes']

#데이터 확인
plt.imshow(train_images[0])
plt.imshow(train_mattes[0])
train_images.shape
test_images.shape
train_mattes.shape
test_mattes.shape
train_images.dtype
test_images.dtype
train_mattes.dtype
test_mattes.dtype


#0이 아닌 값 출력
print(list(filter(lambda x: x != 0,train_images.reshape(-1)))[:10])
print(list(filter(lambda x: x != 0,test_images.reshape(-1)))[:10])
print(list(filter(lambda x: x != 0,train_mattes.reshape(-1)))[:10])
print(list(filter(lambda x: x != 0,test_mattes.reshape(-1)))[:10])

#범위 확인
print(train_images.min(),train_images.max())
print(test_images.min(),test_images.max())
print(train_mattes.min(),train_mattes.max())
print(test_mattes.min(),test_mattes.max())

#mattes를 흑백 shape로 변경
from skimage import color
train_mattes = np.array([color.rgb2gray(img).reshape((100,75,1)) for img in train_mattes])
test_mattes = np.array([color.rgb2gray(img).reshape((100,75,1)) for img in test_mattes])
plt.imshow(np.hstack(train_images[:5]))
plt.imshow(train_images[:5].transpose((1,0,2,3)).reshape(100,-1,3))

plt.imshow(train_mattes[:5].transpose((1,0,2,3)).reshape(100,-1,1),cmap='gray')

#AE모델링
from keras.layers import Dense, Input, Conv2D, UpSampling2D, Flatten, Reshape
from keras.models import Model

def ae_like():
    inputs = Input((100,75,3))
    x = Conv2D(32,3,2,activation='relu',padding='same')(inputs)
    x = Conv2D(64,3,2,activation='relu',padding='same')(x)
    x = Conv2D(128,3,2,activation='relu',padding='same')(x)
    x = Flatten()(x)
    latent = Dense(10)(x)

    x = Dense((13*10*128))(latent)
    x = Reshape((13,10,128))(x)

    x = UpSampling2D(size=(2,2))(x)
    x = Conv2D(128,(2,2),(1,1),activation='relu',padding='valid')(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, (1, 1), (1, 1), activation='relu', padding='valid')(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(32, (1, 2), (1, 1), activation='relu', padding='valid')(x)

    x = Conv2D(1,(1,1),(1,1),activation='sigmoid')(x)
    model = Model(inputs,x)
    return model

model = ae_like()
model.summary()

model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])

history = model.fit(train_images,train_mattes,validation_data=(test_images,test_mattes),epochs=25,verbose=1)

plt.plot(history.history['accuracy'], label = "accuracy")
plt.plot(history.history['loss'], label = "loss")
plt.plot(history.history['val_accuracy'], label = "val_accuracy")
plt.plot(history.history['val_loss'], label = "val_loss")
plt.legend(loc = "upper left")
plt.show()

#결과 확인
res = model.predict(test_images[0:1])

plt.imshow(np.concatenate([res[0],test_mattes[0]]).reshape((2,-1,75,1)).transpose([1,0,2,3]).reshape((100,-1)),cmap='gray')

plt.imshow( np.concatenate([(res[0] > 0.5).astype(np.float64),test_mattes[0]]).reshape((2,-1,75,1)).transpose([1,0,2,3]).reshape((100,-1)),cmap='gray')

five = (model.predict(test_images[:5] > 0.5).astype(np.float64))

plt.imshow(np.concatenate([five,test_mattes[:5]]).transpose([1,0,2,3]).reshape((100,-1)),cmap='gray')

plt.imshow(test_images[:5].transpose((1, 0, 2, 3)).reshape((100, -1, 3)))

plt.imshow(five[2].reshape(100,75,1) * test_images[2])

#u-net 모델링
from keras.layers import Dense, Input, Conv2D, Conv2DTranspose, Flatten, Reshape
from keras.models import Model
from keras.layers import BatchNormalization, Dropout, Activation, MaxPool2D, concatenate

def conv2d_block(x,channel):
    x = Conv2D(channel,3,padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(channel,3,padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def unet_like():
    inputs = Input((100,75,3))

    c1 = conv2d_block(inputs,16)
    p1 = MaxPool2D((2,2))(c1)
    p1 = Dropout(0.1)(p1)

    c2= conv2d_block(p1,32)
    p2 = MaxPool2D((2,2))(c2)
    p2 = Dropout(0.1)(p2)

    c3 = conv2d_block(p2, 64)
    p3 = MaxPool2D((2, 2))(c3)
    p3 = Dropout(0.1)(p3)

    c4 = conv2d_block(p3, 128)
    p4 = MaxPool2D((2, 2))(c4)
    p4 = Dropout(0.1)(p4)

    c5 = conv2d_block(p4, 256)

    u6 = Conv2DTranspose(128,2,2,output_padding=(0,1))(c5)
    u6 = concatenate([u6,c4])
    u6 = Dropout(0.1)(u6)
    c6 = conv2d_block(u6,128)

    u7 = Conv2DTranspose(64, 2, 2, output_padding=(1, 0))(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(0.1)(u7)
    c7 = conv2d_block(u7, 64)

    u8 = Conv2DTranspose(32, 2, 2, output_padding=(0, 1))(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(0.1)(u8)
    c8 = conv2d_block(u8, 32)

    u9 = Conv2DTranspose(16, 2, 2, output_padding=(0, 1))(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(0.1)(u9)
    c9 = conv2d_block(u9, 16)

    outputs = Conv2D(1, (1, 1), activation="sigmoid")(c9)
    model = Model(inputs,outputs)
    return model

model = unet_like()
model.summary()

model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])

history = model.fit(train_images,train_mattes,validation_data=(test_images,test_mattes),epochs=25,verbose=1)

plt.plot(history.history['accuracy'], label = "accuracy")
plt.plot(history.history['loss'], label = "loss")
plt.plot(history.history['val_accuracy'], label = "val_accuracy")
plt.plot(history.history['val_loss'], label = "val_loss")
plt.legend(loc = "upper left")
plt.show()

#결과 확인
res = model.predict( test_images[2:3] )
imgs = np.concatenate([res.reshape((100, 75, 1)), test_mattes[2]]).reshape((2, -1, 75, 1)).transpose((1,0, 2, 3)).reshape((100, -1))
plt.imshow(imgs, cmap="gray")

imgs = np.concatenate([(res > 0.5).astype(np.float64).reshape((100, 75, 1)), test_mattes[2] ]).reshape((2, -1, 75, 1)).transpose((1,0, 2, 3)).reshape((100, -1))
plt.imshow(imgs, cmap="gray")

five = (model.predict(test_images[:5]) > 0.5).astype(np.float64)
plt.imshow( np.concatenate([five , test_mattes[:5]], axis=2).transpose((1, 0, 2, 3)).reshape(100, -1) , cmap="gray")
plt.imshow(test_images[:5].transpose((1,0, 2, 3)).reshape((100, -1, 3)))

plt.imshow(test_images[2] * test_mattes[2].reshape((100, 75 ,1)))

plt.imshow(test_images[4] * model.predict(test_images[4:5]).reshape((100, 75 ,1)))