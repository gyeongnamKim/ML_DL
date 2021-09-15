import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from keras.layers import Dense
from keras.models import Sequential
from IPython.display import Image
warnings.filterwarnings('ignore')
SEED = 34

#데이터 불러오기
celeb = np.load('./multi_rnn/data/celeba_small.npz')
celeb['train_images']

#데이터 확인
x = celeb['train_images'][3]
y = celeb['train_labels'][3]
x.shape
y.shape

plt.imshow(x)
plt.show()
print(y)

#데이터 분류
train_images = celeb['train_images']
train_labels = celeb['train_labels']
test_images = celeb['test_images']
test_labels = celeb['test_labels']

#데이터에 0이 아닌 값 출력
train_images[train_images != 0][:10]

#dtype 확인
train_images.dtype
train_labels.dtype

#이미지 데이터 범위 확인
print(min(train_images.reshape(-1)),max(train_images.reshape(-1)))
print(min(train_labels.reshape(-1)),max(train_labels.reshape(-1)))
train_images.shape
train_labels.shape
test_images.shape
test_labels.shape

#labels 원 핫 인코딩
from tensorflow.keras.utils import to_categorical
train_male_labels, train_smile_labels = np.split(train_labels,2,axis=1)
test_male_labels, test_smile_labels = np.split(test_labels,2,axis=1)
train_male_labels = to_categorical(train_male_labels)
train_smile_labels =to_categorical(train_smile_labels)
test_male_labels =to_categorical(test_male_labels)
test_smile_labels =to_categorical(test_smile_labels)
train_labels_final = np.concatenate([train_male_labels,train_smile_labels],axis=1)
test_labels_final = np.concatenate([test_male_labels,test_smile_labels],axis=1)
train_labels_final.shape
test_labels_final.shape

#이미지 확인
train_images[:5].shape
plt.imshow(np.hstack(train_images[:5]))
plt.show()
plt.imshow(train_images[:5].transpose((1,0,2,3)).reshape(72,-1,3))
plt.show()
print(train_labels_final[:5])

#모델 생성
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, Input, Dense, Flatten
def simple_model():
    inputs = Input((72,59,3))

    x = Conv2D(32,3,activation='relu')(inputs)
    x = MaxPool2D(2)(x)
    x = Conv2D(64,3,activation='relu')(x)
    x = MaxPool2D(2)(x)
    x = Flatten()(x)
    outputs = Dense(2,activation='softmax')(x)
    model = Model(inputs,outputs)
    return model

#모델 할당
gender_model = simple_model()
smile_model = simple_model()

gender_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
smile_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
gender_model.get_weights()[0][0][0][0]
smile_model.get_weights()[0][0][0][0]

#모델 학습
gender_history = gender_model.fit(train_images,train_male_labels,validation_data=(test_images,test_male_labels),epochs=15,verbose=2)
smile_history = smile_model.fit(train_images,train_smile_labels,validation_data=(test_images,test_smile_labels),epochs=15,verbose=2)

#모델 학습 진행사항
plt.plot(gender_history.history['accuracy'], label = 'gender_accuracy')
plt.plot(gender_history.history['loss'], label = 'gender_loss')
plt.plot(gender_history.history['val_accuracy'], label = 'gender_val_accuracy')
plt.plot(gender_history.history['val_loss'], label = 'gender_val_loss')

plt.plot(smile_history.history['accuracy'], label = 'smile_accuracy')
plt.plot(smile_history.history['loss'], label = 'smile_loss')
plt.plot(smile_history.history['val_accuracy'], label = 'smile_val_accuracy')
plt.plot(smile_history.history['val_loss'], label = 'smile_val_loss')

plt.legend(loc='upper left')
plt.show()

#완성된 모델에 이미지 넣기
gender_res = gender_model.predict(test_images[1:2])
gender_res.shape
smile_res = smile_model.predict(test_images[1:2])
smile_res.shape

plt.imshow(test_images[1])
plt.bar(range(2), gender_res[0])
plt.bar(np.array(range(2)) + 0.3, test_male_labels[1])
plt.show()
print(gender_res)
plt.bar(range(2), gender_res[0])
plt.bar(np.array(range(2)) + 0.3, test_male_labels[1])
plt.show()
print(gender_res)

#멀티 아웃풋 모델링
from keras.layers import Concatenate
def multi_model():
    inputs = Input((72,59,3))

    l1 = Conv2D(32,3,activation='relu')(inputs)
    l2 = MaxPool2D(2)(l1)
    l3 = Conv2D(64,3,activation='relu')(l2)
    l4 = MaxPool2D(2)(l3)
    l5 = Conv2D(64,3,activation='relu')(l4)
    l6 = MaxPool2D(2)(l5)

    l7 = Flatten()(l6)
    latent_vector = Dense(64,activation='relu')(l7)

    gender_output = Dense(2,activation='softmax')(latent_vector)
    smile_output = Dense(2,activation='softmax')(latent_vector)

    outputs = Concatenate(axis=1)([gender_output,smile_output])
    model = Model(inputs,outputs)
    return model

model_multi1 = multi_model()
model_multi1.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

multi_history1 = model_multi1.fit(train_images,train_labels_final,validation_data=(test_images,test_labels_final),epochs=15,verbose=1)
train_labels_final

#모델 학습 진행 확인
plt.plot(multi_history1.history['accuracy'], label = 'accuracy')
plt.plot(multi_history1.history['loss'], label = 'loss')
plt.plot(multi_history1.history['val_accuracy'], label = 'val_accuracy')
plt.plot(multi_history1.history['val_loss'], label = 'val_loss')
plt.legend(loc='upper left')
plt.show()
multi_history1.history.keys()

#완성 모델에 이미지 넣기
res = model_multi1.predict(test_images[3:4])
plt.plot(multi_history1.history['accuracy'], label = 'accuracy')
plt.plot(multi_history1.history['loss'], label = 'loss')
plt.plot(multi_history1.history['val_accuracy'], label = 'val_accuracy')
plt.plot(multi_history1.history['val_loss'], label = 'val_loss')
plt.show()

#모델 분리
gender_model_2 = Model(inputs= model_multi1.inputs,outputs = model_multi1.get_layer('dense_10').output)
gender_model_2.summary()
plt.imshow(test_images[0])
x = gender_model_2.predict(test_images[0:1])
x.argmax()

smile_model_2 = Model(inputs= model_multi1.inputs,outputs = model_multi1.get_layer('dense_9').output)
gender_model_2.summary()
plt.imshow(test_images[0])
x = gender_model_2.predict(test_images[0:1])
x.argmax()