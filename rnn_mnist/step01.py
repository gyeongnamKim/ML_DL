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

# 데이터를 불러오는 코드를 작성해주세요.
mnist = keras.datasets.mnist
((train_images,train_label),(test_images,test_label)) = mnist.load_data()

#데이터의 크기 확인
print(f'train_images : {train_images.shape}')
print(f'train_images : {train_label.shape}')
print(f'train_images : {test_images.shape}')
print(f'train_images : {train_label.shape}')

#이미지를 plt로 출력
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid()
plt.show()

#train_images에서 0이 아닌 값들을 출력
list(filter(lambda x:x != 0,train_images[0].reshape(-1)))[:10]

#train_images의 dtype 출력
print(train_images.dtype)

#train/test 이미지 데이터의 범위 확인
train_images[0]
print(list(filter(lambda x:x != 0 ,train_images[0].reshape(-1)))[:10])
print(list(filter(lambda x:x != 0 ,train_label.reshape(-1)))[:10])
print(list(filter(lambda x:x != 0 ,test_images[0].reshape(-1)))[:10])
print(list(filter(lambda x:x != 0 ,test_label.reshape(-1)))[:10])

#train/test 이미지 데이터의 최소/최대값을 출력
print(min(train_images.reshape(-1)),max(train_images.reshape(-1)))
print(min(train_label.reshape(-1)),max(train_label.reshape(-1)))
print(min(test_images.reshape(-1)),max(test_images.reshape(-1)))
print(min(test_label.reshape(-1)),max(test_label.reshape(-1)))

#정수형을 실수형으로 변경 후 dtype으로 비교
train_images = train_images.astype(np.float64)
test_images = test_images.astype(np.float64)

# 데이터 0-1 노말라이즈 수행
train_images = train_images / 255.0
test_images = test_images / 255.0

#노멀라이즈 확인
print(list(filter(lambda x:x != 0 ,train_images[0].reshape(-1)))[:10])
print(list(filter(lambda x:x != 0 ,train_label.reshape(-1)))[:10])
print(list(filter(lambda x:x != 0 ,test_images[0].reshape(-1)))[:10])
print(list(filter(lambda x:x != 0 ,test_label.reshape(-1)))[:10])

#train_image의 이미지를 5장 획득하여 (5, 28, 28)의 shape을 출력
plt.imshow(train_images[:5].transpose(1,0,2).reshape(28,-1),cmap='gray')
plt.show()

#랜덤 노이즈 (28,28) 생성
plt.imshow(np.random.normal(0.0,0.1,(28,28)),cmap='gray')
plt.show()

#train_images의 5번째 이미지와 가우시안 노이즈 (28, 28)를 생성 한 뒤
#각각 tensor를 더한 뒤 noisy_image 변수에 할당
noisy_image = train_images[5] + np.random.normal(0.5,0.1,(28,28))
plt.imshow(noisy_image,cmap='gray')
plt.show()

# max값을 1로 조절
noisy_image[noisy_image > 1.0] = 1.0
plt.imshow(noisy_image,cmap='gray')
plt.show()

#랜덤 노이즈를 추가한 train_noisy_images와 test_noisy_images를 생성
train_noisy_images = train_images + np.random.normal(0.5,0.1,train_images.shape)
train_noisy_images[train_noisy_images > 1.0]=1.0
test_noisy_images = test_images + np.random.normal(0.5,0.1,test_images.shape)
test_noisy_images[test_noisy_images > 1.0]=1.0

#plt.imshow(np.hstack(train_noisy_images[:5]),cmap='gray')
plt.imshow(train_noisy_images[:5].transpose(1,0,2).reshape(28,-1),cmap='gray')
plt.show()

#labels에 onehot 인코딩을 적용하여 (배치 사이즈, 클래스 개수)의 shape으로 변경
from tensorflow.keras.utils import to_categorical
print(train_label.shape,test_label.shape)
train_label = to_categorical(train_label,10)
test_label = to_categorical(test_label,10)
print(train_label.shape,test_label.shape)

#해당 학습셋을 처리하는 이미지 classification 모델 작성
from keras.layers import SimpleRNN
from keras.layers import Dense, Input
from keras.models import Model

inputs = Input(shape=(28,28))
x1 = SimpleRNN(64,activation='tanh')(inputs)
x2 = Dense(10,activation='softmax')(x1)
model = Model(inputs,x2)

model.summary()

#만든 모델에 로스와 옵티마이저, 메트릭을 설정
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#만든 모델에 train_noisy_images를 학습
history = model.fit(train_noisy_images,train_label,validation_data=(test_noisy_images,test_label),
                    epochs=5,verbose=2)

#학습 진행 사항 출력
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend(loc='upper left')
plt.show()

#완성된 모델에서 test_noisy_image를 1장 넣고 결과를 res 변수에 저정
res = model.predict(test_noisy_images[0:1])
res.shape

# test_noisy_images[0], test_images[0]를 width 방향으로 결합하여 plt로 출력
plt.imshow(np.concatenate([test_noisy_images[0],test_images[0]],axis=1),cmap='gray')
plt.show()

#res와 test_labels[0]의 결과를 plt.bar로 확인
plt.bar(range(10),res[0],color='red')
plt.bar(np.array(range(10)) +0.35, test_label[0])
plt.show()

#모델을 저장
model.save('./lecture001.h5')

#모델을 로드
load_model = tf.keras.models.load_model('./lecture001.h5')

# 로드한 모델을 test 데이터로 평가
loss, acc = load_model.evaluate(test_noisy_images, test_label, verbose=2)
print(loss, acc)
loss, acc = model.evaluate(test_noisy_images, test_label, verbose=2)
print(loss, acc)

#모델을 내 컴퓨터에 저장
from google.colab import files
files.download('./lecture001.h5')