from numpy import expand_dims
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import  ImageDataGenerator
from matplotlib import pyplot as plt


img=load_img('./chap05/data/bird.jpg')
data=img_to_array(img)
#width_shift_range 이용한 이미지 증가
img_data=expand_dims(data, 0)
data_gen=ImageDataGenerator(zoom_range=[0.4,1.5])
data_iter=data_gen.flow(img_data, batch_size=1)
fig=plt.figure(figsize=(30,30))
for i in range(9):
    plt.subplot(3, 3, i+1)
    batch=data_iter.next()
    image=batch[0].astype('uint16')
    plt.imshow(image)
plt.show()