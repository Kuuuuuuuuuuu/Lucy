import zipfile
import tensorflow as tf
import numpy as np
import cv2
import pandas
import sys
from IPython.display import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
from keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization

from imutils import face_utils
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import dlib
import cv2

import face_recognition # 얼굴 인식
from PIL import Image, ImageDraw
print(tf.__version__)

'''
path = 'C:/Users/konyang/Desktop/Ku/kuu/lucy/archive.zip'
zip_object = zipfile.ZipFile(file=path, mode='r') # 압축파일 자료 읽기
zip_object.extractall('C:/Users/konyang/Desktop/Ku/kuu/lucy') # zip 파일 내 모든 자료 압축 해제 할 위치
zip_object.close() # 집파일 해체
'''

train_generator = ImageDataGenerator(rotation_range=10,  # Degree range for random rotations
                                     zoom_range=0.2,  # Float or [lower, upper]. Range for random zoom. If a float, [lower, upper] = [1-zoom_range, 1+zoom_range]
                                     horizontal_flip=True,  # Randomly flip inputs horizontally
                                     rescale=1/255)  # Rescaling by 1/255 to normalize

train_dataset = train_generator.flow_from_directory(directory='C:/Users/konyang/Desktop/Ku/kuu/lucy/train',
                                                    target_size=(48, 48),  # Tuple of integers (height, width), defaults to (256, 256)
                                                    class_mode='categorical',
                                                    batch_size=16,  # Size of the batches of data (default: 32)
                                                    shuffle=True,  # Whether to shuffle the data (default: True) If set to False, sorts the data in alphanumeric order
                                                    seed=10)

test_generator = ImageDataGenerator(rescale=1/255)

test_dataset = test_generator.flow_from_directory(directory='C:/Users/konyang/Desktop/Ku/kuu/lucy/test',
                                                  target_size=(48, 48),
                                                  class_mode='categorical',
                                                  batch_size=1,
                                                  shuffle=False,
                                                  seed=10)


# CNN 모델 설계
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization

num_classes = 7
num_detectors = 32
width, height = 48, 48

network = Sequential()

network.add(Conv2D(filters=num_detectors, kernel_size=3, activation='relu', padding='same', input_shape=(width, height, 3)))
network.add(BatchNormalization())
network.add(Conv2D(filters=num_detectors, kernel_size=3, activation='relu', padding='same'))
network.add(BatchNormalization())
network.add(MaxPooling2D(pool_size=(2, 2)))
network.add(Dropout(0.2))

network.add(Conv2D(2*num_detectors, 3, activation='relu', padding='same'))
network.add(BatchNormalization())
network.add(Conv2D(2*num_detectors, 3, activation='relu', padding='same'))
network.add(BatchNormalization())
network.add(MaxPooling2D(pool_size=(2, 2)))
network.add(Dropout(0.2))

network.add(Conv2D(2*2*num_detectors, 3, activation='relu', padding='same'))
network.add(BatchNormalization())
network.add(Conv2D(2*2*num_detectors, 3, activation='relu', padding='same'))
network.add(BatchNormalization())
network.add(MaxPooling2D(pool_size=(2, 2)))
network.add(Dropout(0.2))

network.add(Conv2D(2*2*2*num_detectors, 3, activation='relu', padding='same'))
network.add(BatchNormalization())
network.add(Conv2D(2*2*2*num_detectors, 3, activation='relu', padding='same'))
network.add(BatchNormalization())
network.add(MaxPooling2D(pool_size=(2, 2)))
network.add(Dropout(0.2))

network.add(Flatten())

network.add(Dense(2*2*num_detectors, activation='relu'))
network.add(BatchNormalization())
network.add(Dropout(0.2))

network.add(Dense(2*num_detectors, activation='relu'))
network.add(BatchNormalization())
network.add(Dropout(0.2))

network.add(Dense(num_classes, activation='softmax'))

'''
path = 'C:/Users/konyang/Desktop/Ku/kuu/lucy/archive.zip'
zip_object = zipfile.ZipFile(file=path, mode='r') # 압축파일 자료 읽기
zip_object.extractall('C:/Users/konyang/Desktop/Ku/kuu/lucy') # zip 파일 내 모든 자료 압축 해제 할 위치
zip_object.close() # 집파일 해체
'''

# 모델 훈련
network.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy']) # 옵티마이저 손실함수 평가지표

epochs = 1 # 70 정도면 70% 나옴

network.fit(train_dataset, epochs=epochs)

# 모델 성능 평가
network.evaluate(test_dataset)
preds = network.predict(test_dataset) #7가지 클래스에 대한 확률 값, 가장 큰 값이 최종 예측 클래스
preds = np.argmax(preds, axis=1) # 가장 큰 값 클래스
preds

# 예측값과 실제 클래스 값 비교
accuracy_score(test_dataset.classes, preds)
