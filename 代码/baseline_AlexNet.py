# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 10:07:59 2019

@author: Administrator
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

#原始数据读入
train = pd.read_csv('./data/train.csv')
test_x = pd.read_csv('./data/test.csv')
train_y = train['label']
del train['label']
train_x = train

#展示灰度矩阵图像
def show(gray_matrix):
    plt.subplot(111)
    plt.imshow(gray_matrix,cmap=plt.get_cmap('gray'))
    
def show_with_index(train_x,index):
    show(np.array(train_x.iloc[index]).reshape((28,28)))


from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam

#训练数据生成
X_train = np.array(train_x).reshape(-1,1,28,28)/255.0
X_test = np.array(test_x).reshape(-1,1,28,28)/255.0
y_train = np_utils.to_categorical(train_y, num_classes=10)

#模型设计(LeNet)
#28*28*1 -> 28*28*32(24M) -> 14*14*32(6M) -> 14*14*64(12M) -> 7*7*64(3M) -> 1024 -> 10
model = Sequential()

# Conv1 64
model.add(Convolution2D(64,5,strides=1,padding='same',data_format='channels_first',batch_input_shape = (None, 1, 28, 28)))
model.add(Activation('relu'))
model.add(MaxPooling2D(2,2,'same',data_format='channels_first'))
# Conv2 128
model.add(Convolution2D(128,5,strides=1,padding='same',data_format='channels_first'))
model.add(Activation('relu'))
model.add(MaxPooling2D(2,2,'same',data_format='channels_first'))
# Conv3 256
model.add(Convolution2D(256,3,strides=1,padding='same',data_format='channels_first'))
model.add(Activation('relu'))
model.add(MaxPooling2D(2,2,'same',data_format='channels_first'))
#4*4*256=4096
#数据平展成一维
model.add(Flatten())

#fc*3
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

#optimizer
adam = Adam(lr=1e-4)

#设置模型损失和优化器以及参考参数
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training ------------')
model.fit(X_train, y_train, epochs=3, batch_size=64)

#print('\nTesting ------------')
## Evaluate the model with the metrics we defined earlier
#loss, accuracy = model.evaluate(X_test, y_test)
#
#print('\ntest loss: ', loss)
#print('\ntest accuracy: ', accuracy)

y_test=model.predict_classes(X_test)
submission = pd.read_csv('./data/sample_submission.csv')
submission['Label']=y_test
submission.to_csv('./data/submission.csv',index=0)
