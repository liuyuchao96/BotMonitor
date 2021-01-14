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
X_train = np.array(train_x).reshape(-1,1,28,28)/255.
X_test = np.array(test_x).reshape(-1,1,28,28)/255.
y_train = np_utils.to_categorical(train_y, num_classes=10)

#模型设计(LeNet)
#28*28*1 -> 28*28*32(24M) -> 14*14*32(6M) -> 14*14*64(12M) -> 7*7*64(3M) -> 1024 -> 10
model = Sequential()

# Conv layer1 output shape (32,28,28)
model.add(Convolution2D(
        #图片输入形式
        batch_input_shape = (None, 1, 28, 28),#batch,channels,rows,columns
        data_format='channels_first',#if channel_last: then (batch, rows, columns, channels)
        #指定滤波器方式
        filters = 32,#滤波器数量
        kernel_size = 5,#滤波器（卷积核）尺寸（5*5*1）
        #卷积方式
        padding = 'same',#输出尺寸和输入尺寸保持一致
        strides = 1,#卷积步长
        ))
model.add(Activation('relu'))

#Pooling layer1(max pooling) output shape(32,14,14)
#池化是一种特殊的卷积，指定了函数和参数
model.add(MaxPooling2D(
        pool_size = 2,
        strides = 2,#不重复
        padding  = 'same',#这里的padding和卷积层的padding是不同的，same 表示pooling时会自动空缺补0， 而valid则不会
        data_format='channels_first'
        ))

#从后面开始就不用指定输入的格式了，只要指定输出的尺寸
#model.add(Convolution2D(
#        data_format='channel_first',#if channel_last: then (batch, rows, columns, channels)
#        #指定滤波器方式
#        filters = 64,#滤波器数量
#        kernel_size = 5,#滤波器（卷积核）尺寸（5*5*1）
#        #卷积方式
#        padding = 'same',#输出尺寸和输入尺寸保持一致
#        strides = 1,#卷积步长
#        ))
#另一种写法
#model.add(Convolution2D(64,5,1,'same','channels_first'))
model.add(Convolution2D(64,5,strides=1,padding='same',data_format='channels_first'))
model.add(Activation('relu'))
model.add(MaxPooling2D(2,2,'same',data_format='channels_first'))

#数据平展成一维
model.add(Flatten())

#全连接层
model.add(Dense(1024))
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
model.fit(X_train, y_train, epochs=1, batch_size=64)

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
