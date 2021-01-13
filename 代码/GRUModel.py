import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, GRU
from keras.layers import Conv2D, MaxPooling2D
import os

def load_data():
    paths = ['mnist/train-labels-idx1-ubyte.gz', 'mnist/train-images-idx1-ubyte.gz',
             'mnist/test-labels-idx1-ubyte.gz', 'mnist/test-images-idx1-ubyte.gz']
    with open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    with open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16)
    with open(paths[2], 'rb') as lapath:
        y_test = np.frombuffer(lapath.read(), np.uint8, offset=8)
    with open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8, offset=16)
    return (x_train, y_train), (x_test, y_test)
(x_train, y_train), (x_test, y_test) = load_data()

# 将类别转换成独热编码
num_classes = 256
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = keras.utils.to_categorical(x_train, num_classes)
x_test = keras.utils.to_categorical(x_test, num_classes)

x_train /= 255 # 归一化
x_test /= 255 # 归一化

epochs = 3
batch_size = 8 # 8个数据包
model = Sequential()
input_dim = 256 # 每个字节经onehot编码后有256维
input_length = 100 # 每个数据包包含100个字节
model.add(LSTM(units=256, return_sequences=True, input_shape=(x_train.shape[0], 256)))
model.add(Dense(1))
