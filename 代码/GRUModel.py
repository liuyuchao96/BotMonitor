import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, GRU
from keras.layers import Conv2D, MaxPooling2D
import os
import packet2idx as pi

def load_data():
    paths = ['source_data\\train', 'source_data\\test']
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for dirname in os.listdir(paths[0]): # 带标签的文件夹名
        path = os.path.join(paths[0], dirname)
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            (x, y) = pi.getMatrixFromEachPacket(file_path)
            x_train.append(x[0])
            y_train.append(y[0])
    for dirname in os.listdir(paths[1]): # 带标签的文件夹名
        path = os.path.join(paths[1], dirname)
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            (x, y) = pi.getMatrixFromEachPacket(file_path)
            x_test.append(x[0])
            y_test.append(y[0])
    print(y_train)
    return (x_train, y_train),  (x_test, y_test)

# 加载数据
(x_train, y_train), (x_test, y_test) = load_data()

num_classes = 256
epochs = 3
batch_size = 8 # 8个数据包
model = Sequential()
input_dim = 256 # 每个字节经onehot编码后有256维
input_length = 100 # 每个数据包包含100个字节
model.add(LSTM(units=256, return_sequences=True, input_shape=(x_train[0].shape[0], num_classes)))
model.add(GRU(units=256, return_sequences=True, input_shape=(x_train.shape[0], num_classes)))
model.add(Dense(1))

if __name__ == '__main__':

    print(x_train[0].shape[0])
    print(y_train)
    # paths = ['source_data/train', 'source_data/test']
    # for dirname in os.listdir(paths[0]): # 带标签的文件夹名
    #     path = os.path.join(paths[0], dirname)
    #     for filename in os.listdir(path):
    #         file_path = os.path.join(path, filename)
    #         print(file_path.split("\\")[1])