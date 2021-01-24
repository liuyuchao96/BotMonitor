import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, LSTM, GRU
from keras.layers import Conv2D, MaxPooling2D
import os
import packet2idx as pi
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
            x_train.append(x)
            y_train.append(y)
    for dirname in os.listdir(paths[1]): # 带标签的文件夹名
        path = os.path.join(paths[1], dirname)
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            (x, y) = pi.getMatrixFromEachPacket(file_path)
            x_test.append(x)
            y_test.append(y)
    x_train = np.array(x_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[2] * x_train.shape[1], x_train.shape[-1]))
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[2] * x_test.shape[1], x_test.shape[-1]))
    y_test = np.array(y_test)
    return (x_train, y_train),  (x_test, y_test)

# 加载数据
(x_train, y_train), (x_test, y_test) = load_data()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

num_classes = 256
epochs = 3
batch_size = 4 # 8个数据包
input_dim = 256 # 每个字节经onehot编码后有256维
time_stamps = x_train[0].shape[0] # 每个数据包包含100个字节

model = Sequential()
model.add(GRU(units=100, return_sequences=True, input_shape=(time_stamps, input_dim)))
model.add(Dense(units=256))
model.add(GRU(units=256, return_sequences=False))
model.add(Dense(units=10))
# model.add(Dense(units=1, activation="softmax")) # 全连接层
model.summary() # 查看网络输入输出结构
# model.compile(loss='mae', optimizer='adam', metrics=['acc'])
output = model.predict(x_train)
print(output.shape)

# 拟合模型
# history = model.fit(x_train, y_train, batch_size=batch_size, # 按batch_size大小从x,y生成增强数据
#                                 # flow_from_directory()从路径生成增强数据,和flow方法相比最大的优点在于不用
#                     epochs=epochs,
#                     # steps_per_epoch=x_train.shape[0]//batch_size,
#                     # validation_steps=x_train.shape[0]//batch_size,
#                     validation_data=(x_test, y_test),
#                     )
#
# # #######可视化#######
# #
# # 绘制训练 & 测试的准确率值
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.savefig('tradition_cnn_test_acc.png')
# plt.show()
#
# # 绘制训练 & 测试的损失值
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Valid'], loc='upper left')
# plt.savefig('tradition_cnn_valid_loss.png')
# plt.show()

if __name__ == '__main__':
    print("hello world")
    # paths = ['source_data/train', 'source_data/test']
    # for dirname in os.listdir(paths[0]): # 带标签的文件夹名
    #     path = os.path.join(paths[0], dirname)
    #     for filename in os.listdir(path):
    #         file_path = os.path.join(path, filename)
    #         print(file_path.split("\\")[1])