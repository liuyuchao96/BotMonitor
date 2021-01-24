import gzip
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, GRU
from keras.layers import Conv2D, MaxPooling2D
import os
import matplotlib.pyplot as plt
import functools
import packet2idx as pi

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第3块显卡

# 定义常量
batch_size = 4
epochs = 5
data_augmentation = True #图像增强
# num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models_cnn&gru')
cnn_model_name = 'trained_model_cnn.h5'
gru_model_name = 'trained_model_gru.h5'
final_model_name = 'final_trained_model.h5'

num_classes = 2
epochs = 3
# batch_size = 4 # 8个数据包
input_dim = 256 # 每个字节经onehot编码后有256维
time_stamps = 800 # 每条流取前8个数据包，每个数据包包含100个字节

# 数据预处理（数据包五元组分流，转图像，转idx等）
# 数据加载
def load_data_for_cnn():
    paths = ['mnist/train-labels-idx1-ubyte.gz', 'mnist/train-images-idx3-ubyte.gz',
             'mnist/test-labels-idx1-ubyte.gz', 'mnist/test-images-idx3-ubyte.gz']
    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16)
    with gzip.open(paths[2], 'rb') as lapath:
        y_test = np.frombuffer(lapath.read(), np.uint8, offset=8)
    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8, offset=16)
    return (x_train, y_train), (x_test, y_test)

def load_data_for_gru():
    paths = ['source_data\\train', 'source_data\\test']
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for dirname in os.listdir(paths[0]): # 带标签的文件夹名
        path = os.path.join(paths[0], dirname)
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            x = pi.getMatrixFromEachPacket(file_path)
            x_train.append(x)
            label = int(file_path.split('\\')[2])
            y_train.append(label)
            # y_train.append(y)
    for dirname in os.listdir(paths[1]): # 带标签的文件夹名
        path = os.path.join(paths[1], dirname)
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            x = pi.getMatrixFromEachPacket(file_path)
            x_test.append(x)
            label = int(file_path.split('\\')[2])
            # y_test.append(y)
            y_test.append(label)
    x_train = np.array(x_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[2] * x_train.shape[1], x_train.shape[-1]))
    y_train = np.array(y_train)
    y_train = np.reshape(y_train, (y_train.shape[0], 1))
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[2] * x_test.shape[1], x_test.shape[-1]))
    y_test = np.array(y_test)
    y_test = np.reshape(y_test, (y_test.shape[0], 1))
    return (x_train, y_train),  (x_test, y_test)

def CNN_Model_output(x):
    # 将类别转换成独热编码
    # y = keras.utils.to_categorical(y, num_classes)
    x = np.reshape(x, [-1, 1, 32, 32])
    x = x.astype('float32')

    x /= 255  # 归一化

    model = Sequential()
    # 第一层
    model.add(Conv2D(  # 图片输入形式
        batch_input_shape=(None, 1, 32, 32),  # batch,channels,rows,columns # 第一层需要指出图像的大小
        data_format='channels_first',  # if channel_last: then (batch, rows, columns, channels)
        # 指定滤波器方式
        filters=32,  # 滤波器数量
        kernel_size=3,  # 滤波器（卷积核）尺寸（5*5*1）
        # 卷积方式
        padding='same',  # 输出尺寸和输入尺寸保持一致
        strides=1,  # 卷积步长
    ))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # 第二层
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # 第三层
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    # model.add(Activation('softmax'))

    # # 初始化RMSprop优化器
    # opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
    #
    # # 使用RMSprop优化器
    # model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
    output = model.predict(x, batch_size=batch_size)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, cnn_model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)
    return output

def GRU_Model_output(x):
    model = Sequential()
    model.add(GRU(units=100, return_sequences=True, input_shape=(time_stamps, input_dim)))
    model.add(Dense(units=256))
    model.add(GRU(units=256, return_sequences=False))
    model.add(Dense(units=10))
    # model.add(Dense(units=1, activation="softmax")) # 全连接层
    model.summary()  # 查看网络输入输出结构
    # model.compile(loss='mae', optimizer='adam', metrics=['acc'])
    output = model.predict(x, batch_size=4)
    # 模型保存
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, gru_model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)
    return output

def main():
    (x_train_cnn, y_train_cnn), (x_test_cnn, y_test_cnn) = load_data_for_cnn()
    (x_train_gru, y_train_gru), (x_test_gru, y_test_gru) = load_data_for_gru()
    output_cnn_train = CNN_Model_output(x_train_cnn)
    output_cnn_test = CNN_Model_output(x_test_cnn)
    output_gru_train = GRU_Model_output(x_train_gru)
    output_gru_test = GRU_Model_output(x_test_gru)
    x_train = np.concatenate((output_cnn_train, output_gru_train), axis=1)
    x_test = np.concatenate((output_cnn_test, output_gru_test), axis=1)
    y_train = y_train_cnn
    y_test = y_test_gru
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    # 初始化RMSprop优化器
    opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)

    # 使用RMSprop优化器
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])

    # 拟合模型
    history = model.fit(x_train, y_train, batch_size=batch_size, # 按batch_size大小从x,y生成增强数据
                                    # flow_from_directory()从路径生成增强数据,和flow方法相比最大的优点在于不用
                        epochs=epochs,
                        # steps_per_epoch=x_train.shape[0]//batch_size,
                        # validation_steps=x_train.shape[0]//batch_size,
                        validation_data=(x_test, y_test),
                        )
    model.summary()
    # 模型保存
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, final_model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)
    # 绘制训练 & 测试的准确率值
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model, accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('tradition_cnn_test_acc.png')
    plt.show()

    # 绘制训练 & 测试的损失值
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc='upper left')
    plt.savefig('tradition_cnn_valid_loss.png')
    plt.show()
if __name__ == '__main__':
    (x_train, y_train),  (x_test, y_test) = load_data_for_gru()
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    print(y_train)
