# -*- coding: UTF-8 -*-
from tensorflow.python.keras.utils import get_file
import gzip
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import matplotlib.pyplot as plt
import functools
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 使用第3块显卡
# 数据加载及预处理
def load_data():
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
(x_train, y_train), (x_test, y_test) = load_data()
print(x_train.shape)
batch_size = 32
num_classes = 2
epochs = 5
data_augmentation = True #图像增强
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models_cnn')
model_name = 'keras_fashion_trained_model.h5'

# 将类别转换成独热编码
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
x_train = np.reshape(x_train, [-1, 1, 32, 32])
x_test = np.reshape(x_test, [-1, 1, 32, 32])
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255 # 归一化
x_test /= 255 # 归一化

print(x_train.shape)

model = Sequential()

# 第一层
model.add(Conv2D(#图片输入形式
        batch_input_shape = (None, 1, 32, 32),#batch,channels,rows,columns # 第一层需要指出图像的大小
        data_format='channels_first',#if channel_last: then (batch, rows, columns, channels)
        #指定滤波器方式
        filters = 32,#滤波器数量
        kernel_size = 3,#滤波器（卷积核）尺寸（5*5*1）
        #卷积方式
        padding = 'same',#输出尺寸和输入尺寸保持一致
        strides = 1, #卷积步长
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
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# 初始化RMSprop优化器
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# 使用RMSprop优化器
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])

######train#####
if not data_augmentation:
    print('Not using data augmentation')
    history = model.fit(x_train, y_train, batch_size=batch_size,
                        epochs=epochs, validation_data=(x_test, y_test),
                        shuffle=True)
else:
    print('Using real-time data augmentation.')
    # 数据预处理与实时数据增强
    datagen = ImageDataGenerator(
        featurewise_center=False, samplewise_center=False,
        featurewise_std_normalization=False, samplewise_std_normalization=False,
        zca_whitening=False, zca_epsilon=1e-06,
        rotation_range=0, width_shift_range=0.1,
        height_shift_range=0.1, shear_range=0,
        zoom_range=0, channel_shift_range=0,
        fill_mode='nearest', cval=0.,
        horizontal_flip=True, vertical_flip=False,
        rescale=None, preprocessing_function=None,
        data_format=None, validation_split=0.0
    )
    datagen.fit(x_train)
    print(x_train.shape[0]//batch_size) # 取整
    print(x_train.shape[0]/batch_size) # 保留小数

    # 拟合模型
    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), # 按batch_size大小从x,y生成增强数据
                                  # flow_from_directory()从路径生成增强数据,和flow方法相比最大的优点在于不用
                                  epochs=epochs,
                                  steps_per_epoch=x_train.shape[0]//batch_size,
                                  validation_data=(x_test, y_test),
                                  workers=10 # 在使用基于进程的线程时，最多需要启动的进程数量。
                                  )

    model.summary()
    # 模型保存
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        model_path = os.path.join(save_dir, model_name)
        model.save(model_path)
        print('Saved trained model at %s ' % model_path)

    #######可视化#######

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
    print("hello world")