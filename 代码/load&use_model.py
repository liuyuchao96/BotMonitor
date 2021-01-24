import numpy as np
from keras.models import load_model
import os
from PIL import Image
import flow2idx as fi
import packet2idx as pi

def main():
    # 加载cnn模型
    model_dir = os.path.join(os.getcwd(), 'saved_models_cnn&gru')
    cnn_model_name = 'trained_model_cnn.h5'
    gru_model_name = 'trained_model_gru.h5'
    final_model_name = 'final_trained_model.h5'
    cnn_model_path = os.path.join(model_dir, cnn_model_name)
    gru_model_path = os.path.join(model_dir, gru_model_name)
    final_model_path = os.path.join(model_dir, final_model_name)
    cnn_model = load_model(cnn_model_path)  #选取自己的.h模型名称
    gru_model = load_model(gru_model_path)  #选取自己的.h模型名称
    final_model = load_model(final_model_path)  #选取自己的.h模型名称

    #处理数据
    data_dir = os.path.join(os.getcwd(), 'tcp_syn')
    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        x_cnn = fi.getMatrixFromPcap(file_path, fi.PNG_SIZE) # 生成流图像矩阵
        x_gru = np.array(pi.getMatrixFromEachPacket(file_path))
        x_gru = np.reshape(x_gru, (1, x_gru.shape[0] * x_gru.shape[1], x_gru.shape[-1])) #
        print(x_gru.shape)
        x_cnn = np.reshape(x_cnn, [-1, 1, 32, 32])
        x_cnn = x_cnn.astype('float32')

        x_cnn /= 255  # 归一化
        output_cnn = cnn_model.predict(x_cnn)
        output_gru = gru_model.predict(x_gru, 4)
        x = np.concatenate((output_cnn, output_gru), axis=1)
        y = np.argmax(final_model.predict(x), axis=-1)
        print("该流为第%d类" % y)

if __name__ == '__main__':
    main()
