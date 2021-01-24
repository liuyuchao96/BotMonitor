from tensorflow.python.client import device_lib
import tensorflow as tf
if __name__ == '__main__':
    print(device_lib.list_local_devices())
    gpu_device_name = tf.test.gpu_device_name()
    print(gpu_device_name)