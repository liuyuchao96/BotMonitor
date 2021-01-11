# -*- coding: UTF-8 -*-
import dpkt
from sklearn.preprocessing import OneHotEncoder
import os
import binascii
import numpy

# 选取的数据包个数
SELECTED_PACKET_NUM = 4
# 选取的字节数 (B)
SELECTED_BYTES = 100
# 从每个网络流中的前p个数据包中，提取前q字节的数据，组成p * q维矩阵
train = []
def getMatrixFromPacket(file_path):
    f = open(file = file_path, mode = 'rb')
    pcap = dpkt.pcap.Reader(f) #先按.pcap格式解析，若解析不了，则按pcapng格式解析
    i = 0
    for ts, buf in pcap:
        if i == SELECTED_PACKET_NUM:
            break
        buf_hex = binascii.hexlify(buf)
        try:
            eth = dpkt.ethernet.Ethernet(buf) #解包，物理层
            if not isinstance(eth.data, dpkt.ip.IP): #解包，网络层，判断网络层是否存在，
                continue
            ip = eth.data
            # 如果仅仅只是ip层数据包就丢弃
            if not len(ip.data):
                continue
        except IOError:
            print ("解析失败")
        else:
            if (len(buf_hex) < SELECTED_BYTES * 2):
                buf_hex = buf_hex.ljust(SELECTED_BYTES * 2, b'0')
            buf_hex = buf_hex[0 : SELECTED_BYTES * 2]
            source_array = numpy.array([int(buf_hex[i: i + 2], 16) for i in range(0, SELECTED_BYTES * 2, 2)])
            print (len(buf_hex))
            print("第{}个包: {}".format(i, binascii.hexlify(buf)))
            print("第{}个包的前100字节形成的数组: {}".format(i, source_array))
            encode = getOneHotEncode(source_array)
            train.append(encode)
            # print("第{}个包的onehot编码: {}".format(i, encode))
            i = i + 1
    f.close()
    # 如果网络流个数少于SELECTED_PACKET_NUM个，则补0x00
    # print(i)
    while i < SELECTED_PACKET_NUM:
        source_array = numpy.zeros((100,), dtype= numpy.int)
        print("第{}个包的前100字节形成的数组: {}".format(i, source_array))
        encode = getOneHotEncode(source_array)
        train.append(encode)
        # print("第{}个包的onehot编码: {}".format(i, encode))
        i = i + 1
    return source_array

#对从每个网络流中提取的p * q的矩阵，进行one-hot编码，转成q * 256维pcaket向量
def getOneHotEncode(values):
    # print (len(values))
    # 这里categories代表每维特征编码的位数，这里编码256位，因为1B能表示（0-255）的数值。
    onehot_encoder = OneHotEncoder(sparse=False, categories=[range(256)])
    # 行表示记录，列表示特征。之前是1 * 100的矩阵，先转成100 * 1，表示只有一维特征要进行编码，有100条记录待编码
    integer_encoded = values.reshape(len(values), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # print(len(onehot_encoded[0]))
    return onehot_encoded

if __name__ == '__main__':
    path = 'tcp_syn'
    for i, file_name in enumerate(os.listdir(path)):
        file_path = os.path.join(path, file_name)
        source_array = getMatrixFromPacket(file_path)
    print(train[0])
