# -*- coding: UTF-8 -*-
import dpkt
import collections  #有序字典需要的模块
from sklearn.preprocessing import OneHotEncoder
import time
import os
import binascii
import numpy

# 选取的数据包个数
SELECTED_PACKET_NUM = 8
# 选取的字节数 (B)
SELECTED_BYTES = 100

def getMatrixFromPacket(file_path):
    f = open(file = file_path, mode = 'rb')
    pcap = dpkt.pcap.Reader(f) #先按.pcap格式解析，若解析不了，则按pcapng格式解析
    # except:
    # print "it is not pcap ... format, pcapng format..."
    # pcap = dpkt.pcapng.Reader(f) #解析pcapng会失败，建议使用scapy库的rdpcap去解析
       #接下来就可以对pcap做进一步解析了，记住在使用结束后最好使用f.close()关掉打开的文件，虽然程序运行结束后，
       #系统会自己关掉，但是养成好习惯是必不可少的。当前变量pcap中是按照“间戳：单包”的格式存储着各个单包
    #将时间戳和包数据分开，一层一层解析，其中ts是时间戳，buf存放对应的包
    all_pcap_data = collections.OrderedDict() #有序字典
    all_pcap_data_hex = collections.OrderedDict() #有序字典,存十六进制形式
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
            # #解包，判断传输层协议是否是TCP，即当你只需要TCP时，可用来过滤
            # if not isinstance(ip.data, dpkt.tcp.TCP):
            #     continue
            # if not isinstance(ip.data, dpkt.udp.UDP):#解包，判断传输层协议是否是UDP
            #   continue
            #传输层负载数据，基本上分析流量的人都是分析这部分数据，即应用层负载流量
            transf_data = ip.data
            #如果应用层负载长度为0，即该包为单纯的tcp包，没有负载，则丢弃
            # if not len(transf_data.data):
            #     continue
            #将时间戳与应用层负载按字典形式有序放入字典中，方便后续分析.
            all_pcap_data[ts] = transf_data.data
            all_pcap_data_hex[ts] = binascii.hexlify(transf_data.data)
            i = i + 1
        except IOError:
            print ("解析失败")
        else:
            if (len(buf_hex) < SELECTED_BYTES * 2):
                buf_hex = buf_hex.ljust(SELECTED_BYTES * 2, b'0')
            buf_hex = buf_hex[0 : SELECTED_BYTES * 2]
            source_arry = numpy.array([int(buf_hex[i: i + 2], 16) for i in range(0, SELECTED_BYTES * 2, 2)])
            print (len(buf_hex))
            print("第{}个包: {}".format(i, binascii.hexlify(buf)))
            print("第{}个包的前100字节: {}".format(i, buf_hex))
            print (source_arry)
    f.close()
    #验证结果，打印保存的数据包的抓包以及对应的包的应用层负载长度
    test_ts = 0
    for ts, app_data in all_pcap_data.items():
        #将时间戳转换成日期
        print (time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(ts)) ,":",len(app_data))
        test_ts = ts
    #打印最后一个包的十六进制形式，因为加密数据在命令行打印会出现大量乱码和错行，故在此不做演示打印包的字符形式
    print ("\n最后一个包负载的十六进制******\n%s", all_pcap_data_hex[test_ts],"\n")
    pass
if __name__ == '__main__':
    file_name = '172.16.1.126_49157_23.218.156.26_80_1467763845.pcap'
    file_path = os.path.join('tcp_syn', file_name)
    print (file_path)
    getMatrixFromPacket(file_path)
