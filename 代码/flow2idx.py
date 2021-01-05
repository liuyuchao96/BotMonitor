import numpy
from PIL import Image
import binascii
import os
# 图片的尺寸为32 * 32
PNG_SIZE = 32
# 选取的字节数 (B)
SELECTED_BYTES = PNG_SIZE * PNG_SIZE
# pcap包文件头大小(B)
PCAP_HEADER_SIZE = 24

# 从网络流中获取用于生成图片的灰度值矩阵
def getMatrixFromPcap(fileName, columns):
    with open(fileName, 'rb') as file:
        content = file.read()
    # 将文件中的数据转成二进制并用16进制表示
    hex_content = binascii.hexlify(content)
    print (len(hex_content))
    # 考虑到pcap文件头部是文件的信息，与网络流数据无关，故跳过该部分数据
    selected_max_index = (SELECTED_BYTES + PCAP_HEADER_SIZE) * 2
    if len(hex_content) < selected_max_index:
        # 对于长度小于1024B的流填充0x00，也就是填充的最大位置为selected_max_index处，才能保证能选择到1024B的数据
        hex_content = hex_content.ljust(selected_max_index, b'0')
    hex_content = hex_content[PCAP_HEADER_SIZE * 2 : selected_max_index]
    # 由于hex_content是16进制的二进制数据，因此1B是两个16进制数据，将2个16进制数据合并成一个int类型数值，表示灰度值（0到255）
    source_array = numpy.array([int(hex_content[i : i + 2], 16) for i in range(0, SELECTED_BYTES * 2, 2)])
    rows = int(SELECTED_BYTES / columns)
    #改变原始数组形状，从1*1024 -> 到32*32
    matrix = numpy.reshape(source_array[: rows * columns], (-1, rows))
    matrix = numpy.uint8(matrix)
    return matrix

# 将矩阵转成图片
def getImage(matrix):
    image = Image.fromarray(matrix)
    return image

if __name__ == '__main__':
    # main()
    path = 'tcp_syn'
    # dir_path = 'png\train'
    for i, d in enumerate(os.listdir(path)):
        # for f in os.listdir(os.path.join(path, d)):
        fileName = os.path.join(path, d)
        matrix = getMatrixFromPcap(fileName, PNG_SIZE)
        image = getImage(matrix)
        png_path = r"png\train\{}.png".format(d[:-5])
        image.save(png_path)
