# 说明： 氨基酸理化信息编码
# 输入： data
# 输出： data_X
# 来源： 2019年论文“TOXIFY: a deep learning approach to classify animal venom proteins”,总结了500个AA属性，这些属性反应了极性、二级结构、分子体积、密码子多样性和静电荷
import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec
pad_sequences = tf.contrib.keras.preprocessing.sequence.pad_sequences
def one_hot(data, windows):
    # define input string
    data = data
    length = len(data)
    # define empty array
    data_X = np.zeros((length, windows, 21))
    data_Y = []
    for i in range(length):
        x = data[i].split()
        # get label
        data_Y.append(int(x[1]))
        # define universe of possible input values
        alphabet = 'ACDEFGHIKLMNPQRSTVWY-BJOUXZ'
        # define a mapping of chars to integers
        char_to_int = dict((c, i) for i, c in enumerate(alphabet))
        # integer encode input data
        integer_encoded = [char_to_int[char] for char in x[2]]
        # one hot encode
        j = 0
        for value in integer_encoded:
            if value in [21, 22, 23, 24, 25, 26]:
                for k in range(21):
                    data_X[i][j][k] = 0.05
            else:
                data_X[i][j][value] = 1.0
            j = j + 1
    data_Y = np.array(data_Y)
    return data_X, data_Y


#主函数
if __name__ == '__main__':
    # 从文件读取序列片段（验证，阳性+阴性）
    f_r_test = open("./test_lihua.txt","r", encoding='utf-8')
    f_w = open("./test_lihua.txt","w", encoding='utf-8')

    # 预测序列片段构建
    test_data = f_r_test.readlines()
    # 关闭文件
    f_r_test.close()
    f_w.close()




