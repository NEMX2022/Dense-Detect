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


def Phy_Chem_Inf_1(data, windows):#本文旨在构建的氨基酸理化性质库
    #疏水性、亲水性、侧链质量、侧链体积、平均极性、静电荷、二级结构
    letterDict = {}
    letterDict["A"] = [0.6200, -0.5, 15, 27.5, -0.06, -0.146, -1.302]
    letterDict["C"] = [0.2900, -1.0000, 47.0000,  44.6, 1.36, -0.255, 0.465]
    letterDict["D"] = [-0.9000, 3.0000, 59.0000,  40, -0.8, -3.242, 0.302]
    letterDict["E"] = [-0.7400, 3.0000, 73.0000,  62, -0.77, -0.837, -1.453]
    letterDict["F"] = [1.1900, -2.5000, 91.0000, 115.5, 1.27, 0.412, -0.590]
    letterDict["G"] = [0.4800, 0, 1.0000, 0, -0.41, 2.064, 1.652]
    letterDict["H"] = [-0.4000, -0.5000, 82.0000,  79, 0.49, -0.078, -0.417]
    letterDict["I"] = [1.3800, -1.8000, 57.0000, 93.5, 1.31, 0.816, -0.547]
    letterDict["K"] = [-1.5000, 3.0000, 73.0000, 100, -1.18, 1.648, -0.561]
    letterDict["L"] = [1.0600, -1.8000, 57.0000, 93.5, 1.21, -0.912, -0.987]
    letterDict["M"] = [0.6400, -1.3000, 75.0000, 94.1, 1.27, 1.212, -1.524]
    letterDict["N"] = [-0.7800, 0.2000, 58.0000, 58.7, -0.48, 0.933, 0.828]
    letterDict["P"] = [0.1200, 0, 42.0000, 41.9, 0, -1.392, 2.081]
    letterDict["Q"] = [-0.8500, 0.2000, 72.0000, 80.7, -0.73, -1.853, -0.179]
    letterDict["R"] = [-2.5300, 3.0000, 101.0000, 105, -0.84, 2.897, -0.055]
    letterDict["S"] = [-0.1800, 0.3000, 31.0000, 29.3, -0.5, -2.647, 1.399]
    letterDict["T"] = [-0.0500, -0.4000, 45.0000, 51.3, -0.27, 1.313, 0.326]
    letterDict["V"] = [1.0800, -1.5000, 43.0000, 71.5, 1.09, -1.262, -0.279]
    letterDict["W"] = [0.8100, -3.4000, 130.0000, 145.5, 0.88, -0.184, 0.009]
    letterDict["Y"] = [0.2600, -2.3000, 107.0000, 117.3, 0.33, 1.512, 0.830]
    letterDict["-"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    letterDict["B"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    letterDict["J"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    letterDict["O"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    letterDict["U"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    letterDict["X"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    letterDict["Z"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # define input string
    data = data
    length = len(data)
    # define empty array
    data_X = np.zeros((length, windows, 7))
    for i in range(length):
        x = data[i].split()
        # 编码氨基酸理化属性
        j = 0
        for AA in x[2]:
            for index, value in enumerate(letterDict[AA]):
                data_X[i][j][index] = value
            j = j + 1

    return data_X

def Phy_Chem_Inf_2(data, windows=41):#本文旨在构建的氨基酸理化性质库
    #疏水性、亲水性、平均极性、静电荷、二级结构
    letterDict = {}
    letterDict["A"] = [0.6200, -0.5,  -0.06, -0.146, -1.302]
    letterDict["C"] = [0.2900, -1.0000,  1.36, -0.255, 0.465]
    letterDict["D"] = [-0.9000, 3.0000,  -0.8, -3.242, 0.302]
    letterDict["E"] = [-0.7400, 3.0000,  -0.77, -0.837, -1.453]
    letterDict["F"] = [1.1900, -2.5000,  1.27, 0.412, -0.590]
    letterDict["G"] = [0.4800, 0, -0.41, 2.064, 1.652]
    letterDict["H"] = [-0.4000, -0.5000,  0.49, -0.078, -0.417]
    letterDict["I"] = [1.3800, -1.8000,  1.31, 0.816, -0.547]
    letterDict["K"] = [-1.5000, 3.0000,  -1.18, 1.648, -0.561]
    letterDict["L"] = [1.0600, -1.8000,  1.21, -0.912, -0.987]
    letterDict["M"] = [0.6400, -1.3000,  1.27, 1.212, -1.524]
    letterDict["N"] = [-0.7800, 0.2000, -0.48, 0.933, 0.828]
    letterDict["P"] = [0.1200, 0,  0, -1.392, 2.081]
    letterDict["Q"] = [-0.8500, 0.2000,  -0.73, -1.853, -0.179]
    letterDict["R"] = [-2.5300, 3.0000,  -0.84, 2.897, -0.055]
    letterDict["S"] = [-0.1800, 0.3000,  -0.5, -2.647, 1.399]
    letterDict["T"] = [-0.0500, -0.4000,  -0.27, 1.313, 0.326]
    letterDict["V"] = [1.0800, -1.5000, 1.09, -1.262, -0.279]
    letterDict["W"] = [0.8100, -3.4000,  0.88, -0.184, 0.009]
    letterDict["Y"] = [0.2600, -2.3000, 0.33, 1.512, 0.830]
    letterDict["-"] = [0.0, 0.0, 0.0, 0.0, 0.0]
    letterDict["B"] = [0.0, 0.0, 0.0, 0.0, 0.0]
    letterDict["J"] = [0.0, 0.0, 0.0, 0.0, 0.0]
    letterDict["O"] = [0.0, 0.0, 0.0, 0.0, 0.0]
    letterDict["U"] = [0.0, 0.0, 0.0, 0.0, 0.0]
    letterDict["X"] = [0.0, 0.0, 0.0, 0.0, 0.0]
    letterDict["Z"] = [0.0, 0.0, 0.0, 0.0, 0.0]

    # define input string
    data = data
    length = len(data)
    # define empty array
    data_X = np.zeros((length, windows, 5))
    for i in range(length):
        x = data[i].split()
        # 编码氨基酸理化属性
        j = 0
        for AA in x[2]:
            for index, value in enumerate(letterDict[AA]):
                data_X[i][j][index] = value
            j = j + 1
    return data_X

def Phy_Chem_Inf_3(data, windows=41):
    # 疏水性、亲水性、侧链质量、pK1(α - COOH)、pK2(-NH3)、PI、埋藏残基的平均体积、分子量、侧链体积、平均极性
    letterDict = {}
    letterDict["A"] = [0.62, -0.5, 15, 2.35, 9.87, 6.11, 91.5, 89.09, 27.5, -0.06]
    letterDict["C"] = [0.2900, -1.0000, 47.0000, 1.7100, 10.7800, 5.0200, 117.7, 121.15, 44.6, 1.36]
    letterDict["D"] = [-0.9000, 3.0000, 59.0000, 1.8800, 9.6000, 2.9800, 124.5, 133.1, 40, -0.8]
    letterDict["E"] = [-0.7400, 3.0000, 73.0000, 2.1900, 9.6700, 3.0800, 155.1, 147.13, 62, -0.77]
    letterDict["F"] = [1.1900, -2.5000, 91.0000, 2.5800, 9.2400, 5.9100, 203.4, 165.19, 115.5, 1.27]
    letterDict["G"] = [0.4800, 0, 1.0000, 2.3400, 9.6000, 6.0600, 66.4, 75.07, 0, -0.41]
    letterDict["H"] = [-0.4000, -0.5000, 82.0000, 1.7800, 8.9700, 7.6400, 167.3, 155.16, 79, 0.49]
    letterDict["I"] = [1.3800, -1.8000, 57.0000, 2.3200, 9.7600, 6.0400, 168.8, 131.17, 93.5, 1.31]
    letterDict["K"] = [-1.5000, 3.0000, 73.0000, 2.2000, 8.9000, 9.4700, 171.3, 146.19, 100, -1.18]
    letterDict["L"] = [1.0600, -1.8000, 57.0000, 2.3600, 9.6000, 6.0400, 167.9, 131.17, 93.5, 1.21]
    letterDict["M"] = [0.6400, -1.3000, 75.0000, 2.2800, 9.2100, 5.7400, 170.8, 149.21, 94.1, 1.27]
    letterDict["N"] = [-0.7800, 0.2000, 58.0000, 2.1800, 9.0900, 10.7600, 135.2, 132.12, 58.7, -0.48]
    letterDict["P"] = [0.1200, 0, 42.0000, 1.9900, 10.6000, 6.3000, 129.3, 115.13, 41.9, 0]
    letterDict["Q"] = [-0.8500, 0.2000, 72.0000, 2.1700, 9.1300, 5.6500, 161.1, 146.15, 80.7, -0.73]
    letterDict["R"] = [-2.5300, 3.0000, 101.0000, 2.1800, 9.0900, 10.7600, 202, 174.2, 105, -0.84]
    letterDict["S"] = [-0.1800, 0.3000, 31.0000, 2.2100, 9.1500, 5.6800, 99.1, 105.09, 29.3, -0.5]
    letterDict["T"] = [-0.0500, -0.4000, 45.0000, 2.1500, 9.1200, 5.6000, 122.1, 119.12, 51.3, -0.27]
    letterDict["V"] = [1.0800, -1.5000, 43.0000, 2.2900, 9.7400, 6.0200, 141.7, 117.15, 71.5, 1.09]
    letterDict["W"] = [0.8100, -3.4000, 130.0000, 2.3800, 9.3900, 5.8800, 237.6, 204.24, 145.5, 0.88]
    letterDict["Y"] = [0.2600, -2.3000, 107.0000, 2.2000, 9.1100, 5.6300, 203.6, 181.19, 117.3, 0.33]
    letterDict["-"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    letterDict["B"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    letterDict["J"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    letterDict["O"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    letterDict["U"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    letterDict["X"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    letterDict["Z"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # define input string
    data = data
    length = len(data)
    # define empty array
    data_X = np.zeros((length, windows, 10))
    for i in range(length):
        x = data[i].split()
        # 编码氨基酸理化属性
        j = 0
        for AA in x[2]:
            for index, value in enumerate(letterDict[AA]):
                data_X[i][j][index] = value
            j = j + 1

    return data_X

def Phy_Chem_Inf_4(data, windows=46):#选出来的旨在构建的氨基酸理化性质库 共15种理化信息
    letterDict = {}  # 10种结构，1种电荷，2种能量，1种疏水性，1种其他性质
    letterDict["A"] = [4.349,1.2,1.34,1.08,0.687,0.34,0.99,1.2,0.946,0.328,-2.34,-0.729,18.56,0,5.04]
    letterDict["C"] = [4.686,1,1.07,1.22,0.263,-0.18,2.32,0.8,0.481,0,5.03,-0.408,17.84,0,2.2]
    letterDict["D"] = [4.765,0.7,3.32,0.86,0.632,0.06,1.18,0.8,1.311,3.379,-0.48,-0.545,17.94,0,5.26]
    letterDict["E"] = [4.295,0.7,2.2,1.09,0.669,0.2,1.36,2.2,0.698,0,1.3,-0.532,17.97,0,6.07]
    letterDict["F"] = [4.663,1,0.8,0.96,0.577,0.15,1.25,0.5,0.963,1.336,2.57,-0.454,17.95,0,3.72]
    letterDict["G"] = [3.972,0.8,2.07,0.85,0.67,-0.88,1.4,0.3,0.36,0.5,-1.06,-0.86,18.57,0,7.09]
    letterDict["H"] = [4.63,1.2,1.27,1.02,0.594,-0.09,1.06,0.7,2.168,1.204,-3,-0.519,18.64,1,2.99]
    letterDict["I"] = [4.224,0.8,0.66,0.98,0.564,-0.03,0.81,0.9,1.283,2.078,7.26,-0.361,19.21,0,4.32]
    letterDict["K"] = [4.358,1.7,0.61,1.01,0.407,-0.11,0.91,0.6,1.203,0.835,1.56,-0.508,18.36,1,6.31]
    letterDict["L"] = [4.385,1,0.54,1.04,0.541,0.2,1.26,0.9,1.192,0.414,1.09,-0.462,19.01,0,9.88]
    letterDict["M"] = [4.513,1,0.7,1.11,0.328,0.43,1,0.3,0,0.982,0.62,-0.518,18.49,0,1.85]
    letterDict["N"] = [4.755,1.2,2.49,1.05,0.489,-0.33,1.15,0.7,0.432,1.498,2.81,-0.597,18.24,0,5.94]
    letterDict["P"] = [4.471,1,2.12,0.91,0.6,-0.81,0,2.6,2.093,0.415,-0.15,0,18.77,0,6.22]
    letterDict["Q"] = [4.373,1,1.49,0.95,0.527,0.01,1.52,0.7,1.615,0,0.16,-0.492,18.51,0,4.5]
    letterDict["R"] = [4.396,1.7,0.95,0.93,0.59,0.22,1.19,0.7,1.128,2.088,1.6,-0.535,0,1,3.73]
    letterDict["S"] = [4.498,1.5,0.94,0.95,0.692,-0.35,1.5,0.7,0.523,1.089,1.93,-0.278,18.06,0,8.05]
    letterDict["T"] = [4.346,1,1.09,1.15,0.713,-0.37,1.18,0.8,1.961,1.732,0.19,-0.367,17.71,0,5.2]
    letterDict["V"] = [4.184,0.8,1.32,1.03,0.529,0.13,1.01,1.1,0.409,0.946,2.06,-0.323,18.98,0,6.19]
    letterDict["W"] = [4.702,1,-4.65,1.17,0.632,0.07,1.33,2.1,1.925,1.781,3.59,-0.455,16.87,0,2.1]
    letterDict["Y"] = [4.604,1,-0.17,0.8,0.495,-0.31,1.09,1.8,0.802,0,-2.58,-0.439,18.23,0,3.32]
    letterDict["-"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    letterDict["B"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    letterDict["J"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    letterDict["O"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    letterDict["U"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    letterDict["X"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    letterDict["Z"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # define input string
    data = data
    length = len(data)
    # define empty array
    data_X = np.zeros((length, windows, 15))
    for i in range(length):
        x = data[i].split()
        # 编码氨基酸理化属性
        j = 0
        for AA in x[2]:
            for index, value in enumerate(letterDict[AA]):
                data_X[i][j][index] = value
            j = j + 1

    return data_X

#主函数
if __name__ == '__main__':
    # 从文件读取序列片段（验证，阳性+阴性）
    f_r_test = open("D:/BERT-peptide/train_and_test/Homo-tiny/test_lihua.txt","r", encoding='utf-8')
   # f_r_test = open("D:/BERT-peptide/Homo-sapines/24/train_lihua24.txt", "r", encoding='utf-8')
    f_w = open("D:/test_lihua.txt","w", encoding='utf-8')

    # 预测序列片段构建
    test_data = f_r_test.readlines()
    # 关闭文件
    f_r_test.close()
    # 理化属性信息
    test_X_2 = Phy_Chem_Inf_2(test_data, windows=41)
    #test_X_3 = word2vec1(test_data,46)
    f_w.writelines(str(test_X_2))
    #f_w.writelines(str(test_X_3))
    f_w.close()



