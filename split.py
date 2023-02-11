# -*- coding: utf-8 -*-
"""
创建于2021年10月29日
划分训练集、测试集
@author:Juan Wang
"""

import re
import random
import time

if __name__ == '__main__':
    # 分割训练集（80%）和测试集（20%）
    try:
        # 打开文件
        f_r = open('C:/Users/WangJuan/Desktop/实验结果/实验3：物种实验/Candida_albicans/data/Candi_neg.txt', 'r', encoding='utf-8')
        f_w1 = open('C:/Users/WangJuan/Desktop/实验结果/实验3：物种实验/Candida_albicans/data/Candi_train_neg.txt', 'w', encoding='utf-8')
        f_w2 = open('C:/Users/WangJuan/Desktop/实验结果/实验3：物种实验/Candida_albicans/data/Candi_test_neg.txt', 'w', encoding='utf-8')
    except IOError as e:
        print('open file error!', e)
        #exit()
    else:
        # 正确打开文件后，读取文件内容
        lines = f_r.readlines()
        # 创建空列表保存肽id，创建空字典保存肽序列
        idlist = []         #保存id信息
        sequence_dict = {}  #保存序列信息
        protein_id = ""
        i = 1
        # 循环读取行内容，保存肽id和序列
        for line in lines:
            # print(line[:6])
            if line[:7] == '>Label:':
                idlist.append(line[7:].strip().split('\n'))
                protein_id = line[7:].strip().split('\n')
                i = i + 1
            else:
                sequence_dict[protein_id[0]] = line

        Train_idlist = random.sample(idlist, int(i * 0.8))
        # 保存训练集和测试集到文件
        for Train_idlist_ in Train_idlist:
            f_w1.write(">Label:%s\n" % Train_idlist_)
            f_w1.write(sequence_dict[Train_idlist_[0]])
        for Train_idlist_ in Train_idlist:
            idlist.remove(Train_idlist_)
        for Test_idlist_ in idlist:
            f_w2.write(">Label:%s\n" % Test_idlist_)
            f_w2.write(sequence_dict[Test_idlist_[0]])
        f_r.close()
        f_w1.close()
        f_w2.close()
        print("We have %d train data and %d test data." % (len(Train_idlist), len(idlist)))
    print("Dataset segmentation successfully!!")
