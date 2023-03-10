import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存，按需分配
# config.gpu_options.per_process_gpu_memory_fraction = 0.15
session = tf.Session(config=config)
import numpy as np
import csv
# import matplotlib.pyplot as plt
from sklearn import metrics
from keras.models import load_model
from keras.utils import to_categorical

import sys



# 说明： 性能评估函数
# 输入： predictions 预测结果，Y_test 实际标签，verbose 日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
# 输出： [sn, sp, acc, pre, f1, mcc, gmean, auroc, aupr] 验证指标结果
def perform_eval_2(predictions, Y_test, verbose=0):
    # class_label = np.uint8([round(x) for x in predictions[:, 0]]) # round()函数进行四舍五入
    # R_ = np.uint8(Y_test)
    # R = np.asarray(R_)
    class_label = np.uint8(np.argmax(predictions, axis=1))
    R = np.asarray(np.uint8([sublist[1] for sublist in Y_test]))

    CM = metrics.confusion_matrix(R, class_label, labels=None)
    CM = np.double(CM)  # CM[0][0]：TN，CM[0][1]：FP，CM[1][0]：FN，CM[1][1]：TP

    # 计算各项指标
    sn = (CM[1][1]) / (CM[1][1] + CM[1][0])  # TP/(TP+FN)
    sp = (CM[0][0]) / (CM[0][0] + CM[0][1])  # TN/(TN+FP)
    acc = (CM[1][1] + CM[0][0]) / (CM[1][1] + CM[0][0] + CM[0][1] + CM[1][0])  # (TP+TN)/(TP+TN+FP+FN)
    pre = (CM[1][1]) / (CM[1][1] + CM[0][1])  # TP/(TP+FP)
    f1 = (2 * CM[1][1]) / (2 * CM[1][1] + CM[0][1] + CM[1][0])  # 2*TP/(2*TP+FP+FN)
    mcc = (CM[1][1] * CM[0][0] - CM[0][1] * CM[1][0]) / np.sqrt((CM[1][1] + CM[0][1]) * (CM[1][1] + CM[1][0]) * (CM[0][0] + CM[0][1]) * (CM[0][0] + CM[1][0]))  # (TP*TN-FP*FN)/((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^1/2
    gmean = np.sqrt(sn * sp)
    fpr, tpr, auc_thresholds = metrics.roc_curve(y_true=R, y_score=np.asarray(predictions)[:, 1], pos_label=1)
    auroc = metrics.auc(fpr, tpr)
    precision, recall, pr_thresholds = metrics.precision_recall_curve(y_true=R, probas_pred=np.asarray(predictions)[:, 1], pos_label=1)
    aupr = metrics.auc(recall, precision)
    # auroc = metrics.roc_auc_score(y_true=R, y_score=np.asarray(predictions)[:, 1], average="macro")
    # aupr = metrics.average_precision_score(y_true=R, y_score=np.asarray(predictions)[:, 1], average="macro")

    if verbose == 1:
        print("Sn(Recall):", "{:.4f}".format(sn), "Sp:", "{:.4f}".format(sp), "Acc:", "{:.4f}".format(acc),
              "Pre(PPV):", "{:.4f}".format(pre), "F1:", "{:.4f}".format(f1), "MCC:", "{:.4f}".format(mcc),
              "G-mean:", "{:.4f}".format(gmean), "AUROC:", "{:.4f}".format(auroc), "AUPR:", "{:.4f}".format(aupr))

    return [sn, sp, acc, pre, f1, mcc, gmean, auroc, aupr]

# 说明： 实验结果保存到文件
# 输入： 文件标识符和结果
# 输出： 无
def write_res_2(filehandle, res):
    filehandle.write("Sn(Recall): %s Sp: %s Acc: %s Pre(PPV): %s F1: %s MCC: %s G-mean: %s AUROC: %s AUPR: %s\n" %
                     ("{:.4f}".format(res[0]),
                      "{:.4f}".format(res[1]),
                      "{:.4f}".format(res[2]),
                      "{:.4f}".format(res[3]),
                      "{:.4f}".format(res[4]),
                      "{:.4f}".format(res[5]),
                      "{:.4f}".format(res[6]),
                      "{:.4f}".format(res[7]),
                      "{:.4f}".format(res[8]))
                     )
    filehandle.flush()
    return


if __name__ == '__main__':

    # 超参数设置
    WINDOWS = 46

    # 打开保存结果的文件
    res_file = open("./result.txt", "w", encoding='utf-8')
    # 创建空列表，保存预测结果
    res = []

    # 提取序列片段（阳性+阴性）
    # 打开阴阳数据集文件
    f_r = open("./test_lihua.txt", "r", encoding='utf-8')
    # 正确打开文件后，读取文件内容
    Test_data = f_r.readlines()
    f_r.close()

    # 数据编码
    # 理化属性信息
    from lihua import one_hot
    # one_hot编码序列片段
    test_X_1, test_label = one_hot(Test_data, windows=WINDOWS)
    test_label = to_categorical(test_label, num_classes=2)

    # 加载模型
    model = load_model('./model/Anoph.h5')
    model.summary()

    # 任务预测
    predictions = model.predict(x=test_X_1, verbose=0) 
    result = []

    for i in range(len(Test_data)):
        result.append(predictions[i][1])
    index_value = sorted(enumerate(result), reverse=True, key=lambda x: x[1])
    for i in range(len(index_value)):
        data = Test_data[index_value[i][0]].split()
        res_file.write(data[0] + "\t" + str(index_value[i][1]) + "\t" + data[1] + "\n")
        res_file.flush()

    # 验证预测结果
    res = perform_eval_2(predictions, test_label, verbose=1)
    # 将测试集预测结果写入文件
    write_res_2(res_file, res)
    res_file.close()
    
    print("Test data predicted Successfully!!")
