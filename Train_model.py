import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0";
import tensorflow as tf
config=tf.ConfigProto()
config.gpu_options.allow_growth = True
# #config.gpu_options.per_process_gpu_memory_fraction = 0.3
# session = tf.Session(config=config)
# session = tf.compat.v1.Session(config=config)

import numpy as np
import time
import glob
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras import backend as K
from sklearn import metrics
from keras.optimizers import Adam
from keras.layers import Input, Conv1D, Embedding, AveragePooling1D, GlobalAveragePooling1D, BatchNormalization, Dropout, Flatten, Dense, \
    Activation, Concatenate, Reshape, GlobalMaxPooling1D, Add, Permute, multiply, Lambda, Conv2D, MaxPooling1D, Bidirectional, LSTM
from keras.models import Model
from keras.engine.topology import Layer
from keras.regularizers import l2
from scipy.interpolate import interp1d
from keras.utils import to_categorical
from scipy import interp
from on_lstm import ONLSTM
onlstm = ONLSTM(128, 32, return_sequences=True, dropconnect=0.25)

build_model1_WEIGHT_FILE = os.path.join(os.getcwd(), 'D:/peptide-2/weights1', 'model1_weights.hdf5')

class Extract_outputs(Layer):
    def __init__(self, outputdim, **kwargs):
        # self.input_spec = [InputSpec(ndim='3+')]
        self.outputdim = outputdim
        super(Extract_outputs, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return tuple([None, input_shape[1], self.outputdim])

    def call(self, x, mask=None):
        x = x[:, :, :self.outputdim]
        # return K.batch_flatten(x)
        return x


# 定义密集卷积块中单个卷积层
def conv_factory(x, concat_axis, filters, dropout_rate=None, weight_decay=1e-4):
    """x = BatchNormalization(axis=concat_axis,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)"""
    x = Activation('elu')(x)
    x = Conv1D(filters=filters,
               kernel_size=3,
               kernel_initializer="he_uniform",
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x

# 定义transition层
def transition(x, concat_axis, filters, dropout_rate=None, weight_decay=1e-4):
    """x = BatchNormalization(axis=concat_axis,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)"""
    x = Activation('elu')(x)
    x = Conv1D(filters=filters,
               kernel_size=1,
               kernel_initializer="he_uniform",
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x_1 = AveragePooling1D(pool_size=2, strides=2)(x)
    x_2 = MaxPooling1D(pool_size=2, strides=2)(x)
    x = Concatenate(axis=-1)([x_1, x_2])
    return x

# 定义密集卷积块
def denseblock(x, concat_axis, layers, filters, growth_rate, dropout_rate=None, weight_decay=1e-4):
    list_feature_map = [x]
    for i in range(layers):
        x = conv_factory(x, concat_axis, growth_rate,
                         dropout_rate, weight_decay)
        list_feature_map.append(x)
        x = Concatenate(axis=concat_axis)(list_feature_map)
        filters = filters + growth_rate
    return x, filters


# 通道注意力
def channel_attention(input_feature, ratio=16):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature._keras_shape[channel_axis]

    shared_layer_one = Dense(channel // ratio,
                         kernel_initializer='he_normal',
                         activation='relu',
                         use_bias=True,
                         bias_initializer='zeros')

    shared_layer_two = Dense(channel,
                         kernel_initializer='he_normal',
                         use_bias=True,
                         bias_initializer='zeros')


    avg_pool = GlobalAveragePooling1D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel // ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel)

    max_pool = GlobalMaxPooling1D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel // ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('hard_sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
       cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])

# 空间注意力
def spatial_attention(input_feature):
    kernel_size = 7
    if K.image_data_format() == "channels_first":
        channel = input_feature._keras_shape[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature._keras_shape[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool._keras_shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool._keras_shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat._keras_shape[-1] == 2
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          activation='hard_sigmoid',
                          strides=1,
                          padding='same',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    assert cbam_feature._keras_shape[-1] == 1
    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])

# 构建卷积块注意块
def cbam_block(cbam_feature, ratio=16):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in CBAM: Convolutional Block Attention Module.
    """

    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature


# 定义focal_loss损失函数
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss0(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(1e-8 + pt_1)) - K.sum(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + 1e-8))
    return focal_loss0

# 定义categorical_focal_loss损失函数
def categorical_focal_loss(gamma=2.0, alpha=0.25):
    """
    Implementation of Focal Loss from the paper in multiclass classification
    Formula:
        loss = -alpha*((1-p)^gamma)*log(p)
    Parameters:
        alpha -- the same as wighting factor in balanced cross entropy
        gamma -- focusing parameter for modulating factor (1-p)
    Default value:
        gamma -- 2.0 as mentioned in the paper
        alpha -- 0.25 as mentioned in the paper
    """
    def focal_loss1(y_true, y_pred):
        # Define epsilon so that the backpropagation will not K_FOLD in NaN
        # for 0 divisor case
        epsilon = 1e-7
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediction value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate cross entropy
        cross_entropy = -y_true * K.log(y_pred)
        # Calculate weight that consists of  modulating factor and weighting factor
        alpha1 = tf.where(tf.equal(y_true, [1.0, 0.0]), y_true * (1.0 - alpha), y_true * alpha)
        weight = alpha1 * y_true * K.pow((K.ones_like(y_pred) - y_pred), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.sum(loss, axis=1)
        return loss
    return focal_loss1

# 定义binary_focal_loss损失函数
def binary_focal_loss(gamma=2.0, alpha=0.25):
    """
    Implementation of Focal Loss from the paper in multiclass classification
    Formula:
        loss = -alpha_t*((1-p_t)^gamma)*log(p_t)
        p_t = y_pred, if y_true = 1
        p_t = 1-y_pred, otherwise
        alpha_t = alpha, if y_true=1
        alpha_t = 1-alpha, otherwise
        cross_entropy = -log(p_t)
    Parameters:
        alpha -- the same as wighting factor in balanced cross entropy
        gamma -- focusing parameter for modulating factor (1-p)
    Default value:
        gamma -- 2.0 as mentioned in the paper
        alpha -- 0.25 as mentioned in the paper
    """
    def focal_loss2(y_true, y_pred):
        # Define epsilon so that the backpropagation will not K_FOLD in NaN
        # for 0 divisor case
        epsilon = 1e-7
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediciton value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate p_t
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        # Calculate alpha_t
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        # Calculate cross entropy
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.sum(loss, axis=1)
        return loss
    return focal_loss2

def build_model(windows=46, concat_axis=-1, denseblocks=1, layers=3, filters=96, Cov1Ds=0,
                growth_rate=32, kernel_size_1=3, kernel_size_2=9, padding="same", dropout_rate=0.8, weight_decay=1e-4):
    input1 = Input(shape=(windows, 21))
    input2 = Input(shape=(windows, 15))# Homo0.75 mus 0.85
    input = Input(shape=(windows, 36))
    ########################################################################
    '''
    # 模块一： 序列信息-密集CNN模型
    x_1 = Conv1D(filters=filters, kernel_size=kernel_size_1,
                 kernel_initializer="he_uniform",
                 padding=padding, use_bias=False,
                 kernel_regularizer=l2(weight_decay))(input1)
    x_1 = Activation('relu')(x_1)
    x_1 = BatchNormalization(axis=concat_axis,
                             gamma_regularizer=l2(weight_decay),
                             beta_regularizer=l2(weight_decay))(x_1)
    x_1 = Dropout(dropout_rate)(x_1)
    # Add Cov1Ds
    filters_1 = filters
    for i in range(Cov1Ds):
        # Add Cov1D
        filters_1 = filters_1 + filters
        x_1 = Conv1D(filters=filters_1, kernel_size=kernel_size_2,
                     kernel_initializer="he_uniform",
                     padding=padding, use_bias=False,
                     kernel_regularizer=l2(weight_decay))(x_1)
        x_1 = Activation('relu')(x_1)
        x_1 = BatchNormalization(axis=concat_axis,
                                 gamma_regularizer=l2(weight_decay),
                                 beta_regularizer=l2(weight_decay))(x_1)
        x_1 = Dropout(dropout_rate)(x_1)  # 46*96
    x_1 = BatchNormalization(axis=concat_axis,
                             gamma_regularizer=l2(weight_decay),
                             beta_regularizer=l2(weight_decay))(x_1)
    # output1 = Dropout(0.2)(output1) Mus
    x_1 = Dropout(dropout_rate)(x_1)
    '''
    ############ 密集CNN
    x_1 = Conv1D(filters=filters, kernel_size=3,
                 kernel_initializer="he_normal",
                 padding="same", use_bias=False,
                 kernel_regularizer=l2(weight_decay))(input1)
    x_1 = BatchNormalization(axis=concat_axis,
                             gamma_regularizer=l2(weight_decay),
                             beta_regularizer=l2(weight_decay))(x_1)
    x_1 = Dropout(dropout_rate)(x_1)
    # Add denseblocks
    for i in range(denseblocks - 1):
        # Add denseblock
        x_1, filters = denseblock(x_1, concat_axis=concat_axis, layers=layers,
                                    filters=filters, growth_rate=growth_rate,
                                    dropout_rate=dropout_rate, weight_decay=weight_decay)
        # Add transition
        x_1 = transition(x_1, concat_axis=concat_axis, filters=filters,
                         dropout_rate=dropout_rate, weight_decay=weight_decay)
    # The last denseblock
    # Add denseblock
    x_1, filters = denseblock(x_1, concat_axis=concat_axis, layers=layers,
                                filters=filters, growth_rate=growth_rate,
                                dropout_rate=dropout_rate, weight_decay=weight_decay)
    x_1 = Activation('elu')(x_1)
    x_1 = cbam_block(x_1)
    x_1 = BatchNormalization(axis=concat_axis,
                             gamma_regularizer=l2(weight_decay),
                             beta_regularizer=l2(weight_decay))(x_1)
    print(x_1.shape)
    # 人类 0,4416(1个)  1, 4416（2个） 2, 5520（3个） 3, 3696（4个）    4, 2160   5,1056
    # 小鼠 1,6048    2，7440
    x_1 = Reshape((1, 2, 4416))(x_1)
    ######### OnLSTM
    # x_2 = onlstm(input1)
    # x_2 = Bidirectional(LSTM(units=128, dropout=dropout_rate, return_sequences=True, kernel_regularizer=l2(weight_decay), merge_mode='concat'))(input1)
    x_2 = BatchNormalization(axis=concat_axis,
                             gamma_regularizer=l2(weight_decay),
                             beta_regularizer=l2(weight_decay))(x_2)
    # x_2 = Dropout(dropout_rate)(x_2)
    x_2 = Dropout(dropout_rate)(x_2)
    print(x_2.shape)
    # x_2 = LSTM(units=128, dropout=0.25, return_sequences=True, kernel_regularizer=l2(1e-4))(input)
    x_2 = Reshape((1, 2, 2944))(x_2)     #人类 32,736  64,1472  96,2208  128,2944   160,3680  192,4416
    # x_2 = Reshape((1, 2, 4032))(x_2)  # 小鼠值 2, 4032
    #################################################################################
    x = Concatenate(axis=-1)([x_1, x_2])
    '''
    # 采用二维CNN捕获氨基酸之间的长距离依赖关系的特征
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization(axis=concat_axis,
                             gamma_regularizer=l2(weight_decay),
                             beta_regularizer=l2(weight_decay))(x)
    x = Dropout(dropout_rate)(x)
    x = Activation('elu')(x)
    '''
    x = Flatten()(x)
    #x = Dropout(dropout_rate)(x)
    x = Dropout(dropout_rate)(x)
    ##################################################################################
    # 全连接层进行预测
    x = Dense(units=2, activation="softmax", use_bias=False,
              kernel_initializer='he_normal',
              kernel_regularizer=l2(weight_decay))(x)

    model = Model(inputs=[input1], outputs=[x], name="DenseBlock")
    optimizer = Adam(lr=2e-4, epsilon=1e-8)
    #optimizer = SGD(lr=1e-3, decay=1e-5, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    #model.compile(loss=[categorical_focal_loss(gamma=2.0, alpha=0.25)], optimizer=optimizer, metrics=['accuracy'])

    return model

# 说明： 性能评估函数
# 输入： predictions 预测结果，Y_test 实际标签，verbose 日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
# 输出： [sn, sp, acc, pre, f1, mcc, gmean, auroc, aupr] 验证指标结果
def perform_eval_1(predictions, Y_test, verbose=0):
    #class_label = np.uint8([round(x) for x in predictions[:, 0]]) # round()函数进行四舍五入
    #R_ = np.uint8(Y_test)
    #R = np.asarray(R_)
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
    auroc = metrics.roc_auc_score(y_true=R, y_score=np.asarray(predictions)[:, 1], average="macro")
    aupr = metrics.average_precision_score(y_true=R, y_score=np.asarray(predictions)[:, 1], average="macro")

    if verbose == 1:
        print("Sn(Recall):", "{:.4f}".format(sn), "Sp:", "{:.4f}".format(sp), "Acc:", "{:.4f}".format(acc),
              "Pre(PPV):", "{:.4f}".format(pre), "F1:", "{:.4f}".format(f1), "MCC:", "{:.4f}".format(mcc),
              "G-mean:", "{:.4f}".format(gmean), "AUROC:", "{:.4f}".format(auroc), "AUPR:", "{:.4f}".format(aupr))

    return [sn, sp, acc, pre, f1, mcc, gmean, auroc, aupr]


# 说明： 实验结果保存到文件
# 输入： 文件标识符和结果
# 输出： 无
def write_res_1(filehandle, res, fold=0):
    filehandle.write("Fold: " + str(fold) + " ")
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

# 说明： loss-epoch，acc-epoch曲线作图函数
def figure(history, K_FOLD, fold, windows):
    """
    iters = np.arange(len(history_dict["loss"]))
    plt.figure()
    # acc
    plt.plot(iters, history_dict["acc"], color='r', label='train acc')
    # loss
    plt.plot(iters, history_dict["loss"], color='g', label='train loss')
    # val_acc
    plt.plot(iters, history_dict["val_acc"], color='b', label='val acc')
    # val_loss
    plt.plot(iters, history_dict["val_loss"], color='k', label='val loss')
    plt.grid(True) # 设置网格线
    plt.xlabel('epochs')
    plt.ylabel('loss-acc')
    plt.legend(loc="upper right") #设置图例位置
    plt.savefig("./%d折交叉第%d折.png" % (K_FOLD, fold))  # 保存图片
    """
    def show_train_history(train_history, train_metrics, validation_metrics):
        plt.plot(train_history.history[train_metrics])
        plt.plot(train_history.history[validation_metrics])
        plt.title('Train History')
        plt.grid(True)  # 设置网格线
        plt.ylabel(train_metrics)
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left')  # 设置图例位置

    # 画图显示训练过程
    def plt_fig(history, K_FOLD, fold, windows):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        show_train_history(history, 'acc', 'val_acc')
        plt.subplot(1, 2, 2)
        show_train_history(history, 'loss', 'val_loss')
        plt.savefig("D:/peptide-2/results/picture/%d折交叉第%d折-%d.png" % (K_FOLD, fold, windows), dpi=350)  # 保存图片
        plt.close()

    return plt_fig(history, K_FOLD, fold, windows)


if __name__ == '__main__':

    # 超参数设置
    BATCH_SIZE = 256
    K_FOLD = 1
    N_EPOCH = 500
    WINDOWS = 46

    # 打开保存结果的文件
    res_file = open("D:/peptide-2/results/Anoph.txt", "w", encoding='utf-8')
    #res_file = open("D:/peptide-2/results/Anoph.txt", "w", encoding='utf-8')
    # 创建空列表，保存每折的结果
    res = []
    # 交叉验证开始
    tprs = []
    pres = []
    aurocs = []
    auprs = []
    mean_fpr = np.linspace(0, 1, 200)
    mean_rec = np.linspace(0, 1, 200)
    plt.figure(figsize=(12, 5))
    # 分层交叉验证
    for fold in range(K_FOLD):
    #for fold in [3, 4]:

        # 从文件读取序列片段（训练+验证，阳性+阴性）
        # f_r_train = open("D:/BERT-peptide/10-fold/46-5-fold/Homo/Homo_train-%d.txt" %(fold), "r", encoding='utf-8')
        # f_r_test = open("D:/BERT-peptide/10-fold/46-5-fold/Homo/Homo_test-%d.txt" % (fold), "r", encoding='utf-8')
        # 小鼠独立测试集
        # f_r_train = open("D:/BERT-peptide/train_and_test/Mus63-100/train_lihua.txt", "r", encoding='utf-8')
        # f_r_test = open("D:/BERT-peptide/train_and_test/Mus63-100/test_lihua.txt", "r", encoding='utf-8')
        #f_r_train = open("D:/BERT-peptide/10-fold/46-5-fold/Mus_train-%d.txt" %(fold), "r", encoding='utf-8')
        #f_r_test = open("D:/BERT-peptide/10-fold/46-5-fold/Mus_test-%d.txt" % (fold), "r", encoding='utf-8')
        #f_r_train = open("C:/Users/WangJuan/Desktop/实验结果/实验3：物种实验/Anoph/data-5-fold/Anoph_train-%d.txt" %(fold), "r", encoding='utf-8')
        #f_r_test = open("C:/Users/WangJuan/Desktop/实验结果/实验3：物种实验/Anoph/data-5-fold/Anoph_test-%d.txt" % (fold), "r", encoding='utf-8')
        f_r_train = open("C:/Users/WangJuan/Desktop/实验结果/实验3：物种实验/Anoph/data/train_lihua.txt", "r", encoding='utf-8')
        f_r_test = open("C:/Users/WangJuan/Desktop/实验结果/实验3：物种实验/Anoph/data/test_lihua.txt", "r", encoding='utf-8')

        # 训练序列片段构建
        train_data = f_r_train.readlines()

        # 预测序列片段构建
        test_data = f_r_test.readlines()

        # 关闭文件
        f_r_train.close()
        f_r_test.close()

        # 数据编码
        # 数据编码
        from lihua import one_hot, Phy_Chem_Inf_4

        # one_hot编码序列片段
        train_X_1, train_Y = one_hot(train_data, windows=WINDOWS)
        train_Y = to_categorical(train_Y, num_classes=2)
        test_X_1, test_Y = one_hot(test_data, windows=WINDOWS)
        test_Y = to_categorical(test_Y, num_classes=2)
        # 理化属性信息
        train_X_2 = Phy_Chem_Inf_4(train_data, windows=WINDOWS)
        test_X_2 = Phy_Chem_Inf_4(test_data, windows=WINDOWS)
        # 数组拼接
        train_X = np.concatenate((train_X_1, train_X_2), axis=2)
        test_X = np.concatenate((test_X_1, test_X_2), axis=2)

        # 引入模型
        model = build_model(windows=WINDOWS)
        # 打印模型
        model.summary()

        # 训练模型
        print("fold:", (fold))

        # 早停
        call = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto', patience=20, restore_best_weights=True)
        history = model.fit(x=[train_X_1], y=train_Y, batch_size=BATCH_SIZE, epochs=N_EPOCH, shuffle=True,
                            callbacks=[call], verbose=2, validation_data=([test_X_1], test_Y))
        # 无早停
        #history = model.fit(x=[train_X], y=train_Y, batch_size=BATCH_SIZE, epochs=N_EPOCH, shuffle=True, class_weight={0: 1.0, 1: 9.0}, verbose=1, validation_data=([test_X], test_Y))

        # 得到预测结果
        predictions = model.predict(x=[test_X_1], verbose=0)

        # 验证预测结果
        res = perform_eval_1(predictions, test_Y, verbose=1)

        # 将结果写入文件
        write_res_1(res_file, res, fold)

        # 画loss-epoch，acc-epoch曲线
        figure(history, K_FOLD, fold, WINDOWS)

        # 画ROC曲线
        R = np.asarray(np.uint8([sublist[1] for sublist in test_Y]))
        plt.subplot(1, 2, 1)
        fpr, tpr, auc_thresholds = metrics.roc_curve(y_true=R, y_score=np.asarray(predictions)[:, 1], pos_label=1)
        # 计算AUROC，并保存
        auroc_score = metrics.auc(fpr, tpr)
        aurocs.append(auroc_score)
        # interp1d：1维插值，并把结果添加到tprs列表中
        f1 = interp1d(fpr, tpr, kind='linear')
        interp1d_tpr = f1(mean_fpr)
        interp1d_tpr[0] = 0.0
        interp1d_tpr[-1] = 1.0
        tprs.append(interp1d_tpr)
        # 画图，只需要plt.plot(fpr, tpr)，变量auc_score只是记录auc的值，通过auc()函数计算
        plt.plot(fpr, tpr, lw=1, alpha=0.6, label='ROC fold %d (%0.4f)' % (fold, auroc_score))

        # 画PR曲线
        plt.subplot(1, 2, 2)
        precision, recall, pr_thresholds = metrics.precision_recall_curve(y_true=R,
                                                                          probas_pred=np.asarray(predictions)[:, 1],
                                                                          pos_label=1)
        # 计算AUPR，并保存
        aupr_score = metrics.auc(recall, precision)
        auprs.append(aupr_score)
        # interp1d：1维插值，并把结果添加到pres列表中
        f2 = interp1d(recall, precision, kind='linear')
        interp1d_pre = f2(mean_rec)
        interp1d_pre[0] = 1.0
        interp1d_pre[-1] = 0.0
        pres.append(interp1d_pre)
        # 画图，只需要plt.plot(recall, precision)，变量aupr_score只是记录aupr的值，通过auc()函数计算
        plt.plot(recall, precision, lw=1, alpha=0.6, label='PR fold %d (%0.4f)' % (fold, aupr_score))

        # 保存训练好的模型(既保存了模型图结构，又保存了模型参数)
        #model.save('D:/peptide-2/results/Anoph-%d_fold%d.h5' % (WINDOWS, fold))
        model.save('C:/Users/WangJuan/Desktop/实验结果/实验3：物种实验/result/model/Anoph.h5')
    '''
    # 画第一个子图
    plt.subplot(1, 2, 1)
    # 画对角线
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=1.0)
    mean_tpr = np.mean(tprs, axis=0)
    mean_auroc = np.mean(aurocs)
    std_auroc = np.std(aurocs)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.4f $\pm$ %0.4f)' % (mean_auroc, std_auroc), lw=2,
             alpha=1.0)
    # plt.set(xlim=[-0.05, 1.05], xlabel='False Positive Rate', ylim=[-0.05, 1.05], ylabel='True Positive Rate', title="Receiver operating characteristic curves")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xticks(fontproperties='Times New Roman', size=12)
    plt.yticks(fontproperties='Times New Roman', size=12)
    plt.xlabel('FPR', fontdict={'family': 'Times New Roman', 'size': 12})
    plt.ylabel('TPR', fontdict={'family': 'Times New Roman', 'size': 12})
    plt.title('ROC curves', fontdict={'family': 'Times New Roman', 'size': 14})
    plt.legend(loc="lower right", prop={'family': 'Times New Roman', 'size': 10})

    # 画第二个子图
    plt.subplot(1, 2, 2)
    mean_pre = np.mean(pres, axis=0)
    mean_aupr = np.mean(auprs)
    std_aupr = np.std(auprs)
    plt.plot(mean_rec, mean_pre, color='b', label=r'Mean PR (AUPR = %0.4f $\pm$ %0.4f)' % (mean_aupr, std_aupr), lw=2,
             alpha=1.0)
    # plt.set(xlim=[-0.05, 1.05], xlabel='Recall', ylim=[-0.05, 1.05], ylabel='Precision', title="Precision-Recall curves")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xticks(fontproperties='Times New Roman', size=12)
    plt.yticks(fontproperties='Times New Roman', size=12)
    plt.xlabel('Recall', fontdict={'family': 'Times New Roman', 'size': 12})
    plt.ylabel('Precision', fontdict={'family': 'Times New Roman', 'size': 12})
    plt.title('PR curves', fontdict={'family': 'Times New Roman', 'size': 14})
    plt.legend(loc="lower right", prop={'family': 'Times New Roman', 'size': 10})
    plt.savefig("D:/peptide-2/results/Anoph-%d折交叉-%d.png" % (K_FOLD, WINDOWS), dpi=350)
    plt.close()

    # 关闭文件
    res_file.close()
    ##########################################################################################
    time.sleep(1)
    # 计算10折交叉验证结果每项指标的均值
    sn = []
    sp = []
    acc = []
    pre = []
    f1 = []
    mcc = []
    gmean = []
    auroc = []
    aupr = []

    f_r = open("D:/peptide-2/results/Anoph.txt", "r", encoding='utf-8')
    lines = f_r.readlines()

    for line in lines:
        x = line.split()
        sn.append(float(x[3]))
        sp.append(float(x[5]))
        acc.append(float(x[7]))
        pre.append(float(x[9]))
        f1.append(float(x[11]))
        mcc.append(float(x[13]))
        gmean.append(float(x[15]))
        auroc.append(float(x[17]))
        aupr.append(float(x[19]))

    mean_sn = np.mean(sn)
    mean_sp = np.mean(sp)
    mean_acc = np.mean(acc)
    mean_pre = np.mean(pre)
    mean_f1 = np.mean(f1)
    mean_mcc = np.mean(mcc)
    mean_gmean = np.mean(gmean)
    mean_auroc = np.mean(auroc)
    mean_aupr = np.mean(aupr)

    std_sn = np.std(sn)
    std_sp = np.std(sp)
    std_acc = np.std(acc)
    std_pre = np.std(pre)
    std_f1 = np.std(f1)
    std_mcc = np.std(mcc)
    std_gmean = np.std(gmean)
    std_auroc = np.std(auroc)
    std_aupr = np.std(aupr)

    print("mean_sn:", "{:.4f}".format(mean_sn), "mean_sp:", "{:.4f}".format(mean_sp), "mean_acc:",
          "{:.4f}".format(mean_acc),
          "mean_pre:", "{:.4f}".format(mean_pre), "mean_f1:", "{:.4f}".format(mean_f1), "mean_mcc:",
          "{:.4f}".format(mean_mcc),
          "mean_gmean:", "{:.4f}".format(mean_gmean), "mean_auroc:", "{:.4f}".format(mean_auroc), "mean_aupr:",
          "{:.4f}".format(mean_aupr))
    print("std_sn:", "{:.4f}".format(std_sn), "std_sp:", "{:.4f}".format(std_sp), "std_acc:", "{:.4f}".format(std_acc),
          "std_pre:", "{:.4f}".format(std_pre), "std_f1:", "{:.4f}".format(std_f1), "std_mcc:",
          "{:.4f}".format(std_mcc),
          "std_gmean:", "{:.4f}".format(std_gmean), "std_auroc:", "{:.4f}".format(std_auroc), "std_aupr:",
          "{:.4f}".format(std_aupr))

    f_w = open("D:/peptide-2/results/Anoph-test_elu-%d.txt" % (WINDOWS), "w", encoding='utf-8')
    f_w.write(
        "mean_sn: %s mean_sp: %s mean_acc: %s mean_pre: %s mean_f1: %s mean_mcc: %s mean_gmean: %s mean_auroc: %s mean_aupr: %s\n" %
        ("{:.4f}".format(mean_sn),
         "{:.4f}".format(mean_sp),
         "{:.4f}".format(mean_acc),
         "{:.4f}".format(mean_pre),
         "{:.4f}".format(mean_f1),
         "{:.4f}".format(mean_mcc),
         "{:.4f}".format(mean_gmean),
         "{:.4f}".format(mean_auroc),
         "{:.4f}".format(mean_aupr))
    )
    f_w.write(
        "std_sn: %s std_sp: %s std_acc: %s std_pre: %s std_f1: %s std_mcc: %s std_gmean: %s std_auroc: %s std_aupr: %s\n" %
        ("{:.4f}".format(std_sn),
         "{:.4f}".format(std_sp),
         "{:.4f}".format(std_acc),
         "{:.4f}".format(std_pre),
         "{:.4f}".format(std_f1),
         "{:.4f}".format(std_mcc),
         "{:.4f}".format(std_gmean),
         "{:.4f}".format(std_auroc),
         "{:.4f}".format(std_aupr))
    )
    f_w.flush()

    # 关闭文件
    f_w.close()
    f_r.close()
    '''
