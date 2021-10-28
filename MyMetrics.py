from collections import Counter

import numpy as np


def gmean(y_true, y_pred):
    n_class_i = {}
    tr_class_i = {}
    for y_true1, y_pred1 in zip(y_true, y_pred):
        # 统计各类样本数
        if y_true1 in n_class_i:
            n_class_i[y_true1] += 1
        else:
            n_class_i[y_true1] = 1

        # 预测正确
        if y_true1 == y_pred1:
            if y_pred1 in tr_class_i:
                tr_class_i[y_pred1] += 1
            else:
                tr_class_i[y_pred1] = 1

    # 求解 gmean
    m = len(y_true)
    gmean = 1
    for key in y_true:
        if key not in tr_class_i:
            tr_class_i[key] = 0
        gmean *= tr_class_i[key]/n_class_i[key]
    gmean = pow(gmean, 1/m)

    return gmean

def mAUC(y_true, y_pred):
    # TODO mAUC
    return 0

def acc(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    d1 = (y_true == y_pred)
    n_yes = sum(d1[d1 == True])
    acc1 = n_yes/len(y_true)

    return acc1


def ppv(y_true, y_pred):
    n_sample = len(y_true)
    y_true_counter = Counter(y_true)
    PPV_val = {}
    for key in y_true_counter:
        # 计算每个类的 ppv
        d1 = np.zeros(n_sample, dtype="int32")
        index1 = np.where(y_true == key)[0]
        for i in index1:
            d1[i] = 1
        d2 = np.zeros(n_sample, dtype="int32")
        index2 = np.where(y_pred == key)[0]
        for i in index2:
            d2[i] = 1

        TP = np.sum((d1*d2) == 1)
        FP = 0
        for i in range(len(d1)):
            if d1[i] == 0 and d2[i] == 1:
                FP = FP + 1
        if TP+FP != 0:
            PPV_val[key] = TP/(TP+FP)
        else:
            PPV_val[key] = 0

    return PPV_val

def PFC(y_true, all_y_pred):
    """

    :param y_true:
    :param all_y_pred: 所有分类器的预测结果
    :return:
    """
    M = len(all_y_pred)
    PFC_value = []
    credit = np.zeros((M, M), dtype="float32")
    acc_credit = np.zeros((M, ), dtype="float32")

    # credit
    for i in range(M):
        failure_pattern_1 = np.zeros(len(y_true), dtype="int32")
        for k in range(len(y_true)):
            if y_true[k] == all_y_pred[i][k]:
                failure_pattern_1[k] = 1
        for j in range(M):
            if i != j:
                failure_pattern_2 = np.zeros(len(y_true), dtype="int32")
                for k in range(len(y_true)):
                    if y_true[k] == all_y_pred[j][k]:
                        failure_pattern_2[k] = 1
                credit[i][j] = distance(failure_pattern_1, failure_pattern_2)

    # acc_credit
    for i in range(M):
        for j in range(M):
            acc_credit[i] += credit[i][j]

    # PFC
    for i in range(M):
        PFC_value.append(acc_credit[i]/(M-1))

    return PFC_value


def distance(failure_pattern_1, failure_pattern_2):
    failure_pattern_1 = np.array(failure_pattern_1)
    failure_pattern_2 = np.array(failure_pattern_2)
    d1 = []
    for n1, n2 in zip(failure_pattern_1, failure_pattern_2):
        if n1 != n2:
            d1.append(1)
        else:
            d1.append(0)
    d1 = np.array(d1)
    hamming_distance = sum(d1[d1 == 1])
    zero_nums = sum(failure_pattern_1 == 0) + sum(failure_pattern_2 == 0)
    if hamming_distance != 0 and zero_nums != 0:
        fixed_distance = hamming_distance/zero_nums
    else:
        fixed_distance = 0

    return fixed_distance