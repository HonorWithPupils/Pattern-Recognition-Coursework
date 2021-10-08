import numpy as np
from matplotlib import pyplot as plt

def singleMetics(P, T):
    TP = ((P==1) & (T==1)).sum()
    FP = ((P==0) & (T==1)).sum()
    FN = ((P==1) & (T==0)).sum()
    TN = ((P==0) & (T==0)).sum()
    Accuracy = (TP+TN)/len(P)
    Precision = TP/(TP+FP)
    Recall = TP/(TP+FN)
    F1 = (2*Precision*Recall)/(Precision+Recall)
    return { 'TP':TP, 'FP':FP, 'FN':FN, 'TN':TN, 'Accuracy':Accuracy,
             'Precision':Precision, 'Recall':Recall, 'F1':F1}

def PRcurve(y_prob, y_true, plot=False):
    # 排序
    sorted_indices = np.argsort(y_prob)[::-1]
    y_prob = y_prob[sorted_indices]
    y_true = y_true[sorted_indices]
    
    precisions = []
    recalls = []
    thresholds = []
    
    # 如果 y_prob 没有 1，则手动添加一个 (0,1) 坐标
    # 防止 Precision 分母/分子 皆为 0
    if y_prob[0] != 1:
        precisions.append(1)
        recalls.append(0)
        thresholds.append(1)
    
    # 初始化：全部预测为 False
    TP = 0
    FP = 0
    FN = y_true.sum().item()
    for prob, true in zip(y_prob, y_true):
        if true == 1:
            TP += 1
            FN -= 1
        else:
            FP += 1
        P = TP/(TP+FP)
        R = TP/(TP+FN)
        precisions.append(P)
        recalls.append(R)
        thresholds.append(prob)
        # 当 Recall 为 1 时，就可以退出了
        if R == 1:
            break
    
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    thresholds = np.array(thresholds)
    
    if plot:
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.plot(recalls, precisions)
        plt.fill_between(recalls, precisions, color='lightblue')
    
    return precisions, recalls, thresholds

def AP(y_prob, y_true):
    P, R, _ = PRcurve(y_prob, y_true)
    maxPs = list(set(P))
    maxPs.sort(reverse=True)
    S = 0
    prevr = 0
    for maxP in maxPs:
        r = R[np.where(P == maxP)[0].max()]
        S += maxP*(r-prevr)
        prevr = r
        if r == 1:
            break
    return S

def BEP(y_prob, y_true):
    P, R, _ = PRcurve(y_prob, y_true)
    return P[P<=R].max()

def ROCcurve(y_prob, y_true, plot=False):
    # 排序
    sorted_indices = np.argsort(y_prob)[::-1]
    y_prob = y_prob[sorted_indices]
    y_true = y_true[sorted_indices]
    
    FPR = []
    TRP = []
    thresholds = []
    
    # 初始化：全部预测为 False
    TP = 0
    FP = 0
    FN = y_true.sum().item()
    TN = len(y_true) - FN
    fpr = FP/(FP+TN)
    trp = TP/(TP+FN)
    FPR.append(fpr)
    TRP.append(trp)
    thresholds.append(1.)
    
    for prob, true in zip(y_prob, y_true):
        if true == 1:
            TP += 1
            FN -= 1
        else:
            FP += 1
            TN -= 1
        fpr = FP/(FP+TN)
        trp = TP/(TP+FN)
        FPR.append(fpr)
        TRP.append(trp)
        thresholds.append(prob)
    
    FPR = np.array(FPR)
    TRP = np.array(TRP)
    thresholds = np.array(thresholds)
    
    if plot:
        plt.xlabel('FPR')
        plt.ylabel('TRP')
        plt.plot(FPR, TRP)
        plt.fill_between(FPR, TRP, color='lightblue')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    return TRP, FPR, thresholds

def AUC(y_prob, y_true):
    TRP, FPR, _ = ROCcurve(y_prob, y_true, plot=False)
    prevfpr = 0
    S = 0
    for trp, fpr in zip(TRP, FPR):
        S += (fpr-prevfpr)*trp
        prevfpr = fpr
    return S