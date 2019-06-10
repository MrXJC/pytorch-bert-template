import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score, classification_report


def F1(outputs, labels):
    preds = np.argmax(outputs, axis=1)
    #classifiction_metric(preds, labels)
    return f1_score(y_true=labels, y_pred=preds)


def classifiction_metric(preds, labels):
    """ 分类任务的评价指标， 传入的数据需要是 numpy 类型的 """

    labels_list = [i for i in range(2)]
    report = classification_report(labels, preds, labels=labels_list, target_names=['0', '1'], digits=5)
    print(report)

def acc(outputs, labels):
    preds = np.argmax(outputs, axis=1)
    return (preds == labels).mean()


def spearman(outputs, labels):
    preds = np.argmax(outputs, axis=1)
    return spearmanr(preds, labels)[0]


def pearson(preds, labels):
    return pearsonr(preds, labels)[0]


def matthews(outputs, labels):
    preds = np.argmax(outputs, axis=1)
    return matthews_corrcoef(labels, preds)
