import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
from sklearn.model_selection import KFold
from torch import multiprocessing
from matplotlib import pyplot as plt

multiprocessing.set_sharing_strategy('file_system')

def usegpu(gpu_ids):
    for s in gpu_ids.split(','):
        try:
            int(s)
        except ValueError as e:
            print("Invalid gpu id: {}".format(s))
            raise ValueError
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    if gpu_ids:
        if torch.cuda.is_available():
            use_gpu = True
        else:
            use_gpu = False
    else:
        use_gpu = False
    return use_gpu

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cal_roc(y_true, y_pred):
    FPR, recall, thresholds = roc_curve(y_true, y_pred)
    area = roc_auc_score(y_true, y_pred)
    return area, FPR, recall, thresholds
def compute_mAP(y_true,y_pred):
    aps = []
    y_true=y_true.transpose((1,0))
    y_pred=y_pred.transpose((1,0))
    for i in range(y_true.shape[0]):
        aps.append(average_precision_score(y_true[i],y_pred[i]))
    return np.mean(aps)

def plot_roc(y_true, y_pred, title='ROC',path=''):
    area, FPR, recall, thresholds = cal_roc(y_true, y_pred)
    # fig = plt.figure()
    plt.plot(FPR, recall, color='red', label='ROC curve (area=%0.2f)' % area)
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.xlim([0.05, 1.05])
    plt.ylim([0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('Recall')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.savefig(path)

def get_tp_tn_fp_fn_numpy(y_true,y_pred):
    seg_inv, gt_inv = np.logical_not(y_pred), np.logical_not(y_true)
    tp = float(np.logical_and(y_pred, y_true).sum())  # float for division
    tn = np.logical_and(seg_inv, gt_inv).sum()
    fp = np.logical_and(y_pred, gt_inv).sum()
    fn = np.logical_and(seg_inv, y_true).sum()
    return tp, tn, fp, fn

def get_cls_index_numpy(tp, tn, fp, fn):
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (fn + tn)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    f1_score = 2 * tp / (2 * tp + fp + fn + 1e-6)

    return acc,sensitivity,specificity,ppv,npv,f1_score,

def get_tp_tn_fp_fn_torch(y_true,y_pred):
    y_true = y_true.bool()
    y_pred = y_pred.bool()
    seg_inv, gt_inv = ~y_pred, ~y_true
    tp = (y_pred & y_true).float().sum()  # float for division
    tn = (seg_inv & gt_inv).float().sum()
    fp = (y_pred & gt_inv).float().sum()
    fn = (seg_inv & y_true).float().sum()
    return tp,tn,fp,fn

def get_seg_index_torch(tp,tn,fp,fn):
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (fn + tn)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    f1_score = 2 * tp / (2 * tp + fp + fn + 1e-6)
    iou = (tp + 1e-6) / (tp + fn + fp + 1e-6)
    return acc, sensitivity, specificity, ppv, npv, f1_score, iou

def get_cls_index(y_true,y_pred,y_prob):
    seg_inv, gt_inv = np.logical_not(y_pred), np.logical_not(y_true)
    tp = float(np.logical_and(y_pred, y_true).sum())  # float for division
    tn = np.logical_and(seg_inv, gt_inv).sum()
    fp = np.logical_and(y_pred, gt_inv).sum()
    fn = np.logical_and(seg_inv, y_true).sum()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (fn + tn)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    f1_score = 2 * tp / (2 * tp + fp + fn + 1e-6)
    auc = roc_auc_score(y_true, y_prob)
    return acc,sensitivity,specificity,ppv,npv,f1_score,auc

def get_cls_index_torch(y_true,y_pred,y_prob):
    y_true = y_true.bool()
    y_pred = y_pred.bool()
    seg_inv, gt_inv = ~y_pred, ~y_true
    tp = (y_pred & y_true).float().sum()  # float for division
    tn = (seg_inv & gt_inv).float().sum()
    fp = (y_pred & gt_inv).float().sum()
    fn = (seg_inv & y_true).float().sum()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (fn + tn)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    f1_score = 2 * tp / (2 * tp + fp + fn + 1e-6)
    auc = roc_auc_score(y_true, y_prob)
    return acc,sensitivity,specificity,ppv,npv,f1_score,auc

def get_split_idx(samples, ratio):
    random_index = list(range(len(samples)))
    random.shuffle(random_index)
    return random_index[:int(len(random_index) * ratio)], random_index[int(len(random_index) * ratio):]

def get_k_fold_idx(samples, fold=3,val_ratio=0.2, test_ratio=0.2, random_state=0):
    if fold==1:
        indexs, test_index = get_split_idx(samples, 1 - test_ratio)
        train_index, val_index = get_split_idx(indexs, 1 - val_ratio)
        all_indexs = []
        all_indexs.append([
            [indexs[i] for i in train_index],
            [indexs[i] for i in val_index],
            test_index
        ])
        return all_indexs
    else:
        indexs, test_index = get_split_idx(samples, 1 - test_ratio)
        # 交叉验证数据
        kf = KFold(n_splits=fold, shuffle=True, random_state=random_state)
        all_indexs = []
        for train_index, valid_index in kf.split(indexs):
            all_indexs.append([
                [indexs[i] for i in train_index],
                [indexs[i] for i in valid_index],
                test_index
            ])
        return all_indexs