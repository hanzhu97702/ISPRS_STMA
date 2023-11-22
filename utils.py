import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import argparse

def print_args(args):
    for k, v in zip(args.keys(), args.values()):
        print("{0}: {1}".format(k, v))

class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

def accuracy(output, target, topk, ignore_index):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.squeeze().cpu().detach().numpy()
    target = target.cpu().detach().numpy()

    if ignore_index is None:
        correct_k = 100 * float(np.count_nonzero(pred == target)) / target.size
    else:
        mask = pred == target
        mask[np.where(target == ignore_index)] = False
        total = np.sum(np.where(target != ignore_index, 1, 0))
        correct_k =  100 * np.sum(mask) / total

    res = []
    res.append(torch.from_numpy(np.array(correct_k)))
    return res, target, pred

def output_metric(tar, pre, mode=False):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)

    num_classes = int(np.max(pre)) + 1
    f1 = f1_score(pre, tar, num_classes, all=mode, ignore_index=0)
    iou = IoU(pre, tar, num_classes, all_iou=False, ignore_index=0)

    return OA, f1, iou

def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)

    return OA, AA_mean, Kappa, AA

def IoU(pred, gt, num_classes, all_iou=False, ignore_index=None):
    '''Compute the IoU by class and return mean IoU'''
    iou = []
    for i in range(num_classes):
        if i == ignore_index:
            continue
        if np.sum(gt == i) == 0:
            iou.append(np.NaN)
            continue
        TP = np.sum(np.logical_and(pred == i, gt == i))
        FP = np.sum(np.logical_and(pred == i, gt != i))
        FN = np.sum(np.logical_and(pred != i, gt == i))
        iou.append(TP / (TP + FP + FN))
    '''nanmean: if a class is not present in the image, it's a NaN'''
    result = [np.nanmean(iou), iou] if all_iou else np.nanmean(iou)
    return result

def f1_score(pred, gt, num_classes, all=False, ignore_index=None):
    '''Compute the F1 by class and return mean F1'''
    f1 = []
    for i in range(num_classes):
        if i == ignore_index:
            continue
        if np.sum(gt == i) == 0:
            f1.append(np.NaN)
            continue
        TP = np.sum(np.logical_and(pred == i, gt == i))
        FP = np.sum(np.logical_and(pred == i, gt != i))
        FN = np.sum(np.logical_and(pred != i, gt == i))
        prec = TP / (TP + FP)
        recall = TP / (TP + FN)
        try:
            result = 2 * TP / (2 * TP + FP + FN)
        except ZeroDivisionError:
            result = 0
        f1.append(result)
    result = [np.nanmean(f1), f1] if all else np.nanmean(f1)
    return result
