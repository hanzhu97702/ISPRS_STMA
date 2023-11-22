import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import numpy as np
from model import STMA
from utils import AvgrageMeter, accuracy, output_metric, print_args
from criterion import CrossEntropyLoss2d
from save import save_to_pth, load_from_pth
import matplotlib.pyplot as plt
import scipy.io as sio
import argparse
import dataset
import random
import time
import os

parser = argparse.ArgumentParser(description='Multitemporal Crop Mapping')
parser.add_argument('--fix_random', action='store_true', help='fix randomness')
parser.add_argument('--season_flag', action='store_true', help='whether consider temporal effect')
parser.add_argument('--save_model_flag', action='store_true', help='whether save model train parameters')
parser.add_argument('--load_model_flag', action='store_true', help='whether load model saved parameters')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--gpu_id', default='1', help='gpu id')
parser.add_argument('--epoches', default=500, type=int, help='number of epoch')
parser.add_argument('--test_freq', default=5, type=int, help='number of eval times')
# dataset parameters
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--patch', default=128, type=int, help='input data size')
parser.add_argument('--dataset', choices=['Germany', 'Germany_S2'], default='Germany', type=str, help='dataset to use')
# model parameters
parser.add_argument('--emb_dim', default=328, type=int, help='embedding size')
parser.add_argument('--mlp_dim', default=16, type=int, help='mlp dimension size')
parser.add_argument('--num_heads', default=4, type=int, help='number of head')
parser.add_argument('--attn_dropout_rate', default=0.0, type=float)
parser.add_argument('--dropout_rate', default=0.1, type=float)
# optimizer parameters
parser.add_argument('--learning_rate', default=1e-3, type=float)
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--power', default=0.9, type=float)
args = parser.parse_args()

print_args(vars(args))
print("**************************************************")

def train_epoch(model, train_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (vh_list, vv_list, batch_target) in enumerate(train_loader):
        vh_list = vh_list.cuda()
        vv_list = vv_list.cuda()
        batch_target = batch_target.cuda()

        input_tensor = torch.cat((vh_list.unsqueeze(2), vv_list.unsqueeze(2)), dim=2)

        optimizer.zero_grad()
        batch_pred, batch_aux = model(input_tensor)
        loss_main = criterion(batch_pred, batch_target)
        loss_aux = criterion(batch_aux, batch_target)
        loss = 0.5*loss_main + 0.5*loss_aux

        loss=loss_main
        loss.backward()
        optimizer.step()

        prec1, target, pred = accuracy(batch_pred, batch_target, topk=(1,), ignore_index=0)
        n = batch_target.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, target)
        pre = np.append(pre, pred)

    return top1.avg, objs.avg, tar, pre

def valid_epoch(model, valid_loader):
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (vh_list, vv_list, batch_target) in enumerate(valid_loader):
        vh_list = vh_list.cuda()
        vv_list = vv_list.cuda()
        batch_target = batch_target.cuda()

        input_tensor = torch.cat((vh_list.unsqueeze(2), vv_list.unsqueeze(2)), dim=2)
        batch_pred, batch_aux = model(input_tensor)

        prec1, target, pred = accuracy(batch_pred, batch_target, topk=(1,), ignore_index=0)
        tar = np.append(tar, target)
        pre = np.append(pre, pred)

    return tar, pre

def test_epoch(model, test_loader):
    label_total = []
    label_gt = []
    for batch_idx, (vh_list, vv_list, batch_target) in enumerate(test_loader):
        vh_list = vh_list.cuda()
        vv_list = vv_list.cuda()
        batch_target = batch_target.cuda()

        input_tensor = torch.cat((vh_list.unsqueeze(2), vv_list.unsqueeze(2)), dim=2)
        output_label, feature = model(input_tensor)
        label = batch_target[0].cpu().detach().numpy()

        pred = output_label.cpu().detach().numpy().transpose(0, 2, 3, 1)
        seg_pred = np.asarray(np.argmax(pred, axis=3), dtype = np.uint8)
        label_total.append(seg_pred)

        label = batch_target.cpu().detach().numpy()
        label_gt.append(label)

    return label_total, label_gt

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    if torch.cuda.is_available():
        print('GPU is true')
        print('Cuda Version: {}'.format(torch.version.cuda))
        print('Using GPU: {}'.format(args.gpu_id))
    else:
        print('CPU is true')
    print("**************************************************")

    if args.fix_random:
        manualSeed = args.seed
        np.random.seed(manualSeed)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        torch.cuda.manual_seed(manualSeed)
        torch.cuda.manual_seed_all(manualSeed)

        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.benchmark = True

    # create dataset and model
    train_loader, valid_loader, test_loader, num_classes, band, ColumnOver, RowOver, row, col = dataset.getdata(args.dataset, args.patch, args.batch_size)
    # create model
    print("Create model")
    model = STMA(
            input_band = band,
            emb_dim = args.emb_dim,
            mlp_dim = args.mlp_dim,
            num_heads = args.num_heads,
            num_classes = num_classes,
            attn_dropout_rate = args.attn_dropout_rate,
            dropout_rate = args.dropout_rate)

    model = model.cuda()
    # criterion
    criterion = CrossEntropyLoss2d(ignore_index = -100).cuda()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay = args.weight_decay)

    print("Start training")
    tic = time.time()
    for epoch in range(args.epoches):
        model.train()
        train_acc, train_obj, tar_t, pre_t = train_epoch(model, train_loader, criterion, optimizer)
        OA1, mF1, mIoU1 = output_metric(tar_t, pre_t)
        print("Epoch: {:03d} train_loss: {:.4f} train_acc: {:.4f}"
                    .format(epoch+1, train_obj, train_acc))
        print("OA: {:.4f} mF1: {:.4f} mIoU: {:.4f}"
                    .format(OA1, mF1, mIoU1))
        print("************************************************")

        if (epoch % args.test_freq == 0) | (epoch == args.epoches - 1):
            model.eval()
            tar_v, pre_v = valid_epoch(model, valid_loader)
            OA2, mF2, mIoU2 = output_metric(tar_v, pre_v)
            print("*****************Testing Result****************")
            print("OA: {:.4f} mF1: {:.4f} mIoU: {:.4f}"
                        .format(OA2, mF2, mIoU2))
            print("************************************************")
    toc = time.time()
    print("Running Time: {:.2f}".format(toc-tic))
    print("**************************************************")

    if args.save_model_flag:
        model_name = ["STMA"]
        RESULT_DIR = "./model_save/{}/".format("_".join(model_name))
        save_to_pth(model, os.path.join(RESULT_DIR, "stma.pth"))

    if args.load_model_flag:
        model_name = ["STMA"]
        RESULT_DIR = "./model_save/{}/".format("_".join(model_name))
        load_from_pth(model, os.path.join(RESULT_DIR, "stma.pth"))

    # test phase and output the final segmentation map
    model.eval()
    label_total, label_gt = test_epoch(model, test_loader)
    label_full = dataset.get_full_label(label_total, (row, col), args.patch, RowOver, ColumnOver)
    label_full_gt = dataset.get_full_label(label_gt, (row, col), args.patch, RowOver, ColumnOver)

    label_full_gt_reshape = label_full_gt.reshape(row*col)
    label_full_reshape = label_full.reshape(row*col)
    # evaluate label result
    print("******************Final Result********************")
    OA_final, mF1_final, mIoU_final = output_metric(label_full_gt_reshape, label_full_reshape, mode=True)
    print("OA: {:.4f}, mF1: {:.4f}, mIoU: {:.4f}".format(OA_final, mF1_final[0], mIoU_final))
    print("**************************************************")
    print("Each F1: {}".format(mF1_final[1:]))
    print("**************************************************")
    print("**************************************************")
    print(np.unique(label_full))
    print("**************************************************")

    plt.imshow(label_full)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    sio.savemat('results.mat',{'output': label_full, 'label':label_full_gt})

if __name__ == '__main__':
    main()
