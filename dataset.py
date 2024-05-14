import torch
import torch.utils.data as Data

from random import random
import numpy as np

def getdata(dataset, patch, batchsize):
    vh_list, vv_list, label_list, num_classes, row, col, band = read_data(dataset, mode='train')
    # non-overlay crop for test data
    vh_test_list, vv_test_list, label_test_list, ColumnOver, RowOver = read_data(dataset, mode='test')
    # numpy to tensor
    vh = torch.from_numpy(vh_list.transpose(0,3,2,1)).type(torch.FloatTensor)
    vv = torch.from_numpy(vv_list.transpose(0,3,2,1)).type(torch.FloatTensor)
    label = torch.from_numpy(label_list).type(torch.LongTensor)
    label_train_set = Data.TensorDataset(vh, vv, label)

    vh_test = torch.from_numpy(vh_test_list.transpose(0,3,2,1)).type(torch.FloatTensor)
    vv_test = torch.from_numpy(vv_test_list.transpose(0,3,2,1)).type(torch.FloatTensor)
    label_test = torch.from_numpy(label_test_list).type(torch.LongTensor)
    label_test_set = Data.TensorDataset(vh_test, vv_test, label_test)

    # generate shuffle data ignore_index for train_set
    rate_train_test = 0.6
    shuffled_indices = np.random.permutation(len(vh_list))
    train_idx = shuffled_indices[:int(rate_train_test * len(vh_list))]
    val_idx = shuffled_indices[int(rate_train_test * len(vh_list)):]

    train_loader = Data.DataLoader(label_train_set, batch_size = batchsize, drop_last = True, sampler = Data.SubsetRandomSampler(train_idx))
    valid_loader = Data.DataLoader(label_train_set, batch_size = batchsize, drop_last = False, sampler = Data.SubsetRandomSampler(val_idx))
    test_loader = Data.DataLoader(label_test_set, batch_size = 1, shuffle = False)

    return train_loader, test_loader, test_loader, num_classes, band, ColumnOver, RowOver, row, col

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def read_data(dataset, mode):
    if dataset == 'Germany':
        num_classes = 16 # 0~15
        row, col, band = 3731, 5095, 41
        data_file = './data/train_list.npz'
        flag = 1
    elif dataset == 'Germany_S2':
        num_classes = 16 # 0~15
        row, col, band = 3731, 5095, 41
        data_file = './data/S2/train_list_s2_for_train.npz'
        flag = 2
    else:
        raise ValueError("Unknown dataset")

    # flag=1 represents test data for original area in 2017~2018;
    # flag=2 represents test data for S2 study area in 2020~2021;

    print("Current flag value is {}".format(flag))

    temporal_data = np.load(data_file)
    if mode == 'train':
        vh_list = temporal_data['vh_list']
        vv_list = temporal_data['vv_list']
        label_list = temporal_data['label_list']
        return vh_list, vv_list, label_list, num_classes, row, col, band
    elif mode == 'test':
        ColumnOver, RowOver = 19, 103
        if flag == 1:
            vh_list = temporal_data['vh_test_list']
            vv_list = temporal_data['vv_test_list']
            label_list = temporal_data['label_test_list']
        elif flag == 2:
            temporal_data_s2 = np.load('./data/S2/train_list_s2.npz')
            vh_list = temporal_data_s2['vh_test_list']
            vv_list = temporal_data_s2['vv_test_list']
            label_list = temporal_data_s2['label_test_list']
        else:
            raise ValueError("Unknown flag value")
        return vh_list, vv_list, label_list, ColumnOver, RowOver
    else:
        raise ValueError("Wrong input mode")

def get_full_label(label_total, result_shape, patch, RowOver, ColumnOver):
    result = np.zeros(result_shape, np.uint8)

    rr = int(result_shape[0] / patch)
    cc = int(result_shape[1] / patch)
    # j represent the number of row
    jj = 0
    for i, img in enumerate(label_total):
        img = img.squeeze()

        if i < rr * cc:
            if  (i + 1) % cc == 0:
                result[jj * patch : jj * patch + patch, (cc - 1) * patch : cc * patch] = img
                jj = jj + 1
            else:
                result[jj * patch : jj * patch + patch, (i - cc * jj) * patch : (i - cc * jj) * patch + patch] = img
        elif  i >= rr * cc  and  i < (rr * cc + rr):
            if i == rr * cc:
                jj = 0
            else:
                jj = jj + 1
            result[jj * patch : jj * patch + patch, result_shape[1] - ColumnOver : result_shape[1]] = img[:, patch - ColumnOver : patch]
        elif i >= (rr * cc + rr) and i < len(label_total) - 1:
            if i == rr * cc + rr:
                jj = 0
            else:
                jj = jj + 1
            result[result_shape[0] - RowOver : result_shape[0], jj * patch : jj * patch + patch] = img[patch - RowOver : patch, :]
        else:
            result[result_shape[0] - RowOver : result_shape[0], result_shape[1] - ColumnOver : result_shape[1]] = img[patch - RowOver : patch, patch - ColumnOver : patch]
    return result
