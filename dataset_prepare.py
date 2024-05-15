import torch
import h5py
import numpy as np
from scipy.io import loadmat

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def read_data(dataset):
    if dataset == 'Germany':
        file_name1 = r'./VH_cut.mat'
        file_name2 = r'./VV_cut.mat'
        file_name3 = r'./label.mat'
    else:
        raise ValueError("Unknown dataset")

    with h5py.File(file_name1, 'r') as f:
        f = h5py.File(file_name1, 'r')
        # print(f.keys())
    vh = np.transpose(f['VH'])

    with h5py.File(file_name2, 'r') as f:
        f = h5py.File(file_name2, 'r')
        # print(f.keys())
    vv = np.transpose(f['VV'])

    label_data = loadmat(file_name3)
    label = label_data['label'] ## for 2017~2018
    # label = label_data['label_s2'] ## for 2020~2021

    ## normalize data
    vh1 = normalization(vh)
    vv1 = normalization(vv)

    return vh1, vv1, label

def slide_crop(image, patch, overlay):
    height, width = image.shape[0], image.shape[1]
    crop_list = []

    # middle region
    for i in range(int((height - patch * overlay) / (patch * (1 - overlay)))):
        for j in range(int((width - patch * overlay) / (patch * (1 - overlay)))):
            if len(image.shape) == 2:
                cropped = image[int(i * patch * (1 - overlay)) : int(i * patch * (1 - overlay)) + patch,
                                int(j * patch * (1 - overlay)) : int(j * patch * (1 - overlay)) + patch]
            else:
                cropped = image[int(i * patch * (1 - overlay)) : int(i * patch * (1 - overlay)) + patch,
                                int(j * patch * (1 - overlay)) : int(j * patch * (1 - overlay)) + patch, :]
            crop_list.append(cropped)
    # last column region
    for i in range(int((height - patch * overlay) / (patch * (1 - overlay)))):
        if len(image.shape) == 2:
            cropped = image[int(i * patch * (1 - overlay)) : int(i * patch * (1 - overlay)) + patch,
                            (width - patch) : width]
        else:
            cropped = image[int(i * patch * (1 - overlay)) : int(i * patch * (1 - overlay)) + patch,
                            (width - patch) : width, :]
        crop_list.append(cropped)
    # last row region
    for j in range(int((width - patch * overlay) / (patch * (1 - overlay)))):
        if len(image.shape) == 2:
            cropped = image[(height - patch) : height,
                            int(j * patch * (1 - overlay)) : int(j * patch * (1 - overlay)) + patch]
        else:
            cropped = image[(height - patch) : height,
                            int(j * patch * (1 - overlay)) : int(j * patch * (1 - overlay)) + patch, :]
        crop_list.append(cropped)
    # bottom right region
    if len(image.shape) == 2:
        cropped = image[(height - patch) : height,
                        (width - patch) : width]
    else:
        cropped = image[(height - patch) : height,
                        (width - patch) : width, :]
    crop_list.append(cropped)
    print('Number of Cropped Image: {}'.format(len(crop_list)))

    if overlay == 0:
        ColumnOver = int((height - patch * overlay) % (patch * (1 - overlay)) + patch * overlay // 2)
        RowOver = int((width - patch * overlay) % (patch * (1 - overlay)) + patch * overlay // 2)
        print('Number of ColumnOver and RowOver: {}, {}'.format(ColumnOver, RowOver))
        return np.array(crop_list), ColumnOver, RowOver
    else:
        return np.array(crop_list)

def main():
    dataset = 'Germany'
    vh, vv, label = read_data(dataset)

    patch, overlay = 128, 0.5
    # for training dataset
    vh_list = slide_crop(vh, patch, overlay)
    vv_list = slide_crop(vv, patch, overlay)
    label_list = slide_crop(label, patch, overlay)
    # non-overlay crop for valid data
    vh_test_list, ColumnOver, RowOver = slide_crop(vh, patch, overlay=0)
    vv_test_list, _, _ = slide_crop(vv, patch, overlay=0)
    label_test_list, _, _ = slide_crop(label, patch, overlay=0)

    np.savez('train_list.npz', vh_list=vh_list, vv_list=vv_list, label_list=label_list,
                    vh_test_list=vh_test_list, vv_test_list=vv_test_list, label_test_list=label_test_list)

if __name__ == '__main__':
    main()
