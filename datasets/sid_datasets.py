'''
@File: sid_datasets.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 1月 17, 2024
@HomePage: https://github.com/YanJieWen
'''


import os.path as osp

import PIL.Image
import torch
import torch.utils.data as data
import datasets.utils as util
import torch.nn.functional as F
import random
import cv2
import numpy as np
import glob
import os
import functools



class Dataset_SIDImage(data.Dataset):
    def __init__(self,gt_root,lq_root,phase='train',train_size=[960,512]):
        self.cache_data = True
        self.half_N_frames = 5 // 2
        self.GT_root, self.LQ_root = gt_root, lq_root
        self.data_info = {'path_LQ': [], 'path_GT': [],
                          'folder': [], 'idx': [], 'border': []}
        self.imgs_LQ, self.imgs_GT = {}, {}
        self.phase = phase
        self.train_size = train_size
        subfolders_LQ_origin = util.glob_file_list(self.LQ_root)
        subfolders_GT_origin = util.glob_file_list(self.GT_root)
        assert  len(subfolders_LQ_origin)==len(subfolders_GT_origin),'not paired data'
        subfolders_LQ = []
        subfolders_GT = []
        if self.phase=='train':
            for mm in range(len(subfolders_LQ_origin)):
                name = os.path.basename(subfolders_LQ_origin[mm])
                if '0' in name[0] or '2' in name[0]:
                    subfolders_LQ.append(subfolders_LQ_origin[mm])
                    subfolders_GT.append(subfolders_GT_origin[mm])
        else:
            for mm in range(len(subfolders_LQ_origin)):
                name = os.path.basename(subfolders_LQ_origin[mm])
                if '1' in name[0]:
                    subfolders_LQ.append(subfolders_LQ_origin[mm])
                    subfolders_GT.append(subfolders_GT_origin[mm])
        for subfolder_LQ, subfolder_GT in zip(subfolders_LQ, subfolders_GT):
            subfolder_name = osp.basename(subfolder_LQ)
            img_paths_LQ = util.glob_file_list(subfolder_LQ)
            img_paths_GT = util.glob_file_list(subfolder_GT)
            max_idx = len(img_paths_LQ)
            self.data_info['path_LQ'].extend(
                img_paths_LQ)
            self.data_info['path_GT'].extend(img_paths_GT)
            self.data_info['folder'].extend([subfolder_name] * max_idx)
            for i in range(max_idx):
                self.data_info['idx'].append('{}/{}'.format(i, max_idx))
            border_l = [0] * max_idx
            for i in range(self.half_N_frames):
                border_l[i] = 1
                border_l[max_idx - i - 1] = 1
            self.data_info['border'].extend(border_l)
            if self.cache_data:
                self.imgs_LQ[subfolder_name] = img_paths_LQ
                self.imgs_GT[subfolder_name] = img_paths_GT

    def __getitem__(self, index):
        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)
        border = self.data_info['border'][index]

        img_LQ_path = self.imgs_LQ[folder][idx]
        img_LQ_path = [img_LQ_path]
        img_GT_path = self.imgs_GT[folder][0]
        img_GT_path = [img_GT_path]

        if self.phase == 'train':
            img_LQ = util.read_img_seq2(img_LQ_path, self.train_size)
            img_GT = util.read_img_seq2(img_GT_path, self.train_size)
            img_LQ = img_LQ[0]
            img_GT = img_GT[0]

            img_LQ_l = [img_LQ]
            img_LQ_l.append(img_GT)
            rlt = util.augment_torch(
                img_LQ_l, True,True)
            img_LQ = rlt[0]
            img_GT = rlt[1]

        elif self.phase == 'val':
            img_LQ = util.read_img_seq2(img_LQ_path, self.train_size)
            img_GT = util.read_img_seq2(img_GT_path, self.train_size)
            img_LQ = img_LQ[0]
            img_GT = img_GT[0]

        else:
            img_LQ = util.read_img_seq2(img_LQ_path, self.train_size)
            img_GT = util.read_img_seq2(img_GT_path, self.train_size)
            img_LQ = img_LQ[0]
            img_GT = img_GT[0]
        return {
            'lq': img_LQ,
            'gt': img_GT,
            # 'nf': img_nf,
            'folder': folder,
            'idx': self.data_info['idx'][index],
            'border': border,
            'lq_path': img_LQ_path[0],
            'gt_path': img_GT_path[0]
        }
    def __len__(self):
        return len(self.data_info['path_LQ'])






if __name__ == '__main__':
    from torchvision import transforms as ts
    import matplotlib.pyplot as plt
    import random
    from torch.utils.data import DataLoader
    gt_root, lq_root = '../parired_datasets/sid/long_sid2/', '../parired_datasets/sid/short_sid2/'
    data = Dataset_SIDImage(gt_root, lq_root,phase='val')
    print(f'共计{int(len(data))}个输入样本')
    train_loader = DataLoader(data, batch_size=4, shuffle=True, num_workers=0, sampler=None)
#     # for idx in random.sample(range(len(data)), k=5):
#     #     lq_im = PIL.Image.fromarray(np.uint8(data[idx]['lq'].numpy().transpose(1,2,0)*255.))
#     #     gt_im = PIL.Image.fromarray(np.uint8(data[idx]['gt'].numpy().transpose(1,2,0) * 255.))
#     #     imgs = [lq_im,gt_im]
#     #     for i in range(1,len(imgs)+1):
#     #         plt.subplot(1,2,i)
#     #         plt.imshow(imgs[i-1])
#     #     plt.show()
    for i,data in enumerate(train_loader):
        if i==0:
            img_lq = np.uint8(data['lq'].numpy().transpose(0,2,3,1)*255.)
            img_gt = np.uint8(data['gt'].numpy().transpose(0,2,3,1)*255.)
            imgs = [PIL.Image.fromarray(im) for im in img_lq]+[PIL.Image.fromarray(im) for im in img_gt]
            print(img_gt.shape)
            for i in range(1,len(imgs)+1):
                plt.subplot(2,4,i)
                plt.imshow(imgs[i-1])
            plt.show()
        else:
            break
