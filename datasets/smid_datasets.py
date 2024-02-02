'''
@File: smid_datasets.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 1月 03, 2024
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


class Dataset_SMIDImage(data.Dataset):
    def __init__(self, gt_root,lq_root,phase='train',train_size=[960,512]):
        super(Dataset_SMIDImage, self).__init__()
        self.cache_data = True
        self.half_N_frames = 5 // 2
        self.GT_root, self.LQ_root = gt_root,lq_root
        self.data_info = {'path_LQ': [], 'path_GT': [],
                          'folder': [], 'idx': [], 'border': []}
        self.imgs_LQ, self.imgs_GT = {}, {}
        self.phase = phase
        self.train_size = train_size
        testing_dir = []
        data_root = os.path.dirname(self.GT_root)

        f = open(os.path.join(data_root, 'test_list.txt'))
        lines = f.readlines()
        for mm in range(len(lines)):
            this_line = lines[mm].strip()
            testing_dir.append(this_line)

        subfolders_LQ = util.glob_file_list(self.LQ_root)
        subfolders_GT = util.glob_file_list(self.GT_root)

        for subfolder_LQ, subfolder_GT in zip(subfolders_LQ, subfolders_GT):
            subfolder_name = osp.basename(subfolder_LQ)

            if phase == 'train':
                if (subfolder_name in testing_dir):
                    continue
            else:  # val, test
                if not (subfolder_name in testing_dir):
                    continue

            img_paths_LQ = util.glob_file_list(subfolder_LQ)
            img_paths_GT_all = util.glob_file_list(subfolder_GT)
            img_paths_GT = []
            for mm in range(len(img_paths_GT_all)):
                if '.ARW' in img_paths_GT_all[mm] or 'half' in img_paths_GT_all[mm]:
                    continue
                img_paths_GT.append(img_paths_GT_all[mm])

            max_idx = len(img_paths_LQ)
            self.data_info['path_LQ'].extend(
                img_paths_LQ)  # list of path str of images
            self.data_info['path_GT'].extend(img_paths_GT)
            self.data_info['folder'].extend([subfolder_name] * max_idx)
            for i in range(max_idx):
                self.data_info['idx'].append('{}/{}'.format(i, max_idx))#[lqid/lenlq]

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

        img_LQ_path = self.imgs_LQ[folder][idx:idx + 1]
        img_GT_path = self.imgs_GT[folder][0]
        img_GT_path = [img_GT_path]

        img_LQ = util.read_img_seq2(img_LQ_path, self.train_size)
        img_LQ = img_LQ[0]
        img_GT = util.read_img_seq2(img_GT_path, self.train_size)
        img_GT = img_GT[0]

        if self.phase == 'train':

            # LQ_size = self.opt['LQ_size']
            # GT_size = self.opt['GT_size']

            # _, H, W = img_GT.shape  # real img size

            # rnd_h = random.randint(0, max(0, H - GT_size))
            # rnd_w = random.randint(0, max(0, W - GT_size))
            # img_LQ = img_LQ[:, rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size]
            # img_GT = img_GT[:, rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size]

            img_LQ_l = [img_LQ]
            img_LQ_l.append(img_GT)
            rlt = util.augment_torch(
                img_LQ_l, True, True)
            img_LQ = rlt[0]
            img_GT = rlt[1]

        # img_nf = img_LQ.clone().permute(1, 2, 0).numpy() * 255.0
        # img_nf = cv2.blur(img_nf, (5, 5))
        # img_nf = img_nf * 1.0 / 255.0
        # img_nf = torch.Tensor(img_nf).float().permute(2, 0, 1)

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


# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     import random
#     from torchvision import transforms as ts
#     from torch.utils.data import DataLoader
#     gt_root,lq_root = '../parired_datasets/smid/SMID_Long_np/','../parired_datasets/smid/SMID_LQ_np/'
#     data = Dataset_SMIDImage(gt_root,lq_root,phase='train')
#     train_loader = DataLoader(data, batch_size=4, shuffle=True, num_workers=0, sampler=None)
#     print(f'共计{int(len(data))}个输入样本')
#     import random
#     for idx in random.sample(range(len(data)),k=5):
#         lq_im = PIL.Image.fromarray(np.uint8(data[idx]['lq'].numpy().transpose(1,2,0)*255.))
#         gt_im = PIL.Image.fromarray(np.uint8(data[idx]['gt'].numpy().transpose(1, 2, 0) * 255.))
#         plt.imshow(lq_im)
#         plt.imshow(gt_im)
#         lq_im.save(f'../demo/sid_{idx}_lq.jpg')
#         gt_im.save(f'../demo/sid_{idx}_gt.jpg')
#         plt.show()




    # from tqdm import tqdm
    # for i,data in enumerate(train_loader):
    #     if i==0:
    #         img_lq = np.uint8(data['lq'].numpy().transpose(0,2,3,1)*255.)
    #         img_gt = np.uint8(data['gt'].numpy().transpose(0,2,3,1)*255.)
    #         imgs = [PIL.Image.fromarray(im) for im in img_lq]+[PIL.Image.fromarray(im) for im in img_gt]
    #         for i in range(1,len(imgs)+1):
    #             plt.subplot(2,4,i)
    #             plt.imshow(imgs[i-1])
    #         plt.show()
    #     else:
    #         break


