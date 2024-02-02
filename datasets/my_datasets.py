# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: my_datasets.py
# @Author: ---
# @Institution: --- CSU&BUCEA, ---, China
# @Homepage: ---@---https://github.com/YanJieWen
# @E-mail: ---@---obitowen@csu.edu.cn
# @Site: 
# @Time: 12月 14, 2023
# ---

import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import os

from datasets.gen_paried_path import paired_paths_from_folder
from datasets.transforms import *

class Dataset_PariedImage(Dataset):
    
    def __init__(self,data_root,mean,std,gt_size=128,data_type='Train',geometric_augs=True):
        super(Dataset_PariedImage,self).__init__()
        self.mean = mean if not None else None
        self.std = std if not None else None
        dir_name = ['input','target']
        base_dir = os.path.join(data_root,data_type)
        self.gt_folder,self.lq_folder = os.path.join(base_dir,dir_name[1]),os.path.join(base_dir,dir_name[0])
        self.filename_tmpl = '{}'

        self.paths = paired_paths_from_folder([self.lq_folder,self.gt_folder],['lq','gt'],self.filename_tmpl)
        self.geometric_augs =geometric_augs
        self.data_type = data_type
        self.gt_size = gt_size


    def __getitem__(self, idx):
        index = idx%len(self.paths)
        gt_path = self.paths[index]['gt_path']
        try:
            # img_gt= np.array(Image.open(gt_path).convert('RGB')).astype(np.float32)/255.
            #用PIL读取图片不包含图片旋转信息需要exitif手动旋转，cv2能自动旋转图片，还是靠得住。。。
            img_gt = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        except Exception as e:
            print("gt path {} not working".format(gt_path))
        lq_path = self.paths[index]['lq_path']
        # img_lq = np.array(Image.open(lq_path).convert('RGB')).astype(np.float32)/255.
        img_lq = cv2.cvtColor(cv2.imread(lq_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        if self.data_type=='Train':
            img_gt, img_lq = padding(img_gt, img_lq , self.gt_size)
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, self.gt_size, 1,
                                                gt_path)
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)
        img_gt,img_lq = F.to_tensor(img_gt),F.to_tensor(img_lq)
        if self.mean is not None and self.std is not None:
            img_gt,img_lq = normalize(img_gt,self.mean,self.std),normalize(img_lq,self.mean,self.std)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }


        return None


    def __len__(self):
        return len(self.paths)





# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     import random
#     from torchvision import transforms as ts
#     from torch.utils.data import DataLoader
#     root = '../parired_datasets/LOLv1'
#     data = Dataset_PariedImage(root,None,None)
#     train_loader = DataLoader(data,batch_size=4,shuffle=True,num_workers=0,sampler=None)
#     print(next(iter(train_loader))['lq'].shape)
#     for idx in random.sample(range(0,len(data)),k=5):
#         per_data = data[idx]
#         lq = per_data['lq']
#         gt = per_data['gt']
#         lq = ts.ToPILImage()(lq)
#         gt = ts.ToPILImage()(gt)
#         imgs = [lq, gt]
#         for i in range(1,3):
#             plt.subplot(1,2,i)
#             plt.imshow(imgs[i-1])
#         plt.show()