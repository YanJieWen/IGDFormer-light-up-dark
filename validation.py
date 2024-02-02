'''
@File: validation.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 1月 06, 2024
@HomePage: https://github.com/YanJieWen
'''

import torch


import d2l.torch as d2l
import os
from functools import partial
import matplotlib.pyplot as plt
import cv2

from train_utils import *
from models import metrics

model = create_model(40,1,[1,2,2],attn_type='Mixing_attention_new',if_pretrained=True,
                     weights_root='./save_weights/restore+swin-sid/ORFormer--381.pth')
result = create_train_val_loader('./parired_datasets/sid/',mean=None,std=None,gt_size=[960,512],batch_size=1,read_type='SID')
_,val_loader = result
test = partial(pad_test, window_size=4)
metrictor = d2l.Accumulator(3)
metrictor.reset()
min_max = (0,1)
device = d2l.try_gpu()
model.to(device)
model.eval()
cnt = 0
with torch.no_grad():
    for idx, val_data in enumerate(val_loader):
        data_lq = val_data['lq'].to(device)
        data_gt = val_data['gt'].to(device)
        pred = test(lq=data_lq, model=model)
        sr_img = tensor2img(pred)
        gt_img = tensor2img(data_gt)
        lq_img = tensor2img(data_lq)
        psnr_func = getattr(metrics, 'calculate_psnr')
        ssim_func = getattr(metrics, 'calculate_ssim')
        psnr_metrics = psnr_func(sr_img, gt_img)
        ssim_metrics = ssim_func(sr_img, gt_img)
        metrictor.add(psnr_metrics, ssim_metrics, 1)
        if psnr_metrics>21 and ssim_metrics>0.65:#保存一部分实例
            if cnt <= 30:
                if "idx" in val_data.keys():
                    file_name = val_data['idx'][0].replace('/','_')
                else:
                    file_name = cnt
                cv2.imwrite(os.path.join('./demo/sid/pred',f'pred_{file_name}_idx.jpg'),sr_img)
                cv2.imwrite(os.path.join('./demo/sid/gt', f'gt_{file_name}_idx.jpg'), gt_img)
                cv2.imwrite(os.path.join('./demo/sid/lq', f'lq_{file_name}_idx.jpg'), lq_img)
                cnt+=1
            else:
                continue
        if (idx+1)%100==0:
            print(f'[{idx}]: PSNR->{metrictor[0]/metrictor[-1]};SSIM-->{metrictor[1]/metrictor[-1]}')

print(f'PSNR->{metrictor[0]/metrictor[-1]};SSIM-->{metrictor[1]/metrictor[-1]}')