# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: train_utils.py
# @Author: ---
# @Institution: --- CSU&BUCEA, ---, China
# @Homepage: ---@---https://github.com/YanJieWen
# @E-mail: ---@---obitowen@csu.edu.cn
# @Site: 
# @Time: 12月 18, 2023
# ---

import importlib
import math
import random

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
from functools import partial
import torch.nn.functional as F
import sys
from typing import List
import numpy as np
import cv2
import d2l.torch as d2l
import os

# import datasets as ds
# from datasets.my_datasets import Dataset_PariedImage
# from datasets.smid_datasets import Dataset_SMIDImage

from models.image_restoration_model import Mixing_Augment
from models.RetinexFormer_arch import RetinexFormer
from models import metrics
# from models.original_retinex import RetinexFormer

dataset_filenames = [os.path.splitext(file)[0] for file in os.listdir('./datasets/') if file.endswith('_datasets.py')]
_dataset_modules = [
    importlib.import_module(f'datasets.{file_name}')
    for file_name in dataset_filenames
]

def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        """根据step数返回一个学习率倍率因子"""
        if x >= warmup_iters:  # 当迭代数大于给定的warmup_iters时，倍率因子为1
            return 1
        alpha = float(x) / warmup_iters
        # 迭代过程中倍率因子从warmup_factor -> 1
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
def create_train_val_loader(data_root:str=None,mean:list=None,std:list=None,
                            data_type:List[str]=['Train','Test'],gt_size:int=128,batch_size:int=8,read_type:str="LOLV1"):
    if read_type in ['LOLV1','FiveK','LOLV2','all']:
        for m in _dataset_modules:
            data_cont = getattr(m,'Dataset_PariedImage',None)
            if data_cont is not None:
                break
        if data_cont is None:
            raise ValueError('Dataset_PariedImage is not found!')
        train_set = data_cont(data_root,mean,std,gt_size=gt_size,data_type=data_type[0],geometric_augs=True)
        val_set = data_cont(data_root,mean,std,gt_size=gt_size,data_type=data_type[1],geometric_augs=False)
        train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=0,sampler=None)
        val_loader = DataLoader(val_set,batch_size=1,shuffle=False,num_workers=0,sampler=None)
    elif read_type=='SMID':
        for m in _dataset_modules:
            data_cont = getattr(m,'Dataset_SMIDImage',None)
            if data_cont is not None:
                break
        if data_cont is None:
            raise ValueError('Dataset_SMIDImage is not found!')
        gt_root = os.path.join(data_root,'SMID_Long_np/')
        lq_root = os.path.join(data_root,'SMID_LQ_np/')
        train_set = data_cont(gt_root=gt_root,lq_root=lq_root,phase='train')
        val_set = data_cont(gt_root=gt_root,lq_root=lq_root,phase='val')
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, sampler=None)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=True, num_workers=0, sampler=None)
    elif read_type=='SID':
        for m in _dataset_modules:
            data_cont = getattr(m,'Dataset_SIDImage',None)
            if data_cont is not None:
                break
        if data_cont is None:
            raise ValueError('Dataset_SIDImage is not found!')
        gt_root = os.path.join(data_root, 'long_sid2/')
        lq_root = os.path.join(data_root, 'short_sid2/')
        train_set = data_cont(gt_root=gt_root, lq_root=lq_root, phase='train')
        val_set = data_cont(gt_root=gt_root, lq_root=lq_root, phase='val')
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, sampler=None)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0, sampler=None)
    else:
        raise ValueError(f'{read_type} is not exsist!')

    return train_loader,val_loader


def create_model(n_feat,stage,num_blocks,attn_type='Restormer',if_pretrained=False,weights_root=None):
    assert attn_type in ['IGAB','Restormer','Swin_Transformer','Mixing_attention','Mixing_attention_new'], print(f'{attn_type} is not in all attention styles!')
    model = RetinexFormer(n_feat=n_feat,stage=stage,num_blocks=num_blocks,att_type=attn_type)
    # model = RetinexFormer(n_feat=n_feat, stage=stage, num_blocks=num_blocks)

    if if_pretrained and weights_root is not None:
        model.load_state_dict(torch.load(weights_root,map_location='cpu')['model'],strict=True)
    else:
        pass
    return model


def train_one_epoch(model,optimizer,data_loader,device,epoch,accelerator,loss_fun,lr_scheduler,pre_feq,warmup=False,
                    mini_gt_size=256,gt_size=512):
    model.train()
    mixsing_agumentation = Mixing_Augment(mixup_beta=1.2, use_identity=True, device = accelerator.device)
    _lr_scheduler = None
    if epoch==0 and warmup:
        warmup_factor = 1.0/1000
        warmup_iters = min(1000, len(data_loader) - 1)
        _lr_scheduler = warmup_lr_scheduler(optimizer,warmup_iters,warmup_factor)
        _lr_scheduler = accelerator.prepare(lr_scheduler)
    mloss = torch.zeros(1).to(device)
    for i, datas in enumerate(data_loader):
        #添加一个随机的crop
        lq = datas['lq'].to(device)
        gt = datas['gt'].to(device)
        if mini_gt_size<gt_size:
            x0 = int((gt_size-mini_gt_size)*random.random())
            y0 = int((gt_size-mini_gt_size)*random.random())
            x1 = x0+mini_gt_size
            y1 = y0+mini_gt_size
            lq = lq[:,:,x0:x1,y0:y1]
            gt = gt[:,:,x0:x1,y0:y1]
        gt,lq = mixsing_agumentation(gt,lq)
        preds = model(lq)
        loss = loss_fun(preds,gt)
        loss_value = loss.item()
        mloss = (mloss * i + loss_value) / (i + 1)#更新平均损失
        if (i+1)%pre_feq==0:
            print(f'{i+1}/{epoch}-->current L1loss is {np.round(mloss.item(),3)}-->current lr is {optimizer.param_groups[0]["lr"]}')
        if not math.isfinite(loss_value):  # 当计算的损失为无穷大时停止训练
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        optimizer.zero_grad()
        accelerator.backward(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
        optimizer.step()
        if _lr_scheduler is not None:
            _lr_scheduler.step()
        else:
            lr_scheduler.step()
        now_lr = optimizer.param_groups[0]["lr"]
    return mloss, now_lr


def pad_test(lq,model,window_size):
    mod_pad_h,mod_pad_w = 0,0
    _,_,h,w = lq.size()
    if h%window_size!=0:
        mod_pad_h = window_size-h%window_size
    if w%window_size!=0:
        mod_pad_w = window_size-w%window_size
    img = F.pad(lq,(0,mod_pad_w,0,mod_pad_h),mode='reflect')
    model.eval()
    with torch.no_grad():
        pred = model(img)
    output = pred
    _,_,h,w = output.size()
    return output[:,:,0:h-mod_pad_h,0:w-mod_pad_w]



def tensor2img(tensor,rgb2bgr=True,out_type=np.uint8,min_max=(0,1)):
    '''

    Args:
        tensor: [1,c,h,w]
        rgb2bgr: bool
        out_type: np.uint8
        min_max: [0,1]

    Returns:[h,w,c]->bgr->numpy

    '''
    _tensor = tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
    _tensor = (_tensor-min_max[0])/(min_max[1]-min_max[0])
    img_np = _tensor.numpy()
    img_np = img_np.transpose(1,2,0)
    if rgb2bgr:
        img_np = cv2.cvtColor(img_np,cv2.COLOR_RGB2BGR)
    if out_type==np.uint8:
        img_np = (img_np*255.).round()
    result = img_np.astype(out_type)
    return result

def evaluate(model,data_loader,window_size,acc=None):
    test = partial(pad_test,window_size=window_size)
    metrictor = d2l.Accumulator(3)
    metrictor.reset()
    for idx,val_data in enumerate(data_loader):
        data_lq = val_data['lq']
        data_gt = val_data['gt']
        pred = test(lq=data_lq,model=model)
        sr_img = tensor2img(pred)
        gt_img = tensor2img(data_gt)
        sr_img = acc.gather_for_metrics(sr_img)
        gt_img = acc.gather_for_metrics(gt_img)
        psnr_func = getattr(metrics,'calculate_psnr')
        ssim_func = getattr(metrics,'calculate_ssim')
        psnr_metrics = psnr_func(sr_img,gt_img)
        ssim_metrics = ssim_func(sr_img,gt_img)
        metrictor.add(psnr_metrics,ssim_metrics,1)
        if idx>1500:
            break
    return metrictor[0]/metrictor[-1],metrictor[1]/metrictor[-1]

