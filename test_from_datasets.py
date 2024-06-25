import sys

from train_utils import create_train_val_loader,create_model
import torchvision.transforms as tts
from train_utils import tensor2img
from models import metrics

import cv2
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
import sys
import d2l.torch as d2l

import pandas as pd

data_root = './data/sid/'
mean = None
std = None
batch_size = 1
data_type = 'SID'

att_type = 'Mixing_attention_new'
device = d2l.try_gpu()
factor = 4
weights_dir = './weights/sid.pth'

result_dir = f'./results/{data_type}/pred/'
result_dir_input = f'./results/{data_type}/lq/'
result_dir_gt = f'./results/{data_type}/gt/'


result = create_train_val_loader(data_root,mean,std,batch_size=batch_size,
                                 read_type=data_type)
train_loader,val_loader = result
model = create_model(40,1,[1,2,2],attn_type=att_type)
model.load_state_dict(torch.load(weights_dir,map_location='cpu')['model'],strict=True)

model.to(device)
psnr = []
ssim = []
metric_dict = {}
with torch.no_grad():
    model.eval()
    cnt = 0
    for data_batch in tqdm(val_loader,file=sys.stdout):
        input = data_batch['lq']
        #将输出图像padding到4的倍率
        h,w = input.shape[2],input.shape[3]
        h_,w_ = ((h + factor) // factor) * \
                factor, ((w + factor) // factor) * factor
        padh = h_ - h if h % factor != 0 else 0
        padw = w_ - w if w % factor != 0 else 0
        input_ = F.pad(input, (0, padw, 0, padh), 'reflect')
        restored = model(input_.to(device))

        restored = restored[:, :, :h, :w]
        psnr_func = getattr(metrics, 'calculate_psnr')
        ssim_func = getattr(metrics, 'calculate_ssim')
        pred = tensor2img(restored)
        target = tensor2img(data_batch['gt'])
        lq = tensor2img(input)
        psnr.append(psnr_func(pred,target))
        ssim.append(ssim_func(pred,target))
        str_cnt = str(cnt).zfill(5)
        os.makedirs(result_dir,exist_ok=True)
        os.makedirs(result_dir_input,exist_ok=True)
        os.makedirs(result_dir_gt,exist_ok=True)
        cv2.imwrite(os.path.join(result_dir,f'{str_cnt}.png'),pred)
        cv2.imwrite(os.path.join(result_dir_input,f'{str_cnt}.png'),lq)
        cv2.imwrite(os.path.join(result_dir_gt, f'{str_cnt}.png'), target)
        cnt += 1
        if cnt>500:
            break
    metric_dict = {'psnrs':psnr,'ssim':ssim}
    df = pd.DataFrame(metric_dict)
    df.to_excel(os.path.join('./results/',f'{data_type}.xlsx'))


