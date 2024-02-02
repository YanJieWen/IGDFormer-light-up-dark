# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: demo.py
# @Author: ---
# @Institution: --- CSU&BUCEA, ---, China
# @Homepage: ---@---https://github.com/YanJieWen
# @E-mail: ---@---obitowen@csu.edu.cn
# @Site: 
# @Time: 12月 18, 2023
# ---

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as ts
import cv2


from train_utils import create_model
min_max = (0,1)
model = create_model(40,1,[1,2,2],attn_type='Mixing_attention_new')
# model = create_model(40,1,[1,2,2],attn_type='IGAB')
# print(torch.load('./save_weights/LOL_v1.pth'))
model.load_state_dict(torch.load('./save_weights/restore+swin-all/ORFormer--170.pth',map_location='cpu')['model'])
# model.load_state_dict(torch.load('./save_weights/SID.pth',map_location='cpu')["params"])
with torch.no_grad():
    model.eval()
    demo_img = Image.open('./demo/00000684.jpg').convert('RGB')
    print(demo_img.size)
    demo_tensor = torch.as_tensor((np.array(demo_img)/255).astype(np.float32).transpose(2,0,1)[None,:,:,:])
    pred = model(demo_tensor)
    _tensor =  pred.squeeze(0).float().detach().cpu().clamp_(*min_max)
    _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])
    img_np = _tensor.numpy()
    img_np = img_np.transpose(1, 2, 0)
    img_np = (img_np * 255.0).round()
    img_np = img_np.astype(np.uint8)
    pred_img = Image.fromarray(img_np)
    pred_img.save('results.png')
    plt.imshow(pred_img)
    plt.show()


