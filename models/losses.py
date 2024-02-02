# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: losses.py
# @Author: ---
# @Institution: --- CSU&BUCEA, ---, China
# @Homepage: ---@---https://github.com/YanJieWen
# @E-mail: ---@---obitowen@csu.edu.cn
# @Site: 
# @Time: 12月 17, 2023
# ---

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class L1Loss(nn.Module):
    def __init__(self,loss_weight=1.0,reduction='mean'):
        super(L1Loss,self).__init__()
        if reduction not in ['none','mean','sum']:
            raise ValueError('Unsupported reduction mode!')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self,pred,tgt,weight=None,**kwargs):
        return self.loss_weight*F.l1_loss(pred,tgt,weight,reduction=self.reduction)

class MSELoss(nn.Module):
    def __init__(self,loss_weight=1.0,reduction='mean'):
        super(MSELoss,self).__init__()
        if reduction not in ['none','mean','sum']:
            raise ValueError('Unsupported reduction mode!')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self,pred,tgt,weight=None,**kwargs):
        return self.loss_weight*F.mse_loss(pred,tgt,weight,reduction=self.reduction)



class PSNRLoss(nn.Module):
    def __init__(self,loss_weight=1.0,reduction='mean',toY=False):
        super(PSNRLoss,self).__init__()
        assert reduction=='mean'

        self.loss_weight = loss_weight
        self.scale = 10/np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481,128.553,24.966]).reshape(1,3,1,1)
        self.first = True

    def forward(self,pred,tgt):
        assert len(pred.size())==4
        if self.toY:
            if self.first:
                self.coef  = self.coef.to(pred.device)
                self.first=False
            pred = (pred*self.coef).sum(dim=1).unsqueeze(dim=1)+16.
            tgt = (tgt * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred,tgt = pred/255.,tgt/255.
            pass
        assert len(pred.size())==4

        return self.loss_weight*self.scale*torch.log(((pred-tgt)**2).mean(dim=(1,2,3))+13-8).mean()