# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: base_model.py
# @Author: ---
# @Institution: --- CSU&BUCEA, ---, China
# @Homepage: ---@---https://github.com/YanJieWen
# @E-mail: ---@---obitowen@csu.edu.cn
# @Site: 
# @Time: 12月 14, 2023
# ---

import torch
import random

from models.RetinexFormer_arch import RetinexFormer



class Mixing_Augment:
    #按照一定比例打乱并且混合样本https://arxiv.org/abs/1710.09412
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(
            torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1, 1)).item()

        r_index = torch.randperm(target.size(0)).to(self.device)

        target = lam * target + (1 - lam) * target[r_index, :]#打乱为什么要混合？
        input_ = lam * input_ + (1 - lam) * input_[r_index, :]

        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments) - 1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_



# class ImageCleanModel(BaseModel):
#     def __init__(self,mixup:str=False,
#                  device:str=None,
#                  ):
#         super(ImageCleanModel,self).__init__()
#         if mixup:
#             self.mixup_aug =Mixing_Augment(1.2,use_identity=False,device=device)
#         self.net_g = RetinexFormer()
