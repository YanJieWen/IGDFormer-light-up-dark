# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: RetinexFormer_arch.py
# @Author: ---
# @Institution: --- CSU&BUCEA, ---, China
# @Homepage: ---@---https://github.com/YanJieWen
# @E-mail: ---@---obitowen@csu.edu.cn
# @Site: 
# @Time: 12月 14, 2023
# ---

import torch
import torch.nn as nn
from models import awsome_attentions


#光照估计
class Illumination_Estimator(nn.Module):
    def __init__(self,n_fea_middel,n_fea_in=4,n_fea_out =3):
        super(Illumination_Estimator,self).__init__()
        self.conv1 = nn.Conv2d(n_fea_in,n_fea_middel,kernel_size=1,bias=True)
        self.depth_conv = nn.Conv2d(n_fea_middel,n_fea_middel,kernel_size=5,padding=2,bias=True,groups=n_fea_in)
        self.conv2 = nn.Conv2d(n_fea_middel,n_fea_out,kernel_size=1,bias=True)
    def forward(self,img):
        mean_c = img.mean(dim=1).unsqueeze(1)#[b,1,h,w]
        input = torch.cat([img,mean_c],dim=1)#[b,4,h,w]
        # print(input.shape,mean_c.shape)
        x_1 = self.conv1(input)#[b,c,h,w]
        illu_fea = self.depth_conv(x_1)#[b,c,h,w]
        illu_map = self.conv2(illu_fea)#[b,3,h,w]
        return illu_fea,illu_map
#去噪器->Transformer-based
class Denoiser(nn.Module):
    def __init__(self,in_dim=3,out_dim=3,dim=31,level=2,num_blocks=[2,4,4],att_type='IGAB'):
        super(Denoiser,self).__init__()
        self.dim = dim
        self.level = level
        self.att_type = att_type
        self.embedding = nn.Conv2d(in_dim,self.dim,3,1,1,bias=False)
        attn = getattr(awsome_attentions,att_type)

        #encoder blocks
        dim_level = dim
        self.encoder_layers = nn.ModuleList([])
        for i in range(level):
            self.encoder_layers.append(nn.ModuleList([attn(dim=dim_level,num_blocks=num_blocks[i],dim_head=dim,heads=dim_level//dim),
                                       nn.Conv2d(dim_level,dim_level*2,4,2,1,bias=False),
                                       nn.Conv2d(dim_level,dim_level*2,4,2,1,bias=False)]))
            dim_level*=2
        #瓶颈层
        self.bottleneck = attn(dim=dim_level,dim_head=dim,heads=dim_level//dim,num_blocks=num_blocks[-1])

        #decoder blocks
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            # if self.att_type == 'Swin_Transformer':
            #     se_net = getattr(awsome_attentions, 'SE_net')
            # else:
            #     se_net = nn.Identity
            self.decoder_layers.append(nn.ModuleList([nn.ConvTranspose2d(dim_level,dim_level//2,stride=2,kernel_size=2,padding=0,output_padding=0),
                                                      nn.Conv2d(dim_level,dim_level//2,1,1,bias=False),
                                                      # se_net(in_channels=dim_level, ratio=16),
                                                      attn(dim=dim_level//2,num_blocks=num_blocks[level-1-i],dim_head=dim,heads=(dim_level//2)//dim)]))
            dim_level//=2
        #output_head
        self.mapping = nn.Conv2d(self.dim,out_dim,3,1,1,bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1,inplace=True)
        self.apply(self._init_weights)
    def _init_weights(self,m):
        if isinstance(m,nn.Linear):
            nn.init.trunc_normal_(m.weight,std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias,0)
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.weight,1.0)
            nn.init.constant_(m.bias,0)


    def forward(self,x,illu_fea):
        '''

        Args:
            x: Tensor[b,3,h,w]
            illu_fea:Tensor[b,c,h,w]

        Returns:[b,c,h,w]

        '''
        fea = self.embedding(x)#[b,c,h,w]

        #encoder
        fea_encoder = []
        illu_fea_list = []
        for (igab,feadonwsample,illufeadownsample) in self.encoder_layers:
            fea = igab(fea,illu_fea)
            illu_fea_list.append(illu_fea)
            fea_encoder.append(fea)
            fea = feadonwsample(fea)
            illu_fea = illufeadownsample(illu_fea)
        fea = self.bottleneck(fea,illu_fea)

        #decoder
        for i,(feaupsample,fution,lewinblock) in enumerate(self.decoder_layers):
            fea = feaupsample(fea)
            # fea = fution(se_net(torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1)))
            fea = fution(torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            illu_fea = illu_fea_list[self.level-1-i]
            fea = lewinblock(fea,illu_fea)

        out = self.mapping(fea)+x
        return out



class RetinexFormer_Single_Stage(nn.Module):
    def __init__(self,in_channels,out_channels,n_feat,level,num_blocks,att_type='IGAB'):
        super(RetinexFormer_Single_Stage,self).__init__()
        self.estimator = Illumination_Estimator(n_feat)#光照估计器->增强图像和增强特征
        self.denoiser = Denoiser(in_dim=in_channels,out_dim=out_channels,dim=n_feat,level=level,num_blocks=num_blocks,att_type=att_type)

    def forward(self,img):
        illu_fea,illu_map = self.estimator(img)#(3,h,w)->(c,h,w),(3,h,w)
        input_img = img*illu_map+img#这一部分实质上也是恢复器的一部分操作
        output_img = self.denoiser(input_img,illu_fea)#去噪恢复器
        return output_img

class RetinexFormer(nn.Module):
    def __init__(self,in_channels=3,out_channels=3,n_feat=31,stage=3,num_blocks=[1,1,1],att_type='IGAB'):
        '''

        Args:
            in_channels: 3
            out_channels: 3
            n_feat: 40
            stage: 3
            num_blocks: [1,2,2]
            att_type: [IGAB,Restormer,Swin_Transformer]
        '''
        super(RetinexFormer,self).__init__()
        self.stage = stage
        modules_body =[RetinexFormer_Single_Stage(in_channels=in_channels, out_channels=out_channels,
                                                  n_feat=n_feat, level=2, num_blocks=num_blocks,att_type=att_type)
                        for _ in range(stage)]
        self.body = nn.Sequential(*modules_body)


    def forward(self,x):
        out = self.body(x)

        return out


#
# if __name__ == '__main__':
#     from thop import profile
#     from thop import clever_format
#     model = RetinexFormer(stage=1,n_feat=40,num_blocks=[1,2,2],att_type='Mixing_attention_new').cuda()
#     x = torch.randn((1,3,256,256)).cuda()
#     # for name, param in model.named_parameters():
#     #     if param.grad is None:
#     #         print(name)
#     flops, params = profile(model, inputs=(x,))
#     flops_, params_ = clever_format([flops, params], "%.3f")
#     print(f'GFLOPS-->{flops_}\t Params-->{params_}')
