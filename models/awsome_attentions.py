# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: awsome_attentions.py
# @Author: ---
# @Institution: --- CSU&BUCEA, ---, China
# @Homepage: ---@---https://github.com/YanJieWen
# @E-mail: ---@---obitowen@csu.edu.cn
# @Site: 
# @Time: 12月 15, 2023
# ---
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
import numbers
import numpy as np


def window_partion(x,window_size:int):#[b,h,w,c]->[N,ws,ws,c]
    B,H,W,C = x.shape
    x = x.view(B,H//window_size,window_size,W//window_size,window_size,C)# [b,h/win,win,w/win,win,c]
    windows = x.permute(0,1,3,2,4,5).contiguous().view(-1,window_size,window_size,C)
    return windows#[B*num_windows,mh,mw,c]

def window_reverse(windows,window_size,h,w):#[N,ws,ws,c]->[b,h,w,c]
    b = int(windows.shape[0]/(h*w/window_size/window_size))
    x = windows.view(b,h//window_size,w//window_size,window_size,window_size,-1)
    x = x.permute(0,1,3,2,4,5).contiguous().view(b,h,w,-1)
    return x



class IGAB(nn.Module):
    def __init__(self,dim,dim_head,heads=8,num_blocks=2):
        '''
        Illumination-Guided Attention Block:https://arxiv.org/abs/2303.06705
        Args:
            dim:
            dim_head:
            heads:
            num_blocks:
        '''
        super(IGAB,self).__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([IG_MSA(dim=dim,dim_head=dim_head,heads=heads),
                                             PreNorm(dim,FeedForward(dim=dim))]))

    def forward(self,x,illu_fea):
        '''

        Args:
            x:[b,c,h,w]
            ill_fea: [b,c,h,w]

        Returns:

        '''
        x = x.permute(0,2,3,1)#[b,h,w,c]
        for (attn,ff) in self.blocks:
            x = attn(x,illu_fea_trans=illu_fea.permute(0,2,3,1))+x
            x = ff(x)+x
        out = x.permute(0,3,1,2)
        return out

class IG_MSA(nn.Module):
    def __init__(self,dim,dim_head,heads):
        super(IG_MSA,self).__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim,dim_head*heads,bias=False)
        self.to_k = nn.Linear(dim,dim_head*heads,bias=False)
        self.to_v = nn.Linear(dim,dim_head*heads,bias=False)
        self.rescale = nn.Parameter(torch.ones(heads,1,1))#[h,1,1]
        self.proj = nn.Linear(dim_head*heads,dim,bias=True)
        self.pos_emb = nn.Sequential(nn.Conv2d(dim,dim,3,1,1,bias=False,groups=dim),nn.GELU(),
                                     nn.Conv2d(dim,dim,3,1,1,bias=False,groups=dim))
        self.dim = dim

    def forward(self,x_in,illu_fea_trans):
        '''

        Args:
            x_in: [b,h,w,c]
            illu_fea_trans:[b,h,w,c]

        Returns: [b,h,w,c]

        '''
        b,h,w,c = x_in.shape
        x = x_in.reshape(b,h*w,c)#[b,hw,c]
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        illu_attn = illu_fea_trans.flatten(1,2)#[b,hw,c]
        q,k,v,illu_attn = map(lambda t: rearrange(t,'b n (h d) -> b h n d',h=self.num_heads),
                              (q_inp,k_inp,v_inp,illu_attn))
        v = v*illu_attn

        q,k,v = map(lambda t: torch.transpose(t,-2,-1),(q,k,v))#[b,h,n,d]->[b,h,d,n]
        q,k = map(lambda t: F.normalize(t,dim=-1,p=2),(q,k))
        attn = k@q.transpose(-2,-1)#[b,h,d,d]
        attn = attn*self.rescale
        attn = attn.softmax(dim=-1)
        x = attn@v#[b,h,d,n]
        x = x.permute(0,3,1,2)#[b,hw,h,d]
        x = x.reshape(b,h*w,self.num_heads*self.dim_head)
        out_c = self.proj(x).view(b,h,w,c)
        out_p = self.pos_emb(v_inp.reshape(b,h,w,c).permute(0,3,1,2)).permute(0,2,3,1)#[b,h,w,c]
        out = out_c+out_p
        return out




class PreNorm(nn.Module):
    def __init__(self,dim,fn):
        super(PreNorm,self).__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
    def forward(self,x,*args,**kwargs):
        x = self.norm(x)
        return self.fn(x,*args,**kwargs)




class FeedForward(nn.Module):
    def __init__(self,dim,ratio=4):
        super(FeedForward,self).__init__()
        self.net = nn.Sequential(nn.Conv2d(dim,dim*ratio,1,1,bias=False),nn.GELU(),
                                 nn.Conv2d(dim*ratio,dim*ratio,3,1,1,bias=False,groups=dim*ratio),
                                 nn.GELU(),nn.Conv2d(dim*ratio,dim,1,1,bias=False))

    def forward(self,x):
        out = self.net(x.permute(0,3,1,2))
        return out.permute(0,2,3,1)#[b,h,w,c]


#============================================================

class Restormer(nn.Module):
    def __init__(self,dim,dim_head,heads=8,num_blocks=2):
        super(Restormer,self).__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
                self.blocks.append(nn.ModuleList([MDTA(dim=dim, dim_head=dim_head, heads=heads),#这个部分有两个变量传入前置不用LN
                                                  LayerNorm(dim),
                                                  FDFN(dim)]))

    def forward(self, x, illu_fea):
        '''

        Args:
            x:[b,c,h,w]
            illu_fea:[b,c,h,w]

        Returns:

        '''
        for (attn,ff,ln2) in self.blocks:
            x = x+attn(x, illu_fea)#[b,c,h,w]->[b,c,h,w]
            x = x+ff(ln2(x))#[b,c,h,w]->[b,c,h,w]
        return x
class MDTA(nn.Module):
    def __init__(self,dim,dim_head,heads):
        '''
        Restormer: https://github.com/swz30/Restormer
        Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang;
        Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 5728-5739
        '''
        super(MDTA,self).__init__()
        self.num_heads = heads
        self.temperature = nn.Parameter(torch.ones(heads,1,1))

        self.qkv = nn.Conv2d(dim,dim_head*heads*3,1,bias=False)
        self.qkv_dwconv = nn.Conv2d(dim_head*heads*3,dim_head*heads*3,3,1,1,groups=dim_head*heads*3,
                                    bias=False)
        self.project_out = nn.Conv2d(dim_head*heads,dim,1,bias=False)



    def forward(self,x_in,illu_fea_trans):
        '''

        Args:
            x_in: [b,c,h,w]
            illu_fea_trans: [b,c,h,w]

        Returns:[b,c,h,w]

        '''
        b,c,h,w = x_in.shape
        qkv = self.qkv_dwconv(self.qkv(x_in))
        q,k,v = qkv.chunk(3,dim=1)
        #[b,h,d,n]
        q,k,v,illu_fea_trans = map(lambda t:rearrange(t,'b (n d) h w -> b n d (h w)',n=self.num_heads),(q,k,v,illu_fea_trans))
        q = F.normalize(q,dim=-1)
        k = F.normalize(k,dim=-1)

        attn = (q @ k.transpose(-2,-1))*self.temperature#[b,h,d,d]
        attn = attn.softmax(dim=-1)

        v = v*illu_fea_trans

        out = attn @ v#[b,h,d,n]
        out = rearrange(out,'b n d (h w)->b (n d) h w',n=self.num_heads,h=h,w=w)
        out = self.project_out(out)#[b,c,h,w]

        return out


class FDFN(nn.Module):
    def __init__(self,dim,ratio=2.66):
        super(FDFN,self).__init__()

        hidden_features = int(dim*ratio)
        self.project_in = nn.Conv2d(dim,hidden_features*2,1,bias=False)
        self.dwconv = nn.Conv2d(hidden_features*2,hidden_features*2,kernel_size=3,stride=1,padding=1,groups=hidden_features*2,bias=False)

        self.project_out = nn.Conv2d(hidden_features,dim,1,bias=False)

    def forward(self,x):
        '''

        Args:
            x: [b,c,h,w]

        Returns: [b,c,h,w]

        '''
        x  = self.project_in(x)
        x1,x2 = self.dwconv(x).chunk(2,dim=1)
        x = F.gelu(x1)*x2
        x = self.project_out(x)
        return x

class BiasFree_LayerNorm(nn.Module):
    def __init__(self,normalized_shapes):
        super(BiasFree_LayerNorm,self).__init__()
        if isinstance(normalized_shapes,numbers.Integral):
            normalized_shapes = (normalized_shapes,)
        normalized_shapes = torch.Size(normalized_shapes)
        self.weight = nn.Parameter(torch.ones(normalized_shapes))
        self.normalized_shapes = normalized_shapes
    def forward(self,x):
        sigma = x.var(-1,keepdim=True,unbiased=False)
        return x/torch.sqrt(sigma+1e-5)*self.weight

class LayerNorm(nn.Module):
    def __init__(self,dim):
        super(LayerNorm,self).__init__()
        self.body = BiasFree_LayerNorm(dim)

    def forward(self,x):
        h,w = x.shape[-2:]
        x = rearrange(x,'b c h w -> b (h w) c')
        x = self.body(x)
        x = rearrange(x, 'b (h w) c->b c h w',h=h,w=w)
        return x

#============================================================


class Swin_Transformer(nn.Module):
    '''
    Swin-Transformer:https://arxiv.org/abs/2103.14030
    '''
    def __init__(self,dim,dim_head,heads=8,num_blocks=2,window_size=7,qkv_bias=True,drop=0.1,mlp_ratio=4,
                 attn_drops=0.1,norm_layer=nn.LayerNorm):
        super(Swin_Transformer,self).__init__()
        self.window_size = window_size
        self.shifted_size = window_size//2
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList(SwinTransformerBlock(dim=dim,dim_head=dim_head,num_heads=heads,
                                                                  window_size=window_size,mlp_ratio=mlp_ratio,
                                                                  qkv_bias=qkv_bias,drop=drop,attn_drops=attn_drops,
                                                                  norm_layer=norm_layer,
                                                                  shift_size=0 if (i%2==0) else self.shifted_size,
                                                                  ) for i in range(2)))

    def create_mask(self,x):
        '''
        仅在移位attention中被调用
        Args:
            x:[b,c,h,w]

        Returns:

        '''
        b,h,w,c = x.shape
        hp = int(np.ceil(h/self.window_size))*self.window_size
        wp = int(np.ceil(w/self.window_size))*self.window_size
        img_mask = torch.zeros((1,hp,wp,1),device=x.device)#[1,hp,wp,1]
        #
        h_slices = (slice(0,-self.window_size),slice(-self.window_size,-self.shifted_size),
                    slice(-self.shifted_size,None))
        w_slices = (slice(0,-self.window_size),slice(-self.window_size,-self.shifted_size),
                    slice(-self.shifted_size,None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:,h,w,:] = cnt
                cnt+=1
        mask_window =window_partion(img_mask,window_size=self.window_size)#[(h/s)*(w/s),s,s,c]
        mask_window = mask_window.view(-1, self.window_size * self.window_size)#[N,s*s]展平操作
        attn_mask = mask_window.unsqueeze(1) - mask_window.unsqueeze(2)#[N,1,s*s]-[N,s*s,1]=[N,s*s,s*s]
        attn_mask = attn_mask.masked_fill(attn_mask!=0,float('-inf')).masked_fill(attn_mask==0,float(0.0))


        return attn_mask

    def forward(self,x,illu_fea):
        '''

        Args:
            x: [b,c,h,w]
            illu_fea: [b,c,h,w]

        Returns:[b,c,h,w]

        '''
        #1.创建蒙版判别连续像素块
        b,c,h,w = x.shape
        attn_mask = self.create_mask(x.permute(0,2,3,1))#[N,s*s,s*s]
        for (wmsa,swmsa) in self.blocks:
            x = wmsa(x,illu_fea,attn_mask=None)
            x = swmsa(x,illu_fea,attn_mask=attn_mask)
        return x



class SwinTransformerBlock(nn.Module):
    def __init__(self,dim,dim_head,num_heads,window_size,shift_size,mlp_ratio,qkv_bias,drop,attn_drops,norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU):
        super(SwinTransformerBlock,self).__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0<=self.shift_size<self.window_size
        self.norm1 = norm_layer(dim)
        self.attn = Windowattention(dim=dim,dim_heads=dim_head,window_size=(self.window_size,self.window_size), heads=num_heads,
                                    qkv_bias=qkv_bias,attn_drop=attn_drops,proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim*mlp_ratio)
        #实现windowattention
        self.mlp = Mlp(in_features=dim,hidden_features=mlp_hidden_dim,act_layer=act_layer,drop=drop)

    def forward(self,x,illu_fea,attn_mask):
        b,c,h,w = x.shape
        x = x.permute(0, 2, 3, 1)
        short_cut = x.reshape(b,h*w,c)
        x = self.norm1(x)
        #[b,h,w,c]
        illu_fea = illu_fea.permute(0,2,3,1)#[b,h,w,c]
        pad_l=pad_t=0
        pad_r = (self.window_size-w%self.window_size)%self.window_size
        pad_b = (self.window_size-h % self.window_size) % self.window_size
        x = F.pad(x,(0,0,pad_l,pad_r,pad_t,pad_b))
        illu_fea = F.pad(illu_fea,(0,0,pad_l,pad_r,pad_t,pad_b))
        _,hp,wp,_ = x.shape
        #(b,hp,wp,c),(b,hp,wp,c)
        if self.shift_size>0:
            shifted_x = torch.roll(x,shifts=(-self.shift_size,-self.shift_size),dims=(1,2))
            shifted_illu_fea = torch.roll(illu_fea,shifts=(-self.shift_size,-self.shift_size),dims=(1,2))
        else:
            shifted_x = x
            shifted_illu_fea = illu_fea
            attn_mask = None

        #窗口划分
        x_window = window_partion(shifted_x,self.window_size)#[N,ws,ws,c]
        x_window = x_window.view(-1,self.window_size*self.window_size,c)

        illu_fea_window = window_partion(shifted_illu_fea, self.window_size)  # [N,ws,ws,c]
        illu_fea_window = illu_fea_window.view(-1, self.window_size * self.window_size, c)#[N,ws*ws,c]

        #IG-w-msa/IG-sw-msa-->[N,ws*ws,c]
        attn_windows = self.attn(x_window,illu_fea_window,attn_mask=attn_mask)#[N,ws*ws,c]

        attn_windows = attn_windows.view(-1,self.window_size,self.window_size,c)
        shifted_x = window_reverse(attn_windows,self.window_size,hp,wp)#[b,hp,wp,c]

        #逆移位
        if self.shift_size>0:
            x = torch.roll(shifted_x,(self.shift_size,self.shift_size),(1,2))
        else:
            x = shifted_x


        #由于前面进行了padding需要恢复到原始尺寸
        if pad_r>0 or pad_b>0:
            x = x[:,:h,:w,:].contiguous()

        x = x.view(b,h*w,c)
        x = short_cut+x
        x = x+self.mlp(self.norm2(x))
        x = x.view(b,h,w,c).permute(0,3,1,2)#恢复到原来的尺度
        return x






class Windowattention(nn.Module):
    def __init__(self,dim,dim_heads,heads,window_size,qkv_bias,attn_drop,proj_drop):
        super(Windowattention,self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.dim_heads = dim_heads
        self.scale = dim_heads**-0.5
        self.num_heads = heads

        #相对位置查询表[(2M-1*2M-1),h]
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2*window_size[0]-1)*(2*window_size[1]-1),heads))
        #制作相对位置成对矩阵
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        relative_flatten = torch.flatten(coords, 1)
        relative_coords = relative_flatten[:, :, None] - relative_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)  # [m2,m2]

        self.qkv = nn.Linear(dim,dim_heads*heads*3,bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_heads*heads,dim)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x_in,illu_fea,attn_mask):
        '''

        Args:
            x_in: [N,ws*ws,c]
            illu_fea: [N,ws*ws,c]
            attn_mask: [N,s*s,s*s]

        Returns:[N,ws*ws,c]

        '''
        b,l,c =x_in.shape
        qkv = self.qkv(x_in).reshape(b, l, 3, self.num_heads, self.dim_heads).permute(2, 0, 3, 1, 4)#[3,_b,h,l,d]
        q, k, v = torch.unbind(qkv, dim=0)#[_b,h,l,d]
        q = q*self.scale
        attn = (q @ k.transpose(-2,-1))#[_b,h,l,l]
        illu_attn = rearrange(illu_fea,'b l (h d)->b h l d',h=self.num_heads)
        v = v*illu_attn

        #相对位置编码
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0]*self.window_size[1],self.window_size[0]*self.window_size[1],-1)
        relative_position_bias = relative_position_bias.permute(2,0,1).contiguous().unsqueeze(0)
        attn = attn+relative_position_bias

        #sw-msa掩码不连续像素块的注意力得分
        if attn_mask is not None:
            nw = attn_mask.shape[0]
            attn = attn.view(b//nw,nw,self.num_heads,l,l)+attn_mask.unsqueeze(1).unsqueeze(0)#[1,nw,1,l,l]
            attn = attn.view(-1, self.num_heads, l, l)#[nw,h,l,l]
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x= (attn @ v).transpose(1,2).reshape(b,l,c)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x



class Mlp(nn.Module):
    def __init__(self,in_features,hidden_features,act_layer,drop):
        super(Mlp,self).__init__()
        out_features = in_features
        self.fc1 = nn.Linear(in_features,hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features,in_features)
        self.drop2 = nn.Dropout(drop)
    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


#==================================================================================
#SE-NET
class SE_net(nn.Module):
    def __init__(self,in_channels,ratio):
        super(SE_net,self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1,1))
        self.compress = nn.Conv2d(in_channels,in_channels//ratio,1,1,0)
        self.excitation = nn.Conv2d(in_channels//ratio,in_channels,1,1,0)

    def forward(self,x):
        short_cut = x
        x = self.squeeze(x)
        x = self.compress(x)
        x = F.relu(x)
        x = self.excitation(x)
        x = F.sigmoid(x)
        x = x*short_cut
        return x
#==================================================================================
class Mixing_attention(nn.Module):
    def __init__(self,dim,dim_head,heads=8,num_blocks=2,window_size=7,qkv_bias=True,drop=0.1,mlp_ratio=4,
                 attn_drops=0.1,norm_layer=nn.LayerNorm):
        super(Mixing_attention,self).__init__()
        self.window_size = window_size
        self.shifted_size = window_size // 2
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([IG_MSA(dim=dim,dim_head=dim_head,heads=heads),PreNorm(dim,FeedForward(dim=dim)),
                                              SwinTransformerBlock(dim=dim, dim_head=dim_head, num_heads=heads,
                                                                  window_size=window_size, mlp_ratio=mlp_ratio,
                                                                  qkv_bias=qkv_bias, drop=drop, attn_drops=attn_drops,
                                                                  norm_layer=norm_layer,
                                                                  shift_size=0),
                                              SwinTransformerBlock(dim=dim, dim_head=dim_head, num_heads=heads,
                                                                   window_size=window_size, mlp_ratio=mlp_ratio,
                                                                   qkv_bias=qkv_bias, drop=drop, attn_drops=attn_drops,
                                                                   norm_layer=norm_layer,
                                                                   shift_size=self.shifted_size)
                                              ]))

    def create_mask(self, x):
        '''
        仅在移位attention中被调用
        Args:
            x:[b,c,h,w]

        Returns:

        '''
        b, h, w, c  = x.shape
        hp = int(np.ceil(h / self.window_size)) * self.window_size
        wp = int(np.ceil(w / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, hp, wp, 1), device=x.device)  # [1,hp,wp,1]
        #
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shifted_size),
                    slice(-self.shifted_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shifted_size),
                    slice(-self.shifted_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_window = window_partion(img_mask, window_size=self.window_size)  # [(h/s)*(w/s),s,s,c]
        mask_window = mask_window.view(-1, self.window_size * self.window_size)  # [N,s*s]展平操作
        attn_mask = mask_window.unsqueeze(1) - mask_window.unsqueeze(2)  # [N,1,s*s]-[N,s*s,1]=[N,s*s,s*s]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float('-inf')).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, illu_fea):
        '''

        Args:
            x: [b,c,h,w]
            illu_fea: [b,c,h,w]

        Returns:[b,c,h,w]

        '''
        # 1.创建蒙版判别连续像素块
        b, c, h, w = x.shape
        attn_mask = self.create_mask(x.permute(0, 2, 3, 1))  # [N,s*s,s*s]
        for (igab,ff,wmsa, swmsa) in self.blocks:
            x = x.permute(0,2,3,1)#[b,h,w,c]
            x = igab(x, illu_fea_trans=illu_fea.permute(0, 2, 3, 1)) + x
            x = ff(x) + x
            x = x.permute(0,3,1,2)#[b,c,h,w]
            x = wmsa(x, illu_fea, attn_mask=None)
            x = swmsa(x, illu_fea, attn_mask=attn_mask)
        return x

#======================================================================================================================
class Mixing_attention_new(nn.Module):
    def __init__(self,dim,dim_head,heads=8,num_blocks=2,window_size=7,qkv_bias=True,drop=0.1,mlp_ratio=4,
                 attn_drops=0.1,norm_layer=nn.LayerNorm):
        super(Mixing_attention_new,self).__init__()
        self.window_size = window_size
        self.shifted_size = window_size // 2
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([MDTA(dim=dim,dim_head=dim_head,heads=heads),LayerNorm(dim),FDFN(dim),
                                              SwinTransformerBlock(dim=dim, dim_head=dim_head, num_heads=heads,
                                                                  window_size=window_size, mlp_ratio=mlp_ratio,
                                                                  qkv_bias=qkv_bias, drop=drop, attn_drops=attn_drops,
                                                                  norm_layer=norm_layer,
                                                                  shift_size=0),
                                              SwinTransformerBlock(dim=dim, dim_head=dim_head, num_heads=heads,
                                                                   window_size=window_size, mlp_ratio=mlp_ratio,
                                                                   qkv_bias=qkv_bias, drop=drop, attn_drops=attn_drops,
                                                                   norm_layer=norm_layer,
                                                                   shift_size=self.shifted_size)
                                              ]))

    def create_mask(self, x):
        '''
        仅在移位attention中被调用
        Args:
            x:[b,c,h,w]

        Returns: [N,w*w,w*w]

        '''
        b, h, w, c  = x.shape
        hp = int(np.ceil(h / self.window_size)) * self.window_size
        wp = int(np.ceil(w / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, hp, wp, 1), device=x.device)  # [1,hp,wp,1]
        #
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shifted_size),
                    slice(-self.shifted_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shifted_size),
                    slice(-self.shifted_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_window = window_partion(img_mask, window_size=self.window_size)  # [(h/s)*(w/s),s,s,c]
        mask_window = mask_window.view(-1, self.window_size * self.window_size)  # [N,s*s]展平操作
        attn_mask = mask_window.unsqueeze(1) - mask_window.unsqueeze(2)  # [N,1,s*s]-[N,s*s,1]=[N,s*s,s*s]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float('-inf')).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, illu_fea):
        '''

        Args:
            x: [b,c,h,w]
            illu_fea: [b,c,h,w]

        Returns:[b,c,h,w]

        '''
        # 1.创建蒙版判别连续像素块
        b, c, h, w = x.shape
        attn_mask = self.create_mask(x.permute(0, 2, 3, 1))  # [N,s*s,s*s]
        for (mdtn,ln,ff,wmsa, swmsa) in self.blocks:
            x = x+mdtn(x,illu_fea)
            x = x+ff(ln(x))
            x = wmsa(x, illu_fea, attn_mask=None)
            x = swmsa(x, illu_fea, attn_mask=attn_mask)
        return x