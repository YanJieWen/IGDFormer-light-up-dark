# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: train_engine.py
# @Author: ---
# @Institution: --- CSU&BUCEA, ---, China
# @Homepage: ---@---https://github.com/YanJieWen
# @E-mail: ---@---obitowen@csu.edu.cn
# @Site: 
# @Time: 12月 14, 2023
# ---
import argparse
import os.path
import d2l.torch as d2l
import numpy as np
import torch.optim.lr_scheduler
from accelerate import Accelerator
#=====================================================
from train_utils import *
from models.losses import L1Loss
from models.custom_scheduler import CosineAnnealingRestartCyclicLR



def main(args):
    data_root = args.data_root
    batch_size =args.batch_size
    gt_size = args.gt_size
    mean = args.mean
    std = args.std
    weights_save = args.weights_save
    resume = args.resume
    att_type = args.att_type
    num_epoch = args.num_epoch
    data_type = args.data_type


    if not os.path.exists(weights_save):
        os.makedirs(weights_save)

    accelerator = Accelerator()
    device = accelerator.device
    print(f'Using {device.type} device training')
    #step1: 定义数据
    result = create_train_val_loader(data_root,mean,std,gt_size=gt_size,batch_size=batch_size,read_type=data_type)
    train_loader,val_loader = result
    # print(next(iter(val_loader))['lq_path'])

    #step2: 搭建模型
    model = create_model(40,1,[1,2,2],attn_type=att_type)
    # model = create_model(40, 1, [1, 2, 2])
    model.to(device)
    #step3: 初始化寻训练器需要训练的参数，优化器，学习率调度，损失函数
    current_metric = 0.

    for name,parm in model.named_parameters():
        parm.requires_grad = True
    parms = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parms,lr=2e-4,betas=(0.9,0.999))
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=92000,T_mult=1,eta_min=0.000001)
    total_iters = args.iters
    cycle_iter = args.cycle_iter
    per_epoch_iters = int(len(train_loader))
    num_epoch = int(np.ceil(total_iters/per_epoch_iters))
    print(f'There are total {num_epoch}.....')
    #按照iter来修改学习率
    lr_scheduler = CosineAnnealingRestartCyclicLR(optimizer,periods=[int(cycle_iter[0]), int(cycle_iter[1])],restart_weights=[1,1],eta_mins=[0.0003,0.000001],last_epoch=-1)
    loss = L1Loss()


    #是否继续学习
    start_epoch = args.start_epoch
    if resume!='':
        checkpoint = torch.load(resume,map_location='cpu')
        model = accelerator.unwrap_model(model)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
    print("the training process from epoch{}...".format(start_epoch))
    #包裹加速器
    model,optimizer,train_loader,lr_scheduler = accelerator.prepare(model,optimizer,train_loader,lr_scheduler)
    # model,optimizer,train_loader,lr_scheduler,val_loader = accelerator.prepare(model,optimizer,train_loader,lr_scheduler,val_loader)


    #开展训练
    train_loss = []
    learnin_rate = []
    val_metric = []
    f = open(f'./{att_type}.txt', 'w')
    for epoch in range(start_epoch,num_epoch):
        mloss,lr = train_one_epoch(model,optimizer,train_loader,accelerator.device,epoch,accelerator,loss,lr_scheduler,pre_feq=50,warmup=True,
                                   gt_size=args.gt_size,mini_gt_size=args.mini_gt_size)
        train_loss.append(mloss.item())
        learnin_rate.append(lr)
        # lr_scheduler.step()

        #评估模型
        avg_psnr,avg_ssim = evaluate(model,val_loader,window_size=4,acc=accelerator)
        accelerator.print(f'[{int(epoch)}/{int(num_epoch)}]-->l1loss: {np.round(mloss.item(), 4)}\t Lr: {lr}\t'
              f'PSNR:{np.round(avg_psnr,4)},SSIM:{np.round(avg_ssim,4)}')
        val_metric.append((avg_psnr,avg_ssim))
        #记录损失函数
        log_info = f'[{int(epoch)}/{int(num_epoch)}]-->l1loss: {np.round(mloss.item(), 4)}\t Lr: {lr}\t PSNR:{np.round(avg_psnr,4)},SSIM:{np.round(avg_ssim,4)}'

        f.write(log_info+'\n')
        #保存模型
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        save_files = {
            'model': unwrapped_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch}

        if avg_psnr > current_metric:
            torch.save(save_files,'./save_weights/ORFormer--{}.pth'.format(int(epoch)))
            current_metric = avg_psnr
        else:
            pass
    f.close()
    #绘图
    if len(train_loss)!=0 and len(learnin_rate)!=0:
        from plot_utils import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learnin_rate)

    if len(val_metric)!=0:
        from plot_utils import plot_val
        plot_val(val_metric)








if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_root',default='./parired_datasets/sid/',type=str,help='数据根目录')
    parser.add_argument('--data_type', default='SID', type=str, help='读取的数据类型')
    parser.add_argument('--batch_size', default=4, type=int, help='批大小')
    parser.add_argument('--gt_size', default=128, type=int, help='输入图像的尺寸大小')
    parser.add_argument('--mean', default=None, help='均值')
    parser.add_argument('--std', default=None, help='标准差')
    parser.add_argument('--weights_save', default='./save_weights/', type=str, help='存放权重的根目录')
    parser.add_argument('--resume', default='', type=str, help='继续训练的权重路径')
    parser.add_argument('--start_epoch', default=0, type=int, help='继续训练的轮数')
    parser.add_argument('--num_epoch', default=100, type=int, help='训练总轮数')
    parser.add_argument('--mini_gt_size', default=256, type=int, help='最小的尺寸')
    parser.add_argument('--iters', default=300000, type=int, help='更新的总次数')
    parser.add_argument('--cycle_iter', default=[92000,208000], type=int, help='学习率调度参数')
    parser.add_argument('--att_type', default='Mixing_attention_new', type=str,help="['IGAB','Restormer','Swin_Transformer','Mixing_attention','Mixing_attention_new']")

    args = parser.parse_args()
    print(args)

    main(args)



