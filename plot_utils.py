# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: plot_utils.py
# @Author: ---
# @Institution: --- CSU&BUCEA, ---, China
# @Homepage: ---@---https://github.com/YanJieWen
# @E-mail: ---@---obitowen@csu.edu.cn
# @Site: 
# @Time: 12月 18, 2023
# ---

import numpy as np
import datetime
import matplotlib.pyplot as plt

def plot_loss_and_lr(train_loss, learning_rate):
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    try:
        x = list(range(len(train_loss)))
        fig, ax1 = plt.subplots(1, 1)
        ax1.plot(x, train_loss, 'r', label='loss')
        ax1.set_xlabel("step")
        ax1.set_ylabel("loss")
        ax1.set_title("Train Loss and lr")
        plt.legend(loc='best')

        ax2 = ax1.twinx()
        ax2.plot(x, learning_rate, label='lr')
        ax2.set_ylabel("learning rate")
        ax2.set_xlim(0, len(train_loss))  # 设置横坐标整数间隔
        plt.legend(loc='best')

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(handles1 + handles2, labels1 + labels2, loc='upper right')

        fig.subplots_adjust(right=0.8)  # 防止出现保存图片显示不全的情况
        fig.savefig('./loss_and_lr{}.png'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
        plt.close()
        print("successful save loss curve! ")
    except Exception as e:
        print(e)


def plot_val(val_metric):
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    fig,ax1 = plt.subplots()
    if isinstance(val_metric,list):
        val_metric = np.array(val_metric)#[E,2]
    _epoch = np.arange(val_metric.shape[0])
    ax1.plot(_epoch, val_metric[:,0],color='#AE2012', linestyle='--',linewidth=1.5)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("PSNR", color="#AE2012",fontsize=12)
    ax1.tick_params("y", color="#AE2012")

    ax2 = ax1.twinx()
    ax2.plot(_epoch, val_metric[:,1],  color='#0A9396', linewidth=1.5)
    ax2.set_ylabel("SSIM", color="#0A9396", fontsize=12)
    ax2.tick_params("y", color="#0A9396")
    fig.savefig('./val_metrics{}.png'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    plt.close()
    print("successful save loss curve! ")