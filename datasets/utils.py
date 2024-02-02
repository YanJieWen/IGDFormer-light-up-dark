'''
@File: utils.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 1月 03, 2024
@HomePage: https://github.com/YanJieWen
'''

import numpy as np
import cv2
import torch
import glob
import os

import random



def _read_img_lmdb(env, key, size):
    """read image from lmdb with key (w/ and w/o fixed size)
    size: (C, H, W) tuple"""
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    img_flat = np.frombuffer(buf, dtype=np.uint8)
    C, H, W = size
    img = img_flat.reshape(H, W, C)
    return img
def read_img2(env, path, size=None):
    """read image by cv2 or from lmdb
    return: Numpy float32, HWC, BGR, [0,1]"""
    if env is None:  # img
        img = np.load(path)
        if img is None:
            print(path)
        if size is not None:
            img = cv2.resize(img, (size[0], size[1]))
            # img = cv2.resize(img, size)
    else:
        img = _read_img_lmdb(env, path, size)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img
def read_img_seq2(path, size=None):
    """Read a sequence of images from a given folder path
    Args:
        path (list/str): list of image paths/image folder path

    Returns:
        imgs (Tensor): size (T, C, H, W), RGB, [0, 1]
    """
    # print(path)
    if type(path) is list:
        img_path_l = path
    else:
        img_path_l = sorted(glob.glob(os.path.join(path, '*')))

    img_l = [read_img2(None, v, size) for v in img_path_l]
    # stack to Torch tensor
    imgs = np.stack(img_l, axis=0)
    try:
        imgs = imgs[:, :, :, [2, 1, 0]]
    except Exception:
        import ipdb; ipdb.set_trace()
    imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(imgs, (0, 3, 1, 2)))).float()
    return imgs

def augment_torch(img_list, hflip=True, rot=True):
    """horizontal flip OR rotate (0, 90, 180, 270 degrees)"""
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    # rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = flip(img, 2)
        if vflip:
            img = flip(img, 1)
        # if rot90:
        #     # import pdb; pdb.set_trace()
        #     img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]
def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

def glob_file_list(root):
    return glob.glob(os.path.join(root,"*"))