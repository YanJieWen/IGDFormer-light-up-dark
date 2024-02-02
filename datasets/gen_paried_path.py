# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: gen_paried_path.py
# @Author: ---
# @Institution: --- CSU&BUCEA, ---, China
# @Homepage: ---@---https://github.com/YanJieWen
# @E-mail: ---@---obitowen@csu.edu.cn
# @Site: 
# @Time: 12月 14, 2023
# ---

import os

def paired_paths_from_folder(folders, keys, filename_tmpl):
    '''
    将成对输入转为List[Dict]
    Args:
        folders: 成对图像路径
        keys: lq gt
        filename_tmpl: {}

    Returns: List[Dict]

    '''
    assert len(folders) == 2, (
        'The len of folders should be 2 with [input_folder, gt_folder]. '
        f'But got {len(folders)}')
    assert len(keys) == 2, (
        'The len of keys should be 2 with [input_key, gt_key]. '
        f'But got {len(keys)}')
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    input_paths = list(os.listdir(input_folder))
    gt_paths = list(os.listdir(gt_folder))
    assert len(input_paths) == len(gt_paths), (
        f'{input_key} and {gt_key} datasets have different number of images: '
        f'{len(input_paths)}, {len(gt_paths)}.')

    paths = []

    for idx in range(len(gt_paths)):
        gt_path = gt_paths[idx]
        basename, ext = os.path.splitext(gt_path)#返回gt的文件名和扩展名
        input_path = input_paths[idx]
        basename_input, ext_input = os.path.splitext(os.path.basename(input_path))#返回lq
        input_name = f'{filename_tmpl.format(basename)}{ext_input}'
        input_path = os.path.join(input_folder,input_name)
        gt_path = os.path.join(gt_folder,gt_path)
        paths.append(
            dict([(f'{input_key}_path', input_path),
                  (f'{gt_key}_path', gt_path)]))

    return paths
