# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import time

import mmcv
import torch

from mmedit.apis import init_model, restoration_inference
from mmedit.core import srocc, plcc

import pandas as pd
from tqdm import tqdm
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='ClipIQA demo')
    parser.add_argument('--config', default='configs/clipiqa/clipiqa_attribute_test.py', help='test config file path')
    parser.add_argument('--checkpoint', default=None, help='checkpoint file')
    parser.add_argument('--file_path', default='D:/Datasets/PIPAL/Distortion/Distortion_1/',
                        help='path to input image file')
    parser.add_argument('--txt_folder', default='D:/Datasets/PIPAL/Distortion/Train_Label/',
                        help='path to input txt files')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    model = init_model(args.config, args.checkpoint, device=torch.device('cuda', args.device))

    # 读取所有txt文件
    txt_files = [f for f in os.listdir(args.txt_folder) if f.endswith('.txt')]
    image_names = []
    mos_scores = []

    for txt_file in txt_files:
        with open(os.path.join(args.txt_folder, txt_file), 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                image_names.append(parts[0])
                mos_scores.append(float(parts[1]))

    pred_scores = []
    runtimes = []

    for image_name in tqdm(image_names):
        start_time = time.time()
        output, attributes = restoration_inference(model, os.path.join(args.file_path, image_name),
                                                   return_attributes=True)
        end_time = time.time()

        output = output.float().detach().cpu().numpy()
        pred_scores.append(attributes[0])
        runtimes.append(end_time - start_time)

    # 写入 output.txt
    with open('output.txt', 'w') as output_file:
        for img_name, score in zip(image_names, pred_scores):
            output_file.write(f'{img_name},{score}\n')

    # 写入 readme.txt
    avg_runtime = np.mean(runtimes)
    with open('readme.txt', 'w') as readme_file:
        readme_file.write(f'runtime per image[s]:{avg_runtime:.2f}\n')
        readme_file.write('CPU[1]/GPU[0]:0\n')  # 假设我们使用GPU
        readme_file.write('Extra Data[1]/No Extra Data[0]:0\n')  # 假设我们不使用额外数据
        readme_file.write('Other description: Solution based on the given configuration and model.\n')


if __name__ == '__main__':
    main()
