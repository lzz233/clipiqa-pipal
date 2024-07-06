import argparse
import os
import time
import threading

import mmcv
import torch
from mmedit.apis import init_model, restoration_inference
from mmedit.core import srocc, plcc

import pandas as pd
from tqdm import tqdm
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from io import BytesIO


def parse_args():
    parser = argparse.ArgumentParser(description='ClipIQA demo')
    parser.add_argument('--config', default='configs/clipiqa/clipiqa_attribute_test.py', help='test config file path')
    parser.add_argument('--checkpoint', default=None, help='checkpoint file')
    parser.add_argument('--file_path', default='D:/Datasets/PIPAL/Distortion/Distortion_1/',
                        help='path to input image file')
    parser.add_argument('--txt_folder', default='D:/Datasets/PIPAL/Distortion/Train_Label/',
                        help='path to input txt files')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    parser.add_argument('--sample_fraction', type=float, default=0.001, help='fraction of data to sample for testing')
    args = parser.parse_args()
    return args


def draw_radar_chart(values):
    categories = ['Composite', 'Detail', 'Noise', 'Color', 'Contrast']
    values = values.reshape(-1)
    values = np.append(values, values[0])

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='skyblue', alpha=0.6)
    ax.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9])
    ax.set_yticklabels(["0.1", "0.3", "0.5", "0.7", "0.9"], color="grey", size=12)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color="grey", size=12)

    return fig


def show_radar_chart(attributes):
    fig = draw_radar_chart(attributes)
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    radar = Image.open(buf)
    photo = ImageTk.PhotoImage(radar)
    return photo


def process_image(model, file_path, image_name):
    try:
        start_time = time.time()
        output, attributes = restoration_inference(model, os.path.join(file_path, image_name), return_attributes=True)
        end_time = time.time()
        runtime = end_time - start_time
        output = output.float().detach().cpu().numpy()
        return output, attributes, runtime
    except Exception as e:
        print(f"Error processing image {image_name}: {e}")
        return None, None, None


def main():
    args = parse_args()
    model = init_model(args.config, args.checkpoint, device=torch.device('cuda', args.device))

    txt_files = [f for f in os.listdir(args.txt_folder) if f.endswith('.txt')]
    image_names = []
    mos_scores = []

    for txt_file in txt_files:
        with open(os.path.join(args.txt_folder, txt_file), 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                image_names.append(parts[0])
                mos_scores.append(float(parts[1]))

    sample_size = int(len(image_names) * args.sample_fraction)
    image_names = image_names[:sample_size]
    mos_scores = mos_scores[:sample_size]

    pred_scores = []
    runtimes = []

    root = tk.Tk()
    root.title("Image Quality Assessment")
    root.geometry("1600x1200")

    image_label = tk.Label(root)
    image_label.place(x=350, y=250)  # 设置图片位置
    image_name_label = tk.Label(root, text="", font=("Arial", 14))
    image_name_label.place(x=400, y=550)  # 设置图片名称标签位置

    radar_label = tk.Label(root)
    radar_label.place(x=850, y=150)  # 设置雷达图位置
    scores_label = tk.Label(root, text="Scores", font=("Arial", 14))
    scores_label.place(x=1100, y=770)  # 设置“scores”标签位置

    def update_gui(image, radar, image_name):
        image_label.config(image=image)
        image_label.image = image
        image_name_label.config(text=image_name)
        radar_label.config(image=radar)
        radar_label.image = radar
        root.update()

    def process_images():
        for image_name in tqdm(image_names):
            output, attributes, runtime = process_image(model, args.file_path, image_name)
            if output is not None and attributes is not None:
                pred_scores.append(attributes[0])
                runtimes.append(runtime)

                image_path = os.path.join(args.file_path, image_name)
                image = Image.open(image_path)
                photo = ImageTk.PhotoImage(image)
                radar = show_radar_chart(attributes)

                root.after(0, update_gui, photo, radar, image_name)
                time.sleep(0.6)  # 切换图片的时间间隔

    thread = threading.Thread(target=process_images)
    thread.start()

    root.mainloop()

    # 保存结果
    with open('output.txt', 'w') as output_file:
        for img_name, score in zip(image_names, pred_scores):
            output_file.write(f'{img_name},{score}\n')

    avg_runtime = np.mean(runtimes)
    with open('readme.txt', 'w') as readme_file:
        readme_file.write(f'runtime per image[s]:{avg_runtime:.2f}\n')
        readme_file.write('CPU[1]/GPU[0]:0\n')
        readme_file.write('Extra Data[1]/No Extra Data[0]:0\n')
        readme_file.write('Other description: Solution based on the given configuration and model.\n')


if __name__ == '__main__':
    main()
