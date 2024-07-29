import os
import sys

from numpy.core.defchararray import add
dirname = os.path.dirname(__file__)
class_path = os.path.join(dirname, '..')
sys.path.append(class_path)
import numpy as np
import math
from tqdm import tqdm

def isclose(item_1: float, item_2: float):
    return math.isclose(item_1, item_2, abs_tol=1e-7)

def index_point(full, part):
    part_index_list = []
    # pbar = tqdm(total=len(full))
    for index, line in enumerate(full):
        for part_line in part:
            if isclose(line[0], part_line[0]) and isclose(line[1], part_line[1]) and isclose(line[2], part_line[2]):
                part_index_list.append(index)
        # pbar.update(1)
    return part_index_list

def read_xyz_without_normal(file_path='data/foot.xyz'):
    part = []
    with open(file_path, 'r') as file:
        for line in file.readlines():
            line = line.split()
            part.append([line[0], line[1], line[2]])
    part = np.array(part).astype(np.float)
    return part

def add_to_label_index_list(file_path, label_index_list, full):
    print(f'Start indexing {file_path}...')
    part = read_xyz_without_normal(file_path)
    part_index_list = index_point(full, part)
    label_index_list.append(part_index_list)

def get_hand_split_label(data_location: str):
    label_index_list = []
    full = read_xyz_without_normal(file_path=f'data/{data_location}/{data_location}_downsample_denoise.xyz')

    ## head
    add_to_label_index_list(f'data/{data_location}/{data_location}_head.xyz', label_index_list, full)

    # left foot
    add_to_label_index_list(f'data/{data_location}/{data_location}_left_foot.xyz', label_index_list, full)

    # right foot
    add_to_label_index_list(f'data/{data_location}/{data_location}_right_foot.xyz', label_index_list, full)

    # hand
    add_to_label_index_list(f'data/{data_location}/{data_location}_left_hand.xyz', label_index_list, full)
    add_to_label_index_list(f'data/{data_location}/{data_location}_right_hand.xyz', label_index_list, full)

    # add label
    print('Start adding label...')
    label = np.full(len(full), 100)
    for index, list in enumerate(label_index_list):
        for item in list:
            label[item] = index

    for index, value in enumerate(label):
        if label[index] == 100:
            label[index] = len(label_index_list)

    return label

if __name__ == "__main__":
    full = read_xyz_without_normal(file_path='data/005420/005420_denoising_downsampling.xyz')
    label = get_hand_split_label()

    # add color
    print('Start adding color...')
    label_colours = [[237, 87, 54], [234, 255, 86], [130, 113, 0],
                        [72, 192, 163], [66, 76, 80], [68, 206, 246],
                        [209, 217, 224], [238, 222, 176], [98, 42, 29],
                        [65, 85, 93]]
    label_colours = np.array(label_colours)

    color_cloud = []
    for i in range(len(full)):
        line = [full[int(i)][0], full[int(i)][1], full[int(i)][2], *label_colours[label[i]]]
        color_cloud.append(line)

    with open('result/hand_split_part_result.txt', 'w') as file:
        for i, line in enumerate(color_cloud):
            file.write(f'{line[0]}; {line[1]}; {line[2]}; {line[3]}; {line[4]}; {line[5]}\n')