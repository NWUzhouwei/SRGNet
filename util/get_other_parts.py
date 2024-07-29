import os
import sys

from numpy.core.defchararray import add
dirname = os.path.dirname(__file__)
class_path = os.path.join(dirname, '..')
sys.path.append(class_path)
import numpy as np
import math

from util.add_label_to_part import read_xyz_without_normal, add_to_label_index_list

def read_xyz_with_normal(file_path='data/foot.xyz'):
    part = []
    with open(file_path, 'r') as file:
        for line in file.readlines():
            line = line.split()
            part.append([line[0], line[1], line[2], line[3], line[4], line[5]])
    part = np.array(part).astype(np.float)
    return part

full = read_xyz_without_normal(file_path='data/demo_foot_clean.xyz')

label_index_list = []
full = read_xyz_without_normal(file_path='data/demo_foot_clean.xyz')

## head
add_to_label_index_list('data/head.xyz', label_index_list, full)

# left foot
add_to_label_index_list('data/left_foot.xyz', label_index_list, full)

# right foot
add_to_label_index_list('data/right_foot.xyz', label_index_list, full)

# hand
add_to_label_index_list('data/hand.xyz', label_index_list, full)

# add label
print('Start adding label...')
label = np.full(len(full), 100)
for index, list in enumerate(label_index_list):
    for item in list:
        label[item] = index

# get body
full_with_normal = read_xyz_with_normal(file_path='data/demo_foot_clean.xyz')
body = []
for index, value in enumerate(label):
    if label[index] == 100:
        body.append(full_with_normal[index])

body = np.array(body)
print('body.shape', body.shape)

with open('result/other_parts.xyz', 'w') as file:
    for i, line in enumerate(body):
        file.write(f'{line[0]}; {line[1]}; {line[2]}; {line[3]}; {line[4]}; {line[5]}\n')
