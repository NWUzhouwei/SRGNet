import os
import sys
dirname = os.path.dirname(__file__)
class_path = os.path.join(dirname, '..')
sys.path.append(class_path)

import numpy as np
import torch
from util.srg import srg

use_cuda = False

data = []
xyz = []
normal = []
with open('data/demo_foot_clean.xyz', 'r') as file:
    for line in file.readlines():
        data.append([float(x) for x in line.split()])
        line = line.split()
        xyz.append([float(line[0])/5, float(line[1])/5, float(line[2])/5])
        normal.append([line[3], line[4], line[5]])
data = np.array(data).astype(np.float)
xyz = np.array(xyz).astype(np.float)
normal = np.array(normal).astype(np.float)

data_batch = [data, data]
xyz_batch = [xyz, xyz]


## srg
srg_label, srg_result = srg('data/demo_foot_clean.xyz')
print('len(srg_result):', len(srg_result))
srg_result = torch.from_numpy(np.array([srg_label, srg_label])).long()
print('srg_result.shape:', srg_result.shape)
if use_cuda:
    srg_result = srg_result.cuda()

## train
from model.net2 import *

print("start training...")

train_data = np.array(xyz_batch)
tensor = torch.from_numpy(train_data.transpose((0, 2, 1))).type(torch.FloatTensor)
if use_cuda:
    tensor = tensor.cuda()
class_num = 7
model = get_model(class_num) 
if use_cuda:
    model.cuda()
model.train()
# loss_fn = torch.nn.CrossEntropyLoss()
loss_fn = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.1)
l_inds = srg_result

result = []
max_iter = 2500
for batch_idx in range(max_iter):
    optimizer.zero_grad()
    output = model(tensor)[0]
    output = torch.transpose(output, 1, 2)
    print('output.shape:', output.shape)

    _, target = torch.max(output, 1)  # BN * 1
    print('target.shape:', target.shape)
    print('target', target.data.cpu().numpy()[0][0], target.data.cpu().numpy()[0][1], target.data.cpu().numpy()[0][-1], target.data.cpu().numpy()[0][-2])
    result = target
    if use_cuda:
        target = target.cuda()
    loss = loss_fn(output, srg_result)
    loss.backward()
    optimizer.step()
    print(batch_idx, '/', max_iter, ':', class_num, loss.item())

## save result
label_colours = np.random.randint(255, size=(class_num, 3))
label_colours = [[237, 87, 54], [234, 255, 86], [130, 113, 0],
                    [72, 192, 163], [66, 76, 80], [68, 206, 246],
                    [209, 217, 224], [238, 222, 176], [98, 42, 29],
                    [65, 85, 93]]
label_colours = np.array(label_colours)
color_cloud = []

for i in range(len(data)):
    line = [data[int(i)][0], data[int(i)][1], data[int(i)][2], *label_colours[int(result[0][i])]]
    color_cloud.append(line)

with open('result/srg_pointnet2_result.txt', 'w') as file:
    for i, line in enumerate(color_cloud):
        file.write(f'{line[0]}; {line[1]}; {line[2]}; {line[3]}; {line[4]}; {line[5]}\n')

segment_list = []
for i in range(class_num):
    segment_list.append([])

print('result shape:', np.array(result.cpu()).shape)
for index, item in enumerate(result[0]):
    segment_list[item].append(index)

for i, part in enumerate(segment_list):
    with open(f'result/srg_pointnet2_part_{i}.txt', 'w') as file:
        for item in part:
            file.write(f'{data[item][0]}; {data[item][1]}; {data[item][2]}\n')

