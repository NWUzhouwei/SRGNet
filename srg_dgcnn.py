import numpy as np
import torch
import argparse

use_cuda = True

parser = argparse.ArgumentParser(description='SRG DGCNN')
parser.add_argument('--data_location', type=str, default='005420', help='data location')
args = parser.parse_args()

data_location = args.data_location
data = []
xyz = []
normal = []
with open(f'data/{data_location}/{data_location}_downsample_denoise.xyz', 'r') as file:
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
from util.srg import srg
srg_label, srg_result = srg(f'data/{data_location}/{data_location}_downsample_denoise.xyz', segment_num=5)
print('len(srg_result):', len(srg_result))
srg_result = torch.from_numpy(np.array([srg_label, srg_label])).long()
print('srg_result.shape:', srg_result.shape)
if use_cuda:
    srg_result = srg_result.cuda()
l_inds = srg_result

## train
from model.model import *

print("Start training...")
train_data = np.array(xyz_batch)
tensor = torch.from_numpy(train_data).type(torch.FloatTensor)
# print('tensor shape:', tensor.shape)

means = tensor.mean(1, keepdim=True)
# print('mean shape', means.shape)

deviations = tensor.std(1, keepdim=True)
# print('deviations shape', deviations.shape)

tensor = (tensor - means) / deviations
print('tensor shape:', tensor.shape)

if use_cuda:
    tensor = tensor.cuda()
class_num = 6
class Args:
    k = 10
    task = 'segment'
    feat_dims = 20
    encoder = 'dgcnn_seg'
    shape = 'sphere'
    dropout = 0.1
    eval = False
    loss = 'softmax'     
    seg_no_class_label = True
args = Args
model = SegmentationNet(args, seg_num_all=class_num)
if use_cuda:
    model.cuda()
model.train()

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.000005, momentum=0.001)

from get_miou import get_miou
from util.add_label_to_part import get_hand_split_label
group_result = get_hand_split_label(data_location)
iou_list = []

max_iter = 2000
for batch_idx in range(max_iter):
    optimizer.zero_grad()
    output = model(tensor)[0]
    _, result = torch.max(output, 1)
    iou = get_miou(result[0].cpu().numpy(), group_result)
    loss = loss_fn(output, srg_result)
    loss.backward()
    optimizer.step()
    if batch_idx % 10 == 0:
        print(batch_idx, '/', max_iter, ':', 'iou: ', iou, 'loss: ', loss.item())
        iou_list.append(iou)


## save result
label_colours = np.random.randint(255, size=(class_num, 3))
label_colours = [[237, 87, 54], [234, 255, 86], [130, 113, 0],
                    [72, 192, 163], [66, 76, 80], [68, 206, 246],
                    [209, 217, 224], [238, 222, 176], [98, 42, 29],
                    [65, 85, 93]]
label_colours = np.array(label_colours)
color_cloud = []

with open(f'result/{data_location}/srg_dgcnn_label.txt', 'w') as file:
    for i, item in enumerate(result[0]):
        file.write(f'{item}\n')

for i in range(len(data)):
    line = [data[int(i)][0], data[int(i)][1], data[int(i)][2], *label_colours[int(result[0][i])]]
    color_cloud.append(line)

with open(f'result/{data_location}/srg_dgcnn_result.txt', 'w') as file:
    for i, line in enumerate(color_cloud):
        file.write(f'{line[0]}; {line[1]}; {line[2]}; {line[3]}; {line[4]}; {line[5]}\n')

with open(f'result/{data_location}/srg_dgcnn_iou.txt', 'w') as file:
    for i, item in enumerate(iou_list):
        file.write(f'{item}\n')