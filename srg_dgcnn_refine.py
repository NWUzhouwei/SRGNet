import numpy as np
import torch
use_cuda = True

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
from util.srg import srg
srg_label, srg_result = srg('data/demo_foot_clean.xyz')
print('len(srg_result):', len(srg_result))
srg_result = np.array([srg_label, srg_label]).astype(int)
print('srg_result.shape:', srg_result.shape)

## train
from model.model import *

print("start training...")

train_data = np.array(xyz_batch)
# tensor = torch.from_numpy(train_data.transpose((0, 2, 1))).type(torch.FloatTensor)
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
    dataset = 'modelnet40'
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
optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.1)
l_inds = srg_result

from get_miou import get_miou
from util.add_label_to_part import get_hand_split_label
group_result = get_hand_split_label()

result = []
max_iter = 2500
iou_list = []
for batch_idx in range(max_iter):
    optimizer.zero_grad()
    output = model(tensor)[0]
    print('output.shape:', output.shape)
    _, target = torch.max(output, 1)  # BN * 1
    refined_target = target[0].data.cpu().numpy()
    print('refined_target.shape:', refined_target.shape)

    # refinement
    for i in range(len(l_inds)):
        labels_per_sp = refined_target[l_inds[i]]
        u_labels_per_sp = np.unique(labels_per_sp)
        hist = np.zeros(len(u_labels_per_sp))
        for j in range(len(hist)):
            hist[j] = len(np.where(labels_per_sp == u_labels_per_sp[j])[0])
        refined_target[l_inds[i]] = u_labels_per_sp[np.argmax(hist)]

    # print('target.shape:', target.shape)
    # print('target', target.data.cpu().numpy()[0][0], target.data.cpu().numpy()[0][1], target.data.cpu().numpy()[0][-1], target.data.cpu().numpy()[0][-2])
    result = target
    iou = get_miou(result[0], group_result)
    refined_target = torch.from_numpy(np.array([refined_target, refined_target]))
    if use_cuda:
        refined_target = refined_target.cuda()
    loss = loss_fn(output, refined_target)
    loss.backward()
    optimizer.step()
    if batch_idx % 10 == 0:
        print(batch_idx, '/', max_iter, ':', 'iou: ', iou, 'loss: ', loss.item())
        iou_list.append(iou)

## save result
label_colours = np.random.randint(255, size=(class_num, 3))
label_colours = [[237, 87, 54], [234, 255, 86], [130, 113, 0],
                    [72, 192, 163], [68, 14, 37], [68, 206, 246],
                    [209, 217, 224], [238, 222, 176], [98, 42, 29],
                    [65, 85, 93]]
label_colours = np.array(label_colours)
color_cloud = []

for i in range(len(data)):
    line = [data[int(i)][0], data[int(i)][1], data[int(i)][2], *label_colours[int(result[0][i])]]
    color_cloud.append(line)

segment = []
for i in range(class_num):
    segment.append([])

print('result shape:', np.array(result.cpu()).shape)
for index, item in enumerate(result[0]):
    segment[item].append(index)
           
# for i, piece in enumerate(segment):
#     with open(f'result/srg_pointnet2_label_{i}.txt', 'w') as file:
#         for j, item in enumerate(piece):
#             file.write(f'{segment[i][j]}\n')

for i, piece in enumerate(segment):
    with open(f'result/srg_dgcnn_part_{i}.xyz', 'w') as file:
        for j, item in enumerate(piece):
            file.write(f'{xyz[j][0]}; {xyz[j][1]}; {xyz[j][2]}\n')

with open('result/srg_dgcnn_refine_result.txt', 'w') as file:
    for i, line in enumerate(color_cloud):
        file.write(f'{line[0]}; {line[1]}; {line[2]}; {line[3]}; {line[4]}; {line[5]}\n')

with open('result/srg_dgcnn_refine_iou.txt', 'w') as file:
    for i, item in enumerate(iou_list):
        file.write(f'{item}\n')