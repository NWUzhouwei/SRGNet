import numpy as np
import torch
import argparse
use_cuda = True

parser = argparse.ArgumentParser(description='Kmeans DGCNN')
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


## kmeans
from sklearn.cluster import KMeans
data = np.array(data)
kmeans = KMeans(n_clusters=6, random_state=0).fit(data)
labels = kmeans.labels_
u_labels = np.unique(labels)
l_inds = []
for i in range(len(u_labels)):
    l_inds.append(np.where(labels == u_labels[i])[0])

target = torch.from_numpy(np.array([labels, labels])).long()
print('target.shape', target.shape)
if use_cuda:
    target = target.cuda()

## train
from model.net import *

print("Start training...")
train_data = np.array(xyz_batch)
tensor = torch.from_numpy(train_data).type(torch.FloatTensor)
# print('tensor shape:', tensor.shape)

means = tensor.mean(1, keepdim=True)
# print('mean shape', means.shape)

deviations = tensor.std(1, keepdim=True)
# print('deviations shape', deviations.shape)

tensor = (tensor - means) / deviations
tensor = tensor.permute(0, 2, 1)
print('tensor shape:', tensor.shape)

if use_cuda:
    tensor = tensor.cuda()
class_num = 6
model = get_model(class_num, with_rgb=False)
if use_cuda:
    model.cuda()
model.train()

loss_fn = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.1)

from get_miou import get_miou
from util.add_label_to_part import get_hand_split_label

max_iter = 2000
iou_list = []
group_result = get_hand_split_label(data_location)
for batch_idx in range(max_iter):
    optimizer.zero_grad()
    output = model(tensor)[0]
    output = output.permute(0, 2, 1)
    _, result = torch.max(output, 1)
    iou = get_miou(result[0].cpu().numpy(), group_result)
    loss = loss_fn(output, target)
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

with open(f'result/{data_location}/kmeans_pointnet_label.txt', 'w') as file:
    for i, item in enumerate(result[0]):
        file.write(f'{item}\n')

for i in range(len(data)):
    line = [data[int(i)][0], data[int(i)][1], data[int(i)][2], *label_colours[int(result[0][i])]]
    color_cloud.append(line)

with open(f'result/{data_location}/kmeans_pointnet_result.txt', 'w') as file:
    for i, line in enumerate(color_cloud):
        file.write(f'{line[0]}; {line[1]}; {line[2]}; {line[3]}; {line[4]}; {line[5]}\n')

with open(f'result/{data_location}/kmeans_pointnet_iou.txt', 'w') as file:
    for i, item in enumerate(iou_list):
        file.write(f'{item}\n')