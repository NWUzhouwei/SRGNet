from typing import Union
import numpy as np
import math

def isclose(item_1: float, item_2: float):
    return math.isclose(item_1, item_2, abs_tol=1e-7)

def get_miou(output, target):
    result_list = []
    for i in range(max(target) + 1):
        tmp_list = []
        for j in range(2):
            flag = True
            while flag:
                index = np.random.randint(len(target))
                if target[index] == i:
                    flag = False
                    # print(index, target[index])
                    tmp_list.append(index)
        result_list.append(tmp_list)

    # print(result_list)
    # print('output before', output)
    # print('target', target)

    for i, list in enumerate(result_list):
        for j, chosen_index in enumerate(list):
            for k, output_label in enumerate(output):
                if output[k] == output[chosen_index] and k != chosen_index:
                    output[k] = target[chosen_index] 
            output[chosen_index] = target[chosen_index] 

    # print('output after', output)

    intersection = 0
    for i, item in enumerate(target):
        if isclose(target[i], output[i]):
            intersection += 1
    
    miou = intersection / len(target)
    # print('miou', miou)
    return miou

if __name__ == "__main__":
    output = []
    # with open('result/hand_split_dgcnn_label.txt', 'r') as file:
    # with open('result/srg_dgcnn_label.txt', 'r') as file:
    # with open('result/kmeans_dgcnn_label.txt', 'r') as file:
    # with open('result/kmeans_pointnet2_label.txt', 'r') as file:
    with open('result/kmeans_pointnet_label.txt', 'r') as file:
    # with open('result/srg_pointnet2_label.txt', 'r') as file:
    # with open('result/srg_pointnet_label.txt', 'r') as file:
        while True:
            line = file.readline()
            if str(line) != '':
                output.append(int(line))
            if not line:
                break
    print('output.shape', len(output))

    from util.add_label_to_part import get_hand_split_label

    print('Start hand splitting...')
    target = get_hand_split_label()
    print('target.shape', len(target))

    # target = np.random.randint(5, size=20)
    # output = target + 10
    
    # output = target + 10
    iou = get_miou(output, target)
    print('iou', iou)

    for i in range(2000):
        if i % 30 == 0:
            with open(f'result/srg_dgcnn/srg_dgcnn_label_{i}.txt', 'r') as file:
                while True:
                    line = file.readline()
                    if str(line) != '':
                        output.append(int(line))
                    if not line:
                        break
            # print('output.shape', len(output))

            iou = get_miou(output, target)
            print('iter: ', 'iou', iou)