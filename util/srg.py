import numpy as np
from sklearn.neighbors import NearestNeighbors
import random

def get_dist(point_1, point_2):
    distance = np.sqrt(np.sum(np.square(point_1 - point_2)))
    return distance

def srg(data_path, segment_length_threshold = 700, euc_dis_threshold = 18, normal_threshold = 2.0, segment_num = 6):
    data = []
    xyz = []
    normal = []
    with open(data_path, 'r') as file:
        for line in file.readlines():
            data.append([float(x) for x in line.split()])
            line = line.split()
            xyz.append([line[0], line[1], line[2]])
            normal.append([line[3], line[4], line[5]])
    data = np.array(data).astype(np.float)
    xyz = np.array(xyz).astype(np.float)
    normal = np.array(normal).astype(np.float)

    nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(xyz)
    _, indices = nbrs.kneighbors(xyz)

    mark = np.zeros(len(data))
    result = []

    for epoch in range(segment_num):
        print(f'epoch {epoch}')
        re = []
        re_index = []
        seed_list = []
        flag = 1

        mark_zero_index_list = []
        for index, item in enumerate(mark):
            if item == 0:
                mark_zero_index_list.append(index)
            else:
                continue

        while len(re) <= segment_length_threshold:
            seed_list.append(mark_zero_index_list[random.randint(
                0,
                len(mark_zero_index_list) - 1)])
            while (len(seed_list) > 0):
                seed = int(seed_list[0])
                seed_list.pop(0)
                mark[seed] = flag

                for i, neighbor in enumerate(indices[seed]):
                    normal_dist = get_dist(normal[seed], normal[neighbor])
                    euc_dist = get_dist(xyz[seed], xyz[neighbor])
                    if (normal_dist < normal_threshold
                            and euc_dist < euc_dis_threshold
                            and mark[neighbor] != flag):
                        re.append(xyz[neighbor])
                        re_index.append(neighbor)
                        mark[neighbor] = flag
                        seed_list.append(neighbor)

            if len(re) > segment_length_threshold:
                print('re:', np.array(re).shape)
                result.append(re_index)

    for result_index, segment in enumerate(result):
        print(result_index, np.array(segment).shape)

    all_index = list(range(len(data)))
    other_part_index = all_index

    for result_index, segment in enumerate(result):
        # print(result_index, np.array(segment).shape)
        other_part_index = list(set(other_part_index) - set(segment))

    print(f'other_part_index: {np.array(other_part_index).shape}')
    result.append(other_part_index)

    label = np.zeros(len(data))
    for result_index, segment in enumerate(result):
        for _, item in enumerate(segment):
            label[item] = result_index

    return label, result


if __name__ == "__main__":
    data_path = 'data/005420/005420_denoising_downsampling.xyz'
    label, generate_result = srg(data_path)
    with open('result/label_result.txt', 'w') as file:
        for item in label:
            file.write(f'{int(item)}\n')

    with open('result/generate_result.txt', 'w') as file:
        for item in generate_result:
            for result in generate_result:
                file.write(f'{str(result)} ')
            file.write('\n')

    data = []
    xyz = []
    normal = []
    with open(data_path, 'r') as file:
        for line in file.readlines():
            data.append([float(x) for x in line.split()])
            line = line.split()
            xyz.append([line[0], line[1], line[2]])
            normal.append([line[3], line[4], line[5]])
    data = np.array(data).astype(np.float)
    xyz = np.array(xyz).astype(np.float)
    normal = np.array(normal).astype(np.float)

    class_num = 11
    label_colours = np.random.randint(255, size=(class_num, 3))
    label_colours = [[237, 87, 54], [234, 255, 86], [130, 113, 0],
                     [72, 192, 163], [66, 76, 80], [68, 206, 246],
                     [209, 217, 224], [238, 222, 176], [98, 42, 29],
                     [65, 85, 93]]
    label_colours = np.array(label_colours)
    color_cloud = []

    for i in range(len(data)):
        line = [*xyz[int(i)], *label_colours[int(label[i])]]
        color_cloud.append(line)

    with open('result/point_result.txt', 'w') as file:
        for i, line in enumerate(color_cloud):
            file.write(
                f'{line[0]}; {line[1]}; {line[2]}; {line[3]}; {line[4]}; {line[5]}\n'
            )
