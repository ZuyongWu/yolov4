import numpy as np
import xml.etree.ElementTree as ET
import glob
import random
import os


def cas_iou(box, cluster):
    x = np.minimum(cluster[:, 0], box[0])
    y = np.minimum(cluster[:, 1], box[1])

    intersection = x * y
    area1 = box[0] * box[1]

    area2 = cluster[:, 0] * cluster[:, 1]
    iou = intersection / (area1 + area2 - intersection)

    return iou


def avg_iou(box, cluster):
    return np.mean([np.max(cas_iou(box[i], cluster)) for i in range(box.shape[0])])


def kmeans(box, k):
    row = box.shape[0]
    distance = np.empty((row, k))
    last_clu = np.zeros((row,))
    np.random.seed()

    cluster = box[np.random.choice(row, k, replace=False)]
    while True:
        for j in range(row):
            distance[j] = 1 - cas_iou(box[j], cluster)
        near = np.argmin(distance, axis=1)

        if (last_clu == near).all():
            break

        for j in range(k):
            cluster[j] = np.median(
                box[near == j], axis=0)
        last_clu = near

    return cluster


def load_data(path_):
    data_ = []

    for xml_file in glob.glob('{}/*xml'.format(path_)):
        tree = ET.parse(xml_file)
        height = int(tree.findtext('./size/height'))
        width = int(tree.findtext('./size/width'))

        for obj in tree.iter('object'):
            xmin = int(float(obj.findtext('bndbox/xmin'))) / width
            ymin = int(float(obj.findtext('bndbox/ymin'))) / height
            xmax = int(float(obj.findtext('bndbox/xmax'))) / width
            ymax = int(float(obj.findtext('bndbox/ymax'))) / height

            xmin = np.float64(xmin)
            ymin = np.float64(ymin)
            xmax = np.float64(xmax)
            ymax = np.float64(ymax)

            data_.append([xmax - xmin, ymax - ymin])
    return np.array(data_)


def make_anchors_txt(data_directory, input_shape):
    size = input_shape
    anchors_num = 9

    path = os.path.join(data_directory, "annotations")
    data = load_data(path)

    out = kmeans(data, anchors_num)
    out = out[np.argsort(out[:, 0])]
    print('acc:{:.2f}%'.format(avg_iou(data, out) * 100))
    print(out * size)
    data = out * size
    f = open(os.path.join(data_directory, "anchors.txt"), 'w')
    row = np.shape(data)[0]
    for i in range(row):
        if i == 0:
            x_y = "%d,%d" % (data[i][0], data[i][1])
        else:
            x_y = ", %d,%d" % (data[i][0], data[i][1])
        f.write(x_y)
    f.close()


if __name__ == '__main__':
    pass
