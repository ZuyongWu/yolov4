"""
@File    :   utils.py    
@Contact :   156618056@qq.com
@License :   Free
@Modified:   2020/8/14 19:06
@Author  :   Karol Wu
@Version :   1.0 
@Des     :   None
"""
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from nets.yolo4 import yolo_body
from nets.loss import yolo_loss
from keras.backend.tensorflow_backend import set_session
from utils.utils import get_random_data, get_random_data_with_Mosaic, rand, WarmUpCosineDecayScheduler


def dummy_loss(y_true, y_pred):
    return y_pred


def get_anchors(anchors_path):
    """loads the anchors from a file"""
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, mosaic=False):
    """data generator for fit_generator"""
    n = len(annotation_lines)
    i = 0
    flag = True
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            if mosaic:
                if flag and (i + 4) < n:
                    image, box = get_random_data_with_Mosaic(annotation_lines[i:i + 4], input_shape)
                    i = (i + 1) % n
                else:
                    image, box = get_random_data(annotation_lines[i], input_shape)
                    i = (i + 1) % n
                flag = bool(1 - flag)
            else:
                image, box = get_random_data(annotation_lines[i], input_shape)
                i = (i + 1) % n
            image_data.append(image)
            box_data.append(box)
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


# ---------------------------------------------------#
#   读入xml文件，并输出y_true
# ---------------------------------------------------#
def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
    # 一共有三个特征层数
    num_layers = len(anchors) // 3
    # 先验框
    # 678为 142,110,  192,243,  459,401
    # 345为 36,75,  76,55,  72,146
    # 012为 12,16,  19,36,  40,28
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')  # 416,416
    # 读出xy轴，读出长宽
    # 中心点(m,n,2)
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    # 计算比例
    true_boxes[..., 0:2] = boxes_xy / input_shape[:]
    true_boxes[..., 2:4] = boxes_wh / input_shape[:]

    # m张图
    m = true_boxes.shape[0]
    # 得到网格的shape为13,13;26,26;52,52
    grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
    # y_true的格式为(m,13,13,3,85)(m,26,26,3,85)(m,52,52,3,85)
    y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes),
                       dtype='float32') for l in range(num_layers)]
    # [1,9,2]
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    # 长宽要大于0才有效
    valid_mask = boxes_wh[..., 0] > 0

    for b in range(m):
        # 对每一张图进行处理
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0: continue
        # [n,1,2]
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        # 计算真实框和哪个先验框最契合
        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)
        # 维度是(n) 感谢 消尽不死鸟 的提醒
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    # floor用于向下取整
                    i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                    # 找到真实框在特征层l中第b副图像对应的位置
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b, t, 4].astype('int32')
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5 + c] = 1

    return y_true

