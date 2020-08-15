"""
@File    :   make_txt.py    
@Contact :   156618056@qq.com
@License :   Free
@Modified:   2020/8/13 22:55
@Author  :   Karol Wu
@Version :   1.0 
@Des     :   None
"""
import os
import xml.etree.ElementTree as ET
from utils.kmeans_for_anchors import make_anchors_txt


def convert_annotation(path_to_xml, txt_file, classes):
    xml_file = open(path_to_xml, encoding="UTF-8")
    tree = ET.parse(xml_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xml_box = obj.find('bndbox')
        b = (int(xml_box.find('xmin').text), int(xml_box.find('ymin').text), int(xml_box.find('xmax').text),
             int(xml_box.find('ymax').text))
        txt_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


def makeTxt(data_directory, classes, input_shape):
    annotations_dir = os.path.join(data_directory, "annotations")
    images_dir = os.path.join(data_directory, "images")

    class_txt = os.path.join(data_directory, "classes.txt")
    if not os.path.exists(class_txt):
        with open(class_txt, 'w') as f:
            for cls in classes:
                f.write(cls)
                f.write('\n')

    train_txt = os.path.join(data_directory, "train.txt")
    if not os.path.exists(train_txt):
        train_file = open(train_txt, 'w')

        images_name = os.listdir(images_dir)
        for name in images_name:
            path_to_image = os.path.join(images_dir, name)
            path_to_xml = os.path.join(annotations_dir, str(name.split(".")[0]) + ".xml")

            train_file.write(path_to_image)
            convert_annotation(path_to_xml, train_file, classes)
            train_file.write('\n')
        train_file.close()

    anchors_txt = os.path.join(data_directory, "anchors.txt")
    if not os.path.exists(anchors_txt):
        make_anchors_txt(data_directory, input_shape)


if __name__ == '__main__':
    makeTxt(data_directory="facial_mask", classes=["facial mask"])
