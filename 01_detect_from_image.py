"""
@File    :   01_detect_from_image.py
@Contact :   156618056@qq.com
@License :   Free
@Modified:   2020/7/29 14:55
@Author  :   Karol Wu
@Version :   1.0 
@Des     :   None
"""
from yolov4.Detection import ObjectDetection


model_path = 'model_data/yolo4_weight.h5'
classes_path = 'model_data/coco_classes.txt'
anchors_path = 'model_data/yolo_anchors.txt'

path_to_image = "images/street.jpg"

detector = ObjectDetection(model_path=model_path, classes_path=classes_path, anchors_path=anchors_path)
detector.detectObjectsFromImage(input_file=path_to_image, confidence=0.6, custom_class=['person'])
