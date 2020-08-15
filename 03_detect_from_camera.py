"""
@File    :   03_detect_from_camera.py
@Contact :   156618056@qq.com
@License :   Free
@Modified:   2020/7/29 21:12
@Author  :   Karol Wu
@Version :   1.0 
@Des     :   None
"""
from yolov4.Detection import ObjectDetection


model_path = 'model_data/yolo4_weight.h5'
classes_path = 'model_data/coco_classes.txt'
anchors_path = 'model_data/yolo_anchors.txt'

camera_out_path = "video/detected/camera_detection.mp4"

detector = ObjectDetection(model_path=model_path, classes_path=classes_path, anchors_path=anchors_path)
detector.detectObjectsFromCamera(out_to_file=camera_out_path, save_video=True)
