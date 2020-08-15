"""
@File    :   02_detect_from_video.py
@Contact :   156618056@qq.com
@License :   Free
@Modified:   2020/7/29 15:17
@Author  :   Karol Wu
@Version :   1.0 
@Des     :   None
"""
from yolov4.Detection import ObjectDetection
from utils.video_add_audio import add_audio_to_video


model_path = 'model_data/yolo4_weight.h5'
classes_path = 'model_data/coco_classes.txt'
anchors_path = 'model_data/yolo_anchors.txt'

path_to_video = "video/outside_wander.mp4"
out_to_file = "video/detected/outside_wander.mp4"

detector = ObjectDetection(model_path=model_path, classes_path=classes_path, anchors_path=anchors_path)
detector.detectObjectsFromVideo(input_file=path_to_video, out_to_file=out_to_file, resized=0.5)

add_audio_to_video(video_path_audio=path_to_video, video_path_no_audio=out_to_file)
