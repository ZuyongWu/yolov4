"""
@File    :   05_detect_from_your_model.py
@Contact :   156618056@qq.com
@License :   Free
@Modified:   2020/7/30 16:28
@Author  :   Karol Wu
@Version :   1.0 
@Des     :   None
"""
from yolov4.Detection import ObjectDetection
from utils.video_add_audio import add_audio_to_video


def detect_image():
    path_to_image = "images/test_facial_mask.jpeg"

    detector = ObjectDetection(model_path=model_path, classes_path=classes_path, anchors_path=anchors_path)
    detector.detectObjectsFromImage(input_file=path_to_image, confidence=0.6, custom_class=None)


def detect_video():
    path_to_video = "video/outside_wander.mp4"
    out_to_file = "video/detected/outside_wander_custom.mp4"

    # detector = ObjectDetection(model_path=model_path, classes_path=classes_path, anchors_path=anchors_path)
    # detector.detectObjectsFromVideo(input_file=path_to_video, out_to_file=out_to_file,
    #                                 confidence=0.3, resized=0.5)
    add_audio_to_video(video_path_audio=path_to_video, video_path_no_audio=out_to_file)


def detect_camera():
    detector = ObjectDetection(model_path=model_path, classes_path=classes_path, anchors_path=anchors_path)
    detector.detectObjectsFromCamera()


if __name__ == '__main__':
    model_path = 'facial_mask/models/epoch100-loss12.122-val_loss12.702.h5'
    classes_path = 'facial_mask/classes.txt'
    anchors_path = 'facial_mask/anchors.txt'

    # detect_image()
    # detect_video()
    detect_camera()
