"""
@File    :   __init__.py.py    
@Contact :   156618056@qq.com
@License :   Free
@Modified:   2020/8/13 20:29
@Author  :   Karol Wu
@Version :   1.0 
@Des     :   None
"""
import os
import numpy as np
import colorsys
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
from nets.yolo4 import yolo_body, yolo_eval
from utils.utils import letterbox_image
import cv2
import time


class ObjectDetection:
    def __init__(self, model_path, classes_path, anchors_path):
        self.classes_path = classes_path
        self.model_path = model_path
        assert self.model_path.endswith('.h5'), 'model or weights must be a .h5 file.'

        self.anchors_path = anchors_path
        self.score = 0.1
        self.iou = 0.3
        self.model_image_size = (416, 416)

        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()

        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        np.random.seed(10101)
        np.random.shuffle(self.colors)
        np.random.seed(None)

        self.yolo_model = None
        self.input_image_shape = K.placeholder(shape=(2,))
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        with open(self.classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        with open(self.anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        try:
            self.yolo_model = load_model(self.model_path, compile=False)
        except Exception:
            self.yolo_model = yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(self.model_path)
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(self.model_path))

        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           num_classes, self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_per_image(self, image, confidence=0.4, custom_class=None):
        """"""
        new_image_size = self.model_image_size
        boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        font = ImageFont.truetype(font='font/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in list(enumerate(out_classes)):
            predicted_class = self.class_names[c]
            if custom_class and predicted_class not in custom_class:
                continue

            score = out_scores[i]
            if score < confidence:
                continue

            box = out_boxes[i]
            top, left, bottom, right = box
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for j in range(thickness):
                draw.rectangle(
                    [left + j, top + j, right - j, bottom - j],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        return image

    def detectObjectsFromImage(self, input_file=None, out_to_file=None, confidence=0.4, custom_class=None):
        """"""
        assert os.path.isfile(input_file), "input must be a file"
        if out_to_file is None:
            base_dir = "/".join(input_file.split("/")[:-1]) + "/detected"
            os.makedirs(base_dir, exist_ok=True)
            out_to_file = base_dir + "/" + input_file.split("/")[-1]

        t0 = time.time()

        image = Image.open(input_file)
        detect_image = self.detect_per_image(image, confidence, custom_class)
        detect_image.save(out_to_file)

        print('Done. (%.3fs)' % (time.time() - t0))

    def detectObjectsFromVideo(self, input_file=None, out_to_file=None, resized=None, confidence=0.4, custom_class=None):
        """"""
        assert os.path.isfile(input_file), "input must be a file"
        if out_to_file is None:
            base_dir = "/".join(input_file.split("/")[:-1]) + "/detected"
            os.makedirs(base_dir, exist_ok=True)
            out_to_file = base_dir + "/" + input_file.split("/")[-1].split(".")[0] + ".mp4"

        fps = 0.0
        input_video = cv2.VideoCapture(input_file)
        length = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_per_second = int(input_video.get(cv2.CAP_PROP_FPS))
        frame_width = int(input_video.get(3))
        frame_height = int(input_video.get(4))

        output_video = cv2.VideoWriter(out_to_file, cv2.VideoWriter_fourcc(*'mp4v'), frames_per_second, (frame_width, frame_height))
        video_frames_count = 0
        while input_video.isOpened():
            t1 = time.time()
            ret, frame = input_video.read()

            if ret:
                video_frames_count += 1

                # BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # convert to Image
                image = Image.fromarray(np.uint8(frame))
                # convert image to array
                frame = np.array(self.detect_per_image(image, confidence, custom_class))
                # RGB to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                output_video.write(frame)

                fps = (fps + (1. / (time.time() - t1))) / 2
                if resized:
                    frame = cv2.resize(frame, (0, 0), fx=resized, fy=resized)
                cv2.imshow("video", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
                print("\rFrames: {}/{}, FPS: {}".format(video_frames_count, length, round(fps, 2)), end="")
            else:
                cv2.destroyAllWindows()
                break

    def detectObjectsFromCamera(self, out_to_file=None, save_video=False,
                                confidence=0.4, custom_class=None, frames_per_second=30):
        """"""
        if save_video:
            assert out_to_file, "you must set path for output file"

        fps = 0.0
        input_video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        frame_width = int(input_video.get(3))
        frame_height = int(input_video.get(4))

        output_video = None
        if save_video:
            output_video = cv2.VideoWriter(out_to_file, cv2.VideoWriter_fourcc(*'mp4v'), frames_per_second, (frame_width, frame_height))

        while input_video.isOpened():
            t1 = time.time()
            ret, frame = input_video.read()

            if ret:
                real_image = frame.copy()
                # BGR to RGB
                real_image = cv2.cvtColor(real_image, cv2.COLOR_BGR2RGB)
                # convert to image
                image = Image.fromarray(np.uint8(real_image))
                # convert image to array
                detect_image = np.array(self.detect_per_image(image, confidence, custom_class))
                # RGB to BGR
                detect_image = cv2.cvtColor(detect_image, cv2.COLOR_RGB2BGR)

                fps = (fps + (1. / (time.time() - t1))) / 2
                detect_image = cv2.putText(detect_image, "FPS = %.2f" % fps, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("video", detect_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

                if save_video:
                    output_video.write(detect_image)
