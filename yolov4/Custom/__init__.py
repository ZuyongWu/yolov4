"""
@File    :   __init__.py.py    
@Contact :   156618056@qq.com
@License :   Free
@Modified:   2020/8/13 20:30
@Author  :   Karol Wu
@Version :   1.0
@Des     :   None
"""
import os
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import Input, Lambda
from keras.utils import plot_model
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from nets.yolo4 import yolo_body
from nets.loss import yolo_loss
from keras.backend.tensorflow_backend import set_session
from utils.utils import get_random_data, get_random_data_with_Mosaic, rand, WarmUpCosineDecayScheduler

from utils.make_txt import makeTxt
from yolov4.Custom.utils import get_anchors, data_generator, preprocess_true_boxes, dummy_loss

import warnings
warnings.filterwarnings("ignore")


class Custom_Object_Detect_Training:
    """

    """
    def __init__(self, input_shape=(288, 288)):
        """
        input_shape=(416, 416) or (608, 608) or ..., depends on your memory
        """
        self.data_directory = None
        self.classes = None
        self.pretrain_model = None
        self.batch_size = None
        self.epochs = None
        self.log_dir = None
        self.anchors_path = None
        self.classes_path = None
        self.annotation_path = None

        self.input_shape = input_shape

    def setDataDirectory(self, data_directory=None, object_names=None):
        self.data_directory = data_directory
        self.classes = object_names
        makeTxt(data_directory=data_directory, classes=object_names, input_shape=self.input_shape[0])

        self.classes_path = os.path.join(self.data_directory, "classes.txt")
        self.annotation_path = os.path.join(self.data_directory, "train.txt")
        self.anchors_path = os.path.join(self.data_directory, "anchors.txt")
        self.log_dir = os.path.join(self.data_directory, "models/")

    def setTrainConfig(self, pretrain_model=None, batch_size=2, epochs=100):
        self.pretrain_model = pretrain_model
        self.batch_size = batch_size
        self.epochs = epochs

    def trainModel(self, mosaic=True, cosine_scheduler=True, label_smoothing=0.1):
        """
        """
        anchors = get_anchors(self.anchors_path)
        num_classes = len(self.classes)
        num_anchors = len(anchors)

        K.clear_session()

        image_input = Input(shape=(None, None, 3))
        h, w = self.input_shape

        print('Create YOLOv4 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
        # model_body = Model(image_input, [P5_output, P4_output, P3_output])
        model_body = yolo_body(image_input, num_anchors // 3, num_classes)

        print('Load weights {}.'.format(self.pretrain_model))
        model_body.load_weights(self.pretrain_model, by_name=True, skip_mismatch=True)

        # y_true = [Input(shape=(h//32,w//32,3,cls+5), Input(shape=(h//16,w//16,3,cls+5), Input(shape=(h//8,w//8,3,cls+5)]
        y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[i], w // {0: 32, 1: 16, 2: 8}[i], num_anchors // 3, num_classes + 5))
                  for i in range(3)]

        # model_body.output = [P5_output, P4_output, P3_output]
        loss_input = [*model_body.output, *y_true]
        model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                            arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5,
                                       'label_smoothing': label_smoothing})(loss_input)

        model = Model([model_body.input, *y_true], model_loss)
        # plot_model(model, to_file="yolov4_loss_model.png", show_shapes=True, show_layer_names=True)

        logging = TensorBoard(log_dir=self.log_dir)
        checkpoint = ModelCheckpoint(self.log_dir + 'epoch{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                     monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

        val_split = 0.1
        with open(self.annotation_path) as f:
            lines = f.readlines()
        np.random.seed(10101)
        np.random.shuffle(lines)
        np.random.seed(None)
        num_val = int(len(lines) * val_split)
        num_train = len(lines) - num_val

        # ------------------------------------------------------#
        #   backbone extract general feature in network
        #   freeze some head layers can speed up training, and prevent weights from influence in early epoch
        # ------------------------------------------------------#
        freeze_layers = 249
        for i in range(freeze_layers):
            model_body.layers[i].trainable = False
        print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model_body.layers)))

        init_epoch = 0
        freeze_epoch = self.epochs // 2
        batch_size = self.batch_size * 2
        learning_rate_base = 1e-3

        if cosine_scheduler:
            warm_up_epoch = int((freeze_epoch - init_epoch) * 0.2)
            total_steps = int((freeze_epoch - init_epoch) * num_train / batch_size)
            warm_up_steps = int(warm_up_epoch * num_train / batch_size)

            reduce_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                                   total_steps=total_steps,
                                                   warmup_learning_rate=1e-4,
                                                   warmup_steps=warm_up_steps,
                                                   hold_base_rate_steps=num_train,
                                                   min_learn_rate=1e-6)
            model.compile(optimizer=Adam(), loss=dummy_loss)
        else:
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, min_lr=1e-6)
            model.compile(optimizer=Adam(learning_rate_base), loss=dummy_loss)

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

        model.fit_generator(
            data_generator(lines[:num_train], batch_size, self.input_shape, anchors, num_classes, mosaic=mosaic),
            steps_per_epoch=max(1, num_train // batch_size),
            validation_data=data_generator(lines[num_train:], batch_size, self.input_shape, anchors, num_classes, mosaic=False),
            validation_steps=max(1, num_val // batch_size),
            epochs=freeze_epoch,
            initial_epoch=init_epoch,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping])

        model.save_weights(self.log_dir + 'trained_weights_stage_1.h5')

        for i in range(freeze_layers):
            model_body.layers[i].trainable = True

        print("\n\nStarting Training all Layers....\n\n")

        batch_size = self.batch_size

        learning_rate_base = 1e-4
        if cosine_scheduler:
            warm_up_epoch = int((self.epochs - freeze_epoch) * 0.2)
            total_steps = int((self.epochs - freeze_epoch) * num_train / batch_size)
            warm_up_steps = int(warm_up_epoch * num_train / batch_size)
            reduce_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                                   total_steps=total_steps,
                                                   warmup_learning_rate=1e-5,
                                                   warmup_steps=warm_up_steps,
                                                   hold_base_rate_steps=num_train // 2,
                                                   min_learn_rate=1e-6)
            model.compile(optimizer=Adam(), loss=dummy_loss)
        else:
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, min_lr=1e-6)
            model.compile(optimizer=Adam(learning_rate_base), loss=dummy_loss)

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

        model.fit_generator(
            data_generator(lines[:num_train], batch_size, self.input_shape, anchors, num_classes, mosaic=mosaic),
            steps_per_epoch=max(1, num_train // batch_size),
            validation_data=data_generator(lines[num_train:], batch_size, self.input_shape, anchors, num_classes, mosaic=False),
            validation_steps=max(1, num_val // batch_size),
            epochs=self.epochs,
            initial_epoch=freeze_epoch,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping])

        model.save_weights(self.log_dir + 'last1.h5')
