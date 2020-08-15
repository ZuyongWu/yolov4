"""
@File    :   04_custom_training.py    
@Contact :   156618056@qq.com
@License :   Free
@Modified:   2020/8/13 23:16
@Author  :   Karol Wu
@Version :   1.0 
@Des     :   None
"""
from yolov4.Custom import Custom_Object_Detect_Training


path_to_model = "model_data/yolo4_voc_weights.h5"

trainer = Custom_Object_Detect_Training()
trainer.setDataDirectory(data_directory="facial_mask", object_names=["facial mask"])
trainer.setTrainConfig(batch_size=2, epochs=100, pretrain_model=path_to_model)

trainer.trainModel(mosaic=False)  # default setting is True, means using mosaic to do data augmentation
