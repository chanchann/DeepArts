# -*- coding: utf-8 -*-

from deeparts.core.models.complex.od.models import deepartsTFModelsObjectDetector


deepartsTFModelsObjectDetector(
    origin_dataset_path="/Users/aaron/od_api/VOCdevkit/VOC2007/yd_train_clean",
    tfrecord_dataset_path="/Users/aaron/test_deeparts/dataset/",
    fine_tune_model_name="SSD ResNet50 V1 FPN 640x640 (RetinaNet50)",
    model_save_path="~/test_deeparts/",
    batch_size=1,
).run()
