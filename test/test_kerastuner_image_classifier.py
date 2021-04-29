# -*- coding: utf-8 -*-
# @Date         : 2021-01-20
# @Author       : AaronJny
# @LastEditTime : 2021-01-20
# @FilePath     : /deeparts/test/test_kerastuner_image_classifier.py
# @Desc         :
from deeparts.core.models.classifier.kerastuner.image_classifier import deepartsKerasTunerImageClassifier

path = './images2'
deeparts_classifier = deepartsKerasTunerImageClassifier(path, path)
deeparts_classifier.run()
