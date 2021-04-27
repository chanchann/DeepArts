# -*- coding: utf-8 -*-

from deeparts.core.models.classifier.kerastuner.image_classifier import deepartsKerasTunerImageClassifier

path = './images2'
deeparts_classifier = deepartsKerasTunerImageClassifier(path, path)
deeparts_classifier.run()
