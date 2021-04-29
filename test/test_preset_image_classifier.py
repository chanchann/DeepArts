# -*- coding: utf-8 -*-

from deeparts.core.models.image import (
    deepartsResNet50V2ImageClassifier as ImageClassifier,
)

# path = "./images2/"
# path = "/app/images2"
ImageClassifier(
    origin_dataset_path="/Users/aaron/test_kaggle_api/chinese-click-demo-api/data",
    target_dataset_path="/Users/aaron/test_deeparts",
    model_save_path="/Users/aaron/test_deeparts",
    epochs=10,
).run()
