# -*- coding: utf-8 -*-
# @Date         : 2021-01-21
# @Author       : AaronJny
# @LastEditTime : 2021-04-05
# @FilePath     : /deeparts/deeparts/core/models/classifier/preset/pre_trained.py
# @Desc         : 封装tf.keras里设置的预训练模型，并对外提供支持
import os
from abc import ABC

import tensorflow as tf
from jinja2 import Template
from deeparts.core.models.classifier import deepartsImageClassifier
from loguru import logger


class deepartsPreTrainedImageClassifier(deepartsImageClassifier, ABC):
    def __init__(self, net_name, *args, **kwargs):
        super(deepartsPreTrainedImageClassifier, self).__init__(*args, **kwargs)
        self.net_name = net_name

    def define_model(self) -> tf.keras.Model:
        pre_trained_net: tf.keras.Model = getattr(
            tf.keras.applications, self.net_name
        )()
        pre_trained_net.trainable = False
        # 记录densenet
        self.pre_trained_net = pre_trained_net
        model = tf.keras.Sequential(
            [
                pre_trained_net,
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(len(self.classes_num_dict), activation="softmax"),
            ]
        )
        return model

    def train(self):
        # callbacks
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            self.model_save_path, monitor="val_accuracy", save_best_only=True
        )
        if self.do_fine_tune and self.freeze_epochs_ratio:
            # 如果选择了fine tune，则至少冻结训练一个epoch
            pre_train_epochs = max(1, int(self.freeze_epochs_ratio * self.epochs))
        else:
            pre_train_epochs = 0
        train_epochs = self.epochs - pre_train_epochs
        if pre_train_epochs:
            logger.info(
                f"分两步训练，冻结训练{pre_train_epochs}个epochs，解冻训练{train_epochs}个epochs..."
            )
        # 训练
        if pre_train_epochs:
            logger.info("冻结 pre-trained 模型，开始预训练 ...")
            self.model.fit(
                self.train_dataset.for_fit(),
                initial_epoch=0,
                epochs=pre_train_epochs,
                steps_per_epoch=self.train_dataset.steps,
                validation_data=self.dev_dataset.for_fit(),
                validation_steps=self.dev_dataset.steps,
                callbacks=[
                    checkpoint,
                ],
            )
        if train_epochs:
            logger.info("解冻 pre-trained 模型，继续训练 ...")
            self.pre_trained_net.trainable = True
            self.model.fit(
                self.train_dataset.for_fit(),
                initial_epoch=pre_train_epochs,
                epochs=self.epochs,
                steps_per_epoch=self.train_dataset.steps,
                validation_data=self.dev_dataset.for_fit(),
                validation_steps=self.dev_dataset.steps,
                callbacks=[
                    checkpoint,
                ],
            )
        logger.info("加载最优参数，输出验证集结果 ...")
        # self.model.load_weights(self.model_save_path, by_name=True)
        self.model.evaluate(self.dev_dataset.for_fit(), steps=self.dev_dataset.steps)

    def get_call_code(self):
        """返回模型定义和模型调用的代码"""
        if not self._call_code:
            template_path = os.path.join(
                os.path.dirname(__file__), "templates/deepartsPreTrainedImageClassifier.txt"
            )
            with open(template_path, "r") as f:
                text = f.read()
            data = {
                "net_name": self.net_name,
                "num_classes": len(self.classes_num_dict),
                "num_classes_map": str(self.classes_num_dict_rev),
                "image_size": self.image_size,
                "model_path": self.model_save_path,
                "data_preprocess_template": self.train_dataset.generate_preprocess_code(),
            }
            template = Template(text)
            code = template.render(**data)
            self._call_code = code
        return self._call_code

    def save_code(self):
        """导出模型定义和模型调用的代码"""
        code = self.get_call_code()
        code_file_name = "deeparts-code.py"
        code_path = os.path.join(self.project_save_path, code_file_name)
        with open(code_path, "w") as f:
            f.write(code)


class deepartsLeNetImageClassifier(deepartsPreTrainedImageClassifier):
    def __init__(self, *args, **kwargs):
        kwargs["net_name"] = "LeNet"
        kwargs["with_image_net"] = False
        kwargs["do_fine_tune"] = False
        kwargs["image_size"] = 32
        super(deepartsLeNetImageClassifier, self).__init__(*args, **kwargs)

    def define_model(self) -> tf.keras.Model:
        # todo: 为模型增加dropout和正则，以适当减轻过拟合
        model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(self.image_size, self.image_size, 3)),
                tf.keras.layers.Conv2D(6, (5, 5), padding="same"),
                # 添加BN层，将数据调整为均值0，方差1
                tf.keras.layers.BatchNormalization(),
                # 最大池化层，池化后图片长宽减半
                tf.keras.layers.MaxPooling2D((2, 2), 2),
                # relu激活层
                tf.keras.layers.ReLU(),
                # 第二个卷积层
                tf.keras.layers.Conv2D(16, (5, 5)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D((2, 2), 2),
                tf.keras.layers.ReLU(),
                # 将节点展平为(None,-1)的形式，以作为全连接层的输入
                tf.keras.layers.Flatten(),
                # 第一个全连接层，120个节点
                tf.keras.layers.Dense(120),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                # 第二个全连接层
                tf.keras.layers.Dense(84),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                # tf.keras.layers.Dropout(0.3),
                # 输出层，使用softmax激活
                tf.keras.layers.Dense(len(self.classes_num_dict), activation="softmax"),
            ]
        )
        # use_regularizer = True
        # if use_regularizer:
        #     for layer in model.layers:
        #         if hasattr(layer, "kernel_regularizer"):
        #             layer.kernel_regularizer = tf.keras.regularizers.l2(0.01)
        return model

    def train(self):
        # callbacks
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            self.model_save_path, monitor="val_accuracy", save_best_only=True
        )
        self.model.fit(
            self.train_dataset.for_fit(),
            epochs=self.epochs,
            steps_per_epoch=self.train_dataset.steps,
            validation_data=self.dev_dataset.for_fit(),
            validation_steps=self.dev_dataset.steps,
            callbacks=[
                checkpoint,
            ],
        )
        logger.info("加载最优参数，输出验证集结果 ...")
        self.model.load_weights(self.model_save_path)
        self.model.evaluate(self.dev_dataset.for_fit(), steps=self.dev_dataset.steps)


class deepartsDenseNet121ImageClassifier(deepartsPreTrainedImageClassifier):
    def __init__(self, net_name="DenseNet121", **kwargs):
        super(deepartsDenseNet121ImageClassifier, self).__init__(net_name, **kwargs)


class deepartsDenseNet169ImageClassifier(deepartsPreTrainedImageClassifier):
    def __init__(self, net_name="DenseNet169", **kwargs):
        super(deepartsDenseNet169ImageClassifier, self).__init__(net_name, **kwargs)


class deepartsDenseNet201ImageClassifier(deepartsPreTrainedImageClassifier):
    def __init__(self, net_name="DenseNet201", **kwargs):
        super(deepartsDenseNet201ImageClassifier, self).__init__(net_name, **kwargs)


class deepartsVGG16ImageClassifier(deepartsPreTrainedImageClassifier):
    def __init__(self, net_name="VGG16", **kwargs):
        super(deepartsVGG16ImageClassifier, self).__init__(net_name, **kwargs)


class deepartsVGG19ImageClassifier(deepartsPreTrainedImageClassifier):
    def __init__(self, net_name="VGG19", **kwargs):
        super(deepartsVGG19ImageClassifier, self).__init__(net_name, **kwargs)


class deepartsMobileNetImageClassifier(deepartsPreTrainedImageClassifier):
    def __init__(self, net_name="MobileNet", **kwargs):
        super(deepartsMobileNetImageClassifier, self).__init__(net_name, **kwargs)


class deepartsMobileNetV2ImageClassifier(deepartsPreTrainedImageClassifier):
    def __init__(self, net_name="MobileNetV2", **kwargs):
        super(deepartsMobileNetV2ImageClassifier, self).__init__(net_name, **kwargs)


class deepartsInceptionResNetV2ImageClassifier(deepartsPreTrainedImageClassifier):
    def __init__(self, net_name="InceptionResNetV2", **kwargs):
        super(deepartsInceptionResNetV2ImageClassifier, self).__init__(net_name, **kwargs)


class deepartsInceptionV3ImageClassifier(deepartsPreTrainedImageClassifier):
    def __init__(self, net_name="InceptionV3", **kwargs):
        super(deepartsInceptionV3ImageClassifier, self).__init__(net_name, **kwargs)


class deepartsNASNetMobileImageClassifier(deepartsPreTrainedImageClassifier):
    def __init__(self, net_name="NASNetMobile", **kwargs):
        super(deepartsNASNetMobileImageClassifier, self).__init__(net_name, **kwargs)


class deepartsNASNetLargeImageClassifier(deepartsPreTrainedImageClassifier):
    def __init__(self, net_name="NASNetLarge", **kwargs):
        super(deepartsNASNetLargeImageClassifier, self).__init__(net_name, **kwargs)


class deepartsResNet50ImageClassifier(deepartsPreTrainedImageClassifier):
    def __init__(self, net_name="ResNet50", **kwargs):
        super(deepartsResNet50ImageClassifier, self).__init__(net_name, **kwargs)


class deepartsResNet50V2ImageClassifier(deepartsPreTrainedImageClassifier):
    def __init__(self, net_name="ResNet50V2", **kwargs):
        super(deepartsResNet50V2ImageClassifier, self).__init__(net_name, **kwargs)


class deepartsResNet101ImageClassifier(deepartsPreTrainedImageClassifier):
    def __init__(self, net_name="ResNet101", **kwargs):
        super(deepartsResNet101ImageClassifier, self).__init__(net_name, **kwargs)


class deepartsResNet101V2ImageClassifier(deepartsPreTrainedImageClassifier):
    def __init__(self, net_name="ResNet101V2", **kwargs):
        super(deepartsResNet101V2ImageClassifier, self).__init__(net_name, **kwargs)


class deepartsResNet152ImageClassifier(deepartsPreTrainedImageClassifier):
    def __init__(self, net_name="ResNet152", **kwargs):
        super(deepartsResNet152ImageClassifier, self).__init__(net_name, **kwargs)


class deepartsResNet152V2ImageClassifier(deepartsPreTrainedImageClassifier):
    def __init__(self, net_name="ResNet152V2", **kwargs):
        super(deepartsResNet152V2ImageClassifier, self).__init__(net_name, **kwargs)


class deepartsMobileNetV3SmallImageClassifier(deepartsPreTrainedImageClassifier):
    def __init__(self, net_name="MobileNetV3Small", **kwargs):
        super(deepartsMobileNetV3SmallImageClassifier, self).__init__(net_name, **kwargs)


class deepartsMobileNetV3LargeImageClassifier(deepartsPreTrainedImageClassifier):
    def __init__(self, net_name="MobileNetV3Large", **kwargs):
        super(deepartsMobileNetV3LargeImageClassifier, self).__init__(net_name, **kwargs)


class deepartsXceptionImageClassifier(deepartsPreTrainedImageClassifier):
    def __init__(self, net_name="Xception", **kwargs):
        super(deepartsXceptionImageClassifier, self).__init__(net_name, **kwargs)


class deepartsEfficientNetB0ImageClassifier(deepartsPreTrainedImageClassifier):
    def __init__(self, net_name="EfficientNetB0", **kwargs):
        super(deepartsEfficientNetB0ImageClassifier, self).__init__(net_name, **kwargs)


class deepartsEfficientNetB1ImageClassifier(deepartsPreTrainedImageClassifier):
    def __init__(self, net_name="EfficientNetB1", **kwargs):
        super(deepartsEfficientNetB1ImageClassifier, self).__init__(net_name, **kwargs)


class deepartsEfficientNetB2ImageClassifier(deepartsPreTrainedImageClassifier):
    def __init__(self, net_name="EfficientNetB2", **kwargs):
        super(deepartsEfficientNetB2ImageClassifier, self).__init__(net_name, **kwargs)


class deepartsEfficientNetB3ImageClassifier(deepartsPreTrainedImageClassifier):
    def __init__(self, net_name="EfficientNetB3", **kwargs):
        super(deepartsEfficientNetB3ImageClassifier, self).__init__(net_name, **kwargs)


class deepartsEfficientNetB4ImageClassifier(deepartsPreTrainedImageClassifier):
    def __init__(self, net_name="EfficientNetB4", **kwargs):
        super(deepartsEfficientNetB4ImageClassifier, self).__init__(net_name, **kwargs)


class deepartsEfficientNetB5ImageClassifier(deepartsPreTrainedImageClassifier):
    def __init__(self, net_name="EfficientNetB5", **kwargs):
        super(deepartsEfficientNetB5ImageClassifier, self).__init__(net_name, **kwargs)


class deepartsEfficientNetB6ImageClassifier(deepartsPreTrainedImageClassifier):
    def __init__(self, net_name="EfficientNetB6", **kwargs):
        super(deepartsEfficientNetB6ImageClassifier, self).__init__(net_name, **kwargs)


class deepartsEfficientNetB7ImageClassifier(deepartsPreTrainedImageClassifier):
    def __init__(self, net_name="EfficientNetB7", **kwargs):
        super(deepartsEfficientNetB7ImageClassifier, self).__init__(net_name, **kwargs)