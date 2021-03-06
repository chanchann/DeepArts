# -*- coding: utf-8 -*-

import tensorflow as tf


def extract_image_and_label_from_record(example_string):
    feature_description = {  # 定义Feature结构，告诉解码器每个Feature的类型是什么
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "num": tf.io.FixedLenFeature([], tf.int64),
    }
    feature_dict = tf.io.parse_single_example(example_string, feature_description)
    feature_dict["image"] = tf.io.decode_jpeg(feature_dict["image"], channels=3)
    # 将标签转成one-hot形式
    y = tf.cast(feature_dict["label"], dtype=tf.int32)
    y = tf.one_hot(y, tf.cast(feature_dict["num"], dtype=tf.int32))
    return feature_dict["image"], y


def normalized_image(image, label, image_size):
    # min_size = 28
    # max_size = 224
    # # 限定图片大小
    # image_size = max(min_size, image.shape[0])
    # image_size = min(image_size, max_size)
    # 缩放图片
    x = tf.image.resize(image, [image_size, image_size])
    # 将图片的像素值缩放到[0,1]之间
    x = tf.cast(x, dtype=tf.float32) / 255.0
    return x, label


def normalized_image_with_imagenet(image, label, image_size):
    # imagenet数据集均值
    image_mean = [0.485, 0.456, 0.406]
    # imagenet数据集标准差
    image_std = [0.299, 0.224, 0.225]
    # 缩放图片
    x = tf.image.resize(image, [image_size, image_size])
    # 将图片的像素值缩放到[0,1]之间
    x = tf.cast(x, dtype=tf.float32) / 255.0
    # 归一化
    x = (x - image_mean) / image_std
    return x, label
