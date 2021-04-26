import json
import os
import re
import time

from flask import request
from flask.blueprints import Blueprint

from deeparts.backend import status_code_wrapper
from deeparts.backend.model import TrainProject, db
# from deeparts.core.models import image

api_v1_blueprint = Blueprint("api_v1_blueprint", __name__, url_prefix="/api/v1")

ENGINE_LIST = [
    {"index": 1, "name": "预设模型", "tip": "预先定义好的、结构固定的模型"},
    # {"index": 2, "name": "AutoKeras", "tip": ""},
    # {"index": 3, "name": "KerasTuner", "tip": ""},
    # {"index": 4, "name": "NNI", "tip": ""},
]

def check_path_correct(path):
    """检查给定路径是否符合要求，不符合要求则会抛出异常
    Args:
        path (str): 待检查路径
    """
    if not os.path.exists(path):
        raise Exception(f"指定路径 {path} 不存在！")
    if not os.path.isdir(path):
        raise Exception(f"{path} 必须是文件夹！")

@api_v1_blueprint.route("/")
def index():
    return "hello world"

@api_v1_blueprint.route("/hello/")
def hello():
    return "hello!!!"

# @api_v1_blueprint.route("/image/classifier/engines/")
# @status_code_wrapper()
# def get_engine_list():
#     data = ENGINE_LIST
#     return data

# @api_v1_blueprint.route("/hello/")
# def hello():
#     return "hello!!!"
