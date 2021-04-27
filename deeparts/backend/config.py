# -*- coding: utf-8 -*-
# @Author       : AaronJny
# @LastEditTime : 2021-01-28
# @FilePath     : /app/deeparts/backend/config.py
# @Desc         :
import os


class Config:

    deeparts_DIR = os.path.expanduser("~/.deeparts")

    SQLALCHEMY_DATABASE_URI = f"sqlite:///{deeparts_DIR}/deeparts.db"

    PORT = 7788
