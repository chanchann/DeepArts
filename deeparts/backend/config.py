# -*- coding: utf-8 -*-

import os


class Config:

    deeparts_DIR = os.path.expanduser("~/.deeparts")

    SQLALCHEMY_DATABASE_URI = f"sqlite:///{deeparts_DIR}/deeparts.db"

    PORT = 7788
