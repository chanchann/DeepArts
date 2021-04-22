import os

class Config:
    DEEPARTS_DIR = os.path.expanduser("~/.deeparts")
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{DEEPARTS_DIR}/deeparts.db"
    PORT = 7788