from flask import Flask
from flask_cors import CORS

from deeparts.backend.config import Config
from deeparts.backend.model import db

def create_app():
    # app = Flask(__name__, static_folder="../dist", template_folder="../dist", static_url_path="")
    app = Flask(__name__)
    app.config.from_object(Config)
    cors = CORS()

    cors.init_app(app)
    db.init_app(app)
    with app.app_context():
        # 初始化数据库
        db.create_all()
    from deeparts.backend.v1 import api_v1_blueprint
    app.register_blueprint(api_v1_blueprint)

    return app

def run():
    app = create_app()
    app.run(host="127.0.0.1", port = Config.PORT)

if __name__ == "__main__":
    run()