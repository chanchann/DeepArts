from flask import Flask

def create_app():
    app = Flask(
        __name__, static_folder="../dist", template_folder="../dist", static_url_path=""
    )
    return app

def run():
    app = create_app()
    app.run(host="0.0.0.0", port = 8888)

if __name__ == "__main__":
    run()