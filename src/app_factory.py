from flask import Flask
from dotenv import load_dotenv

from src.core.config.config import Config
from src.routes import api
from flask_cors import CORS

def create_app() -> Flask:
    load_dotenv()

    app = Flask(__name__)
    config = Config().dev_config
    app.env = config.ENV

    app.register_blueprint(api, url_prefix="/api")
    CORS(app,
         resources={r"/api/*": {"origins": "http://localhost:5173"}},
         methods=["GET", "POST", "OPTIONS"]
         )


    return app




