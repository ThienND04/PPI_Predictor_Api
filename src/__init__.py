from flask import Flask
from src.core.config.config import Config
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
config = Config().dev_config
app.env = config.ENV

from src.routes import api
app.register_blueprint(api, url_prefix="/api")


