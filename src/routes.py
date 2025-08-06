from flask import Blueprint
from src.api.routes.predict import predictRouter

api = Blueprint('api', __name__)

api.register_blueprint(predictRouter, url_prefix="/predict")