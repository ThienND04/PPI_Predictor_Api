from flask import Blueprint
from src.api.routes.predict import predictRouter
from src.api.routes.auth import authRouter

api = Blueprint('api', __name__)

api.register_blueprint(predictRouter, url_prefix="/predict")
api.register_blueprint(authRouter, url_prefix="/auth")