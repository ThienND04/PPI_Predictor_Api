from src.models import db
from sqlalchemy.sql import func

class PredictionResult(db.Model):
    __tablename__ = 'prediction_results'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    model_name = db.Column(db.String(64), nullable=False)
    protein1_id = db.Column(db.String(255), nullable=False)
    protein2_id = db.Column(db.String(255), nullable=False)
    score = db.Column(db.Float, nullable=False)
    label = db.Column(db.String(32), nullable=False)
    threshold = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime(timezone=True), server_default=func.now(), nullable=False)
