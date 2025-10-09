from sqlalchemy import func
from datetime import datetime
from src.models import db


class VerificationCode(db.Model):
    __tablename__ = 'verification_codes'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    code = db.Column(db.String(6), nullable=False, index=True)
    expires_at = db.Column(db.DateTime(timezone=True), nullable=False)
    used = db.Column(db.Boolean, nullable=False, default=False)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, server_default=func.now())

    user = db.relationship('User', back_populates='verifications')


