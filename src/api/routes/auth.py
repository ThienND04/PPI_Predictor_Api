from flask import Blueprint, request, jsonify
from sqlalchemy import and_
from datetime import datetime, timedelta, timezone
import os
import re
import random

from src.models import db
from src.models.user import User
from src.models.verification import VerificationCode
from src.models.password_reset import PasswordResetCode
from src.utils.security import hash_password, verify_password, generate_jwt
from src.utils.email_utils import send_verification_email, send_reset_otp_email
from src.api.schemas.AuthSchemas import RegisterInput, VerifyInput, ResendCodeInput, LoginInput, ForgotPasswordInput, ResetPasswordInput, ChangePasswordInput


authRouter = Blueprint('auth_routes', __name__)


def _validate_email(email: str) -> bool:
    return re.match(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$", email) is not None


@authRouter.route('/register', methods=['POST'])
def register():
    try:
        # Accept JSON or form-encoded
        raw = request.get_json(silent=True)
        if raw is None:
            raw = request.form.to_dict() if request.form else {}

        parsed = RegisterInput.model_validate(raw)
        email = parsed.email.lower()
        full_name = parsed.full_name.strip()
        password = parsed.password

        if not _validate_email(email) or len(password) < 6:
            return jsonify({"error": "Invalid email or password format"}), 400
        if not full_name:
            return jsonify({"error": "Full name is required"}), 400

        existing = User.query.filter_by(email=email).first()
        if existing:
            return jsonify({"error": "Email already registered"}), 409

        user = User(email=email, full_name=full_name, password_hash=hash_password(password))
        db.session.add(user)
        db.session.commit()

        return jsonify({"message": "User registered successfully"}), 201
    except Exception as e:
        db.session.rollback()
        print(f"Error in register: {e}")
        # Validation errors
        try:
            from pydantic import ValidationError
            if isinstance(e, ValidationError):
                return jsonify({"error": "Invalid email or password format"}), 400
        except Exception:
            pass
        return jsonify({"error": "Internal server error"}), 500


@authRouter.route('/verify', methods=['POST'])
def verify():
    try:
        raw = request.get_json(silent=True)
        if raw is None:
            raw = request.form.to_dict() if request.form else {}
        parsed = VerifyInput.model_validate(raw)
        email = parsed.email.lower()
        code = parsed.code.strip()

        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"error": "User not found"}), 400

        now = datetime.now(timezone.utc)
        v = VerificationCode.query.filter(
            VerificationCode.user_id == user.id,
            VerificationCode.code == code,
            VerificationCode.used.is_(False),
            VerificationCode.expires_at > now
        ).order_by(VerificationCode.created_at.desc()).first()

        if not v:
            return jsonify({"error": "Invalid or expired verification code"}), 400

        user.is_verified = True
        v.used = True
        db.session.commit()
        return jsonify({"message": "Email verified successfully"})
    except Exception as e:
        db.session.rollback()
        try:
            from pydantic import ValidationError
            if isinstance(e, ValidationError):
                return jsonify({"error": "Validation error", "details": e.errors()}), 400
        except Exception:
            pass
        return jsonify({"error": "Server error"}), 500


@authRouter.route('/resend-code', methods=['POST'])
def resend_code():
    try:
        raw = request.get_json(silent=True)
        if raw is None:
            raw = request.form.to_dict() if request.form else {}
        parsed = ResendCodeInput.model_validate(raw)
        email = parsed.email.lower()

        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"error": "User not found"}), 400
        if user.is_verified:
            return jsonify({"message": "User already verified"}), 200

        # Optionally mark old codes as used
        VerificationCode.query.filter_by(user_id=user.id, used=False).update({VerificationCode.used: True})

        exp_min = int(os.getenv('VERIFICATION_EXP_MIN', '10'))
        code = f"{random.randint(0, 999999):06d}"
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=exp_min)
        v = VerificationCode(user_id=user.id, code=code, expires_at=expires_at, used=False)
        db.session.add(v)
        db.session.commit()

        frontend_base = os.getenv('FRONTEND_BASE_URL')
        link = None
        if frontend_base:
            link = f"{frontend_base.rstrip('/')}/verify?email={email}&code={code}"

        try:
            send_verification_email(email, code, link)
        except Exception:
            return jsonify({"error": "Failed to send verification email"}), 500

        return jsonify({"message": "Verification code resent"})
    except Exception as e:
        db.session.rollback()
        try:
            from pydantic import ValidationError
            if isinstance(e, ValidationError):
                return jsonify({"error": "Validation error", "details": e.errors()}), 400
        except Exception:
            pass
        return jsonify({"error": "Server error"}), 500


@authRouter.route('/login', methods=['POST'])
def login():
    try:
        raw = request.get_json(silent=True)
        if raw is None:
            raw = request.form.to_dict() if request.form else {}
        parsed = LoginInput.model_validate(raw)
        email = parsed.email.lower()
        password = parsed.password

        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"error": "Email not found"}), 404
        if not verify_password(password, user.password_hash):
            return jsonify({"error": "Invalid password"}), 401

        token = generate_jwt({"sub": user.id, "email": user.email})
        return jsonify({"message": "Login successful", "token": token}), 200
    except Exception as e:
        print(f"Error in login: {e}")
        try:
            from pydantic import ValidationError
            if isinstance(e, ValidationError):
                return jsonify({"error": "Invalid email or password format"}), 400
        except Exception:
            pass
        return jsonify({"error": "Internal server error"}), 500


@authRouter.route('/forgot-password', methods=['POST'])
def forgot_password():
    try:
        raw = request.get_json(silent=True)
        if raw is None:
            raw = request.form.to_dict() if request.form else {}
        parsed = ForgotPasswordInput.model_validate(raw)
        email = parsed.email.lower()

        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"error": "Email not found"}), 404

        exp_min = int(os.getenv('RESET_OTP_EXP_MIN', '10'))
        otp_code = f"{random.randint(100000, 999999):06d}"
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=exp_min)

        reset = PasswordResetCode(email=email, otp_code=otp_code, expires_at=expires_at)
        db.session.add(reset)
        db.session.commit()

        try:
            send_reset_otp_email(email, otp_code, exp_min)
        except Exception as e:
            print(f"Error in send_reset_otp_email: {e}")
            return jsonify({"error": "Failed to send OTP email"}), 500

        return jsonify({"message": "OTP sent to your email."})
    except Exception as e:
        try:
            from pydantic import ValidationError
            if isinstance(e, ValidationError):
                return jsonify({"error": "Validation error", "details": e.errors()}), 400
        except Exception:
            pass
        return jsonify({"error": "Server error"}), 500


@authRouter.route('/reset-password', methods=['POST'])
def reset_password():
    try:
        raw = request.get_json(silent=True)
        if raw is None:
            raw = request.form.to_dict() if request.form else {}
        parsed = ResetPasswordInput.model_validate(raw)
        email = parsed.email.lower()
        otp_code = parsed.otp_code.strip()
        new_password = parsed.new_password

        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"error": "Email not found"}), 404

        now = datetime.now(timezone.utc)
        otp = PasswordResetCode.query.filter(
            PasswordResetCode.email == email,
            PasswordResetCode.otp_code == otp_code,
            PasswordResetCode.expires_at > now
        ).order_by(PasswordResetCode.created_at.desc()).first()

        if not otp:
            return jsonify({"error": "Invalid or expired OTP"}), 400

        if len(new_password) < 6:
            return jsonify({"error": "Invalid password format"}), 400

        user.password_hash = hash_password(new_password)
        db.session.delete(otp)
        db.session.commit()

        return jsonify({"message": "Password reset successfully"}), 200
    except Exception as e:
        try:
            from pydantic import ValidationError
            if isinstance(e, ValidationError):
                return jsonify({"error": "Invalid password format"}), 400
        except Exception:
            pass
        return jsonify({"error": "Internal server error"}), 500


@authRouter.route('/change-password', methods=['POST'])
def change_password():
    try:
        raw = request.get_json(silent=True)
        if raw is None:
            raw = request.form.to_dict() if request.form else {}
        parsed = ChangePasswordInput.model_validate(raw)
        email = parsed.email.lower()
        old_password = parsed.old_password
        new_password = parsed.new_password

        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"error": "User not found"}), 404

        if not verify_password(old_password, user.password_hash):
            return jsonify({"error": "Old password incorrect"}), 401

        if len(new_password) < 6 or verify_password(new_password, user.password_hash):
            return jsonify({"error": "New password invalid"}), 400

        user.password_hash = hash_password(new_password)
        db.session.commit()
        return jsonify({"message": "Password changed successfully"}), 200
    except Exception as e:
        try:
            from pydantic import ValidationError
            if isinstance(e, ValidationError):
                return jsonify({"error": "New password invalid"}), 400
        except Exception:
            pass
        return jsonify({"error": "Internal server error"}), 500


