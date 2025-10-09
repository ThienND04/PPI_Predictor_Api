from flask import Flask
from dotenv import load_dotenv

from src.core.config.config import Config
from src.routes import api
from flask_cors import CORS
from src.extensions import limiter
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.engine.url import make_url
import os
from sqlalchemy import create_engine, text, inspect
from sqlalchemy_utils import database_exists, create_database

def create_app() -> Flask:
    load_dotenv()

    app = Flask(__name__)
    config = Config().dev_config
    app.env = config.ENV

    # SQLAlchemy setup via connection string from env
    # Accept several common env var names for flexibility
    connection_string = (
        os.getenv('DB_CONNECTION_STRING')
        or os.getenv('DATABASE_URL')
        or os.getenv('SQLALCHEMY_DATABASE_URI')
    )

    if not connection_string:
        raise RuntimeError('Database connection string is not set (expected DB_CONNECTION_STRING, DATABASE_URL, or SQLALCHEMY_DATABASE_URI)')

    # Normalize legacy PostgreSQL scheme if present (SQLAlchemy expects postgresql://)
    if connection_string.startswith('postgres://'):
        connection_string = 'postgresql://' + connection_string[len('postgres://'):]

    # Enforce sslmode=require for Postgres/Neon if not explicitly set
    if connection_string.startswith('postgresql://') and 'sslmode=' not in connection_string:
        separator = '&' if '?' in connection_string else '?'
        connection_string = f"{connection_string}{separator}sslmode=require"

    app.config['SQLALCHEMY_DATABASE_URI'] = connection_string
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    from src.models import db
    db.init_app(app)

    # Bootstrap database and tables idempotently
    try:
        # Create database if missing (when credentials have create privilege)
        if not database_exists(connection_string):
            print("[DB] Database not found, creating…")
            create_database(connection_string)
            print("[DB] Database created.")

        # Ensure tables exist
        with app.app_context():
            inspector = inspect(db.engine)
            existing_tables = set(inspector.get_table_names())
            required_tables = {"users", "verification_codes", "password_reset_codes"}

            if not required_tables.issubset(existing_tables):
                print("[DB] Creating missing tables…")
                # Import models to register metadata
                from src.models.user import User  # noqa: F401
                from src.models.verification import VerificationCode  # noqa: F401
                from src.models.password_reset import PasswordResetCode  # noqa: F401
                db.create_all()
                print("[DB] Creating missing tables… done.")
    except Exception as e:
        # Do not crash app startup if we cannot create; log and proceed
        print(f"[DB] Bootstrap warning: {e}")

    # Init global limiter
    limiter.init_app(app)

    # error handler for 429
    @app.errorhandler(429)
    def ratelimit_handler(e):
        return {"error": "Rate limit exceeded. Please try again later."}, 429

    app.register_blueprint(api, url_prefix="/api")
    CORS(app,
         resources={r"/api/*": {"origins": ["http://localhost:5173", "http://127.0.0.1:5173"]}},
         methods=["GET", "POST", "OPTIONS", "PUT", "DELETE"],
         allow_headers=["Content-Type", "Authorization", "Accept"],
         supports_credentials=True
         )


    return app




