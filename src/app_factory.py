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
from flasgger import Swagger

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
            required_tables = {"users", "verification_codes", "password_reset_codes", "prediction_results"}

            if not required_tables.issubset(existing_tables):
                print("[DB] Creating missing tables…")
                # Import models to register metadata
                from src.models.user import User  # noqa: F401
                from src.models.verification import VerificationCode  # noqa: F401
                from src.models.password_reset import PasswordResetCode  # noqa: F401
                from src.models.prediction_result import PredictionResult  # noqa: F401
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
    
    # Configure Swagger UI
    swagger_config = {
        "headers": [],
        "specs": [
            {
                "endpoint": "apispec",
                "route": "/apispec.json",
                "rule_filter": lambda rule: True,
                "model_filter": lambda tag: True,
            }
        ],
        "static_url_path": "/flasgger_static",
        "swagger_ui": True,
        "specs_route": "/apidocs/"
    }
    
    swagger_template = {
        "swagger": "2.0",
        "info": {
            "title": "PPI Predictor API",
            "description": "API for Protein-Protein Interaction (PPI) Prediction using machine learning models",
            "version": "1.0.0",
            "contact": {
                "name": "API Support"
            }
        },
        "basePath": "/api",
        "schemes": ["http", "https"],
        "securityDefinitions": {
            "Bearer": {
                "type": "apiKey",
                "name": "Authorization",
                "in": "header",
                "description": "JWT Authorization header using the Bearer scheme. Example: 'Bearer {token}'"
            }
        },
        "consumes": ["application/json"],
        "produces": ["application/json"],
        "definitions": {
            "Error": {
                "type": "object",
                "properties": {
                    "error": {
                        "type": "string",
                        "example": "Error message"
                    }
                }
            },
            "ValidationError": {
                "type": "object",
                "properties": {
                    "error": {
                        "type": "string",
                        "example": "Validation error"
                    },
                    "details": {
                        "type": "array",
                        "items": {
                            "type": "object"
                        }
                    }
                }
            },
            "PredictionRecord": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "integer",
                        "example": 123
                    },
                    "model_name": {
                        "type": "string",
                        "example": "MCAPST5"
                    },
                    "protein1_id": {
                        "type": "string",
                        "example": "P12345"
                    },
                    "protein2_id": {
                        "type": "string",
                        "example": "Q98765"
                    },
                    "score": {
                        "type": "number",
                        "format": "float",
                        "example": 0.7845
                    },
                    "label": {
                        "type": "string",
                        "enum": ["interaction", "no_interaction"],
                        "example": "interaction"
                    },
                    "timestamp": {
                        "type": "string",
                        "format": "date-time",
                        "example": "2026-01-29T10:30:00Z"
                    }
                }
            },
            "PredictionResponse": {
                "type": "object",
                "properties": {
                    "protein1": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "example": "P12345"
                            }
                        }
                    },
                    "protein2": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "example": "Q98765"
                            }
                        }
                    },
                    "model": {
                        "type": "string",
                        "example": "MCAPST5"
                    },
                    "score": {
                        "type": "number",
                        "format": "float",
                        "example": 0.7845
                    },
                    "label": {
                        "type": "string",
                        "enum": ["interaction", "no_interaction"],
                        "example": "interaction"
                    },
                    "threshold": {
                        "type": "number",
                        "example": 0.5
                    },
                    "timestamp": {
                        "type": "string",
                        "format": "date-time",
                        "example": "2026-01-29T10:30:00Z"
                    }
                }
            },
            "HistoryResponse": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "example": "123"
                    },
                    "total_records": {
                        "type": "integer",
                        "example": 25
                    },
                    "predictions": {
                        "type": "array",
                        "items": {
                            "$ref": "#/definitions/PredictionRecord"
                        }
                    }
                }
            }
        }
    }
    
    Swagger(app, config=swagger_config, template=swagger_template)
    
    CORS(app,
         resources={
             r"/api/*": {"origins": ["http://localhost:5173", "http://127.0.0.1:5173"]},
             r"/apidocs/*": {"origins": "*"},
             r"/apispec.json": {"origins": "*"},
             r"/flasgger_static/*": {"origins": "*"}
         },
         methods=["GET", "POST", "OPTIONS", "PUT", "DELETE"],
         allow_headers=["Content-Type", "Authorization", "Accept"],
         supports_credentials=True
         )


    return app




