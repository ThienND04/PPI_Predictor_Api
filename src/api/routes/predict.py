from logging import Logger
from datetime import datetime, timezone

from flask import Blueprint, request, jsonify
from src.api.schemas.PredictInput import PredictInput
from pydantic import ValidationError

from src.services.registry import get_runner
from src.core.logger import get_logger

logger = get_logger(__name__)

predictRouter = Blueprint('api_routes', __name__)

@predictRouter.route('', methods=['POST'])
def predict():
    try:
        json_data = request.get_json()
        if not json_data:
            return jsonify({"error": "No JSON data provided"}), 400
        data = PredictInput(**json_data)

        runner = get_runner(data.model)
        result = runner.predict(data.id1, data.seq1, data.id2, data.seq2)

        score = result
        threshold = 0.5
        label = "interaction" if score is not None and score >= threshold else "no_interaction"

        response = {
            "protein1": {"id": data.id1},
            "protein2": {"id": data.id2},
            "model": data.model,
            "score": round(score, 4) if score is not None else None,
            "label": label,
            "threshold": threshold,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        }
        return jsonify(response)
    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        return jsonify({"error": repr(e.errors()[0])}), 500
    except ValueError as e:
        logger.error(f"Value error: {e}")
        return jsonify({"error": str(e)}), 400
    except TimeoutError:
        logger.error("Model prediction timeout")
        return jsonify({"error": "Prediction timeout"}), 408
    except Exception as e:
        logger.error(f"Unexpected error in predict: {e}", exc_info=True)
        return jsonify({"error": "Server error"}), 500
