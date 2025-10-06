from logging import Logger
from datetime import datetime, timezone

from flask import Blueprint, request, jsonify, send_file
from src.api.schemas.PredictInput import PredictInput
from pydantic import ValidationError

from src.services.registry import get_runner
from src.core.logger import get_logger
from src.utils.fasta_parser import parse_fasta, parse_pairs
import io
import csv
import tempfile
import os

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


@predictRouter.route('/batch', methods=['POST'])
def predict_batch():
    try:
        if 'fasta_file' not in request.files or 'pairs_file' not in request.files:
            return jsonify({"error": "Missing required files 'fasta_file' and/or 'pairs_file'"}), 400

        fasta_file = request.files['fasta_file']
        pairs_file = request.files['pairs_file']

        if fasta_file.filename == '' or pairs_file.filename == '':
            return jsonify({"error": "Empty filename for 'fasta_file' or 'pairs_file'"}), 400

        # Optional: choose model via form field, fallback to default
        model_name = request.form.get('model', None)
        if model_name is None:
            # Try to use default model from existing single predict flow
            # If PredictInput enforces model, we can set a sensible default here
            model_name = 'MCAPST5'

        # Prepare runner once (do not reload per pair)
        runner = get_runner(model_name)

        # Parse FASTA and pairs in a streaming-friendly manner
        fasta_map = parse_fasta(fasta_file.stream)
        pairs_iter = list(parse_pairs(pairs_file.stream))

        threshold = 0.5
        successful_count = 0
        failed_count = 0

        # Prepare temp file to write results
        batch_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
        tmp_path = tmp_file.name
        tmp_file.close()  # Will reopen with text mode for csv writer

        try:
            with open(tmp_path, 'w', newline='') as f:
                # Write only successful lines: "id1 id2 score" (space separated), no header
                for id1, id2 in pairs_iter:
                    try:
                        if id1 not in fasta_map or id2 not in fasta_map:
                            failed_count += 1
                            continue

                        seq1 = fasta_map[id1]
                        seq2 = fasta_map[id2]

                        score = runner.predict(id1, seq1, id2, seq2)
                        if score is None:
                            failed_count += 1
                            continue
                        f.write(f"{id1} {id2} {score:.4f}\n")
                        successful_count += 1
                    except Exception:
                        failed_count += 1

            resp = send_file(tmp_path, mimetype='text/plain', as_attachment=True, download_name='ppi_results.txt')
            # Attach metadata as headers
            resp.headers['X-Model'] = model_name
            resp.headers['X-Threshold'] = str(threshold)
            resp.headers['X-Timestamp'] = batch_timestamp
            resp.headers['X-Total-Pairs'] = str(len(pairs_iter))
            resp.headers['X-Successful-Predictions'] = str(successful_count)
            resp.headers['X-Failed-Predictions'] = str(failed_count)
            return resp
        finally:
            # Best-effort cleanup: remove file after sending if possible via after_this_request could be better,
            # but keep simple here. Do not remove immediately as send_file needs the path.
            # Caller OS will eventually clean temp files if server process restarts.
            pass
    except ValueError as e:
        logger.error(f"Value error in batch: {e}")
        return jsonify({"error": str(e)}), 400
    except TimeoutError:
        logger.error("Model prediction timeout in batch")
        return jsonify({"error": "Prediction timeout"}), 408
    except Exception as e:
        logger.error(f"Unexpected error in predict_batch: {e}", exc_info=True)
        return jsonify({"error": "Server error"}), 500
