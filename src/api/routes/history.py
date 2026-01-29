from flask import Blueprint, request, jsonify, current_app
from typing import List, Dict, Any, Optional
from src.utils.security import verify_jwt
from sqlalchemy import text, inspect
from src.models import db
from src.api.schemas.HistorySchemas import HistoryResponse
from flasgger import swag_from
import os

historyRouter = Blueprint("historyRouter", __name__)

@historyRouter.get("/")
@swag_from(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'static/swagger_specs/history/get_all.yml'))
def get_history():
    print("Received /history request")
    user_id = None
    try:
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            token = auth_header.split(' ', 1)[1].strip()
            payload = verify_jwt(token)
            user_id = payload.get('sub')
    except Exception as e:
        print(f"JWT verification failed: {e}")
        return jsonify({"error": "Unauthorized"}), 401
    
    print("User ID:", user_id)

    # Query database for prediction history that matches the schema
    try:
        bind = db.session.get_bind()
        if bind is None:
            raise RuntimeError("Database engine not available")

        inspector = inspect(bind)
        records: Optional[List[Dict[str, Any]]] = []
        existing_tables = set(inspector.get_table_names())

        if "prediction_results" in existing_tables:
            cols_info = inspector.get_columns("prediction_results")
            table_cols = [c.get("name") for c in cols_info if c.get("name")]

            # ORDER BY preference
            order_by_clause = ""
            if "created_at" in table_cols:
                order_by_clause = " ORDER BY created_at DESC, id DESC" if "id" in table_cols else " ORDER BY created_at DESC"
            elif "id" in table_cols:
                order_by_clause = " ORDER BY id DESC"

            # Only select columns needed by PredictionRecordOut (+ created_at if exists for timestamp)
            select_cols = [
                "id",
                "user_id",
                "model_name",
                "protein1_id",
                "protein2_id",
                "score",
                "label",
            ]
            include_created = "created_at" in table_cols
            if include_created:
                select_cols.append("created_at")

            select_list = ", ".join(select_cols)
            query = text(
                f"SELECT {select_list} FROM prediction_results WHERE user_id = :uid{order_by_clause}"
            )

            result = db.session.execute(query, {"uid": user_id})
            rows = result.fetchall()
            print(f"Fetched {len(rows)} records for user {user_id}")

            predictions: List[Dict[str, Any]] = []
            for row in rows:
                mapping = getattr(row, "_mapping", None)
                if mapping is not None:
                    row_dict: Dict[str, Any] = dict(mapping)
                else:
                    row_dict = {}
                    for idx, name in enumerate(select_cols):
                        try:
                            row_dict[name] = row[idx]
                        except Exception:
                            pass

                # Build timestamp from created_at if present
                timestamp_val = None
                if include_created:
                    created_at_val = row_dict.get("created_at")
                    try:
                        import datetime as _dt
                        if isinstance(created_at_val, (_dt.datetime, _dt.date)):
                            dt = created_at_val if isinstance(created_at_val, _dt.datetime) else _dt.datetime.combine(created_at_val, _dt.time())
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=_dt.timezone.utc)
                            timestamp_val = dt.astimezone(_dt.timezone.utc).isoformat().replace("+00:00", "Z")
                        elif created_at_val is not None:
                            timestamp_val = str(created_at_val)
                    except Exception:
                        timestamp_val = None

                # Map exactly to PredictionRecordOut fields
                rec: Dict[str, Any] = {
                    "id": row_dict.get("id"),
                    "model_name": row_dict.get("model_name"),
                    "protein1_id": row_dict.get("protein1_id"),
                    "protein2_id": row_dict.get("protein2_id"),
                    "score": float(row_dict["score"]) if row_dict.get("score") is not None else None,
                    "label": row_dict.get("label"),
                    "timestamp": timestamp_val,
                }
                predictions.append(rec)

            records = predictions
        else:
            records = []

        response_model = HistoryResponse(
            user_id=user_id,
            total_records=len(records),
            predictions=records,
        )
        payload_dict = (
            response_model.model_dump() if hasattr(response_model, "model_dump") else response_model.dict()
        )
        print(jsonify(payload_dict))
        return jsonify(payload_dict), 200

    except Exception as e:
        print(f"Error fetching history: {e}")
        return jsonify({"error": "Database query failed"}), 500

@historyRouter.get("/<int:history_id>")
@swag_from(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'static/swagger_specs/history/get_by_id.yml'))
def get_history_by_id(history_id: int):
    user_id: Optional[Any] = None
    try:
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            token = auth_header.split(' ', 1)[1].strip()
            payload = verify_jwt(token)
            user_id = payload.get('sub')
    except Exception as e:
        print(f"JWT verification failed: {e}")
        return jsonify({"error": "Unauthorized"}), 401

    if user_id is None:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        bind = db.session.get_bind()
        if bind is None:
            raise RuntimeError("Database engine not available")
        inspector = inspect(bind)

        if "prediction_results" not in set(inspector.get_table_names()):
            return jsonify({"error": "Not found"}), 404

        cols_info = inspector.get_columns("prediction_results")
        col_names = {c.get("name") for c in cols_info}

        select_cols = ["id", "user_id", "model_name", "protein1_id", "protein2_id", "score", "label"]
        if "created_at" in col_names:
            select_cols.append("created_at")

        select_list = ", ".join(select_cols)
        row = db.session.execute(
            text(f"SELECT {select_list} FROM prediction_results WHERE id = :hid"),
            {"hid": history_id}
        ).fetchone()
        if not row:
            return jsonify({"error": "Not found"}), 404

        mapping = getattr(row, "_mapping", None)
        def get_val(key: str):
            if mapping is not None and key in mapping:
                return mapping[key]
            try:
                idx = select_cols.index(key)
                return row[idx]
            except Exception:
                return None

        owner_uid = str(get_val("user_id"))
        # print(f"Owner UID: {owner_uid}, Requester UID: {user_id}")
        # print(type(owner_uid), type(user_id))
        if owner_uid != user_id:
            return jsonify({"error": "Forbidden"}), 403

        created_at_val = get_val("created_at") if "created_at" in select_cols else None
        timestamp = None
        if created_at_val is not None:
            try:
                import datetime as _dt
                if isinstance(created_at_val, (_dt.datetime, _dt.date)):
                    dt = created_at_val if isinstance(created_at_val, _dt.datetime) else _dt.datetime.combine(created_at_val, _dt.time())
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=_dt.timezone.utc)
                    timestamp = dt.astimezone(_dt.timezone.utc).isoformat().replace("+00:00", "Z")
                else:
                    timestamp = str(created_at_val)
            except Exception:
                timestamp = None

        return jsonify({
            "id": get_val("id"),
            "model_name": get_val("model_name"),
            "protein1_id": get_val("protein1_id"),
            "protein2_id": get_val("protein2_id"),
            "score": get_val("score"),
            "label": get_val("label"),
            "timestamp": timestamp
        }), 200

    except Exception as e:
        print(f"Error fetching history by id: {e}")
        return jsonify({"error": "Database query failed"}), 500

@historyRouter.delete("/<int:history_id>")
@swag_from(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'static/swagger_specs/history/delete_by_id.yml'))
def delete_history_by_id(history_id: int):
    user_id: Optional[Any] = None
    try:
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            token = auth_header.split(' ', 1)[1].strip()
            payload = verify_jwt(token)
            user_id = payload.get('sub')
    except Exception as e:
        print(f"JWT verification failed: {e}")
        return jsonify({"error": "Unauthorized"}), 401

    if user_id is None:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        bind = db.session.get_bind()
        if bind is None:
            raise RuntimeError("Database engine not available")
        inspector = inspect(bind)

        if "prediction_results" not in set(inspector.get_table_names()):
            return jsonify({"error": "Not found"}), 404

        # Fetch owner user_id for the record
        sel = text("SELECT user_id FROM prediction_results WHERE id = :hid")
        row = db.session.execute(sel, {"hid": history_id}).fetchone()
        if not row:
            return jsonify({"error": "Not found"}), 404

        mapping = getattr(row, "_mapping", None)
        owner_uid = mapping["user_id"] if mapping is not None and "user_id" in mapping else (row[0] if len(row) > 0 else None)
        owner_uid = str(owner_uid)

        if owner_uid != user_id:
            return jsonify({"error": "Forbidden"}), 403

        # Authorized: delete the record
        del_res = db.session.execute(text("DELETE FROM prediction_results WHERE id = :hid"), {"hid": history_id})
        db.session.commit()
        deleted = getattr(del_res, "rowcount", None) or 0
        return jsonify({"deleted": deleted, "id": history_id}), 200

    except Exception as e:
        print(f"Error deleting history: {e}")
        db.session.rollback()
        return jsonify({"error": "Database query failed"}), 500

@historyRouter.delete("/")
@swag_from(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'static/swagger_specs/history/delete_all.yml'))
def delete_all_history():
    # Authenticate and get user_id from token
    user_id: Optional[Any] = None
    try:
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            token = auth_header.split(' ', 1)[1].strip()
            payload = verify_jwt(token)
            user_id = payload.get('sub')
    except Exception as e:
        print(f"JWT verification failed: {e}")
        return jsonify({"error": "Unauthorized"}), 401

    if user_id is None:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        bind = db.session.get_bind()
        if bind is None:
            raise RuntimeError("Database engine not available")
        inspector = inspect(bind)

        if "prediction_results" not in set(inspector.get_table_names()):
            return jsonify({"deleted": 0}), 200
        
        print("aaaa")

        del_res = db.session.execute(text("DELETE FROM prediction_results WHERE user_id = :uid"), {"uid": user_id})
        db.session.commit()
        deleted = getattr(del_res, "rowcount", None) or 0
        return jsonify({"deleted": deleted}), 200

    except Exception as e:
        print(f"Error deleting all history: {e}")
        db.session.rollback()
        return jsonify({"error": "Database query failed"}), 500
