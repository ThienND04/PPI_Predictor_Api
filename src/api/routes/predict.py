from flask import Blueprint, request, jsonify
# from services.predict_service import predict_interaction

predictRouter = Blueprint('api_routes', __name__)

@predictRouter.route('/', methods=['POST'])
def predict():
    data = request.get_json()
    seq1 = data.get("protein1")
    seq2 = data.get("protein2")

    if not seq1 or not seq2:
        return jsonify({"error": "Missing input sequences"}), 400

    # result = predict_interaction(seq1, seq2)
    result = {"interacts": True, "confidence": 0.87}
    return jsonify({"result": result})