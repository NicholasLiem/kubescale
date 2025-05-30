from flask import Blueprint, jsonify, current_app
from utils.logging_config import setup_logging

logger = setup_logging()
prediction_bp = Blueprint('prediction', __name__)

# TODO: NEED REWORK FOR MANUAL PREDICTION INVOCATION

# @prediction_bp.route('/predict', methods=['GET'])
# def predict():
#     prediction_service = current_app.prediction_service
#     notification_service = current_app.notification_service
    
#     if not prediction_service.is_initialized:
#         return jsonify({
#             "error": "Model not yet initialized",
#             "message": "Waiting for sufficient data collection (200 data points required)",
#             "current_points": len(prediction_service.history) if not prediction_service.history.empty else 0,
#             "required_points": 200
#         }), 503
    
#     forecast, confidence = prediction_service.predict_traffic()
#     anomaly = prediction_service.detect_traffic_anomaly()
    
#     if anomaly["spike_detected"]:
#         notification_service.notify_brain_controller(
#             True, anomaly["predicted_value"], anomaly["time_to_spike"]
#         )
    
#     return jsonify({
#         "forecast": forecast.tolist() if forecast is not None else [],
#         "confidence": float(confidence),
#         "spike_detected": anomaly["spike_detected"],
#         "predicted_value": float(anomaly["predicted_value"]),
#         "current_value": float(anomaly["current_value"]),
#         "time_to_spike_minutes": anomaly["time_to_spike"]
#     })