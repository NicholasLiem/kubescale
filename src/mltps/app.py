import os
from flask import Flask, request, jsonify
import numpy as np
from models.arima import ARIMAModel
import time
import threading

from config import (
    PROMETHEUS_URL,
    BRAIN_CONTROLLER_URL,
    PREDICTION_INTERVAL_MINUTES,
    PREDICTION_WINDOW_SIZE,
    MODEL_UPDATE_INTERVAL_MINUTES,
    NAMESPACE,
    CONFIDENCE_THRESHOLD
)
from services.metrics_service import MetricsService
from services.prediction_service import PredictionService
from services.notification_service import NotificationService
from services.metrics_transformer_service import MetricsTransformerService
from utils.logging_config import setup_logging

# Set up logging
logger = setup_logging()

# Initialize Flask app
app = Flask(__name__)

# Initialize services
metrics_service = MetricsService(PROMETHEUS_URL)
prediction_service = PredictionService(
    metrics_service, 
    prediction_interval_minutes=PREDICTION_INTERVAL_MINUTES,
    window_size=PREDICTION_WINDOW_SIZE,
    confidence_threshold=CONFIDENCE_THRESHOLD,
    model_update_interval_minutes=MODEL_UPDATE_INTERVAL_MINUTES
)
notification_service = NotificationService(BRAIN_CONTROLLER_URL, NAMESPACE)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route('/predict', methods=['GET'])
def predict():
    prediction_service.update_model()
    forecast, confidence = prediction_service.predict_traffic()
    
    spike_detected, predicted_value, time_to_spike = prediction_service.detect_spike()
    
    if spike_detected:
        notification_service.notify_brain_controller(
            True, predicted_value, time_to_spike
        )
    
    return jsonify({
        "forecast": forecast.tolist() if forecast is not None else [],
        "confidence": float(confidence),
        "spike_detected": spike_detected,
        "predicted_value": float(predicted_value),
        "time_to_spike_minutes": time_to_spike
    })

#
# BELOW are test endpoints for local testing
# 
@app.route('/force-prediction', methods=['POST'])
def force_prediction():
    """Force a prediction and notify brain controller (for testing)"""
    data = request.json
    scale_to = data.get("replica_count", 3)
    deployment = data.get("deployment_name", "s0-warm-pool")
    
    try:
        success, response = notification_service.notify_brain_controller(
            True, 
            predicted_value=scale_to * 100,  # Approximate value based on replica count
            time_to_spike=0,
            deployment_name=deployment
        )
        
        return jsonify({
            "success": success,
            "response": response if success else "Failed to notify controller"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/query-prometheus', methods=['POST'])
def query_prometheus():
    """Test endpoint to run arbitrary Prometheus queries"""
    try:
        data = request.json
        if not data or 'query' not in data:
            return jsonify({
                "error": "Missing required field: 'query'",
                "example": {
                    "query": 'sum(rate(container_cpu_usage_seconds_total{namespace="default", pod=~"s[0-2].*|gw-nginx.*"}[5m])) by (pod)',
                    "start_time": "optional timestamp in seconds",
                    "end_time": "optional timestamp in seconds",
                    "step": "optional step like 1m, 5m, etc."
                }
            }), 400
            
        query = data.get('query')
        start_time = data.get('start_time')
        end_time = data.get('end_time')
        step = data.get('step', '1m')
        
        logger.info(f"Executing Prometheus query: {query}")
        
        result = metrics_service.get_prometheus_data(
            query=query,
            start_time=start_time,
            end_time=end_time,
            step=step
        )
                    
        return jsonify({
            "query": query,
            "raw_data": result
        })
        
    except Exception as e:
        logger.error(f"Error executing Prometheus query: {e}")
        return jsonify({"error": str(e)}), 500


# TODO: Implement this properly
def prediction_loop():
    """Background task to periodically make predictions"""
    # Initial delay to allow system to start up
    time.sleep(10)
    
    logger.info("Starting prediction loop...")
    
    # Initialize model with sufficient data
    initialization_attempts = 0
    while not prediction_service.is_initialized and initialization_attempts < 12:
        logger.info(f"Attempting to initialize prediction model (attempt {initialization_attempts+1})...")
        initialized = prediction_service.initialize_model()
        
        if initialized:
            logger.info("✅ Prediction model successfully initialized!")
            break
            
        initialization_attempts += 1
        logger.info(f"Waiting for more data before next initialization attempt...")
        time.sleep(30)
    
    if not prediction_service.is_initialized:
        logger.warning("⚠️ Could not initialize prediction model after maximum attempts")
        
    # Main prediction loop
    while True:
        try:
            # Update or initialize the model if needed
            if not prediction_service.is_initialized:
                prediction_service.initialize_model()
            else:
                prediction_service.update_model()
                
            # Make prediction if model is ready
            if prediction_service.is_initialized:
                forecast, confidence = prediction_service.predict_traffic()
                
                spike_detected, predicted_value, time_to_spike = prediction_service.detect_spike()
                
                if spike_detected:
                    logger.info(f"🔥 SPIKE ALERT: Predicted value of {predicted_value:.2f} "
                               f"expected in {time_to_spike} minutes "
                               f"(confidence: {confidence:.2f})")
                               
                    notification_service.notify_brain_controller(
                        True, predicted_value, time_to_spike
                    )
                else:
                    logger.info(f"✓ Normal traffic predicted (confidence: {confidence:.2f})")
            else:
                logger.info("⏳ Waiting for model initialization...")
                
        except Exception as e:
            logger.error(f"❌ Error in prediction loop: {e}")
            
        # Sleep until next prediction cycle
        logger.info(f"Sleeping for {PREDICTION_INTERVAL_MINUTES} minutes until next prediction cycle")
        time.sleep(30)

if __name__ == '__main__':        
    # Start Flask app
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)