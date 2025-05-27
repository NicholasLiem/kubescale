import os
from flask import Flask, request, jsonify
from services.metrics_transformer_service import MetricsTransformerService

from config import config
from services.metrics_service import MetricsService
from services.prediction_service import PredictionService
from services.notification_service import NotificationService
from utils.logging_config import setup_logging

# Set up logging
logger = setup_logging()

# Initialize Flask app
app = Flask(__name__)

# Initialize services
metrics_service = MetricsService(config.prometheus_url)
prediction_service = PredictionService(
    metrics_service, 
    update_interval_seconds=config.get("update_interval_seconds", 60),
    window_size=config.get("prediction_window_size", 4),
    confidence_threshold=config.get("confidence_threshold", 0.7),
    model_update_interval_minutes=config.get("model_update_interval_minutes", 3),
    min_training_points=config.get("min_training_points", 60)
)
notification_service = NotificationService(config.brain_controller_url, config.namespace)

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
        transformer = MetricsTransformerService()
        raw_df = transformer.prometheus_to_dataframe(result)
        prepared_df = transformer.prepare_for_arima(
                df=raw_df,
                metric_type="cpu",
                resample_freq="15s",
                fillna_method='ffill',
                aggregate=True
            )
        
        data_prepared_df = prepared_df.to_json()
                    
        return jsonify({
            "query": query,
            "data": data_prepared_df
        })
        
    except Exception as e:
        logger.error(f"Error executing Prometheus query: {e}")
        return jsonify({"error": str(e)}), 500

def prediction_loop():    
    logger.info("Starting prediction loop...")
    
    # # Initialize model
    # for attempt in range(5):
    #     logger.info(f"Attempting to initialize prediction model (attempt {attempt+1})...")
    #     if prediction_service.initialize_model():
    #         logger.info("✅ Prediction model successfully initialized!")
    #         break
    #     time.sleep(30)
    
    # # Main prediction loop
    # while True:
    #     try:
    #         # Update model
    #         prediction_service.update_model()
            
    #         # Skip prediction if model isn't ready
    #         if not prediction_service.is_initialized:
    #             logger.info("⏳ Waiting for model initialization...")
    #             time.sleep(5)
    #             continue
                
    #         # Make prediction and check for anomalies
    #         _, confidence = prediction_service.predict_traffic()
    #         anomaly = prediction_service.detect_traffic_anomaly()
            
    #         if anomaly["spike_detected"]:
    #             logger.info(f"🔥 SPIKE ALERT: Predicted value of {anomaly['predicted_value']:.2f} "
    #                        f"(confidence: {anomaly['confidence']:.2f})")
                           
    #             notification_service.notify_brain_controller(
    #                 True, anomaly["predicted_value"], anomaly["time_to_spike"]
    #             )
                
    #         elif anomaly["spike_ending"]:
    #             logger.info(f"🔽 SPIKE ENDING: Traffic returning to normal levels "
    #                    f"(current: {anomaly['current_value']:.2f}, confidence: {anomaly['confidence']:.2f})")
                
    #             notification_service.notify_brain_controller(
    #                 False, anomaly["current_value"], 0, 
    #                 message="traffic_normalizing"
    #             )
                
    #         else:
    #             logger.info(f"✓ Normal traffic predicted (confidence: {confidence:.2f})")
                
    #     except Exception as e:
    #         logger.error(f"❌ Error in prediction loop: {e}")
            
    #     time.sleep(5)

if __name__ == '__main__':        
    # Start Flask app
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)