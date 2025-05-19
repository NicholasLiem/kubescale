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
    namespace = data.get("namespace", "warm-pool")
    
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

@app.route('/feed-and-predict', methods=['POST'])
def feed_and_predict():
    """
    Endpoint to feed custom data to the ARIMA model and get predictions.
    Expected JSON body format:
    {
        "metric_type": "requests_per_second",  // or "cpu_usage", "memory_usage"
        "values": [1.2, 1.5, 1.8, 2.1, ...],   // historical values to feed to the model
        "prediction_steps": 10                  // number of steps to predict ahead
    }
    """
    try:
        data = request.json
        if not data or 'metric_type' not in data or 'values' not in data:
            return jsonify({
                "error": "Missing required fields",
                "example": {
                    "metric_type": "requests_per_second",
                    "values": [1.2, 1.5, 1.8, 2.1, 2.5, 2.2, 2.3, 2.4, 2.7, 3.0],
                    "prediction_steps": 10
                }
            }), 400
            
        metric_type = data.get('metric_type')
        values = data.get('values')
        steps = data.get('prediction_steps', 10)
        
        if not isinstance(values, list) or len(values) < 10:
            return jsonify({
                "error": "Values must be a list with at least 10 data points"
            }), 400
            
        if metric_type not in ["requests_per_second", "cpu_usage", "memory_usage"]:
            return jsonify({
                "error": "Invalid metric_type. Must be one of: requests_per_second, cpu_usage, memory_usage"
            }), 400
            
        # Create a temporary ARIMA model
        model = ARIMAModel(order=(5, 1, 0))
        # Train the model with provided data
        model.train(values)
        
        if not model.is_trained:
            return jsonify({
                "error": "Failed to train model with provided data"
            }), 500
            
        # Make prediction
        forecast, intervals, confidence = model.predict(steps=steps)
        
        # Calculate if a spike is detected
        current_avg = np.mean(values[-5:])
        forecast_max = max(forecast)
        is_spike = forecast_max > current_avg * 1.5  # 50% increase
        
        # Determine when spike will occur (if any)
        spike_time = 0
        if is_spike:
            spike_indices = [i for i, val in enumerate(forecast) if val > current_avg * 1.5]
            if spike_indices:
                spike_time = min(spike_indices)  # First occurrence of spike

        # Transform boolean to json
        is_spike = bool(is_spike)
        
        return jsonify({
            "forecast": forecast,
            "confidence_intervals": intervals,
            "confidence_score": float(confidence),
            "spike_detected": is_spike,
            "spike_time_steps": spike_time if is_spike else None,
            "current_average": float(current_avg),
            "max_forecast": float(forecast_max)
        })
        
    except Exception as e:
        logger.error(f"Error in feed-and-predict endpoint: {e}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/prometheus-to-predict', methods=['POST'])
def prometheus_to_predict():
    """
    Endpoint to query Prometheus, transform the data, and make predictions.
    Expected JSON body format:
    {
        "query": "sum(rate(container_cpu_usage_seconds_total{namespace=\"default\", pod=~\"s[0-2].*|gw-nginx.*\"}[5m])) by (pod)",
        "start_time": "optional timestamp in seconds",
        "end_time": "optional timestamp in seconds",
        "step": "optional step like 1m, 5m, etc.",
        "metric_type": "cpu_usage",
        "resample_freq": "1min",
        "pod_name": "optional specific pod to analyze",
        "prediction_steps": 10
    }
    """
    try:
        data = request.json
        if not data or 'query' not in data:
            return jsonify({
                "error": "Missing required field: 'query'",
                "example": {
                    "query": 'sum(rate(container_cpu_usage_seconds_total{namespace="default", pod=~"s[0-2].*|gw-nginx.*"}[5m])) by (pod)',
                    "metric_type": "cpu_usage",
                    "start_time": "optional timestamp in seconds",
                    "end_time": "optional timestamp in seconds",
                    "step": "1m",
                    "resample_freq": "1min",
                    "pod_name": "optional specific pod name",
                    "prediction_steps": 10
                }
            }), 400
            
        query = data.get('query')
        start_time = data.get('start_time')
        end_time = data.get('end_time')
        step = data.get('step', '1m')
        metric_type = data.get('metric_type', 'cpu_usage')
        resample_freq = data.get('resample_freq', '1min')
        pod_name = data.get('pod_name')
        prediction_steps = data.get('prediction_steps', 10)
        
        logger.info(f"Executing Prometheus query: {query}")
        
        # 1. Get raw data from Prometheus
        prometheus_data = metrics_service.get_prometheus_data(
            query=query,
            start_time=start_time,
            end_time=end_time,
            step=step
        )
        
        # 2. Create transformer service and process the data
        transformer = MetricsTransformerService()
        
        try:
            # Convert Prometheus data to DataFrame
            raw_df = transformer.prometheus_to_dataframe(prometheus_data)
            
            # Prepare for ARIMA modeling
            prepared_df = transformer.prepare_for_arima(
                df=raw_df,
                metric_type=metric_type,
                resample_freq=resample_freq,
                fillna_method='ffill',
                pod_name=pod_name
            )
            
            if prepared_df.empty:
                return jsonify({
                    "error": "No valid data points after transformation"
                }), 400
                
            # For simplicity, we'll use the first column if pod_name wasn't specified
            if pod_name is None and not prepared_df.empty:
                column_name = prepared_df.columns[0]
                logger.info(f"No pod specified, using first pod in results: {column_name}")
                values = prepared_df[column_name].values.tolist()
            else:
                values = prepared_df[pod_name].values.tolist()
            
            # 3. Train ARIMA model with the processed data
            model = ARIMAModel(order=(5, 1, 0))
            model.train(values)
            
            if not model.is_trained:
                return jsonify({
                    "error": "Failed to train model with processed Prometheus data"
                }), 500
                
            # 4. Make prediction
            forecast, intervals, confidence = model.predict(steps=prediction_steps)
            
            # 5. Calculate if a spike is detected
            current_avg = np.mean(values[-5:])
            forecast_max = max(forecast)
            is_spike = forecast_max > current_avg * 1.5  # 50% increase
            
            # 6. Determine when spike will occur (if any)
            spike_time = None
            if is_spike:
                spike_indices = [i for i, val in enumerate(forecast) if val > current_avg * 1.5]
                if spike_indices:
                    spike_time = min(spike_indices)  # First occurrence of spike
            
            # Build response with both raw and transformed data
            return jsonify({
                "query": query,
                "pod_analyzed": pod_name if pod_name else prepared_df.columns[0],
                "data_points": len(values),
                "prediction_results": {
                    "forecast": forecast,
                    "confidence_intervals": intervals,
                    "confidence_score": float(confidence),
                    "spike_detected": bool(is_spike),
                    "spike_time_steps": spike_time,
                    "current_average": float(current_avg),
                    "max_forecast": float(forecast_max)
                },
                "sample_values": values[-10:] if len(values) > 10 else values  # Just show recent values
            })
        
        except KeyError as e:
            return jsonify({
                "error": f"Pod not found in results: {str(e)}",
                "available_pods": list(raw_df.columns) if 'raw_df' in locals() and hasattr(raw_df, 'columns') else []
            }), 400
            
    except Exception as e:
        logger.error(f"Error in prometheus-to-predict endpoint: {e}")
        return jsonify({"error": str(e)}), 500

def prediction_loop():
    """Background task to periodically make predictions"""
    while True:
        try:
            prediction_service.update_model()
            forecast, confidence = prediction_service.predict_traffic()
            
            spike_detected, predicted_value, time_to_spike = prediction_service.detect_spike()
            
            if spike_detected:
                notification_service.notify_brain_controller(
                    True, predicted_value, time_to_spike
                )
                
        except Exception as e:
            logger.error(f"Error in prediction loop: {e}")
            
        time.sleep(PREDICTION_INTERVAL_MINUTES * 60)

if __name__ == '__main__':    
    # Start background prediction task
    prediction_thread = threading.Thread(target=prediction_loop, daemon=True)
    prediction_thread.start()
    
    # Start Flask app
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)