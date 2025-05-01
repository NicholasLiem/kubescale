from flask import Flask, request, jsonify
import logging
import os
import json
import requests
import pandas as pd
import numpy as np
from prometheus_client import Gauge, start_http_server
from statsmodels.tsa.arima.model import ARIMA
import time
from config import (
    PROMETHEUS_URL,
    BRAIN_CONTROLLER_URL,
    PREDICTION_INTERVAL_MINUTES,
    PREDICTION_WINDOW_SIZE,
    MODEL_UPDATE_INTERVAL_MINUTES,
    NAMESPACE,
    CONFIDENCE_THRESHOLD
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mltps")

# Initialize Flask app
app = Flask(__name__)

# Initialize metrics for Prometheus export
traffic_prediction = Gauge('mltps_traffic_prediction', 'Predicted request rate')
prediction_confidence = Gauge('mltps_prediction_confidence', 'Confidence in traffic prediction')
model_accuracy = Gauge('mltps_model_accuracy', 'ARIMA model accuracy')

class TrafficPredictor:
    def __init__(self):
        self.model = None
        self.history = []
        self.last_update = 0
        self.last_prediction = None
        self.prediction_history = []
        
    def get_prometheus_data(self, query, start_time=None, end_time=None, step='1m'):
        """Get data from Prometheus using a PromQL query"""
        try:
            if start_time is None:
                start_time = time.time() - 3600  # Last hour by default
            if end_time is None:
                end_time = time.time()
                
            response = requests.get(
                f"{PROMETHEUS_URL}/api/v1/query_range",
                params={
                    "query": query,
                    "start": start_time,
                    "end": end_time,
                    "step": step
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                if result["status"] == "success" and len(result["data"]["result"]) > 0:
                    return result["data"]["result"][0]["values"]
                else:
                    logger.warning(f"No data found for query: {query}")
                    return []
            else:
                logger.error(f"Error fetching Prometheus data: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Exception when querying Prometheus: {e}")
            return []
    
    def update_model(self):
        """Update ARIMA model with latest data"""
        current_time = time.time()
        
        # Only update model if it's been long enough since last update
        if current_time - self.last_update < MODEL_UPDATE_INTERVAL_MINUTES * 60:
            return
        
        # Example query for HTTP request rate (adjust for your specific metrics)
        query = 'sum(rate(http_requests_total{namespace="default"}[5m]))'
        data = self.get_prometheus_data(query)
        
        if not data:
            logger.warning("No data available to update model")
            return
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(data, columns=["timestamp", "value"])
        df["value"] = df["value"].astype(float)
        
        # Store in history
        self.history = df
        
        # Fit ARIMA model
        try:
            # A simple ARIMA model with parameters (p=5, d=1, q=0)
            # You'll want to tune these parameters based on your data
            model = ARIMA(df["value"].values, order=(5, 1, 0))
            self.model = model.fit()
            logger.info("ARIMA model updated successfully")
            self.last_update = current_time
        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {e}")
    
    def predict_traffic(self, steps_ahead=PREDICTION_WINDOW_SIZE):
        """Predict traffic for next n intervals"""
        if self.model is None:
            self.update_model()
            if self.model is None:
                logger.error("Cannot make prediction: no model available")
                return None, 0
                
        try:
            # Generate forecast
            forecast = self.model.forecast(steps=steps_ahead)
            
            # Calculate confidence
            # This is a simple heuristic - replace with a proper confidence calculation
            confidence = min(0.95, 1.0 - self.model.resid.std() / np.mean(self.history["value"].astype(float)))
            
            prediction = {
                "forecast": forecast.tolist(),
                "confidence": confidence,
                "timestamp": time.time()
            }
            
            self.last_prediction = prediction
            self.prediction_history.append(prediction)
            
            # Update prometheus metrics
            traffic_prediction.set(np.max(forecast))
            prediction_confidence.set(confidence)
            
            return forecast, confidence
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None, 0
    
    def detect_spike(self):
        """Detect if a traffic spike is predicted"""
        if self.last_prediction is None:
            return False, 0, 0
            
        forecast = self.last_prediction["forecast"]
        confidence = self.last_prediction["confidence"]
        
        # Simple spike detection: max forecast exceeds current average by 50%
        if len(self.history) > 0:
            current_avg = np.mean(self.history["value"].astype(float)[-10:])  # avg of last 10 points
            max_forecast = np.max(forecast)
            
            if max_forecast > current_avg * 1.5 and confidence > CONFIDENCE_THRESHOLD:
                # When spike is predicted to occur (in minutes)
                spike_time = np.argmax(forecast) * PREDICTION_INTERVAL_MINUTES
                return True, max_forecast, spike_time
        
        return False, 0, 0
        
    def notify_brain_controller(self, spike_detected, predicted_value=0, time_to_spike=0):
        """Notify brain controller about prediction"""
        if not spike_detected:
            return
            
        try:
            # Estimate needed replicas based on predicted traffic
            # This is a placeholder calculation
            current_replicas = 1  # Default
            replica_count = max(1, int(predicted_value / 100))  # 1 replica per 100 requests/sec
            
            # Prepare scale request
            scale_request = {
                "replica_count": replica_count,
                "deployment_name": "placeholder-deployment",
                "namespace": NAMESPACE
            }
            
            # Call brain controller API
            response = requests.post(
                f"{BRAIN_CONTROLLER_URL}/ml-callback/scale",
                json=scale_request
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully notified brain controller to scale to {replica_count} replicas")
            else:
                logger.error(f"Failed to notify brain controller: {response.status_code} {response.text}")
                
        except Exception as e:
            logger.error(f"Error notifying brain controller: {e}")


# Initialize predictor
predictor = TrafficPredictor()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route('/metrics', methods=['GET'])
def metrics():
    # Prometheus metrics endpoint is handled by prometheus_client
    return "Use the /metrics endpoint provided by prometheus_client"

@app.route('/predict', methods=['GET'])
def predict():
    predictor.update_model()
    forecast, confidence = predictor.predict_traffic()
    
    spike_detected, predicted_value, time_to_spike = predictor.detect_spike()
    
    if spike_detected:
        predictor.notify_brain_controller(True, predicted_value, time_to_spike)
    
    return jsonify({
        "forecast": forecast.tolist() if forecast is not None else [],
        "confidence": confidence,
        "spike_detected": spike_detected,
        "predicted_value": float(predicted_value),
        "time_to_spike_minutes": time_to_spike
    })

@app.route('/force-prediction', methods=['POST'])
def force_prediction():
    """Force a prediction and notify brain controller (for testing)"""
    data = request.json
    scale_to = data.get("replica_count", 3)
    
    try:
        scale_request = {
            "replica_count": scale_to,
            # TODO: Add logic to determine the deployment name and namespace dynamically
            "deployment_name": "placeholder-deployment",
            "namespace": NAMESPACE
        }
        
        response = requests.post(
            f"{BRAIN_CONTROLLER_URL}/ml-callback/scale",
            json=scale_request
        )
        
        return jsonify({
            "success": response.status_code == 200,
            "response": response.json() if response.status_code == 200 else response.text
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

def prediction_loop():
    """Background task to periodically make predictions"""
    while True:
        try:
            predictor.update_model()
            forecast, confidence = predictor.predict_traffic()
            spike_detected, predicted_value, time_to_spike = predictor.detect_spike()
            
            if spike_detected:
                predictor.notify_brain_controller(True, predicted_value, time_to_spike)
                
        except Exception as e:
            logger.error(f"Error in prediction loop: {e}")
            
        time.sleep(PREDICTION_INTERVAL_MINUTES * 60)

if __name__ == '__main__':
    start_http_server(8000)
    
    import threading
    prediction_thread = threading.Thread(target=prediction_loop, daemon=True)
    prediction_thread.start()
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)