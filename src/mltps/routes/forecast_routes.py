import time
import numpy as np
import datetime
from flask import Blueprint, jsonify, request, current_app
from utils.visualization import generate_forecast_plot
from utils.logging_config import setup_logging

logger = setup_logging()
forecast_bp = Blueprint('forecast', __name__)

@forecast_bp.route('/forecast', methods=['GET'])
def get_forecast():
    """Get detailed forecast for the next 40 data points with timestamps and visualization"""
    prediction_service = current_app.prediction_service
    
    # Check if model is initialized
    if not prediction_service.is_initialized:
        return jsonify({
            "error": "Model not yet initialized",
            "message": "Waiting for sufficient data collection",
            "current_points": len(prediction_service.history) if not prediction_service.history.empty else 0,
            "required_points": prediction_service.min_training_points
        }), 503
    
    try:
        # Get optional parameters from query string
        steps = int(request.args.get('steps', 40))
        include_confidence = request.args.get('confidence', 'true').lower() == 'true'
        include_plot = request.args.get('plot', 'true').lower() == 'true'
        
        # Ensure reasonable limits
        steps = min(max(steps, 1), 100)
        
        # Update model with latest data before forecasting
        logger.info("🔄 Refreshing model with latest data for forecast...")
        prediction_service.update_model()
        
        # Make forecast
        forecast, base_confidence = prediction_service.predict_traffic_extended(steps)
        
        if forecast is None:
            return jsonify({
                "error": "Failed to generate forecast",
                "message": "Model prediction failed"
            }), 500
        
        # Generate timestamps for forecast
        current_time = datetime.datetime.now()
        timestamps = []
        for i in range(steps):
            future_time = current_time + datetime.timedelta(seconds=(i + 1) * 15)
            timestamps.append(future_time.isoformat())
        
        # Get current and historical data for context
        current_data = prediction_service._get_current_series()
        current_value = float(current_data.tail(1).iloc[0]) if current_data is not None and not current_data.empty else 0.0
        
        # Get last 60 data points for visualization context
        historical_data = current_data.tail(60) if current_data is not None and not current_data.empty else None
        
        # Detect potential spikes in forecast
        anomaly = prediction_service.detect_traffic_anomaly()
        
        # Build response
        response = {
            "forecast": {
                "values": forecast.tolist(),
                "timestamps": timestamps,
                "steps": steps,
                "interval_seconds": 15
            },
            "current_value": current_value,
            "confidence": float(base_confidence),
            "spike_analysis": {
                "spike_detected": anomaly["spike_detected"],
                "spike_ending": anomaly["spike_ending"],
                "predicted_peak": float(np.max(forecast)),
                "time_to_peak_minutes": float(np.argmax(forecast) * 15 / 60),
                "confidence": float(anomaly["confidence"])
            },
            "model_info": {
                "transform_method": prediction_service.transform_method,
                "best_params": prediction_service.best_params,
                "last_update": prediction_service.last_update,
                "data_points": len(prediction_service.history) if not prediction_service.history.empty else 0
            }
        }
        
        # Add confidence intervals if requested
        if include_confidence and prediction_service.best_model:
            try:
                pred = prediction_service.best_model.get_forecast(steps=steps)
                conf_int = pred.conf_int(alpha=0.05)
                
                lower_bound = prediction_service.inverse_transform(conf_int.iloc[:, 0])
                upper_bound = prediction_service.inverse_transform(conf_int.iloc[:, 1])
                
                response["forecast"]["confidence_interval"] = {
                    "lower_bound": lower_bound.tolist(),
                    "upper_bound": upper_bound.tolist(),
                    "confidence_level": 0.95
                }
            except Exception as e:
                logger.warning(f"Failed to calculate confidence intervals: {e}")
        
        # Add visualization if requested
        if include_plot and historical_data is not None:
            try:
                plot_base64 = generate_forecast_plot(historical_data, forecast, timestamps, 
                                                   response.get("forecast", {}).get("confidence_interval"))
                response["visualization"] = {
                    "plot_base64": plot_base64,
                    "plot_type": "time_series_forecast"
                }
            except Exception as e:
                logger.warning(f"Failed to generate plot: {e}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error generating forecast: {e}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

@forecast_bp.route('/forecast/summary', methods=['GET'])
def get_forecast_summary():
    """Get a simplified forecast summary for quick checks"""
    prediction_service = current_app.prediction_service
    
    if not prediction_service.is_initialized:
        return jsonify({
            "error": "Model not initialized",
            "is_ready": False
        }), 503
    
    try:
        # Get short-term forecast (next 10 points = 2.5 minutes)
        forecast, confidence = prediction_service.predict_traffic_extended(10)
        
        if forecast is None:
            return jsonify({"error": "Forecast failed"}), 500
        
        # Get current value
        current_data = prediction_service._get_current_series()
        current_value = float(current_data.tail(1).iloc[0]) if current_data is not None and not current_data.empty else 0.0
        
        # Calculate summary statistics
        forecast_max = float(np.max(forecast))
        forecast_min = float(np.min(forecast))
        forecast_avg = float(np.mean(forecast))
        
        # Trend analysis
        trend = "stable"
        if forecast_avg > current_value * 1.1:
            trend = "increasing"
        elif forecast_avg < current_value * 0.9:
            trend = "decreasing"
        
        # Spike detection
        anomaly = prediction_service.detect_traffic_anomaly()
        
        return jsonify({
            "is_ready": True,
            "current_value": current_value,
            "forecast_summary": {
                "next_2_5_minutes": {
                    "max": forecast_max,
                    "min": forecast_min,
                    "average": forecast_avg,
                    "trend": trend
                }
            },
            "alerts": {
                "spike_detected": anomaly["spike_detected"],
                "spike_ending": anomaly["spike_ending"],
                "confidence": float(anomaly["confidence"])
            },
            "model_confidence": float(confidence),
            "timestamp": time.time()
        })
        
    except Exception as e:
        logger.error(f"Error generating forecast summary: {e}")
        return jsonify({"error": str(e)}), 500