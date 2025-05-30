from flask import Blueprint, jsonify, request, current_app
from services.metrics_transformer_service import MetricsTransformerService
from utils.logging_config import setup_logging

logger = setup_logging()
test_bp = Blueprint('test', __name__)

@test_bp.route('/force-prediction', methods=['POST'])
def force_prediction():
    """Force a prediction and notify brain controller (for testing)"""
    data = request.json
    scale_to = data.get("replica_count", 3)
    deployment = data.get("deployment_name", "s0-warm-pool")
    
    try:
        notification_service = current_app.notification_service
        success, response = notification_service.notify_brain_controller(
            True, 
            predicted_value=scale_to * 100,
            time_to_spike=0,
            deployment_name=deployment
        )
        
        return jsonify({
            "success": success,
            "response": response if success else "Failed to notify controller"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@test_bp.route('/query-prometheus', methods=['POST'])
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
        
        metrics_service = current_app.metrics_service
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

@test_bp.route('/model-status', methods=['GET'])
def model_status():
    """Get current model status and data collection progress"""
    prediction_service = current_app.prediction_service
    
    current_points = len(prediction_service.history) if not prediction_service.history.empty else 0
    required_points = prediction_service.min_training_points
    
    # Calculate estimated time to sufficient data
    if current_points > 0 and current_points < required_points:
        # Assuming 15s intervals, calculate remaining time
        remaining_points = required_points - current_points
        estimated_minutes = (remaining_points * 15) / 60  # Convert to minutes
    else:
        estimated_minutes = 0
    
    return jsonify({
        "is_initialized": prediction_service.is_initialized,
        "current_data_points": current_points,
        "required_data_points": required_points,
        "data_collection_progress": min(100, (current_points / required_points) * 100),
        "estimated_time_to_ready_minutes": max(0, estimated_minutes),
        "last_model_update": prediction_service.last_update,
        "transform_method": prediction_service.transform_method,
        "best_params": prediction_service.best_params
    })