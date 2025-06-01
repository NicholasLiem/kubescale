from flask import Blueprint, jsonify, request, current_app
from services.metrics_transformer_service import MetricsTransformerService
from utils.logging_config import setup_logging

logger = setup_logging()
test_bp = Blueprint('test', __name__)

# Inter service comm testing
@test_bp.route('/force-scaling', methods=['POST'])
def force_prediction():
    """Force a scaling and notify brain controller (for testing)"""
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

@test_bp.route('/forecast-visualization', methods=['GET', 'POST'])
def forecast_visualization():
    """Generate and return forecast visualization"""
    try:
        # Get parameters from request
        if request.method == 'POST':
            data = request.json or {}
        else:
            data = request.args.to_dict()
        
        steps = int(data.get('steps', 100))
        include_confidence = data.get('include_confidence', 'true').lower() == 'true'
        format_type = data.get('format', 'base64')  # 'base64' or 'binary'
        
        # Get prediction service
        prediction_service = getattr(current_app, 'prediction_service', None)
        if not prediction_service:
            return jsonify({"error": "Prediction service not available"}), 500
        
        # Ensure model is initialized
        if not prediction_service.is_initialized:
            initialized = prediction_service.initialize_model()
            if not initialized:
                return jsonify({
                    "error": "Model not initialized and initialization failed",
                    "suggestion": "Ensure sufficient data is available and try again"
                }), 500
        
        logger.info(f"Generating forecast visualization with {steps} steps")
        
        # Generate the visualization
        result = prediction_service.generate_forecast_plot(
            steps=steps,
            include_confidence=include_confidence
        )
        
        if not result:
            return jsonify({"error": "Failed to generate visualization"}), 500
        
        # Return based on requested format
        if format_type == 'binary':
            # Return as binary PNG
            import base64
            from flask import Response
            
            plot_binary = base64.b64decode(result['plot_base64'])
            return Response(
                plot_binary,
                mimetype='image/png',
                headers={'Content-Disposition': 'inline; filename=forecast.png'}
            )
        else:
            # Return as JSON with base64 encoded image
            return jsonify({
                "success": True,
                "plot_base64": result['plot_base64'],
                "forecast_data": result['forecast_data'],
                "spikes": result['spikes'],
                "model_info": result['model_info'],
                "parameters": {
                    "steps": steps,
                    "include_confidence": include_confidence
                }
            })
        
    except Exception as e:
        logger.error(f"Error generating forecast visualization: {e}")
        return jsonify({"error": str(e)}), 500