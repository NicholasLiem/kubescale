from flask import Blueprint, jsonify, current_app

health_bp = Blueprint('health', __name__)

@health_bp.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@health_bp.route('/model-status', methods=['GET'])
def model_status():
    """Get current model status and data collection progress"""
    prediction_service = current_app.prediction_service
    
    current_points = len(prediction_service.history) if not prediction_service.history.empty else 0
    required_points = prediction_service.min_training_points
    
    # Calculate estimated time to sufficient data
    if current_points > 0 and current_points < required_points:
        remaining_points = required_points - current_points
        estimated_minutes = (remaining_points * 15) / 60
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