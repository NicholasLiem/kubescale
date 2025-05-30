import os
from flask import Flask
from config import config
from services.metrics_service import MetricsService
from services.prediction_service import PredictionService
from services.notification_service import NotificationService
from utils.logging_config import setup_logging
from routes.health_routes import health_bp
from routes.prediction_routes import prediction_bp
from routes.test_routes import test_bp
from background.prediction_loop import PredictionLoop

# Set up logging
logger = setup_logging()

def create_app():
    """Application factory pattern"""
    app = Flask(__name__)
    
    # Initialize services
    metrics_service = MetricsService(config.prometheus_url)
    prediction_service = PredictionService(
        metrics_service, 
        window_size=config.get("prediction_window_size", 40),
        confidence_threshold=config.get("confidence_threshold", 0.7),
        min_training_points=config.get("min_training_points", 200)
    )
    notification_service = NotificationService(config.brain_controller_url, config.namespace)
    
    # Store services in app context for access in routes
    app.metrics_service = metrics_service
    app.prediction_service = prediction_service
    app.notification_service = notification_service
    
    # Register blueprints
    app.register_blueprint(health_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(test_bp)
    
    # Start background prediction loop
    prediction_loop = PredictionLoop(
        prediction_service, 
        metrics_service,
        notification_service
    )
    prediction_loop.start_background_thread()
    logger.info("🚀 Background prediction loop started")
    
    return app

app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)