import os

# Prometheus configuration
PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "http://prometheus-kube-prometheus-prometheus.default.svc.cluster.local:9090")

# Brain controller configuration
BRAIN_CONTROLLER_URL = os.environ.get("BRAIN_CONTROLLER_URL", "http://brain.kube-scale.svc.cluster.local:8080")

# Deployment to scale
NAMESPACE = os.environ.get("NAMESPACE", "default")

# Prediction configuration
PREDICTION_INTERVAL_MINUTES = int(os.environ.get("PREDICTION_INTERVAL_MINUTES", "5"))
PREDICTION_WINDOW_SIZE = int(os.environ.get("PREDICTION_WINDOW_SIZE", "12"))  # 12 intervals
MODEL_UPDATE_INTERVAL_MINUTES = int(os.environ.get("MODEL_UPDATE_INTERVAL_MINUTES", "10"))
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.7"))