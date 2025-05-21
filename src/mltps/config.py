import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("mltps")

class Config:
    # Default configuration
    _defaults = {
        # Service URLs
        "prometheus_url": "http://prometheus-kube-prometheus-prometheus.monitoring.svc.cluster.local:9090",
        "brain_controller_url": "http://brain.kube-scale.svc.cluster.local:8080",
        
        # General settings
        "namespace": "default",
        
        # Prediction settings
        "update_interval_seconds": 60,
        "prediction_window_size": 12,
        "model_update_interval_minutes": 10,
        "confidence_threshold": 0.7,
        "min_training_points": 15,
        
        # Advanced model parameters
        "arima_order": [3, 1, 1],
        "metrics_resample_freq": "1min",
        "metrics_fillna_method": "ffill"
    }
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._config = cls._defaults.copy()
            cls._instance._load_from_file()
        return cls._instance
    
    def _load_from_file(self):
        """Load configuration from file"""
        config_file = os.environ.get('MLTPS_CONFIG_FILE', 'mltps_config.json')
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                    self._config.update(file_config)
                logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logger.error(f"Failed to load configuration from {config_file}: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value"""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value at runtime"""
        self._config[key] = value
        logger.info(f"Configuration updated: {key} = {value}")
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """Update multiple configuration values at once"""
        self._config.update(config_dict)
        logger.info(f"Configuration updated with {len(config_dict)} values")
    
    def save_to_file(self, filename: Optional[str] = None) -> bool:
        """Save current configuration to a file"""
        if filename is None:
            filename = os.environ.get('MLTPS_CONFIG_FILE', 'mltps_config.json')
        
        try:
            with open(filename, 'w') as f:
                json.dump(self._config, f, indent=2)
            logger.info(f"Configuration saved to {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to save configuration to {filename}: {e}")
            return False
    
    def __getattr__(self, name):
        """Allow attribute-style access to configuration values"""
        if name in self._config:
            return self._config[name]
        raise AttributeError(f"Config has no attribute '{name}'")


config = Config()