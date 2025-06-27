import time
import threading
import numpy as np
from utils.logging_config import setup_logging

logger = setup_logging()

class PredictionLoop:
    def __init__(self, prediction_service, metrics_service, notification_service):
        self.prediction_service = prediction_service
        self.metrics_service = metrics_service
        self.notification_service = notification_service
        self.MODEL_UPDATE_INTERVAL = 30 * 60  # 10 minutes
        self.DATA_CHECK_INTERVAL = 60  # 1 minute  
        self.PREDICTION_INTERVAL = 60  # 1 minute
        self._stop_event = threading.Event()
    
    def start_background_thread(self):
        """Start the prediction loop in a background thread"""
        thread = threading.Thread(target=self.prediction_loop, daemon=True)
        thread.start()
        logger.info("🚀 Background prediction loop started")
    
    def stop(self):
        """Stop the prediction loop"""
        self._stop_event.set()
    
    def prediction_loop(self):
        """Main prediction loop"""
        logger.info("🚀 Starting prediction loop...")
        
        # Phase 1: Wait for sufficient data
        if not self._wait_for_sufficient_data():
            logger.error("❌ Failed to collect sufficient data")
            return
        
        # Phase 2: Initialize model
        if not self._initialize_model():
            logger.error("❌ Failed to initialize model")
            return
        
        # Phase 3: Main monitoring loop
        logger.info("🔄 Starting continuous prediction and monitoring...")
        
        last_model_update = time.time()
        prediction_count = 0
        
        while not self._stop_event.is_set():
            try:
                prediction_count += 1
                
                # Update model if needed
                last_model_update = self._update_model_if_needed(last_model_update)
                
                # Skip predictions if model isn't ready
                if not self.prediction_service.is_initialized:
                    logger.warning("⏳ Model not initialized - skipping prediction")
                    time.sleep(self.PREDICTION_INTERVAL)
                    continue
                
                if not self._fit_model_with_updated_data():
                    logger.warning(" Failing to fit model with newest data point")
                    continue

                # Make predictions and check for anomalies
                self._notify_spike_forecast()
                
            except Exception as e:
                logger.error(f"❌ Error in main prediction loop iteration #{prediction_count}: {e}")
            
            time.sleep(self.PREDICTION_INTERVAL)
    
    def _wait_for_sufficient_data(self):
        """Wait until we have enough data points for model initialization"""
        logger.info("📊 Waiting for sufficient data collection (200 points = ~50 minutes)...")
        cpu_query = 'sum(rate(container_cpu_usage_seconds_total{namespace="default", pod=~"s[0-2].*|gw-nginx.*"}[1m])) by (pod)'

        while not self._stop_event.is_set():
            try:
                raw_data = self.metrics_service.get_prometheus_data(query=cpu_query, step='15s')
                
                if raw_data and 'data' in raw_data and 'result' in raw_data['data']:
                    raw_df = self.prediction_service.transformer.prometheus_to_dataframe(raw_data)
                    if not raw_df.empty:
                        prepared_df = self.prediction_service.transformer.prepare_for_arima(
                            df=raw_df, metric_type="cpu", resample_freq="15s", 
                            fillna_method='ffill', aggregate=True
                        )
                        
                        current_points = len(prepared_df)
                        required_points = self.prediction_service.min_training_points
                        progress = (current_points / required_points) * 100
                        
                        logger.info(f"📈 Data collection: {current_points}/{required_points} points ({progress:.1f}%)")
                        
                        if current_points >= required_points:
                            logger.info("✅ Sufficient data collected!")
                            return True
                        
                        remaining_time = ((required_points - current_points) * 15) / 60
                        logger.info(f"⏳ Need {required_points - current_points} more points (~{remaining_time:.1f}min)")
                time.sleep(self.DATA_CHECK_INTERVAL)
                
            except Exception as e:
                logger.error(f"❌ Error checking data availability: {e}")
                time.sleep(self.DATA_CHECK_INTERVAL)
        
        return False
    
    def _initialize_model(self):
        """Initialize the prediction model with retry logic"""
        max_attempts = 3
        
        for attempt in range(max_attempts):
            if self._stop_event.is_set():
                return False
                
            logger.info(f"🔄 Initializing model (attempt {attempt+1}/{max_attempts})...")
            try:
                if self.prediction_service.initialize_model():
                    logger.info("✅ Prediction model successfully initialized!")
                    return True
                else:
                    logger.warning(f"⚠️ Model initialization failed (attempt {attempt+1})")
            except Exception as e:
                logger.error(f"❌ Error during model initialization: {e}")
            
            if attempt < max_attempts - 1:
                time.sleep(30)
        
        logger.error("❌ Failed to initialize model after all attempts")
        return False
    
    def _update_model_if_needed(self, last_update_time):
        """Update model if enough time has passed"""
        current_time = time.time()
        
        if current_time - last_update_time >= self.MODEL_UPDATE_INTERVAL:
            logger.info("🔄 Updating model with latest data...")
            if self.prediction_service.update_model():
                logger.info("✅ Model updated successfully")
                return current_time
            else:
                logger.warning("⚠️ Model update failed")
                return last_update_time
        
        return last_update_time
    
    def _fit_model_with_updated_data(self):
        """Fit current model with new data"""
        return self.prediction_service.fit_with_current_data()
    
    def _notify_spike_forecast(self):
        try:
            result = self.prediction_service.predict_spikes()
            if result is None:
                logger.warning("⚠️ No spike forecast available")
                return
            
            formatted_spikes = []
            for spike in result:
                formatted_spike = {
                    'index': int(spike['index']) if isinstance(spike['index'], (np.integer, np.int64)) else spike['index'],
                    'time': spike['time'].isoformat() if hasattr(spike['time'], 'isoformat') else str(spike['time']),
                    'value': float(spike['value']) if isinstance(spike['value'], (np.floating, np.float64)) else spike['value'],
                    'spike_id': int(spike['spike_id']) if isinstance(spike['spike_id'], (np.integer, np.int64)) else spike['spike_id'],
                    'type': str(spike['type']),
                    'time_from_now': float(spike['time_from_now']) if isinstance(spike['time_from_now'], (np.floating, np.float64)) else spike['time_from_now']
                }
                formatted_spikes.append(formatted_spike)
            
            if formatted_spikes:
                logger.info(f"📈 Spike forecast generated: {len(formatted_spikes)} spikes predicted")
                for spike in formatted_spikes:
                    logger.info(f"Spike at {spike['time']} - Value: {spike['value']}, Type: {spike['type']}")

            self.notification_service.notify_brain_controller(formatted_spikes)
                
        except Exception as e:
            logger.error(f"❌ Error in prediction detection: {e}")