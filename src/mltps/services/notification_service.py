import requests
import logging

logger = logging.getLogger("mltps")

class NotificationService:
    def __init__(self, brain_controller_url: str, namespace: str):
        """
        Service to notify brain controller about predictions
        
        Args:
            brain_controller_url: URL of brain controller API
            namespace: Kubernetes namespace
        """
        self.brain_controller_url = brain_controller_url
        self.namespace = namespace
        
    def notify_brain_controller(self, formatted_spikes):
        """Notify brain controller about prediction"""
        if not formatted_spikes:
            logger.warning("No spikes to notify brain controller about")
            return False, None
        
        try:
            forecast_request = {
                'success': True,
                'spikes': formatted_spikes,
            }

            response = requests.post(
                f"{self.brain_controller_url}/ml-callback/spike-forecast",
                json=forecast_request
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully notified brain controller about spike forecast")
                return True, response.json()
            else:
                logger.error(f"Failed to notify brain controller: {response.status_code} {response.text}")
                return False, None
                
        except Exception as e:
            logger.error(f"Error notifying brain controller: {e}")
            return False, None