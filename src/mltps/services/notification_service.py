import requests
import logging
from typing import Dict, Any

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
        
    def notify_brain_controller(self, spike_detected: bool, predicted_value=0, time_to_spike=0, deployment_name=None):
        """Notify brain controller about prediction"""
        if not spike_detected:
            return
            
        try:
            target_deployment = deployment_name or "s0-warm-pool"
            target_namespace = self.namespace
            
            # TODO: To change this
            current_replicas = 1  # Default
            replica_count = max(1, int(predicted_value / 100))  # 1 replica per 100 requests/sec
            
            scale_request = {
                "replica_count": replica_count,
                "deployment_name": target_deployment,
                "namespace": target_namespace
            }
            
            response = requests.post(
                f"{self.brain_controller_url}/ml-callback/scale",
                json=scale_request
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully notified brain controller to scale to {replica_count} replicas")
                return True, response.json()
            else:
                logger.error(f"Failed to notify brain controller: {response.status_code} {response.text}")
                return False, None
                
        except Exception as e:
            logger.error(f"Error notifying brain controller: {e}")
            return False, None