import requests
import time
import logging
from typing import Dict

logger = logging.getLogger("mltps")

class MetricsService:
    def __init__(self, prometheus_url: str):
        """
        Service for fetching and handling metrics data
        
        Args:
            prometheus_url: URL to Prometheus API endpoint
        """
        self.prometheus_url = prometheus_url
        
    def get_prometheus_data(self, query: str, start_time=None, end_time=None, step='1m') -> Dict:
        """
        Get data from Prometheus using a PromQL query
        
        Args:
            query: PromQL query string
            start_time: Start time for the query (Unix timestamp)
            end_time: End time for the query (Unix timestamp)
            step: Resolution step for the query
            
        Returns:
            Dict with Prometheus response data or empty dict on error
        """
        try:
            if start_time is None:
                start_time = time.time() - 3600  # Last hour by default
            if end_time is None:
                end_time = time.time()
                
            response = requests.get(
                f"{self.prometheus_url}/api/v1/query_range",
                params={
                    "query": query,
                    "start": start_time,
                    "end": end_time,
                    "step": step
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                if result["status"] == "success" and len(result["data"]["result"]) > 0:
                    return result
                else:
                    logger.warning(f"No data found for query: {query}")
                    return {}
            else:
                logger.error(f"Error fetching Prometheus data: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"Exception when querying Prometheus: {e}")
            return {}