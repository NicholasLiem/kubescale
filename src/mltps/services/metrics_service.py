import requests
import time
import logging
import pandas as pd
from typing import List, Dict, Any, Optional

logger = logging.getLogger("mltps")

class MetricsService:
    def __init__(self, prometheus_url: str):
        """
        Service for fetching and handling metrics data
        
        Args:
            prometheus_url: URL to Prometheus API endpoint
        """
        self.prometheus_url = prometheus_url
        
    def get_prometheus_data(self, query: str, start_time=None, end_time=None, step='1m') -> List:
        """Get data from Prometheus using a PromQL query"""
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
                    return result["data"]["result"][0]["values"]
                else:
                    logger.warning(f"No data found for query: {query}")
                    return []
            else:
                logger.error(f"Error fetching Prometheus data: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Exception when querying Prometheus: {e}")
            return []
            
    def get_request_rate_data(self, namespace="default", window_minutes=60) -> pd.DataFrame:
        """Get HTTP request rate data from Prometheus"""
        # Example query for HTTP request rate
        query = f'sum(rate(http_requests_total{{namespace="{namespace}"}}[5m]))'
        data = self.get_prometheus_data(
            query,
            start_time=time.time() - (window_minutes * 60)
        )
        
        if not data:
            return pd.DataFrame(columns=["timestamp", "value"])
            
        # Convert to pandas DataFrame
        df = pd.DataFrame(data, columns=["timestamp", "value"])
        df["value"] = df["value"].astype(float)
        
        return df