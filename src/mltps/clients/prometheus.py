import aiohttp
import logging
import time
from typing import Dict, List, Any, Optional
import urllib.parse

logger = logging.getLogger(__name__)

class PrometheusClient:
    def __init__(self, base_url: str):
        """
        Client for Prometheus API.
        
        Args:
            base_url: URL to Prometheus API endpoint
        """
        self.base_url = base_url.rstrip('/')
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            self.session = None
    
    async def _ensure_session(self):
        """Ensure HTTP session is created"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
    
    # async def get_metrics(self, queries: Dict[str, str], time_window: int = 60) -> Dict[str, List[float]]:
    #     """
    #     Get metrics from Prometheus.
        
    #     Args:
    #         queries: Dict of metric name to Prometheus query
    #         time_window: Time window in minutes for the query
            
    #     Returns:
    #         Dict of metric name to list of values
    #     """
    #     await self._ensure_session()
        
    #     end_time = int(time.time())
    #     start_time = end_time - (time_window * 60)
    #     step = '15s'  # Query resolution
        
    #     results = {}
        
    #     for metric_name, query in queries.items():
    #         try:
    #             # URL encode the query
    #             encoded_query = urllib.parse.quote(query)
                
    #             # Build the query URL
    #             url = f"{self.base_url}/api/v1/query_range"
    #             params = {
    #                 'query': query,
    #                 'start': start_time,
    #                 'end': end_time,
    #                 'step': step
    #             }
                
    #             async with self.session.get(url, params=params) as response:
    #                 if response.status != 200:
    #                     error_text = await response.text()
    #                     logger.error(f"Prometheus API error ({response.status}): {error_text}")
    #                     continue
                        
    #                 data = await response.json()
                    
    #                 if data.get('status') == 'success' and 'data' in data and # filepath: mltps/clients/prometheus.py