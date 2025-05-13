import requests
import time
import logging
from typing import Dict, Optional

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
    
    def get_pod_metrics(self, namespace: str = "default", metric_type: str = "cpu", 
                      window_minutes: int = 10, step: str = "1m") -> Dict:
        """
        Get pod metrics from Prometheus
        
        Args:
            namespace: Kubernetes namespace
            metric_type: Type of metric ("cpu" or "memory")
            window_minutes: Time window in minutes
            step: Prometheus step interval
            
        Returns:
            Raw Prometheus metrics data
        """
        if metric_type == "cpu":
            query = f'sum(rate(container_cpu_usage_seconds_total{{namespace="{namespace}", ' \
                    f'pod=~".*"}}[10m])) by (pod)'
        elif metric_type == "memory":
            query = f'sum(container_memory_usage_bytes{{namespace="{namespace}", pod=~".*"}}) by (pod)'
        else:
            logger.error(f"Unsupported metric type: {metric_type}")
            return {}
            
        return self.get_prometheus_data(
            query=query,
            start_time=time.time() - (window_minutes * 60),
            end_time=time.time(),
            step=step
        )
        
    def get_pod_health_data(self, namespace: str, pod_name: Optional[str] = None, 
                          time_window_minutes: int = 10) -> Dict:
        """
        Get pod health status data from Prometheus
        
        Args:
            namespace: Kubernetes namespace to query
            pod_name: Optional pod name filter
            time_window_minutes: Time window in minutes
            
        Returns:
            Raw Prometheus data for pod health metrics
        """
        pod_filter = f'namespace="{namespace}"'
        if pod_name:
            pod_filter += f', pod="{pod_name}"'
            
        # Query for pod startup failures
        startup_query = f'kube_pod_status_phase{{phase="Failed", {pod_filter}}}'
        
        # Query for container restarts
        restart_query = f'changes(kube_pod_container_status_restarts_total{{{pod_filter}}}[{time_window_minutes}m])'
        
        # Get the data
        startup_data = self.get_prometheus_data(
            startup_query, 
            time.time() - (time_window_minutes * 60)
        )
        
        restart_data = self.get_prometheus_data(
            restart_query,
            time.time() - (time_window_minutes * 60)
        )
        
        return {
            "startup_failures": startup_data,
            "container_restarts": restart_data
        }