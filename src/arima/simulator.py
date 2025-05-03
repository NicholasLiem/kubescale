#!/usr/bin/env python3
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
import logging
import sys
from pathlib import Path

# Add the arima module to the path
sys.path.append(str(Path(__file__).parent.parent))

from arima.data_transformer import PrometheusDataTransformer
from arima.arima_model import ARIMAModeler

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('arima.simulator')

class ARIMASimulator:
    """
    End-to-end simulator for ARIMA forecasting using Prometheus metrics
    """
    
    def __init__(self, examples_dir: str):
        """
        Initialize the simulator
        
        Args:
            examples_dir: Directory containing example Prometheus data files
        """
        self.examples_dir = examples_dir
        self.transformer = PrometheusDataTransformer()
        self.modeler = ARIMAModeler()
        logger.info("ARIMA Simulator initialized")
        
    def load_example_data(self, filename: str) -> Dict:
        """
        Load example Prometheus data from a JSON file
        
        Args:
            filename: Name of the JSON file in the examples directory
            
        Returns:
            Dictionary containing the Prometheus data
        """
        try:
            filepath = os.path.join(self.examples_dir, filename)
            logger.info(f"Loading example data from {filepath}")
            
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            return data.get('raw_data', {})
        except Exception as e:
            logger.error(f"Error loading example data: {e}")
            return {}
    
    def run_cpu_simulation(self, pod_name: str = "s0-55bb4b4958-scd6b", forecast_steps: int = 10):
        """
        Run a simulation with CPU metrics data
        
        Args:
            pod_name: Name of the pod to analyze
            forecast_steps: Number of steps to forecast
        """
        logger.info("Starting CPU simulation")
        
        # 1. Load the data
        raw_data = self.load_example_data("cpu-usage.json")
        
        # 2. Transform to DataFrame
        cpu_df = self.transformer.prometheus_to_dataframe(raw_data)
        print("\n--- Raw CPU DataFrame ---")
        print(f"Shape: {cpu_df.shape}")
        print(f"Columns: {cpu_df.columns.tolist()}")
        print(cpu_df.head())
        print("\nData Types:")
        print(cpu_df.dtypes)
        
        # 3. Prepare for ARIMA
        cpu_for_arima = self.transformer.prepare_for_arima(
            cpu_df, 
            metric_type="cpu",
            pod_name=pod_name
        )
        print("\n--- Processed CPU DataFrame for ARIMA ---")
        print(f"Shape: {cpu_for_arima.shape}")
        print(cpu_for_arima.head())
        
        # 4. Plot the processed data
        self.transformer.plot_time_series(
            cpu_for_arima, 
            title=f"CPU Usage for {pod_name} (% of core)"
        )
        
        # 5. Train-test split
        if len(cpu_for_arima) >= 3:  # Need at least 3 points for meaningful split
            split_idx = len(cpu_for_arima) - 1  # Leave last point for testing
            train_data = cpu_for_arima.iloc[:split_idx]
            test_data = cpu_for_arima.iloc[split_idx:]
            
            logger.info(f"Train data shape: {train_data.shape}")
            logger.info(f"Test data shape: {test_data.shape}")
            
            # 6. Train the model
            try:
                model = self.modeler.train_model(train_data.iloc[:, 0])
                
                # 7. Evaluate on test data
                metrics = self.modeler.evaluate_model(model, test_data.iloc[:, 0])
                
                # 8. Make a forecast
                forecast, conf_int = self.modeler.forecast(model, steps=forecast_steps)
                
                # 9. Plot the forecast
                self.modeler.plot_forecast(
                    train_data.iloc[:, 0], 
                    forecast, 
                    conf_int,
                    title=f"ARIMA Forecast for {pod_name} CPU Usage"
                )
            except Exception as e:
                logger.error(f"Error in ARIMA modeling: {e}")
        else:
            logger.warning("Not enough data points for train-test split")
    
    def run_memory_simulation(self, pod_name: str = "s0-55bb4b4958-scd6b", forecast_steps: int = 10):
        """
        Run a simulation with Memory metrics data
        
        Args:
            pod_name: Name of the pod to analyze
            forecast_steps: Number of steps to forecast
        """
        logger.info("Starting Memory simulation")
        
        # 1. Load the data
        raw_data = self.load_example_data("memory-usage.json")
        
        # 2. Transform to DataFrame
        memory_df = self.transformer.prometheus_to_dataframe(raw_data)
        print("\n--- Raw Memory DataFrame ---")
        print(f"Shape: {memory_df.shape}")
        print(f"Columns: {memory_df.columns.tolist()}")
        print(memory_df.head())
        
        # 3. Prepare for ARIMA
        memory_for_arima = self.transformer.prepare_for_arima(
            memory_df, 
            metric_type="memory",
            pod_name=pod_name
        )
        print("\n--- Processed Memory DataFrame for ARIMA ---")
        print(f"Shape: {memory_for_arima.shape}")
        print(memory_for_arima.head())
        
        # 4. Plot the processed data
        self.transformer.plot_time_series(
            memory_for_arima, 
            title=f"Memory Usage for {pod_name} (MB)"
        )
        
        # 5. Train-test split
        if len(memory_for_arima) >= 3:  # Need at least 3 points for meaningful split
            split_idx = len(memory_for_arima) - 1  # Leave last point for testing
            train_data = memory_for_arima.iloc[:split_idx]
            test_data = memory_for_arima.iloc[split_idx:]
            
            # 6. Train the model
            try:
                model = self.modeler.train_model(train_data.iloc[:, 0])
                
                # 7. Evaluate on test data
                metrics = self.modeler.evaluate_model(model, test_data.iloc[:, 0])
                
                # 8. Make a forecast
                forecast, conf_int = self.modeler.forecast(model, steps=forecast_steps)
                
                # 9. Plot the forecast
                self.modeler.plot_forecast(
                    train_data.iloc[:, 0], 
                    forecast, 
                    conf_int,
                    title=f"ARIMA Forecast for {pod_name} Memory Usage"
                )
            except Exception as e:
                logger.error(f"Error in ARIMA modeling: {e}")
        else:
            logger.warning("Not enough data points for train-test split")
    
    def run_all_simulations(self):
        """Run all available simulations"""
        logger.info("Running all simulations")
        
        # Run CPU simulation
        self.run_cpu_simulation()
        
        # Run Memory simulation
        self.run_memory_simulation()
        
        logger.info("All simulations completed")

if __name__ == "__main__":
    # Get the examples directory
    current_dir = Path(__file__).parent.parent.parent
    examples_dir = os.path.join(current_dir, "examples", "prometheus")
    
    # Create and run the simulator
    simulator = ARIMASimulator(examples_dir)
    simulator.run_all_simulations()