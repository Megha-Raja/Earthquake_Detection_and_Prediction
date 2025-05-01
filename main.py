#!/usr/bin/env python
"""
Main execution file for the Earthquake Detection & Prediction System.
This script coordinates the data collection, processing, analysis, and visualization.
"""

import os
import findspark
findspark.init()  # Initialize findspark to locate Spark on the system

from pyspark.sql import SparkSession
from datetime import datetime, timedelta

# Import project modules
import data_collector
import spark_processor
import spark_ml
import visualizer

def create_spark_session():
    """Create and return a Spark session."""
    return (SparkSession.builder
            .appName("Earthquake Detection System")
            .config("spark.driver.memory", "4g")
            .config("spark.executor.memory", "4g")
            .getOrCreate())

def ensure_data_directory():
    """Create data directories if they don't exist."""
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/models', exist_ok=True)

def main():
    """Main execution function."""
    print("Starting Earthquake Detection & Prediction System...")
    
    # Create necessary directories
    ensure_data_directory()
    
    # Initialize Spark session
    spark = create_spark_session()
    print(f"Spark version: {spark.version}")
    
    # Set time periods for data collection
    end_date = datetime.utcnow()
    start_date_month = end_date - timedelta(days=30)
    start_date_year = end_date - timedelta(days=365)
    
    # Collect earthquake data
    print("Collecting recent earthquake data (last 30 days)...")
    monthly_data_path = data_collector.fetch_and_save_earthquake_data(
        start_date_month.strftime("%Y-%m-%d"), 
        end_date.strftime("%Y-%m-%d"),
        "data/raw/monthly_earthquakes.csv"
    )
    
    print("Collecting historical earthquake data (last year)...")
    yearly_data_path = data_collector.fetch_and_save_earthquake_data(
        start_date_year.strftime("%Y-%m-%d"), 
        end_date.strftime("%Y-%m-%d"),
        "data/raw/yearly_earthquakes.csv"
    )
    
    # Process data with PySpark
    print("Processing earthquake data with PySpark...")
    earthquake_df = spark_processor.load_earthquake_data(spark, monthly_data_path)
    
    # Perform basic analysis
    print("Performing basic analysis...")
    spark_processor.basic_earthquake_analysis(earthquake_df)
    
    # Save processed data
    print("Saving processed data...")
    spark_processor.save_processed_data(earthquake_df, "data/processed/processed_earthquakes")
    
    # Machine learning analysis
    print("Performing machine learning analysis...")
    ml_model = spark_ml.train_clustering_model(spark, yearly_data_path)
    
    # Create visualizations
    print("Creating visualizations...")
    visualizer.create_earthquake_map(monthly_data_path, "data/processed/earthquake_map.html")
    visualizer.create_analysis_plots(monthly_data_path, "data/processed/")
    
    # Stop Spark session
    print("Stopping Spark session...")
    spark.stop()
    
    print("\nProcessing complete. Run the dashboard with:")
    print("streamlit run dashboard.py")

if __name__ == "__main__":
    main()