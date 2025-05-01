#!/usr/bin/env python
"""
PySpark processing module for analyzing earthquake data.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, max, min, desc, year, month, dayofmonth, hour
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType, IntegerType
import os

def define_schema():
    """Define the schema for earthquake data."""
    return StructType([
        StructField("id", StringType(), True),
        StructField("magnitude", DoubleType(), True),
        StructField("place", StringType(), True),
        StructField("time", TimestampType(), True),
        StructField("updated", TimestampType(), True),
        StructField("url", StringType(), True),
        StructField("detail", StringType(), True),
        StructField("felt", IntegerType(), True),
        StructField("cdi", DoubleType(), True),
        StructField("mmi", DoubleType(), True),
        StructField("alert", StringType(), True),
        StructField("status", StringType(), True),
        StructField("tsunami", IntegerType(), True),
        StructField("sig", IntegerType(), True),
        StructField("net", StringType(), True),
        StructField("code", StringType(), True),
        StructField("ids", StringType(), True),
        StructField("sources", StringType(), True),
        StructField("types", StringType(), True),
        StructField("longitude", DoubleType(), True),
        StructField("latitude", DoubleType(), True),
        StructField("depth", DoubleType(), True),
        StructField("date", StringType(), True),
        StructField("hour", IntegerType(), True)
    ])

def load_earthquake_data(spark, data_path):
    """
    Load earthquake data from CSV into a Spark DataFrame.
    
    Args:
        spark (SparkSession): Active Spark session
        data_path (str): Path to the CSV file containing earthquake data
        
    Returns:
        pyspark.sql.DataFrame: Spark DataFrame containing earthquake data
    """
    try:
        if not os.path.exists(data_path):
            print(f"Data file not found: {data_path}")
            return None
            
        schema = define_schema()
        
        df = (spark.read
              .format("csv")
              .option("header", "true")
              .option("inferSchema", "false")
              .schema(schema)
              .load(data_path))
        
        # Validate data
        if df.count() == 0:
            print("No data loaded from file")
            return None
            
        # Register the DataFrame as a temporary SQL table
        df.createOrReplaceTempView("earthquakes")
        
        print(f"Loaded {df.count()} earthquake records into Spark DataFrame")
        return df
    except Exception as e:
        print(f"Error loading earthquake data: {str(e)}")
        return None

def basic_earthquake_analysis(df):
    """
    Perform basic analysis on earthquake data.
    
    Args:
        df (pyspark.sql.DataFrame): Spark DataFrame containing earthquake data
    """
    try:
        if df is None:
            print("No data available for analysis")
            return
            
        # Show the schema
        print("\nDataFrame Schema:")
        df.printSchema()
        
        # Show a sample of the data
        print("\nSample Data:")
        df.show(5, truncate=False)
        
        # Basic statistics
        print("\nBasic Statistics for Magnitude:")
        df.select("magnitude").describe().show()
        
        # Count earthquakes by magnitude range
        print("\nEarthquakes by Magnitude Range:")
        df.groupBy(
            (col("magnitude").cast("int")).alias("magnitude_range")
        ).count().orderBy("magnitude_range").show()
        
        # Top 10 regions with most earthquakes
        print("\nTop 10 Regions with Most Earthquakes:")
        df.groupBy("place").count().orderBy(desc("count")).show(10, truncate=False)
        
        # Average depth by magnitude range
        print("\nAverage Depth by Magnitude Range:")
        df.groupBy(
            (col("magnitude").cast("int")).alias("magnitude_range")
        ).agg(
            avg("depth").alias("avg_depth"),
            count("*").alias("count")
        ).orderBy("magnitude_range").show()
        
        # Time-based analysis (by hour)
        print("\nEarthquakes by Hour of Day:")
        df.groupBy("hour").count().orderBy("hour").show(24)
    except Exception as e:
        print(f"Error in basic analysis: {str(e)}")

def region_analysis(df):
    """
    Analyze earthquake data by region.
    
    Args:
        df (pyspark.sql.DataFrame): Spark DataFrame containing earthquake data
    """
    try:
        if df is None:
            print("No data available for region analysis")
            return None
            
        # Define regions based on longitude/latitude
        region_df = df.withColumn(
            "region",
            (
                # This is a simplified region mapping - would need refinement for production
                (col("longitude").between(-170, -30) & col("latitude").between(15, 90)).cast("int") * 1 + # North America
                (col("longitude").between(-85, -30) & col("latitude").between(-60, 15)).cast("int") * 2 + # South America
                (col("longitude").between(-30, 40) & col("latitude").between(35, 90)).cast("int") * 3 + # Europe
                (col("longitude").between(40, 180) & col("latitude").between(0, 90)).cast("int") * 4 + # Asia
                (col("longitude").between(-30, 60) & col("latitude").between(-40, 35)).cast("int") * 5 + # Africa
                (col("longitude").between(110, 180) & col("latitude").between(-50, 0)).cast("int") * 6 + # Australia
                (
                    ~(col("longitude").between(-170, 180) & col("latitude").between(-60, 90))
                ).cast("int") * 7 # Pacific and others
            )
        )
        
        # Map region codes to names
        region_mapping = {
            1: "North America",
            2: "South America",
            3: "Europe",
            4: "Asia", 
            5: "Africa",
            6: "Australia",
            7: "Pacific and Others"
        }
        
        # Replace region codes with names
        for code, name in region_mapping.items():
            region_df = region_df.replace(code, name, "region")
        
        # Analyze earthquakes by region
        print("\nEarthquakes by Region:")
        region_df.groupBy("region").agg(
            count("*").alias("count"),
            avg("magnitude").alias("avg_magnitude"),
            max("magnitude").alias("max_magnitude"),
            avg("depth").alias("avg_depth")
        ).orderBy(desc("count")).show(truncate=False)
        
        return region_df
    except Exception as e:
        print(f"Error in region analysis: {str(e)}")
        return None

def time_series_analysis(df):
    """
    Perform time series analysis on earthquake data.
    
    Args:
        df (pyspark.sql.DataFrame): Spark DataFrame containing earthquake data
    """
    # Extract date components
    time_df = df.withColumn("year", year("time")) \
                .withColumn("month", month("time")) \
                .withColumn("day", dayofmonth("time"))
    
    # Aggregate by day
    print("\nEarthquakes by Day (last 30 days):")
    daily_counts = time_df.groupBy("year", "month", "day") \
                         .agg(count("*").alias("earthquakes_count"), 
                              avg("magnitude").alias("avg_magnitude")) \
                         .orderBy("year", "month", "day")
    
    daily_counts.show(30)
    
    # Aggregate by hour
    print("\nEarthquakes by Hour of Day:")
    hourly_counts = df.groupBy("hour") \
                     .agg(count("*").alias("earthquakes_count")) \
                     .orderBy("hour")
    
    hourly_counts.show(24)
    
    return daily_counts, hourly_counts

def save_processed_data(df, output_path):
    """
    Save processed earthquake data to parquet format.
    
    Args:
        df (pyspark.sql.DataFrame): Spark DataFrame containing processed data
        output_path (str): Path to save the processed data
    """
    try:
        if df is None:
            print("No data available to save")
            return
            
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as parquet (efficient columnar storage format)
        df.write.mode("overwrite").parquet(output_path)
        print(f"Saved processed data to {output_path}")
        
        # Also save important aggregations for dashboard
        # Region analysis
        region_df = region_analysis(df)
        if region_df is not None:
            region_df.write.mode("overwrite").parquet(f"{output_path}_regions")
        
        # Time series
        daily_counts, hourly_counts = time_series_analysis(df)
        if daily_counts is not None:
            daily_counts.write.mode("overwrite").parquet(f"{output_path}_daily")
        if hourly_counts is not None:
            hourly_counts.write.mode("overwrite").parquet(f"{output_path}_hourly")
    except Exception as e:
        print(f"Error saving processed data: {str(e)}")

if __name__ == "__main__":
    # Test the Spark processing functions
    spark = SparkSession.builder \
        .appName("Earthquake Analysis Test") \
        .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow") \
        .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow") \
        .getOrCreate()
    
    test_data_path = "data/raw/test_earthquakes.csv"
    
    if os.path.exists(test_data_path):
        df = load_earthquake_data(spark, test_data_path)
        if df is not None:
            basic_earthquake_analysis(df)
            region_analysis(df)
            time_series_analysis(df)
            save_processed_data(df, "data/processed/test_processed")
    else:
        print(f"Test data not found at {test_data_path}")
    
    spark.stop()