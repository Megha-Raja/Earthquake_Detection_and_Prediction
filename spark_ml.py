#!/usr/bin/env python
"""
Machine learning module using PySpark MLlib for earthquake analysis.
"""

import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, count, avg, max, min, desc

from spark_processor import load_earthquake_data

def prepare_features(df):
    """
    Prepare features for machine learning.
    
    Args:
        df (pyspark.sql.DataFrame): Spark DataFrame containing earthquake data
        
    Returns:
        pyspark.sql.DataFrame: DataFrame with feature vector column
    """
    try:
        # Select only numeric columns for clustering
        numeric_cols = ["magnitude", "longitude", "latitude", "depth"]
        
        # Validate data
        for col_name in numeric_cols:
            if col_name not in df.columns:
                print(f"Column {col_name} not found in DataFrame")
                return None
        
        # Drop rows with null values
        df_clean = df.na.drop(subset=numeric_cols)
        
        # Check if we have enough data
        if df_clean.count() < 10:
            print("Not enough data points after cleaning")
            return None
        
        # Create feature vector
        assembler = VectorAssembler(
            inputCols=numeric_cols,
            outputCol="features"
        )
        
        return assembler.transform(df_clean)
    except Exception as e:
        print(f"Error preparing features: {str(e)}")
        return None

def train_clustering_model(spark, data_path, k=5):
    """
    Train a K-Means clustering model to identify earthquake patterns.
    
    Args:
        spark (SparkSession): Active Spark session
        data_path (str): Path to the CSV file containing earthquake data
        k (int): Number of clusters
        
    Returns:
        pyspark.ml.clustering.KMeansModel: Trained K-Means model
    """
    try:
        # Load data
        df = load_earthquake_data(spark, data_path)
        if df is None:
            raise ValueError("Failed to load earthquake data")
        
        # Prepare features
        df_features = prepare_features(df)
        if df_features is None:
            raise ValueError("Failed to prepare features")
        
        # Train K-Means model
        kmeans = KMeans().setK(k).setSeed(42)
        model = kmeans.fit(df_features)
        
        # Make predictions
        predictions = model.transform(df_features)
        
        # Evaluate the model
        evaluator = ClusteringEvaluator()
        silhouette = evaluator.evaluate(predictions)
        print(f"Silhouette score with {k} clusters: {silhouette}")
        
        # Analyze clusters
        print("\nCluster Centers:")
        centers = model.clusterCenters()
        for i, center in enumerate(centers):
            print(f"Cluster {i}: {center}")
        
        # Count earthquakes in each cluster
        print("\nEarthquakes per Cluster:")
        predictions.groupBy("prediction").count().orderBy("prediction").show()
        
        # Save the model
        model_path = "data/models/kmeans_model"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.write().overwrite().save(model_path)
        print(f"Model saved to {model_path}")
        
        # Save predictions for visualization
        predictions.select("id", "magnitude", "longitude", "latitude", "depth", "prediction") \
            .write.mode("overwrite").parquet("data/processed/clustered_earthquakes")
        
        return model
    except Exception as e:
        print(f"Error training clustering model: {str(e)}")
        return None

def train_magnitude_prediction_model(spark, data_path):
    """
    Train a linear regression model to predict earthquake magnitude based on depth and location.
    
    Args:
        spark (SparkSession): Active Spark session
        data_path (str): Path to the CSV file containing earthquake data
        
    Returns:
        pyspark.ml.regression.LinearRegressionModel: Trained regression model
    """
    try:
        # Load data
        df = load_earthquake_data(spark, data_path)
        if df is None:
            raise ValueError("Failed to load earthquake data")
        
        # Prepare features for regression
        feature_cols = ["latitude", "longitude", "depth"]
        
        # Validate data
        for col_name in feature_cols + ["magnitude"]:
            if col_name not in df.columns:
                print(f"Column {col_name} not found in DataFrame")
                return None
        
        # Clean data
        df_clean = df.na.drop(subset=feature_cols + ["magnitude"])
        
        # Check if we have enough data
        if df_clean.count() < 10:
            print("Not enough data points after cleaning")
            return None
        
        # Create feature vector
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features"
        )
        
        # Create pipeline
        pipeline = Pipeline(stages=[assembler])
        df_features = pipeline.fit(df_clean).transform(df_clean)
        
        # Split data
        train_data, test_data = df_features.randomSplit([0.8, 0.2], seed=42)
        
        # Train linear regression model
        lr = LinearRegression(
            featuresCol="features",
            labelCol="magnitude",
            maxIter=10,
            regParam=0.1,
            elasticNetParam=0.8
        )
        
        lr_model = lr.fit(train_data)
        
        # Evaluate model
        predictions = lr_model.transform(test_data)
        
        # Print model statistics
        trainingSummary = lr_model.summary
        print(f"RMSE: {trainingSummary.rootMeanSquaredError}")
        print(f"R2: {trainingSummary.r2}")
        
        # Save the model
        model_path = "data/models/magnitude_prediction_model"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        lr_model.write().overwrite().save(model_path)
        print(f"Magnitude prediction model saved to {model_path}")
        
        return lr_model
    except Exception as e:
        print(f"Error training magnitude prediction model: {str(e)}")
        return None

if __name__ == "__main__":
    # Test the ML functions
    spark = SparkSession.builder \
        .appName("Earthquake ML Test") \
        .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow") \
        .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow") \
        .getOrCreate()
    
    test_data_path = "data/raw/test_earthquakes.csv"
    
    if os.path.exists(test_data_path):
        train_clustering_model(spark, test_data_path)
        train_magnitude_prediction_model(spark, test_data_path)
    else:
        print(f"Test data not found at {test_data_path}")
    
    spark.stop()