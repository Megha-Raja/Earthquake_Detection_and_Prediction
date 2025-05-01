#!/usr/bin/env python
"""
Pattern analysis module for earthquake data using scikit-learn.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import joblib
import os

class EarthquakePatternAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans = None
        self.isolation_forest = None
        self.pca = None
        
    def prepare_features(self, df):
        """Prepare features for analysis."""
        # Select numeric features
        features = ['magnitude', 'depth', 'latitude', 'longitude']
        
        # Handle missing values
        df_clean = df.dropna(subset=features)
        
        # Scale features
        X = self.scaler.fit_transform(df_clean[features])
        
        return X, df_clean
    
    def perform_clustering(self, df, n_clusters=5):
        """Perform K-means clustering on earthquake data."""
        try:
            # Prepare features
            X, df_clean = self.prepare_features(df)
            
            # Perform clustering
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = self.kmeans.fit_predict(X)
            
            # Add cluster labels to dataframe
            df_clean['cluster'] = clusters
            
            # Calculate cluster statistics
            cluster_stats = df_clean.groupby('cluster').agg({
                'magnitude': ['mean', 'min', 'max', 'count'],
                'depth': 'mean'
            }).round(2)
            
            # Save model
            os.makedirs('data/models', exist_ok=True)
            joblib.dump(self.kmeans, 'data/models/kmeans_model.joblib')
            
            return df_clean, cluster_stats
            
        except Exception as e:
            print(f"Error in clustering: {str(e)}")
            return None, None
    
    def detect_anomalies(self, df, contamination=0.1):
        """Detect anomalous earthquakes using Isolation Forest."""
        try:
            # Prepare features
            X, df_clean = self.prepare_features(df)
            
            # Fit isolation forest
            self.isolation_forest = IsolationForest(contamination=contamination, random_state=42)
            anomalies = self.isolation_forest.fit_predict(X)
            
            # Add anomaly labels (-1 for anomalies, 1 for normal)
            df_clean['is_anomaly'] = anomalies
            
            # Save model
            os.makedirs('data/models', exist_ok=True)
            joblib.dump(self.isolation_forest, 'data/models/anomaly_detector.joblib')
            
            return df_clean
            
        except Exception as e:
            print(f"Error in anomaly detection: {str(e)}")
            return None
    
    def analyze_patterns(self, df):
        """Analyze patterns in earthquake data."""
        try:
            # Prepare features
            X, df_clean = self.prepare_features(df)
            
            # Perform PCA for dimensionality reduction
            self.pca = PCA(n_components=2)
            pca_result = self.pca.fit_transform(X)
            
            # Add PCA components to dataframe
            df_clean['pca1'] = pca_result[:, 0]
            df_clean['pca2'] = pca_result[:, 1]
            
            # Calculate temporal patterns
            df_clean['hour'] = pd.to_datetime(df_clean['time']).dt.hour
            hourly_patterns = df_clean.groupby('hour').agg({
                'magnitude': ['mean', 'count'],
                'depth': 'mean'
            }).round(2)
            
            # Calculate spatial patterns
            spatial_patterns = df_clean.groupby('place').agg({
                'magnitude': ['mean', 'count', 'max'],
                'depth': 'mean'
            }).round(2)
            
            return {
                'pca_data': df_clean,
                'hourly_patterns': hourly_patterns,
                'spatial_patterns': spatial_patterns,
                'explained_variance': self.pca.explained_variance_ratio_
            }
            
        except Exception as e:
            print(f"Error in pattern analysis: {str(e)}")
            return None
    
    def predict_magnitude(self, df):
        """Predict earthquake magnitude based on location and depth."""
        try:
            from sklearn.ensemble import RandomForestRegressor
            
            # Prepare features
            features = ['depth', 'latitude', 'longitude']
            target = 'magnitude'
            
            # Handle missing values
            df_clean = df.dropna(subset=features + [target])
            
            # Split data
            X = df_clean[features]
            y = df_clean[target]
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # Make predictions
            predictions = model.predict(X)
            
            # Add predictions to dataframe
            df_clean['predicted_magnitude'] = predictions
            df_clean['magnitude_error'] = abs(df_clean[target] - predictions)
            
            # Save model
            os.makedirs('data/models', exist_ok=True)
            joblib.dump(model, 'data/models/magnitude_predictor.joblib')
            
            return df_clean
            
        except Exception as e:
            print(f"Error in magnitude prediction: {str(e)}")
            return None

if __name__ == "__main__":
    # Test the pattern analyzer
    import data_collector
    
    # Load some test data
    df = data_collector.fetch_realtime_earthquakes(hours=24, min_magnitude=4.0)
    
    if not df.empty:
        analyzer = EarthquakePatternAnalyzer()
        
        # Test clustering
        clustered_df, cluster_stats = analyzer.perform_clustering(df)
        if clustered_df is not None:
            print("\nCluster Statistics:")
            print(cluster_stats)
        
        # Test anomaly detection
        anomaly_df = analyzer.detect_anomalies(df)
        if anomaly_df is not None:
            print("\nAnomaly Detection Results:")
            print(f"Found {len(anomaly_df[anomaly_df['is_anomaly'] == -1])} anomalies")
        
        # Test pattern analysis
        patterns = analyzer.analyze_patterns(df)
        if patterns is not None:
            print("\nPattern Analysis Results:")
            print("Explained variance ratio:", patterns['explained_variance'])
        
        # Test magnitude prediction
        predicted_df = analyzer.predict_magnitude(df)
        if predicted_df is not None:
            print("\nMagnitude Prediction Results:")
            print("Average error:", predicted_df['magnitude_error'].mean()) 