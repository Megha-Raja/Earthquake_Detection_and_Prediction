import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import joblib
import os
from datetime import datetime, timedelta

class EarthquakeRiskAnalyzer:
    def __init__(self):
        self.models_dir = 'data/models'
        os.makedirs(self.models_dir, exist_ok=True)
        self.scaler = StandardScaler()
        
    def identify_active_zones(self, df):
        """Identify active earthquake zones using DBSCAN clustering."""
        try:
            # Prepare features for clustering
            features = df[['latitude', 'longitude']].copy()
            features_scaled = self.scaler.fit_transform(features)
            
            # Perform DBSCAN clustering with adjusted parameters
            dbscan = DBSCAN(eps=1.0, min_samples=3)  # Reduced eps and min_samples to get more zones
            zones = dbscan.fit_predict(features_scaled)
            
            # Calculate zone statistics
            df['zone'] = zones
            zone_stats = df[df['zone'] != -1].groupby('zone').agg({
                'magnitude': ['count', 'mean', 'max'],
                'latitude': 'mean',
                'longitude': 'mean'
            }).round(2)
            
            # Filter out zones with too few earthquakes
            zone_stats = zone_stats[zone_stats[('magnitude', 'count')] >= 3]
            
            return zone_stats
        except Exception as e:
            print(f"Error in active zone identification: {str(e)}")
            return None

    def predict_future_earthquakes(self, df, days_ahead=30):
        """
        Predict future earthquake locations and magnitudes.
        Returns a DataFrame with predicted locations and magnitudes.
        """
        try:
            # Identify active zones
            zone_stats = self.identify_active_zones(df)
            if zone_stats is None or zone_stats.empty:
                return None
            
            # Prepare features for prediction
            features = ['latitude', 'longitude', 'depth']
            target = 'magnitude'
            
            # Clean data
            df_clean = df.dropna(subset=features + [target])
            
            # Scale features
            X = self.scaler.fit_transform(df_clean[features])
            y = df_clean[target]
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # Generate predictions for active zones
            predictions = []
            
            # For each active zone
            for zone in zone_stats.index:
                zone_data = zone_stats.loc[zone]
                zone_lat = zone_data[('latitude', 'mean')]
                zone_lon = zone_data[('longitude', 'mean')]
                zone_mag_mean = zone_data[('magnitude', 'mean')]
                zone_mag_max = zone_data[('magnitude', 'max')]
                zone_count = zone_data[('magnitude', 'count')]
                
                # Calculate zone radius based on historical data
                zone_radius = 0.3  # Reduced radius for more focused predictions
                
                # Generate points within the zone
                num_points = min(15, max(5, int(zone_count / 2)))  # Adjust number of points based on zone activity
                lat_points = np.linspace(zone_lat - zone_radius, zone_lat + zone_radius, num_points)
                lon_points = np.linspace(zone_lon - zone_radius, zone_lon + zone_radius, num_points)
                
                # Predict for each point in the zone
                for lat in lat_points:
                    for lon in lon_points:
                        # Use historical average depth for the zone
                        features = np.array([[lat, lon, df_clean['depth'].mean()]])
                        features_scaled = self.scaler.transform(features)
                        pred_magnitude = model.predict(features_scaled)[0]
                        
                        # Adjust prediction based on zone characteristics
                        pred_magnitude = min(pred_magnitude, zone_mag_max * 1.2)  # Cap at 120% of historical max
                        
                        # Only include significant predictions
                        if pred_magnitude >= 4.0:
                            # Calculate confidence based on historical data
                            confidence = 'High' if zone_count > 10 and pred_magnitude <= zone_mag_max else 'Medium'
                            
                            predictions.append({
                                'latitude': lat,
                                'longitude': lon,
                                'predicted_magnitude': pred_magnitude,
                                'confidence': confidence,
                                'zone': zone
                            })
            
            # Convert to DataFrame
            predictions_df = pd.DataFrame(predictions)
            
            # Save model
            joblib.dump(model, os.path.join(self.models_dir, 'magnitude_predictor.joblib'))
            
            return predictions_df
            
        except Exception as e:
            print(f"Error in future earthquake prediction: {str(e)}")
            return None 