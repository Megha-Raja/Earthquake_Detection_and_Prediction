#!/usr/bin/env python
"""
Visualization module for earthquake data.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap, MarkerCluster
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

def create_earthquake_map(data_path, output_path):
    """
    Create an interactive map of earthquakes.
    
    Args:
        data_path (str): Path to the CSV file containing earthquake data
        output_path (str): Path to save the HTML map
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Filter out rows with missing coordinates
    df = df.dropna(subset=['latitude', 'longitude', 'magnitude'])
    
    # Create base map centered on the average coordinates
    map_center = [df['latitude'].mean(), df['longitude'].mean()]
    earthquake_map = folium.Map(location=map_center, zoom_start=3, tiles="cartodbpositron")
    
    # Create a marker cluster group
    marker_cluster = MarkerCluster().add_to(earthquake_map)
    
    # Define magnitude color function
    def get_color(magnitude):
        if magnitude < 1.0:
            return 'green'
        elif magnitude < 3.0:
            return 'blue'
        elif magnitude < 5.0:
            return 'orange'
        else:
            return 'red'
    
    # Add markers for each earthquake
    for idx, row in df.iterrows():
        # Create popup text
        popup_text = f"""
        <strong>Magnitude:</strong> {row['magnitude']}<br>
        <strong>Location:</strong> {row['place']}<br>
        <strong>Depth:</strong> {row['depth']} km<br>
        <strong>Time:</strong> {row['time']}<br>
        """
        
        # Add marker with popup
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=row['magnitude'] * 1.5,
            popup=folium.Popup(popup_text, max_width=300),
            color=get_color(row['magnitude']),
            fill=True,
            fill_opacity=0.7
        ).add_to(marker_cluster)
    
    # Add heat map layer
    heat_data = [[row['latitude'], row['longitude'], row['magnitude']] for idx, row in df.iterrows()]
    HeatMap(heat_data, radius=15, gradient={0.4: 'blue', 0.65: 'lime', 0.8: 'orange', 1: 'red'}).add_to(earthquake_map)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index:9999; background-color:white; 
    padding: 10px; border: 2px solid grey; border-radius: 5px; font-size: 14px;">
    <p><strong>Magnitude:</strong></p>
    <p><span style="color:green;">●</span> &lt; 1.0</p>
    <p><span style="color:blue;">●</span> 1.0 - 2.9</p>
    <p><span style="color:orange;">●</span> 3.0 - 4.9</p>
    <p><span style="color:red;">●</span> 5.0+</p>
    </div>
    '''
    earthquake_map.get_root().html.add_child(folium.Element(legend_html))
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save map
    earthquake_map.save(output_path)
    print(f"Map saved to {output_path}")
    
    return earthquake_map

def create_analysis_plots(data_path, output_dir):
    """
    Create various analysis plots for earthquake data.
    
    Args:
        data_path (str): Path to the CSV file containing earthquake data
        output_dir (str): Directory to save the plot images
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Convert time to datetime
    df['time'] = pd.to_datetime(df['time'])
    
    # Plot 1: Magnitude Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='magnitude', bins=20, kde=True)
    plt.title('Earthquake Magnitude Distribution')
    plt.xlabel('Magnitude')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'magnitude_distribution.png'))
    plt.close()
    
    # Plot 2: Depth vs Magnitude
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='depth', y='magnitude', hue='magnitude', palette='viridis', size='magnitude', sizes=(20, 200), alpha=0.7)
    plt.title('Earthquake Depth vs. Magnitude')
    plt.xlabel('Depth (km)')
    plt.ylabel('Magnitude')
    plt.savefig(os.path.join(output_dir, 'depth_vs_magnitude.png'))
    plt.close()
    
    # Plot 3: Earthquakes over time
    plt.figure(figsize=(12, 6))
    df_sorted = df.sort_values('time')
    plt.plot(df_sorted['time'], df_sorted['magnitude'], 'o-', alpha=0.6)
    plt.title('Earthquake Magnitude Over Time')
    plt.xlabel('Date')
    plt.ylabel('Magnitude')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'magnitude_over_time.png'))
    plt.close()
    
    # Plot 4: Earthquake frequency by hour of day
    plt.figure(figsize=(10, 6))
    df['hour'] = df['time'].dt.hour
    hour_counts = df.groupby('hour').size()
    sns.barplot(x=hour_counts.index, y=hour_counts.values)
    plt.title('Earthquake Frequency by Hour of Day')
    plt.xlabel('Hour (UTC)')
    plt.ylabel('Number of Earthquakes')
    plt.savefig(os.path.join(output_dir, 'hourly_frequency.png'))
    plt.close()
    
    # Plot 5: Heatmap of earthquake locations
    if len(df) > 1:
        plt.figure(figsize=(12, 8))
        
        # Create a 2D histogram of earthquake locations
        heatmap, xedges, yedges = np.histogram2d(
            df['longitude'], 
            df['latitude'], 
            bins=50,
            range=[[-180, 180], [-90, 90]]
        )
        
        # Create a custom colormap
        colors = ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"]
        cmap = LinearSegmentedColormap.from_list("earthquake_cmap", colors)
        
        plt.imshow(
            heatmap.T, 
            extent=[-180, 180, -90, 90], 
            origin='lower', 
            cmap=cmap,
            aspect='auto'
        )
        
        plt.colorbar(label='Earthquake Density')
        plt.title('Global Earthquake Density')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        
        # Add a simple world map outline
        try:
            from matplotlib.patches import Rectangle
            for x in range(-180, 180, 30):
                for y in range(-90, 90, 30):
                    plt.axvline(x=x, color='gray', linestyle='--', alpha=0.3)
                    plt.axhline(y=y, color='gray', linestyle='--', alpha=0.3)
        except Exception as e:
            print(f"Could not add grid lines: {e}")
        
        plt.savefig(os.path.join(output_dir, 'location_heatmap.png'))
        plt.close()
    
    print(f"Analysis plots saved to {output_dir}")

def create_cluster_visualization(cluster_data_path, output_dir):
    """
    Create visualizations for clustered earthquake data.
    
    Args:
        cluster_data_path (str): Path to the parquet file containing clustered data
        output_dir (str): Directory to save the plot images
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load clustered data
        df = pd.read_parquet(cluster_data_path)
        
        # Plot: Clusters on map
        plt.figure(figsize=(14, 10))
        scatter = plt.scatter(df['longitude'], df['latitude'], 
                   c=df['prediction'], 
                   s=df['magnitude'] * 3,
                   alpha=0.6, 
                   cmap='viridis', 
                   edgecolors='k')
        
        plt.colorbar(scatter, label='Cluster')
        plt.title('Earthquake Clusters')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'earthquake_clusters.png'))
        plt.close()
        
        # Plot: Cluster statistics
        plt.figure(figsize=(12, 6))
        cluster_stats = df.groupby('prediction').agg({
            'magnitude': ['mean', 'min', 'max', 'count'],
            'depth': ['mean', 'min', 'max']
        })
        
        cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns.values]
        cluster_stats = cluster_stats.reset_index()
        
        sns.barplot(x='prediction', y='magnitude_count', data=cluster_stats)
        plt.title('Number of Earthquakes per Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Count')
        plt.savefig(os.path.join(output_dir, 'cluster_counts.png'))
        plt.close()
        
        # Plot: Average magnitude per cluster
        plt.figure(figsize=(12, 6))
        sns.barplot(x='prediction', y='magnitude_mean', data=cluster_stats)
        plt.title('Average Magnitude per Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Average Magnitude')
        plt.savefig(os.path.join(output_dir, 'cluster_magnitudes.png'))
        plt.close()
        
        print(f"Cluster visualizations saved to {output_dir}")
        
    except Exception as e:
        print(f"Error creating cluster visualizations: {e}")

if __name__ == "__main__":
    # Test the visualization functions
    test_data_path = "data/raw/test_earthquakes.csv"
    
    if os.path.exists(test_data_path):
        create_earthquake_map(test_data_path, "data/processed/test_earthquake_map.html")
        create_analysis_plots(test_data_path, "data/processed/")
    else:
        print(f"Test data not found at {test_data_path}")
    
    # Test cluster visualization if data exists
    test_cluster_path = "data/processed/clustered_earthquakes"
    if os.path.exists(test_cluster_path):
        create_cluster_visualization(test_cluster_path, "data/processed/")