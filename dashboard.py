#!/usr/bin/env python
"""
Streamlit dashboard for the Earthquake Detection & Prediction System.
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
from streamlit_folium import folium_static
import pyspark
from pyspark.sql import SparkSession
from risk_analyzer import EarthquakeRiskAnalyzer

# Import project modules
import data_collector
import spark_processor
import spark_ml
import pattern_analyzer

# Set page configuration
st.set_page_config(
    page_title="Earthquake Detection System",
    page_icon="🌎",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def load_earthquake_data(days=30, min_magnitude=0):
    """Load earthquake data from USGS API."""
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    # Check if we already have this data cached as a file
    file_path = f"data/raw/earthquakes_{days}days_{min_magnitude}mag.csv"
    
    # If file doesn't exist or is older than 6 hours, fetch new data
    if not os.path.exists(file_path) or (datetime.now() - datetime.fromtimestamp(os.path.getmtime(file_path))).seconds > 21600:
        data_path = data_collector.fetch_and_save_earthquake_data(
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
            file_path,
            min_magnitude
        )
    else:
        data_path = file_path
    
    # Load the data
    if data_path and os.path.exists(data_path):
        return pd.read_csv(data_path)
    else:
        st.error("Failed to load earthquake data")
        return pd.DataFrame()

@st.cache_resource
def create_spark_session():
    """Create and return a Spark session."""
    try:
        import findspark
        findspark.init()
        
        return (SparkSession.builder
                .appName("Earthquake Dashboard")
                .config("spark.driver.memory", "2g")
                .config("spark.executor.memory", "2g")
                .config("spark.ui.enabled", "false")
                .config("spark.driver.host", "localhost")
                .config("spark.driver.bindAddress", "localhost")
                .getOrCreate())
    except Exception as e:
        st.error(f"Failed to create Spark session: {str(e)}")
        return None

def create_map(df):
    """Create an interactive folium map with earthquake data."""
    # Filter out rows with missing coordinates
    df_map = df.dropna(subset=['latitude', 'longitude', 'magnitude'])
    
    # Create base map
    m = folium.Map(location=[20, 0], zoom_start=2, tiles="cartodbpositron")
    
    # Create marker cluster
    marker_cluster = MarkerCluster().add_to(m)
    
    # Define color function for magnitude
    def get_color(magnitude):
        if magnitude < 2.0:
            return 'green'
        elif magnitude < 4.0:
            return 'blue'
        elif magnitude < 6.0:
            return 'orange'
        else:
            return 'red'
    
    # Add markers for each earthquake
    for idx, row in df_map.iterrows():
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
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index:9999; background-color:white; 
    padding: 10px; border: 2px solid grey; border-radius: 5px;">
    <p><strong>Magnitude:</strong></p>
    <p><span style="color:green;">●</span> &lt; 2.0</p>
    <p><span style="color:blue;">●</span> 2.0 - 3.9</p>
    <p><span style="color:orange;">●</span> 4.0 - 5.9</p>
    <p><span style="color:red;">●</span> 6.0+</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

def run_spark_analysis(df, num_clusters=5):
    """Run Spark analysis on the earthquake data."""
    try:
        # Create temporary CSV for Spark to read
        temp_csv_path = "data/temp/dashboard_data.csv"
        os.makedirs(os.path.dirname(temp_csv_path), exist_ok=True)
        df.to_csv(temp_csv_path, index=False)
        
        # Get Spark session
        spark = create_spark_session()
        
        # Load data into Spark
        spark_df = spark_processor.load_earthquake_data(spark, temp_csv_path)
        
        # Run basic analysis
        spark_processor.basic_earthquake_analysis(spark_df)
        
        # Run clustering
        model = spark_ml.train_clustering_model(spark, temp_csv_path, k=num_clusters)
        
        # Load clustered data for visualization
        clustered_df = pd.read_parquet("data/processed/clustered_earthquakes")
        
        return clustered_df
    except Exception as e:
        st.error(f"Error running Spark analysis: {e}")
        return None

def run_pattern_analysis(df, num_clusters=5):
    """Run pattern analysis on the earthquake data."""
    try:
        analyzer = pattern_analyzer.EarthquakePatternAnalyzer()
        
        # Perform clustering
        clustered_df, cluster_stats = analyzer.perform_clustering(df, n_clusters=num_clusters)
        if clustered_df is None:
            st.error("Failed to perform clustering analysis")
            return None
            
        # Detect anomalies
        anomaly_df = analyzer.detect_anomalies(df)
        if anomaly_df is None:
            st.error("Failed to detect anomalies")
            return None
            
        # Analyze patterns
        patterns = analyzer.analyze_patterns(df)
        if patterns is None:
            st.error("Failed to analyze patterns")
            return None
            
        # Predict magnitudes
        predicted_df = analyzer.predict_magnitude(df)
        if predicted_df is None:
            st.error("Failed to predict magnitudes")
            return None
            
        return {
            'clustered': clustered_df,
            'cluster_stats': cluster_stats,
            'anomalies': anomaly_df,
            'patterns': patterns,
            'predictions': predicted_df
        }
    except Exception as e:
        st.error(f"Error running pattern analysis: {str(e)}")
        return None

def show_map(df):
    """Display interactive earthquake map."""
    st.subheader("Interactive Earthquake Map")
    
    # Create and display map
    with st.spinner("Generating map..."):
        m = create_map(df)
        folium_static(m, width=1000, height=600)
    
    st.markdown("""
    **How to Use the Map:**
    - Each circle represents an earthquake
    - The size of the circle shows how strong the earthquake was
    - The color indicates the magnitude:
      * Green: Small earthquakes (< 2.0)
      * Blue: Medium earthquakes (2.0 - 3.9)
      * Orange: Strong earthquakes (4.0 - 5.9)
      * Red: Very strong earthquakes (6.0+)
    - Click on any circle to see details about that earthquake
    - Use the + and - buttons to zoom in and out
    - Drag to move around the map
    """)

def show_basic_analysis(df):
    """Display basic earthquake analysis."""
    st.subheader("Earthquake Analysis")
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Magnitude distribution
        st.subheader("How Strong Are the Earthquakes?")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=df, x='magnitude', bins=20, kde=True, ax=ax)
        ax.set_xlabel('Magnitude')
        ax.set_ylabel('Number of Earthquakes')
        st.pyplot(fig)
        st.markdown("""
        **What This Chart Shows:**
        - This chart shows how many earthquakes of each strength occurred
        - Most earthquakes are small (left side of the graph)
        - Strong earthquakes are rare (right side of the graph)
        - The orange line shows the overall pattern
        - This helps understand what strength of earthquakes are most common
        """)
    
    with col2:
        # Earthquakes over time
        st.subheader("When Did the Earthquakes Happen?")
        fig, ax = plt.subplots(figsize=(10, 6))
        df_sorted = df.sort_values('time')
        ax.plot(df_sorted['time'], df_sorted['magnitude'], 'o-', alpha=0.6)
        ax.set_xlabel('Date')
        ax.set_ylabel('Magnitude')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown("""
        **What This Chart Shows:**
        - This chart shows when earthquakes happened
        - Each dot represents one earthquake
        - The height of the dot shows how strong it was
        - The line connects the dots to show the sequence
        - This helps see if there are more earthquakes at certain times
        """)

def show_risk_analysis(df):
    """Display future earthquake predictions on a map."""
    st.header("Future Earthquake Predictions")
    
    # Initialize risk analyzer
    analyzer = EarthquakeRiskAnalyzer()
    
    # Get predictions
    with st.spinner("Analyzing earthquake patterns and generating predictions..."):
        predictions = analyzer.predict_future_earthquakes(df)
    
    if predictions is not None and not predictions.empty:
        # Create map
        m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], 
                      zoom_start=4)
        
        # Add predicted earthquake markers
        for _, row in predictions.iterrows():
            # Color based on predicted magnitude and confidence
            if row['confidence'] == 'High':
                color = 'red' if row['predicted_magnitude'] >= 6.0 else \
                       'orange' if row['predicted_magnitude'] >= 5.0 else 'yellow'
            else:
                color = 'lightred' if row['predicted_magnitude'] >= 6.0 else \
                       'lightorange' if row['predicted_magnitude'] >= 5.0 else 'lightyellow'
            
            # Create popup text
            popup_text = f"""
            <strong>Predicted Magnitude:</strong> {row['predicted_magnitude']:.1f}<br>
            <strong>Confidence:</strong> {row['confidence']}<br>
            <strong>Zone:</strong> {row['zone']}
            """
            
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=8,
                color=color,
                fill=True,
                popup=popup_text
            ).add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; bottom: 50px; left: 50px; z-index:9999; 
             background-color:white; padding: 10px; border: 2px solid grey; 
             border-radius: 5px; font-size: 14px;">
        <p><strong>Predicted Magnitude:</strong></p>
        <p><span style="color:red;">●</span> ≥ 6.0 (High Confidence)</p>
        <p><span style="color:orange;">●</span> 5.0 - 5.9 (High Confidence)</p>
        <p><span style="color:yellow;">●</span> 4.0 - 4.9 (High Confidence)</p>
        <p><span style="color:lightred;">●</span> ≥ 6.0 (Medium Confidence)</p>
        <p><span style="color:lightorange;">●</span> 5.0 - 5.9 (Medium Confidence)</p>
        <p><span style="color:lightyellow;">●</span> 4.0 - 4.9 (Medium Confidence)</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Display map
        st.write("### Predicted Earthquake Locations (Next 30 Days)")
        st.write("Predictions are based on historical patterns and active zones")
        st.write("Darker colors indicate higher confidence predictions")
        folium_static(m)
        
        # Display prediction statistics
        st.write("### Prediction Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### By Magnitude Range")
            mag_stats = predictions.groupby(pd.cut(predictions['predicted_magnitude'], 
                                                 bins=[4, 5, 6, float('inf')],
                                                 labels=['4.0-4.9', '5.0-5.9', '6.0+'])).size()
            st.write(mag_stats)
            
        with col2:
            st.write("#### By Confidence Level")
            conf_stats = predictions['confidence'].value_counts()
            st.write(conf_stats)
    else:
        st.error("Unable to generate predictions. Please check the data and try again.")

def main():
    # Set title
    st.title("🌎 Earthquake Detection & Prediction System")
    
    # Sidebar for filters
    st.sidebar.header("Data Filters")
    
    # Time period selection
    days = st.sidebar.slider("Days to analyze", 1, 365, 30)
    
    # Magnitude filter
    min_magnitude = st.sidebar.slider("Minimum magnitude", 0.0, 9.0, 2.5, 0.1)
    
    # Load data
    with st.spinner("Loading earthquake data..."):
        df = load_earthquake_data(days, min_magnitude)
    
    if df.empty:
        st.error("No earthquake data available. Please check your internet connection and try again.")
        return
    
    # Convert time to datetime
    df['time'] = pd.to_datetime(df['time'])
    
    # Show data stats
    st.header("Earthquake Data Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Earthquakes", len(df))
    col2.metric("Average Magnitude", f"{df['magnitude'].mean():.2f}")
    col3.metric("Maximum Magnitude", f"{df['magnitude'].max():.2f}")
    col4.metric("Average Depth", f"{df['depth'].mean():.2f} km")
    
    # Add tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Interactive Map", "Basic Analysis", "Future Predictions", "Raw Data"])
    
    with tab1:
        show_map(df)
    
    with tab2:
        show_basic_analysis(df)
        
    with tab3:
        show_risk_analysis(df)
        
    with tab4:
        st.header("Raw Earthquake Data")
        st.write("View and explore the raw earthquake data below:")
        
        # Format the data for display
        display_df = df.copy()
        display_df['time'] = display_df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Add search and filter options
        search_term = st.text_input("Search by location:", "")
        if search_term:
            display_df = display_df[display_df['place'].str.contains(search_term, case=False, na=False)]
        
        # Show the data
        st.dataframe(
            display_df,
            column_config={
                "time": "Time",
                "magnitude": "Magnitude",
                "depth": "Depth (km)",
                "latitude": "Latitude",
                "longitude": "Longitude",
                "place": "Location"
            },
            hide_index=True
        )

if __name__ == "__main__":
    main()