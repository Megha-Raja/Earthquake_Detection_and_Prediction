#!/usr/bin/env python
"""
Data collector module for fetching earthquake data from USGS API.
"""

import requests
import pandas as pd
import time
import os
from datetime import datetime

def fetch_earthquake_data(start_date, end_date, min_magnitude=0):
    """
    Fetch earthquake data from USGS API for a specified time period.
    
    Args:
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
        min_magnitude (float): Minimum earthquake magnitude to include
        
    Returns:
        dict: JSON response from USGS API containing earthquake data
    """
    base_url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    
    params = {
        'format': 'geojson',
        'starttime': start_date,
        'endtime': end_date,
        'minmagnitude': min_magnitude
    }
    
    print(f"Fetching earthquake data from {start_date} to {end_date}...")
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching earthquake data: {e}")
        return None

def process_earthquake_json(earthquake_json):
    """
    Process the JSON response from USGS API into a pandas DataFrame.
    
    Args:
        earthquake_json (dict): JSON response from USGS API
        
    Returns:
        pandas.DataFrame: Processed earthquake data
    """
    if not earthquake_json or 'features' not in earthquake_json:
        print("No earthquake data to process")
        return pd.DataFrame()
    
    # Extract features from the JSON response
    earthquakes = []
    
    for feature in earthquake_json['features']:
        props = feature['properties']
        coords = feature['geometry']['coordinates']
        
        earthquake = {
            'id': feature['id'],
            'magnitude': props.get('mag'),
            'place': props.get('place'),
            'time': datetime.fromtimestamp(props.get('time') / 1000) if props.get('time') else None,
            'updated': datetime.fromtimestamp(props.get('updated') / 1000) if props.get('updated') else None,
            'url': props.get('url'),
            'detail': props.get('detail'),
            'felt': props.get('felt'),
            'cdi': props.get('cdi'),
            'mmi': props.get('mmi'),
            'alert': props.get('alert'),
            'status': props.get('status'),
            'tsunami': props.get('tsunami'),
            'sig': props.get('sig'),
            'net': props.get('net'),
            'code': props.get('code'),
            'ids': props.get('ids'),
            'sources': props.get('sources'),
            'types': props.get('types'),
            'longitude': coords[0] if coords and len(coords) > 0 else None,
            'latitude': coords[1] if coords and len(coords) > 1 else None,
            'depth': coords[2] if coords and len(coords) > 2 else None,
        }
        
        earthquakes.append(earthquake)
    
    # Create DataFrame from the list of dictionaries
    df = pd.DataFrame(earthquakes)
    
    # Convert time columns to datetime
    if 'time' in df.columns:
        df['date'] = df['time'].dt.date
        df['hour'] = df['time'].dt.hour
    
    return df

def fetch_and_save_earthquake_data(start_date, end_date, output_path, min_magnitude=0):
    """
    Fetch earthquake data and save it to a CSV file.
    
    Args:
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
        output_path (str): Path to save the CSV file
        min_magnitude (float): Minimum earthquake magnitude to include
        
    Returns:
        str: Path to the saved CSV file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Fetch data
    earthquake_json = fetch_earthquake_data(start_date, end_date, min_magnitude)
    
    if earthquake_json:
        # Process data
        df = process_earthquake_json(earthquake_json)
        
        if not df.empty:
            # Save to CSV
            df.to_csv(output_path, index=False)
            print(f"Saved {len(df)} earthquake records to {output_path}")
            return output_path
        else:
            print("No earthquake data to save")
            return None
    else:
        print("Failed to fetch earthquake data")
        return None

def fetch_realtime_earthquakes(hours=1, min_magnitude=0):
    """
    Fetch real-time earthquake data for the last specified hours.
    
    Args:
        hours (int): Number of hours to look back
        min_magnitude (float): Minimum earthquake magnitude to include
        
    Returns:
        pandas.DataFrame: Real-time earthquake data
    """
    end_date = datetime.utcnow()
    start_date = end_date - pd.Timedelta(hours=hours)
    
    earthquake_json = fetch_earthquake_data(
        start_date.strftime("%Y-%m-%d %H:%M:%S"),
        end_date.strftime("%Y-%m-%d %H:%M:%S"),
        min_magnitude
    )
    
    if earthquake_json:
        return process_earthquake_json(earthquake_json)
    else:
        return pd.DataFrame()

if __name__ == "__main__":
    # Test the data collection functions
    today = datetime.utcnow().strftime("%Y-%m-%d")
    one_month_ago = (datetime.utcnow() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    
    output_path = "data/raw/test_earthquakes.csv"
    fetch_and_save_earthquake_data(one_month_ago, today, output_path, min_magnitude=4.5)