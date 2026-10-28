#!/usr/bin/env python3
"""
Data anonymization and cleaning pipeline for household water consumption data.

This script reads raw water consumption data, cleans it, translates labels,
removes outliers, and anonymizes sensitive information.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Define translation mappings
ATTACHED_TO_MAPPING = {
    'Vatten WC stol': 'Toilet',
    'Tappvatten Kök': 'Kitchen tap',
    'Tvättställ': 'Washbasin',
    'Dusch': 'Shower',
    'Tvättmaskin': 'Washing machine'
}

ROOM_TYPE_MAPPING = {
    'Boenderum': 'Living room',
    'WC': 'Bathroom',
    'Kök': 'Kitchen',
    'Gemensamt': 'Shared space',
    'WC gemensamt': 'Shared bathroom',
    'Gemensam tvättstuga': 'Shared laundry'
}


def load_data(filepath):
    """Load raw CSV data with semicolon delimiter."""
    try:
        df = pd.read_csv(filepath, sep=';', encoding='utf-8')
        print(f"Loaded {len(df)} rows from {filepath}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)


def translate_labels(df):
    """Translate Swedish labels to English."""
    if 'attached_to' in df.columns:
        df['attached_to'] = df['attached_to'].map(
            lambda x: ATTACHED_TO_MAPPING.get(x, x)
        )
    
    if 'room_type' in df.columns:
        df['room_type'] = df['room_type'].map(
            lambda x: ROOM_TYPE_MAPPING.get(x, x)
        )
    
    return df


def remove_outliers_and_zeros(df, value_col='value'):
    """Remove outliers (value > 0.3 m3) and zero-value readings."""
    initial_count = len(df)
    
    # Remove outliers
    df = df[df[value_col] <= 0.3]
    outliers_removed = initial_count - len(df)
    
    # Remove zero-value readings
    initial_count = len(df)
    df = df[df[value_col] > 0]
    zeros_removed = initial_count - len(df)
    
    print(f"Removed {outliers_removed} outlier readings (value > 0.3 m3)")
    print(f"Removed {zeros_removed} zero-value readings")
    
    return df


def anonymize_identifiers(df):
    """Replace sensitive identifiers with random anonymous IDs."""
    anonymization_map = {}
    
    for col in ['apartment', 'cluster', 'sensor_id', 'room_number']:
        if col in df.columns:
            unique_vals = df[col].unique()
            # Create anonymous IDs (e.g., "APT_001", "SENSOR_042")
            prefix = col.upper()[:3]
            if col == 'room_number':
                prefix = 'ROOM'
            anon_ids = {val: f"{prefix}_{i:03d}" for i, val in enumerate(unique_vals)}
            anonymization_map[col] = anon_ids
            df[col] = df[col].map(anon_ids)
            print(f"Anonymized {col}: {len(anon_ids)} unique values")
    
    return df, anonymization_map


def remove_columns(df):
    """Remove sensitive columns."""
    cols_to_remove = ['ID', 'user_id', 'aggregated_event_id']
    cols_to_remove = [col for col in cols_to_remove if col in df.columns]
    if cols_to_remove:
        df = df.drop(columns=cols_to_remove)
        print(f"Removed columns: {', '.join(cols_to_remove)}")
    return df


def add_temporal_features(df, timestamp_col='timestamp'):
    """Add temporal features from timestamp column."""
    if timestamp_col not in df.columns:
        print(f"Warning: {timestamp_col} column not found, skipping temporal features")
        return df
    
    # Convert to datetime if not already
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
    
    # Extract temporal features
    df['date'] = df[timestamp_col].dt.date
    df['hour'] = df[timestamp_col].dt.hour
    df['day_of_week'] = df[timestamp_col].dt.dayofweek
    df['day_name'] = df[timestamp_col].dt.day_name()
    df['month'] = df[timestamp_col].dt.month
    df['year'] = df[timestamp_col].dt.year
    
    print("Added temporal features: date, hour, day_of_week, day_name, month, year")
    
    return df


def save_anonymized_data(df, output_filepath):
    """Save anonymized data as semicolon-delimited CSV."""
    try:
        df.to_csv(output_filepath, sep=';', index=False, encoding='utf-8')
        print(f"Saved anonymized data to {output_filepath}")
        print(f"Final dataset: {len(df)} rows, {len(df.columns)} columns")
        return True
    except Exception as e:
        print(f"Error saving data: {e}")
        return False


def main():
    """Main anonymization pipeline."""
    # Paths
    script_dir = Path(__file__).parent
    raw_data_path = script_dir / '../data/HSB_data_dump.csv'
    output_path = script_dir / 'anonymized_data.csv'
    
    print("=" * 60)
    print("Water Consumption Data Anonymization Pipeline")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading raw data...")
    df = load_data(raw_data_path)
    print(f"   Columns: {', '.join(df.columns)}")
    
    # Translate labels
    print("\n2. Translating Swedish labels to English...")
    df = translate_labels(df)
    
    # Remove outliers and zeros
    print("\n3. Removing outliers and zero-value readings...")
    df = remove_outliers_and_zeros(df)
    
    # Remove sensitive columns
    print("\n4. Removing sensitive columns...")
    df = remove_columns(df)
    
    # Anonymize identifiers
    print("\n5. Anonymizing identifiers...")
    df, anon_map = anonymize_identifiers(df)
    
    # Add temporal features
    print("\n6. Adding temporal features...")
    df = add_temporal_features(df)
    
    # Save anonymized data
    print("\n7. Saving anonymized data...")
    if save_anonymized_data(df, output_path):
        print("\n" + "=" * 60)
        print("Anonymization complete!")
        print("=" * 60)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
