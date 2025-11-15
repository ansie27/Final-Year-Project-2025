"""
Utility functions for Green Supply Chain Risk Management Project

This module provides helper functions and utilities used across the project.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime


def ensure_directory(path):
    """
    Ensure a directory exists, create if it doesn't.
    
    Parameters:
    -----------
    path : str or Path
        Directory path
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def save_dataframe(df, filepath, index=False):
    """
    Save DataFrame to CSV file.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to save
    filepath : str or Path
        Output file path
    index : bool
        Whether to include index in output
    """
    ensure_directory(Path(filepath).parent)
    df.to_csv(filepath, index=index)
    print(f"✓ Saved: {filepath}")


def load_dataframe(filepath):
    """
    Load DataFrame from CSV file.
    
    Parameters:
    -----------
    filepath : str or Path
        Input file path
        
    Returns:
    --------
    pd.DataFrame
        Loaded DataFrame
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    return pd.read_csv(filepath)


def print_section_header(title, width=70):
    """
    Print a formatted section header.
    
    Parameters:
    -----------
    title : str
        Section title
    width : int
        Width of the header
    """
    print("\n" + "="*width)
    print(title.center(width))
    print("="*width + "\n")


def print_progress(message, step=None, total=None):
    """
    Print progress message.
    
    Parameters:
    -----------
    message : str
        Progress message
    step : int, optional
        Current step number
    total : int, optional
        Total number of steps
    """
    if step is not None and total is not None:
        print(f"[{step}/{total}] {message}")
    else:
        print(f"  {message}")


def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to validate
    required_columns : list, optional
        List of required column names
    min_rows : int
        Minimum number of rows required
        
    Returns:
    --------
    bool
        True if valid, raises ValueError if invalid
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    if len(df) < min_rows:
        raise ValueError(f"DataFrame must have at least {min_rows} rows")
    
    if required_columns:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True


def calculate_percentage(value, total):
    """
    Calculate percentage.
    
    Parameters:
    -----------
    value : float
        Value
    total : float
        Total value
        
    Returns:
    --------
    float
        Percentage
    """
    if total == 0:
        return 0.0
    return (value / total) * 100


def format_number(value, decimals=2):
    """
    Format number with specified decimal places.
    
    Parameters:
    -----------
    value : float
        Number to format
    decimals : int
        Number of decimal places
        
    Returns:
    --------
    str
        Formatted number string
    """
    return f"{value:.{decimals}f}"


def get_timestamp():
    """
    Get current timestamp as string.
    
    Returns:
    --------
    str
        Timestamp string
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_json(data, filepath):
    """
    Save data to JSON file.
    
    Parameters:
    -----------
    data : dict or list
        Data to save
    filepath : str or Path
        Output file path
    """
    ensure_directory(Path(filepath).parent)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"✓ Saved JSON: {filepath}")


def load_json(filepath):
    """
    Load data from JSON file.
    
    Parameters:
    -----------
    filepath : str or Path
        Input file path
        
    Returns:
    --------
    dict or list
        Loaded data
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def safe_divide(numerator, denominator, default=0.0):
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Parameters:
    -----------
    numerator : float
        Numerator
    denominator : float
        Denominator
    default : float
        Default value if division by zero
        
    Returns:
    --------
    float
        Result of division or default
    """
    if denominator == 0:
        return default
    return numerator / denominator


def get_memory_usage(df):
    """
    Get memory usage of DataFrame in MB.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame
        
    Returns:
    --------
    float
        Memory usage in MB
    """
    return df.memory_usage(deep=True).sum() / 1024**2


def print_dataframe_info(df, name="DataFrame"):
    """
    Print summary information about DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame
    name : str
        Name of the DataFrame
    """
    print(f"\n{name} Information:")
    print(f"  Shape: {df.shape}")
    print(f"  Memory Usage: {get_memory_usage(df):.2f} MB")
    print(f"  Missing Values: {df.isnull().sum().sum()}")
    print(f"  Duplicate Rows: {df.duplicated().sum()}")


if __name__ == "__main__":
    print("Utility Module")
    print("Import this module to use utility functions in your pipeline.")

