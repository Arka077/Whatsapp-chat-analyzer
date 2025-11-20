"""
Date utility functions
"""

from datetime import datetime, timedelta
import pandas as pd
from typing import Tuple, Optional

def parse_date(date_str: str) -> datetime:
    """Parse date string to datetime"""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except:
        return datetime.now()

def get_date_range(start: str, end: str) -> Tuple[datetime, datetime]:
    """Get datetime tuple from date strings"""
    return (parse_date(start), parse_date(end))

def filter_df_by_date_range(
    df: pd.DataFrame,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """Filter dataframe by date range"""
    df_copy = df.copy()
    df_copy['only_date'] = pd.to_datetime(df_copy['only_date'])
    
    mask = (df_copy['only_date'].astype(str) >= start_date) & \
           (df_copy['only_date'].astype(str) <= end_date)
    
    return df_copy[mask]

def get_preset_ranges() -> dict:
    """Get preset date ranges"""
    today = datetime.now()
    
    return {
        "Last 7 Days": (
            (today - timedelta(days=7)).strftime("%Y-%m-%d"),
            today.strftime("%Y-%m-%d")
        ),
        "Last 30 Days": (
            (today - timedelta(days=30)).strftime("%Y-%m-%d"),
            today.strftime("%Y-%m-%d")
        ),
        "Last 90 Days": (
            (today - timedelta(days=90)).strftime("%Y-%m-%d"),
            today.strftime("%Y-%m-%d")
        ),
        "Last Year": (
            (today - timedelta(days=365)).strftime("%Y-%m-%d"),
            today.strftime("%Y-%m-%d")
        ),
    }

def validate_date_range(start: str, end: str) -> bool:
    """Validate date range"""
    try:
        start_dt = parse_date(start)
        end_dt = parse_date(end)
        return start_dt <= end_dt
    except:
        return False

def get_date_range_label(start: str, end: str) -> str:
    """Get human-readable date range label"""
    return f"{start} to {end}"