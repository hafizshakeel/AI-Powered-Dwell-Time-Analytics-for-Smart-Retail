"""
File handling utilities for the Streamlit application.
"""
import os
import tempfile
from typing import List, Tuple
import json


def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary location and return the path"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name


def save_zones_to_json(zones: List[List[Tuple[int, int]]], file_path: str) -> None:
    """Save zones to a JSON file"""
    # Convert tuples to lists for JSON serialization
    json_zones = []
    for zone in zones:
        json_zone = []
        for point in zone:
            json_zone.append([int(point[0]), int(point[1])])
        json_zones.append(json_zone)
    
    with open(file_path, 'w') as f:
        json.dump(json_zones, f) 