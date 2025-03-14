"""
Video processing module for handling static video files.
"""
import os
import sys
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Optional, Any
import supervision as sv
import time

# Add the parent directory to sys.path to allow imports from calculate_zone_time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import modules from calculate_zone_time
from calculate_zone_time.core.detector import ZoneDetector
from calculate_zone_time.core.tracker import ZoneTracker
from calculate_zone_time.visualization.annotator import ZoneTimeAnnotator


def process_video(
    video_path: str,
    zones_file: str,
    display: bool = True,
    model_path: str = "yolov8n.pt",
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.5,
    alert_threshold: int = 3,
    alert_duration_threshold: int = 5,
    display_every_n_frames: int = 5,
    save_csv: bool = False
) -> Optional[Tuple[pd.DataFrame, int]]:
    """
    Process a video file for zone occupancy tracking.
    
    Args:
        video_path: Path to the video file
        zones_file: Path to the zones JSON file
        display: Whether to display the video during processing
        model_path: Path to the YOLOv8 model to use for detection
        confidence_threshold: Confidence threshold for object detection
        iou_threshold: IoU threshold for non-max suppression
        alert_threshold: Number of people in a zone to trigger an alert
        alert_duration_threshold: Duration in seconds for an alert to be considered
        display_every_n_frames: Display every nth frame (to reduce visual clutter)
        save_csv: Whether to save the results to a CSV file
        
    Returns:
        Tuple of results DataFrame and total people detected, or None if processing failed
    """
    try:
        # Load zones from file
        with open(zones_file, 'r') as f:
            zones_data = json.load(f)
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            st.error(f"Error: Could not open video file {video_path}")
            return None
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Convert zones to polygon zones
        zones = []
        for zone_points in zones_data:
            polygon = np.array(zone_points)
            zones.append(sv.PolygonZone(
                polygon=polygon,
                triggering_anchors=(sv.Position.CENTER,)
            ))
        
        # Initialize components from calculate_zone_time
        detector = ZoneDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold
        )
        
        tracker = ZoneTracker(zones=zones, fps=fps)
        annotator = ZoneTimeAnnotator()
        
        # Initialize results storage
        results_data = []
        zone_occupancy_history = {i: [] for i in range(len(zones))}
        current_alerts = {}
        alert_history = []
        total_people_detected = 0
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Create a single placeholder for frame display
        if display:
            frame_display = st.empty()
        
        # Process video frames
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects
            detections = detector.detect(frame)
            if detections is not None and len(detections) > 0:
                total_people_detected = max(total_people_detected, len(detections))
            
            # Track objects and update zone times
            tracker.update(detections)
            
            # Get statistics
            stats = tracker.get_statistics()
            zone_counts = {i: count for i, count in enumerate(stats["current_objects_in_zone"])}
            
            # Store results for this frame
            timestamp = frame_idx / fps
            formatted_time = str(timedelta(seconds=int(timestamp)))
            
            # Update zone occupancy history
            for zone_id, count in zone_counts.items():
                zone_occupancy_history[zone_id].append((timestamp, count))
            
            # Process alerts for each zone
            for zone_id in range(len(zones)):
                zone_detections = tracker.get_zone_detections(zone_id)
                count = len(zone_detections) if zone_detections is not None else 0
                
                # Check for alerts
                if count >= alert_threshold:
                    if zone_id not in current_alerts:
                        current_alerts[zone_id] = {
                            'start_time': timestamp,
                            'count': count
                        }
                    else:
                        current_alerts[zone_id]['count'] = max(
                            current_alerts[zone_id]['count'], 
                            count
                        )
                
                # Handle alert expiration
                elif zone_id in current_alerts:
                    alert_duration = timestamp - current_alerts[zone_id]['start_time']
                    if alert_duration >= alert_duration_threshold:
                        alert_history.append({
                            'zone_id': zone_id,
                            'start_time': current_alerts[zone_id]['start_time'],
                            'end_time': timestamp,
                            'duration': alert_duration,
                            'max_count': current_alerts[zone_id]['count']
                        })
                    del current_alerts[zone_id]
            
            # Store frame data
            for zone_id, count in zone_counts.items():
                results_data.append({
                    'frame': frame_idx,
                    'timestamp': timestamp,
                    'formatted_time': formatted_time,
                    'zone_id': zone_id,
                    'count': count
                })
            
            # Annotate frame
            if display and (frame_idx % display_every_n_frames == 0):  # Only display every nth frame
                # Create annotated frame
                annotated_frame = frame.copy()
                
                # Draw zones and statistics using the annotator from calculate_zone_time
                annotated_frame = annotator.annotate_zones(annotated_frame, zones)
                annotated_frame = annotator.annotate_statistics(
                    annotated_frame, 
                    stats, 
                    position=(10, 30), 
                    line_spacing=30
                )
                
                # Draw time in zone for each detection
                for zone_id in range(len(zones)):
                    # Get formatted labels for this zone
                    labels = tracker.get_formatted_labels(zone_id)
                    zone_detections = tracker.get_zone_detections(zone_id)
                    
                    if zone_detections is not None and len(zone_detections) > 0:
                        # Use the official annotate_detections method instead of manual drawing
                        annotated_frame = annotator.annotate_detections(
                            annotated_frame,
                            zone_detections,
                            zone_id,
                            labels
                        )
                
                # Display the annotated frame
                frame_display.image(annotated_frame, channels="BGR", use_container_width=True)
                
                # Update progress bar
                progress = min(frame_idx / frame_count, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {frame_idx}/{frame_count} ({progress:.1%})")
            
            frame_idx += 1
        
        # Clean up
        cap.release()
        
        # Store alert history and zone occupancy history in session state
        st.session_state.alert_history = alert_history
        st.session_state.zone_occupancy_history = zone_occupancy_history
        st.session_state.current_alerts = current_alerts
        
        # Create results DataFrame
        results_df = pd.DataFrame(results_data)
        
        # Save results to CSV if requested
        if save_csv and len(results_data) > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"zone_occupancy_{timestamp}.csv"
            results_df.to_csv(csv_filename, index=False)
            st.success(f"Results saved to {csv_filename}")
        
        return results_df, total_people_detected
    
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None 