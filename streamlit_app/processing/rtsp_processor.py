"""
RTSP stream processing module for handling live video streams.
"""
import os
import sys
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import json
import threading
import time
from typing import Dict, List, Tuple, Optional, Any
import supervision as sv

# Add the parent directory to sys.path to allow imports from calculate_zone_time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import modules from calculate_zone_time
from calculate_zone_time.core.detector import ZoneDetector
from calculate_zone_time.core.tracker import ZoneTracker
from calculate_zone_time.visualization.annotator import ZoneTimeAnnotator


class RTSPProcessor:
    """Class for processing RTSP streams with zone occupancy tracking."""
    
    def __init__(
        self,
        rtsp_url: str,
        zones_file: str,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.5,
        alert_threshold: int = 3,
        alert_duration_threshold: int = 5
    ):
        """
        Initialize the RTSP processor.
        
        Args:
            rtsp_url: URL of the RTSP stream
            zones_file: Path to the zones JSON file
            model_path: Path to the YOLOv8 model to use for detection
            confidence_threshold: Confidence threshold for object detection
            iou_threshold: IoU threshold for non-max suppression
            alert_threshold: Number of people in a zone to trigger an alert
            alert_duration_threshold: Duration in seconds for an alert to be considered
        """
        self.rtsp_url = rtsp_url
        self.zones_file = zones_file
        self.alert_threshold = alert_threshold
        self.alert_duration_threshold = alert_duration_threshold
        
        # Load zones from file
        with open(zones_file, 'r') as f:
            self.zones_data = json.load(f)
        
        # Convert zones to polygon zones
        self.zones = []
        for zone_points in self.zones_data:
            polygon = np.array(zone_points)
            self.zones.append(sv.PolygonZone(
                polygon=polygon,
                triggering_anchors=(sv.Position.CENTER,)
            ))
        
        # Initialize components from calculate_zone_time
        self.detector = ZoneDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold
        )
        
        # Initialize tracker with zones (use_clock_timer=True for RTSP streams)
        self.tracker = ZoneTracker(zones=self.zones, use_clock_timer=True)
        
        # Initialize annotator
        self.annotator = ZoneTimeAnnotator()
        
        # Initialize results storage
        self.results_data = []
        self.zone_occupancy_history = {i: [] for i in range(len(self.zones))}
        self.current_alerts = {}
        self.alert_history = []
        self.total_people_detected = 0
        
        # Initialize frame storage
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # Initialize processing state
        self.is_running = False
        self.start_time = None
        self.processing_thread = None
    
    def start(self):
        """Start processing the RTSP stream."""
        if not self.is_running:
            self.is_running = True
            self.start_time = time.time()
            self.processing_thread = threading.Thread(target=self._process_stream)
            self.processing_thread.daemon = True
            self.processing_thread.start()
    
    def stop(self):
        """Stop processing the RTSP stream."""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
    
    def get_current_frame(self):
        """Get the current processed frame."""
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def _process_stream(self):
        """Process the RTSP stream in a loop."""
        # Initialize video capture
        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            st.error(f"Error: Could not open RTSP stream {self.rtsp_url}")
            self.is_running = False
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Process frames
        frame_idx = 0
        
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                # Try to reconnect
                cap.release()
                time.sleep(1.0)
                cap = cv2.VideoCapture(self.rtsp_url)
                continue
            
            # Detect objects
            detections = self.detector.detect(frame)
            if detections is not None and len(detections) > 0:
                self.total_people_detected = max(self.total_people_detected, len(detections))
            
            # Track objects and update zone times
            self.tracker.update(detections)
            
            # Get statistics
            stats = self.tracker.get_statistics()
            zone_counts = {i: count for i, count in enumerate(stats["current_objects_in_zone"])}
            
            # Calculate timestamp
            elapsed_time = time.time() - self.start_time
            formatted_time = str(timedelta(seconds=int(elapsed_time)))
            
            # Update zone occupancy history
            for zone_id, count in zone_counts.items():
                self.zone_occupancy_history[zone_id].append((elapsed_time, count))
                
                # Limit history size to prevent memory issues
                if len(self.zone_occupancy_history[zone_id]) > 3600:  # Limit to 1 hour at 1 fps
                    self.zone_occupancy_history[zone_id] = self.zone_occupancy_history[zone_id][-1800:]
            
            # Process alerts for each zone
            for zone_id in range(len(self.zones)):
                zone_detections = self.tracker.get_zone_detections(zone_id)
                count = len(zone_detections) if zone_detections is not None else 0
                
                # Check for alerts
                if count >= self.alert_threshold:
                    if zone_id not in self.current_alerts:
                        self.current_alerts[zone_id] = {
                            'start_time': elapsed_time,
                            'count': count
                        }
                    else:
                        self.current_alerts[zone_id]['count'] = max(
                            self.current_alerts[zone_id]['count'], 
                            count
                        )
                
                # Handle alert expiration
                elif zone_id in self.current_alerts:
                    alert_duration = elapsed_time - self.current_alerts[zone_id]['start_time']
                    if alert_duration >= self.alert_duration_threshold:
                        self.alert_history.append({
                            'zone_id': zone_id,
                            'start_time': self.current_alerts[zone_id]['start_time'],
                            'end_time': elapsed_time,
                            'duration': alert_duration,
                            'max_count': self.current_alerts[zone_id]['count']
                        })
                    del self.current_alerts[zone_id]
            
            # Store frame data
            for zone_id, count in zone_counts.items():
                self.results_data.append({
                    'frame': frame_idx,
                    'timestamp': elapsed_time,
                    'formatted_time': formatted_time,
                    'zone_id': zone_id,
                    'count': count
                })
                
                # Limit results data size to prevent memory issues
                if len(self.results_data) > 10000:
                    self.results_data = self.results_data[-5000:]
            
            # Annotate frame
            annotated_frame = frame.copy()
            
            # Draw zones and statistics using the annotator from calculate_zone_time
            annotated_frame = self.annotator.annotate_zones(annotated_frame, self.zones)
            annotated_frame = self.annotator.annotate_statistics(
                annotated_frame, 
                stats, 
                position=(10, 30), 
                line_spacing=30
            )
            
            # Draw time in zone for each detection
            for zone_id in range(len(self.zones)):
                # Get formatted labels for this zone
                labels = self.tracker.get_formatted_labels(zone_id)
                zone_detections = self.tracker.get_zone_detections(zone_id)
                
                if zone_detections is not None and len(zone_detections) > 0:
                    # Use the official annotate_detections method instead of manual drawing
                    annotated_frame = self.annotator.annotate_detections(
                        annotated_frame,
                        zone_detections,
                        zone_id,
                        labels
                    )
            
            # Update the current frame
            with self.frame_lock:
                self.current_frame = annotated_frame
            
            frame_idx += 1
        
        # Clean up
        cap.release()


def process_rtsp_stream(
    rtsp_url: str,
    zones_file: str,
    model_path: str = "yolov8n.pt",
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.5,
    alert_threshold: int = 3,
    alert_duration_threshold: int = 5,
    duration: int = 60,  # Duration in seconds to process the stream
    display_every_n_frames: int = 5,  # Display every nth frame (to reduce visual clutter)
    save_csv: bool = False
) -> Optional[Tuple[pd.DataFrame, int]]:
    """
    Process an RTSP stream for zone occupancy tracking.
    
    Args:
        rtsp_url: RTSP URL of the stream
        zones_file: Path to the zones JSON file
        model_path: Path to the YOLOv8 model to use for detection
        confidence_threshold: Confidence threshold for object detection
        iou_threshold: IoU threshold for non-max suppression
        alert_threshold: Number of people in a zone to trigger an alert
        alert_duration_threshold: Duration in seconds for an alert to be considered
        duration: Duration in seconds to process the stream
        display_every_n_frames: Display every nth frame (to reduce visual clutter)
        save_csv: Whether to save the results to a CSV file
        
    Returns:
        Tuple of results DataFrame and total people detected, or None if processing failed
    """
    try:
        processor = RTSPProcessor(
            rtsp_url=rtsp_url,
            zones_file=zones_file,
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            alert_threshold=alert_threshold,
            alert_duration_threshold=alert_duration_threshold
        )
        
        # Start processing
        processor.start()
        
        # Create a placeholder for frame display
        frame_display = st.empty()
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process for the specified duration
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < duration:
            # Get the current frame
            frame = processor.get_current_frame()
            
            # Display the frame
            if frame is not None and frame_count % display_every_n_frames == 0:
                frame_display.image(frame, channels="BGR", use_container_width=True)
            
            # Update progress
            elapsed = time.time() - start_time
            progress = min(elapsed / duration, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Processing: {progress:.1%} complete")
            
            # Sleep to reduce CPU usage
            time.sleep(0.1)
            frame_count += 1
        
        # Stop processing
        processor.stop()
        
        # Clean up
        progress_bar.empty()
        status_text.empty()
        
        # Save results to CSV if requested
        if save_csv and processor.results_data:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"rtsp_results_{timestamp}.csv"
            results_df = pd.DataFrame(processor.results_data)
            results_df.to_csv(csv_filename, index=False)
            st.success(f"Results saved to {csv_filename}")
        
        # Update session state
        st.session_state.alert_history = processor.alert_history
        st.session_state.zone_occupancy_history = processor.zone_occupancy_history
        st.session_state.current_alerts = processor.current_alerts
        
        # Return results
        results_df = pd.DataFrame(processor.results_data)
        return results_df, processor.total_people_detected
        
    except Exception as e:
        st.error(f"Error processing RTSP stream: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None 