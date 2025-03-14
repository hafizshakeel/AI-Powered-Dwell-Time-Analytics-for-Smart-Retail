"""Main Streamlit application for zone occupancy tracking."""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import json
import tempfile
from datetime import datetime
import time
from typing import Dict, List, Tuple, Optional, Any


# Import modules from the modular structure
from utils.file_utils import save_uploaded_file, save_zones_to_json
from utils.zone_utils import draw_zones_interface, display_zone_preview
from processing.video_processor import process_video
from processing.rtsp_processor import process_rtsp_stream
from analytics.dashboard import display_heatmap_analysis, display_alert_history
from alerts.alert_system import display_current_alerts


def initialize_session_state():
    """Initialize session state variables."""
    if 'video_info' not in st.session_state:
        st.session_state.video_info = None
    
    if 'video_path' not in st.session_state:
        st.session_state.video_path = None
    
    if 'rtsp_url' not in st.session_state:
        st.session_state.rtsp_url = ""
    
    if 'zones' not in st.session_state:
        st.session_state.zones = []
    
    if 'zones_file' not in st.session_state:
        st.session_state.zones_file = None
    
    if 'zones_ready' not in st.session_state:
        st.session_state.zones_ready = False
    
    if 'current_frame' not in st.session_state:
        st.session_state.current_frame = None
    
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    
    if 'results_df' not in st.session_state:
        st.session_state.results_df = None
    
    if 'total_people_detected' not in st.session_state:
        st.session_state.total_people_detected = 0
    
    if 'zone_colors' not in st.session_state:
        st.session_state.zone_colors = [
            "#FF5733", "#33FF57", "#3357FF", "#F033FF", "#FF33F0", 
            "#33FFF0", "#F0FF33", "#FF3333", "#33FF33", "#3333FF"
        ]


def set_page_config():
    """Set page configuration and styling."""
    st.set_page_config(
        page_title="Zone Occupancy Tracker",
        page_icon="👥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown(
        """
        <style>
        .header {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 20px;
            color: #1E88E5;
            border-bottom: 1px solid #1E88E5;
            padding-bottom: 5px;
        }
        .sub-header {
            font-size: 20px;
            font-weight: bold;
            margin-top: 15px;
            margin-bottom: 10px;
            color: #333;
        }
        .info-text {
            background-color: #E3F2FD;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 15px;
        }
        .stButton>button {
            width: 100%;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def sidebar_menu():
    """Display sidebar menu and handle input selection."""
    st.sidebar.markdown('<div class="header">Zone Occupancy Tracker</div>', unsafe_allow_html=True)
    
    # Input selection
    st.sidebar.markdown('<div class="sub-header">Input Selection</div>', unsafe_allow_html=True)
    input_type = st.sidebar.radio("Select Input Type", ["Video File", "RTSP Stream"])
    
    # Zone configuration
    st.sidebar.markdown('<div class="sub-header">Zone Configuration</div>', unsafe_allow_html=True)
    
    # Option to upload a predefined zones JSON file
    uploaded_zones_file = st.sidebar.file_uploader("Upload Zones JSON (Optional)", type=["json"])
    
    if uploaded_zones_file is not None:
        # Save uploaded zones file
        zones_path = save_uploaded_file(uploaded_zones_file)
        st.session_state.zones_file = zones_path
        
        # Load zones from file to display info
        try:
            with open(zones_path, 'r') as f:
                zones_data = json.load(f)
            
            st.sidebar.success(f"Loaded {len(zones_data)} zones from file")
            st.session_state.zones = zones_data
            st.session_state.zones_ready = True
        except Exception as e:
            st.sidebar.error(f"Error loading zones file: {str(e)}")
    
    if input_type == "Video File":
        # Video file upload
        uploaded_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
        
        if uploaded_file is not None:
            # Save uploaded file
            video_path = save_uploaded_file(uploaded_file)
            
            # Get video info
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                
                # Read first frame
                ret, frame = cap.read()
                if ret:
                    st.session_state.current_frame = frame
                
                cap.release()
                
                # Store video info
                st.session_state.video_info = {
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "frame_count": frame_count,
                    "duration": duration
                }
                st.session_state.video_path = video_path
                
                # Display video info
                st.sidebar.markdown('<div class="info-text">', unsafe_allow_html=True)
                st.sidebar.markdown(f"**Video Information:**")
                st.sidebar.markdown(f"- Resolution: {width}x{height}")
                st.sidebar.markdown(f"- FPS: {fps:.2f}")
                st.sidebar.markdown(f"- Duration: {duration:.2f} seconds")
                st.sidebar.markdown(f"- Frames: {frame_count}")
                st.sidebar.markdown('</div>', unsafe_allow_html=True)
            else:
                st.sidebar.error("Failed to open video file")
    else:
        # RTSP stream input
        rtsp_url = st.sidebar.text_input("RTSP URL", value=st.session_state.rtsp_url)
        
        if rtsp_url and rtsp_url != st.session_state.rtsp_url:
            st.session_state.rtsp_url = rtsp_url
            
            # Test connection
            cap = cv2.VideoCapture(rtsp_url)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # Read first frame
                ret, frame = cap.read()
                if ret:
                    st.session_state.current_frame = frame
                
                cap.release()
                
                # Store video info
                st.session_state.video_info = {
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "frame_count": 0,
                    "duration": 0
                }
                
                # Display stream info
                st.sidebar.markdown('<div class="info-text">', unsafe_allow_html=True)
                st.sidebar.markdown(f"**Stream Information:**")
                st.sidebar.markdown(f"- Resolution: {width}x{height}")
                st.sidebar.markdown(f"- FPS: {fps:.2f}")
                st.sidebar.markdown('</div>', unsafe_allow_html=True)
            else:
                st.sidebar.error("Failed to connect to RTSP stream")
    
    # Processing parameters
    st.sidebar.markdown('<div class="sub-header">Processing Parameters</div>', unsafe_allow_html=True)
    
    # Model selection
    model_options = {
        "YOLOv8n": "yolov8n.pt",
        "YOLOv8s": "yolov8s.pt",
        "YOLOv8m": "yolov8m.pt",
        "YOLOv8l": "yolov8l.pt",
        "YOLOv8x": "yolov8x.pt"
    }
    
    model_name = st.sidebar.selectbox(
        "Detection Model",
        options=list(model_options.keys()),
        index=0,
        help="Select the YOLOv8 model to use for detection. Larger models are more accurate but slower."
    )
    model_path = model_options[model_name]
    
    confidence_threshold = st.sidebar.slider(
        "Detection Confidence",
        min_value=0.1,
        max_value=1.0,
        value=0.3,
        step=0.05
    )
    
    iou_threshold = st.sidebar.slider(
        "IoU Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Intersection over Union threshold for non-max suppression"
    )
    
    alert_threshold = st.sidebar.slider(
        "Alert Threshold (people)",
        min_value=1,
        max_value=10,
        value=3,
        step=1
    )
    
    alert_duration_threshold = st.sidebar.slider(
        "Alert Duration Threshold (seconds)",
        min_value=1,
        max_value=30,
        value=5,
        step=1
    )
    
    display_every_n_frames = st.sidebar.slider(
        "Display Frequency (every N frames)",
        min_value=1,
        max_value=30,
        value=5,
        step=1,
        help="Higher values will make the display less cluttered but less smooth"
    )
    
    save_csv = st.sidebar.checkbox(
        "Save Results to CSV",
        value=False,
        help="Save the processing results to a CSV file"
    )
    
    if input_type == "RTSP Stream":
        stream_duration = st.sidebar.slider(
            "Stream Processing Duration (seconds)",
            min_value=10,
            max_value=300,
            value=60,
            step=10
        )
    else:
        stream_duration = 60  # Default value for video files
    
    # Return processing parameters
    return {
        "input_type": input_type,
        "model_path": model_path,
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
        "alert_threshold": alert_threshold,
        "alert_duration_threshold": alert_duration_threshold,
        "stream_duration": stream_duration,
        "display_every_n_frames": display_every_n_frames,
        "save_csv": save_csv
    }


def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Set page configuration
    set_page_config()
    
    # Display sidebar menu and get processing parameters
    params = sidebar_menu()
    
    # Main content
    st.markdown('<div class="header">Zone Occupancy Tracker</div>', unsafe_allow_html=True)
    
    # Create tabs
    setup_tab, analytics_tab, alert_history_tab = st.tabs([
        "Setup & Processing", 
        "Heatmap Analysis", 
        "Alert History"
    ])
    
    with setup_tab:
        # Check if video is loaded
        if st.session_state.video_info is None:
            st.info("Please upload a video file or enter an RTSP URL in the sidebar.")
        else:
            # Zone drawing interface
            if not st.session_state.zones_ready:
                draw_zones_interface()
            else:
                # Display zone preview
                if st.session_state.current_frame is not None and st.session_state.zones:
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        display_zone_preview(st.session_state.current_frame, st.session_state.zones)
                    
                    with col2:
                        # Processing controls
                        st.markdown('<div class="sub-header">Processing Controls</div>', unsafe_allow_html=True)
                        
                        # Start button
                        start_button = st.button("Start Processing")
                        
                        if start_button or (st.session_state.zones_ready and not st.session_state.processing_complete):
                            # Auto-processing message
                            auto_process_msg = st.empty()
                            if not start_button and st.session_state.zones_ready:
                                auto_process_msg.info("Video and zones are ready. Starting processing automatically...")
                            
                            # Process based on input type
                            if params["input_type"] == "Video File" and st.session_state.video_path:
                                with st.spinner("Processing video..."):
                                    results = process_video(
                                        video_path=st.session_state.video_path,
                                        zones_file=st.session_state.zones_file,
                                        display=True,
                                        model_path=params["model_path"],
                                        confidence_threshold=params["confidence_threshold"],
                                        iou_threshold=params["iou_threshold"],
                                        alert_threshold=params["alert_threshold"],
                                        alert_duration_threshold=params["alert_duration_threshold"],
                                        display_every_n_frames=params["display_every_n_frames"],
                                        save_csv=params["save_csv"]
                                    )
                                
                                # Clear auto-processing message
                                auto_process_msg.empty()
                                
                                # Store results
                                if results is not None:
                                    results_df, total_people_detected = results
                                    st.session_state.results_df = results_df
                                    st.session_state.total_people_detected = total_people_detected
                                    st.session_state.processing_complete = True
                                    
                                    # Display summary
                                    st.success("Video processing complete!")
                                    st.markdown(f"**Total people detected:** {total_people_detected}")
                                    st.markdown(f"**Processed frames:** {len(results_df) // len(st.session_state.zones)}")
                            
                            elif params["input_type"] == "RTSP Stream" and st.session_state.rtsp_url:
                                with st.spinner("Processing RTSP stream..."):
                                    results = process_rtsp_stream(
                                        rtsp_url=st.session_state.rtsp_url,
                                        zones_file=st.session_state.zones_file,
                                        model_path=params["model_path"],
                                        confidence_threshold=params["confidence_threshold"],
                                        iou_threshold=params["iou_threshold"],
                                        alert_threshold=params["alert_threshold"],
                                        alert_duration_threshold=params["alert_duration_threshold"],
                                        duration=params["stream_duration"],
                                        display_every_n_frames=params["display_every_n_frames"],
                                        save_csv=params["save_csv"]
                                    )
                                
                                # Clear auto-processing message
                                auto_process_msg.empty()
                                
                                # Store results
                                if results is not None:
                                    results_df, total_people_detected = results
                                    st.session_state.results_df = results_df
                                    st.session_state.total_people_detected = total_people_detected
                                    st.session_state.processing_complete = True
                                    
                                    # Display summary
                                    st.success("Stream processing complete!")
                                    st.markdown(f"**Total people detected:** {total_people_detected}")
                                    st.markdown(f"**Processed frames:** {len(results_df) // len(st.session_state.zones)}")
                        
                        # Display current alerts if processing is complete
                        if st.session_state.processing_complete and 'current_alerts' in st.session_state:
                            display_current_alerts(
                                st.session_state.current_alerts, 
                                params["alert_threshold"]
                            )
    
    # Analytics tab
    with analytics_tab:
        if st.session_state.processing_complete and st.session_state.results_df is not None:
            display_heatmap_analysis(st.session_state.results_df)
        else:
            st.info("Process a video or stream to view analytics.")
    
    # Alert history tab
    with alert_history_tab:
        if st.session_state.processing_complete:
            display_alert_history()
        else:
            st.info("Process a video or stream to view alert history.")


if __name__ == "__main__":
    main()
