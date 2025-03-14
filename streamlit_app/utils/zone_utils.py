"""
Zone drawing and management utilities for the Streamlit application.
"""
import os
import tempfile
import cv2
import numpy as np
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import supervision as sv
from typing import List, Tuple

from .file_utils import save_zones_to_json


def draw_zones_interface():
    """Interface for drawing zones on a video frame"""
    st.markdown('<div class="sub-header">Draw Zones</div>', unsafe_allow_html=True)
    
    # Check if zones are already loaded from a file
    if st.session_state.zones_ready and st.session_state.zones_file:
        st.success(f"Zones already loaded from file: {st.session_state.zones_file}")
        
        # Display the loaded zones
        if st.session_state.current_frame is not None and st.session_state.zones:
            frame = st.session_state.current_frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create a clean copy of the frame to show final zones
            zone_frame = rgb_frame.copy()
            
            # Draw each zone with a different color
            for i, zone in enumerate(st.session_state.zones):
                color_idx = i % len(st.session_state.zone_colors)
                color_hex = st.session_state.zone_colors[color_idx]
                # Convert hex to RGB
                color = sv.Color.from_hex(color_hex)
                # Draw the zone
                zone_frame = sv.draw_polygon(
                    scene=zone_frame,
                    polygon=np.array(zone),
                    color=color,
                    thickness=2
                )
                
                # Add zone number
                centroid = np.mean(zone, axis=0).astype(int)
                cv2.putText(
                    zone_frame,
                    f"Zone {i+1}",
                    (centroid[0], centroid[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
            
            # Display the zones
            st.image(zone_frame, use_container_width=True, caption="Loaded Zones")
        
        return
    
    # Get the first frame of the video
    if st.session_state.video_info and st.session_state.current_frame is not None:
        frame = st.session_state.current_frame
        height, width = frame.shape[:2]
        
        # Instructions
        st.markdown("""
        <div class="info-text">
            <b>Instructions:</b><br>
            1. Click on the image to add points to the current zone<br>
            2. Press 'Complete Zone' when you've added all points for a zone<br>
            3. Press 'New Zone' to start drawing a new zone<br>
            4. Press 'Clear All Zones' to start over
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize session state for current zone points if not exists
        if 'current_zone_points' not in st.session_state:
            st.session_state.current_zone_points = []
        
        # Convert frame to RGB for display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create a copy of the frame to draw on
        drawing_frame = rgb_frame.copy()
        
        # Draw existing zones
        if st.session_state.zones:
            for i, zone in enumerate(st.session_state.zones):
                color_idx = i % len(st.session_state.zone_colors)
                color_hex = st.session_state.zone_colors[color_idx]
                # Convert hex to RGB
                color = sv.Color.from_hex(color_hex)
                # Draw the zone
                drawing_frame = sv.draw_polygon(
                    scene=drawing_frame,
                    polygon=np.array(zone),
                    color=color,
                    thickness=2
                )
                # Add zone number
                centroid = np.mean(zone, axis=0).astype(int)
                cv2.putText(
                    drawing_frame,
                    f"Zone {i+1}",
                    (centroid[0], centroid[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
        
        # Draw current zone points
        if st.session_state.current_zone_points:
            # Draw points
            for i, point in enumerate(st.session_state.current_zone_points):
                cv2.circle(drawing_frame, point, 5, (255, 0, 0), -1)
                cv2.putText(
                    drawing_frame,
                    str(i + 1),
                    (point[0] + 10, point[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA
                )
            
            # Draw lines between points
            for i in range(len(st.session_state.current_zone_points) - 1):
                cv2.line(
                    drawing_frame, 
                    st.session_state.current_zone_points[i], 
                    st.session_state.current_zone_points[i + 1], 
                    (255, 0, 0), 
                    2
                )
            
            # Draw line from last point to first point if more than 2 points
            if len(st.session_state.current_zone_points) > 2:
                cv2.line(
                    drawing_frame, 
                    st.session_state.current_zone_points[-1], 
                    st.session_state.current_zone_points[0], 
                    (255, 0, 0), 
                    2,
                    cv2.LINE_AA
                )
        
        # Display the image and get coordinates when clicked
        clicked_coords = streamlit_image_coordinates(
            drawing_frame,
            key="zone_image"
        )
        
        # Handle click on image
        if clicked_coords:
            point = (clicked_coords['x'], clicked_coords['y'])
            if point not in st.session_state.current_zone_points:
                st.session_state.current_zone_points.append(point)
                st.rerun()
        
        # Control buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Complete Zone"):
                if len(st.session_state.current_zone_points) >= 3:
                    st.session_state.zones.append(st.session_state.current_zone_points)
                    st.session_state.current_zone_points = []
                    st.success(f"Zone {len(st.session_state.zones)} added!")
                    st.rerun()
                else:
                    st.error("A zone must have at least 3 points")
        
        with col2:
            if st.button("New Zone"):
                st.session_state.current_zone_points = []
                st.rerun()
        
        with col3:
            if st.button("Clear All Zones"):
                st.session_state.zones = []
                st.session_state.current_zone_points = []
                st.success("All zones cleared!")
                st.rerun()
        
        # Display current zones
        if st.session_state.zones:
            st.markdown('<div class="sub-header">Current Zones</div>', unsafe_allow_html=True)
            
            # Create a clean copy of the frame to show final zones
            zone_frame = rgb_frame.copy()
            
            # Draw each zone with a different color
            for i, zone in enumerate(st.session_state.zones):
                color_idx = i % len(st.session_state.zone_colors)
                color_hex = st.session_state.zone_colors[color_idx]
                # Convert hex to RGB
                color = sv.Color.from_hex(color_hex)
                # Draw the zone
                zone_frame = sv.draw_polygon(
                    scene=zone_frame,
                    polygon=np.array(zone),
                    color=color,
                    thickness=2
                )
                
                # Add zone number
                centroid = np.mean(zone, axis=0).astype(int)
                cv2.putText(
                    zone_frame,
                    f"Zone {i+1}",
                    (centroid[0], centroid[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
            
            # Display the zones
            st.image(zone_frame, use_container_width=True, caption="Defined Zones")
            
            # Save zones button
            if st.button("Save Zones and Continue"):
                # Save zones to a temporary file
                zones_file = os.path.join(tempfile.gettempdir(), "zones.json")
                save_zones_to_json(st.session_state.zones, zones_file)
                st.session_state.zones_file = zones_file
                st.success(f"Zones saved to {zones_file}")
                st.session_state.zones_ready = True
                st.rerun()


def display_zone_preview(frame, zones):
    """Display a preview of zones on a frame"""
    if frame is not None and zones:
        # Create a small preview of the zones
        rgb_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
        
        # Draw zones on the frame
        for i, zone in enumerate(zones):
            color_idx = i % len(st.session_state.zone_colors)
            color_hex = st.session_state.zone_colors[color_idx]
            color = sv.Color.from_hex(color_hex)
            rgb_frame = sv.draw_polygon(
                scene=rgb_frame,
                polygon=np.array(zone),
                color=color,
                thickness=2
            )
            # Add zone number
            centroid = np.mean(zone, axis=0).astype(int)
            cv2.putText(
                rgb_frame,
                f"Zone {i+1}",
                (centroid[0], centroid[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
        
        # Display a smaller version of the image
        h, w = rgb_frame.shape[:2]
        preview_width = 300
        preview_height = int(h * (preview_width / w))
        preview = cv2.resize(rgb_frame, (preview_width, preview_height))
        st.image(preview, caption="Zones Preview", use_container_width=True) 