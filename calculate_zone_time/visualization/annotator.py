"""
Visualization utilities for zone time calculation.
"""
from typing import List, Optional, Dict

import cv2
import numpy as np
import supervision as sv
from supervision.draw.color import Color, ColorPalette


class ZoneTimeAnnotator:
    """
    Annotator for visualizing zone time tracking.
    
    This class provides methods to annotate frames with zone polygons,
    object detections, and time spent in zones.
    """
    
    def __init__(
        self,
        colors: List[str] = None,
        text_color: str = "#000000",
        text_scale: float = 0.5,
        text_thickness: int = 1,
        box_thickness: int = 2,
    ):
        """
        Initialize the ZoneTimeAnnotator.
        
        Args:
            colors: List of hex color codes for zones and detections.
            text_color: Hex color code for text.
            text_scale: Scale factor for text size.
            text_thickness: Thickness of text.
            box_thickness: Thickness of bounding boxes.
        """
        # Default colors if none provided
        if colors is None:
            colors = ["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"]
            
        self.colors = ColorPalette.from_hex(colors)
        self.text_color = Color.from_hex(text_color)
        self.text_scale = text_scale
        self.text_thickness = text_thickness
        self.box_thickness = box_thickness
        
        # Initialize annotators
        self.color_annotator = sv.ColorAnnotator(color=self.colors)
        self.label_annotator = sv.LabelAnnotator(
            color=self.colors,
            text_color=self.text_color,
            text_scale=text_scale,
            text_thickness=text_thickness
        )
        
    def annotate_zones(
        self,
        scene: np.ndarray,
        zones: List[sv.PolygonZone]
    ) -> np.ndarray:
        """
        Draw zone polygons on the frame.
        
        Args:
            scene: Input frame.
            zones: List of polygon zones.
            
        Returns:
            np.ndarray: Annotated frame with zone polygons.
        """
        annotated_frame = scene.copy()
        
        for idx, zone in enumerate(zones):
            annotated_frame = sv.draw_polygon(
                scene=annotated_frame,
                polygon=zone.polygon,
                color=self.colors.by_idx(idx),
                thickness=self.box_thickness
            )
            
        return annotated_frame
    
    def annotate_detections(
        self,
        scene: np.ndarray,
        detections: sv.Detections,
        zone_idx: int,
        labels: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Draw detections and labels on the frame.
        
        Args:
            scene: Input frame.
            detections: Object detections.
            zone_idx: Index of the zone for color lookup.
            labels: Optional list of labels for detections.
            
        Returns:
            np.ndarray: Annotated frame with detections and labels.
        """
        if detections is None or len(detections) == 0:
            return scene.copy()
            
        annotated_frame = scene.copy()
        custom_color_lookup = np.full(detections.class_id.shape, zone_idx)
        
        # Draw bounding boxes
        annotated_frame = self.color_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            custom_color_lookup=custom_color_lookup
        )
        
        # Draw labels if provided
        if labels and len(labels) > 0:
            annotated_frame = self.label_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
                labels=labels,
                custom_color_lookup=custom_color_lookup
            )
            
        return annotated_frame
    
    def annotate_fps(
        self,
        scene: np.ndarray,
        fps: float,
        position: tuple = (20, 40)
    ) -> np.ndarray:
        """
        Draw FPS counter on the frame.
        
        Args:
            scene: Input frame.
            fps: Current frames per second.
            position: Position to draw the FPS counter.
            
        Returns:
            np.ndarray: Annotated frame with FPS counter.
        """
        annotated_frame = scene.copy()
        
        # Create a small panel for FPS
        panel_width = 100
        panel_height = 40
        panel_top_left = (position[0] - 10, position[1] - 25)
        panel_bottom_right = (panel_top_left[0] + panel_width, panel_top_left[1] + panel_height)
        
        # Draw semi-transparent panel
        overlay = annotated_frame.copy()
        cv2.rectangle(
            overlay, 
            panel_top_left, 
            panel_bottom_right, 
            (30, 30, 30),  # Dark gray
            -1  # Filled rectangle
        )
        # Apply the overlay with transparency
        alpha = 0.7  # Transparency factor
        cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)
        
        # Add a title bar
        title_bar_height = 20
        cv2.rectangle(
            annotated_frame,
            (panel_top_left[0], panel_top_left[1]),
            (panel_bottom_right[0], panel_top_left[1] + title_bar_height),
            (215, 55, 0),  # Orange title bar for FPS
            -1  # Filled rectangle
        )
        
        # Add title text
        cv2.putText(
            annotated_frame,
            "FPS",
            (panel_top_left[0] + 10, panel_top_left[1] + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),  # White text
            1,
            cv2.LINE_AA
        )
        
        # Add FPS value
        cv2.putText(
            annotated_frame,
            f"{fps:.1f}",
            (panel_top_left[0] + 15, panel_top_left[1] + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,  # Larger text for FPS value
            (0, 255, 255),  # Cyan text
            1,
            cv2.LINE_AA
        )
        
        return annotated_frame
        
    def annotate_statistics(
        self,
        scene: np.ndarray,
        stats: Dict,
        position: tuple = (20, 80),
        line_spacing: int = 30
    ) -> np.ndarray:
        """
        Draw statistics on the frame.
        
        Args:
            scene: Input frame.
            stats: Dictionary containing statistics.
            position: Position to start drawing the statistics.
            line_spacing: Vertical spacing between lines.
            
        Returns:
            np.ndarray: Annotated frame with statistics.
        """
        annotated_frame = scene.copy()
        
        # Get zone counts
        zone_counts = stats.get("current_objects_in_zone", [])
        
        # Create a simple panel for zone counts
        panel_width = 200
        panel_height = (len(zone_counts) + 1) * line_spacing
        panel_top_left = (position[0] - 10, position[1] - 25)
        panel_bottom_right = (panel_top_left[0] + panel_width, panel_top_left[1] + panel_height)
        
        # Draw semi-transparent panel
        overlay = annotated_frame.copy()
        cv2.rectangle(
            overlay, 
            panel_top_left, 
            panel_bottom_right, 
            (30, 30, 30),  # Dark gray
            -1  # Filled rectangle
        )
        # Apply the overlay with transparency
        alpha = 0.7  # Transparency factor
        cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)
        
        # Add a title bar
        title_bar_height = 30
        cv2.rectangle(
            annotated_frame,
            (panel_top_left[0], panel_top_left[1]),
            (panel_bottom_right[0], panel_top_left[1] + title_bar_height),
            (0, 120, 215),  # Blue title bar
            -1  # Filled rectangle
        )
        
        # Add title text
        cv2.putText(
            annotated_frame,
            "ZONE OCCUPANCY",
            (panel_top_left[0] + 10, panel_top_left[1] + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),  # White text
            1,
            cv2.LINE_AA
        )
        
        # Draw zone counts with zone-specific colors
        start_y = panel_top_left[1] + title_bar_height + 25
        for idx, count in enumerate(zone_counts):
            # Convert the zone color to BGR for cv2
            zone_color = self.colors.by_idx(idx)
            bgr_color = (int(zone_color.b * 255), int(zone_color.g * 255), int(zone_color.r * 255))
            
            y_pos = start_y + idx * line_spacing
            cv2.putText(
                annotated_frame,
                f"Zone {idx+1}: {count} people",
                (panel_top_left[0] + 15, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,  # Slightly larger text
                bgr_color,  # Use the zone color
                1,
                cv2.LINE_AA
            )
            
        return annotated_frame

    def annotate_alert(
        self,
        scene: np.ndarray,
        alert_text: str,
        position: tuple = (20, 400)
    ) -> np.ndarray:
        """
        Draw alert text on the frame.
        
        Args:
            scene: Input frame.
            alert_text: Alert text to display.
            position: Position to draw the alert text.
            
        Returns:
            np.ndarray: Annotated frame with alert text.
        """
        annotated_frame = scene.copy()
        
        # Split text into lines
        lines = alert_text.strip().split('\n')
        
        # Calculate panel dimensions
        line_height = 25
        panel_width = 300
        panel_height = len(lines) * line_height + 10
        panel_top_left = (position[0] - 10, position[1] - 25)
        panel_bottom_right = (panel_top_left[0] + panel_width, panel_top_left[1] + panel_height)
        
        # Draw semi-transparent panel
        overlay = annotated_frame.copy()
        cv2.rectangle(
            overlay, 
            panel_top_left, 
            panel_bottom_right, 
            (30, 30, 30),  # Dark gray
            -1  # Filled rectangle
        )
        # Apply the overlay with transparency
        alpha = 0.7  # Transparency factor
        cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)
        
        # Add a title bar
        title_bar_height = 30
        cv2.rectangle(
            annotated_frame,
            (panel_top_left[0], panel_top_left[1]),
            (panel_bottom_right[0], panel_top_left[1] + title_bar_height),
            (215, 0, 0),  # Red title bar for alerts
            -1  # Filled rectangle
        )
        
        # Add title text (first line)
        cv2.putText(
            annotated_frame,
            lines[0],
            (panel_top_left[0] + 10, panel_top_left[1] + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),  # White text
            1,
            cv2.LINE_AA
        )
        
        # Add remaining lines
        for i, line in enumerate(lines[1:], 1):
            cv2.putText(
                annotated_frame,
                line,
                (panel_top_left[0] + 10, panel_top_left[1] + title_bar_height + i * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),  # White text
                1,
                cv2.LINE_AA
            )
        
        return annotated_frame
