#!/usr/bin/env python
"""
Script to interactively draw zones on the front end for the Zone Time Tracker application.
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


class ZoneDrawer:
    """Class for interactively drawing zones on the front end on an image or video frame."""

    def __init__(self, image: np.ndarray):
        """Initialize the zone drawer.
        
        Args:
            image: The image or video frame to draw zones on.
        """
        self.image = image.copy()
        self.original_image = image.copy()
        self.zones = []
        self.current_zone = []
        self.drawing = False
        self.window_name = "Zone Drawing Tool"
        self.colors = [
            (230, 25, 75),   # Red
            (60, 180, 75),   # Green
            (255, 225, 25),  # Yellow
            (0, 130, 200),   # Blue
            (245, 130, 48),  # Orange
            (145, 30, 180),  # Purple
            (70, 240, 240),  # Cyan
            (240, 50, 230),  # Magenta
        ]
        
        # Create window and set mouse callback
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Add instructions to the image
        self.add_instructions()
    
    def add_instructions(self):
        """Add instructions to the image."""
        instructions = [
            "Instructions:",
            "- Click to add points to the current zone",
            "- Press ENTER to complete the current zone",
            "- Press ESC to cancel the current zone",
            "- Press 's' to save all zones",
            "- Press 'r' to reset all zones",
            "- Press 'q' to quit without saving"
        ]
        
        # Add a semi-transparent overlay for instructions
        overlay = self.image.copy()
        cv2.rectangle(overlay, (10, 10), (400, 180), (0, 0, 0), -1)
        alpha = 0.7
        self.image = cv2.addWeighted(overlay, alpha, self.image, 1 - alpha, 0)
        
        # Add text
        for i, text in enumerate(instructions):
            cv2.putText(
                self.image,
                text,
                (20, 40 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
    
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback function for drawing zones.
        
        Args:
            event: The mouse event.
            x: The x-coordinate of the mouse.
            y: The y-coordinate of the mouse.
            flags: Additional flags.
            param: Additional parameters.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add point to current zone
            self.current_zone.append((x, y))
            self.drawing = True
            self.update_image()
    
    def update_image(self):
        """Update the image with the current zones."""
        # Reset image
        self.image = self.original_image.copy()
        
        # Draw completed zones
        for i, zone in enumerate(self.zones):
            color = self.colors[i % len(self.colors)]
            self.draw_polygon(zone, color, True)
        
        # Draw current zone
        if self.current_zone:
            color = self.colors[len(self.zones) % len(self.colors)]
            self.draw_polygon(self.current_zone, color, False)
        
        # Add instructions
        self.add_instructions()
        
        # Show image
        cv2.imshow(self.window_name, self.image)
    
    def draw_polygon(self, points: List[Tuple[int, int]], color: Tuple[int, int, int], closed: bool):
        """Draw a polygon on the image.
        
        Args:
            points: List of points (x, y) defining the polygon.
            color: Color of the polygon (B, G, R).
            closed: Whether to close the polygon.
        """
        if not points:
            return
        
        # Draw points
        for i, point in enumerate(points):
            cv2.circle(self.image, point, 5, color, -1)
            cv2.putText(
                self.image,
                str(i + 1),
                (point[0] + 10, point[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                1,
                cv2.LINE_AA
            )
        
        # Draw lines between points
        for i in range(len(points) - 1):
            cv2.line(self.image, points[i], points[i + 1], color, 2)
        
        # Close the polygon if needed
        if closed and len(points) > 2:
            cv2.line(self.image, points[-1], points[0], color, 2)
    
    def complete_zone(self):
        """Complete the current zone and add it to the list of zones."""
        if len(self.current_zone) >= 3:
            self.zones.append(self.current_zone)
            self.current_zone = []
            self.drawing = False
            self.update_image()
            print(f"Zone {len(self.zones)} completed")
        else:
            print("A zone must have at least 3 points")
    
    def cancel_zone(self):
        """Cancel the current zone."""
        self.current_zone = []
        self.drawing = False
        self.update_image()
        print("Current zone canceled")
    
    def reset_zones(self):
        """Reset all zones."""
        self.zones = []
        self.current_zone = []
        self.drawing = False
        self.update_image()
        print("All zones reset")
    
    def save_zones(self, output_path: str):
        """Save the zones to a JSON file.
        
        Args:
            output_path: Path to save the zones configuration.
        """
        if not self.zones:
            print("No zones to save")
            return
        
        # Convert zones to the expected format
        zones_data = [zone for zone in self.zones]
        
        # Save to JSON
        with open(output_path, "w") as f:
            json.dump(zones_data, f, indent=2)
        
        print(f"Saved {len(self.zones)} zones to {output_path}")
    
    def run(self):
        """Run the zone drawing tool."""
        print("Zone Drawing Tool")
        print("=================")
        print("Click to add points to the current zone")
        print("Press ENTER to complete the current zone")
        print("Press ESC to cancel the current zone")
        print("Press 's' to save all zones")
        print("Press 'r' to reset all zones")
        print("Press 'q' to quit without saving")
        
        while True:
            cv2.imshow(self.window_name, self.image)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13:  # ENTER
                self.complete_zone()
            elif key == 27:  # ESC
                self.cancel_zone()
            elif key == ord("s"):  # Save
                return True
            elif key == ord("r"):  # Reset
                self.reset_zones()
            elif key == ord("q"):  # Quit
                return False


def get_first_frame(source_path: str) -> np.ndarray:
    """Get the first frame from a video or image.
    
    Args:
        source_path: Path to the video or image.
    
    Returns:
        The first frame or image.
    """
    # Check if the source is a video or image
    if source_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        # Video
        cap = cv2.VideoCapture(source_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Could not read video: {source_path}")
        
        return frame
    else:
        # Image
        frame = cv2.imread(source_path)
        
        if frame is None:
            raise ValueError(f"Could not read image: {source_path}")
        
        return frame


def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Draw zones for Zone Time Tracker")
    parser.add_argument(
        "--source_path",
        type=str,
        required=True,
        help="Path to the source image or video",
    )
    parser.add_argument(
        "--zone_configuration_path",
        type=str,
        required=True,
        help="Path to save the zone configuration",
    )
    args = parser.parse_args()
    
    # Get the first frame
    try:
        frame = get_first_frame(args.source_path)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Create zone drawer
    drawer = ZoneDrawer(frame)
    
    # Run the zone drawer
    should_save = drawer.run()
    
    # Save zones if requested
    if should_save:
        drawer.save_zones(args.zone_configuration_path)
    
    # Clean up
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
