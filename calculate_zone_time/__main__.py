"""
Main entry point for the zone time calculation application.
"""
import time
from collections import deque
from typing import List, Optional

import cv2
import numpy as np
import supervision as sv

from .core.detector import ZoneDetector
from .core.tracker import ZoneTracker
from .utils.cli import parse_arguments
from .utils.general import load_zones_config
from .utils.video import (
    get_frame_generator,
    get_video_info,
    initialize_video_writer,
    setup_display_window,
)
from .visualization.annotator import ZoneTimeAnnotator


class FPSCounter:
    """Tracks and calculates frames per second."""
    
    def __init__(self, avg_frames=30):
        """
        Initialize the FPS counter.
        
        Args:
            avg_frames: Number of frames to average FPS over.
        """
        self.fps_buffer = deque(maxlen=avg_frames)
        self.last_time = time.time()
    
    def update(self):
        """Update the FPS counter with the current frame."""
        current_time = time.time()
        self.fps_buffer.append(1 / (current_time - self.last_time))
        self.last_time = current_time
    
    def get_fps(self):
        """Get the current FPS."""
        return sum(self.fps_buffer) / len(self.fps_buffer) if self.fps_buffer else 0


def process_video_file(args):
    """
    Process a video file for zone time calculation.
    
    Args:
        args: Command-line arguments.
    """
    # Get video information
    width, height, fps, _ = get_video_info(args.source_video_path)
    
    # Load zone configuration
    polygons = load_zones_config(file_path=args.zone_configuration_path)
    zones = [
        sv.PolygonZone(
            polygon=polygon,
            triggering_anchors=(sv.Position.CENTER,),
        )
        for polygon in polygons
    ]
    
    # Initialize detector
    detector = ZoneDetector(
        model_path=args.weights,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
        device=args.device,
        classes=args.classes
    )
    
    # Initialize tracker
    tracker = ZoneTracker(zones=zones, fps=fps)
    
    # Initialize FPS counter
    fps_counter = FPSCounter()
    
    # Initialize annotator
    annotator = ZoneTimeAnnotator()
    
    # Initialize video writer if requested
    video_writer = None
    if args.save_video:
        video_writer = initialize_video_writer(
            args.output_video_path,
            width,
            height,
            fps
        )
    
    # Set up display window if showing video
    if args.display:
        setup_display_window(
            "Zone Time Tracking",
            args.display_width,
            args.display_height
        )
    
    # Process video frames
    frame_generator = get_frame_generator(args.source_video_path)
    print("Processing video... Press 'q' to quit")
    
    for frame in frame_generator:
        # Detect objects
        detections = detector.detect(frame)
        
        # Update tracker with all detections
        tracker.update(detections)
        
        # Update FPS counter
        fps_counter.update()
        
        # Create annotated frame
        annotated_frame = frame.copy()
        
        # Draw zones
        annotated_frame = annotator.annotate_zones(annotated_frame, zones)
        
        # Draw statistics
        stats = tracker.get_statistics()
        annotated_frame = annotator.annotate_statistics(annotated_frame, stats)
        
        # Draw detections and labels for each zone
        for idx in range(len(zones)):
            zone_detections = tracker.get_zone_detections(idx)
            if zone_detections is not None and len(zone_detections) > 0:
                labels = tracker.get_formatted_labels(idx)
                annotated_frame = annotator.annotate_detections(
                    annotated_frame,
                    zone_detections,
                    idx,
                    labels
                )
        
        # Write frame to output video if requested
        if args.save_video and video_writer:
            video_writer.write(annotated_frame)
        
        # Display frame if requested
        if args.display:
            cv2.imshow("Zone Time Tracking", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    
    # Clean up
    if args.save_video and video_writer:
        video_writer.release()
    
    if args.display:
        cv2.destroyAllWindows()


def process_rtsp_stream(args):
    """
    Process an RTSP stream for zone time calculation.
    
    Args:
        args: Command-line arguments.
    """
    from .utils.general import get_stream_frames_generator
    
    # Load zone configuration
    polygons = load_zones_config(file_path=args.zone_configuration_path)
    zones = [
        sv.PolygonZone(
            polygon=polygon,
            triggering_anchors=(sv.Position.CENTER,),
        )
        for polygon in polygons
    ]
    
    # Initialize detector
    detector = ZoneDetector(
        model_path=args.weights,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
        device=args.device,
        classes=args.classes
    )
    
    # Initialize tracker with clock-based timer for real-time streams
    tracker = ZoneTracker(zones=zones, use_clock_timer=True)
    
    # Initialize FPS counter
    fps_counter = FPSCounter()
    
    # Initialize annotator
    annotator = ZoneTimeAnnotator()
    
    # Process stream frames
    frame_generator = get_stream_frames_generator(args.rtsp_url)
    print("Processing stream... Press 'q' to quit")
    
    first_frame = True
    width, height = 0, 0
    video_writer = None
    
    for frame in frame_generator:
        # Get dimensions from first frame
        if first_frame:
            height, width = frame.shape[:2]
            
            # Set up display window if showing video
            if args.display:
                setup_display_window(
                    "Zone Time Tracking",
                    args.display_width,
                    args.display_height
                )
            
            # Initialize video writer if requested
            if args.save_video:
                video_writer = initialize_video_writer(
                    args.output_video_path,
                    width,
                    height,
                    30  # Assume 30 FPS for stream
                )
                
            first_frame = False
        
        # Detect objects
        detections = detector.detect(frame)
        
        # Update tracker with all detections
        tracker.update(detections)
        
        # Update FPS counter
        fps_counter.update()
        
        # Create annotated frame
        annotated_frame = frame.copy()
        
        # Draw zones
        annotated_frame = annotator.annotate_zones(annotated_frame, zones)
        
        # Draw statistics
        stats = tracker.get_statistics()
        annotated_frame = annotator.annotate_statistics(annotated_frame, stats)
        
        # Draw detections and labels for each zone
        for idx in range(len(zones)):
            zone_detections = tracker.get_zone_detections(idx)
            if zone_detections is not None and len(zone_detections) > 0:
                labels = tracker.get_formatted_labels(idx)
                annotated_frame = annotator.annotate_detections(
                    annotated_frame,
                    zone_detections,
                    idx,
                    labels
                )
        
        # Write frame to output video if requested
        if args.save_video and video_writer:
            video_writer.write(annotated_frame)
        
        # Display frame if requested
        if args.display:
            cv2.imshow("Zone Time Tracking", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    
    # Clean up
    if args.save_video and video_writer:
        video_writer.release()
    
    if args.display:
        cv2.destroyAllWindows()


def main():
    """
    Main entry point for the application.
    """
    args = parse_arguments()
    
    if args.rtsp_url:
        process_rtsp_stream(args)
    else:
        process_video_file(args)


if __name__ == "__main__":
    main()
