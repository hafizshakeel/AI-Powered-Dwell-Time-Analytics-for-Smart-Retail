"""
Command-line interface utilities for zone time calculation.
"""
import argparse
from typing import List, Optional

from ..config.settings import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_DEVICE,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_MODEL_PATH,
    DEFAULT_ZONE_CONFIG_PATH,
    DEFAULT_DISPLAY_WIDTH,
    DEFAULT_DISPLAY_HEIGHT,
)


def parse_arguments():
    """
    Parse command-line arguments for the zone time calculation application.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Calculate and track time spent by objects in defined zones."
    )
    
    # Input source arguments
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--source_video_path",
        type=str,
        help="Path to the source video file.",
    )
    source_group.add_argument(
        "--rtsp_url",
        type=str,
        help="Complete RTSP URL for the video stream.",
    )
    
    # Zone configuration
    parser.add_argument(
        "--zone_configuration_path",
        type=str,
        default=DEFAULT_ZONE_CONFIG_PATH,
        help=f"Path to the zone configuration JSON file. Default is '{DEFAULT_ZONE_CONFIG_PATH}'.",
    )
    
    # Model settings
    parser.add_argument(
        "--weights",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the model weights file. Default is '{DEFAULT_MODEL_PATH}'.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help=f"Computation device ('cpu', 'mps' or 'cuda'). Default is '{DEFAULT_DEVICE}'.",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help=f"Confidence level for detections (0 to 1). Default is {DEFAULT_CONFIDENCE_THRESHOLD}.",
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=DEFAULT_IOU_THRESHOLD,
        help=f"IOU threshold for non-max suppression. Default is {DEFAULT_IOU_THRESHOLD}.",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        type=int,
        default=[],
        help="List of class IDs to track. If empty, all classes are tracked.",
    )
    
    # Display options
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display the processed video in a window.",
    )
    parser.add_argument(
        "--display_width",
        type=int,
        default=DEFAULT_DISPLAY_WIDTH,
        help=f"Width of the display window. Default is {DEFAULT_DISPLAY_WIDTH}.",
    )
    parser.add_argument(
        "--display_height",
        type=int,
        default=DEFAULT_DISPLAY_HEIGHT,
        help=f"Height of the display window. Default is {DEFAULT_DISPLAY_HEIGHT}.",
    )
    
    # Output options
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="Save the processed video to a file.",
    )
    parser.add_argument(
        "--output_video_path",
        type=str,
        default="output.mp4",
        help="Path to save the processed video. Default is 'output.mp4'.",
    )
    
    return parser.parse_args()
