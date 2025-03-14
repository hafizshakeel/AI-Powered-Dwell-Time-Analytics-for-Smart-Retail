"""
Video processing utilities for zone time calculation.
"""
from typing import Generator, Tuple

import cv2
import numpy as np
import supervision as sv


def get_video_info(video_path: str) -> Tuple[int, int, int, int]:
    """
    Get information about a video file.

    Args:
        video_path (str): Path to the video file.

    Returns:
        Tuple[int, int, int, int]: Width, height, fps, and total frames of the video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Error: Could not open video file at {video_path}.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()
    return width, height, fps, total_frames


def get_frame_generator(video_path: str) -> Generator[np.ndarray, None, None]:
    """
    Create a generator that yields frames from a video file.

    Args:
        video_path (str): Path to the video file.

    Yields:
        np.ndarray: The next frame from the video file.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Error: Could not open video file at {video_path}.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()


def setup_display_window(window_name: str, width: int, height: int) -> None:
    """
    Set up a display window for showing processed video frames.

    Args:
        window_name (str): Name of the window.
        width (int): Width of the window.
        height (int): Height of the window.
    """
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)


def resize_frame_for_display(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Resize a frame for display purposes.

    Args:
        frame (np.ndarray): The input frame.
        width (int): Target width.
        height (int): Target height.

    Returns:
        np.ndarray: Resized frame.
    """
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def initialize_video_writer(
    output_path: str, width: int, height: int, fps: int
) -> cv2.VideoWriter:
    """
    Initialize a video writer for saving processed frames.

    Args:
        output_path (str): Path to save the output video.
        width (int): Width of the output video.
        height (int): Height of the output video.
        fps (int): Frames per second of the output video.

    Returns:
        cv2.VideoWriter: Initialized video writer object.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))
