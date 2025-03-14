#!/usr/bin/env python
"""
Script to stream video files as RTSP streams for testing purposes.
"""
import argparse
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2


def get_video_files(directory: str) -> List[str]:
    """Get all video files in a directory.
    
    Args:
        directory: Directory to search for video files.
    
    Returns:
        List of video file paths.
    """
    video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
    video_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(root, file))
    
    return video_files


def check_ffmpeg() -> bool:
    """Check if FFmpeg is installed.
    
    Returns:
        True if FFmpeg is installed, False otherwise.
    """
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def check_rtsp_server() -> bool:
    """Check if the RTSP server is running.
    
    Returns:
        True if the RTSP server is running, False otherwise.
    """
    try:
        # Try to connect to the RTSP server
        import socket
        
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        result = s.connect_ex(("localhost", 8554))
        s.close()
        
        return result == 0
    except Exception:
        return False


def start_rtsp_server() -> Optional[subprocess.Popen]:
    """Start the RTSP server using Docker.
    
    Returns:
        Subprocess object if the server was started, None otherwise.
    """
    print("Starting RTSP server...")
    
    # Check if Docker is installed
    try:
        subprocess.run(
            ["docker", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Error: Docker is not installed or not in PATH.")
        return None
    
    # Start the RTSP server
    try:
        # Create a simple configuration file
        config_path = os.path.join(os.getcwd(), "rtsp-simple-server.yml")
        with open(config_path, "w") as f:
            f.write("protocols: [tcp]\n")
            f.write("paths:\n")
            f.write("  all:\n")
            f.write("    source: publisher\n")
        
        # Start the server
        process = subprocess.Popen(
            [
                "docker",
                "run",
                "--rm",
                "-p",
                "8554:8554",
                "-v",
                f"{config_path}:/rtsp-simple-server.yml",
                "aler9/rtsp-simple-server:v1.3.0",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        # Wait for the server to start
        time.sleep(2)
        
        # Check if the server is running
        if not check_rtsp_server():
            print("Error: Failed to start RTSP server.")
            process.terminate()
            return None
        
        print("RTSP server started successfully.")
        return process
    except Exception as e:
        print(f"Error starting RTSP server: {e}")
        return None


def stream_video(video_path: str, stream_name: str) -> Optional[subprocess.Popen]:
    """Stream a video file to the RTSP server.
    
    Args:
        video_path: Path to the video file.
        stream_name: Name of the RTSP stream.
    
    Returns:
        Subprocess object if the stream was started, None otherwise.
    """
    if not check_ffmpeg():
        print("Error: FFmpeg is not installed or not in PATH.")
        return None
    
    # Get video information
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return None
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    # Start streaming
    print(f"Streaming {video_path} to rtsp://localhost:8554/{stream_name}")
    
    # Build FFmpeg command
    command = [
        "ffmpeg",
        "-re",  # Read input at native frame rate
        "-i", video_path,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-b:v", "1000k",
        "-f", "rtsp",
        f"rtsp://localhost:8554/{stream_name}",
    ]
    
    # Start FFmpeg
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return process
    except Exception as e:
        print(f"Error starting FFmpeg: {e}")
        return None


def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Stream video files as RTSP streams")
    parser.add_argument(
        "--video_directory",
        type=str,
        required=True,
        help="Directory containing video files",
    )
    parser.add_argument(
        "--number_of_streams",
        type=int,
        default=1,
        help="Number of streams to create",
    )
    args = parser.parse_args()
    
    # Check if the directory exists
    if not os.path.isdir(args.video_directory):
        print(f"Error: Directory not found: {args.video_directory}")
        return
    
    # Get video files
    video_files = get_video_files(args.video_directory)
    if not video_files:
        print(f"Error: No video files found in {args.video_directory}")
        return
    
    # Limit the number of streams
    num_streams = min(args.number_of_streams, len(video_files))
    
    # Start RTSP server if not already running
    server_process = None
    if not check_rtsp_server():
        server_process = start_rtsp_server()
        if server_process is None:
            return
    
    # Start streaming
    stream_processes = []
    try:
        for i in range(num_streams):
            video_path = video_files[i % len(video_files)]
            stream_name = f"live{i}.stream"
            
            process = stream_video(video_path, stream_name)
            if process is not None:
                stream_processes.append(process)
                print(f"Stream {i+1}/{num_streams} started: rtsp://localhost:8554/{stream_name}")
        
        # Keep the streams running
        print("\nStreams are now running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\nStopping streams...")
    
    finally:
        # Stop all streams
        for process in stream_processes:
            process.terminate()
        
        # Stop the server if we started it
        if server_process is not None:
            server_process.terminate()
        
        print("All streams stopped.")


if __name__ == "__main__":
    main()
