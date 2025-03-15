# 🕒 Zone Time Calculation Module

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-brightgreen.svg)](https://github.com/ultralytics/ultralytics)
[![Supervision](https://img.shields.io/badge/Supervision-0.15.0+-orange.svg)](https://github.com/roboflow/supervision)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A robust, modular system for tracking and calculating how long people spend in defined zones within videos or live streams.


## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
  - [Processing a Video File](#processing-a-video-file)
  - [Processing a Live Stream](#processing-a-live-stream)
  - [Common Options](#common-options)
- [Zone Configuration](#-zone-configuration)
  - [Drawing Zones](#drawing-zones)
- [Visualization](#-visualization)
- [RTSP Streaming for Testing](#-rtsp-streaming-for-testing)
- [Troubleshooting](#-troubleshooting)

## 🔭 Overview

The Zone Time Calculation module provides a comprehensive solution for tracking people and measuring the time they spend in predefined zones. This is particularly useful for retail analytics, queue management, and space utilization analysis. The system uses YOLOv8 for object detection and ByteTrack for object tracking, providing accurate and reliable results even in crowded scenes.

## ✨ Features

- 🎯 **Accurate Detection**: Uses YOLOv8 models for reliable person detection
- 🔄 **Robust Tracking**: Implements ByteTrack algorithm for consistent object tracking
- ⏱️ **Time Measurement**: Calculates precise time spent by each person in defined zones
- 🖥️ **Real-time Processing**: Supports both video files and live RTSP streams
- 🎨 **Customizable Visualization**: Color-coded zones with time labels and statistics
- 📊 **Data Export**: Save results to CSV for further analysis
- 🛠️ **Configurable**: Easily adjust detection parameters and tracking settings
- 🔍 **Interactive Zone Definition**: User-friendly tools for defining monitoring zones

## 🗂️ Project Structure

```
calculate_zone_time/
├── core/                     # Core detection and tracking components
│   ├── detector.py           # YOLOv8 object detection implementation
│   └── tracker.py            # ByteTrack object tracking and time calculation
├── visualization/            # Visualization components
│   └── annotator.py          # Frame annotation with zones, detections, and statistics
├── utils/                    # Helper utilities
│   ├── cli.py                # Command-line interface utilities
│   ├── general.py            # General helper functions
│   ├── timers.py             # Time tracking implementations
│   └── video.py              # Video processing utilities
├── config/                   # Configuration settings
│   └── settings.py           # Default parameters and constants
├── scripts/                  # Utility scripts
│   ├── draw_zones_cli.py     # Command-line zone drawing tool
│   ├── draw_zones_ui.py      # UI-based zone drawing tool
│   └── stream_from_file.py   # RTSP streaming utility
└── __main__.py               # Main entry point for the application
```

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/hafizshakeel/AI-Powered-Dwell-Time-Analytics-for-Smart-Retail.git
cd zone-time-calculation

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .

# Install additional dependencies
pip install ultralytics opencv-python-headless
```

## 🚀 Usage

### Processing a Video File

To analyze a video file and calculate zone times:

```bash
python -m calculate_zone_time --source_video_path videos/retail_store.mp4 --zone_configuration_path zones/store_zones.json --display
```

### Processing a Live Stream

To analyze a live RTSP stream:

```bash
python -m calculate_zone_time --rtsp_url rtsp://camera.example.com/stream --zone_configuration_path zones/store_zones.json --display
```

### Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `--weights` | YOLOv8 model to use | `yolov8n.pt` |
| `--device` | Device for inference (cpu/cuda) | `cpu` |
| `--confidence_threshold` | Detection confidence threshold | `0.3` |
| `--classes` | Classes to track (e.g., 0 for people) | `[0]` |
| `--display` | Show video while processing | `False` |
| `--save_video` | Save processed video | `False` |
| `--output_video_path` | Path for saving video | `output.mp4` |
| `--save_csv` | Save results to CSV file | `False` |
| `--output_csv_path` | Path for saving CSV results | `results.csv` |



### Drawing Zones

The module includes interactive tools for defining zones:

#### Command-line Interface

```bash
python -m calculate_zone_time.scripts.draw_zones_cli --source_path videos/retail_store.mp4 --zone_configuration_path zones/store_zones.json
```

Controls:
- Left-click to add points
- Enter to finish current zone and start a new one
- Escape to clear current zone
- 's' to save zones
- 'q' to quit

#### UI Interface

```bash
python -m calculate_zone_time.scripts.draw_zones_ui --source_path videos/retail_store.mp4 --zone_configuration_path zones/store_zones.json
```

The UI interface via Streamlit provides a more user-friendly experience with buttons for adding, editing, and deleting zones.

## 📊 Visualization

The processed video display includes:

- **Zone Boundaries**: Color-coded polygons showing each defined zone
- **Person Detection**: Bounding boxes around detected people
- **Zone Occupancy**: Number of people in each zone (color-coded to match the zone)
- **Time Labels**: For each person in a zone, a label shows their ID and time spent (MM:SS format)
- **FPS Counter**: Current processing speed in frames per second
- **Statistics**: Total people count, average time spent, and other metrics

## 🎥 RTSP Streaming for Testing

For testing purposes, you can create RTSP streams from local video files:

### Prerequisites

1. **Docker**: You need to have Docker installed on your system.
2. **FFmpeg**: Required for video streaming.

### Starting RTSP Streams

```bash
python -m calculate_zone_time.scripts.stream_from_file --video_directory videos/ --number_of_streams 1
```

This will:
1. Start an RTSP server using Docker
2. Stream the video files in a loop
3. Create RTSP URLs like `rtsp://localhost:8554/live0.stream`, `rtsp://localhost:8554/live1.stream`, etc.

### Using the RTSP Streams

```bash
python -m calculate_zone_time --rtsp_url rtsp://localhost:8554/live0.stream --zone_configuration_path zones/store_zones.json --display
```

### Stopping the RTSP Server

```bash
docker kill rtsp_server
```

## 🔍 Troubleshooting

### RTSP Streaming Issues

#### "404 Not Found" or "No one is publishing to path" Errors

This usually means the FFmpeg process isn't successfully publishing to the RTSP server.

**Solution:**
1. Check if the Docker container is running: `docker ps`
2. Check if FFmpeg is running: `tasklist | findstr ffmpeg`
3. Manually start FFmpeg to stream the video:
   ```bash
   ffmpeg -re -i <video_path> -c:v libx264 -preset ultrafast -tune zerolatency -b:v 1000k -f rtsp rtsp://localhost:8554/live0.stream
   ```

#### VLC Cannot Open the Stream

If VLC shows "unable to open the MRL" error:
1. Make sure the RTSP server is running
2. Make sure FFmpeg is actively streaming to the server
3. Try accessing the stream with the main application first

### General Troubleshooting Steps

1. Restart the Docker container
2. Restart the FFmpeg process
3. Check Docker logs for errors: `docker logs rtsp_server`
4. Ensure your firewall isn't blocking the RTSP port (8554)



### Custom Models

You can use custom YOLOv8 models trained on your specific data:

```bash
python -m calculate_zone_time --source_video_path videos/retail_store.mp4 --zone_configuration_path zones/store_zones.json --weights path/to/custom_model.pt --display
```

### Performance Optimization

For better performance on resource-constrained systems:

1. Use smaller YOLOv8 models (yolov8n.pt)
2. Reduce input resolution: `--input_size 320`
3. Process every Nth frame: `--process_every_n_frames 2`
4. Disable visualization if not needed: omit `--display`

