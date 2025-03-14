# 📊 Wait Time Analysis Streamlit Application

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30.0+-red.svg)](https://streamlit.io/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-brightgreen.svg)](https://github.com/ultralytics/ultralytics)
[![Supervision](https://img.shields.io/badge/Supervision-0.15.0+-orange.svg)](https://github.com/roboflow/supervision)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A powerful, user-friendly web application for analyzing wait times and zone occupancy in retail environments using computer vision and deep learning.


## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Installation](#-installation)
- [Usage Guide](#-usage-guide)
  - [Input Selection](#1-select-input-source)
  - [Configuration](#2-configure-detection-settings)
  - [Zone Drawing](#3-draw-zones)
  - [Video Processing](#4-process-video)
  - [Analytics Dashboard](#5-view-analytics)
- [Architecture](#-architecture)
- [Advanced Features](#-advanced-features)
- [Troubleshooting](#-troubleshooting)
- [Future Enhancements](#-future-enhancements)

## 🔭 Overview

The Wait Time Analysis Streamlit application provides a comprehensive, user-friendly interface for analyzing how long people spend in defined zones within retail environments. By leveraging state-of-the-art computer vision models (YOLOv8) and object tracking algorithms (ByteTrack), the application delivers accurate insights into customer behavior, queue management, and space utilization.

## ✨ Features

- 🎥 **Multiple Input Sources**: Upload videos, connect to RTSP streams, or use sample videos
- 🖌️ **Interactive Zone Drawing**: Intuitive interface for defining custom monitoring zones
- 🧠 **Configurable Detection**: Choose from different YOLOv8 models with adjustable parameters
- 🔄 **Real-time Processing**: Watch the analysis happen with visual feedback and progress tracking
- 📈 **Comprehensive Analytics**: Interactive dashboards with statistics, charts, and visualizations
- 📊 **Data Export**: Download results as CSV for further analysis in external tools
- ⚙️ **Advanced Settings**: Fine-tune detection confidence, alert thresholds, and display options
- 🚨 **Alert System**: Get notified when zones exceed occupancy thresholds

## 🔧 Installation

### Prerequisites

- Python 3.8+
- Git (for cloning the repository)

### Setup

```bash
# Clone the repository
git clone https://github.com/hafizshakeel/Real-Time-Dwell-Time-Analysis-for-Retail-Environments
cd Real-Time-Dwell-Time-Analysis-for-Retail-Environments

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# Using the convenience script
python run_streamlit.py

# Or directly with Streamlit
streamlit run streamlit_app/app.py
```

## 🚀 Usage Guide

### 1. Select Input Source

The application supports multiple input sources:

<table>
  <tr>
    <td width="33%"><b>Upload Video</b><br>Upload your own video files (MP4, AVI, MOV) for analysis.</td>
    <td width="33%"><b>RTSP Stream</b><br>Connect to live RTSP camera streams by providing the URL.</td>
    <td width="33%"><b>Sample Video</b><br>Use the included sample videos for testing and demonstration.</td>
  </tr>
</table>

### 2. Configure Detection Settings

In the sidebar, you can configure various detection and processing parameters:

| Setting | Description | Default |
|---------|-------------|---------|
| **Model** | YOLOv8 model variant (nano to extra large) | `yolov8n.pt` |
| **Confidence Threshold** | Minimum detection confidence (0.1-1.0) | `0.3` |
| **Alert Threshold** | Number of people to trigger an alert | `3` |
| **Alert Duration Threshold** | Minimum duration for an alert (seconds) | `5` |
| **Display Frequency** | How often frames are displayed during processing | `5` |
| **Save Results to CSV** | Enable to save processing results | `False` |

### 3. Draw Zones

The application provides an intuitive interface for defining monitoring zones:

<table>
  <tr>
    <td width="50%">
      <b>Drawing Controls:</b><br>
      • Click on the video frame to add points<br>
      • Click "Add Zone" to save the current polygon<br>
      • Click "New Zone" to start drawing another zone<br>
      • Click "Clear All Zones" to start over<br>
      • Click "Save Zones and Continue" when finished
    </td>
    <td width="50%">
      <b>Tips:</b><br>
      • Create closed polygons for accurate results<br>
      • Each zone is color-coded for easy identification<br>
      • You can create multiple zones to monitor different areas<br>
      • Zones can be any shape (not just rectangles)<br>
      • Avoid overlapping zones for clearer analytics
    </td>
  </tr>
</table>

### 4. Process Video

After configuring zones, click "Start Processing" to begin the analysis:

- **Real-time Visualization**: Watch as people are detected and tracked through zones
- **Progress Tracking**: Monitor completion percentage with a progress bar
- **Live Metrics**: View total people count, average time, and processing speed (FPS)
- **Visual Feedback**: See bounding boxes, zone boundaries, and time labels

### 5. View Analytics

After processing completes, explore the comprehensive analytics dashboard:

#### Overview Tab
- Key metrics (total people, zones, average and max time)
- Average time spent in each zone (bar chart)
- Distribution of people across zones (pie chart)

#### Time Analysis Tab
- Distribution of time spent in zones (histogram)
- Average time per person (scatter plot)
- Time trends over the video duration (line chart)

#### Zone Occupancy Tab
- Number of people in each zone over time (line chart)
- Zone occupancy heatmap (color-coded by occupancy level)
- Peak occupancy times and patterns

#### Detailed Data Tab
- Comprehensive statistics for each zone
- Raw data explorer with filtering and sorting
- Download options for further analysis

## 🏗️ Architecture

The application follows a modular architecture for maintainability and extensibility:

```
streamlit_app/
├── app.py                  # Main Streamlit application
├── processing/             # Video and RTSP processing modules
│   ├── video_processor.py  # Handles video file processing
│   └── rtsp_processor.py   # Handles RTSP stream processing
├── analytics/              # Data visualization components
│   └── dashboard.py        # Analytics dashboard implementation
├── alerts/                 # Alert system components
│   └── alert_system.py     # Alert detection and management
├── utils/                  # Helper utilities
│   ├── file_utils.py       # File handling utilities
│   └── zone_utils.py       # Zone drawing and management
└── README.md               # Documentation
```

The application integrates with the `calculate_zone_time` module for core detection, tracking, and time calculation functionality.

## 🔍 Advanced Features

### Alert System

The application includes a sophisticated alert system that:
- Monitors zone occupancy in real-time
- Triggers alerts when occupancy exceeds thresholds
- Tracks alert duration and severity
- Provides visual notifications during processing
- Maintains an alert history for review

### Custom Model Support

You can use your own custom-trained YOLOv8 models:
1. Place your model file (e.g., `custom_model.pt`) in the application directory
2. Select "Custom" from the model dropdown
3. Enter the path to your model file

### Performance Optimization

For better performance on resource-constrained systems:
- Use smaller YOLOv8 models (nano or small)
- Increase the display frequency value to reduce rendering overhead
- Process shorter video clips or lower resolution videos
- Disable CSV saving if not needed

## 🔧 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **Application crashes with large videos** | Try processing a shorter clip or using a smaller model |
| **No objects detected** | Adjust the confidence threshold or try a different model |
| **Zone drawing not working** | Click "New Zone" to reset the drawing mode |
| **RTSP connection fails** | Verify the URL is correct and the stream is active |
| **Slow processing speed** | Increase display frequency or use a smaller model |
| **Memory errors** | Close other applications or restart with more memory |

### Getting Help

If you encounter issues not covered here:
1. Check the console output for error messages
2. Verify all dependencies are correctly installed
3. Try restarting the application
4. Check for updates to the application or dependencies

