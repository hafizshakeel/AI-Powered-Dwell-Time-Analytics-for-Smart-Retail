"""
Configuration settings for the zone time calculation module.
"""

# Default model settings
DEFAULT_MODEL_PATH = "yolov8n.pt"
DEFAULT_CONFIDENCE_THRESHOLD = 0.3
DEFAULT_IOU_THRESHOLD = 0.7
DEFAULT_DEVICE = "cpu"

# Default class IDs to track (empty list means track all classes)
DEFAULT_CLASSES = []

# Default person class ID in COCO dataset
PERSON_CLASS_ID = 0

# Visualization settings
COLORS = ["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"]
TEXT_COLOR = "#000000"
BACKGROUND_COLOR = "#FFFFFF"

# Default zone configuration file
DEFAULT_ZONE_CONFIG_PATH = "zones.json"

# FPS settings for video processing
DEFAULT_FPS = 30

# Display settings
DEFAULT_DISPLAY_WIDTH = 640
DEFAULT_DISPLAY_HEIGHT = 360
