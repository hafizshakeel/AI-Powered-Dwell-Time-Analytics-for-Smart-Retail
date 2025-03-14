"""
FPS counter utility for tracking frames per second.
"""
import time
from collections import deque


class FPSCounter:
    """Tracks and calculates frames per second."""
    
    def __init__(self, avg_frames=30):
        """
        Initialize the FPS counter.
        
        Args:
            avg_frames: Number of frames to average for FPS calculation
        """
        self.prev_time = time.time()
        self.frame_times = deque(maxlen=avg_frames)
    
    def update(self):
        """Update the FPS counter with a new frame."""
        curr_time = time.time()
        self.frame_times.append(curr_time - self.prev_time)
        self.prev_time = curr_time
    
    def get_fps(self):
        """Get the current FPS."""
        if not self.frame_times:
            return 0
        return 1.0 / (sum(self.frame_times) / len(self.frame_times)) 