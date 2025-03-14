"""
Object tracking module for zone time calculation.

This module provides a tracker class that handles object tracking and time calculation
for objects in predefined zones. It supports both FPS-based and clock-based timing.
"""
from typing import Dict, List, Optional

import numpy as np
import supervision as sv

from ..utils.timers import FPSBasedTimer, ClockBasedTimer


class ZoneTracker:
    """
    Object tracker for zone time calculation.
    
    This class handles object tracking and time calculation for objects in zones.
    It uses ByteTrack for object tracking and supports both FPS-based and clock-based
    timing for calculating the time objects spend in zones.
    """
    
    def __init__(
        self,
        zones: List[sv.PolygonZone],
        fps: Optional[int] = None,
        use_clock_timer: bool = False,
        tracker_kwargs: Optional[Dict] = None
    ):
        """
        Initialize the ZoneTracker.
        
        Args:
            zones: List of polygon zones to track objects in.
            fps: Frames per second of the video. Required if use_clock_timer is False.
            use_clock_timer: Whether to use clock-based timing instead of FPS-based.
            tracker_kwargs: Additional keyword arguments for the ByteTrack tracker.
        """
        if not zones:
            raise ValueError("At least one zone must be provided")
            
        self.zones = zones
        
        # Initialize tracker
        tracker_params = tracker_kwargs or {"minimum_matching_threshold": 0.5}
        self.tracker = sv.ByteTrack(**tracker_params)
        
        # Initialize timers for each zone
        self.timers = []
        for _ in zones:
            if use_clock_timer:
                self.timers.append(ClockBasedTimer())
            else:
                if fps is None:
                    raise ValueError("FPS must be provided when using FPS-based timer")
                self.timers.append(FPSBasedTimer(fps=fps))
                
        # Initialize storage for detections and times
        self.zone_detections = [None] * len(zones)
        self.zone_times = [None] * len(zones)
        
        # Initialize statistics
        self._stats = {
            "total_tracked_objects": 0,
            "max_time_in_zone": [0] * len(zones),
            "avg_time_in_zone": [0] * len(zones),
            "current_objects_in_zone": [0] * len(zones)
        }
        
    def update(self, detections: sv.Detections) -> None:
        """
        Update tracking and time calculations for all zones.
        
        Args:
            detections: Detections from the current frame.
        """
        if detections is None or len(detections) == 0:
            return
            
        # Find detections in any zone
        in_any_zone = np.zeros(len(detections), dtype=bool)
        for zone in self.zones:
            in_any_zone = np.logical_or(in_any_zone, zone.trigger(detections))
        
        # Only track detections that are in zones
        zone_detections = detections[in_any_zone]
        if len(zone_detections) == 0:
            return
            
        # Update tracking
        tracked_detections = self.tracker.update_with_detections(zone_detections)
        if len(tracked_detections) == 0:
            return
            
        # Update total tracked objects count
        self._stats["total_tracked_objects"] = len(np.unique(tracked_detections.tracker_id))
        
        # Process each zone
        for idx, zone in enumerate(self.zones):
            # Get detections in this zone
            detections_in_zone = tracked_detections[zone.trigger(tracked_detections)]
            
            # Update time for detections in this zone
            time_in_zone = self.timers[idx].tick(detections_in_zone)
            
            # Store results
            self.zone_detections[idx] = detections_in_zone
            self.zone_times[idx] = time_in_zone
            
            # Update statistics
            self._stats["current_objects_in_zone"][idx] = len(detections_in_zone)
            if len(detections_in_zone) > 0 and len(time_in_zone) > 0:
                self._stats["max_time_in_zone"][idx] = max(
                    self._stats["max_time_in_zone"][idx],
                    np.max(time_in_zone)
                )
                self._stats["avg_time_in_zone"][idx] = np.mean(time_in_zone)
            
    def get_zone_detections(self, zone_idx: int) -> Optional[sv.Detections]:
        """
        Get detections in a specific zone.
        
        Args:
            zone_idx: Index of the zone.
            
        Returns:
            sv.Detections: Detections in the specified zone, or None if no detections.
        """
        self._check_zone_index(zone_idx)
        return self.zone_detections[zone_idx]
    
    def get_zone_times(self, zone_idx: int) -> Optional[np.ndarray]:
        """
        Get time spent by objects in a specific zone.
        
        Args:
            zone_idx: Index of the zone.
            
        Returns:
            np.ndarray: Time spent by each object in the zone, or None if no objects.
        """
        self._check_zone_index(zone_idx)
        return self.zone_times[zone_idx]
    
    def get_formatted_labels(self, zone_idx: int) -> List[str]:
        """
        Get formatted time labels for objects in a zone.
        
        Args:
            zone_idx: Index of the zone.
            
        Returns:
            List[str]: Formatted time labels for each object in the zone.
        """
        self._check_zone_index(zone_idx)
        
        if self.zone_detections[zone_idx] is None or len(self.zone_detections[zone_idx]) == 0:
            return []
            
        return [
            f"#{tracker_id} {int(time // 60):02d}:{int(time % 60):02d}"
            for tracker_id, time in zip(
                self.zone_detections[zone_idx].tracker_id,
                self.zone_times[zone_idx]
            )
        ]
        
    def get_all_zone_detections(self) -> sv.Detections:
        """
        Get all detections from all zones combined.
        
        Returns:
            sv.Detections: Combined detections from all zones.
        """
        valid_detections = [d for d in self.zone_detections if d is not None and len(d) > 0]
        
        if not valid_detections:
            return sv.Detections.empty()
            
        return sv.Detections.merge(valid_detections)
        
    def get_statistics(self) -> Dict:
        """
        Get tracking statistics.
        
        Returns:
            Dict: Dictionary containing tracking statistics.
        """
        return self._stats.copy()
        
    def reset_statistics(self) -> None:
        """
        Reset tracking statistics.
        """
        self._stats = {
            "total_tracked_objects": 0,
            "max_time_in_zone": [0] * len(self.zones),
            "avg_time_in_zone": [0] * len(self.zones),
            "current_objects_in_zone": [0] * len(self.zones)
        }
    
    def _check_zone_index(self, zone_idx: int) -> None:
        """
        Check if zone index is valid.
        
        Args:
            zone_idx: Index of the zone to check.
            
        Raises:
            IndexError: If zone_idx is out of range.
        """
        if not 0 <= zone_idx < len(self.zones):
            raise IndexError(f"Zone index {zone_idx} out of range (0-{len(self.zones)-1})")
        
    def __str__(self) -> str:
        """
        Get a string representation of the tracker.
        
        Returns:
            str: String representation.
        """
        return f"ZoneTracker(zones={len(self.zones)}, tracked_objects={self._stats['total_tracked_objects']})"
