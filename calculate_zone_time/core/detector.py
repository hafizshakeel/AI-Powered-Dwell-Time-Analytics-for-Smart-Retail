"""
Object detection module for zone time calculation.

This module provides a detector class that handles object detection using YOLOv8 models.
It supports different device types (CPU, CUDA) and allows filtering by class IDs.
"""
from typing import List, Optional, Dict, Any

import numpy as np
import torch
from ultralytics import YOLO

import supervision as sv
from ..config.settings import DEFAULT_MODEL_PATH, DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_IOU_THRESHOLD, PERSON_CLASS_ID
from ..utils.general import find_in_list


class ZoneDetector:
    """
    Object detector for zone time calculation.
    
    This class handles object detection using the YOLO model from Ultralytics.
    It provides methods to detect objects in frames and filter them by class.
    """
    
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        device: str = "cpu",
        classes: Optional[List[int]] = None
    ):
        """
        Initialize the ZoneDetector.
        
        Args:
            model_path: Path to the YOLO model weights.
            confidence_threshold: Confidence threshold for detections (0.0 to 1.0).
            iou_threshold: IoU threshold for non-max suppression (0.0 to 1.0).
            device: Device to run inference on ('cpu' or 'cuda').
            classes: List of class IDs to detect. If None, only people (class 0) are detected.
        """
        # Validate parameters
        if not (0 <= confidence_threshold <= 1):
            raise ValueError(f"Confidence threshold must be between 0 and 1, got {confidence_threshold}")
        if not (0 <= iou_threshold <= 1):
            raise ValueError(f"IoU threshold must be between 0 and 1, got {iou_threshold}")
            
        # Check device availability
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA is not available, falling back to CPU")
            device = "cpu"
            
        # Load model
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load model from {model_path}: {str(e)}")
            
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.classes = classes if classes is not None else [PERSON_CLASS_ID]
        
        # Cache model info
        self._model_info = {
            "model_path": model_path,
            "model_type": self.model.type,
            "model_task": self.model.task,
            "classes": self.model.names
        }
        
    def detect(self, frame: np.ndarray) -> sv.Detections:
        """
        Detect objects in a frame.
        
        Args:
            frame: Input frame as a numpy array (H, W, C).
            
        Returns:
            sv.Detections: Detected objects with bounding boxes, confidence scores, and class IDs.
        """
        # Validate frame
        if frame is None:
            return sv.Detections.empty()
            
        try:
            # Run inference
            results = self.model(
                frame,
                verbose=False,
                conf=self.confidence_threshold,
                device=self.device
            )[0]
            
            # Convert to Supervision Detections
            detections = sv.Detections.from_ultralytics(results)
            
            # Filter by class if specified
            if self.classes:
                detections = detections[find_in_list(detections.class_id, self.classes)]
                
            # Apply non-max suppression
            detections = detections.with_nms(threshold=self.iou_threshold)
            
            return detections
            
        except Exception as e:
            print(f"Detection error: {str(e)}")
            return sv.Detections.empty()
            
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dict[str, Any]: Dictionary containing model information.
        """
        return self._model_info.copy()
        
    def __str__(self) -> str:
        """
        Get a string representation of the detector.
        
        Returns:
            str: String representation.
        """
        return f"ZoneDetector(model={self._model_info['model_path']}, confidence={self.confidence_threshold}, device={self.device})"
