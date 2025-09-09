"""
Core detection engine for parking space occupancy detection

This module contains the main detection algorithms and processing logic.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import threading
import time
from abc import ABC, abstractmethod

from models.parking_space import ParkingSpace, SpaceStatus, DetectionParameters
from core.exceptions import DetectionException

class DetectionStrategy(ABC):
    """Abstract base class for detection strategies"""
    
    @abstractmethod
    def detect_occupancy(self, roi: np.ndarray, parameters: DetectionParameters) -> Tuple[bool, float]:
        """
        Detect occupancy in a region of interest
        
        Args:
            roi: Region of interest (cropped image)
            parameters: Detection parameters
            
        Returns:
            Tuple of (is_occupied, confidence_score)
        """
        pass

class EdgeDetectionStrategy(DetectionStrategy):
    """Edge-based detection strategy using Canny edge detection"""
    
    def detect_occupancy(self, roi: np.ndarray, parameters: DetectionParameters) -> Tuple[bool, float]:
        """
        Detect occupancy using edge detection
        
        Args:
            roi: Region of interest
            parameters: Detection parameters
            
        Returns:
            Tuple of (is_occupied, confidence_score)
        """
        if roi.size == 0:
            return False, 0.0
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            if parameters.blur_kernel > 1:
                kernel_size = parameters.blur_kernel if parameters.blur_kernel % 2 == 1 else parameters.blur_kernel + 1
                gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
            
            # Apply Canny edge detection
            edges = cv2.Canny(gray, parameters.threshold1, parameters.threshold2)
            
            # Apply morphological operations to clean up edges
            if parameters.morphology_kernel > 0:
                kernel = np.ones((parameters.morphology_kernel, parameters.morphology_kernel), np.uint8)
                edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # Count edge pixels
            edge_pixels = cv2.countNonZero(edges)
            total_pixels = roi.shape[0] * roi.shape[1]
            edge_ratio = edge_pixels / total_pixels if total_pixels > 0 else 0
            
            # Determine occupancy
            is_occupied = parameters.min_pixels < edge_pixels < parameters.max_pixels
            
            # Calculate confidence based on how well the edge count fits the expected range
            if is_occupied:
                mid_point = (parameters.min_pixels + parameters.max_pixels) / 2
                distance_from_mid = abs(edge_pixels - mid_point)
                max_distance = (parameters.max_pixels - parameters.min_pixels) / 2
                confidence = max(0.0, 1.0 - (distance_from_mid / max_distance))
            else:
                # Lower confidence when outside expected range
                if edge_pixels < parameters.min_pixels:
                    confidence = max(0.0, 1.0 - (parameters.min_pixels - edge_pixels) / parameters.min_pixels)
                else:
                    confidence = max(0.0, 1.0 - (edge_pixels - parameters.max_pixels) / parameters.max_pixels)
            
            return is_occupied, min(confidence, 1.0)
            
        except Exception as e:
            raise DetectionException(f"Edge detection failed: {e}")

class HybridDetectionStrategy(DetectionStrategy):
    """Hybrid detection combining multiple methods"""
    
    def detect_occupancy(self, roi: np.ndarray, parameters: DetectionParameters) -> Tuple[bool, float]:
        """
        Detect occupancy using hybrid approach (edges + color analysis)
        """
        if roi.size == 0:
            return False, 0.0
        
        try:
            # Edge detection component
            edge_strategy = EdgeDetectionStrategy()
            edge_occupied, edge_confidence = edge_strategy.detect_occupancy(roi, parameters)
            
            # Color variance component
            color_occupied, color_confidence = self._color_variance_detection(roi, parameters)
            
            # Combine results with weighted average
            edge_weight = 0.7
            color_weight = 0.3
            
            combined_confidence = (edge_confidence * edge_weight + color_confidence * color_weight)
            combined_occupied = combined_confidence > parameters.confidence_threshold
            
            return combined_occupied, combined_confidence
            
        except Exception as e:
            raise DetectionException(f"Hybrid detection failed: {e}")
    
    def _color_variance_detection(self, roi: np.ndarray, parameters: DetectionParameters) -> Tuple[bool, float]:
        """Color variance based detection"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Calculate standard deviation (measure of variance)
        std_dev = np.std(gray)
        
        # Higher variance typically indicates presence of objects
        # Normalize to 0-1 range (assuming max std dev of 100 for 8-bit images)
        variance_score = min(std_dev / 100.0, 1.0)
        
        # Consider occupied if variance is above threshold
        is_occupied = variance_score > 0.3
        
        return is_occupied, variance_score

class ParkingDetector:
    """
    Main parking detection engine
    """
    
    def __init__(self, strategy: DetectionStrategy = None):
        self.strategy = strategy or EdgeDetectionStrategy()
        self.detection_history = {}  # Space ID -> list of recent detections
        self.history_length = 5  # Number of recent detections to keep
        self._lock = threading.Lock()
    
    def set_strategy(self, strategy: DetectionStrategy) -> None:
        """Set the detection strategy"""
        self.strategy = strategy
    
    def detect_spaces(self, frame: np.ndarray, spaces: List[ParkingSpace], 
                     parameters: DetectionParameters) -> List[ParkingSpace]:
        """
        Detect occupancy for all parking spaces
        
        Args:
            frame: Input frame
            spaces: List of parking spaces to detect
            parameters: Detection parameters
            
        Returns:
            Updated list of parking spaces with detection results
        """
        if frame is None or len(frame.shape) != 3:
            raise DetectionException("Invalid input frame")
        
        updated_spaces = []
        
        with self._lock:
            for space in spaces:
                try:
                    # Extract region of interest
                    roi = self._extract_roi(frame, space)
                    
                    # Perform detection
                    is_occupied, confidence = self.strategy.detect_occupancy(roi, parameters)
                    
                    # Apply temporal smoothing
                    smoothed_occupied, smoothed_confidence = self._apply_temporal_smoothing(
                        space.id, is_occupied, confidence)
                    
                    # Update space status
                    updated_space = ParkingSpace(
                        id=space.id,
                        x=space.x,
                        y=space.y,
                        width=space.width,
                        height=space.height,
                        status=SpaceStatus.OCCUPIED if smoothed_occupied else SpaceStatus.VACANT,
                        confidence=smoothed_confidence
                    )
                    
                    updated_spaces.append(updated_space)
                    
                except Exception as e:
                    # If detection fails, keep previous status
                    space.confidence = 0.0
                    updated_spaces.append(space)
                    print(f"Detection failed for space {space.id}: {e}")
        
        return updated_spaces
    
    def _extract_roi(self, frame: np.ndarray, space: ParkingSpace) -> np.ndarray:
        """Extract region of interest from frame"""
        height, width = frame.shape[:2]
        
        # Ensure coordinates are within frame bounds
        x1 = max(0, min(space.x, width - 1))
        y1 = max(0, min(space.y, height - 1))
        x2 = max(x1 + 1, min(space.x + space.width, width))
        y2 = max(y1 + 1, min(space.y + space.height, height))
        
        roi = frame[y1:y2, x1:x2]
        return roi
    
    def _apply_temporal_smoothing(self, space_id: int, is_occupied: bool, 
                                confidence: float) -> Tuple[bool, float]:
        """
        Apply temporal smoothing to reduce flickering
        
        Args:
            space_id: ID of the parking space
            is_occupied: Current detection result
            confidence: Current confidence score
            
        Returns:
            Smoothed detection result and confidence
        """
        # Initialize history for new spaces
        if space_id not in self.detection_history:
            self.detection_history[space_id] = []
        
        # Add current detection to history
        history = self.detection_history[space_id]
        history.append((is_occupied, confidence))
        
        # Keep only recent detections
        if len(history) > self.history_length:
            history.pop(0)
        
        # Calculate smoothed result
        if len(history) == 1:
            return is_occupied, confidence
        
        # Use majority voting for occupancy
        occupied_votes = sum(1 for occupied, _ in history if occupied)
        smoothed_occupied = occupied_votes > len(history) // 2
        
        # Use weighted average for confidence (recent detections have higher weight)
        weights = [i + 1 for i in range(len(history))]
        weighted_confidence = sum(conf * weight for (_, conf), weight in zip(history, weights))
        total_weight = sum(weights)
        smoothed_confidence = weighted_confidence / total_weight if total_weight > 0 else confidence
        
        return smoothed_occupied, smoothed_confidence
    
    def clear_history(self) -> None:
        """Clear detection history"""
        with self._lock:
            self.detection_history.clear()
    
    def get_detection_stats(self) -> dict:
        """Get detection statistics"""
        with self._lock:
            stats = {
                'total_spaces_tracked': len(self.detection_history),
                'history_length': self.history_length,
                'strategy': self.strategy.__class__.__name__
            }
            return stats

class FrameProcessor:
    """
    Frame processing utilities for the detection system
    """
    
    @staticmethod
    def preprocess_frame(frame: np.ndarray, target_size: Tuple[int, int] = None) -> np.ndarray:
        """
        Preprocess frame for detection
        
        Args:
            frame: Input frame
            target_size: Optional target size (width, height)
            
        Returns:
            Preprocessed frame
        """
        if frame is None:
            return None
        
        processed = frame.copy()
        
        # Resize if target size is specified
        if target_size:
            processed = cv2.resize(processed, target_size)
        
        # Apply noise reduction
        processed = cv2.bilateralFilter(processed, 9, 75, 75)
        
        return processed
    
    @staticmethod
    def draw_spaces(frame: np.ndarray, spaces: List[ParkingSpace], 
                   show_ids: bool = True, show_confidence: bool = False) -> np.ndarray:
        """
        Draw parking spaces on frame
        
        Args:
            frame: Input frame
            spaces: List of parking spaces
            show_ids: Whether to show space IDs
            show_confidence: Whether to show confidence scores
            
        Returns:
            Frame with drawn spaces
        """
        if frame is None:
            return None
        
        result = frame.copy()
        
        # Colors
        VACANT_COLOR = (0, 255, 0)     # Green
        OCCUPIED_COLOR = (0, 0, 255)   # Red
        UNKNOWN_COLOR = (128, 128, 128)  # Gray
        
        for space in spaces:
            # Choose color based on status
            if space.status == SpaceStatus.VACANT:
                color = VACANT_COLOR
            elif space.status == SpaceStatus.OCCUPIED:
                color = OCCUPIED_COLOR
            else:
                color = UNKNOWN_COLOR
            
            # Draw rectangle
            thickness = 2 if space.confidence > 0.5 else 1
            cv2.rectangle(result, 
                         (space.x, space.y), 
                         (space.x + space.width, space.y + space.height), 
                         color, thickness)
            
            # Draw space ID
            if show_ids:
                text_pos = (space.x + 5, space.y + 20)
                cv2.putText(result, f"{space.id}", text_pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw confidence score
            if show_confidence and space.confidence > 0:
                conf_text = f"{space.confidence:.2f}"
                text_pos = (space.x + 5, space.y + space.height - 10)
                cv2.putText(result, conf_text, text_pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return result
    
    @staticmethod
    def draw_statistics(frame: np.ndarray, total: int, occupied: int, 
                       vacant: int, occupancy_rate: float) -> np.ndarray:
        """Draw statistics on frame"""
        if frame is None:
            return None
        
        result = frame.copy()
        
        # Statistics text
        stats_text = [
            f"Total Spaces: {total}",
            f"Occupied: {occupied}",
            f"Vacant: {vacant}",
            f"Occupancy: {occupancy_rate:.1f}%"
        ]
        
        # Background rectangle
        text_height = 25
        bg_height = len(stats_text) * text_height + 20
        cv2.rectangle(result, (10, 10), (250, bg_height), (0, 0, 0), -1)
        cv2.rectangle(result, (10, 10), (250, bg_height), (255, 255, 255), 2)
        
        # Draw text
        for i, text in enumerate(stats_text):
            y_pos = 35 + i * text_height
            cv2.putText(result, text, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result