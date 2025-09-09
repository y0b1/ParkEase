"""
Camera management system for parking detection

This module handles all camera-related operations including initialization,
frame capture, and camera settings management.
"""

import cv2
import numpy as np
import threading
import time
from typing import Optional, Tuple, Callable
from dataclasses import dataclass

from core.exceptions import CameraException

@dataclass
class CameraSettings:
    """Camera configuration settings"""
    width: int = 1280
    height: int = 720
    fps: int = 30
    auto_exposure: bool = True
    brightness: int = 50
    contrast: int = 50
    saturation: int = 50

class CameraManager:
    """
    Manages camera operations for the parking detection system
    """
    
    def __init__(self, source: int = 0):
        self.source = source
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.capture_thread = None
        self.frame_callback = None
        self.settings = CameraSettings()
        
    def initialize_camera(self) -> bool:
        """
        Initialize the camera
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.source)
            
            if not self.cap.isOpened():
                raise CameraException(f"Could not open camera source: {self.source}")
            
            # Apply initial settings
            self._apply_settings()
            
            # Test frame capture
            ret, frame = self.cap.read()
            if not ret:
                raise CameraException("Could not capture initial frame")
            
            with self.frame_lock:
                self.current_frame = frame
            
            return True
            
        except Exception as e:
            raise CameraException(f"Camera initialization failed: {e}")
    
    def _apply_settings(self) -> None:
        """Apply camera settings"""
        if self.cap is None:
            return
        
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.settings.fps)
            
            # Set exposure and other properties if supported
            if not self.settings.auto_exposure:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual exposure
            else:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Auto exposure
            
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.settings.brightness / 100.0)
            self.cap.set(cv2.CAP_PROP_CONTRAST, self.settings.contrast / 100.0)
            self.cap.set(cv2.CAP_PROP_SATURATION, self.settings.saturation / 100.0)
            
        except Exception as e:
            print(f"Warning: Could not apply some camera settings: {e}")
    
    def start_capture(self, frame_callback: Callable[[np.ndarray], None] = None) -> bool:
        """
        Start continuous frame capture
        
        Args:
            frame_callback: Optional callback function for new frames
            
        Returns:
            True if successful, False otherwise
        """
        if self.cap is None:
            raise CameraException("Camera not initialized")
        
        if self.is_running:
            return True
        
        self.frame_callback = frame_callback
        self.is_running = True
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        return True
    
    def stop_capture(self) -> None:
        """Stop frame capture"""
        self.is_running = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
    
    def _capture_loop(self) -> None:
        """Main capture loop running in separate thread"""
        frame_time = 1.0 / self.settings.fps
        
        while self.is_running:
            start_time = time.time()
            
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Warning: Failed to capture frame")
                    continue
                
                # Update current frame
                with self.frame_lock:
                    self.current_frame = frame.copy()
                
                # Call callback if provided
                if self.frame_callback:
                    try:
                        self.frame_callback(frame)
                    except Exception as e:
                        print(f"Frame callback error: {e}")
                
                # Maintain frame rate
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_time - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                print(f"Capture loop error: {e}")
                break
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """
        Get the current frame
        
        Returns:
            Current frame or None if not available
        """
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def capture_single_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame
        
        Returns:
            Captured frame or None if failed
        """
        if self.cap is None:
            raise CameraException("Camera not initialized")
        
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def update_settings(self, settings: CameraSettings) -> bool:
        """
        Update camera settings
        
        Args:
            settings: New camera settings
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.settings = settings
            if self.cap:
                self._apply_settings()
            return True
        except Exception as e:
            print(f"Failed to update camera settings: {e}")
            return False
    
    def get_camera_info(self) -> dict:
        """
        Get camera information
        
        Returns:
            Dictionary with camera properties
        """
        if self.cap is None:
            return {}
        
        try:
            info = {
                'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': self.cap.get(cv2.CAP_PROP_FPS),
                'brightness': self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
                'contrast': self.cap.get(cv2.CAP_PROP_CONTRAST),
                'saturation': self.cap.get(cv2.CAP_PROP_SATURATION),
                'auto_exposure': bool(self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)),
                'backend': self.cap.getBackendName() if hasattr(self.cap, 'getBackendName') else 'Unknown'
            }
            return info
        except Exception as e:
            print(f"Failed to get camera info: {e}")
            return {}
    
    def is_camera_available(self) -> bool:
        """Check if camera is available and working"""
        return self.cap is not None and self.cap.isOpened()
    
    def release(self) -> None:
        """Release camera resources"""
        self.stop_capture()
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        with self.frame_lock:
            self.current_frame = None

class VideoFileManager(CameraManager):
    """
    Video file manager for testing with recorded videos
    """
    
    def __init__(self, video_path: str):
        super().__init__(video_path)
        self.video_path = video_path
        self.total_frames = 0
        self.current_frame_number = 0
        self.is_looping = True
    
    def initialize_camera(self) -> bool:
        """Initialize video file"""
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            
            if not self.cap.isOpened():
                raise CameraException(f"Could not open video file: {self.video_path}")
            
            # Get video properties
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.settings.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.settings.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.settings.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Test frame capture
            ret, frame = self.cap.read()
            if not ret:
                raise CameraException("Could not read first frame from video")
            
            # Reset to beginning
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            with self.frame_lock:
                self.current_frame = frame
            
            return True
            
        except Exception as e:
            raise CameraException(f"Video file initialization failed: {e}")
    
    def _capture_loop(self) -> None:
        """Video capture loop with looping support"""
        frame_time = 1.0 / self.settings.fps
        
        while self.is_running:
            start_time = time.time()
            
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    if self.is_looping and self.total_frames > 0:
                        # Loop back to beginning
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.current_frame_number = 0
                        ret, frame = self.cap.read()
                    
                    if not ret:
                        print("Warning: Failed to read video frame")
                        continue
                
                self.current_frame_number += 1
                
                # Update current frame
                with self.frame_lock:
                    self.current_frame = frame.copy()
                
                # Call callback if provided
                if self.frame_callback:
                    try:
                        self.frame_callback(frame)
                    except Exception as e:
                        print(f"Frame callback error: {e}")
                
                # Maintain frame rate
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_time - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                print(f"Video capture loop error: {e}")
                break
    
    def seek_to_frame(self, frame_number: int) -> bool:
        """Seek to specific frame number"""
        try:
            if 0 <= frame_number < self.total_frames:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                self.current_frame_number = frame_number
                return True
            return False
        except Exception:
            return False
    
    def get_video_info(self) -> dict:
        """Get video file information"""
        info = self.get_camera_info()
        info.update({
            'total_frames': self.total_frames,
            'current_frame': self.current_frame_number,
            'duration': self.total_frames / self.settings.fps if self.settings.fps > 0 else 0,
            'video_path': self.video_path,
            'is_looping': self.is_looping
        })
        return info

def get_available_cameras() -> List[int]:
    """
    Get list of available camera indices
    
    Returns:
        List of available camera indices
    """
    available_cameras = []
    
    # Test camera indices 0-9
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
        cap.release()
    
    return available_cameras