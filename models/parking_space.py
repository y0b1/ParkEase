"""
Data models for the parking detection system

This module contains all data classes and models used throughout the system.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
import json

class SpaceStatus(Enum):
    """Enumeration for parking space status"""
    VACANT = "vacant"
    OCCUPIED = "occupied"
    UNKNOWN = "unknown"

@dataclass
class ParkingSpace:
    """
    Data class representing a parking space
    
    Attributes:
        id: Unique identifier for the parking space
        x: X coordinate of top-left corner
        y: Y coordinate of top-left corner
        width: Width of the parking space
        height: Height of the parking space
        status: Current occupancy status
        confidence: Confidence level of detection (0-100)
    """
    id: int
    x: int
    y: int
    width: int
    height: int
    status: SpaceStatus = SpaceStatus.UNKNOWN
    confidence: float = 0.0
    
    @property
    def is_occupied(self) -> bool:
        """Check if the space is occupied"""
        return self.status == SpaceStatus.OCCUPIED
    
    @property
    def is_vacant(self) -> bool:
        """Check if the space is vacant"""
        return self.status == SpaceStatus.VACANT
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get the center point of the parking space"""
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def area(self) -> int:
        """Get the area of the parking space"""
        return self.width * self.height
    
    def contains_point(self, x: int, y: int) -> bool:
        """Check if a point is inside this parking space"""
        return (self.x <= x <= self.x + self.width and 
                self.y <= y <= self.y + self.height)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height,
            'status': self.status.value,
            'confidence': self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ParkingSpace':
        """Create ParkingSpace from dictionary"""
        return cls(
            id=data['id'],
            x=data['x'],
            y=data['y'],
            width=data['width'],
            height=data['height'],
            status=SpaceStatus(data.get('status', SpaceStatus.UNKNOWN.value)),
            confidence=data.get('confidence', 0.0)
        )

@dataclass
class DetectionParameters:
    """
    Configuration parameters for parking detection
    """
    threshold1: int = 100
    threshold2: int = 200
    min_pixels: int = 150
    max_pixels: int = 800
    blur_kernel: int = 5
    morphology_kernel: int = 3
    area_threshold: float = 0.3
    confidence_threshold: float = 0.7
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'threshold1': self.threshold1,
            'threshold2': self.threshold2,
            'min_pixels': self.min_pixels,
            'max_pixels': self.max_pixels,
            'blur_kernel': self.blur_kernel,
            'morphology_kernel': self.morphology_kernel,
            'area_threshold': self.area_threshold,
            'confidence_threshold': self.confidence_threshold
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DetectionParameters':
        """Create DetectionParameters from dictionary"""
        return cls(**data)

@dataclass
class ParkingLotStats:
    """
    Statistics for the parking lot
    """
    total_spaces: int = 0
    occupied_spaces: int = 0
    vacant_spaces: int = 0
    occupancy_rate: float = 0.0
    last_updated: Optional[str] = None
    
    @property
    def utilization_percentage(self) -> float:
        """Get utilization as percentage"""
        if self.total_spaces == 0:
            return 0.0
        return (self.occupied_spaces / self.total_spaces) * 100
    
    def update_from_spaces(self, spaces: List[ParkingSpace]) -> None:
        """Update statistics from list of parking spaces"""
        self.total_spaces = len(spaces)
        self.occupied_spaces = sum(1 for space in spaces if space.is_occupied)
        self.vacant_spaces = self.total_spaces - self.occupied_spaces
        self.occupancy_rate = self.utilization_percentage
        self.last_updated = time.strftime('%Y-%m-%d %H:%M:%S')

class ParkingLotConfiguration:
    """
    Configuration manager for parking lot setup
    """
    
    def __init__(self, config_file: str = 'parking_config.json'):
        self.config_file = config_file
        self.spaces: List[ParkingSpace] = []
        self.parameters = DetectionParameters()
        self.camera_settings = {
            'source': 0,
            'width': 1280,
            'height': 720,
            'fps': 30
        }
    
    def add_space(self, space: ParkingSpace) -> None:
        """Add a parking space to the configuration"""
        # Ensure unique ID
        space.id = len(self.spaces)
        self.spaces.append(space)
    
    def remove_space(self, space_id: int) -> bool:
        """Remove a parking space by ID"""
        for i, space in enumerate(self.spaces):
            if space.id == space_id:
                del self.spaces[i]
                # Reassign IDs to maintain sequence
                for j, remaining_space in enumerate(self.spaces[i:], start=i):
                    remaining_space.id = j
                return True
        return False
    
    def clear_spaces(self) -> None:
        """Clear all parking spaces"""
        self.spaces.clear()
    
    def get_space_by_id(self, space_id: int) -> Optional[ParkingSpace]:
        """Get a parking space by ID"""
        for space in self.spaces:
            if space.id == space_id:
                return space
        return None
    
    def get_space_at_point(self, x: int, y: int) -> Optional[ParkingSpace]:
        """Get parking space at given coordinates"""
        for space in self.spaces:
            if space.contains_point(x, y):
                return space
        return None
    
    def save_to_file(self) -> bool:
        """Save configuration to file"""
        try:
            config_data = {
                'spaces': [space.to_dict() for space in self.spaces],
                'parameters': self.parameters.to_dict(),
                'camera_settings': self.camera_settings,
                'version': '1.0',
                'created': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False
    
    def load_from_file(self) -> bool:
        """Load configuration from file"""
        try:
            if not os.path.exists(self.config_file):
                return False
                
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
            
            # Load parking spaces
            self.spaces = [
                ParkingSpace.from_dict(space_data) 
                for space_data in config_data.get('spaces', [])
            ]
            
            # Load parameters
            if 'parameters' in config_data:
                self.parameters = DetectionParameters.from_dict(config_data['parameters'])
            
            # Load camera settings
            if 'camera_settings' in config_data:
                self.camera_settings.update(config_data['camera_settings'])
            
            return True
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return False
    
    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """Validate the current configuration"""
        errors = []
        
        if not self.spaces:
            errors.append("No parking spaces defined")
        
        # Check for overlapping spaces
        for i, space1 in enumerate(self.spaces):
            for j, space2 in enumerate(self.spaces[i+1:], start=i+1):
                if self._spaces_overlap(space1, space2):
                    errors.append(f"Spaces {space1.id} and {space2.id} overlap")
        
        # Check space dimensions
        for space in self.spaces:
            if space.width < 20 or space.height < 20:
                errors.append(f"Space {space.id} is too small (minimum 20x20 pixels)")
            if space.area < 400:
                errors.append(f"Space {space.id} area is too small")
        
        return len(errors) == 0, errors
    
    def _spaces_overlap(self, space1: ParkingSpace, space2: ParkingSpace) -> bool:
        """Check if two parking spaces overlap"""
        return not (space1.x + space1.width < space2.x or
                   space2.x + space2.width < space1.x or
                   space1.y + space1.height < space2.y or
                   space2.y + space2.height < space1.y)

import time