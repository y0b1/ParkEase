"""
Custom exceptions for the parking detection system

This module defines all custom exceptions used throughout the system.
"""

class ParkingSystemException(Exception):
    """Base exception for all parking system errors"""
    pass

class CameraException(ParkingSystemException):
    """Exception raised when camera operations fail"""
    pass

class DetectionException(ParkingSystemException):
    """Exception raised when detection operations fail"""
    pass

class ConfigurationException(ParkingSystemException):
    """Exception raised when configuration operations fail"""
    pass

class GUIException(ParkingSystemException):
    """Exception raised when GUI operations fail"""
    pass

class ValidationException(ParkingSystemException):
    """Exception raised when validation fails"""
    pass