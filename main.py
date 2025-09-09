#!/usr/bin/env python3
"""
Smart Parking Detection System
Main entry point for the application

Author: Assistant
Date: 2025
"""

import sys
import os
import tkinter as tk
from tkinter import messagebox
import logging

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from gui.main_window import ParkingDetectionApp
    from core.exceptions import ParkingSystemException
    from utils.logger import setup_logger
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure all required files are in the correct directories")
    sys.exit(1)

def main():
    """Main entry point for the parking detection system"""
    
    # Setup logging
    logger = setup_logger()
    logger.info("Starting Smart Parking Detection System")
    
    try:
        # Create the main application
        root = tk.Tk()
        app = ParkingDetectionApp(root)
        
        # Start the GUI main loop
        root.mainloop()
        
    except ParkingSystemException as e:
        logger.error(f"Parking System Error: {e}")
        messagebox.showerror("System Error", str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        messagebox.showerror("Unexpected Error", f"An unexpected error occurred: {e}")
    finally:
        logger.info("Shutting down Smart Parking Detection System")

if __name__ == "__main__":
    main()