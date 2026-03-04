"""
Rules engine for detecting traffic violations
"""
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class ViolationRulesEngine:
    """Detect different types of traffic violations"""
    
    def __init__(self, config):
        """
        Initialize rules engine
        
        Args:
            config: Configuration object with violation parameters
        """
        self.config = config
        self.violation_types = ['speeding', 'red_light', 'parking', 'toll_evasion']
    
    def detect_speeding(self, vehicle_speed: float, speed_limit: Optional[float] = None) -> Tuple[bool, float]:
        """
        Detect speeding violation
        
        Args:
            vehicle_speed: Detected vehicle speed in km/h
            speed_limit: Speed limit for the road
            
        Returns:
            Tuple of (is_violation, fine_amount)
        """
        if speed_limit is None:
            speed_limit = self.config.SPEED_LIMIT
        
        if vehicle_speed > speed_limit:
            overspeed = vehicle_speed - speed_limit
            fine = self.config.SPEEDING_FINE_PER_KM * overspeed
            logger.info(f"Speeding detected: {vehicle_speed} km/h > {speed_limit} km/h")
            return True, fine
        
        return False, 0
    
    def detect_red_light(self, frame: np.ndarray, 
                        traffic_light_zone: Tuple[int, int, int, int],
                        vehicle_bbox: Tuple[int, int, int, int]) -> bool:
        """
        Detect red light violation
        
        Args:
            frame: Current video frame
            traffic_light_zone: ROI where traffic light is located
            vehicle_bbox: Vehicle bounding box
            
        Returns:
            True if red light violation detected
        """
        # This is a simplified simulation
        # In real implementation, you'd use traffic light detection
        
        # For demo, we'll simulate based on timestamp
        current_second = datetime.now().second
        is_red = current_second % 10 < 5  # Red for first 5 seconds of every 10
        
        if is_red:
            # Check if vehicle crossed stop line
            # Simplified: check if vehicle is beyond a certain point
            x, y, w, h = vehicle_bbox
            stop_line_y = 300  # Example stop line position
            vehicle_bottom = y + h
            
            if vehicle_bottom > stop_line_y:
                logger.info("Red light violation detected")
                return True
        
        return False
    
    def detect_parking_violation(self, frame: np.ndarray,
                                no_parking_zone: Tuple[int, int, int, int],
                                vehicle_bbox: Tuple[int, int, int, int],
                                duration_frames: int = 30) -> bool:
        """
        Detect parking violation
        
        Args:
            frame: Current frame
            no_parking_zone: ROI for no-parking area
            vehicle_bbox: Vehicle bounding box
            duration_frames: How long vehicle has been stationary
            
        Returns:
            True if parking violation
        """
        # Simplified: check if vehicle overlaps with no-parking zone
        x1, y1, x2, y2 = vehicle_bbox
        vz_x1, vz_y1, vz_x2, vz_y2 = no_parking_zone
        
        # Check for overlap
        overlap = not (x2 < vz_x1 or x1 > vz_x2 or y2 < vz_y1 or y1 > vz_y2)
        
        if overlap and duration_frames > 30:  # Stationary for > 1 sec at 30fps
            logger.info("Parking violation detected")
            return True
        
        return False
    
    def calculate_fine(self, violation_type: str, details: Optional[Dict] = None) -> float:
        """
        Calculate fine amount based on violation type
        
        Args:
            violation_type: Type of violation
            details: Additional details for fine calculation
            
        Returns:
            Fine amount
        """
        if violation_type == 'red_light':
            return self.config.RED_LIGHT_FINE
        elif violation_type == 'parking':
            return self.config.PARKING_FINE
        elif violation_type == 'speeding' and details:
            return self.config.SPEEDING_FINE_PER_KM * details.get('overspeed', 0)
        elif violation_type == 'toll_evasion':
            return 2 * details.get('toll_amount', 100)  # Double penalty
        
        return 500  # Default fine