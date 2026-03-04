"""
Visualization utilities for drawing annotations
"""
import cv2
import numpy as np
from typing import Tuple, Optional

class Visualizer:
    """Helper class for drawing on images"""
    
    @staticmethod
    def annotate_plate(image: np.ndarray, bbox: Tuple[int, int, int, int],
                       plate_text: str, violation_type: Optional[str] = None,
                       fine_amount: Optional[float] = None) -> np.ndarray:
        """
        Annotate image with plate detection and violation info
        
        Args:
            image: Input image
            bbox: Bounding box (x1, y1, x2, y2)
            plate_text: Detected plate number
            violation_type: Type of violation if any
            fine_amount: Fine amount if any
            
        Returns:
            Annotated image
        """
        x1, y1, x2, y2 = bbox
        
        # Choose color based on violation
        if violation_type:
            color = (0, 0, 255)  # Red for violation
        else:
            color = (0, 255, 0)  # Green for normal
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw plate text background
        text = plate_text
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), color, -1)
        
        # Draw plate text
        cv2.putText(image, text, (x1 + 5, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw violation info if present
        if violation_type and fine_amount:
            info_text = f"{violation_type}: ₹{fine_amount:,.0f}"
            (info_w, info_h), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Draw below the plate
            cv2.rectangle(image, (x1, y2), (x1 + info_w + 10, y2 + info_h + 10), 
                         (0, 0, 255), -1)
            cv2.putText(image, info_text, (x1 + 5, y2 + info_h + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return image
    
    @staticmethod
    def create_fine_notice(plate_text: str, owner_name: str, 
                          violation: str, amount: float,
                          location: str, timestamp: str) -> np.ndarray:
        """
        Create a fine notice image
        
        Args:
            plate_text: License plate number
            owner_name: Owner's name
            violation: Type of violation
            amount: Fine amount
            location: Location of violation
            timestamp: Time of violation
            
        Returns:
            Fine notice image
        """
        # Create blank image
        img = np.ones((600, 800, 3), dtype=np.uint8) * 255
        
        # Draw border
        cv2.rectangle(img, (10, 10), (790, 590), (0, 0, 255), 3)
        
        # Header
        cv2.putText(img, "TRAFFIC VIOLATION NOTICE", (200, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Fine details
        y_start = 120
        line_height = 40
        
        details = [
            f"License Plate: {plate_text}",
            f"Owner: {owner_name}",
            f"Violation: {violation.upper()}",
            f"Location: {location}",
            f"Time: {timestamp}",
            f"Fine Amount: ₹{amount:,.2f}",
            "",
            "Please pay within 15 days to avoid penalty.",
            "Payment methods: Online, Bank, Traffic Office"
        ]
        
        for i, line in enumerate(details):
            if "Fine Amount" in line:
                color = (0, 0, 255)
                font_scale = 0.9
            else:
                color = (0, 0, 0)
                font_scale = 0.7
            
            cv2.putText(img, line, (50, y_start + i * line_height),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
        
        return img