"""
Post-process OCR text to format license plates correctly
"""
import re
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class PlateTextPostprocessor:
    """Clean and format license plate text"""
    
    def __init__(self, country: str = 'IN'):
        """
        Initialize postprocessor
        
        Args:
            country: Country code for plate format rules
        """
        self.country = country
        self.patterns = {
            'IN': r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{1,4}$',  # Indian format
            'US': r'^[A-Z0-9]{1,7}$',  # US format (simplified)
            'UK': r'^[A-Z]{2}[0-9]{2}[A-Z]{3}$',  # UK format
            'EU': r'^[A-Z]{1,3}[0-9]{1,4}[A-Z]{0,3}$'  # General European
        }
    
    def clean_text(self, text: str) -> str:
        """
        Remove unwanted characters and standardize format
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove spaces, dashes, special characters
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # Remove common OCR errors
        replacements = {
            'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8',
            'G': '6', 'Q': '0', 'D': '0', 'L': '1', 'T': '7'
        }
        
        # Only replace in appropriate positions (heuristic)
        # Usually letters in first part, numbers in last part
        if len(cleaned) >= 4:
            # First 2-3 characters should be letters
            for i in range(min(3, len(cleaned))):
                if cleaned[i].isdigit() and cleaned[i] in replacements.values():
                    # Find the letter it might represent
                    for letter, digit in replacements.items():
                        if digit == cleaned[i]:
                            cleaned = cleaned[:i] + letter + cleaned[i+1:]
                            break
            
            # Last characters should be numbers
            for i in range(max(0, len(cleaned)-4), len(cleaned)):
                if cleaned[i].isalpha() and cleaned[i] in replacements:
                    cleaned = cleaned[:i] + replacements[cleaned[i]] + cleaned[i+1:]
        
        return cleaned
    
    def validate_format(self, plate: str) -> bool:
        """
        Check if plate matches expected format
        
        Args:
            plate: Cleaned plate number
            
        Returns:
            True if format is valid
        """
        pattern = self.patterns.get(self.country, self.patterns['IN'])
        return bool(re.match(pattern, plate))
    
    def format_plate(self, plate: str) -> str:
        """
        Apply standard formatting (e.g., MH12AB1234 -> MH 12 AB 1234)
        
        Args:
            plate: Clean plate number
            
        Returns:
            Formatted plate string
        """
        if self.country == 'IN' and len(plate) >= 10:
            # Format: MH12AB1234 -> MH 12 AB 1234
            return f"{plate[:2]} {plate[2:4]} {plate[4:6]} {plate[6:]}"
        elif len(plate) >= 7:
            # Generic formatting
            return ' '.join(plate[i:i+2] for i in range(0, len(plate), 2))
        
        return plate
    
    def process(self, raw_text: str) -> Optional[str]:
        """
        Complete postprocessing pipeline
        
        Args:
            raw_text: Raw OCR text
            
        Returns:
            Formatted plate number or None if invalid
        """
        if not raw_text:
            return None
        
        # Clean
        cleaned = self.clean_text(raw_text)
        
        # Validate (optional - can be disabled for flexibility)
        # if not self.validate_format(cleaned):
        #     logger.warning(f"Plate {cleaned} doesn't match expected format")
        #     return None
        
        # Format for display
        formatted = self.format_plate(cleaned)
        
        logger.info(f"Processed plate: {raw_text} -> {cleaned} -> {formatted}")
        return formatted