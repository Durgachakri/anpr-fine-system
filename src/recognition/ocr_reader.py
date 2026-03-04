"""
OCR module for reading license plate text
"""
import cv2
import numpy as np
import easyocr
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)

class PlateOCR:
    """OCR reader for license plate text extraction"""
    
    def __init__(self, languages: List[str] = ['en'], gpu: bool = False):
        """
        Initialize OCR reader
        
        Args:
            languages: List of language codes
            gpu: Whether to use GPU
        """
        self.reader = easyocr.Reader(languages, gpu=gpu)
        logger.info(f"Initialized EasyOCR reader with languages: {languages}")
    
    def preprocess_plate(self, plate_image: np.ndarray) -> np.ndarray:
        """
        Preprocess plate image for better OCR
        
        Args:
            plate_image: Cropped plate image
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image.copy()
        
        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(denoised, 255, 
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        return cleaned
    
    def read_plate(self, plate_image: np.ndarray, 
                   confidence_threshold: float = 0.5) -> Tuple[Optional[str], float]:
        """
        Read text from plate image
        
        Args:
            plate_image: Cropped plate image
            confidence_threshold: Minimum confidence for valid reading
            
        Returns:
            Tuple of (plate_text, confidence)
        """
        # Preprocess
        processed = self.preprocess_plate(plate_image)
        
        # Perform OCR
        results = self.reader.readtext(processed)
        
        if not results:
            return None, 0.0
        
        # Combine all detected text
        all_text = []
        total_confidence = 0
        
        for (bbox, text, confidence) in results:
            # Clean text (keep only alphanumeric)
            clean_text = ''.join(c for c in text if c.isalnum()).upper()
            if clean_text and confidence >= confidence_threshold:
                all_text.append(clean_text)
                total_confidence += confidence
        
        if all_text:
            combined_text = ''.join(all_text)
            avg_confidence = total_confidence / len(all_text)
            logger.info(f"Read plate: {combined_text} (conf: {avg_confidence:.2f})")
            return combined_text, avg_confidence
        
        return None, 0.0
    
    def read_plate_multiple_attempts(self, plate_image: np.ndarray, 
                                    attempts: int = 3) -> Tuple[Optional[str], float]:
        """
        Try multiple preprocessing techniques for better results
        
        Args:
            plate_image: Cropped plate image
            attempts: Number of preprocessing variations
            
        Returns:
            Best result tuple (text, confidence)
        """
        best_text = None
        best_confidence = 0
        
        # Different preprocessing variations
        variations = [
            self.preprocess_plate,  # Default preprocessing
            lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY),  # Just grayscale
            lambda x: cv2.GaussianBlur(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), (5,5), 0),  # Blurred
        ]
        
        for preprocess_func in variations[:attempts]:
            try:
                processed = preprocess_func(plate_image)
                results = self.reader.readtext(processed)
                
                if results:
                    # Get best result from this variation
                    for (_, text, conf) in results:
                        clean_text = ''.join(c for c in text if c.isalnum()).upper()
                        if clean_text and conf > best_confidence:
                            best_confidence = conf
                            best_text = clean_text
            except Exception as e:
                logger.debug(f"Preprocessing variation failed: {e}")
                continue
        
        if best_text:
            logger.info(f"Best result: {best_text} (conf: {best_confidence:.2f})")
        
        return best_text, best_confidence