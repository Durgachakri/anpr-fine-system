"""
Image enhancement techniques for license plates
"""
import cv2
import numpy as np
from typing import Tuple, Optional

class PlateEnhancer:
    """Enhance license plate images for better OCR accuracy"""
    
    @staticmethod
    def enhance_contrast(image: np.ndarray) -> np.ndarray:
        """
        Enhance contrast using CLAHE
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        return enhanced
    
    @staticmethod
    def denoise(image: np.ndarray) -> np.ndarray:
        """
        Remove noise from plate image
        """
        # Bilateral filter preserves edges while denoising
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        return denoised
    
    @staticmethod
    def correct_skew(image: np.ndarray) -> np.ndarray:
        """
        Correct skewed license plates
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
        
        if lines is not None:
            # Calculate average angle
            angles = []
            for rho, theta in lines[:, 0]:
                angle = theta * 180 / np.pi - 90
                angles.append(angle)
            
            median_angle = np.median(angles)
            
            # Rotate to correct skew
            if abs(median_angle) > 0.5:
                h, w = image.shape[:2]
                center = (w // 2, h // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                corrected = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                           flags=cv2.INTER_CUBIC,
                                           borderMode=cv2.BORDER_REPLICATE)
                return corrected
        
        return image
    
    @staticmethod
    def super_resolution(image: np.ndarray, scale: float = 2.0) -> np.ndarray:
        """
        Simple super-resolution using interpolation
        """
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        enhanced = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return enhanced
    
    @staticmethod
    def remove_shadows(image: np.ndarray) -> np.ndarray:
        """
        Remove shadows from plate image
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Morphological operations to estimate background
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        background = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
        background = cv2.morphologyEx(background, cv2.MORPH_ERODE, kernel)
        
        # Remove background
        diff = 255 - cv2.absdiff(gray, background)
        
        # Normalize
        normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        
        return normalized
    
    def process(self, image: np.ndarray, apply_all: bool = True) -> np.ndarray:
        """
        Apply all enhancement techniques
        """
        result = image.copy()
        
        if apply_all:
            result = self.correct_skew(result)
            result = self.denoise(result)
            result = self.remove_shadows(result)
            result = self.enhance_contrast(result)
            result = self.super_resolution(result)
        
        return result