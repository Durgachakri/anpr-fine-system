"""
Utility functions for image processing
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class ImageUtils:
    """Helper functions for image operations"""
    
    @staticmethod
    def load_image(image_path: str) -> Optional[np.ndarray]:
        """
        Load image from path with error handling
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
            return image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    @staticmethod
    def save_image(image: np.ndarray, save_path: str) -> bool:
        """
        Save image with directory creation
        """
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), image)
            logger.info(f"Image saved to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            return False
    
    @staticmethod
    def resize_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
        """
        Resize image maintaining aspect ratio
        """
        h, w = image.shape[:2]
        aspect_ratio = w / h
        target_width = int(target_height * aspect_ratio)
        resized = cv2.resize(image, (target_width, target_height))
        return resized
    
    @staticmethod
    def crop_region(image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        """
        Safely crop region from image
        """
        img_h, img_w = image.shape[:2]
        
        # Ensure coordinates are within bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        
        if w <= 0 or h <= 0:
            return image
        
        return image[y:y+h, x:x+w]
    
    @staticmethod
    def add_padding(image: np.ndarray, padding: int = 10) -> np.ndarray:
        """
        Add padding around image
        """
        return cv2.copyMakeBorder(image, padding, padding, padding, padding, 
                                 cv2.BORDER_CONSTANT, value=[255, 255, 255])
    
    @staticmethod
    def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
        """
        Convert to grayscale safely
        """
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.copy()
    
    @staticmethod
    def get_image_info(image: np.ndarray) -> dict:
        """
        Get image metadata
        """
        info = {
            'shape': image.shape,
            'dtype': str(image.dtype),
            'min_value': float(image.min()),
            'max_value': float(image.max()),
            'mean_value': float(image.mean())
        }
        
        if len(image.shape) == 3:
            info['channels'] = image.shape[2]
            info['height'] = image.shape[0]
            info['width'] = image.shape[1]
        else:
            info['channels'] = 1
            info['height'] = image.shape[0]
            info['width'] = image.shape[1]
        
        return info
    
    @staticmethod
    def batch_process(images: List[np.ndarray], function, **kwargs) -> List[np.ndarray]:
        """
        Apply function to multiple images
        """
        results = []
        for img in images:
            try:
                result = function(img, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                results.append(img)  # Return original on error
        return results