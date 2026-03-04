"""
YOLO-based license plate detector with enhanced debugging
"""
import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
import logging
import os
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LicensePlateDetector:
    """Detect license plates in images using YOLO"""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        """
        Initialize the detector with a YOLO model
        
        Args:
            model_path: Path to YOLO model weights
            conf_threshold: Confidence threshold for detections
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        
        # Enhanced debugging information
        logger.info("=" * 50)
        logger.info("LICENSE PLATE DETECTOR INITIALIZATION")
        logger.info("=" * 50)
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Model path provided: {model_path}")
        logger.info(f"Absolute model path: {self.model_path.absolute()}")
        logger.info(f"Model exists: {self.model_path.exists()}")
        
        if self.model_path.exists():
            file_size = self.model_path.stat().st_size
            logger.info(f"File size: {file_size} bytes ({file_size/1024/1024:.2f} MB)")
            
            # Check if file is too small (corrupted)
            if file_size < 6_000_000:  # Less than 6MB
                logger.error(f"⚠️ Model file is too small ({file_size} bytes). It may be corrupted!")
            else:
                logger.info("✅ Model file size looks good")
        else:
            logger.error(f"❌ Model file not found at: {self.model_path.absolute()}")
            logger.info("Checking alternative locations...")
            
            # Check common alternative locations
            alt_paths = [
                Path("yolov8n.pt"),
                Path("models/yolov8n.pt"),
                Path("pretrained/yolov8n.pt"),
                Path("../yolov8n.pt"),
                Path("../../yolov8n.pt")
            ]
            
            for alt_path in alt_paths:
                if alt_path.exists():
                    logger.info(f"✅ Found model at alternative location: {alt_path.absolute()}")
                    self.model_path = alt_path
                    break
        
        # Load YOLO model
        try:
            if self.model_path.exists():
                logger.info(f"Loading model from: {self.model_path.absolute()}")
                self.model = YOLO(str(self.model_path))
                logger.info(f"✅ Successfully loaded YOLO model from {self.model_path}")
            else:
                # Use pretrained model and suggest fine-tuning
                logger.warning(f"Model {self.model_path} not found. Downloading base YOLOv8n...")
                self.model = YOLO('yolov8n.pt')
                logger.info("✅ Base YOLOv8n model downloaded and loaded")
                
                # Save the downloaded model to pretrained folder for future use
                try:
                    save_dir = Path("models/pretrained")
                    save_dir.mkdir(parents=True, exist_ok=True)
                    save_path = save_dir / "yolov8n.pt"
                    
                    # Copy the downloaded model
                    import shutil
                    if Path("yolov8n.pt").exists():
                        shutil.copy("yolov8n.pt", save_path)
                        logger.info(f"✅ Saved model to {save_path} for future use")
                except Exception as e:
                    logger.warning(f"Could not save model: {e}")
                    
        except Exception as e:
            logger.error(f"❌ Failed to load YOLO model: {e}")
            logger.error("Please ensure the model file is not corrupted and try again.")
            sys.exit(1)
        
        logger.info("=" * 50)
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detect license plates in image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of dictionaries containing detection info
        """
        if image is None:
            logger.error("Input image is None")
            return []
        
        logger.info(f"Processing image of shape: {image.shape}")
        
        try:
            # Run inference
            results = self.model(image, conf=self.conf_threshold)[0]
            
            detections = []
            for box in results.boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                
                # Ensure coordinates are within image bounds
                h, w = image.shape[:2]
                x1 = max(0, min(x1, w))
                y1 = max(0, min(y1, h))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))
                
                # Crop plate region
                if y2 > y1 and x2 > x1:
                    plate_crop = image[y1:y2, x1:x2]
                else:
                    logger.warning(f"Invalid bounding box: ({x1}, {y1}, {x2}, {y2})")
                    continue
                
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': confidence,
                    'crop': plate_crop,
                    'center': ((x1 + x2) // 2, (y1 + y2) // 2)
                })
            
            logger.info(f"✅ Detected {len(detections)} license plates")
            return detections
            
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return []
    
    def detect_from_path(self, image_path: str) -> Tuple[List[Dict], np.ndarray]:
        """
        Detect plates from image file path
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (detections, original_image)
        """
        logger.info(f"Reading image from: {image_path}")
        
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        logger.info(f"Image loaded successfully. Shape: {image.shape}")
        
        detections = self.detect(image)
        return detections, image
    
    def batch_detect(self, image_paths: List[str]) -> Dict[str, List[Dict]]:
        """
        Detect plates in multiple images
        
        Args:
            image_paths: List of image paths
            
        Returns:
            Dictionary mapping image paths to detections
        """
        results = {}
        for path in image_paths:
            try:
                logger.info(f"Batch processing: {path}")
                detections, _ = self.detect_from_path(path)
                results[path] = detections
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
                results[path] = []
        
        return results
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        info = {
            'model_path': str(self.model_path),
            'model_exists': self.model_path.exists(),
            'conf_threshold': self.conf_threshold
        }
        
        if self.model_path.exists():
            info['file_size'] = self.model_path.stat().st_size
            info['file_size_mb'] = round(info['file_size'] / (1024 * 1024), 2)
        
        return info