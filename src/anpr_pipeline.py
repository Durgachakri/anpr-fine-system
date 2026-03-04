"""
Main ANPR pipeline integrating all components
"""
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

from src.config.settings import config, BASE_DIR, OUTPUT_DIR
from src.detection.plate_detector import LicensePlateDetector
from src.recognition.ocr_reader import PlateOCR
from src.recognition.text_postprocessor import PlateTextPostprocessor
from src.database.owner_db import OwnerDatabase
from src.database.fine_logger import FineLogger
from src.violation.rules_engine import ViolationRulesEngine
from src.utils.visualization import Visualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ANPRFineSystem:
    """Complete ANPR system with fine issuance simulation"""
    
    def __init__(self, config):
        """
        Initialize the complete ANPR system
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Initialize components
        logger.info("Initializing ANPR Fine System...")
        
        # Detection
        model_path = config.CUSTOM_PLATE_MODEL if Path(config.CUSTOM_PLATE_MODEL).exists() else config.YOLO_MODEL_PATH
        self.detector = LicensePlateDetector(
            model_path=str(model_path),
            conf_threshold=config.CONFIDENCE_THRESHOLD
        )
        
        # Recognition
        self.ocr = PlateOCR(languages=config.OCR_LANGUAGES)
        self.postprocessor = PlateTextPostprocessor(country='IN')
        
        # Database
        self.owner_db = OwnerDatabase(str(config.OWNER_DB_PATH))
        self.fine_logger = FineLogger(str(config.VIOLATION_LOG_PATH))
        
        # Rules
        self.rules_engine = ViolationRulesEngine(config)
        
        # Visualization
        self.visualizer = Visualizer()
        
        logger.info("ANPR Fine System initialized successfully")
    
    def process_image(self, image_path: str, 
                     violation_type: Optional[str] = None,
                     location: str = "Unknown") -> Dict:
        """
        Process a single image through the complete pipeline
        
        Args:
            image_path: Path to input image
            violation_type: Optional violation type (for simulation)
            location: Location where image was captured
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Processing image: {image_path}")
        
        # Load and detect
        detections, image = self.detector.detect_from_path(image_path)
        
        if not detections:
            logger.warning("No license plates detected")
            return {'status': 'no_detection', 'plates': []}
        
        results = []
        annotated_image = image.copy()
        
        for i, detection in enumerate(detections):
            # Get plate crop
            plate_crop = detection['crop']
            bbox = detection['bbox']
            
            # Save cropped plate if configured
            if self.config.SAVE_CROPPED_PLATES:
                crop_path = OUTPUT_DIR / 'processed' / 'detected_plates' / f"plate_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.jpg"
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(crop_path), plate_crop)
            
            # Read plate text
            plate_text, ocr_confidence = self.ocr.read_plate_multiple_attempts(plate_crop)
            
            if plate_text:
                # Post-process
                formatted_plate = self.postprocessor.process(plate_text)
                
                if formatted_plate:
                    # Look up owner
                    owner = self.owner_db.lookup_owner(formatted_plate)
                    
                    # Check for violation (simulated)
                    fine_amount = 0
                    if violation_type:
                        fine_amount = self.rules_engine.calculate_fine(violation_type)
                        
                        # Log the fine
                        fine_data = {
                            'plate_number': formatted_plate,
                            'violation_type': violation_type,
                            'location': location,
                            'fine_amount': fine_amount,
                            'owner_name': owner.get('owner_name', 'Unknown') if owner else 'Unknown',
                            'owner_address': owner.get('address', 'Unknown') if owner else 'Unknown',
                            'evidence_image': str(image_path)
                        }
                        fine_id = self.fine_logger.log_fine(fine_data)
                        logger.info(f"Fine issued: {fine_id} for {fine_amount}")
                    
                    # Store result
                    result = {
                        'plate_number': formatted_plate,
                        'raw_text': plate_text,
                        'ocr_confidence': ocr_confidence,
                        'detection_confidence': detection['confidence'],
                        'bbox': bbox,
                        'owner': owner,
                        'violation': violation_type,
                        'fine_amount': fine_amount,
                        'fine_issued': bool(fine_amount)
                    }
                    results.append(result)
                    
                    # Annotate image
                    annotated_image = self.visualizer.annotate_plate(
                        annotated_image, bbox, formatted_plate, 
                        violation_type, fine_amount
                    )
        
        # Save annotated image
        if self.config.SAVE_ANNOTATED_IMAGES:
            output_path = OUTPUT_DIR / 'processed' / 'annotated' / f"annotated_{Path(image_path).name}"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), annotated_image)
            logger.info(f"Saved annotated image to {output_path}")
        
        return {
            'status': 'success',
            'image_path': image_path,
            'plates': results,
            'total_fines': sum(1 for r in results if r.get('fine_issued')),
            'total_amount': sum(r.get('fine_amount', 0) for r in results)
        }
    
    def process_video(self, video_path: str, 
                     violation_zones: Optional[Dict] = None,
                     output_video: Optional[str] = None) -> List[Dict]:
        """
        Process video file
        
        Args:
            video_path: Path to input video
            violation_zones: Dictionary with violation zone definitions
            output_video: Path to save annotated video
            
        Returns:
            List of detected plates with fines
        """
        logger.info(f"Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize video writer
        if output_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        
        frame_count = 0
        detected_plates = []
        vehicle_tracker = {}  # Track vehicles across frames
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every N frames for efficiency
            if frame_count % self.config.PROCESS_EVERY_N_FRAMES == 0:
                # Detect plates
                detections = self.detector.detect(frame)
                
                for detection in detections:
                    plate_crop = detection['crop']
                    bbox = detection['bbox']
                    
                    # Read plate
                    plate_text, confidence = self.ocr.read_plate(plate_crop)
                    
                    if plate_text and confidence > self.config.OCR_CONFIDENCE_THRESHOLD:
                        formatted_plate = self.postprocessor.process(plate_text)
                        
                        if formatted_plate:
                            # Check if we've seen this plate before
                            if formatted_plate not in vehicle_tracker:
                                vehicle_tracker[formatted_plate] = {
                                    'first_seen': frame_count,
                                    'last_seen': frame_count,
                                    'bbox': bbox
                                }
                            else:
                                vehicle_tracker[formatted_plate]['last_seen'] = frame_count
                            
                            # Check for violations (simplified)
                            # For demo, simulate random violations
                            if violation_zones and frame_count % (fps * 10) == 0:  # Every 10 seconds
                                # Simulate violation detection
                                violation_type = 'speeding'  # In real app, use actual detection
                                fine_amount = self.rules_engine.calculate_fine(violation_type)
                                
                                # Look up owner
                                owner = self.owner_db.lookup_owner(formatted_plate)
                                
                                # Log fine
                                fine_data = {
                                    'plate_number': formatted_plate,
                                    'violation_type': violation_type,
                                    'location': video_path,
                                    'fine_amount': fine_amount,
                                    'owner_name': owner.get('owner_name', 'Unknown'),
                                    'owner_address': owner.get('address', 'Unknown'),
                                    'evidence_image': f"video_frame_{frame_count}.jpg"
                                }
                                fine_id = self.fine_logger.log_fine(fine_data)
                                
                                detected_plates.append({
                                    'plate': formatted_plate,
                                    'fine_id': fine_id,
                                    'amount': fine_amount,
                                    'violation': violation_type,
                                    'timestamp': frame_count / fps
                                })
                            
                            # Annotate frame
                            frame = self.visualizer.annotate_plate(
                                frame, bbox, formatted_plate
                            )
            
            # Write frame
            if output_video:
                out.write(frame)
            
            # Display progress
            if frame_count % (fps * 10) == 0:  # Every 10 seconds
                logger.info(f"Processed {frame_count/fps:.1f} seconds")
            
            frame_count += 1
        
        # Cleanup
        cap.release()
        if output_video:
            out.release()
        
        logger.info(f"Video processing complete. Detected {len(detected_plates)} violations")
        return detected_plates
    
    def generate_report(self, date: Optional[str] = None) -> Dict:
        """
        Generate daily report of violations and fines
        
        Args:
            date: Date string (YYYY-MM-DD)
            
        Returns:
            Report dictionary
        """
        report = self.fine_logger.generate_daily_report(date)
        
        # Add summary statistics
        report['summary'] = {
            'total_revenue': report['total_amount'],
            'average_fine': report['total_amount'] / report['total_fines'] if report['total_fines'] > 0 else 0,
            'most_common_violation': max(report['violation_breakdown'].items(), 
                                        key=lambda x: x[1])[0] if report['violation_breakdown'] else None
        }
        
        # Save report
        report_path = OUTPUT_DIR / 'reports' / f"report_{date}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to {report_path}")
        return report