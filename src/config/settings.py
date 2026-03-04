"""
Configuration settings for ANPR Fine System
"""
import os
from pathlib import Path
import sys

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
OUTPUT_DIR = BASE_DIR / 'output'

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, OUTPUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
    print(f"✅ Directory ready: {dir_path}")

# Ensure database directory exists
(DATA_DIR / 'database').mkdir(parents=True, exist_ok=True)

# ANPR Settings
class ANPRConfig:
    def __init__(self):
        # Detection settings
        self.CONFIDENCE_THRESHOLD = 0.5
        self.IOU_THRESHOLD = 0.45
        self.IMG_SIZE = 640
        
        # OCR settings
        self.OCR_LANGUAGES = ['en']
        self.OCR_CONFIDENCE_THRESHOLD = 0.6
        
        # Violation settings
        self.SPEED_LIMIT = 60  # km/h
        self.RED_LIGHT_FINE = 5000  # INR
        self.SPEEDING_FINE_PER_KM = 100  # INR per km over limit
        self.PARKING_FINE = 1000  # INR
        
        # Video processing
        self.PROCESS_EVERY_N_FRAMES = 5
        
        # Output settings
        self.SAVE_ANNOTATED_IMAGES = True
        self.SAVE_CROPPED_PLATES = True
        self.GENERATE_FINE_NOTICES = True
        
        # Database paths - ADD THESE INSIDE THE CLASS
        self.OWNER_DB_PATH = DATA_DIR / 'database' / 'vehicle_owner_db.csv'
        self.VIOLATION_LOG_PATH = DATA_DIR / 'database' / 'violation_logs.csv'
        
        # Model paths
        self.YOLO_MODEL_PATH = MODELS_DIR / 'pretrained' / 'yolov8n.pt'
        self.CUSTOM_PLATE_MODEL = MODELS_DIR / 'pretrained' / 'yolov8n.pt'
        
        # Also store as strings for compatibility
        self.YOLO_MODEL_PATH_STR = str(self.YOLO_MODEL_PATH)
        self.CUSTOM_PLATE_MODEL_STR = str(self.CUSTOM_PLATE_MODEL)
        self.OWNER_DB_PATH_STR = str(self.OWNER_DB_PATH)
        self.VIOLATION_LOG_PATH_STR = str(self.VIOLATION_LOG_PATH)

# Initialize config
config = ANPRConfig()

# Enhanced debugging
print("=" * 60)
print("ANPR FINE SYSTEM CONFIGURATION")
print("=" * 60)
print(f"BASE_DIR: {BASE_DIR}")
print(f"MODELS_DIR: {MODELS_DIR}")
print(f"YOLO_MODEL_PATH: {config.YOLO_MODEL_PATH}")
print(f"YOLO_MODEL_PATH exists: {config.YOLO_MODEL_PATH.exists()}")
print(f"OWNER_DB_PATH: {config.OWNER_DB_PATH}")
print(f"VIOLATION_LOG_PATH: {config.VIOLATION_LOG_PATH}")

if config.YOLO_MODEL_PATH.exists():
    file_size = config.YOLO_MODEL_PATH.stat().st_size
    print(f"✅ Model file found! Size: {file_size} bytes ({file_size/1024/1024:.2f} MB)")
else:
    print("❌ Model file NOT found!")
    
    # Check alternative locations
    alt_paths = [
        BASE_DIR / 'yolov8n.pt',
        BASE_DIR / 'models' / 'yolov8n.pt',
        BASE_DIR / 'pretrained' / 'yolov8n.pt',
        Path('yolov8n.pt'),
        Path('models/yolov8n.pt'),
    ]
    
    print("\nChecking alternative locations:")
    for alt_path in alt_paths:
        exists = alt_path.exists()
        status = "✅ FOUND" if exists else "❌ Not found"
        if exists:
            file_size = alt_path.stat().st_size
            print(f"  {status}: {alt_path} ({file_size} bytes)")
        else:
            print(f"  {status}: {alt_path}")

print("=" * 60)

# Export commonly used paths
YOLO_MODEL_PATH = config.YOLO_MODEL_PATH
CUSTOM_PLATE_MODEL = config.CUSTOM_PLATE_MODEL
OWNER_DB_PATH = config.OWNER_DB_PATH
VIOLATION_LOG_PATH = config.VIOLATION_LOG_PATH