"""
Test script for ANPR Fine System
Run this file to test all components automatically
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

print("=" * 60)
print("ANPR FINE SYSTEM - COMPLETE TEST SUITE")
print("=" * 60)

# Helper function to ensure path is a folder
def ensure_folder(path):
    """Ensure path is a folder, delete if it's a file"""
    path = Path(path)
    if path.exists() and not path.is_dir():
        print(f"  ⚠️  Removing file: {path} (should be a folder)")
        path.unlink()  # Delete the file
    path.mkdir(parents=True, exist_ok=True)
    return path

# Step 1: Check and create necessary directories
print("\n📁 Step 1: Checking directories...")
directories = [
    "data/raw/images",
    "data/raw/videos",
    "data/raw/sample_cars",
    "data/processed/detected_plates",
    "data/processed/annotated",
    "data/database",
    "models/pretrained",
    "output/fines",
    "output/logs",
    "output/reports"
]

for directory in directories:
    ensure_folder(directory)
    print(f"  ✅ {directory}")

# Step 2: Create sample database if not exists
print("\n📊 Step 2: Setting up database...")
db_file = Path("data/database/vehicle_owner_db.csv")
if not db_file.exists():
    try:
        import pandas as pd
        owner_data = {
            'plate_number': ['RJ14CV0002'],
            'owner_name': ['John Doe', 'Jane Smith', 'Raj Kumar', 'Priya Singh', 'Amit Patel'],
            'address': ['123 Main St, Mumbai', '456 Park Ave, Delhi', '789 Lake Rd, Bangalore', '321 Beach Rd, Chennai', '654 River St, Ahmedabad'],
            'city': ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Ahmedabad'],
            'state': ['MH', 'DL', 'KA', 'TN', 'GJ'],
            'pincode': ['400001', '110001', '560001', '600001', '380001'],
            'phone': ['9876543210', '8765432109', '7654321098', '6543210987', '5432109876'],
            'email': ['john@email.com', 'jane@email.com', 'raj@email.com', 'priya@email.com', 'amit@email.com'],
            'vehicle_model': ['Sedan', 'SUV', 'Hatchback', 'Motorcycle', 'SUV'],
            'registration_date': ['2020-01-15', '2021-03-20', '2022-06-10', '2023-02-28', '2024-01-05']
        }
        pd.DataFrame(owner_data).to_csv(db_file, index=False)
        print("  ✅ Created vehicle_owner_db.csv")
    except ImportError:
        print("  ⚠️  pandas not installed, skipping database creation")
else:
    print("  ✅ vehicle_owner_db.csv exists")

# Step 3: Create violation logs if not exists
log_file = Path("data/database/violation_logs.csv")
if not log_file.exists():
    try:
        import pandas as pd
        violation_data = {
            'fine_id': [],
            'timestamp': [],
            'plate_number': [],
            'violation_type': [],
            'location': [],
            'fine_amount': [],
            'status': [],
            'owner_name': [],
            'owner_address': [],
            'evidence_image': []
        }
        pd.DataFrame(violation_data).to_csv(log_file, index=False)
        print("  ✅ Created violation_logs.csv")
    except ImportError:
        print("  ⚠️  pandas not installed, skipping violation logs creation")
else:
    print("  ✅ violation_logs.csv exists")

# Step 4: Create multiple test images
print("\n🎨 Step 3: Creating test images...")

try:
    # Test Image 1: Simple text image
    img1 = np.ones((400, 600, 3), dtype=np.uint8) * 255
    cv2.putText(img1, 'MH12AB1234', (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    cv2.rectangle(img1, (140, 170), (450, 230), (0, 255, 0), 2)
    cv2.imwrite('data/raw/images/test_plate_1.jpg', img1)
    print("  ✅ Created test_plate_1.jpg (simple text)")

    # Test Image 2: Realistic car with plate
    img2 = np.ones((600, 900, 3), dtype=np.uint8) * 200
    # Car body
    cv2.rectangle(img2, (150, 250), (750, 500), (100, 100, 200), -1)
    # Windows
    cv2.rectangle(img2, (200, 200), (350, 250), (150, 150, 150), -1)
    cv2.rectangle(img2, (550, 200), (700, 250), (150, 150, 150), -1)
    # Wheels
    cv2.circle(img2, (250, 500), 50, (50, 50, 50), -1)
    cv2.circle(img2, (650, 500), 50, (50, 50, 50), -1)
    # License plate
    cv2.rectangle(img2, (350, 400), (550, 450), (255, 255, 255), -1)
    cv2.rectangle(img2, (350, 400), (550, 450), (0, 0, 0), 2)
    cv2.putText(img2, 'MH12AB1234', (370, 435), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.imwrite('data/raw/images/test_plate_2.jpg', img2)
    print("  ✅ Created test_plate_2.jpg (realistic car)")

    # Test Image 3: Different plate format
    img3 = np.ones((400, 600, 3), dtype=np.uint8) * 240
    cv2.putText(img3, 'DL5CQ5678', (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    cv2.rectangle(img3, (140, 170), (450, 230), (255, 0, 0), 2)
    cv2.imwrite('data/raw/images/test_plate_3.jpg', img3)
    print("  ✅ Created test_plate_3.jpg (different plate)")

except Exception as e:
    print(f"  ⚠️  Error creating test images: {e}")

# Step 5: Check if model exists
print("\n🤖 Step 4: Checking YOLO model...")
model_path = Path("models/pretrained/yolov8n.pt")
if model_path.exists():
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"  ✅ Model found: {model_path.name} ({size_mb:.2f} MB)")
else:
    print("  ⚠️  Model not found. Checking alternative locations...")
    
    # Check if model exists in root
    if Path("yolov8n.pt").exists():
        print("  ✅ Found model in root directory. Moving to pretrained folder...")
        import shutil
        shutil.move("yolov8n.pt", "models/pretrained/yolov8n.pt")
        print("  ✅ Model moved successfully")
    else:
        print("  ⚠️  Model not found. Downloading...")
        try:
            from ultralytics import YOLO
            model = YOLO('yolov8n.pt')
            import shutil
            shutil.move('yolov8n.pt', 'models/pretrained/yolov8n.pt')
            print("  ✅ Model downloaded successfully")
        except Exception as e:
            print(f"  ❌ Failed to download model: {e}")

# Step 6: Test all images
print("\n🚀 Step 5: Running tests with all images...")
print("-" * 40)

test_images = [
    "data/raw/images/test_plate_1.jpg",
    "data/raw/images/test_plate_2.jpg",
    "data/raw/images/test_plate_3.jpg"
]

for img_path in test_images:
    print(f"\n📸 Testing: {img_path}")
    print("-" * 20)
    
    # Check if image exists
    if not Path(img_path).exists():
        print(f"  ❌ Image not found: {img_path}")
        continue
    
    # Run the ANPR system
    cmd = f'python main.py --image "{img_path}" --violation speeding --location "Test Location"'
    exit_code = os.system(cmd)
    if exit_code != 0:
        print(f"  ⚠️  Command exited with code {exit_code}")
    print("-" * 20)

print("\n" + "=" * 60)
print("✅ TEST COMPLETE!")
print("📁 Check these folders for results:")
print("  - output/processed/annotated/     (images with boxes)")
print("  - output/processed/detected_plates/ (cropped plates)")
print("  - data/database/violation_logs.csv  (fine records)")
print("=" * 60)

# Step 7: Show summary of fines
print("\n📊 Fine Summary:")
try:
    if log_file.exists():
        import pandas as pd
        df = pd.read_csv(log_file)
        if len(df) > 0:
            print("\nFines issued:")
            print(df.to_string(index=False))
        else:
            print("  No fines issued yet. The test images may not contain detectable license plates.")
            print("  This is normal - YOLO is trained on real photos, not simple drawings.")
    else:
        print("  No violation logs found.")
except Exception as e:
    print(f"  ⚠️  Could not read violation logs: {e}")

print("\n" + "=" * 60)
print("💡 TIPS:")
print("  • If no plates were detected, try using real photos of cars")
print("  • Place any car photo in data/raw/images/ and run:")
print("  • python main.py --image data/raw/images/your_photo.jpg --violation speeding")
print("=" * 60)