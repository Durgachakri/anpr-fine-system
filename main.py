#!/usr/bin/env python
"""
Automatic Number Plate Recognition with Fine Issuance System
Main entry point for the application
"""
import argparse
import sys
from pathlib import Path
import logging
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config.settings import config, BASE_DIR
from src.anpr_pipeline import ANPRFineSystem
from src.utils.visualization import Visualizer

def setup_logging():
    """Configure logging"""
    log_path = BASE_DIR / 'output' / 'logs' / f"anpr_{datetime.now().strftime('%Y%m%d')}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='ANPR System with Fine Issuance')
    
    # Input options
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--video', type=str, help='Path to input video')
    parser.add_argument('--camera', type=int, help='Camera device ID (0, 1, etc.)')
    
    # Processing options
    parser.add_argument('--violation', type=str, choices=['speeding', 'red_light', 'parking', 'toll_evasion'],
                       help='Type of violation to simulate')
    parser.add_argument('--location', type=str, default='Unknown', help='Location name')
    parser.add_argument('--output', type=str, help='Output path for processed media')
    
    # Report options
    parser.add_argument('--report', type=str, help='Generate report for date (YYYY-MM-DD)')
    parser.add_argument('--dashboard', action='store_true', help='Launch web dashboard')
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Initialize system
    logger.info("Starting ANPR Fine System")
    anpr = ANPRFineSystem(config)
    
    # Process based on arguments
    if args.report:
        # Generate report
        report = anpr.generate_report(args.report)
        print(f"\n📊 Report for {args.report}")
        print(f"Total Fines: {report['total_fines']}")
        print(f"Total Amount: ₹{report['total_amount']:,}")
        print(f"Average Fine: ₹{report['summary']['average_fine']:,.2f}")
        print("\nViolation Breakdown:")
        for vtype, count in report['violation_breakdown'].items():
            print(f"  {vtype}: {count}")
    
    elif args.image:
        # Process single image
        logger.info(f"Processing image: {args.image}")
        results = anpr.process_image(
            args.image,
            violation_type=args.violation,
            location=args.location
        )
        
        # Display results
        print(f"\n📸 Processing Results for {args.image}")
        print("=" * 50)
        
        if results['status'] == 'no_detection':
            print("❌ No license plates detected")
        else:
            for plate in results['plates']:
                print(f"\n🚗 Plate: {plate['plate_number']}")
                print(f"   Confidence: {plate['ocr_confidence']:.2%}")
                if plate['owner']:
                    print(f"   Owner: {plate['owner']['owner_name']}")
                    print(f"   Vehicle: {plate['owner']['vehicle_model']}")
                if plate['fine_issued']:
                    print(f"   ⚠️  VIOLATION: {plate['violation']}")
                    print(f"   💰 Fine: ₹{plate['fine_amount']:,}")
            
            print(f"\n📊 Summary: {results['total_fines']} fines issued, Total: ₹{results['total_amount']:,}")
    
    elif args.video:
        # Process video
        logger.info(f"Processing video: {args.video}")
        
        output_path = args.output or str(BASE_DIR / 'output' / 'processed' / f"processed_{Path(args.video).name}")
        
        violations = anpr.process_video(
            args.video,
            output_video=output_path
        )
        
        # Display results
        print(f"\n🎥 Video Processing Complete")
        print(f"Output saved to: {output_path}")
        print(f"\n📊 Detected Violations: {len(violations)}")
        
        for v in violations:
            print(f"  • {v['plate']} - {v['violation']} - ₹{v['amount']:,} at {v['timestamp']:.1f}s")
    
    elif args.camera is not None:
        # Process from camera
        logger.info(f"Starting camera {args.camera}")
        
        import cv2
        cap = cv2.VideoCapture(args.camera)
        
        print("🎥 Camera Mode - Press 'q' to quit, 's' to save frame")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            detections = anpr.detector.detect(frame)
            
            for detection in detections:
                plate_crop = detection['crop']
                plate_text, conf = anpr.ocr.read_plate(plate_crop)
                
                if plate_text:
                    formatted = anpr.postprocessor.process(plate_text)
                    if formatted:
                        # Annotate frame
                        x1, y1, x2, y2 = detection['bbox']
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, formatted, (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('ANPR Camera', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save frame
                save_path = BASE_DIR / 'output' / 'captures' / f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                save_path.parent.mkdir(exist_ok=True)
                cv2.imwrite(str(save_path), frame)
                print(f"📸 Saved to {save_path}")
        
        cap.release()
        cv2.destroyAllWindows()
    
    elif args.dashboard:
        # Launch web dashboard
        print("🌐 Launching web dashboard...")
        from src.web_dashboard.app import run_dashboard
        run_dashboard()
    
    else:
        # No arguments, show help
        parser.print_help()
        print("\n💡 Example usage:")
        print("  python main.py --image data/raw/images/car.jpg --violation speeding --location 'Main Street'")
        print("  python main.py --video data/raw/videos/traffic.mp4 --output output.mp4")
        print("  python main.py --report 2024-01-15")
        print("  python main.py --camera 0")

if __name__ == "__main__":
    main()