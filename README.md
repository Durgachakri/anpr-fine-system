# 🚗 Automatic Number Plate Recognition (ANPR) with Fine Issuance System

A complete ANPR system that detects license plates, reads them using OCR, and simulates traffic fine issuance.

## ✨ Features

- **License Plate Detection**: YOLOv8-based detection
- **OCR Reading**: EasyOCR integration with preprocessing
- **Owner Database**: Mock vehicle registration database
- **Fine Issuance**: Automated fine calculation and logging
- **Multiple Input Sources**: Images, videos, and live camera
- **Violation Types**: Speeding, red light, parking, toll evasion
- **Reporting**: Daily violation reports and statistics
- **Visualization**: Annotated images and fine notices

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <your-repo-url>
cd anpr-fine-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt