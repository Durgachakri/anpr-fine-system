# 🚗 ANPR Fine Issuance System

An Automatic Number Plate Recognition system that detects license 
plates from images, videos, and live camera feeds, and automates 
traffic fine issuance with a web dashboard.

## 🛠️ Tech Stack

- **Language:** Python
- **Computer Vision:** OpenCV
- **Web Framework:** Flask
- **Data Processing:** NumPy, Pandas
- **Frontend:** HTML, CSS

## ✨ Features

- 📸 License plate detection from images, videos, and live camera
- ⚠️ Violation detection — speeding, red light, parking, toll evasion
- 💰 Automated fine calculation and issuance
- 🗃️ Mock vehicle owner database with registration lookup
- 📊 Daily violation reports with statistics and breakdowns
- 🌐 Flask web dashboard for viewing violations and reports

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/Durgachakri/anpr-fine-system.git
cd anpr-fine-system

# Create virtual environment
python -m venv venv
source venv/bin/activate     # Linux/Mac
venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt
```

## 💻 Usage

```bash
# Process an image
python main.py --image car.jpg --violation speeding --location "Main Street"

# Process a video
python main.py --video traffic.mp4 --output output.mp4

# Live camera
python main.py --camera 0

# Generate report
python main.py --report 2024-01-15

# Launch web dashboard
python main.py --dashboard
```
