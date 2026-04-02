import os
import cv2
import easyocr
import pandas as pd
import re
from difflib import get_close_matches
from flask import Flask, render_template, request

app = Flask(__name__)

# Load OCR once
reader = easyocr.Reader(['en'])

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DB_PATH = os.path.join(BASE_DIR, "..", "data", "database", "vehicle_owner_db.csv")

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# -------------------------------
# SMART CORRECTION FUNCTION
# -------------------------------
def smart_correct_plate(text):
    text = list(text)

    for i in range(len(text)):
        ch = text[i]

        if i < 2:
            if ch.isdigit():
                text[i] = chr(ord('A') + int(ch) % 26)

        elif i < 4:
            if ch.isalpha():
                text[i] = '1' if ch in ['I', 'L'] else '0'

        elif 4 <= i < 6:
            if ch.isdigit():
                text[i] = chr(ord('A') + int(ch) % 26)

        else:
            if ch.isalpha():
                if ch in ['O', 'U']:
                    text[i] = '0'
                elif ch in ['I', 'L']:
                    text[i] = '1'
                elif ch == 'Z':
                    text[i] = '2'

    return "".join(text)


# -------------------------------
# OCR FUNCTION
# -------------------------------
def detect_plate(image_path):
    img = cv2.imread(image_path)

    if img is None:
        return ""

    h, w, _ = img.shape

    cropped = img[int(h * 0.5):int(h * 0.85), int(w * 0.2):int(w * 0.9)]

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

    results = reader.readtext(thresh)

    text = ""
    for res in results:
        text += res[1]

    text = re.sub(r'[^A-Z0-9]', '', text.upper())

    return text


# -------------------------------
# FINE LOGIC
# -------------------------------
def calculate_fine(vehicle_type, speed, no_parking, helmet, seatbelt):
    violations = []
    fine = 0

    if speed > 60:
        violations.append("Speeding")
        fine += 1000

    if no_parking:
        violations.append("No Parking")
        fine += 500

    if vehicle_type == "motorcycle":
        if not helmet:
            violations.append("No Helmet")
            fine += 800

    elif vehicle_type in ["sedan", "suv", "hatchback"]:
        if not seatbelt:
            violations.append("No Seatbelt")
            fine += 500

    return violations, fine


# -------------------------------
# ROUTE
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def index():

    # ✅ FIX 1: Reset values (prevents auto display)
    owner = None
    vehicle = None
    plate = None
    violations = None
    fine = None
    error = None

    if request.method == "POST":

        # ✅ FIX 2: Safe file handling
        file = request.files.get('image')

        if not file or file.filename == "":
            return render_template("index.html", error="Please upload an image")

        # ✅ FIX 3: Safe inputs
        try:
            speed = int(request.form.get("speed", 0))
        except:
            return render_template("index.html", error="Invalid speed input")

        no_parking = request.form.get("no_parking") == "yes"
        helmet = request.form.get("helmet") == "yes"
        seatbelt = request.form.get("seatbelt") == "yes"

        # Save image
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # OCR
        raw_plate = detect_plate(filepath)
        detected_plate = smart_correct_plate(raw_plate)

        print("RAW:", raw_plate)
        print("CORRECTED:", detected_plate)

        if not detected_plate:
            return render_template("index.html", error="Plate not detected")

        # Load DB
        df = pd.read_csv(DB_PATH)
        df.columns = df.columns.str.strip()
        df['plate_number'] = df['plate_number'].str.upper().str.strip()

        # -------------------------------
        # FUZZY MATCH
        # -------------------------------
        plate_list = df['plate_number'].tolist()

        best_match = get_close_matches(detected_plate, plate_list, n=1, cutoff=0.6)

        if not best_match:
            return render_template("index.html", error="Plate not found")

        matched_plate = best_match[0]

        match = df[df['plate_number'] == matched_plate]

        owner = match.iloc[0]
        vehicle_type = owner['vehicle_model'].lower()

        # Fine logic
        violations, fine = calculate_fine(
            vehicle_type, speed, no_parking, helmet, seatbelt
        )

        return render_template(
            "index.html",
            plate=matched_plate,
            owner=owner['owner_name'],
            vehicle=owner['vehicle_model'],
            speed=speed,  
            violations=", ".join(violations) if violations else "No violation",
            fine=fine
        )

    # ✅ GET request → empty UI
    return render_template("index.html")
    

# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)