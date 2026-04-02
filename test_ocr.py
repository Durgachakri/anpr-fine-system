import easyocr
import cv2
import re

def detect_plate(image_path):
    reader = easyocr.Reader(['en'])

    img = cv2.imread(image_path)

    h, w, _ = img.shape

    # better crop (tuned for your image)
    cropped = img[int(h*0.6):int(h*0.85), int(w*0.2):int(w*0.95)]

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    # improve contrast
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

    result = reader.readtext(thresh)

    text = ""
    for detection in result:
        text += detection[1]

    clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())

    print("OCR RAW:", text)
    print("OCR CLEAN:", clean_text)

    return clean_text