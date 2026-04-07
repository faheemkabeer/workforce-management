from flask import Flask, request, jsonify, send_from_directory
import os
import cv2
import re
from ultralytics import YOLO
import easyocr
import sqlite3
from collections import Counter
from datetime import datetime

# =========================
# FLASK SETUP
# =========================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# =========================
# DATABASE SETUP
# =========================
DB_PATH = "database/fines.db"
os.makedirs("database", exist_ok=True)

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS fines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_no TEXT,
    timestamp TEXT,
    violation_type TEXT,
    fine_amount INTEGER,
    payment_status TEXT
)
""")
conn.commit()

# =========================
# MODEL SETUP
# =========================
print("Loading YOLO and EasyOCR models...")
model = YOLO(r"C:\Users\FAHEEM\Desktop\datasets\runs\detect\train\weights\best.pt")
reader = easyocr.Reader(['en'], gpu=True)
print("Models loaded successfully.")

# =========================
# HELPER FUNCTIONS
# =========================
def clean_text(text):
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)
    text = text.replace('O', '0')
    text = text.replace('I', '1')
    text = text.replace('Z', '2')
    text = text.replace('S', '5')
    return text

def calculate_fine(plate):
    violations = {
        "overspeed": 1000,
        "no_helmet": 500,
        "signal_jump": 2000,
        "unauthorized_vehicle": 1500
    }
    violation_type = "overspeed"  # Default
    fine = violations[violation_type]
    return violation_type, fine

# =========================
# IMAGE PROCESSING
# =========================
def process_image(image_path):
    frame = cv2.imread(image_path)

    plate_counter = Counter()
    results_list = []

    results = model(frame, conf=0.4)

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        plate_img = frame[y1:y2, x1:x2]

        # Improve OCR accuracy
        plate_img = cv2.resize(plate_img, None, fx=2, fy=2)

        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        ocr_results = reader.readtext(gray)

        merged_text = ""
        for _, text, conf in ocr_results:
            merged_text += clean_text(text)

        plate_pattern = r"[A-Z]{2}\d{2}[A-Z]{2}\d{4}"
        match = re.search(plate_pattern, merged_text)

        if match:
            plate = match.group()
            plate_counter[plate] += 1

            violation_type, fine_amount = calculate_fine(plate)

            cursor.execute("""
                INSERT INTO fines (plate_no, timestamp, violation_type, fine_amount, payment_status)
                VALUES (?, ?, ?, ?, ?)
            """, (
                plate,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                violation_type,
                fine_amount,
                "Pending"
            ))
            conn.commit()

            results_list.append({
                "number": plate,
                "time": datetime.now().strftime("%H:%M:%S"),
                "violation": violation_type,
                "fine": fine_amount
            })

    return results_list

# =========================
# VIDEO PROCESSING
# =========================
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError("Video not found or cannot be opened")

    plate_counter = Counter()
    results_list = []

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        if frame_id % 3 != 0:
            continue  # Skip frames

        results = model(frame, conf=0.4)

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_img = frame[y1:y2, x1:x2]

            plate_img = cv2.resize(plate_img, None, fx=2, fy=2)

            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            ocr_results = reader.readtext(gray)

            merged_text = ""
            for _, text, conf in ocr_results:
                merged_text += clean_text(text)

            plate_pattern = r"[A-Z]{2}\d{2}[A-Z]{2}\d{4}"
            match = re.search(plate_pattern, merged_text)

            if match:
                plate = match.group()
                plate_counter[plate] += 1

                # Avoid duplicate spam
                if plate_counter[plate] < 3:
                    continue

                violation_type, fine_amount = calculate_fine(plate)

                cursor.execute("""
                    INSERT INTO fines (plate_no, timestamp, violation_type, fine_amount, payment_status)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    plate,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    violation_type,
                    fine_amount,
                    "Pending"
                ))
                conn.commit()

                results_list.append({
                    "number": plate,
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "violation": violation_type,
                    "fine": fine_amount
                })

    cap.release()
    return results_list

# =========================
# ROUTES
# =========================
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/anpr', methods=['POST'])
def anpr():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    save_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(save_path)

    try:
        filename = file.filename.lower()

        if filename.endswith(('.mp4', '.avi', '.mov')):
            results = process_video(save_path)

        elif filename.endswith(('.jpg', '.jpeg', '.png')):
            results = process_image(save_path)

        else:
            return jsonify({"error": "Unsupported file format"}), 400

        return jsonify({"plates": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================
# RUN APP
# =========================
if __name__ == '__main__':
    app.run(debug=True)