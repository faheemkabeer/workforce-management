import cv2
import re
import sqlite3
from ultralytics import YOLO
import easyocr
from collections import Counter
from datetime import datetime
import os

# =========================
# CLEAN OCR TEXT
# =========================
def clean_text(text):
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)
    text = text.replace('O', '0')
    text = text.replace('I', '1')
    text = text.replace('Z', '2')
    text = text.replace('S', '5')
    return text

# =========================
# DATABASE SETUP
# =========================
os.makedirs("database", exist_ok=True)
conn = sqlite3.connect("database/fines.db", check_same_thread=False)
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
# FINE LOGIC
# =========================
def calculate_fine(plate):
    # Default overspeed fine (can be extended)
    violations = {
        "overspeed": 1000,
        "no_helmet": 500,
        "signal_jump": 2000,
        "unauthorized_vehicle": 1500
    }
    violation_type = "overspeed"
    fine = violations[violation_type]
    return violation_type, fine

# =========================
# LOAD MODELS
# =========================
print("Loading models...")
model = YOLO(r"C:\Users\FAHEEM\Desktop\datasets\runs\detect\train\weights\best.pt")
reader = easyocr.Reader(['en'], gpu=True)
print("Models loaded successfully.")

# =========================
# VIDEO INPUT
# =========================
video_path = r"C:\Users\FAHEEM\Desktop\datasets\Automatic Number Plate Recognition (ANPR) _ Vehicle Number Plate Recognition (1).mp4"

print("Input video path:")
print(video_path)

# Strict Indian plate regex
plate_pattern = r"[A-Z]{2}\d{2}[A-Z]{2}\d{4}"

# =========================
# VIDEO PROCESSING
# =========================
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError("❌ Video file not found or cannot be opened")

    print("\n--- VIDEO ANPR STARTED ---")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(
        "output_video_anpr.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    frame_id = 0
    detected_plates = set()  # Track plates already fined

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        # Process every 3rd frame
        if frame_id % 3 != 0:
            out.write(frame)
            continue

        results = model(frame, conf=0.4)

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_img = frame[y1:y2, x1:x2]

            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            ocr_results = reader.readtext(gray)
            merged_text = "".join([clean_text(text) for _, text, _ in ocr_results])
            match = re.search(plate_pattern, merged_text)

            if match:
                plate = match.group()

                if plate not in detected_plates:  # Insert only once
                    detected_plates.add(plate)

                    violation_type, fine_amount = calculate_fine(plate)

                    cursor.execute("""
                    INSERT INTO fines (plate_no, timestamp, violation_type, fine_amount, payment_status)
                    VALUES (?, ?, ?, ?, ?)
                    """, (plate, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                          violation_type, fine_amount, "Pending"))
                    conn.commit()

                    print(f"🚨 Fine Added → {plate} | Violation: {violation_type} | ₹{fine_amount}")

                # Draw box on frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame, plate,
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2
                )

        out.write(frame)

    cap.release()
    out.release()
    print("\n🎥 Video saved as output_video_anpr.mp4")

# =========================
# RUN VIDEO PROCESSING
# =========================
print("\nStarting video processing...")
process_video(video_path)

# =========================
# FINAL RESULT
# =========================
print("\n--- DATABASE RECORDS (Unique Plates Only) ---")
cursor.execute("SELECT * FROM fines")
for row in cursor.fetchall():
    print(row)

print("\nProcessing complete.")
