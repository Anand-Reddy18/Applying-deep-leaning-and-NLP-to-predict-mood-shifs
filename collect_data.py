import cv2
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime, timedelta

try:
    from tensorflow.keras.models import load_model
except ImportError:
    print("TensorFlow not found. Please install tensorflow.")
    exit()

# CONFIGURATION
FILE_NAME = "mood_dataset.csv"
ANALYZE_INTERVAL = 2   # seconds

EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
MOOD_COLORS = {
    'happy': (0, 255, 0),       
    'sad': (255, 0, 0),         
    'angry': (0, 0, 255),       
    'neutral': (200, 200, 200), 
    'surprise': (0, 165, 255),  
    'fear': (128, 0, 128),      
    'disgust': (0, 100, 0),
    'sleep': (150, 150, 255),   # Pinkish for sleep
    'away': (100, 100, 100)
}

# --- DATASET & GAP LOGIC ---
unique_hours = set()
last_recorded_time = None
pending_gap = False

if not os.path.exists(FILE_NAME):
    df = pd.DataFrame(columns=["timestamp", "hour", "emotion", "confidence"])
    df.to_csv(FILE_NAME, index=False)
    print("Dataset file created.")
else:
    try:
        existing_df = pd.read_csv(FILE_NAME)
        if not existing_df.empty and "timestamp" in existing_df.columns:
            existing_df["timestamp"] = pd.to_datetime(existing_df["timestamp"])
            unique_hours = set(existing_df["timestamp"].dt.floor('h').astype(str))
            
            last_recorded_time = existing_df["timestamp"].max()
            
            if last_recorded_time is not pd.NaT:
                time_diff = datetime.now() - last_recorded_time
                if time_diff.total_seconds() > 3600:
                    pending_gap = True
                    print(f"Gap detected: {time_diff.total_seconds()/3600:.1f} hours.")
    except Exception as e:
        print("Error reading existing dataset:", e)

# LOAD MODEL
if not os.path.exists("emotion_model.h5"):
    print("emotion_model.h5 not found! Please run train_vision_model.py first.")
    exit()

print("Loading custom emotion model...")
emotion_model = load_model("emotion_model.h5")

# OPENCV DETECTORS (Bulletproof fallback)
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_detector = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")

cap = cv2.VideoCapture(0)

def draw_styled_box(frame, x, y, w, h, color, thickness=2, l=30):
    cv2.line(frame, (x, y), (x + l, y), color, thickness)
    cv2.line(frame, (x, y), (x, y + l), color, thickness)
    cv2.line(frame, (x + w, y), (x + w - l, y), color, thickness)
    cv2.line(frame, (x + w, y), (x + w, y + l), color, thickness)
    cv2.line(frame, (x, y + h), (x + l, y + h), color, thickness)
    cv2.line(frame, (x, y + h), (x, y + h - l), color, thickness)
    cv2.line(frame, (x + w, y + h), (x + w - l, y + h), color, thickness)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - l), color, thickness)
    
def draw_text_panel(frame, text, x, y, bg_color, text_color=(255, 255, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    padding = 10
    box_x_min = x
    box_y_min = y - text_height - padding
    box_x_max = x + text_width + 2 * padding
    box_y_max = y + padding
    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x_min, box_y_min), (box_x_max, box_y_max), bg_color, -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, text, (x + padding + 2, y - 2 + 2), font, font_scale, (0, 0, 0), thickness)
    cv2.putText(frame, text, (x + padding, y - 2), font, font_scale, text_color, thickness)


last_analysis_time = 0
current_emotion = "neutral"
current_confidence = 0.0
closed_eyes_frames = 0
SLEEP_FRAMES_REQUIRED = 30

def fill_gap(gap_start, gap_end, fill_emotion):
    print(f"Filling gap between {gap_start} and {gap_end} with {fill_emotion}...")
    start_hour = gap_start.replace(minute=0, second=0, microsecond=0)
    end_hour = gap_end.replace(minute=0, second=0, microsecond=0)
    curr = start_hour + timedelta(hours=1)
    new_rows = []
    
    while curr < end_hour:
        new_rows.append({
            "timestamp": curr.strftime("%Y-%m-%d %H:%M:%S"),
            "hour": curr.hour,
            "emotion": fill_emotion,
            "confidence": 100.0
        })
        unique_hours.add(curr.strftime("%Y-%m-%d %H:%M:%S"))
        curr += timedelta(hours=1)
    
    if new_rows:
        pd.DataFrame(new_rows).to_csv(FILE_NAME, mode="a", header=False, index=False)
        print(f"Added {len(new_rows)} hours of '{fill_emotion}'.")

print("\nPress 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    img_h, img_w, _ = frame.shape

    # -- GAP POPUP LOGIC --
    if pending_gap:
        draw_text_panel(frame, "GAP DETECTED > 1HR", 50, img_h//2 - 40, (0,0,255))
        draw_text_panel(frame, "Were you sleeping? (Y/N)", 50, img_h//2 + 20, (50,50,50))
        cv2.imshow("Mood Data Collector HUD", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('y') or key == ord('Y'):
            fill_gap(last_recorded_time, datetime.now(), "sleep")
            pending_gap = False
        elif key == ord('n') or key == ord('N'):
            fill_gap(last_recorded_time, datetime.now(), "away")
            pending_gap = False
        elif key == ord('q'):
            break
        continue # Skip processing until answered

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    is_sleeping = False
    
    # -- BULLETPROOF SLEEP DETECTION (OpenCV Eyes) --
    if len(faces) > 0:
        x, y, w, h = faces[0]
        roi_eyes = gray[y:y+int(h/2), x:x+w] # Only search upper half of face
        eyes = eye_detector.detectMultiScale(roi_eyes, scaleFactor=1.1, minNeighbors=3, minSize=(10, 10))
        
        if len(eyes) == 0:
            closed_eyes_frames += 1
        else:
            closed_eyes_frames = 0
            
        if closed_eyes_frames >= SLEEP_FRAMES_REQUIRED:
            is_sleeping = True

    # -- EMOTION DETECTION --
    if time.time() - last_analysis_time > ANALYZE_INTERVAL:
        if is_sleeping:
            current_emotion = "sleep"
            current_confidence = 100.0
            now = datetime.now()
            row = pd.DataFrame([{"timestamp": now.strftime("%Y-%m-%d %H:%M:%S"), "hour": now.hour, "emotion": current_emotion, "confidence": current_confidence}])
            row.to_csv(FILE_NAME, mode="a", header=False, index=False)
            unique_hours.add(now.replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%d %H:%M:%S"))
            print(f"Saved -> SLEEP (Eyes Closed)")
            last_analysis_time = time.time()
        elif len(faces) > 0:
            x, y, w, h = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            try:
                face_resized = cv2.resize(face_roi, (48, 48))
                face_input = np.reshape(face_resized / 255.0, (1, 48, 48, 1))
                pred = emotion_model.predict(face_input, verbose=0)
                max_index = np.argmax(pred[0])
                current_emotion = EMOTION_LABELS[max_index]
                current_confidence = float(pred[0][max_index]) * 100
                
                now = datetime.now()
                row = pd.DataFrame([{"timestamp": now.strftime("%Y-%m-%d %H:%M:%S"), "hour": now.hour, "emotion": current_emotion, "confidence": current_confidence}])
                row.to_csv(FILE_NAME, mode="a", header=False, index=False)
                unique_hours.add(now.replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%d %H:%M:%S"))
                
                print(f"Saved -> {current_emotion.upper()} | {current_confidence:.2f}%")
                last_analysis_time = time.time()
            except Exception as e:
                pass

    # -- DRAW UI --
    for (x, y, w, h) in faces:
        color = MOOD_COLORS.get(current_emotion, (0, 255, 0))
        draw_styled_box(frame, x, y, w, h, color, thickness=2)

    ui_color = MOOD_COLORS.get(current_emotion, (50, 50, 50))
    panel_color = (max(0, ui_color[0]-60), max(0, ui_color[1]-60), max(0, ui_color[2]-60))
    
    draw_text_panel(frame, f"Mood: {current_emotion.upper()}", 20, 50, panel_color)
    draw_text_panel(frame, f"Conf: {current_confidence:.1f}%", 20, 100, (50, 50, 50))
    
    hours_count = len(unique_hours)
    progress_color = (0, 255, 0) if hours_count >= 24 else (0, 165, 255)
    draw_text_panel(frame, f"Hours Data: {hours_count}/24", 20, 150, (50, 50, 50), text_color=progress_color)

    cv2.imshow("Mood Data Collector HUD", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Data collection stopped.")