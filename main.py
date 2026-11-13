 import cv2
import numpy as np
from ultralytics import YOLO
import logging
import threading
import pygame

# -------------------------------
# 🔊 INIT SOUND SYSTEM (STABLE)
# -------------------------------
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("alarm.wav")  # use WAV file

# Hide YOLO logs
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Load pose model
model = YOLO("yolov8n-pose.pt")

cap = cv2.VideoCapture(0)

track_history = {}

SPEED_THRESHOLD = 40
FRAME_THRESHOLD = 6

# Alarm control
alarm_on = False

def play_alarm():
    if not pygame.mixer.get_busy():
        alarm_sound.play()


while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    suspicious_flag = False
    for r in results:
        if r.keypoints is None:
            continue

        keypoints = r.keypoints.xy.cpu().numpy()

        for i, person in enumerate(keypoints):

            # -------------------------------
            # REMOVE INVALID KEYPOINTS
            # -------------------------------
            valid_points = person[~np.isnan(person).any(axis=1)]

            if len(valid_points) < 5:
                continue

            # -------------------------------
            # CENTER POINT
            # -------------------------------
            cx = int(np.mean(valid_points[:, 0]))
            cy = int(np.mean(valid_points[:, 1]))

            # -------------------------------
            # INIT TRACKING
            # -------------------------------
            if i not in track_history:
                track_history[i] = {
                    "positions": [],
                    "suspicious_frames": 0,
                    "fall_frames": 0,
                    "prev_y": cy
                }
            track_history[i]["positions"].append((cx, cy))
            if len(track_history[i]["positions"]) > 10:
                track_history[i]["positions"].pop(0)

            positions = track_history[i]["positions"]

            # -------------------------------
            # SPEED ANALYSIS
            # -------------------------------
            if len(positions) >= 2:
                distances = []
                for j in range(1, len(positions)):
                    d = np.sqrt(
                        (positions[j][0] - positions[j-1][0])**2 +
                        (positions[j][1] - positions[j-1][1])**2
                    )
                    distances.append(d)

                avg_speed = np.mean(distances)
            else:
                avg_speed = 0

            # -------------------------------
            # FALL DETECTION
            # -------------------------------
            prev_y = track_history[i]["prev_y"]
            dy = prev_y - cy
            track_history[i]["prev_y"] = cy

            x_coords = valid_points[:, 0]
            y_coords = valid_points[:, 1]

            width = np.max(x_coords) - np.min(x_coords)
            height = np.max(y_coords) - np.min(y_coords)

            is_horizontal = width > height
            is_fast_drop = dy > 20

            if is_horizontal and is_fast_drop:
                track_history[i]["fall_frames"] += 1
            else:
                track_history[i]["fall_frames"] = 0

            fall_detected = track_history[i]["fall_frames"] >= 3

            # -------------------------------
            # SUSPICIOUS LOGIC
            # -------------------------------
            if avg_speed > SPEED_THRESHOLD or fall_detected:
                track_history[i]["suspicious_frames"] += 1
            else:
                track_history[i]["suspicious_frames"] = 0

            if track_history[i]["suspicious_frames"] >= FRAME_THRESHOLD:
                suspicious_flag = True
                color = (0, 0, 255)

                if fall_detected:
                    text = "FALL DETECTED!"
                else:
                    text = "SUSPICIOUS ACTIVITY"
            else:
                color = (0, 255, 0)
                text = "Normal"

            # -------------------------------
            # DRAW OUTPUT
            # -------------------------------
            cv2.circle(frame, (cx, cy), 5, color, -1)
            cv2.putText(frame, text, (cx, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # -------------------------------
    # 🔊 TERMINAL + ALARM OUTPUT
    # -------------------------------
    if suspicious_flag:
        print("🚨 Suspicious Activity Detected")

        if not alarm_on:
            threading.Thread(target=play_alarm, daemon=True).start()
            alarm_on = True
    else:
        print("Normal")
        alarm_on = False
        pygame.mixer.stop()  # stop alarm when safe

    cv2.imshow("Final Surveillance System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
