import cv2
import mediapipe as mp
import json
import os
from collections import deque
import time
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)


gestures = ["swipe_up", "swipe_down", "swipe_left", "swipe_right", 
            "pointing", "open_palm", "thumb_index_touch", "fist"]

dynamic = ["swipe_up", "swipe_down", "swipe_left", "swipe_right"]
static = ["pointing", "open_palm", "thumb_index_touch", "fist"]
dataset_path = "datasets/gesture_dataset/static/val"

for gesture in gestures:
    os.makedirs(os.path.join(dataset_path, gesture), exist_ok=True)

sequence_length = 30
buffer = deque(maxlen=sequence_length)

def save_landmarks(gesture_name, landmarks):
    gesture_dir = os.path.join(dataset_path, gesture_name)
    timestamp = int(time.time())
    filename = f"{gesture_name}_{timestamp}.json"
    filepath = os.path.join(gesture_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(landmarks, f)
    print(f"Saved {filepath}")

def calculate_distance(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

cap = cv2.VideoCapture(1)
current_gesture = None

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    fps = cap.get(cv2.CAP_PROP_FPS)
    text = f"FPS: {fps:.2f}"

    cv2.putText(frame, text, (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(frame,
                "Press 'g' to choose a gesture, 'z' to clear buffer, 's' to save, 'q' to quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    if current_gesture:
        cv2.putText(frame, f"Current Gesture: {current_gesture}, buffer length: {len(buffer)}/{sequence_length}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []

            xs = [lm.x for lm in hand_landmarks.landmark]
            ys = [lm.y for lm in hand_landmarks.landmark]
            zs = [lm.z for lm in hand_landmarks.landmark]

            # Calculate the mean position (center of the hand)
            mean_x = np.mean(xs)
            mean_y = np.mean(ys)
            mean_z = np.mean(zs)

            # Calculate the ranges
            x_range = max(xs) - min(xs)
            y_range = max(ys) - min(ys)
            z_range = max(zs) - min(zs)

            # Use the maximum range as the hand size
            hand_size = max(x_range, y_range, z_range)

            # Avoid division by zero
            if hand_size == 0:
                hand_size = 1e-6

            for lm in hand_landmarks.landmark:
                landmarks.append([
                    (lm.x - mean_x) / hand_size / 3 + 0.2,
                    (lm.y - mean_y) / hand_size / 3 + 0.2,
                    (lm.z - mean_z) / hand_size / 3 + 0.2,
                ])
            
            buffer.append(landmarks)
            
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            for i, lm in enumerate(landmarks):
                h, w, _ = frame.shape
                x_px, y_px = int(lm[0] * w), int(lm[1] * h)
                if 0 <= x_px < w and 0 <= y_px < h:
                    cv2.circle(frame, (x_px, y_px), 5, (0, 255, 0), -1)  # Green color for normalized landmarks
                
            for connection in mp_hands.HAND_CONNECTIONS:
                start_idx, end_idx = connection

                # Get the start and end landmarks
                start_lm = landmarks[start_idx]
                end_lm = landmarks[end_idx]

                # Convert normalized coordinates back to pixel values
                x_start = int((start_lm[0]) * w)
                y_start = int((start_lm[1]) * h)
                x_end = int((end_lm[0]) * w)
                y_end = int((end_lm[1]) * h)

                # Draw the connection if both points are within frame boundaries
                if (0 <= x_start < w and 0 <= y_start < h and
                    0 <= x_end < w and 0 <= y_end < h):
                    cv2.line(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)  # Green color



    cv2.imshow("Hand Gesture Data Collection", frame)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('g'):
        print("Choose a gesture:")
        for i, gesture in enumerate(gestures):
            print(f"{i}: {gesture}")
        gesture_index = int(input("Enter the gesture number: "))
        current_gesture = gestures[gesture_index]
        buffer.clear()
    elif key == ord('z'):
        buffer.clear()
    elif key == ord('s'):  # Save sequence
        if current_gesture in dynamic and len(buffer) == sequence_length:
            sequence_data = list(buffer)
            save_landmarks(current_gesture, sequence_data)
            buffer.clear()
        elif current_gesture in static and len(buffer) > 0:
            sequence_data = [list(buffer)[-1]]
            save_landmarks(current_gesture, sequence_data)
            buffer.clear()

cap.release()
cv2.destroyAllWindows()
