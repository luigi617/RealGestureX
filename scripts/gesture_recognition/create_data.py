import cv2
import mediapipe as mp
import json
import os
from collections import deque
import time
import numpy as np
from models.GestureClasses import static, dynamic

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)



is_static = False


if is_static:
    gestures = static
    dataset_path = "gesture_dataset/static"
else:
    gestures = dynamic
    dataset_path = "gesture_dataset/dynamic"

for gesture in gestures:
    os.makedirs(os.path.join(dataset_path, gesture), exist_ok=True)

sequence_length = 30
# buffer = deque(maxlen=sequence_length)
buffer = []

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

cap = cv2.VideoCapture(2)
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

            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
            
            if len(buffer) < sequence_length:
                buffer.append(landmarks)
            
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)



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










