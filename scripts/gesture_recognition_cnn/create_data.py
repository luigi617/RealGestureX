# hand_detection_with_mediapipe.py

import cv2
import mediapipe as mp
import os
import json
import time
from collections import deque
import numpy as np
from models.GestureClasses import static, dynamic
import re

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils




is_static = False


if is_static:
    gestures = static
    dataset_path = "datasets/gesture_dataset_cnn/static"
else:
    gestures = dynamic
    dataset_path = "datasets/gesture_dataset_cnn/dynamic"

for gesture in gestures:
    os.makedirs(os.path.join(dataset_path, gesture), exist_ok=True)

sequence_length = 30
# buffer = deque(maxlen=sequence_length)
buffer = []

def extract_hand_bbox(result):

    box = []

    if result.multi_hand_landmarks:
        for landmarks in result.multi_hand_landmarks:
            # Calculate bounding box of the hand
            x_min = min([landmark.x for landmark in landmarks.landmark])-0.03
            y_min = min([landmark.y for landmark in landmarks.landmark])-0.03
            x_max = max([landmark.x for landmark in landmarks.landmark])+0.03
            y_max = max([landmark.y for landmark in landmarks.landmark])+0.03

            # Convert normalized coordinates to pixel values
            h, w, _ = frame.shape
            x_min, y_min, x_max, y_max = int(x_min * w), int(y_min * h), int(x_max * w), int(y_max * h)

            box.append([x_min, y_min, x_max, y_max])

    return box

def save_cropped_image(gesture_name, cropped_image, further_dir = None):
    gesture_dir = os.path.join(dataset_path, gesture_name)
    if further_dir:
        gesture_dir = os.path.join(gesture_dir, further_dir)
    os.makedirs(gesture_dir, exist_ok=True)
    files = os.listdir(gesture_dir)
    number_pattern = re.compile(r'(\d+)')
    max_number = -1
    for file in files:
        if file.endswith('.jpg'):
            match = number_pattern.search(file)
            if match:
                number = int(match.group(1))
                max_number = max(max_number, number)
    index = max_number+1
    filename = f"{gesture_name}_{index}.jpg"
    filepath = os.path.join(gesture_dir, filename)
    cv2.imwrite(filepath, cropped_image)


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

    # if results.multi_hand_landmarks:
    #     pass
        # for hand_landmarks in results.multi_hand_landmarks:
            
            # mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    hand_data = extract_hand_bbox(results)
    if hand_data:
        x_min, y_min, x_max, y_max = hand_data[0]
        cropped_frame = frame[y_min:y_max, x_min:x_max].copy()
        if len(buffer) < sequence_length:
            buffer.append(cropped_frame)
    # if hand_data:
    #     for hand in hand_data:
    #         x_min, y_min, x_max, y_max = hand
    #         cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)


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
        pass
        if current_gesture in dynamic and len(buffer) == sequence_length:
            latest_frames = buffer
            gesture_dir = os.path.join(dataset_path, current_gesture)
            files = os.listdir(gesture_dir)
            number_pattern = re.compile(r'(\d+)')
            max_number = -1
            for file in files:
                match = number_pattern.search(file)
                if match:
                    max_number = max(max_number, int(match.group(1)))

            for f in latest_frames:
                save_cropped_image(current_gesture, f, str(max_number+1))
            buffer.clear()
        elif current_gesture in static and len(buffer) > 0:
            latest_frame = buffer[-1]
            save_cropped_image(current_gesture, latest_frame)
            buffer.clear()

cap.release()
cv2.destroyAllWindows()










