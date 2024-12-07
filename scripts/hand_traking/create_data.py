# hand_detection_with_mediapipe.py

import cv2
import mediapipe as mp
import os
import json
import time
from collections import deque
import numpy as np
import re

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

yolo_dataset_path = "yolo_dataset"
yolo_images_base = os.path.join(yolo_dataset_path, "images")
yolo_labels_base = os.path.join(yolo_dataset_path, "labels")
os.makedirs(yolo_images_base, exist_ok=True)
os.makedirs(yolo_labels_base, exist_ok=True)


def extract_hand_bbox(result, frame):

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

def save_yolo_data(frame, box):
    """
    Save the original frame and corresponding YOLO label file.
    boxes: list of [x_min, y_min, x_max, y_max]
    """
    

    files = os.listdir(yolo_images_base)
    number_pattern = re.compile(r'(\d+)')
    max_number = -1
    for file in files:
        if file.endswith('.jpg'):
            match = number_pattern.search(file)
            if match:
                number = int(match.group(1))
                max_number = max(max_number, number)
    index = max_number+1
    # Save the image
    image_filename = f"{index}.jpg"
    image_path = os.path.join(yolo_images_base, image_filename)
    cv2.imwrite(image_path, frame)
    

    # Create and save the label file
    # YOLO format: class x_center y_center width height (all normalized)
    h, w, _ = frame.shape
    label_path = os.path.join(yolo_labels_base, f"{index}.txt")

    with open(label_path, 'w') as f:
        x_min, y_min, x_max, y_max = box
        x_center = ((x_min + x_max) / 2) / w
        y_center = ((y_min + y_max) / 2) / h
        width = (x_max - x_min) / w
        height = (y_max - y_min) / h
        # Assume class id is 0 for "hand"
        f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


cap = cv2.VideoCapture(2)


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


    cv2.imshow("Hand Gesture Data Collection", frame)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):  # Save sequence
        hand_data = extract_hand_bbox(results, frame)
        if hand_data:
            save_yolo_data(frame, hand_data[0])

cap.release()
cv2.destroyAllWindows()










