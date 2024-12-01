# hand_detection_with_mediapipe.py

import cv2
import mediapipe as mp
import os
import json
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def extract_hand_bbox_and_landmarks(frame):
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    hand_data = []

    # Check if hands are detected
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

            # Collect landmarks and bounding box data
            hand_landmarks = [(lm.x * w, lm.y * h) for lm in landmarks.landmark]
            hand_data.append({
                'bbox': [x_min, y_min, x_max, y_max],
                'landmarks': hand_landmarks
            })

    return hand_data

def save_data(frame, hand_data, index):
    # Create the directories if they do not exist
    os.makedirs('hand_tracking_dataset/image', exist_ok=True)
    os.makedirs('hand_tracking_dataset/hand', exist_ok=True)
    os.makedirs('hand_tracking_dataset/landmark', exist_ok=True)
    os.makedirs('hand_tracking_dataset/cropped_image', exist_ok=True)

    # Save the original frame
    image_filename = f'hand_tracking_dataset/image/frame_{index}.jpg'
    cv2.imwrite(image_filename, frame)

    for hand in hand_data:

        x_min, y_min, x_max, y_max = hand['bbox']
        hand_crop = frame[y_min:y_max, x_min:x_max]
        hand_image_filename = f'hand_tracking_dataset/cropped_image/hand_{index}.jpg'
        cv2.imwrite(hand_image_filename, hand_crop)

        # Save the hand landmarks to a JSON file
        landmarks_filename = f'hand_tracking_dataset/landmark/hand_{index}.json'
        with open(landmarks_filename, 'w') as f:
            json.dump(hand['landmarks'], f)

        yolo_filename = f'hand_tracking_dataset/hand/hand_{index}.txt'

        h, w, _ = frame.shape
        x_min, y_min, x_max, y_max = hand['bbox']
        
        # Convert pixel values to YOLO normalized format
        x_center = (x_min + x_max) / 2 / w
        y_center = (y_min + y_max) / 2 / h
        width = (x_max - x_min) / w
        height = (y_max - y_min) / h

        # Write to YOLO txt file (class_id, x_center, y_center, width, height)
        with open(yolo_filename, 'a') as f:
            f.write(f"0 {x_center} {y_center} {width} {height}\n")

def resize_with_aspect_ratio(frame, target_width=800):
    h, w, _ = frame.shape
    aspect_ratio = w / h
    target_height = int(target_width / aspect_ratio)
    return cv2.resize(frame, (target_width, target_height))
cap = cv2.VideoCapture(2)

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break
    # frame = cv2.resize(frame, (800, 800))
    frame = resize_with_aspect_ratio(frame, target_width=800)



    # Extract hand bounding boxes and landmarks
    hand_data = extract_hand_bbox_and_landmarks(frame)
    print(hand_data)
    
    if hand_data:
        # Save the image, hand crops, and landmarks
        save_data(frame, hand_data, int(time.time()))

        # Draw bounding boxes and landmarks on the frame for visualization
        for hand in hand_data:
            x_min, y_min, x_max, y_max = hand['bbox']
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Draw landmarks
            for lm in hand['landmarks']:
                cv2.circle(frame, (int(lm[0]), int(lm[1])), 5, (0, 0, 255), -1)

    # Display the image with bounding boxes and landmarks
    cv2.imshow('Hand Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

