import json
import cv2
import numpy as np
import mediapipe as mp
import os
from glob import glob
import time

from models.transformGesture import TransformGesture
from utils.utils import preprocess_landmarks

# Initialize MediaPipe Hands for connections
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS

transform = TransformGesture()

def load_gesture_json(json_file_path):

    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        return data  # List of frames
    except Exception as e:
        print(f"Error loading JSON file {json_file_path}: {e}")
        return []

def normalize_landmarks(landmarks, image_width, image_height):
    pixel_landmarks = []
    for lm in landmarks:
        x = int(lm[0] * image_width)
        y = int(lm[1] * image_height)
        pixel_landmarks.append((x, y))
    return pixel_landmarks

def draw_landmarks_on_canvas(canvas, pixel_landmarks, connections):
    for idx, (x, y) in enumerate(pixel_landmarks):
        cv2.circle(canvas, (x, y), radius=5, color=(0, 0, 255), thickness=-1)  # Red dots

    for connection in connections:
        start_idx, end_idx = connection
        if start_idx < len(pixel_landmarks) and end_idx < len(pixel_landmarks):
            start_point = pixel_landmarks[start_idx]
            end_point = pixel_landmarks[end_idx]
            cv2.line(canvas, start_point, end_point, color=(0, 255, 0), thickness=2)  # Green lines

def display_gesture_video(json_file_path, frame_rate=30, canvas_size=(640, 480)):
    frames = load_gesture_json(json_file_path)
    if not frames:
        print("No frames to display.")
        return

    frame_delay = int(1000 / frame_rate)

    canvas_width, canvas_height = canvas_size

    cv2.namedWindow('Gesture Video Playback', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Gesture Video Playback', canvas_width, canvas_height)

    print(f"Displaying video from {json_file_path} at {frame_rate} FPS...")
    print("Press 'q' to quit.")
    next_video = False
    while not next_video:
        for frame_idx, landmarks in enumerate(frames):
            canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255  # White background


            pixel_landmarks = normalize_landmarks(landmarks, canvas_width, canvas_height)

            draw_landmarks_on_canvas(canvas, pixel_landmarks, HAND_CONNECTIONS)

            cv2.putText(
                canvas,
                f"Frame: {frame_idx + 1}/{len(frames)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )

            # Display the frame
            cv2.imshow('Gesture Video Playback', canvas)

            # Wait for the appropriate time or until 'q' is pressed
            key = cv2.waitKey(frame_delay) & 0xFF
            if key == ord('q'):
                print("Video playback interrupted by user.")
                next_video = True
            if next_video: break

    cv2.destroyAllWindows()
    print("Video playback finished.")

def display_all_gesture_videos(dataset_path, frame_rate=30, canvas_size=(640, 480)):
    
    for gesture in os.listdir(dataset_path):
        gesture_dir = os.path.join(dataset_path, gesture)
        if not os.path.isdir(gesture_dir):
            continue

        json_files = glob(os.path.join(gesture_dir, '*.json'))
        for json_file in json_files:
            print(f"Displaying gesture '{gesture}' from file: {json_file}")
            display_gesture_video(json_file, frame_rate, canvas_size)

def display_one_gesture_all_videos(gesture_dir, frame_rate=30, canvas_size=(640, 480)):
    if not os.path.isdir(gesture_dir):
        return
        

    json_files = glob(os.path.join(gesture_dir, '*.json'))
    for json_file in json_files:
        print(f"Displaying gesture '{gesture_dir.split('/')[-1]}' from file: {json_file}")
        display_gesture_video(json_file, frame_rate, canvas_size)

# Example usage
if __name__ == "__main__":
    # Path to a single JSON file
    # single_json_file = 'gesture_dataset/train/swipe_up/swipe_up_1730858820.json'
    # single_json_file = 'gesture_dataset/train/pointing/pointing_1730859898.json'
    # display_gesture_video(single_json_file)

    # Alternatively, display all gesture videos in the dataset
    dataset_directory = 'gesture_dataset/static/rock_on'  # Replace with your dataset directory path
    display_one_gesture_all_videos(dataset_directory)

    # dataset_directory = 'gesture_dataset/train'
    # display_all_gesture_videos(dataset_directory)
