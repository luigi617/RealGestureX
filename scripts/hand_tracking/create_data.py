import cv2
import mediapipe as mp
import os
import re

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

p = "val"
p = "train"
# Change this path to your desired directory structure
yolo_dataset_path = "datasets/yolo_dataset"
yolo_images_base = os.path.join(yolo_dataset_path, "images", p)
yolo_labels_base = os.path.join(yolo_dataset_path, "labels", p)

os.makedirs(yolo_images_base, exist_ok=True)
os.makedirs(yolo_labels_base, exist_ok=True)

def extract_hand_bbox(result, frame):
    box = []
    if result.multi_hand_landmarks:
        for landmarks in result.multi_hand_landmarks:
            # Calculate bounding box of the hand
            x_min = max(min([landmark.x for landmark in landmarks.landmark]) - 0.03, 0)
            y_min = max(min([landmark.y for landmark in landmarks.landmark]) - 0.03, 0)
            x_max = min(max([landmark.x for landmark in landmarks.landmark]) + 0.03, 1)
            y_max = min(max([landmark.y for landmark in landmarks.landmark]) + 0.03, 1)

            # Convert normalized coordinates to pixel values
            h, w, _ = frame.shape
            x_min, y_min, x_max, y_max = int(x_min * w), int(y_min * h), int(x_max * w), int(y_max * h)
            box.append([x_min, y_min, x_max, y_max])
    return box

def save_yolo_data(frame, box, save=True):
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

    image_filename = f"{index}.jpg"
    image_path = os.path.join(yolo_images_base, image_filename)
    if save:
        cv2.imwrite(image_path, frame)
    
    h, w, _ = frame.shape
    label_path = os.path.join(yolo_labels_base, f"{index}.txt")

    with open(label_path, 'w') as f:
        x_min, y_min, x_max, y_max = box
        x_center = ((x_min + x_max) / 2) / w
        y_center = ((y_min + y_max) / 2) / h
        width = (x_max - x_min) / w
        height = (y_max - y_min) / h
        if save:
            f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    return x_center, y_center, width, height

def save_negative_sample(frame, save=True):
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

    image_filename = f"{index}.jpg"
    image_path = os.path.join(yolo_images_base, image_filename)
    if save:
        cv2.imwrite(image_path, frame)

def draw_yolo_boxes(frame, detection, color=(0, 255, 0), thickness=2):

    img_height, img_width = frame.shape[:2]


    x_center, y_center, width, height = detection

    x_center_pix = int(x_center * img_width)
    y_center_pix = int(y_center * img_height)
    width_pix = int(width * img_width)
    height_pix = int(height * img_height)

    x_min = int(x_center_pix - (width_pix / 2))
    y_min = int(y_center_pix - (height_pix / 2))
    x_max = int(x_center_pix + (width_pix / 2))
    y_max = int(y_center_pix + (height_pix / 2))

    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_width - 1, x_max)
    y_max = min(img_height - 1, y_max)

    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness)


    return frame

cap = cv2.VideoCapture(2)

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    
    hand_data = extract_hand_bbox(results, frame)
    if hand_data:
        detections = save_yolo_data(frame, hand_data[0], True)
        # frame = draw_yolo_boxes(frame, detections)
    else:
        save_negative_sample()

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break

    cv2.imshow("Hand Gesture Data Collection", frame)

cap.release()
cv2.destroyAllWindows()
