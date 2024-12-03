# utils/utils.py

import os
import random
import numpy as np
import torch
from torchvision.ops import box_iou

def preprocess_landmarks(landmarks):
    """
    Preprocess hand landmarks by normalizing.

    Args:
        landmarks (list or np.ndarray): List of 63 values (21 landmarks * 3 coordinates).

    Returns:
        torch.Tensor: Normalized landmarks tensor of shape (63,).
    """
    landmarks = np.array(landmarks).reshape(-1, 3)  # Shape: (21, 3)
    # Normalize landmarks relative to the wrist (landmark 0)
    wrist = landmarks[0]
    landmarks = landmarks - wrist  # Shift wrist to origin
    # Scale based on the maximum distance from wrist
    max_dist = np.max(np.linalg.norm(landmarks, axis=1))
    if max_dist > 0:
        landmarks = landmarks / max_dist + 0.5
    landmarks = landmarks.flatten()  # Shape: (63,)
    landmarks = torch.FloatTensor(landmarks)
    return landmarks


def split_data(dir, classes, filetypes):
    train_dir = {}
    val_dir = {}
    test_dir = {}
    for i, cls in enumerate(classes):
        cls_dir = os.path.join(dir, cls)
        if not os.path.isdir(cls_dir): continue
        filetype = filetypes if isinstance(filetypes, str) else filetypes[i]
        files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.endswith(filetype)]
        random.shuffle(files)
        n = len(files)
        split_1 = int(0.8 * n)
        split_2 = split_1 + int(0.1 * n)

        train_dir[cls] = files[:split_1]
        val_dir[cls] = files[split_1:split_2]
        test_dir[cls] = files[split_2:]
    
    return train_dir, val_dir, test_dir

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy




def calculate_mae(pred_landmarks, true_landmarks):
    return torch.mean(torch.abs(pred_landmarks - true_landmarks))

def map_gesture_to_command(gesture):
    """
    Maps a recognized gesture to a specific command.

    Args:
        gesture (str): The recognized gesture label.

    Returns:
        str: The corresponding command.
    """
    gesture_command_mapping = {
        # Static Gestures
        'pointing': 'select_item',
        'open_palm': 'open_menu',
        'thumb_index_touch': 'zoom_in',
        'thumb_middle_touch': 'zoom_out',
        'fist': 'pause',

        # Dynamic Gestures
        'swipe_up': 'scroll_up',
        'swipe_down': 'scroll_down',
        'swipe_left': 'prev_slide',
        'swipe_right': 'next_slide'
    }

    return gesture_command_mapping.get(gesture, 'no_action')



def execute_command(command):
    """
    Executes a specific command based on the recognized gesture.

    Args:
        command (str): The command to execute.
    """
    if command == 'select_item':
        print("Command: Select Item")
        # Implement selection logic here
    elif command == 'open_menu':
        print("Command: Open Menu")
        # Implement menu opening logic here
    elif command == 'zoom_in':
        print("Command: Zoom In")
        # Implement zoom in logic here
    elif command == 'zoom_out':
        print("Command: Zoom Out")
        # Implement zoom out logic here
    elif command == 'pause':
        print("Command: Pause")
        # Implement pause logic here
    elif command == 'scroll_up':
        print("Command: Scroll Up")
        # Implement scroll up logic here
    elif command == 'scroll_down':
        print("Command: Scroll Down")
        # Implement scroll down logic here
    elif command == 'prev_slide':
        print("Command: Previous Slide")
        # Implement previous slide logic here
    elif command == 'next_slide':
        print("Command: Next Slide")
        # Implement next slide logic here
    else:
        pass  # No action