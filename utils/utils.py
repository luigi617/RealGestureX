# utils/utils.py

import numpy as np
import torch

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
        landmarks = landmarks / max_dist
    landmarks = landmarks.flatten()  # Shape: (63,)
    landmarks = torch.FloatTensor(landmarks)
    return landmarks
