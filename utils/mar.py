# mar.py
import cv2
import numpy as np

# Mouth landmark indices (from Mediapipe FaceMesh)
MOUTH = {
    "left": 61,       # left corner
    "right": 291,     # right corner
    "top_outer": 13,  # upper lip outer
    "bottom_outer": 14,  # lower lip outer
    "top_inner": 81,  # upper lip inner
    "bottom_inner": 178 # lower lip inner
}

def mouth_aspect_ratio(landmarks, frame_width, frame_height):# Convert normalized coordinates to pixel values
    
    left = np.array([landmarks[MOUTH["left"]].x * frame_width, landmarks[MOUTH["left"]].y * frame_height])
    right = np.array([landmarks[MOUTH["right"]].x * frame_width, landmarks[MOUTH["right"]].y * frame_height])
    top_outer = np.array([landmarks[MOUTH["top_outer"]].x * frame_width, landmarks[MOUTH["top_outer"]].y * frame_height])
    bottom_outer = np.array([landmarks[MOUTH["bottom_outer"]].x * frame_width, landmarks[MOUTH["bottom_outer"]].y * frame_height])
    top_inner = np.array([landmarks[MOUTH["top_inner"]].x * frame_width, landmarks[MOUTH["top_inner"]].y * frame_height])
    bottom_inner = np.array([landmarks[MOUTH["bottom_inner"]].x * frame_width, landmarks[MOUTH["bottom_inner"]].y * frame_height])

    
    vertical1 = np.linalg.norm(top_outer - bottom_outer)# Compute vertical distances (outer + inner)
    vertical2 = np.linalg.norm(top_inner - bottom_inner)
    vertical = (vertical1 + vertical2) / 2.0

   
    horizontal = np.linalg.norm(left - right) # Horizontal distance (mouth width)

    
    mar = vertical / horizontal# MAR formula
    return mar
