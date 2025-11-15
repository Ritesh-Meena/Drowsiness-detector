import numpy as np
from scipy.spatial import distance as dist

# Mediapipe FaceMesh indices for eye landmarks (6 points per eye)
LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]   # (p1,p2,p3,p4,p5,p6)
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(pts):
    """
    Compute the Eye Aspect Ratio (EAR) for a 6-point eye polygon.
    pts: list of (x, y) with length 6 ordered as [p1,p2,p3,p4,p5,p6]
    """
    A = dist.euclidean(pts[1], pts[5])
    B = dist.euclidean(pts[2], pts[4])
    C = dist.euclidean(pts[0], pts[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)
