import cv2
import numpy as np

LANDMARKS = {
    "nose_tip": 1,
    "chin": 199,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}

MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -63.6, -12.5),      # Chin
    (-43.3, 32.7, -26.0),     # Left eye outer
    (43.3, 32.7, -26.0),      # Right eye outer
    (-28.9, -28.9, -24.1),    # Left mouth corner
    (28.9, -28.9, -24.1),     # Right mouth corner
], dtype=np.float64)


def normalize_angle(a):
    """Convert to [-180, 180] range."""
    a = ((a + 180) % 360) - 180
    return a


def head_pose_angles(landmarks, img_w, img_h):
    """Compute stable head-pose angles (pitch, yaw, roll) from Mediapipe landmarks."""
    idx = LANDMARKS
    image_points = np.array([
        (landmarks[idx["nose_tip"]].x * img_w, landmarks[idx["nose_tip"]].y * img_h),
        (landmarks[idx["chin"]].x * img_w, landmarks[idx["chin"]].y * img_h),
        (landmarks[idx["left_eye_outer"]].x * img_w, landmarks[idx["left_eye_outer"]].y * img_h),
        (landmarks[idx["right_eye_outer"]].x * img_w, landmarks[idx["right_eye_outer"]].y * img_h),
        (landmarks[idx["left_mouth"]].x * img_w, landmarks[idx["left_mouth"]].y * img_h),
        (landmarks[idx["right_mouth"]].x * img_w, landmarks[idx["right_mouth"]].y * img_h),
    ], dtype=np.float64)

    focal_length = img_w
    center = (img_w / 2.0, img_h / 2.0)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    success, rvec, tvec = cv2.solvePnP(
        MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0.0, 0.0, 0.0

    # Convert to rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # Flip Y & Z axes to align with Mediapipe camera orientation
    R[:, 1:3] *= -1

    # Compute Euler angles from the adjusted matrix
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy < 1e-6:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    else:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])

    pitch, yaw, roll = np.degrees([x, y, z])

    # Normalize
    pitch, yaw, roll = map(normalize_angle, (pitch, yaw, roll))

    return pitch, yaw, roll
