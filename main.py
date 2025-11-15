import argparse
import threading
import time
from collections import deque
import cv2
import mediapipe as mp
import numpy as np
import simpleaudio as sa

#utility files from the folder being used
from utils.ear import LEFT_EYE_IDX, RIGHT_EYE_IDX, eye_aspect_ratio
from utils.pose import head_pose_angles
from utils.mar import mouth_aspect_ratio


#camera use
def open_camera(index, width, height):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    print(f"[INFO] Using camera index {index}")
    return cap

#sound file
def play_beep(wav_path):
    try:
        wave_obj = sa.WaveObject.from_wave_file(wav_path)
        wave_obj.play()
    except Exception as e:
        print(f"[WARN] Failed to play sound: {e}")


#display
def draw_hud(frame, ear, mar, pitch, yaw, roll, status, cfg):
    overlay = frame.copy()
    h, w = frame.shape[:2]

    color_alert = (60, 255, 60)
    color_warn = (0, 200, 255)
    color_drowsy = (0, 0, 255)
    color_text = (230, 230, 230)

    cv2.rectangle(overlay, (10, 10), (500, 160), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    color = color_alert if status == "alert" else color_warn if status == "slightly_drowsy" else color_drowsy

    cv2.putText(frame, f"EAR: {ear:.3f} | MAR: {mar:.3f}", (20, 40),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, color_text, 1)
    cv2.putText(frame, f"Pitch: {pitch:.1f} | Yaw: {yaw:.1f} | Roll: {roll:.1f}",
                (20, 70), cv2.FONT_HERSHEY_DUPLEX, 0.7, color_text, 1)
    cv2.putText(frame, f"STATUS: {status.upper()}", (20, 110),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)

    cv2.putText(frame, f"EAR<th:{cfg.ear_thresh:.2f}  MAR>th:{cfg.mar_thresh:.2f}",
                (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_text, 1)
    cv2.putText(frame, f"Pitch>{cfg.pitch_thresh:.1f} Roll>{cfg.roll_thresh:.1f} Yaw>{cfg.yaw_thresh:.1f}",
                (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_text, 1)


# state buffer for reducing false positive results
class LiveState:
    def __init__(self, maxlen=150):
        self.ear_hist = deque(maxlen=maxlen)
        self.mar_hist = deque(maxlen=maxlen)
        self.pitch_hist = deque(maxlen=maxlen)
        self.roll_hist = deque(maxlen=maxlen)
        self.yaw_hist = deque(maxlen=maxlen)
        self.time_hist = deque(maxlen=maxlen)

    def push(self, t, ear, mar, pitch, roll, yaw):
        self.time_hist.append(t)
        self.ear_hist.append(ear)
        self.mar_hist.append(mar)
        self.pitch_hist.append(pitch)
        self.roll_hist.append(roll)
        self.yaw_hist.append(yaw)

    def reset(self):
        self.ear_hist.clear()
        self.mar_hist.clear()
        self.pitch_hist.clear()
        self.roll_hist.clear()
        self.yaw_hist.clear()
        self.time_hist.clear()

    def get_means(self):
        return dict(
            mean_ear=np.mean(self.ear_hist) if self.ear_hist else 0,
            mean_mar=np.mean(self.mar_hist) if self.mar_hist else 0,
            mean_pitch=np.mean(self.pitch_hist) if self.pitch_hist else 0,
            mean_roll=np.mean(self.roll_hist) if self.roll_hist else 0,
            mean_yaw=np.mean(self.yaw_hist) if self.yaw_hist else 0
        )


# Auto calibration
def auto_calibrate(face_mesh, cap, duration=15):
    print(f"[INFO] Calibrating thresholds for {duration}s...")
    start = time.time()
    ears, mars, pitches, rolls, yaws = [], [], [], [], []

    while time.time() - start < duration:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        if not res.multi_face_landmarks:
            cv2.putText(frame, "No face - waiting...", (40, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Calibration", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        lm = res.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]
        left_eye = [(int(lm[i].x*w), int(lm[i].y*h)) for i in LEFT_EYE_IDX]
        right_eye = [(int(lm[i].x*w), int(lm[i].y*h)) for i in RIGHT_EYE_IDX]
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2
        mar = mouth_aspect_ratio(lm, w, h)
        pitch, yaw, roll = head_pose_angles(lm, w, h)
        if pitch is None or yaw is None or roll is None:
            continue

        ears.append(float(ear))
        mars.append(float(mar))
        pitches.append(float(pitch))
        rolls.append(float(roll))
        yaws.append(float(yaw))

        progress = int(((time.time() - start) / duration) * w)
        cv2.rectangle(frame, (0, h - 20), (progress, h - 5), (0, 200, 255), -1)
        cv2.putText(frame, "Calibrating... Hold still, blink normally", (30, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Calibration", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyWindow("Calibration")

    if len(ears) < 10:
        print("[WARN] Not enough calibration data. Using defaults.")
        return 0.3, 0.6, 12, 15, 20

    pitches = np.array(pitches)
    rolls = np.array(rolls)
    yaws = np.array(yaws)

    ear_th = np.median(ears) * 0.85
    mar_th = np.median(mars) * 1.5
    pitch_th = min(15.0, np.std(pitches) * 1.5 + 5.0)
    roll_th = min(20.0, np.std(rolls) * 1.5 + 8.0)
    yaw_th = min(25.0, np.std(yaws) * 1.5 + 10.0)

    print(f"[INFO] Calibrated: EAR={ear_th:.3f}, MAR={mar_th:.3f}, "
          f"Pitch={pitch_th:.1f}, Roll={roll_th:.1f}, Yaw={yaw_th:.1f}")
    return ear_th, mar_th, pitch_th, roll_th, yaw_th


#main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--alarm", default="assets/alarm.wav")
    ap.add_argument("--no-sound", action="store_true")
    args = ap.parse_args()

    cap = open_camera(args.camera, args.width, args.height)
    if not cap:
        print("[ERR] Cannot open camera.")
        return

    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(refine_landmarks=True, max_num_faces=1)

    ear_th, mar_th, pitch_th, roll_th, yaw_th = 0.3, 0.6, 12, 15, 20
    state = LiveState()
    last_beep = 0
    calibrated = False
    face_missing_time = 0

    print("[INFO] Waiting for face...")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        t = time.time()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        if not res.multi_face_landmarks:
            face_missing_time += 1
            if face_missing_time > 30:
                state.reset()
                calibrated = False
            cv2.putText(frame, "No face detected", (50, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Drowsiness Detector", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        face_missing_time = 0
        lm = res.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]
        left_eye = [(int(lm[i].x*w), int(lm[i].y*h)) for i in LEFT_EYE_IDX]
        right_eye = [(int(lm[i].x*w), int(lm[i].y*h)) for i in RIGHT_EYE_IDX]
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2
        mar = mouth_aspect_ratio(lm, w, h)
        pitch, yaw, roll = head_pose_angles(lm, w, h)
        if pitch is None:
            continue

        if (ear < 0.01 and mar < 0.01):
            state.reset()
            calibrated = False

        if not calibrated:
            ear_th, mar_th, pitch_th, roll_th, yaw_th = auto_calibrate(face_mesh, cap, duration=15)
            calibrated = True
            continue

        state.push(t, ear, mar, pitch, roll, yaw)
        feats = state.get_means()

        if feats['mean_ear'] < ear_th or feats['mean_mar'] > mar_th:
            pred = "drowsy"
        elif (feats['mean_ear'] < ear_th * 1.1 or
              abs(feats['mean_pitch']) > pitch_th or
              abs(feats['mean_roll']) > roll_th or
              abs(feats['mean_yaw']) > yaw_th):
            pred = "slightly_drowsy"
        else:
            pred = "alert"

        if pred == "drowsy" and not args.no_sound and time.time() - last_beep > 2:
            threading.Thread(target=play_beep, args=(args.alarm,), daemon=True).start()
            last_beep = time.time()

        draw_hud(frame, ear, mar, pitch, yaw, roll, pred,
                 type("Cfg", (), dict(ear_thresh=ear_th, mar_thresh=mar_th,
                                      pitch_thresh=pitch_th, roll_thresh=roll_th,
                                      yaw_thresh=yaw_th)))

        if abs(pitch) > pitch_th or abs(roll) > roll_th or abs(yaw) > yaw_th:
            cv2.putText(frame, "⚠ HEAD OFF AXIS", (20, 160),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 200, 255), 2)

        cv2.imshow("Drowsiness Detector", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
