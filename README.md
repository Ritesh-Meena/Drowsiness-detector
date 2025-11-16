# Real-Time Drowsiness Detection (EAR + MAR + Head Pose) + auto Calibration
Real-time drowsiness detection using EAR, MAR, and 3D head pose with Mediapipe and OpenCV. Features automatic user calibration, smooth multi-cue prediction, and an audio alert system. Runs on CPU without deep learning. Ideal for driver safety, monitoring, and real-time fatigue detection.

A lightweight, real-time drowsiness detection system using **OpenCV**, **Mediapipe**, and classical facial geometry. The system tracks **eye aspect ratio (EAR)**, **mouth aspect ratio (MAR)**, and **3D head pose** to detect signs of fatigue. Includes **automatic calibration**, **multi-cue analysis**, and an **audio alert**.

## ⭐ Features
- Eye closure detection (EAR)
- Yawning detection (MAR)
- 3D head pose estimation (pitch, yaw, roll)
- Automatic user threshold calibration
- Real-time HUD overlay
- Audio alarm for drowsiness
- No deep learning required — runs entirely on CPU

<img width="610" height="1019" alt="drowsiness_system_architecture" src="https://github.com/user-attachments/assets/b69271ce-f618-446c-ab76-ebdfcb05b9c7" />


## 📦 Installation
### 1. Clone the repository
```bash
git clone https://github.com/Ritesh-Meena/Drowsiness-detector.git
cd drowsiness-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

## ▶️ Running the System
```bash
python main.py
```
### Calibration
<img width="1392" height="860" alt="calibration" src="https://github.com/user-attachments/assets/e58eec0c-57e5-427b-b783-a5f694340470" />

### Alert
<img width="1317" height="826" alt="alert result" src="https://github.com/user-attachments/assets/b92e6c5d-c153-43a6-aa01-145073b12352" />

### Drwosiness due to high EAR value
<img width="1317" height="826" alt="ear result" src="https://github.com/user-attachments/assets/253f3f2b-60df-4d34-89f2-079452fe61d4" />

### Drowsiness due to high MAR value
<img width="1317" height="826" alt="mar result" src="https://github.com/user-attachments/assets/59ca6850-ad43-4977-83a2-1b70c2c574c2" />





## 📁 Project Structure
```
📂 Drowsiness-detect0r/
│
├── main.py        # Main program (EAR + MAR + Head Pose + Calibration)
├── requirements.txt              # Project dependencies
├── README.md                     # Project documentation
│
├── assets/
│   └── alarm.wav                 # Alarm sound file
│
├── utils/                        # Optional (if you separate utilities later)
    ├── ear.py
    ├── mar.py
    └── pose.py
```

## 📝 License
MIT License.
