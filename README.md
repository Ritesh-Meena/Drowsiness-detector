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
