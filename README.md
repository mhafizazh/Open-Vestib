# 👁️ Eye Tracking Application (Dual Camera + Plot Recording)

A real-time eye tracking tool built with PyQt5, OpenCV, and PyQtGraph. This app captures video from two cameras (left and right eyes), detects pupil positions, plots their movement over time, and records..

![Preview](media/2025-06-0313-02-34-ezgif.com-video-to-gif-converter.gif) 

---

## 🧠 Features
- ▶️ Start and stop button for recording added
- 📷 selecting the camera input
- 🎥 Dual-camera live feed (left & right eye)
- 🎯 Real-time pupil detection with crosshair overlay
- 📈 Live plotting of X and Y positions over time
- 🎬 Automatic screen recording (video + plots) saved as `.mp4`
- 💾 Output saved in `result_videos/` folder
- 📷 The safed vedio includes series graph
- 📁 user can select the directory where the file    will be saved

---

## 📦 Dependencies

Ensure Python 3.7+ is installed. Then:

```bash
pip install pyqt5 opencv-python pyqtgraph numpy
```

## 🚀 Usage
1. Clone the repo:
```bash
git clone https://github.com/your-username/eye-tracker-app.git
cd eye-tracker-app
```

2. Run the application:
```bash
python main.py
```
3. Output .mp4 video will be saved in: 
```bash
./result_videos/
```

