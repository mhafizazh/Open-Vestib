from PyQt5 import QtWidgets, QtCore, QtGui
from camera_handler import init_cameras, get_frames
from pupil_detection import detect_pupil
from plotting import PlotManager
import numpy as np
import time
from collections import deque
import cv2
import os



class EyeTrackerApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.left_cam, self.right_cam = init_cameras()
        self.roi_width, self.roi_height = 150, 150
        self.max_data_points = 200
        self.time_data = deque(maxlen=self.max_data_points)
        self.left_x_data, self.left_y_data = deque(), deque()
        self.right_x_data, self.right_y_data = deque(), deque()
        self.last_left_x = self.last_left_y = None
        self.last_right_x = self.last_right_y = None
        self.start_time = time.time()
        self.recording_enabled = False

        # UI setup
        self.setWindowTitle("Eye Tracking System")
        self.resize(1200, 800)
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QGridLayout(self.central_widget)
        self.video_label = QtWidgets.QLabel()
        self.layout.addWidget(self.video_label, 0, 0, 1, 2)

        # Buttons
        self.start_button = QtWidgets.QPushButton("start")
        self.stop_button = QtWidgets.QPushButton("stop")

        self.start_button.clicked.connect(self.start_recording)
        self.stop_button.clicked.connect(self.stop_recording)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)

        self.layout.addLayout(button_layout, 2, 0, 1, 2)  # row 2, spanning 2 columns


        # Create output directory
        os.makedirs("result_videos", exist_ok=True)

        # Initialize screen recording writer (you'll set this properly in update)
        self.video_writer = None

        # Plot manager
        self.plot_manager = PlotManager(self.layout, self.max_data_points)

        # Timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(30)

    def update(self):
        left_frame, right_frame = get_frames(self.left_cam, self.right_cam)
        if left_frame is None or right_frame is None:
            QtWidgets.QMessageBox.critical(self, "Camera Error", "Unable to read from one or both cameras.")
            self.timer.stop()
            return


        # Process left eye
        lh, lw = left_frame.shape[:2]
        lx1 = (lw - self.roi_width) // 2
        ly1 = (lh - self.roi_height) // 2
        left_roi = left_frame[ly1:ly1+self.roi_height, lx1:lx1+self.roi_width].copy()
        left_x, left_y, _ = detect_pupil(left_roi, lw, lh)

        # Process right eye
        rh, rw = right_frame.shape[:2]
        rx1 = (rw - self.roi_width) // 2
        ry1 = (rh - self.roi_height) // 2
        right_roi = right_frame[ry1:ry1+self.roi_height, rx1:rx1+self.roi_width].copy()
        right_x, right_y, _ = detect_pupil(right_roi, rw, rh)

        # Time
        current_time = time.time() - self.start_time

        if left_x is not None and left_y is not None:
            self.last_left_x = left_x
            self.last_left_y = left_y
        if right_x is not None and right_y is not None:
            self.last_right_x = right_x
            self.last_right_y = right_y

        if self.last_left_x is not None and self.last_left_y is not None:
            self.time_data.append(current_time)
            self.left_x_data.append(self.last_left_x)
            self.left_y_data.append(self.last_left_y)

            if self.last_right_x is not None and self.last_right_y is not None:
                self.right_x_data.append(self.last_right_x)
                self.right_y_data.append(self.last_right_y)

            self.plot_manager.update(
                t=current_time,
                lx=self.last_left_x,
                ly=self.last_left_y,
                rx=self.last_right_x,
                ry=self.last_right_y
            )

        # Draw detection markers
        if left_x is not None and left_y is not None:
            cv2.line(left_frame, (left_x - 10, left_y), (left_x + 10, left_y), (0, 255, 255), 1)
            cv2.line(left_frame, (left_x, left_y - 10), (left_x, left_y + 10), (0, 255, 255), 1)

        if right_x is not None and right_y is not None:
            cv2.line(right_frame, (right_x - 10, right_y), (right_x + 10, right_y), (0, 255, 255), 1)
            cv2.line(right_frame, (right_x, right_y - 10), (right_x, right_y + 10), (0, 255, 255), 1)


        # Combine and display
        combined_frame = np.hstack((left_frame, right_frame))

        # Initialize VideoWriter once
        if self.video_writer is None:
            height, width = combined_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_path = os.path.join("result_videos", f"recording_{int(time.time())}.mp4")
            self.video_writer = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

        # ✅ Write frame to video
        if self.recording_enabled and self.video_writer:
            self.video_writer.write(combined_frame)


        # Convert to Qt image and display
        rgb_image = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        self.video_label.setPixmap(QtGui.QPixmap.fromImage(qt_image))


    def closeEvent(self, event):
        self.left_cam.release()
        self.right_cam.release()

        if self.video_writer:
            self.video_writer.release()

        event.accept()
    
    def start_recording(self):
        self.recording_enabled = True
        if self.video_writer is None:
            height, width = self.video_label.height(), self.video_label.width()
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_path = os.path.join("result_videos", f"recording_{int(time.time())}.mp4")
            self.video_writer = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
        print("Recording started")

    def stop_recording(self):
        self.recording_enabled = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        print("Recording stopped")

    

