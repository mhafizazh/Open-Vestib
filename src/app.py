from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QFileDialog
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
        self.recording_filename = None

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
        self.quit_button = QtWidgets.QPushButton("quit")

        self.start_button.clicked.connect(self.start_recording)
        self.stop_button.clicked.connect(self.stop_recording)
        self.quit_button.clicked.connect(self.close)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.quit_button) 

        self.layout.addLayout(button_layout, 2, 0, 1, 2)  # row 2, spanning 2 columns


        # Create output directory
        os.makedirs("../result_videos", exist_ok=True)

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
            # Only append if values are numbers
            if isinstance(self.last_left_x, (int, float)) and isinstance(self.last_left_y, (int, float)):
                self.left_x_data.append(self.last_left_x)
                self.left_y_data.append(self.last_left_y)
        
            if self.last_right_x is not None and self.last_right_y is not None:
                if isinstance(self.last_right_x, (int, float)) and isinstance(self.last_right_y, (int, float)):
                    self.right_x_data.append(self.last_right_x)
                    self.right_y_data.append(self.last_right_y)
        
            self.plot_manager.update(
                t=current_time,
                lx=self.last_left_x if isinstance(self.last_left_x, (int, float)) else 0,
                ly=self.last_left_y if isinstance(self.last_left_y, (int, float)) else 0,
                rx=self.last_right_x if isinstance(self.last_right_x, (int, float)) else 0,
                ry=self.last_right_y if isinstance(self.last_right_y, (int, float)) else 0,
            )
        

        # Draw detection markers
        if left_x is not None and left_y is not None:
            cv2.line(left_frame, (left_x - 10, left_y), (left_x + 10, left_y), (0, 255, 255), 1)
            cv2.line(left_frame, (left_x, left_y - 10), (left_x, left_y + 10), (0, 255, 255), 1)

        if right_x is not None and right_y is not None:
            cv2.line(right_frame, (right_x - 10, right_y), (right_x + 10, right_y), (0, 255, 255), 1)
            cv2.line(right_frame, (right_x, right_y - 10), (right_x, right_y + 10), (0, 255, 255), 1)


        # Combine and display
        # Ensure left_frame and right_frame have the same height before stacking
        lh, lw = left_frame.shape[:2]
        rh, rw = right_frame.shape[:2]
        
        if lh != rh:
            # Resize right_frame to match left_frame's height
            new_rw = int(rw * lh / rh)
            right_frame = cv2.resize(right_frame, (new_rw, lh))
            # Optionally, resize left_frame to match right_frame's height instead:
            # new_lw = int(lw * rh / lh)
            # left_frame = cv2.resize(left_frame, (new_lw, rh))
        
        # Now you can safely stack
        combined_frame = np.hstack((left_frame, right_frame))
        plot_frame = self.get_plot_frame()  # <-- FIXED LINE
        
        if plot_frame.shape[1] != combined_frame.shape[1]:
            plot_frame = cv2.resize(plot_frame, (combined_frame.shape[1], plot_frame.shape[0]))
        
        # Stack video and plot vertically
        final_frame = np.vstack((combined_frame, plot_frame))

        if self.recording_enabled and self.video_writer is None and self.recording_filename:
            height, width = final_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.recording_filename, fourcc, 30, (width, height))
            if not self.video_writer.isOpened():
                print("Trying XVID codec...")
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.video_writer = cv2.VideoWriter(self.recording_filename, fourcc, 30, (width, height))
            if not self.video_writer.isOpened():
                print("Trying MJPG codec...")
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                self.video_writer = cv2.VideoWriter(self.recording_filename, fourcc, 30, (width, height))
            if not self.video_writer.isOpened():
                print("VideoWriter failed to open")
                self.video_writer = None
                self.recording_enabled = False
                return

        # ✅ Write frame to video
        if self.recording_enabled and self.video_writer:
            self.video_writer.write(final_frame)


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
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Recording",
            "",
            "MP4 Files (*.mp4);;All Files (*)",
            options=options
        )
        if not file_path:
            print("Recording canceled by user.")
            return
        # Ensure .mp4 extension
        if not file_path.lower().endswith('.mp4'):
            file_path += '.mp4'
        self.recording_enabled = True
        self.recording_filename = file_path
        print("Recording will start on next frame.")
    

    def get_plot_frame(self):
        """Capture the pyqtgraph plot as a NumPy BGR image, handling row padding."""
        pixmap = self.plot_manager.plot_widget.grab()
        qimage = pixmap.toImage().convertToFormat(QtGui.QImage.Format_RGB888)
        width = qimage.width()
        height = qimage.height()
        bytes_per_line = qimage.bytesPerLine()
        ptr = qimage.bits()
        ptr.setsize(qimage.byteCount())
        arr = np.array(ptr, dtype=np.uint8).reshape((height, bytes_per_line))
        # Only take the actual image width (width * 3 for RGB)
        arr = arr[:, :width * 3]
        arr = arr.reshape((height, width, 3))
        img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return img

    def stop_recording(self):
        self.recording_enabled = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.recording_filename = None
        print("Recording stopped")

    