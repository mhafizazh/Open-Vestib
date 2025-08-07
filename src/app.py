from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QFileDialog
from camera_handler import init_cameras, get_frames
from pygrabber.dshow_graph import FilterGraph
from pupil_detection import detect_pupil
from plotting import PlotManager
import numpy as np
import time
from collections import deque
import cv2
import os
import pandas as pd



class EyeTrackerApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.session_folder = None


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
        # self.left_cam_selector.currentIndexChanged.connect(self.init_selected_cameras)
        # self.right_cam_selector.currentIndexChanged.connect(self.init_selected_cameras)

        # UI setup
        self.setWindowTitle("Eye Tracking System")
        self.resize(1200, 800)
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QGridLayout(self.central_widget)
        self.video_label = QtWidgets.QLabel()
        self.layout.addWidget(self.video_label, 0, 0, 1, 2)

        # --- Add camera selection dropdowns here ---
        self.available_cameras = []
        self.camera_names = []
        graph = FilterGraph()
        device_list = graph.get_input_devices()
        for i, name in enumerate(device_list):
            self.available_cameras.append(i)
            self.camera_names.append(name)
        
        self.left_cam_selector = QtWidgets.QComboBox()
        self.right_cam_selector = QtWidgets.QComboBox()
        for idx, name in zip(self.available_cameras, self.camera_names):
            self.left_cam_selector.addItem(f"{name}", idx)
            self.right_cam_selector.addItem(f"{name}", idx)

        self.left_cam_selector.setCurrentIndex(0)
        self.right_cam_selector.setCurrentIndex(1 if len(self.available_cameras) > 1 else 0)

        self.layout.addWidget(QtWidgets.QLabel("Left Camera:"), 3, 0)
        self.layout.addWidget(self.left_cam_selector, 3, 1)
        self.layout.addWidget(QtWidgets.QLabel("Right Camera:"), 4, 0)
        self.layout.addWidget(self.right_cam_selector, 4, 1)
        self.left_cam_selector.currentIndexChanged.connect(self.init_selected_cameras)
        self.right_cam_selector.currentIndexChanged.connect(self.init_selected_cameras)
        # --- End camera selection dropdowns ---
        self.init_selected_cameras()
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
        text, ok = QtWidgets.QInputDialog.getText(
            self, "Start Recording", "Enter session name:"
        )
        if not ok or not text:
            print("Recording canceled by user.")
            return

        base_name = text.strip().replace(" ", "_")
        self.session_folder = os.path.join("../result_videos", base_name)
        os.makedirs(self.session_folder, exist_ok=True)

        # Set file paths
        self.recording_filename = os.path.join(self.session_folder, f"{base_name}.mp4")
        self.csv_filename = os.path.join(self.session_folder, f"{base_name}.csv")

        self.recording_enabled = True
        print(f"Recording started. Files will be saved to: {self.session_folder}")

    

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

        # Save CSV
        if self.session_folder:
            import pandas as pd

            # Get minimum length across all arrays
            lengths = [
                len(self.time_data),
                len(self.left_x_data),
                len(self.left_y_data),
                len(self.right_x_data),
                len(self.right_y_data),
            ]
            min_len = min(lengths)
            if min_len < 10:
                print("Not enough valid data to save.")
                return

            df = pd.DataFrame({
                'time': list(self.time_data)[:min_len],
                'left_x': list(self.left_x_data)[:min_len],
                'left_y': list(self.left_y_data)[:min_len],
                'right_x': list(self.right_x_data)[:min_len],
                'right_y': list(self.right_y_data)[:min_len],
            })
            df.to_csv(self.csv_filename, index=False)
            print(f"CSV data saved to: {self.csv_filename}")

        self.recording_filename = None
        self.session_folder = None
        print("Recording stopped")



    def init_selected_cameras(self):
        left_index = self.left_cam_selector.currentData()
        right_index = self.right_cam_selector.currentData()
    
        if left_index == right_index:
            QtWidgets.QMessageBox.warning(
                self,
                "Camera Selection Error",
                "Left and Right cameras must be different!"
            )
            # Revert right selector to a different camera
            if self.right_cam_selector.currentIndex() == 0 and self.right_cam_selector.count() > 1:
                self.right_cam_selector.setCurrentIndex(1)
            else:
                self.right_cam_selector.setCurrentIndex(0)
            return
    
        # Release current cameras if they exist
        if hasattr(self, 'left_cam') and self.left_cam is not None:
            self.left_cam.release()
        if hasattr(self, 'right_cam') and self.right_cam is not None:
            self.right_cam.release()
    
        self.left_cam = cv2.VideoCapture(left_index)
        self.right_cam = cv2.VideoCapture(right_index)

    