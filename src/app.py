from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QFileDialog, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox, QSpinBox, QPushButton, QFormLayout
from camera_handler import init_cameras, get_frames
from pygrabber.dshow_graph import FilterGraph
from pupil_detection import detect_pupil
from plotting import PlotManager
from csv_handler import CSVDataHandler
import numpy as np
import time
from collections import deque
import cv2
import os
import pandas as pd
import json
from datetime import datetime
from prediction import estimate_bppv_likelihood


def ellipse_size(e):
    if e is None:
        return np.nan
    # e = ((cx, cy), (MA, ma), angle)  — MA, ma are diameters in pixels
    (_, _), (MA, ma), _ = e
    return round((float(MA) + float(ma)) / 2.0, 2)


class TestSessionDialog(QDialog):
    def __init__(self, base_path, parent=None):
        super().__init__(parent)
        self.base_path = base_path
        self.setWindowTitle("Test Session Information")
        self.setModal(True)
        self.resize(400, 300)
        
        # Initialize data
        self.patient_first_name = ""
        self.patient_last_name = ""
        self.test_type = ""
        self.trial_number = 1
        self.session_date = datetime.now().strftime("%Y%m%d")
        self.session_time = datetime.now().strftime("%H:%M:%S")
        
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Form layout for inputs
        form_layout = QFormLayout()
        
        # Patient first name
        self.first_name_edit = QLineEdit()
        self.first_name_edit.setPlaceholderText("Enter patient's first name")
        form_layout.addRow("First Name:", self.first_name_edit)
        
        # Patient last name
        self.last_name_edit = QLineEdit()
        self.last_name_edit.setPlaceholderText("Enter patient's last name")
        form_layout.addRow("Last Name:", self.last_name_edit)
        
        # Test type dropdown
        self.test_type_combo = QComboBox()
        test_types = [
            "Left Dix Halpike",
            "Right Dix Halpike", 
            "Left BBQ Roll",
            "Right BBQ Roll"
        ]
        self.test_type_combo.addItems(test_types)
        form_layout.addRow("Test Type:", self.test_type_combo)
        
        # Trial number
        self.trial_spinbox = QSpinBox()
        self.trial_spinbox.setMinimum(1)
        self.trial_spinbox.setMaximum(99)
        self.trial_spinbox.setValue(1)
        form_layout.addRow("Trial Number:", self.trial_spinbox)
        
        layout.addLayout(form_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("Start Recording")
        self.cancel_button = QPushButton("Cancel")
        
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
    def get_session_info(self):
        """Get the session information as a dictionary"""
        return {
            'patient_first_name': self.first_name_edit.text().strip(),
            'patient_last_name': self.last_name_edit.text().strip(),
            'test_type': self.test_type_combo.currentText(),
            'trial_number': self.trial_spinbox.value(),
            'session_date': self.session_date,
            'session_time': self.session_time,
            'datetime': datetime.now().isoformat()
        }
        
    def get_folder_name(self):
        """Generate the folder name for the test session"""
        test_type_clean = self.test_type_combo.currentText().replace(" ", "")
        trial_num = self.trial_spinbox.value()
        return f"{test_type_clean}_{trial_num}_{self.session_date}"
        
    def get_patient_folder_name(self):
        """Generate the patient folder name"""
        first_name = self.first_name_edit.text().strip()
        last_name = self.last_name_edit.text().strip()
        return f"{first_name}_{last_name}"


class EyeTrackerApp(QtWidgets.QMainWindow):
    def __init__(self, base_path):
        super().__init__()
        self.base_path = base_path  # Add base_path to the main app
        self.session_folder = None
        
        # Session information
        self.session_info = None
        self.patient_folder = None
        self.test_folder = None

        # Countdown variables
        self.countdown_active = False
        self.countdown_start_time = None
        self.countdown_duration = 3  # 3 seconds

        self.left_cam, self.right_cam = init_cameras()
        self.roi_width, self.roi_height = 150, 150
        self.max_data_points = 200
        
        # Initialize CSV data handler
        self.csv_handler = CSVDataHandler(max_data_points=self.max_data_points, target_fps=30)
        
        # Keep existing data structures for backward compatibility with plotting
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

        # --- Add camera selection dropdowns ---
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

        # Auto-select cameras based on names
        left_cam_index = self.find_camera_by_name("Left Eye")
        right_cam_index = self.find_camera_by_name("Right Eye")
        
        # Set default selections
        if left_cam_index is not None:
            self.left_cam_selector.setCurrentIndex(left_cam_index)
        else:
            self.left_cam_selector.setCurrentIndex(0)
            
        if right_cam_index is not None:
            self.right_cam_selector.setCurrentIndex(right_cam_index)
        else:
            self.right_cam_selector.setCurrentIndex(1 if len(self.available_cameras) > 1 else 0)

        self.layout.addWidget(QtWidgets.QLabel("Left Camera:"), 3, 0)
        self.layout.addWidget(self.left_cam_selector, 3, 1)
        self.layout.addWidget(QtWidgets.QLabel("Right Camera:"), 4, 0)
        self.layout.addWidget(self.right_cam_selector, 4, 1)
        self.left_cam_selector.currentIndexChanged.connect(self.init_selected_cameras)
        self.right_cam_selector.currentIndexChanged.connect(self.init_selected_cameras)
        self.init_selected_cameras()

        # Buttons
        self.start_button = QtWidgets.QPushButton("Start")
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.quit_button = QtWidgets.QPushButton("Quit")

        self.start_button.clicked.connect(self.start_recording)
        self.stop_button.clicked.connect(self.stop_recording)
        self.quit_button.clicked.connect(QtWidgets.QApplication.quit)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.quit_button) 

        self.layout.addLayout(button_layout, 2, 0, 1, 2)

        # Create output directory
        os.makedirs(os.path.join(self.base_path, "result_videos"), exist_ok=True)

        # Initialize screen recording writer
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

        # Handle countdown
        if self.countdown_active:
            elapsed = time.time() - self.countdown_start_time
            remaining = max(0, self.countdown_duration - elapsed)
            
            if remaining <= 0:
                self.countdown_active = False
                self.recording_enabled = True
                print("Countdown finished! Recording started.")
            else:
                countdown_text = f"Recording starts in: {int(remaining) + 1}"
                lh, lw = left_frame.shape[:2]
                rh, rw = right_frame.shape[:2]
                
                if lh != rh:
                    new_rw = int(rw * lh / rh)
                    right_frame = cv2.resize(right_frame, (new_rw, lh))
                
                combined_frame = np.hstack((left_frame, right_frame))
                self.draw_countdown_on_combined_frame(combined_frame, countdown_text)
                
                rgb_image = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                self.video_label.setPixmap(QtGui.QPixmap.fromImage(qt_image))
                return

        # Process left eye
        lh, lw = left_frame.shape[:2]
        lx1 = (lw - self.roi_width) // 2
        ly1 = (lh - self.roi_height) // 2
        left_roi = left_frame[ly1:ly1+self.roi_height, lx1:lx1+self.roi_width].copy()
        left_x, left_y, left_ellipse, _ = detect_pupil(left_roi, lw, lh)
        left_size  = ellipse_size(left_ellipse) 

        # Process right eye
        rh, rw = right_frame.shape[:2]
        rx1 = (rw - self.roi_width) // 2
        ry1 = (rh - self.roi_height) // 2
        right_roi = right_frame[ry1:ry1+self.roi_height, rx1:rx1+self.roi_width].copy()
        right_x, right_y, right_ellipse, _ = detect_pupil(right_roi, rw, rh)
        right_size  = ellipse_size(right_ellipse) 

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
            if isinstance(self.last_left_x, (int, float)) and isinstance(self.last_left_y, (int, float)):
                self.left_x_data.append(self.last_left_x)
                self.left_y_data.append(self.last_left_y)
        
            if self.last_right_x is not None and self.last_right_y is not None:
                if isinstance(self.last_right_x, (int, float)) and isinstance(self.last_right_y, (int, float)):
                    self.right_x_data.append(self.last_right_x)
                    self.right_y_data.append(self.last_right_y)
            # Add data to CSV
            if self.recording_enabled and not self.countdown_active:
                # --- (1) compute pixel-speed (magnitude) in px/sec for each eye ---
                # need at least 2 samples to compute velocity
                if len(self.time_data) >= 2:
                    dt = self.time_data[-1] - self.time_data[-2]
                else:
                    dt = 0

                # Left eye speed
                if dt > 0 and len(self.left_x_data) >= 2 and len(self.left_y_data) >= 2:
                    ldx = (self.left_x_data[-1] - self.left_x_data[-2])
                    ldy = (self.left_y_data[-1] - self.left_y_data[-2])
                    lxv = (ldx**2 + ldy**2) ** 0.5 / dt
                else:
                    lxv = 0

                # Right eye speed
                if dt > 0 and len(self.right_x_data) >= 2 and len(self.right_y_data) >= 2:
                    rdx = (self.right_x_data[-1] - self.right_x_data[-2])
                    rdy = (self.right_y_data[-1] - self.right_y_data[-2])
                    rxv = (rdx**2 + rdy**2) ** 0.5 / dt
                else:
                    rxv = 0

                # --- (2) make ellipse serializable (OpenCV ellipse is a tuple of tuples) ---
                # def ellipse_to_str(e):
                #     if e is None:
                #         return ""
                #     # e = ((cx, cy), (axis1, axis2), angle)
                #     return f"{e[0][0]:.2f},{e[0][1]:.2f}|{e[1][0]:.2f},{e[1][1]:.2f}|{e[2]:.2f}"

                # left_ellipse_str  = ellipse_to_str(left_ellipse)
                # right_ellipse_str = ellipse_to_str(right_ellipse)

                # --- (3) write EVERYTHING to CSV ---
                frame_recorded = self.csv_handler.add_data_point(
                    left_x=self.last_left_x if isinstance(self.last_left_x, (int, float)) else 0,
                    left_y=self.last_left_y if isinstance(self.last_left_y, (int, float)) else 0,
                    right_x=self.last_right_x if isinstance(self.last_right_x, (int, float)) else 0,
                    right_y=self.last_right_y if isinstance(self.last_right_y, (int, float)) else 0,
                    left_velocity=lxv,
                    right_velocity=rxv,
                    left_size=left_size,          # <<< ADDED
                    right_size=right_size         # <<< ADDED
                )




        
        self.plot_manager.update(
            t=current_time,
            lx=self.last_left_x if isinstance(self.last_left_x, (int, float)) else 0,
            ly=self.last_left_y if isinstance(self.last_left_y, (int, float)) else 0,
            rx=self.last_right_x if isinstance(self.last_right_x, (int, float)) else 0,
            ry=self.last_right_y if isinstance(self.last_right_y, (int, float)) else 0,
            left_ellipse=left_ellipse,
            right_ellipse=right_ellipse
        )

        # Draw detection markers
        if left_x is not None and left_y is not None:
            cv2.line(left_frame, (left_x - 10, left_y), (left_x + 10, left_y), (0, 255, 255), 1)
            cv2.line(left_frame, (left_x, left_y - 10), (left_x, left_y + 10), (0, 255, 255), 1)

        if right_x is not None and right_y is not None:
            cv2.line(right_frame, (right_x - 10, right_y), (right_x + 10, right_y), (0, 255, 255), 1)
            cv2.line(right_frame, (right_x, right_y - 10), (right_x, right_y + 10), (0, 255, 255), 1)

                # Draw detection markers
        if left_ellipse is not None:
            # The ellipse object is ((center_x, center_y), (minor_axis, major_axis), angle)
            # We need to adjust the center coordinates back to the full frame
            ellipse_center_x = int(left_ellipse[0][0] + lx1)
            ellipse_center_y = int(left_ellipse[0][1] + ly1)
            
            # Use the average of the axes for the circle's radius
            radius = int((left_ellipse[1][0] + left_ellipse[1][1]) / 4)
            
            # Draw the circle on the left frame
            cv2.circle(left_frame, (ellipse_center_x, ellipse_center_y), radius, (0, 255, 0), 2)

        if right_ellipse is not None:
            # Adjust the center coordinates for the right eye
            ellipse_center_x = int(right_ellipse[0][0] + rx1)
            ellipse_center_y = int(right_ellipse[0][1] + ry1)
            
            # Calculate the radius
            radius = int((right_ellipse[1][0] + right_ellipse[1][1]) / 4)
            
            # Draw the circle on the right frame
            cv2.circle(right_frame, (ellipse_center_x, ellipse_center_y), radius, (0, 255, 0), 2)

        # Combine and display
        lh, lw = left_frame.shape[:2]
        rh, rw = right_frame.shape[:2]
        
        if lh != rh:
            new_rw = int(rw * lh / rh)
            right_frame = cv2.resize(right_frame, (new_rw, lh))
        
        combined_frame = np.hstack((left_frame, right_frame))
        plot_frame = self.get_plot_frame()
        
        if plot_frame.shape[1] != combined_frame.shape[1]:
            plot_frame = cv2.resize(plot_frame, (combined_frame.shape[1], plot_frame.shape[0]))
        
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

        if self.recording_enabled and self.video_writer:
            self.video_writer.write(final_frame)

        rgb_image = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        self.video_label.setPixmap(QtGui.QPixmap.fromImage(qt_image))

    def draw_countdown_on_combined_frame(self, combined_frame, text):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        color = (0, 255, 0)
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        lh, lw = combined_frame.shape[:2]
        text_x = lw - text_width - 10
        text_y = text_height + 10
        cv2.putText(combined_frame, text, (text_x, text_y), font, font_scale, color, thickness)

    def closeEvent(self, event):
        self.left_cam.release()
        self.right_cam.release()
        if self.video_writer:
            self.video_writer.release()
        if self.csv_handler.get_recording_status():
            self.csv_handler.stop_recording()
        event.accept()

    def start_recording(self):
        dialog = TestSessionDialog(self.base_path, self)  # Pass base_path and parent
        if dialog.exec_() != QDialog.Accepted:
            print("Recording canceled by user.")
            return

        self.session_info = dialog.get_session_info()
        patient_folder_name = dialog.get_patient_folder_name()
        test_folder_name = dialog.get_folder_name()
        
        if not self.session_info['patient_first_name'] or not self.session_info['patient_last_name']:
            QtWidgets.QMessageBox.warning(self, "Missing Information", "Please enter both first and last name.")
            return
            
        base_folder = os.path.join(self.base_path, "result_videos")
        self.patient_folder = os.path.join(base_folder, patient_folder_name)
        self.test_folder = os.path.join(self.patient_folder, test_folder_name)
        
        os.makedirs(self.patient_folder, exist_ok=True)
        os.makedirs(self.test_folder, exist_ok=True)
        
        self.session_folder = self.test_folder
        self.recording_filename = os.path.join(self.test_folder, f"{test_folder_name}.mp4")
        self.json_filename = os.path.join(self.test_folder, f"{test_folder_name}.json")
        
        self.csv_handler.start_recording(test_folder_name, base_folder=self.test_folder)
        self.save_session_metadata()
        
        self.countdown_active = True
        self.countdown_start_time = time.time()
        self.recording_enabled = False
        
        print(f"Session started for {patient_folder_name}")
        print(f"Test: {self.session_info['test_type']} - Trial {self.session_info['trial_number']}")
        print(f"Folder: {self.test_folder}")
        print(f"Countdown started. Recording will begin in {self.countdown_duration} seconds...")
        
    def save_session_metadata(self):
        if not self.session_info or not self.json_filename:
            return
            
        metadata = {
            'session_info': self.session_info,
            'folder_structure': {
                'patient_folder': self.patient_folder,
                'test_folder': self.test_folder,
                'video_file': self.recording_filename,
                'csv_file': self.csv_handler.csv_filename,
                'json_file': self.json_filename
            },
            'recording_settings': {
                'target_fps': self.csv_handler.target_fps,
                'roi_width': self.roi_width,
                'roi_height': self.roi_height,
                'max_data_points': self.max_data_points
            },
            'camera_info': {
                'left_camera': self.camera_names[self.left_cam_selector.currentIndex()] if self.camera_names else "Unknown",
                'right_camera': self.camera_names[self.right_cam_selector.currentIndex()] if self.camera_names else "Unknown"
            }
        }
        
        try:
            with open(self.json_filename, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"[INFO] Session metadata saved to: {self.json_filename}")
        except Exception as e:
            print(f"[ERROR] Failed to save metadata: {e}")

    def get_plot_frame(self):
        pixmap = self.plot_manager.plot_widget.grab()
        qimage = pixmap.toImage().convertToFormat(QtGui.QImage.Format_RGB888)
        width = qimage.width()
        height = qimage.height()
        bytes_per_line = qimage.bytesPerLine()
        ptr = qimage.bits()
        ptr.setsize(qimage.byteCount())
        arr = np.array(ptr, dtype=np.uint8).reshape((height, bytes_per_line))
        arr = arr[:, :width * 3]
        arr = arr.reshape((height, width, 3))
        img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return img

    def stop_recording(self):
        self.recording_enabled = False
        self.countdown_active = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        self.csv_handler.stop_recording()
        
        if self.session_info:
            self.save_final_metadata()

        self.recording_filename = None
        self.session_folder = None
        self.session_info = None
        self.patient_folder = None
        self.test_folder = None
        print("Recording stopped")
        
    def save_final_metadata(self):
        if not self.session_info or not self.json_filename:
            return
            
        try:
            with open(self.json_filename, 'r') as f:
                metadata = json.load(f)
        except:
            metadata = {}
            
        metadata['recording_results'] = {
            'total_frames_recorded': self.csv_handler.get_current_frame_number(),
            'recording_duration': self.csv_handler.time_data[-1] if self.csv_handler.time_data else 0,
            'actual_fps': len(self.csv_handler.frame_data) / metadata.get('recording_results', {}).get('recording_duration', 1) if self.csv_handler.frame_data else 0,
            'recording_completed': True,
            'completion_time': datetime.now().isoformat()
        }
        
        try:
            with open(self.json_filename, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"[INFO] Final metadata updated: {self.json_filename}")
        except Exception as e:
            print(f"[ERROR] Failed to update final metadata: {e}")

    def init_selected_cameras(self):
        left_index = self.left_cam_selector.currentData()
        right_index = self.right_cam_selector.currentData()
    
        if left_index == right_index:
            QtWidgets.QMessageBox.warning(
                self,
                "Camera Selection Error",
                "Left and Right cameras must be different!"
            )
            if self.right_cam_selector.currentIndex() == 0 and self.right_cam_selector.count() > 1:
                self.right_cam_selector.setCurrentIndex(1)
            else:
                self.right_cam_selector.setCurrentIndex(0)
            return
    
        if hasattr(self, 'left_cam') and self.left_cam is not None:
            self.left_cam.release()
        if hasattr(self, 'right_cam') and self.right_cam is not None:
            self.right_cam.release()
    
        self.left_cam = cv2.VideoCapture(left_index)
        self.right_cam = cv2.VideoCapture(right_index)

    def find_camera_by_name(self, name_to_find):
        for i, name in enumerate(self.camera_names):
            if name_to_find.lower() in name.lower():
                print(f"[INFO] Found camera '{name}' for '{name_to_find}' at index {i}")
                return i
        print(f"[WARNING] No camera found containing '{name_to_find}'")
        return None
    