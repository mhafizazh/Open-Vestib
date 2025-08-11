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



class TestSessionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
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
    def __init__(self):
        super().__init__()
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
            # If no "Right Eye" camera found, select second camera if available
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
        self.quit_button.clicked.connect(QtWidgets.QApplication.quit)

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

        # Handle countdown
        if self.countdown_active:
            elapsed = time.time() - self.countdown_start_time
            remaining = max(0, self.countdown_duration - elapsed)
            
            if remaining <= 0:
                # Countdown finished, start actual recording
                self.countdown_active = False
                self.recording_enabled = True
                print("Countdown finished! Recording started.")
            else:
                # Still in countdown, don't process data yet
                countdown_text = f"Recording starts in: {int(remaining) + 1}"
                
                # Combine frames first
                lh, lw = left_frame.shape[:2]
                rh, rw = right_frame.shape[:2]
                
                if lh != rh:
                    new_rw = int(rw * lh / rh)
                    right_frame = cv2.resize(right_frame, (new_rw, lh))
                
                combined_frame = np.hstack((left_frame, right_frame))
                
                # Draw countdown on combined frame
                self.draw_countdown_on_combined_frame(combined_frame, countdown_text)
                
                # Convert to Qt image and display
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
            
            # Add data to CSV handler if recording is enabled
            if self.recording_enabled and not self.countdown_active:
                frame_recorded = self.csv_handler.add_data_point(
                    left_x=self.last_left_x if isinstance(self.last_left_x, (int, float)) else 0,
                    left_y=self.last_left_y if isinstance(self.last_left_y, (int, float)) else 0,
                    right_x=self.last_right_x if isinstance(self.last_right_x, (int, float)) else 0,
                    right_y=self.last_right_y if isinstance(self.last_right_y, (int, float)) else 0
                )
                
                # Optional: Print frame recording status for debugging
                # if frame_recorded:
                #     print(f"Frame {self.csv_handler.get_current_frame_number()} recorded at {time.time() - self.start_time:.3f}s")
        
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

    def draw_countdown_on_combined_frame(self, combined_frame, text):
        """Draw countdown text on the combined frame in the top right corner."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8 # Smaller font scale
        thickness = 2
        color = (0, 255, 0) # Green color
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Calculate position to place text in the top right corner
        lh, lw = combined_frame.shape[:2]
        text_x = lw - text_width - 10 # 10 pixels from the right edge
        text_y = text_height + 10 # 10 pixels from the top edge
        
        # Draw text on the combined frame
        cv2.putText(combined_frame, text, (text_x, text_y), font, font_scale, color, thickness)

    def closeEvent(self, event):
        self.left_cam.release()
        self.right_cam.release()

        if self.video_writer:
            self.video_writer.release()
            
        # Stop CSV recording if active
        if self.csv_handler.get_recording_status():
            self.csv_handler.stop_recording()

        event.accept()
    

    
    def start_recording(self):
        # Show the test session dialog
        dialog = TestSessionDialog(self)
        if dialog.exec_() != QDialog.Accepted:
            print("Recording canceled by user.")
            return

        # Get session information
        self.session_info = dialog.get_session_info()
        patient_folder_name = dialog.get_patient_folder_name()
        test_folder_name = dialog.get_folder_name()
        
        # Validate required fields
        if not self.session_info['patient_first_name'] or not self.session_info['patient_last_name']:
            QtWidgets.QMessageBox.warning(self, "Missing Information", "Please enter both first and last name.")
            return
            
        # Create folder structure
        base_folder = "../result_videos"
        self.patient_folder = os.path.join(base_folder, patient_folder_name)
        self.test_folder = os.path.join(self.patient_folder, test_folder_name)
        
        # Create directories
        os.makedirs(self.patient_folder, exist_ok=True)
        os.makedirs(self.test_folder, exist_ok=True)
        
        # Set file paths
        self.session_folder = self.test_folder
        self.recording_filename = os.path.join(self.test_folder, f"{test_folder_name}.mp4")
        self.json_filename = os.path.join(self.test_folder, f"{test_folder_name}.json")
        
        # Start CSV recording
        self.csv_handler.start_recording(test_folder_name, base_folder=self.test_folder)
        
        # Save JSON metadata
        self.save_session_metadata()
        
        # Start countdown instead of recording immediately
        self.countdown_active = True
        self.countdown_start_time = time.time()
        self.recording_enabled = False  # Will be set to True when countdown finishes
        
        print(f"Session started for {patient_folder_name}")
        print(f"Test: {self.session_info['test_type']} - Trial {self.session_info['trial_number']}")
        print(f"Folder: {self.test_folder}")
        print(f"Countdown started. Recording will begin in {self.countdown_duration} seconds...")
        
    def save_session_metadata(self):
        """Save session metadata to JSON file"""
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
        self.countdown_active = False  # Stop countdown if active
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        # Stop CSV recording and save data
        self.csv_handler.stop_recording()

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

    def find_camera_by_name(self, name_to_find):
        """
        Find the index of a camera by its name (case-insensitive partial match)
        
        Args:
            name_to_find (str): Name to search for in camera names
            
        Returns:
            int or None: Index of the camera if found, None otherwise
        """
        for i, name in enumerate(self.camera_names):
            if name_to_find.lower() in name.lower():
                print(f"[INFO] Found camera '{name}' for '{name_to_find}' at index {i}")
                return i
        print(f"[WARNING] No camera found containing '{name_to_find}'")
        return None
    