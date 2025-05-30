import cv2
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import sys
from collections import deque
import time

class EyeTrackerApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize camera streams
        self.left_cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.right_cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        
        # ROI size for each eye
        self.roi_width, self.roi_height = 150, 150
        
        # Data storage for plotting
        self.max_data_points = 200
        self.time_data = deque(maxlen=self.max_data_points)
        self.left_x_data = deque(maxlen=self.max_data_points)
        self.left_y_data = deque(maxlen=self.max_data_points)
        self.right_x_data = deque(maxlen=self.max_data_points)
        self.right_y_data = deque(maxlen=self.max_data_points)
        
        # Initialize with None values
        self.last_left_x = None
        self.last_left_y = None
        self.last_right_x = None
        self.last_right_y = None
        
        # Start time for tracking
        self.start_time = time.time()
        
        # Set up main window
        self.setWindowTitle("Eye Tracking System")
        self.resize(1200, 800)
        
        # Create central widget and layout
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QGridLayout(self.central_widget)
        
        # Create video display widgets
        self.video_label = QtWidgets.QLabel()
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.video_label, 0, 0, 1, 2)
        
        # Create threshold display widgets
        self.left_thresh_label = QtWidgets.QLabel()
        self.left_thresh_label.setAlignment(QtCore.Qt.AlignCenter)
        self.right_thresh_label = QtWidgets.QLabel()
        self.right_thresh_label.setAlignment(QtCore.Qt.AlignCenter)
        
        threshold_layout = QtWidgets.QHBoxLayout()
        threshold_layout.addWidget(self.left_thresh_label)
        threshold_layout.addWidget(self.right_thresh_label)
        self.layout.addLayout(threshold_layout, 1, 0, 1, 2)
        
        # Create PyQtGraph plots
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.plot_widget, 2, 0, 1, 2)
        
        # Create plots
        self.x_plot = self.plot_widget.addPlot(title="X Position vs Time")
        self.x_plot.setLabel('left', "X Position (pixels)")
        self.x_plot.setLabel('bottom', "Time (seconds)")
        self.x_plot.addLegend()
        
        self.y_plot = self.plot_widget.addPlot(title="Y Position vs Time")
        self.y_plot.setLabel('left', "Y Position (pixels)")
        self.y_plot.setLabel('bottom', "Time (seconds)")
        self.y_plot.addLegend()
        
        self.plot_widget.nextRow()
        
        # Create trajectory plot
        # self.traj_plot = self.plot_widget.addPlot(title="Eye Trajectory")
        # self.traj_plot.setLabel('left', "Y Position (pixels)")
        # self.traj_plot.setLabel('bottom', "X Position (pixels)")
        # self.traj_plot.addLegend()
        # self.traj_plot.setAspectLocked(True)
        
        # Create curves for plots
        self.left_x_curve = self.x_plot.plot(pen='r', name="Left Eye X")
        self.right_x_curve = self.x_plot.plot(pen='b', name="Right Eye X")
        
        self.left_y_curve = self.y_plot.plot(pen='r', name="Left Eye Y")
        self.right_y_curve = self.y_plot.plot(pen='b', name="Right Eye Y")
        
        # self.left_traj_curve = self.traj_plot.plot(pen='r', name="Left Eye")
        # self.right_traj_curve = self.traj_plot.plot(pen='b', name="Right Eye")
        
        # Timer for updating the display
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(30)  # Update every 30ms
    
    def detect_pupil(self, roi_frame, original_width, original_height):
        # Convert ROI to grayscale
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Binary threshold - may need adjustment
        _, threshold = cv2.threshold(blurred, 35, 255, cv2.THRESH_BINARY_INV)
        
        # Morphological operations to clean up small noise
        kernel = np.ones((3, 3), np.uint8)
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None, threshold
        
        # Filter contours by size
        min_contour_area = 50
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
        
        if not valid_contours:
            return None, None, threshold
        
        largest_contour = max(valid_contours, key=cv2.contourArea)
        
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return None, None, threshold
        
        # ROI coordinates
        roi_cX = int(M["m10"] / M["m00"])
        roi_cY = int(M["m01"] / M["m00"])
        
        # Convert to full frame coordinates
        cX = roi_cX + (original_width - roi_frame.shape[1]) // 2
        cY = roi_cY + (original_height - roi_frame.shape[0]) // 2
        
        return cX, cY, threshold
    
    def update_plots(self):
        time_array = np.array(self.time_data)
        left_x_array = np.array(self.left_x_data)
        left_y_array = np.array(self.left_y_data)
        right_x_array = np.array(self.right_x_data)
        right_y_array = np.array(self.right_y_data)

        # Plot left eye
        if len(time_array) == len(left_x_array) == len(left_y_array):
            self.left_x_curve.setData(time_array, left_x_array)
            self.left_y_curve.setData(time_array, left_y_array)
            # self.left_traj_curve.setData(left_x_array, left_y_array)

        # Plot right eye
        if len(time_array) == len(right_x_array) == len(right_y_array):
            self.right_x_curve.setData(time_array, right_x_array)
            self.right_y_curve.setData(time_array, right_y_array)
            # self.right_traj_curve.setData(right_x_array, right_y_array)
    
    def update(self):
        # Read frames from both cameras
        left_ret, left_frame = self.left_cam.read()
        right_ret, right_frame = self.right_cam.read()
        
        if not left_ret or not right_ret:
            print("Error reading from cameras")
            self.timer.stop()
            return
        
        # Flip frames if needed (depends on camera orientation)
        left_frame = cv2.flip(left_frame, 1)
        right_frame = cv2.flip(right_frame, 1)
        
        # Process left eye
        left_height, left_width = left_frame.shape[:2]
        left_x1 = (left_width - self.roi_width) // 2
        left_y1 = (left_height - self.roi_height) // 2
        left_x2 = left_x1 + self.roi_width
        left_y2 = left_y1 + self.roi_height
        left_roi = left_frame[left_y1:left_y2, left_x1:left_x2].copy()
        
        left_x, left_y, left_threshold = self.detect_pupil(left_roi, left_width, left_height)
        
        # Process right eye
        right_height, right_width = right_frame.shape[:2]
        right_x1 = (right_width - self.roi_width) // 2
        right_y1 = (right_height - self.roi_height) // 2
        right_x2 = right_x1 + self.roi_width
        right_y2 = right_y1 + self.roi_height
        right_roi = right_frame[right_y1:right_y2, right_x1:right_x2].copy()
        
        right_x, right_y, right_threshold = self.detect_pupil(right_roi, right_width, right_height)
        
        # Get current time
        current_time = time.time() - self.start_time
        
        # Update last known positions if detection is successful
        if left_x is not None and left_y is not None:
            self.last_left_x = left_x
            self.last_left_y = left_y
        if right_x is not None and right_y is not None:
            self.last_right_x = right_x
            self.last_right_y = right_y
        
        # Append time and left eye data if valid
        if self.last_left_x is not None and self.last_left_y is not None:
            self.time_data.append(current_time)
            self.left_x_data.append(self.last_left_x)
            self.left_y_data.append(self.last_left_y)

            # Only append right eye data if it's also valid
            if self.last_right_x is not None and self.last_right_y is not None:
                self.right_x_data.append(self.last_right_x)
                self.right_y_data.append(self.last_right_y)
        
        # Update plots
        self.update_plots()
        
        # Draw results on left frame
        if left_x is not None and left_y is not None:
            cv2.circle(left_frame, (left_x, left_y), 3, (0, 0, 255), -1)
            cv2.line(left_frame, (left_x-10, left_y), (left_x+10, left_y), (0, 255, 255), 1)
            cv2.line(left_frame, (left_x, left_y-10), (left_x, left_y+10), (0, 255, 255), 1)
            cv2.putText(left_frame, f"Left: ({left_x}, {left_y})", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw results on right frame
        if right_x is not None and right_y is not None:
            cv2.circle(right_frame, (right_x, right_y), 3, (0, 0, 255), -1)
            cv2.line(right_frame, (right_x-10, right_y), (right_x+10, right_y), (0, 255, 255), 1)
            cv2.line(right_frame, (right_x, right_y-10), (right_x, right_y+10), (0, 255, 255), 1)
            cv2.putText(right_frame, f"Right: ({right_x}, {right_y})", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Combine frames side by side for display
        combined_frame = np.hstack((left_frame, right_frame))
        
        # Convert to QImage and display
        rgb_image = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        self.video_label.setPixmap(QtGui.QPixmap.fromImage(qt_image))
        
        # # Display threshold images
        # for threshold, label in [(left_threshold, self.left_thresh_label), 
        #                        (right_threshold, self.right_thresh_label)]:
        #     rgb_thresh = cv2.cvtColor(threshold, cv2.COLOR_GRAY2RGB)
        #     h, w, ch = rgb_thresh.shape
        #     bytes_per_line = ch * w
        #     qt_thresh = QtGui.QImage(rgb_thresh.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        #     label.setPixmap(QtGui.QPixmap.fromImage(qt_thresh).scaled(
        #         300, 300, QtCore.Qt.KeepAspectRatio))
        
        # Print positions to console
        if left_x is not None and left_y is not None and right_x is not None and right_y is not None:
            print(f"Left Eye: ({left_x}, {left_y}) | Right Eye: ({right_x}, {right_y})")
    
    def closeEvent(self, event):
        self.left_cam.release()
        self.right_cam.release()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = EyeTrackerApp()
    window.show()
    sys.exit(app.exec_())