import pandas as pd
import numpy as np
from collections import deque
import os
import time

class CSVDataHandler:
    def __init__(self, max_data_points=200, target_fps=30):
        """
        Initialize CSV data handler for eye tracking data
        
        Args:
            max_data_points (int): Maximum number of data points to store in memory
            target_fps (int): Target frame rate for recording (default: 30 FPS)
        """
        self.max_data_points = max_data_points
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps  # Time between frames
        self.frame_number = 0
        self.start_time = None
        self.last_frame_time = None
        
        # Data storage with frame numbers
        self.frame_data = deque(maxlen=max_data_points)
        self.time_data = deque(maxlen=max_data_points)
        self.left_x_data = deque(maxlen=max_data_points)
        self.left_y_data = deque(maxlen=max_data_points)
        self.right_x_data = deque(maxlen=max_data_points)
        self.right_y_data = deque(maxlen=max_data_points)
        
        # Recording state
        self.is_recording = False
        self.session_folder = None
        self.csv_filename = None
        
    def start_recording(self, session_name, base_folder="../result_videos"):
        """
        Start recording session and create output folder
        
        Args:
            session_name (str): Name of the recording session
            base_folder (str): Base folder for saving results
        """
        self.session_folder = os.path.join(base_folder, session_name)
        os.makedirs(self.session_folder, exist_ok=True)
        
        self.csv_filename = os.path.join(self.session_folder, f"{session_name}.csv")
        self.is_recording = True
        self.frame_number = 0
        self.start_time = time.time()
        self.last_frame_time = self.start_time
        
        print(f"[INFO] CSV recording started at {self.target_fps} FPS. Data will be saved to: {self.csv_filename}")
        
    def stop_recording(self):
        """Stop recording and save data to CSV"""
        if not self.is_recording:
            print("[WARNING] No active recording to stop")
            return
            
        self.is_recording = False
        
        # Save data to CSV
        if self.csv_filename and len(self.frame_data) > 0:
            self.save_to_csv()
            print(f"[INFO] Recording stopped. Data saved to: {self.csv_filename}")
        else:
            print("[WARNING] No data to save")
            
        # Reset state
        self.session_folder = None
        self.csv_filename = None
        
    def should_record_frame(self):
        """
        Check if it's time to record the next frame based on target FPS
        
        Returns:
            bool: True if a new frame should be recorded
        """
        if not self.is_recording or self.start_time is None:
            return False
            
        current_time = time.time()
        time_since_last_frame = current_time - self.last_frame_time
        
        return time_since_last_frame >= self.frame_interval
        
    def add_data_point(self, left_x, left_y, right_x, right_y):
        """
        Add a new data point with frame number if it's time for the next frame
        
        Args:
            left_x (float): Left eye X position
            left_y (float): Left eye Y position  
            right_x (float): Right eye X position
            right_y (float): Right eye Y position
        """
        if not self.should_record_frame():
            return False
            
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # Increment frame number
        self.frame_number += 1
        self.last_frame_time = current_time
        
        # Store data
        self.frame_data.append(self.frame_number)
        self.time_data.append(elapsed_time)
        self.left_x_data.append(left_x)
        self.left_y_data.append(left_y)
        self.right_x_data.append(right_x)
        self.right_y_data.append(right_y)
        
        return True
        
    def save_to_csv(self):
        """Save current data to CSV file"""
        if not self.csv_filename:
            print("[ERROR] No CSV filename specified")
            return
            
        if len(self.frame_data) == 0:
            print("[WARNING] No data to save")
            return
            
        # Create DataFrame with frame numbers
        df = pd.DataFrame({
            'Frame#': list(self.frame_data),
            'Time(s)': list(self.time_data),
            'Left_X': list(self.left_x_data),
            'Left_Y': list(self.left_y_data),
            'Right_X': list(self.right_x_data),
            'Right_Y': list(self.right_y_data),
        })
        
        # Save to CSV
        df.to_csv(self.csv_filename, index=False)
        
        # Calculate actual frame rate
        if len(df) > 1:
            total_time = df['Time(s)'].iloc[-1] - df['Time(s)'].iloc[0]
            actual_fps = len(df) / total_time if total_time > 0 else 0
            print(f"[INFO] CSV data saved: {len(df)} frames to {self.csv_filename}")
            print(f"[INFO] Actual frame rate: {actual_fps:.2f} FPS (target: {self.target_fps} FPS)")
        else:
            print(f"[INFO] CSV data saved: {len(df)} frame to {self.csv_filename}")
        
    def get_data_for_plotting(self):
        """
        Get data in format compatible with PlotManager
        
        Returns:
            dict: Dictionary containing time and position data arrays
        """
        return {
            'time': list(self.time_data),
            'left_x': list(self.left_x_data),
            'left_y': list(self.left_y_data),
            'right_x': list(self.right_x_data),
            'right_y': list(self.right_y_data),
        }
        
    def get_current_frame_number(self):
        """Get current frame number"""
        return self.frame_number
        
    def get_recording_status(self):
        """Get current recording status"""
        return self.is_recording
        
    def get_session_info(self):
        """Get current session information"""
        return {
            'session_folder': self.session_folder,
            'csv_filename': self.csv_filename,
            'frame_count': self.frame_number,
            'is_recording': self.is_recording,
            'target_fps': self.target_fps
        }
        
    def clear_data(self):
        """Clear all stored data"""
        self.frame_data.clear()
        self.time_data.clear()
        self.left_x_data.clear()
        self.left_y_data.clear()
        self.right_x_data.clear()
        self.right_y_data.clear()
        self.frame_number = 0
        print("[INFO] All data cleared") 