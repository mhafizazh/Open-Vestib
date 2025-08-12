import pyqtgraph as pg
from collections import deque
import numpy as np
import pandas as pd

def calculate_velocity(p1, p2, t1, t2):
    """Calculates velocity between two points."""
    if p1 is None or p2 is None or t1 is None or t2 is None or t1 == t2:
        return 0.0
    
    dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    delta_t = t2 - t1
    
    return dist / delta_t if delta_t > 0 else 0.0

class PlotManager:
    def __init__(self, layout, max_len=200):
        self.plot_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.plot_widget, 1, 0, 1, 2)

        self.left_x_data = deque(maxlen=max_len)
        self.left_y_data = deque(maxlen=max_len)
        self.right_x_data = deque(maxlen=max_len)
        self.right_y_data = deque(maxlen=max_len)
        self.ellipse_right_data = deque(maxlen=max_len)
        self.ellipse_left_data = deque(maxlen=max_len)
        self.time_data = deque(maxlen=max_len)
        self.left_velocity_data = deque(maxlen=max_len)
        self.right_velocity_data = deque(maxlen=max_len)

        self.x_plot = self.plot_widget.addPlot(title="X Position vs Time")
        self.x_plot.setLabel('left', "X Position")
        self.x_plot.setLabel('bottom', "Time")
        self.left_x_curve = self.x_plot.plot(pen='r', name="Left X")
        self.right_x_curve = self.x_plot.plot(pen='b', name="Right X")

        self.y_plot = self.plot_widget.addPlot(title="Y Position vs Time")
        self.y_plot.setLabel('left', "Y Position")
        self.y_plot.setLabel('bottom', "Time")
        self.left_y_curve = self.y_plot.plot(pen='r', name="Left Y")
        self.right_y_curve = self.y_plot.plot(pen='b', name="Right Y")

    def update(self, t, lx, ly, rx, ry, left_ellipse, right_ellipse):
        self.time_data.append(t)
        self.left_x_data.append(lx)
        self.left_y_data.append(ly)
        self.right_x_data.append(rx)
        self.right_y_data.append(ry)
        self.ellipse_left_data.append(left_ellipse)
        self.ellipse_right_data.append(right_ellipse)

        t_arr = np.array(self.time_data)
        self.left_x_curve.setData(t_arr, self.left_x_data)
        self.right_x_curve.setData(t_arr, self.right_x_data)
        self.left_y_curve.setData(t_arr, self.left_y_data)
        self.right_y_curve.setData(t_arr, self.right_y_data)

        # Calculate and store velocity
            # if len(self.time_data) > 1:
            #     prev_t = self.time_data[-2]
        left_vel = None
        right_vel = None
        if len(self.time_data) > 1:
            prev_t = self.time_data[-2]
            
            # Left eye velocity
            if lx is not None and self.left_x_data[-2] is not None:
                p1_left = (self.left_x_data[-2], self.left_y_data[-2])
                p2_left = (lx, ly)
                left_vel = calculate_velocity(p1_left, p2_left, prev_t, t)

            # Right eye velocity
            if rx is not None and self.right_x_data[-2] is not None:
                p1_right = (self.right_x_data[-2], self.right_y_data[-2])
                p2_right = (rx, ry)
                right_vel = calculate_velocity(p1_right, p2_right, prev_t, t)
        
        self.left_velocity_data.append(left_vel)
        self.right_velocity_data.append(right_vel)
            
        # Debug printing
        left_ellipse_size_str = f"({left_ellipse[1][0]:.2f}, {left_ellipse[1][1]:.2f})" if left_ellipse else "N/A"
        right_ellipse_size_str = f"({right_ellipse[1][0]:.2f}, {right_ellipse[1][1]:.2f})" if right_ellipse else "N/A"
        left_vel_str = f"{left_vel:.2f}" if left_vel is not None else "N/A"
        right_vel_str = f"{right_vel:.2f}" if right_vel is not None else "N/A"
        # print(f"Time: {t:.2f} | L Vel: {left_vel_str} | R Vel: {right_vel_str} | L Ellipse: {left_ellipse_size_str} | R Ellipse: {right_ellipse_size_str}")


    def save_to_csv(self, filename):
        # Ensure all deques have the same length for DataFrame creation
        max_len = len(self.time_data)
        
        # This is a safety check; ideally all deques should be the same length
        data = {
            'time': list(self.time_data),
            'left_x': list(self.left_x_data),
            'left_y': list(self.left_y_data),
            'right_x': list(self.right_x_data),
            'right_y': list(self.right_y_data),
            'left_ellipse': [str(e) for e in self.ellipse_left_data],
            'right_ellipse': [str(e) for e in self.ellipse_right_data],
            'left_velocity': list(self.left_velocity_data),
            'right_velocity': list(self.right_velocity_data),
        }
        
        # All lists must be the same length. We will check and log if they are not.
        for k, v in data.items():
            if len(v) != max_len:
                print(f"[WARNING] CSV Save: Mismatch in length for column '{k}'. Expected {max_len}, got {len(v)}.")
                # Pad the list to match the length of time_data
                data[k] = (list(v) + [None] * (max_len - len(v)))[:max_len]


        if max_len == 0:
            print("[INFO] No data to save.")
            return

        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"[INFO] Data saved to {filename}")