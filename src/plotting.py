import pyqtgraph as pg
from collections import deque
import numpy as np
import pandas as pd

class PlotManager:
    def __init__(self, layout, max_len=200):
        self.plot_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.plot_widget, 1, 0, 1, 2)

        self.left_x_data = deque(maxlen=max_len)
        self.left_y_data = deque(maxlen=max_len)
        self.right_x_data = deque(maxlen=max_len)
        self.right_y_data = deque(maxlen=max_len)
        self.time_data = deque(maxlen=max_len)

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

    def update(self, t, lx, ly, rx, ry):
        self.time_data.append(t)
        self.left_x_data.append(lx)
        self.left_y_data.append(ly)
        self.right_x_data.append(rx)
        self.right_y_data.append(ry)

        t_arr = np.array(self.time_data)
        self.left_x_curve.setData(t_arr, self.left_x_data)
        self.right_x_curve.setData(t_arr, self.right_x_data)
        self.left_y_curve.setData(t_arr, self.left_y_data)
        self.right_y_curve.setData(t_arr, self.right_y_data)

    def save_to_csv(self, filename):
        df = pd.DataFrame({
            'time': list(self.time_data),
            'left_x': list(self.left_x_data),
            'left_y': list(self.left_y_data),
            'right_x': list(self.right_x_data),
            'right_y': list(self.right_y_data),
        })
        df.to_csv(filename, index=False)
        print(f"[INFO] Data saved to {filename}")
