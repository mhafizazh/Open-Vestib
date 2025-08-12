from PyQt5 import QtWidgets
from app import EyeTrackerApp
import sys
import os

if __name__ == "__main__":
    if getattr(sys, 'frozen', False):
        # If the application is run as a bundle, the base path is the executable's directory
        base_path = os.path.dirname(sys.executable)
    else:
        # If running from source, the base path is the project root
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    app = QtWidgets.QApplication(sys.argv)
    window = EyeTrackerApp(base_path=base_path)
    window.show()
    sys.exit(app.exec_())