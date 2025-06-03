from PyQt5 import QtWidgets
from app import EyeTrackerApp
import sys

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = EyeTrackerApp()
    window.show()
    sys.exit(app.exec_())
