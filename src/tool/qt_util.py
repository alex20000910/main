from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QProgressBar, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt
import time
from MDC_cut_utility import IconManager
from PyQt5.QtGui import QPixmap, QIcon
from base64 import b64decode
import matplotlib as mpl
from matplotlib.colors import Colormap

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        icon = IconManager().icon
        pixmap = QPixmap()
        pixmap.loadFromData(b64decode(icon))
        qicon = QIcon(pixmap)
        self.icon = qicon
        self.setWindowIcon(qicon)
        self.setStyleSheet("""
            QWidget {
                background-color: #000;
                color: #EEE;
                font-family: Arial;
                font-size: 24px;
            }
            QMessageBox { font-size: 18pt; }
            QStatusBar {
                background-color: #D7D7D7;
                color: #222;
                font-size: 30px;
            }
            QMenuBar, QMenu, QSlider, QSpinBox, QLineEdit, QLabel, QRadioButton {
                background-color: #000;
                color: #EEE;
            }
            QLineEdit {
                qproperty-alignment: 'AlignCenter';
            }
            QLineEdit:disabled {
                background-color: #444;
                color: #AAA;
            }
            QMenuBar {
                padding: 8px;
            }
            QPushButton {
                background-color: #333;
                color: #EEE;
                font-family: Arial;
                font-weight: bold;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #777;
            }
            QRadioButton::indicator {
                background-color: #999;
                width: 16px;
                height: 16px;
                border-radius: 8px;
            }
            QRadioButton::indicator:checked {
                background-color: #FCFCFC;
                width: 20px;
                height: 20px;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #555;
                color: #FFD700;
            }
            QMenuBar::item {
                background-color: #000;
                color: #EEE;
                font-family: Arial;
            }
            QMenuBar::item:selected {
                background: #555;
                color: #FFD700;
            }
            QMenu {
                background-color: #222;
                color: #EEE;
                font-family: Arial;
            }
            QMenu::item {
                background: #222;
                color: #EEE;
                padding: 6px 24px;
                font-family: Arial;
            }
            QMenu::item:selected {
                background: #FFD700;
                color: #222;
            }
            QToolTip {
                background-color: #222;
                color: #EEE;
                border: 5px solid white;
                font-size: 20pt;
            }
        """)


class ProgressDialog(QDialog):
    def __init__(self, max_val=100, qicon=None):
        super().__init__()
        self.setStyleSheet("""
            QWidget {
                background-color: #222;
                color: #EEE;
                font-family: Arial;
                font-size: 24px;
            }
        """)
        self.setWindowTitle('Progress')
        self.setWindowIcon(qicon)
        self.progress = QProgressBar(self)
        self.progress.setMinimum(0)
        self.progress.setMaximum(max_val)
        self.progress.setValue(0)
        self.label = QLabel(f"Progress: {self.progress.value()}/{self.progress.maximum()}", self)
        self.label.setAlignment(Qt.AlignCenter)
        vbox = QVBoxLayout()
        vbox.addWidget(self.progress)
        vbox.addWidget(self.label)
        self.setLayout(vbox)
        QApplication.processEvents()  # Update the GUI immediately
    
    def increaseProgress(self, text=None):
        value = self.progress.value()
        self.progress.setValue(value + 1)
        if value < self.progress.maximum()-1:
            if text:
                self.label.setText(text)
            else:
                self.label.setText(f"Progress: {self.progress.value()}/{self.progress.maximum()}")
            QApplication.processEvents()
        elif value == self.progress.maximum()-1:
            if text:
                self.label.setText(text)
            else:
                self.label.setText('Almost Done! Please Wait...')
            QApplication.processEvents()
            time.sleep(0.5)
        self.raise_()
        self.activateWindow()

def cmap_register(cmap: Colormap):
    try:
        mpl.colormaps.register(cmap)
    except Exception as e:
        print(f"Colormap {cmap.name} registration failed: {e}")