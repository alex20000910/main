from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QProgressBar, QLabel, QVBoxLayout, QSystemTrayIcon, QMenu
from PyQt5.QtCore import Qt
import time
from MDC_cut_utility import IconManager, MenuIconManager
from PyQt5.QtGui import QPixmap, QIcon
from base64 import b64decode
import matplotlib as mpl
from matplotlib.colors import Colormap
import os
import subprocess
from PIL import Image
import numpy as np

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

class SystemTrayIcon(QSystemTrayIcon):
    def __init__(self, icon, parent=None):
        super().__init__(icon, parent)
        menu = QMenu(parent)
        showAction = menu.addAction("Show")
        showAction.triggered.connect(self.focus)
        exitAction = menu.addAction("Exit")
        exitAction.triggered.connect(parent.close)
        self.setContextMenu(menu)
    
    def focus(self):
        self.parent().raise_()
        self.parent().activateWindow()

def getTrayIcon(icon_none:str, icon_light:str, icon_dark:str) -> QIcon:
    icon_manager = MenuIconManager(qt=True)
    if os.name == 'posix': # macOS
        try:
            result = subprocess.run(
                ['osascript', '-e', 
                'tell application "Finder" to get POSIX path of (desktop picture as alias)'],
                capture_output=True,
                text=True
            )
            if result.stdout.strip():
                out = result.stdout.strip()
            else:
                out = None
        except: # 純色背景無桌布圖檔
            out = None
        try:
            img = Image.open(out)
            w, h = img.width, img.height//20
            cut = img.crop((w//2, 0, w, h))
            cut.thumbnail((w//30, h//15))
            img_gray = cut.convert('L')
            brightness = np.mean(np.array(img_gray))
            if brightness > 128:
                icon_name = icon_light
            else:
                icon_name = icon_dark
        except Exception as e:
            icon_name = icon_none
    else: # Windows
        try:
            import winreg
            registry_path = r'Software\Microsoft\Windows\CurrentVersion\Themes\Personalize'
            registry_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, registry_path)
            value, _ = winreg.QueryValueEx(registry_key, 'AppsUseLightTheme')
            winreg.CloseKey(registry_key)
            icon_name = icon_light if value == 1 else icon_dark
        except:
            icon_name = icon_none
        
    icon = icon_manager.gen_icon(icon_name)[0]
    tray_icon_pixmap = QPixmap()
    tray_icon_pixmap.loadFromData(b64decode(icon))
    return QIcon(tray_icon_pixmap)