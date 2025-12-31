from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider, QLabel, QStatusBar,
    QSpinBox, QPushButton, QHBoxLayout, QLineEdit, QMenuBar, QAction, QRadioButton,
    QButtonGroup, QFileDialog, QProgressBar, QDialog, QTextEdit, QComboBox, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QPainter, QFont, QColor, QIcon, QCursor, QFontMetrics
import pyqtgraph as pg

import os, inspect, time, sys, argparse
from base64 import b64decode
from ctypes import windll

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl

cdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
if os.name == 'nt':
    cdir = cdir[0].upper() + cdir[1:]
app_name = os.path.basename(inspect.stack()[0].filename).removesuffix('.py')
for i in range(5):
    cdir = os.path.dirname(cdir)
    if '.MDC_cut' in os.listdir(cdir):
        break
sys.path.append(os.path.join(cdir, '.MDC_cut'))

from MDC_cut_utility import *
from tool.loader import loadfiles, mloader, eloader, tkDnD_loader, file_loader, data_loader, load_h5, load_json, load_npz, load_txt
from tool.spectrogram import spectrogram, lfs_exp_casa
from tool.util import laplacian_filter  # for originpro: from MDC_cut import *
from tool.util import app_param, MDC_param, EDC_param, Button, MenuIconManager, ToolTip_util, IconManager, origin_util, motion, plots_util, exp_util
from tool.SO_Fitter import SO_Fitter
from tool.CEC import CEC, call_cec
from tool.window import AboutWindow, EmodeWindow, ColormapEditorWindow, c_attr_window, c_name_window, c_excitation_window, c_description_window, VersionCheckWindow, CalculatorWindow, Plot1Window, Plot1Window_MDC_curves, Plot1Window_Second_Derivative, Plot3Window

def get_hwnd():
    try:
        with open('hwnd', 'r') as f:
            hwnd = int(f.read().strip())
        return hwnd
    except Exception:
        return find_window()

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

class c_fermi_level(QDialog):
    def __init__(self, vfe, icon):
        super().__init__()
        self.setStyleSheet("""
            QWidget {
                background-color: #222;
                color: #EEE;
                font-family: Arial;
                font-size: 30px;
                padding: 10px;
            }
        """)
        self.setWindowTitle("Set Fermi Level")
        self.setWindowIcon(icon)
        layout = QHBoxLayout()
        self.label = QLabel("Enter Fermi Level (eV):")
        self.input = QLineEdit()
        self.ovfe = vfe
        self.vfe = vfe
        self.eflag = False
        self.input.setText(str(self.ovfe))
        self.input.setFocus()
        self.input.selectAll()
        self.ok_button = QPushButton("OK")
        self.ok_button.setStyleSheet("""
            QPushButton {
                background-color: #333;
                color: #EEE;
                font-family: Arial;
                font-weight: bold;
                border: 2px solid #555;
                border-radius: 5px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #555;
                color: #FFD700;
            }
        """)
        self.ok_button.clicked.connect(self.accept)
        self.keyPressEvent = lambda event: self.accept(event) if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter else None
        layout.addWidget(self.label)
        layout.addWidget(self.input)
        layout.addWidget(self.ok_button)
        self.setLayout(layout)
        
        
        self.force_to_front()
        
    def showEvent(self, event):
        """當視窗顯示時確保 QLineEdit 獲得焦點"""
        super().showEvent(event)
        # 延遲一下確保視窗完全顯示後再設定焦點
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(100, lambda: self.input.setFocus())
        QTimer.singleShot(100, lambda: self.input.selectAll())
    
    def force_to_front(self):
        """使用 Windows API 強制視窗置頂"""
        hwnd = int(self.winId())
        SWP_NOMOVE = 0x0002
        SWP_NOSIZE = 0x0001
        HWND_TOPMOST = -1
        HWND_NOTOPMOST = -2
        
        # 設為最上層
        windll.user32.SetWindowPos(
            hwnd, HWND_TOPMOST, 0, 0, 0, 0,
            SWP_NOMOVE | SWP_NOSIZE
        )
        
        # 設定焦點
        windll.user32.SetForegroundWindow(hwnd)
        
    def accept(self, event):
        eflag = True
        try:
            vfe = float(self.input.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid number for Fermi Level.")
            eflag = False
            vfe = self.ovfe
        self.vfe = vfe
        self.eflag = eflag
        self.close()
        QApplication.processEvents()

class main(QMainWindow):
    def __init__(self, file, hwnd=None):
        self.lfs = loadfiles(file, name='internal')
        icon = IconManager().icon
        pixmap = QPixmap()
        pixmap.loadFromData(b64decode(icon))
        qicon = QIcon(pixmap)
        self.icon = qicon
        super().__init__()
        self.hwnd=hwnd
        self.setWindowTitle("Raw Data Viewer")
        # self.showFullScreen()
        # geo = self.geometry()
        # self.showNormal()
        # self.resize(geo.width(), geo.height())
        # self.resize(1200, 1000)
        # self.setFixedSize(1200, 1000)
        self.setStyleSheet("""
            QWidget {
                background-color: #000;
                color: #EEE;
                font-family: Arial;
                font-size: 24px;
            }
            QStatusBar {
                background-color: #D7D7D7;
                color: #222;
                font-size: 30px;
            }
            QMenuBar, QMenu, QSlider, QSpinBox, QLineEdit, QLabel, QRadioButton {
                background-color: #000;
                color: #EEE;
            }
            QPushButton {
                background-color: #333;
                color: #EEE;
                font-family: Arial;
                font-weight: bold;
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
        """)
        
        self.setWindowIcon(qicon)
        
        
        # 主視窗
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)  # 主水平佈局
        
        # 左側：資訊顯示框
        left_layout = QVBoxLayout()
        main_layout.addLayout(left_layout, stretch=2)
        
        self.file_name = QComboBox()
        self.file_name.addItems([name for name in self.lfs.name])
        self.file_name.setCurrentIndex(0)
        self.file_name.currentIndexChanged.connect(self.on_file_name_changed)
        
        self.file_name.setStyleSheet("""
            QComboBox {
                background-color: #333;
                color: #FFD700;
                font-family: Consolas;
                font-size: 20px;
                font-weight: bold;
                border: 2px solid #555;
                border-radius: 5px;
                padding: 8px;
                min-height: 40px;
            }
            QComboBox:hover {
                background-color: #444;
            }
            QComboBox::drop-down {
                border: none;
                width: 30px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 10px solid transparent;
                border-right: 10px solid transparent;
                border-top: 12px solid #FFD700;
            }
            QComboBox QAbstractItemView {
                background-color: #222;
                color: #FFD700;
                selection-background-color: #555;
                selection-color: #FFD700;
                font-size: 20px;
            }
        """)
        
        
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)  # 設為唯讀
        self.text_display.setStyleSheet("""
            QTextEdit {
                background-color: #1E1E1E;
                color: #D4D4D4;
                font-family: Consolas;
                font-size: 18px;
                font-weight: bold;
                border: 2px solid #555;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        self.text_display.setPlainText("Waiting for data...")
        self.text_display.wheelEvent = self.on_wheel_event
        
        left_layout.addWidget(self.file_name)
        left_layout.addWidget(self.text_display)
        
        
        # 中間：繪圖區
        mid_layout = QVBoxLayout()
        ploty_layout = QHBoxLayout()    #plot, ylabel
        plotx_layout = QVBoxLayout()    #plot, xlabel


        self.plot = pg.PlotWidget()
        self.lx, self.ly = None, None
        self.plot.setMouseEnabled(x=True, y=True)
        self.plot.setLabel('bottom', '')
        self.plot.getAxis('bottom').setStyle(tickFont=pg.QtGui.QFont("Arial", 18))
        self.plot.setLabel('left', '')
        self.plot.getAxis('left').setStyle(tickFont=pg.QtGui.QFont("Arial", 18))
        self.plot.scene().sigMouseMoved.connect(self.on_mouse_moved)
        self.plot.scene().sigMouseClicked.connect(self.mouse_clicked)
        self.flag_click = False
        self.plot.sigRangeChanged.connect(self.range_changed)
        self.mouse_point = None
        
        self.plotx = pg.PlotWidget()
        self.ploty = pg.PlotWidget()
        self.plotx.setMouseEnabled(x=False, y=False)
        self.ploty.setMouseEnabled(x=False, y=False)
        self.plotx.scene().sigMouseMoved.connect(self.xy_clear)
        self.ploty.scene().sigMouseMoved.connect(self.xy_clear)
        self.plotx.setMaximumHeight(100)
        self.ploty.setMaximumWidth(100)
        self.plotx.hideAxis('bottom')
        self.ploty.hideAxis('left')
        self.plotx.getAxis('left').setStyle(tickFont=pg.QtGui.QFont("Arial", 14))
        self.ploty.getAxis('bottom').setStyle(tickFont=pg.QtGui.QFont("Arial", 14))

        self.menu_bar = QMenuBar()
        self.setMenuBar(self.menu_bar)
        file_menu = self.menu_bar.addMenu("File")
        act_quit = QAction("Quit", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)
        view_menu = self.menu_bar.addMenu("View")
        act_grid = QAction("Show Grid", self)
        act_grid.setCheckable(True)
        act_grid.setChecked(False)
        act_grid.triggered.connect(self.toggle_grid)
        view_menu.addAction(act_grid)
        view_menu.addSeparator()
        self.set_default_colormap()
        for cmap_name in self.cmap_colors_dict.keys():
            act_cmap = QAction(f"Colormap: {cmap_name}", self)
            act_cmap.setCheckable(True)
            act_cmap.setChecked(cmap_name=='prevac_cmap')
            act_cmap.triggered.connect(lambda checked, name=cmap_name: self.set_cmap(name))
            view_menu.addAction(act_cmap)
        
        
        self.KE_pixmap = self.make_axis_label("Kinetic Energy (eV)", font_size=18, vertical=True)
        self.BE_pixmap = self.make_axis_label("Binding Energy (eV)", font_size=18, vertical=True)
        self.mm_pixmap = self.make_axis_label("Position (mm)", font_size=18, vertical=False)
        self.k_pixmap = self.make_axis_label("k (2π/Å)", font_size=18, vertical=False)
        self.deg_pixmap = self.make_axis_label("Angle (deg)", font_size=18, vertical=False)

        self.xlabel = QLabel()
        self.xlabel.setPixmap(self.deg_pixmap)
        self.xlabel.setAlignment(Qt.AlignCenter)

        self.ylabel = QLabel()
        self.set_vfe = None
        self.emode='KE'
        self.eflag = True
        self.vfe = 21.2
        self.ylabel.mousePressEvent = lambda event: self.energy_mode(event)
        self.ylabel.setCursor(Qt.PointingHandCursor)
        self.ylabel.setStyleSheet("""
            QLabel:hover {
                background-color: #555;
                border: 2px solid #FFD700;
            }
        """)
        self.ylabel.setPixmap(self.KE_pixmap)
        self.ylabel.setAlignment(Qt.AlignCenter)
            
        ploty_layout.addWidget(self.ylabel)
        ploty_layout.addWidget(self.plot)
        ploty_layout.addWidget(self.ploty)
        plotx_layout.addWidget(self.plotx)
        
        plotx_layout.addLayout(ploty_layout)
        plotx_layout.addWidget(self.xlabel)
        
        plot_hist_layout = QHBoxLayout()
        plot_hist_layout.setContentsMargins(0, 0, 0, 0)
        self.hist = pg.HistogramLUTWidget()
        # plot_hist_layout.addWidget(self.plot, stretch=4)
        self.set_cmap()
        
        plot_hist_layout.addLayout(plotx_layout, stretch=4)
        plot_hist_layout.addWidget(self.hist, stretch=1)

        # plot_menu_layout.addLayout(plot_hist_layout)
        mid_layout.addLayout(plot_hist_layout)

        main_layout.addLayout(mid_layout, stretch=4)
        
        # 狀態列
        self.statusbar = QStatusBar(self)
        self.setStatusBar(self.statusbar)
        # right_label = QLabel(f"{self.mode.capitalize()} Mode")
        right_label = QLabel("Raw Data")
        right_label.setStyleSheet("background-color: #D7D7D7; color: #000; font-weight: bold; font-size: 30px;")
        # right_label.setFocusPolicy(Qt.NoFocus)
        # right_label.setTextInteractionFlags(Qt.NoTextInteraction)
        self.statusbar.addPermanentWidget(right_label)  # 右側狀態列(有缺陷 #D7D7D7 游標殘留)
        
        self.ind = 0
        self.data = self.lfs.get(self.ind)
        self.resizeEvent = self.xy_clear()
        self.update_plot()
        
        self.showMaximized()
        self.w, self.h = self.width(), self.height()
        
    def energy_mode(self, event):
        if self.set_vfe is None:
            self.set_vfe = c_fermi_level(self.vfe, self.icon)
            QApplication.activeWindow().activateWindow()
            self.set_vfe.exec_()
            self.vfe, self.eflag = self.set_vfe.vfe, self.set_vfe.eflag
            if self.eflag:
                if self.emode == 'KE':
                    self.emode = 'BE'
                else:
                    self.emode = 'KE'
                self.update_plot()
            self.set_vfe = None
    
    def plot_geometry(self):
        axis = self.plot.getAxis('left')
        tick_font = axis.style.get('tickFont', pg.QtGui.QFont("Arial", 18))
        metrics = QFontMetrics(tick_font)
        
        # 取得當前範圍的 tick values
        axis_range = axis.range
        size = axis.height()
        
        try:
            tick_values = axis.tickValues(axis_range[0], axis_range[1], size)            
            max_width = 0
            
            # tickValues 返回的是一個列表,每個元素是 (spacing, [(value, text), ...])
            for level in tick_values:
                if isinstance(level, tuple) and len(level) >= 2:
                    spacing, ticks = level[0], level[1]
                    # ticks 是 [(value, text), ...] 的列表
                    for tick_info in ticks:
                        if isinstance(tick_info, float) and len(ticks) >= 2:
                            text = str(tick_info)
                            # 計算文字寬度
                            width = metrics.horizontalAdvance(text)
                            if '9999' not in text and '0000' not in text:   # 小數極端狀況排除
                                if width > max_width:
                                    max_width = width
            
            # 設定軸寬度
            tick_length = abs(axis.style.get('tickLength', 5))
            tick_offset = axis.style.get('tickTextOffset', 3)
            total_width = max_width + tick_length + tick_offset[0] + 10
            
            if max_width < 65:
                max_width = 65
                
            axis.setWidth(max_width)
            
            # print(f"Max tick label width: {max_width}, Total axis width: {total_width}")
            return max_width
            
        except Exception as e:
            print(f"Error calculating tick label width: {e}")
            import traceback
            traceback.print_exc()
            # 返回預設寬度
            default_width = metrics.horizontalAdvance("-000.00")
            axis.setWidth(default_width + 20)
            return default_width
        
    
    def xy_geometry(self, event=None):
        if event is not None:
            print(event)
        self.plot_geometry()
        if self.flag_click is False:
            self.plotx.clear()
            self.ploty.clear()
        self.plotx.showAxis('left')
        self.ploty.showAxis('bottom')
        vb = self.plot.getViewBox()
        rect = vb.sceneBoundingRect()
        # 使用 setGeometry 設定位置和大小
        plot_geometry = self.plot.geometry()
        plot_x = plot_geometry.x()
        plot_y = plot_geometry.y()
        # 計算左軸佔用的寬度
        left_axis = self.plot.getAxis('left')
        left_axis_width = int(left_axis.width())  # 包含 tick labels 的寬度
        
        # 計算實際繪圖區域相對於 widget 的偏移
        plot_area_width = int(rect.width())  # 繪圖區域的寬度
        
        # 計算 plotx 應該的位置和寬度
        # plotx 應該與主 plot 的繪圖區域對齊
        plotx_height = self.plotx.height()
        plotx_x = plot_x  # widget x + 左軸寬度
        plotx_y = plot_y - plotx_height
        plotx_width = plot_area_width + left_axis_width  # 包含左軸寬度
        
        # 設定 plotx 的幾何位置
        self.plotx.getAxis('left').setWidth(left_axis_width)
        self.plotx.setGeometry(plotx_x, plotx_y, plotx_width, plotx_height)
        
        
        bottom_axis = self.plot.getAxis('bottom')
        bottom_axis_height = int(bottom_axis.height())
        
        plot_area_height = int(rect.height())
        
        
        ploty_width = self.ploty.width()
        ploty_x = plot_x + plot_area_width + left_axis_width
        ploty_y = plot_y
        ploty_height = plot_area_height + bottom_axis_height
        
        self.ploty.getAxis('bottom').setHeight(bottom_axis_height-2)
        self.ploty.setGeometry(ploty_x, ploty_y, ploty_width, ploty_height)
    
    def range_changed(self, *args):
        if self.mouse_point is not None:
            if self.flag_click:
                self.xy_geometry()
                self.update_xy_cut(self.mark_point.x(), self.mark_point.y())
            else:
                self.xy_geometry()
                self.update_xy_cut(self.mouse_point.x(), self.mouse_point.y())
    
    def mouse_clicked(self, event):
        if self.flag_click:
            self.flag_click = False
        else:
            self.flag_click = True
            mark_point = event.scenePos()
            self.mark_point = self.plot.getViewBox().mapSceneToView(mark_point)
    
    def on_mouse_moved(self, pos):
        vb = self.plot.getViewBox()
        rect = vb.sceneBoundingRect()
        self.xy_geometry()
        if rect.contains(pos):
            mouse_point = vb.mapSceneToView(pos)
            self.mouse_point = mouse_point
            self.statusbar.setStyleSheet("font-size: 30px;")
            self.statusbar.showMessage(f"x={mouse_point.x():.2f}  y={mouse_point.y():.2f} data={self.data_interp(mouse_point.x(), mouse_point.y()):.4f}")
            if not self.flag_click:
                self.update_xy_cut(mouse_point.x(), mouse_point.y())
        else:
            self.xy_clear()
    
    def update_xy_cut(self, x, y):
        xdata = self.data.phi.values
        ydata = self.data.eV.values
        xidx = (np.abs(xdata - x)).argmin()
        yidx = (np.abs(ydata - y)).argmin()
        if self.emode == 'BE':
            yidx = (np.abs((ydata - self.vfe) - y)).argmin()
            ydata = ydata - self.vfe
        
        if abs(xdata[xidx] - x) > (xdata[1]-xdata[0])/2:
            self.ploty.clear()
        else:
            self.ploty.plot(self.data.data[:, xidx], ydata, pen=pg.mkPen('w', width=2))
        
        if abs(ydata[yidx] - y) > (ydata[1]-ydata[0])/2:
            self.plotx.clear()
        else:
            self.plotx.plot(xdata, self.data.data[yidx, :], pen=pg.mkPen('w', width=2))
            
        left, right = self.plot.viewRect().left(), self.plot.viewRect().right()
        top, bottom = self.plot.viewRect().top(), self.plot.viewRect().bottom()
        self.plotx.setRange(xRange=(left, right), padding=0)
        self.ploty.setRange(yRange=(top, bottom), padding=0)
        if self.lx is not None:
            self.plot.removeItem(self.lx)
            self.plot.removeItem(self.ly)
            self.lx, self.ly = None, None
        self.lx = self.plot.plot([x, x], [top, bottom], pen=pg.mkPen('y', width=1, style=Qt.DashLine))
        self.ly = self.plot.plot([left, right], [y, y], pen=pg.mkPen('y', width=1, style=Qt.DashLine))
    
    def data_interp(self, x, y):
        if self.emode == 'BE':
            y+=self.vfe
        data = self.data.sel(eV=y, phi=x, method='nearest').values
        return data
    
    def on_file_name_changed(self, value):
        self.flag_click = False
        self.mouse_point = None
        self.xy_clear()
        self.ind = value
        self.data = self.lfs.get(self.ind)
        self.update_plot()
    
    def xy_clear(self):
        if self.flag_click is False:
            self.statusbar.showMessage("")
            self.plotx.clear()
            self.ploty.clear()
            self.plotx.hideAxis('left')
            self.ploty.hideAxis('bottom')
            if self.lx is not None:
                self.plot.removeItem(self.lx)
                self.plot.removeItem(self.ly)
                self.lx, self.ly = None, None
        else:
            self.xy_geometry()
            x = self.mark_point.x()
            y = self.mark_point.y()
            self.statusbar.setStyleSheet("font-size: 30px;")
            self.statusbar.showMessage(f"x={x:.2f}  y={y:.2f} data={self.data_interp(x, y):.4f}")
    
    def update_info_display(self):
        self.attrs = self.data.attrs
        self.lensmode = self.attrs['LensMode']
        st=''
        for _ in self.attrs.keys():
            if _ == 'Description':
                st+=str(_)+': '+str(self.attrs[_]).replace('\n','\n             ')
            elif _ == 'Path':
                pass
            else:
                st+=str(_)+': '+str(self.attrs[_])+'\n'
        """更新資訊顯示框"""
        info_text = f"""{'='*40}
{' '*13}FILE INFORMATION
{'='*40}
{st}
{'='*40}
{' '*13}DATA INFORMATION
{'='*40}
Spectrum Shape: {self.data.data.shape}
Spectrum Dtype: {self.data.data.dtype}
Min Value: {np.min(self.data.data):.4f}
Max Value: {np.max(self.data.data):.4f}
Y-Axis: {self.data.eV.values.min()} to {self.data.eV.values.max()}, {len(self.data.eV)} points
X-Axis: {self.data.phi.values.min()} to {self.data.phi.values.max()}, {len(self.data.phi)} points
"""
        self.text_display.setPlainText(info_text)
    
    def update_plot(self):
        # tring to plot initial data
        self.update_info_display()
        self.plot.clear()
        arr = self.data.data
        img_item = pg.ImageItem(arr.T)
        img_item.setLevels(np.min(arr), np.max(arr))
        self.hist.setImageItem(img_item)
        self.plot.setAspectLocked(False)  # 鎖定比例
        dx = self.data.phi[-1] - self.data.phi[0]
        x=[self.data.phi[0], self.data.phi[-1]]
        xlow, xhigh = self.data.phi[0], self.data.phi[-1]
        if self.lensmode == 'Transmission':
            self.xlabel.setPixmap(self.mm_pixmap)
        elif self.lfs.f_npz[self.ind]:
            self.xlabel.setPixmap(self.k_pixmap)
        else:
            self.xlabel.setPixmap(self.deg_pixmap)
        
        if self.emode == 'BE':
            self.ylabel.setPixmap(self.BE_pixmap)
            y=[self.data.eV[0]-self.vfe, self.data.eV[-1]-self.vfe]
        else:
            self.ylabel.setPixmap(self.KE_pixmap)
            y=[self.data.eV[0], self.data.eV[-1]]
        dy = self.data.eV[-1] - self.data.eV[0]
        self.plot.setLimits(xMin=x[0]-dx/8, xMax=x[-1]+dx/8, yMin=y[0]-dy/8, yMax=y[-1]+dy/8)
        
        self.plot.setRange(xRange=(xlow, xhigh), yRange=(y[0], y[-1]), padding=0)
        # self.rescale(y, x)
        rect = pg.QtCore.QRectF(x[0], y[0], x[-1] - x[0], y[-1] - y[0])  # 真實位置
        img_item.setRect(rect)
        self.plot.addItem(img_item)
    
    @property
    def ind(self):
        return self._ind
    
    @ind.setter
    def ind(self, value):
        l=len(self.lfs.name)
        i = value
        if i > l - 1:
            i = 0
        elif i < 0:
            i = l - 1
        self._ind = i
    
    def on_wheel_event(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self.ind -= 1
            self.data = self.lfs.get(self.ind)
        else:
            self.ind += 1
            self.data = self.lfs.get(self.ind)
        self.file_name.setCurrentIndex(self.ind)
        self.statusbar.setStyleSheet("font-size: 20px;")
        self.statusbar.showMessage(f"Path: {self.lfs.path[self.ind]}")
    
    def toggle_grid(self, checked):
        if checked:
            self.plot.showGrid(x=True, y=True)
        else:
            self.plot.showGrid(x=False, y=False)
            
    def make_axis_label(self, text, font_size=18, vertical=False):
        font = QFont("Arial", font_size, QFont.Bold)
        metrics = pg.QtGui.QFontMetrics(font)
        if vertical:
            w = metrics.height() + 10
            h = metrics.horizontalAdvance(text) + 10
            pixmap = QPixmap(w, h)
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            painter.setFont(font)
            painter.setPen(QColor("white"))
            painter.translate(w//2, h//2)
            painter.rotate(-90)
            painter.drawText(-h//2 + 5, w//2 - 12, text)
            painter.end()
            # label = QLabel()
            # label.setPixmap(pixmap)
            # label.setMinimumWidth(w)
            # label.setAlignment(Qt.AlignCenter)
            return pixmap
        else:
            w = metrics.horizontalAdvance(text) + 10
            h = metrics.height() + 10
            pixmap = QPixmap(w, h)
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            painter.setFont(font)
            painter.setPen(QColor("white"))
            painter.drawText(5, h - 12, text)
            painter.end()
            # label = QLabel()
            # label.setPixmap(pixmap)
            # label.setMinimumHeight(h)
            # label.setAlignment(Qt.AlignCenter)
            return pixmap

    def set_default_colormap(self):
        # Define your custom colors (as RGB tuples)
        # (value,(color))
        custom_colors1 = [(0, (1, 1, 1)),
                        (0.5, (0, 0, 1)),
                        (0.85, (0, 1, 1)),
                        (1, (1, 1, 0.26))]

        # Create a custom colormap
        custom_cmap1 = LinearSegmentedColormap.from_list(
            'custom_cmap1', custom_colors1, N=256)
        mpl.colormaps.register(custom_cmap1)

        # Define your custom colors (as RGB tuples)
        # (value,(color))
        custom_colors2 = [(0, (0, 0.08, 0.16)),
                        (0.2, (0.2, 0.7, 1)),
                        (0.4, (0.28, 0.2, 0.4)),
                        (0.62, (0.9, 0.1, 0.1)),
                        (0.72, (0.7, 0.34, 0.1)),
                        (0.8, (1, 0.5, 0.1)),
                        (1, (1, 1, 0))]

        # Create a custom colormap
        custom_cmap2 = LinearSegmentedColormap.from_list(
            'custom_cmap2', custom_colors2, N=256)
        mpl.colormaps.register(custom_cmap2)

        # Define your custom colors (as RGB tuples)
        # (value,(color))
        custom_colors3 = [(0, (0.88, 0.84, 0.96)),
                        (0.5, (0.32, 0, 0.64)),
                        (0.75, (0, 0, 1)),
                        (0.85, (0, 0.65, 1)),
                        (0.9, (0.2, 1, 0.2)),
                        (0.96, (0.72, 1, 0)),
                        (1, (1, 1, 0))]

        # Create a custom colormap
        custom_cmap3 = LinearSegmentedColormap.from_list(
            'custom_cmap3', custom_colors3, N=256)
        mpl.colormaps.register(custom_cmap3)

        # Define your custom colors (as RGB tuples)
        # (value,(color))
        custom_colors4 = [(0, (1, 1, 1)),
                        (0.4, (0.3, 0, 0.3)),
                        (0.5, (0.3, 0, 0.6)),
                        (0.6, (0, 1, 1)),
                        (0.7, (0, 1, 0)),
                        (0.8, (1, 1, 0)),
                        (1, (1, 0, 0))]

        # Create a custom colormap
        custom_cmap4 = LinearSegmentedColormap.from_list(
            'custom_cmap4', custom_colors4, N=256)
        mpl.colormaps.register(custom_cmap4)
        
        # Define your custom colors (as RGB tuples)
        # (value,(color))
        prevac_colors = [(0, (0.2*0.82, 0.2*0.82, 0.2*0.82)),
                        (0.2, (0.4*0.82, 0.6*0.82, 0.9*0.82)),
                        (0.4, (0, 0.4*0.82, 0)),
                        (0.6, (0.5*0.82, 1*0.82, 0)),
                        (0.8,(1*0.82, 1*0.82, 0)),
                        (1, (1*0.82, 0, 0))]
        # Create a custom colormap
        prevac_cmap = LinearSegmentedColormap.from_list(
            'prevac_cmap', prevac_colors, N=256)
        mpl.colormaps.register(prevac_cmap)
        
        self.cmap_colors_dict={
            'prevac_cmap': len(prevac_cmap._segmentdata['red']),
            'custom_cmap1': len(custom_cmap1._segmentdata['red']),
            'custom_cmap2': len(custom_cmap2._segmentdata['red']),
            'custom_cmap3': len(custom_cmap3._segmentdata['red']),
            'custom_cmap4': len(custom_cmap4._segmentdata['red'])
        }
            
    def set_cmap(self, cmap_name='prevac_cmap'):
        for action in self.menu_bar.actions():
            if action.menu():
                for act in action.menu().actions():
                    if act.text().startswith("Colormap:"):
                        if act.text() == f"Colormap: {cmap_name}":
                            act.setChecked(True)
                        else:
                            act.setChecked(False)
        cmap = plt.get_cmap(cmap_name)
        n = self.cmap_colors_dict[cmap_name]
            
        # 轉換為 pyqtgraph 格式
        # 取 256 個顏色點
        colors = cmap(np.linspace(0, 1, n))
        
        # 轉換為 pyqtgraph 的格式 (0-255 的整數)
        colors_rgb = (colors[:, :3] * 255).astype(np.uint8)
        
        # 創建 ColorMap
        pos = np.linspace(0, 1, n)
        color_map = pg.ColorMap(pos, colors_rgb)
        
        # 應用到 histogram widget
        self.hist.gradient.setColorMap(color_map)
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    hwnd = get_hwnd()
    p = argparse.ArgumentParser(description="Input Raw Data File Path")
    p.add_argument("-f", "--file", help="file path", type=str, nargs='+', required=True)
    args = p.parse_args()
    if args.file:
        file = args.file
    win = main(file, hwnd)
    win.show()
    sys.exit(app.exec())