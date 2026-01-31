import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider, QLabel, QStatusBar,
    QSpinBox, QPushButton, QHBoxLayout, QLineEdit, QMenuBar, QAction, QRadioButton,
    QButtonGroup, QFileDialog, QProgressBar, QDialog
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QPainter, QFont, QColor, QIcon, QCursor
import pyqtgraph as pg
from base64 import b64decode
import cv2, os, inspect
import h5py, time, zarr
if os.name == 'nt':
    from ctypes import windll
import shutil, psutil, argparse
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

cdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
if __name__ == "__main__":
    os.chdir(cdir)
    if os.path.exists('.MDC_cut_DataViewer'):
        shutil.rmtree('.MDC_cut_DataViewer')
    os.mkdir('.MDC_cut_DataViewer')
    os.system(f'attrib +h +s ".MDC_cut_DataViewer"')

sys.path.append(os.path.dirname(cdir))
from tool.util import MenuIconManager
from tool.qt_util import MainWindow, ProgressDialog, cmap_register

def rotate(data: cv2.typing.MatLike, angle: float, size: tuple[int, int]) -> cv2.typing.MatLike:
    """
    for square data
    """
    mat = cv2.getRotationMatrix2D((size[1]/2, size[0]/2), angle, 1)
    data = cv2.warpAffine(data, mat, (size[1], size[0]), flags=cv2.INTER_NEAREST)
    return data

def find_window():
    if sys.platform != "win32":
        return 0
    hwnd = windll.user32.FindWindowW(None, "命令提示字元")
    if not hwnd:
        hwnd = windll.user32.FindWindowW(None, "Command Prompt")
    if not hwnd:
        hwnd = windll.user32.FindWindowW(None, "cmd")
    return hwnd

def det_chunk(density: int, density2: int = 0, dtype: np.dtype=np.float32):
    if density2 == 0:
        density2 = density
    current_mem = psutil.virtual_memory().available/1024**3
    use_mem = current_mem*0.8  # 80%
    print(f"Memory available: {current_mem:.2f} GB, 80% Upper Limit: {use_mem:.2f} GB")
    mem = np.empty((density, density2), dtype=dtype).nbytes/1024**3
    chunk_size = int(use_mem / mem)
    mem = None
    return chunk_size

def disp_zarr_save(input_path, output_path, shape, max_val):
    zarr.save_group(output_path, ang=np.array([0, 1], dtype=np.float32))
    end = shape[0]
    size = det_chunk(shape[1], dtype=np.uint8)
    path = os.path.join(output_path, 'data')
    if size/end <1.2:   # threshold: more than 1.2 times memory available
        # partial load data into memory (light weight RAM usage)
        step = int(min(size, end//1.5))   #fix step
        savez = zarr.open(path, mode='w', shape=shape, dtype=np.uint8)
        for i in range(0, end, step):
            ind = slice(i, min(i + step, end))
            savez[ind,...] = np.asarray(zarr.open(input_path, mode='r')[ind, :, :-1]/max_val*255, dtype=np.uint8)
            print('Progress: %.2f%%'%(min(i + step, end)/end*100))
    else:
        # load all data into memory (heavy RAM usage)
        zdata = np.asarray(zarr.open(input_path, mode='r')[..., :-1]/max_val*255, dtype=np.uint8)
        zarr.open(path, mode='w', shape=shape, dtype=np.uint8)[:] = zdata
    ang = zarr.open(os.path.join(output_path, 'ang'), mode='r+')
    ang[1] = 0

def load_zarr(path: str):
    try:
        data = zarr.open(path, mode='r+', dtype=np.float32)
        xmin,xmax = data[0, 1, -1], data[1, 1, -1]
        ymin,ymax = data[2, 1, -1], data[3, 1, -1]
        E = data[:, 0, -1]
        shape = data.shape[:-1] + (data.shape[2]-1,)
        ang_path = os.path.join(path, '__disp__.zarr', 'ang')
        if os.path.exists(ang_path):
            ang = zarr.open(ang_path, mode='r')
            if ang[0] != ang[-1]:
                os.chdir(path)
                shutil.rmtree('__disp__.zarr')
        # zpath = os.path.join(path, '__disp__.zarr')
        # if not os.path.exists(zpath):
        #     # data = zarr.open(path, mode='r+', dtype=np.float32)[:,:,:data.shape[2]-1]  # Remove the last attribute dimension
        # else:
        #     shape = list(data.shape)
        #     shape[2] -= 1
        #     data = np.zeros(tuple(shape))
        mode = 'standard'
        return mode, shape, xmin, xmax, ymin, ymax, E
    except:
        try:
            data = zarr.open_group(path, mode='r+')
            xmin,xmax = data['attr_array'][0, 1], data['attr_array'][1, 1]
            ymin,ymax = data['attr_array'][2, 1], data['attr_array'][3, 1]
            E = data['attr_array'][:, 0]
            data = data['data']
            shape = data.shape
            mode = 'display'
            return mode, shape, xmin, xmax, ymin, ymax, E
        except Exception as e:
            print(f"Error loading data from {path}: {e}")
            quit()

class SliceBrowser(MainWindow):
    def __init__(self, path=None, hwnd=None):
        super().__init__()
        icon_manager = MenuIconManager(qt=True)
        icon = icon_manager.get_icon('view_3d')
        pixmap = QPixmap()
        pixmap.loadFromData(b64decode(icon))
        qicon = QIcon(pixmap)
        self.icon = qicon
        pbar = ProgressDialog(10, self.icon)
        pbar.resize(self.width()//3, self.height()//4)
        pbar.show()
        pbar.increaseProgress('Loading Zarr Data Cube')
        print(f"Loading Zarr Data Cube from {path}")
        print('Please wait...')
        self.hwnd=hwnd
        if hwnd:
            windll.user32.ShowWindow(hwnd, 9)
            windll.user32.SetForegroundWindow(hwnd)
        t=time.perf_counter()
        self.mode, self.shape, xmin, xmax, ymin, ymax, E = load_zarr(path)
                
        if path is None:
            self.path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        else:
            self.path = path
            
        self.max_value_flag = False
        
        pbar.increaseProgress('Loading Zarr Data Cube')
        e_size, ky_size, kx_size = self.shape
        kx = np.linspace(xmin, xmax, kx_size)
        ky = np.linspace(ymin, ymax, ky_size)
        print(self.shape)
        
        pbar.increaseProgress('Loading Zarr Data Cube')
        print(f"Elapse time: {time.perf_counter()-t:.2f} s")
        pbar.increaseProgress('Setting QtWidgets')

        self.raw_E = E
        self.raw_kx = kx
        self.raw_ky = ky

        self.path_angle = 0
        
        # bin 設定
        self.bin_e = 1
        self.bin_kx = 1
        self.bin_ky = 1

        self.setWindowTitle("Volume Viewer")
        # self.showFullScreen()
        # geo = self.geometry()
        # self.showNormal()
        # self.resize(geo.width(), geo.height())
        # self.resize(1200, 1000)
        # self.setFixedSize(1200, 1000)
        
        self.setWindowIcon(qicon)
        
        
        # 主視窗
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)  # 主水平佈局
        
        # 左側：繪圖區
        left_layout = QVBoxLayout()
        ploty_layout = QHBoxLayout()    #plot, ylabel
        plotx_layout = QVBoxLayout()    #plot, xlabel


        pbar.increaseProgress('Setting QtWidgets')
        self.plot = pg.PlotWidget()
        self.plot.setMouseEnabled(x=True, y=True)
        self.plot.setLabel('bottom', '')
        self.plot.getAxis('bottom').setStyle(tickFont=pg.QtGui.QFont("Arial", 18))
        self.plot.setLabel('left', '')
        self.plot.getAxis('left').setStyle(tickFont=pg.QtGui.QFont("Arial", 18))
        self.plot.scene().sigMouseMoved.connect(self.on_mouse_moved)

        pbar.increaseProgress('Setting QtWidgets')
        self.menu_bar = QMenuBar()
        self.menu_bar.setStyleSheet("""
            QMenuBar {
                padding: 0px;
            }
            """)
        self.setMenuBar(self.menu_bar)
        file_menu = self.menu_bar.addMenu("File")
        act_quit = QAction("Quit", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        if self.mode == 'standard':
            act_save_zarr = QAction("Save as Zarr-Standard", self)
            act_save_zarr.triggered.connect(self.save_as_zarr)
        act_save_zarr_disp = QAction("Save as Zarr-Display", self)
        act_save_zarr_disp.triggered.connect(self.save_as_zarr_disp)
        if self.mode == 'standard':
            file_menu.addAction(act_save_zarr)
        file_menu.addAction(act_save_zarr_disp)
        file_menu.addAction(act_quit)
        view_menu = self.menu_bar.addMenu("View")
        act_grid = QAction("Show Grid", self)
        act_grid.setCheckable(True)
        act_grid.setChecked(False)
        act_grid.triggered.connect(self.toggle_grid)
        view_menu.addAction(act_grid)
        self.cmap_menu = view_menu.addMenu("Colormap")
        self.set_default_colormap()
        for cmap_name in self.cmap_colors_dict.keys():
            act_cmap = QAction(f"{cmap_name}", self)
            act_cmap.setCheckable(True)
            act_cmap.setChecked(cmap_name=='prevac_cmap')
            act_cmap.triggered.connect(lambda checked, name=cmap_name: self.set_cmap(name))
            self.cmap_menu.addAction(act_cmap)
        
        
        self.E_pixmap_x = self.make_axis_label("Kinetic Energy (eV)", font_size=18, vertical=False)
        self.kx_pixmap_x = self.make_axis_label("kx (2π/Å)", font_size=18, vertical=False)
        self.ky_pixmap_x = self.make_axis_label("ky (2π/Å)", font_size=18, vertical=False)
        self.E_pixmap_y = self.make_axis_label("Kinetic Energy (eV)", font_size=18, vertical=True)
        self.kx_pixmap_y = self.make_axis_label("kx (2π/Å)", font_size=18, vertical=True)
        self.ky_pixmap_y = self.make_axis_label("ky (2π/Å)", font_size=18, vertical=True)

        self.xlabel = QLabel()
        self.xlabel.setPixmap(self.kx_pixmap_x)
        self.xlabel.setAlignment(Qt.AlignCenter)
        self.xlabel.setContentsMargins(10, 0, 0, 0)

        self.ylabel = QLabel()
        self.ylabel.setPixmap(self.ky_pixmap_y)
        self.ylabel.setAlignment(Qt.AlignCenter)
            
        ploty_layout.addWidget(self.ylabel)
        ploty_layout.addWidget(self.plot)
        plotx_layout.addLayout(ploty_layout)
        plotx_layout.addWidget(self.xlabel)
        
        rdb_layout = QHBoxLayout()

        pbar.increaseProgress('Setting QtWidgets')
        radio1 = QRadioButton("E")
        radio1.setToolTip("Show E slice (ky-kx)")
        radio1.setChecked(True)
        radio2 = QRadioButton("kx")
        radio2.setToolTip("Show kx slice (E-ky)")
        radio3 = QRadioButton("ky")
        radio3.setToolTip("Show ky slice (E-kx)")

        group = QButtonGroup(central)
        group.addButton(radio1)
        group.addButton(radio2)
        group.addButton(radio3)

        radio1.clicked.connect(lambda: self.on_radio_button_changed("E"))
        radio2.clicked.connect(lambda: self.on_radio_button_changed("kx"))
        radio3.clicked.connect(lambda: self.on_radio_button_changed("ky"))

        rdb_layout.addWidget(radio1)
        rdb_layout.addWidget(radio2)
        rdb_layout.addWidget(radio3)
        left_layout.addWidget(self.menu_bar)
        left_layout.addLayout(rdb_layout)

        # 三個滑桿
        pbar.increaseProgress('Setting QtWidgets')
        self.slider_E = QSlider(Qt.Horizontal)
        self.slider_E.setFixedHeight(50)
        self.slider_E.setMinimum(0)
        self.slider_E.setMaximum(len(E)-1)
        self.slider_E.setValue(0)
        self.slider_E.setTickInterval(1)
        self.slider_E.setSingleStep(1)
        self.label_E = QLabel("E Slice (ky-kx)")
        self.label_E.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.label_E)
        left_layout.addWidget(self.slider_E)

        self.slider_kx = QSlider(Qt.Horizontal)
        self.slider_kx.setFixedHeight(50)
        self.slider_kx.setMinimum(0)
        self.slider_kx.setMaximum(len(kx)-1)
        self.slider_kx.setValue(0)
        self.slider_kx.setTickInterval(1)
        self.slider_kx.setSingleStep(1)
        self.label_kx = QLabel("kx Slice (E-ky)")
        self.label_kx.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.label_kx)
        left_layout.addWidget(self.slider_kx)
        self.label_kx.setVisible(False)
        self.slider_kx.setVisible(False)

        self.slider_ky = QSlider(Qt.Horizontal)
        self.slider_ky.setFixedHeight(50)
        self.slider_ky.setMinimum(0)
        self.slider_ky.setMaximum(len(ky)-1)
        self.slider_ky.setValue(0)
        self.slider_ky.setTickInterval(1)
        self.slider_ky.setSingleStep(1)
        self.label_ky = QLabel("ky Slice (E-kx)")
        self.label_ky.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.label_ky)
        left_layout.addWidget(self.slider_ky)
        self.label_ky.setVisible(False)
        self.slider_ky.setVisible(False)

        # 建立 plot + hist 水平佈局
        plot_hist_layout = QHBoxLayout()
        plot_hist_layout.setContentsMargins(0, 0, 0, 0)
        self.hist = pg.HistogramLUTWidget()
        # plot_hist_layout.addWidget(self.plot, stretch=4)
        plot_hist_layout.addLayout(plotx_layout, stretch=4)
        plot_hist_layout.addWidget(self.hist, stretch=1)

        # plot_menu_layout.addLayout(plot_hist_layout)
        left_layout.addLayout(plot_hist_layout)

        main_layout.addLayout(left_layout, stretch=4)

        # 右側：控制區
        right_layout = QVBoxLayout()
        
        # bin 控制（三軸）
        bin_layout = QVBoxLayout()
        bin_layout.addWidget(QLabel("E bin"))
        self.bin_e_spin = QSpinBox()
        self.bin_e_spin.setMinimum(1)
        self.bin_e_spin.setMaximum(50)
        self.bin_e_spin.setValue(1)
        bin_layout.addWidget(self.bin_e_spin)

        bin_layout.addWidget(QLabel("kx bin"))
        self.bin_kx_spin = QSpinBox()
        self.bin_kx_spin.setMinimum(1)
        self.bin_kx_spin.setMaximum(50)
        self.bin_kx_spin.setValue(1)
        bin_layout.addWidget(self.bin_kx_spin)

        bin_layout.addWidget(QLabel("ky bin"))
        self.bin_ky_spin = QSpinBox()
        self.bin_ky_spin.setMinimum(1)
        self.bin_ky_spin.setMaximum(50)
        self.bin_ky_spin.setValue(1)
        bin_layout.addWidget(self.bin_ky_spin)
        right_layout.addLayout(bin_layout)
        
        # 初始 binning
        pbar.increaseProgress('Initializing')

        self.apply_bin_btn = QPushButton("Apply Bin")
        right_layout.addWidget(self.apply_bin_btn)
        self.apply_bin_btn.clicked.connect(self.on_bin_change)
        self.bin_e_spin.returnPressed = self.on_bin_change
        self.bin_kx_spin.returnPressed = self.on_bin_change
        self.bin_ky_spin.returnPressed = self.on_bin_change
        for spin in [self.bin_e_spin, self.bin_kx_spin, self.bin_ky_spin]:
            spin.lineEdit().returnPressed.connect(self.on_bin_change)
        

        # xlow/xhigh 控制
        self.xlow_edit = QLineEdit(f'{min(xmin, ymin):.3f}')
        self.xhigh_edit = QLineEdit(f'{max(xmax, ymax):.3f}')
        for i in [self.xlow_edit, self.xhigh_edit]:
            i.setStyleSheet("qproperty-alignment: 'AlignLeft';")
        self.xlow_label = QLabel("xlow")
        self.xhigh_label = QLabel("xhigh")
        right_layout.addWidget(self.xlow_label)
        right_layout.addWidget(self.xlow_edit)
        right_layout.addWidget(self.xhigh_label)
        right_layout.addWidget(self.xhigh_edit)

        self.xlow_edit.returnPressed.connect(self.xlim)
        self.xhigh_edit.returnPressed.connect(self.xlim)

        # 旋轉控制
        self.rotate_label = QLabel("Angle (deg)")
        self.rotate_slider = QSlider(Qt.Horizontal)
        self.rotate_slider.setFixedHeight(50)
        self.rotate_slider.setStyleSheet("""
                                    QSlider::handle:horizontal {
                                        background: #007AD9;
                                        width: 40px;      /* 控制滑塊寬度 */
                                        height: 40px;     /* 控制滑塊高度（對水平slider沒影響，但可加大垂直slider）*/
                                        margin: -10px 0;  /* 讓滑塊更突出 */
                                    }
                                    QSlider::groove:horizontal {
                                        height: 10px;     /* 控制滑道粗細 */
                                        background: #bcbcbc;
                                        border-radius: 5px;
                                    }
                                    """)
        self.rotate_slider.setMinimum(0)
        self.rotate_slider.setMaximum(3600)  # 0.1度精度
        self.rotate_slider.setValue(0)
        self.rotate_edit = QLineEdit("0.0")
        self.rotate_edit.setStyleSheet("qproperty-alignment: 'AlignLeft';")
        self.rotate_btn = QPushButton("Rotate Data Cube")
        right_layout.addWidget(self.rotate_label)
        right_layout.addWidget(self.rotate_slider)
        right_layout.addWidget(self.rotate_edit)
        right_layout.addWidget(self.rotate_btn)
        
        self.update_binned_data(init=True)

        # 存檔按鈕
        self.export_btn = QPushButton("Export to HDF5 File")
        self.export_btn.setStyleSheet("""
            QToolTip {
                background-color: #222;
                color: #EEE;
                border: 5px solid white;
                font-size: 20pt;
            }
        """)
        # self.export_btn.setToolTip("Export the current slice to an HDF5 file")
        self.set_exp_h5_btn()
        right_layout.addWidget(self.export_btn)

        right_layout.addStretch()
        main_layout.addLayout(right_layout, stretch=1)

        # 狀態列
        self.statusbar = QStatusBar(self)
        self.setStatusBar(self.statusbar)
        right_label = QLabel(f"{self.mode.capitalize()} Mode")
        right_label.setStyleSheet("background-color: #D7D7D7; color: #000; font-weight: bold; font-size: 30px;")
        # right_label.setFocusPolicy(Qt.NoFocus)
        # right_label.setTextInteractionFlags(Qt.NoTextInteraction)
        self.statusbar.addPermanentWidget(right_label)  # 右側狀態列(有缺陷 #D7D7D7 游標殘留)

        # 綁定事件
        self.slider_E.valueChanged.connect(self.update_E_slice)
        self.slider_kx.valueChanged.connect(self.update_kx_slice)
        self.slider_ky.valueChanged.connect(self.update_ky_slice)
        self.rotate_slider.valueChanged.connect(self.sync_rotate_edit)
        self.rotate_btn.clicked.connect(self.apply_rotation)
        self.rotate_edit.editingFinished.connect(self.sync_rotate_slider)
        self.export_btn.clicked.connect(self.export_slice)

        # 預設顯示 E 切片
        self.current_mode = 'E'
        self.update_E_slice(0, init=True)
        pbar.increaseProgress('Ready')
        pbar.close()
        
        self.showMaximized()
        self.w, self.h = self.width(), self.height()

    @property
    def dtype(self):
        if self.mode == 'standard':
            return np.float32
        elif self.mode == 'display':
            return np.uint8
    
    def data(self, ind: slice):
        if self.mode == 'standard':
            data = zarr.open(self.path, mode='r')[ind, :, :-1]
        elif self.mode == 'display':
            data = zarr.open(os.path.join(self.path, 'data'), mode='r')[ind]
        return data

    @property
    def max_value(self):
        if self.max_value_flag:
            return self.max_value_val
        shape = self.shape
        end = shape[0]
        size = det_chunk(shape[1], dtype=self.dtype)
        if size/end <1.2:   # threshold: more than 1.2 times memory available
            step = int(min(size, end//1.5))   #fix step
            max_value = 1.0
            for i in range(0, end, step):
                ind = slice(i, min(i + step, end))
                max_value = max(max_value, np.max(self.data(ind)))
            self.max_value_flag = True
            self.max_value_val = max_value
        else:
            max_value = np.max(self.data(slice(None)))
        return max_value
    
    def get_raw_data(self, save=False):
        if self.mode == 'standard':
            size = det_chunk(self.shape[1], dtype=np.float32)
            end = self.shape[0]
            if not save:
                if size/end <1.2:
                    self.raw_data = None
                else:
                    self.raw_data = zarr.open(self.path, mode='r')[:, :, :-1] #origin
            else:
                self.raw_data = zarr.open(self.path, mode='r')
        elif self.mode == 'display':
            self.raw_data = zarr.open(os.path.join(self.path, 'data'), mode='r+')

    def toggle_grid(self, checked):
        if checked:
            self.plot.showGrid(x=True, y=True)
        else:
            self.plot.showGrid(x=False, y=False)

    def xlim(self):
        xlow = float(self.xlow_edit.text())
        xhigh = float(self.xhigh_edit.text())
        if xlow >= xhigh:
            self.xlow_edit.setText(str(xhigh))
            self.xhigh_edit.setText(str(xlow))
        self.refresh_slice()

    def on_radio_button_changed(self, mode):
        self.current_mode = mode
        if mode == 'E':
            self.label_E.setVisible(True)
            self.slider_E.setVisible(True)
            self.label_kx.setVisible(False)
            self.slider_kx.setVisible(False)
            self.label_ky.setVisible(False)
            self.slider_ky.setVisible(False)
            self.update_E_slice(self.slider_E.value(),init=True)
        elif mode == 'kx':
            self.label_E.setVisible(False)
            self.slider_E.setVisible(False)
            self.label_kx.setVisible(True)
            self.slider_kx.setVisible(True)
            self.label_ky.setVisible(False)
            self.slider_ky.setVisible(False)
            self.update_kx_slice(self.slider_kx.value(), setlim=False)
        elif mode == 'ky':
            self.label_E.setVisible(False)
            self.slider_E.setVisible(False)
            self.label_kx.setVisible(False)
            self.slider_kx.setVisible(False)
            self.label_ky.setVisible(True)
            self.slider_ky.setVisible(True)
            self.update_ky_slice(self.slider_ky.value(), setlim=False)

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
            painter.drawText(-h//2 + 5, w//2 - 5, text)
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
            painter.drawText(5, h - 5, text)
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
        cmap_register(custom_cmap1)

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
        cmap_register(custom_cmap2)

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
        cmap_register(custom_cmap3)

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
        cmap_register(custom_cmap4)
        
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
        cmap_register(prevac_cmap)
        
        self.cmap_colors_dict={
            'prevac_cmap': len(prevac_cmap._segmentdata['red']),
            'custom_cmap1': len(custom_cmap1._segmentdata['red']),
            'custom_cmap2': len(custom_cmap2._segmentdata['red']),
            'custom_cmap3': len(custom_cmap3._segmentdata['red']),
            'custom_cmap4': len(custom_cmap4._segmentdata['red'])
        }
    
    def set_cmap(self, cmap_name='prevac_cmap'):
        for act in self.cmap_menu.actions():
            if act.text() == f"{cmap_name}":
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
        
    def on_mouse_moved(self, pos):
        vb = self.plot.getViewBox()
        if vb.sceneBoundingRect().contains(pos):
            mouse_point = vb.mapSceneToView(pos)
            self.statusbar.setStyleSheet("font-size: 30px;")
            self.statusbar.showMessage(f"x={mouse_point.x():.2f}  y={mouse_point.y():.2f}")

    def rescale(self, x, y):
        # metrics = pg.QtGui.QFontMetrics(QFont("Arial", 18))
        # axis = self.plot.getAxis('left')  # 或 'left'
        # ticks = axis.tickValues(18, axis.range[0], axis.range[1])
        # # ticks 是 list，每個元素是 (scale, [tick位置list])
        # h = metrics.horizontalAdvance('')
        # for scale, tick_list in ticks:
        #     print(scale, tick_list)
        #     tick = tick_list[:]
        #     if len(tick) > 0:
        #         for t in tick:
        #             if metrics.horizontalAdvance(str(t)) > h:
        #                 h = metrics.horizontalAdvance(str(t))
        #                 mt = t
        # print('maximum ytick:',h, 'longest tick:', mt)
        # self.px, self.py = 46, 0
        # lx, ly = len(x), len(y)
        # self.x_scale = (self.w-self.px-498) / lx
        # self.y_scale = (self.h-290) / ly
        # self.dx, self.dy = lx*self.x_scale, ly*self.y_scale
        
        # self.imrect = pg.QtCore.QRectF(self.px, self.py, self.dx, self.dy)
        return
    
    def det_lim(self, axis, det_array)->tuple[float, float]:
        for i in range(len(axis)):
            if det_array[i]:
                xl=axis[i]
                if i==0:
                    xl+=0.00011
                break
        for i in range(len(axis)-1, -1, -1):
            if det_array[i]:
                xh=axis[i]
                if i==len(axis)-1:
                    xh-=0.00011
                break
        return xl, xh

    def gen_slider_color(self, axis, det)->str:
        text = "stop:0.0 #bcbcbc,"
        xl, xh = self.det_lim(axis, det)
        for i in range(len(axis)):
            if axis[i] >= xl:
                start = axis[i]+0.00011
                for j in range(i, len(axis)):
                    if det[j]==False:
                        end = axis[j-1]-0.00011
                        body = f"""stop:{start-0.0001} #bcbcbc,
                                stop:{start} #00AA00,
                                stop:{end} #00AA00,
                                stop:{end+0.0001} #bcbcbc,
                            """
                        text += body
                        break
                try:
                    xl, xh = self.det_lim(axis[j:], det[j:])
                except:
                    break
        text += "stop:1.0 #bcbcbc);"
        return text
    
    def update_binned_data(self, save=False, indky=None, indkx=None, init=False):
        # 對整個三維資料 binning
        bin_e = self.bin_e_spin.value()
        bin_kx = self.bin_kx_spin.value()
        bin_ky = self.bin_ky_spin.value()
        arr=[]
        if init:
            if self.mode == 'standard':
                path = os.path.join(self.path, '__disp__.zarr')
                try:
                    if os.path.exists(path):
                        self.raw_data_show = zarr.open_group(path, mode='r+')['data']
                        self.path_angle = zarr.open_group(path, mode='r+')['ang'][0]
                        self.rotate_slider.setValue(int(self.path_angle*10))
                        self.rotate_edit.setText(f"{self.path_angle:.1f}")
                        self.prload = True
                    else:
                        input_path = self.path
                        output_path = path
                        disp_zarr_save(input_path, output_path, self.shape, self.max_value)
                        os.system(f'attrib +h +s "{path}"')
                        for name in os.listdir(path):
                            item_path = os.path.join(path, name)
                            if os.path.isfile(item_path):
                                os.system(f'attrib +h +s "{item_path}"')
                            elif os.path.isdir(item_path):
                                os.system(f'attrib +h +s "{item_path}"')
                        self.raw_data_show = zarr.open_group(path, mode='r+')['data']
                        self.prload = False
                except Exception as e:
                    print(e)
                    self.raw_data_show = np.asarray(self.data(slice(None))/self.max_value*255, dtype=np.uint8)
                    self.prload = False
            elif self.mode == 'display':
                self.raw_data_show = zarr.open(os.path.join(self.path, 'data'), mode='r')
                self.path_angle = zarr.open_group(self.path, mode='r+')['ang'][0]
                self.rotate_slider.setValue(int(self.path_angle*10))
                self.rotate_edit.setText(f"{self.path_angle:.1f}")
                self.prload = True
            self.data_show = self.raw_data_show
        elif not save:
            if bin_e == 1 and bin_kx == 1 and bin_ky == 1:
                self.data_show = self.raw_data_show
            else:
                self.data_show = self.bin_data(self.raw_data_show, axis=0, bin_size=bin_e)
                self.data_show = self.bin_data(self.data_show, axis=1, bin_size=bin_ky)
                self.data_show = self.bin_data(self.data_show, axis=2, bin_size=bin_kx)
        else:   #save (only allow path_angle==0)
            self.get_raw_data(save=True)
            if bin_e == 1 and bin_kx == 1 and bin_ky == 1:
                arr = self.raw_data[:, indky, indkx]     #fix
                self.raw_data = None
                # self.data_show = arr
            else:
                arr = self.bin_data(self.raw_data, axis=0, bin_size=bin_e, save=save)     #fix
                self.raw_data = None
                arr = self.bin_data(arr, axis=1, bin_size=bin_ky, save=save)
                arr = self.bin_data(arr, axis=2, bin_size=bin_kx, save=save)
                arr = arr[:, indky, indkx]
                # self.data_show = arr
            
        
        self.E = self.raw_E[:self.data_show.shape[0]*bin_e].reshape(-1, bin_e).mean(axis=1) if bin_e > 1 else self.raw_E
        self.ky = self.raw_ky[:self.data_show.shape[1]*bin_ky].reshape(-1, bin_ky).mean(axis=1) if bin_ky > 1 else self.raw_ky
        self.kx = self.raw_kx[:self.data_show.shape[2]*bin_kx].reshape(-1, bin_kx).mean(axis=1) if bin_kx > 1 else self.raw_kx
        # 更新滑桿
        self.slider_E.setMaximum(len(self.E)-1)
        self.slider_kx.setMaximum(len(self.kx)-1)
        self.slider_ky.setMaximum(len(self.ky)-1)
        det = (self.data_show[:, :, :].sum(axis=2).sum(axis=1) > 0)
        axis=(self.E-self.E[0])/(self.E[-1]-self.E[0])
        el, eh = self.det_lim(axis, det)
        w=eh-el
        self.slider_E.setStyleSheet("""
                                    QSlider::handle:horizontal {
                                        background: #007AD9;
                                        width: 40px;      /* 控制滑塊寬度 */
                                        height: 40px;     /* 控制滑塊高度（對水平slider沒影響，但可加大垂直slider）*/
                                        margin: -10px 0;  /* 讓滑塊更突出 */
                                    }
                                    QSlider::groove:horizontal {
                                        height: 10px;     /* 控制滑道粗細 */
                                        border-radius: 5px;
                                        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                    """+f"""
                                            stop:0.0 #bcbcbc,
                                            stop:{el-0.0001} #bcbcbc,
                                            stop:{el} #2A2A2A,
                                            stop:{el+0.2*w} #669AE6,
                                            stop:{el+0.4*w} #006600,
                                            stop:{eh-0.4*w} #80FF00,
                                            stop:{eh-0.2*w} #FFFF00,
                                            stop:{eh} #FF0000,
                                            stop:{eh+0.0001} #bcbcbc,
                                            stop:1.0 #bcbcbc);
                                    """+"""
                                    }
                                    """)
        det = (self.data_show[:, :, :].sum(axis=1).sum(axis=0) > 0)
        axis=(self.kx-self.kx[0])/(self.kx[-1]-self.kx[0])
        text = self.gen_slider_color(axis, det)
        self.slider_kx.setStyleSheet("""
                                    QSlider::handle:horizontal {
                                        background: #007AD9;
                                        width: 40px;      /* 控制滑塊寬度 */
                                        height: 40px;     /* 控制滑塊高度（對水平slider沒影響，但可加大垂直slider）*/
                                        margin: -10px 0;  /* 讓滑塊更突出 */
                                    }
                                    QSlider::groove:horizontal {
                                        height: 10px;     /* 控制滑道粗細 */
                                        border-radius: 5px;
                                        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                    """+text+"""
                                    }
                                    """)
        det = (self.data_show[:, :, :].sum(axis=2).sum(axis=0) > 0)
        axis=(self.ky-self.ky[0])/(self.ky[-1]-self.ky[0])
        text = self.gen_slider_color(axis, det)
        self.slider_ky.setStyleSheet("""
                                    QSlider::handle:horizontal {
                                        background: #007AD9;
                                        width: 40px;      /* 控制滑塊寬度 */
                                        height: 40px;     /* 控制滑塊高度（對水平slider沒影響，但可加大垂直slider）*/
                                        margin: -10px 0;  /* 讓滑塊更突出 */
                                    }
                                    QSlider::groove:horizontal {
                                        height: 10px;     /* 控制滑道粗細 */
                                        border-radius: 5px;
                                        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                    """+text+"""
                                    }
                                    """)
        
        for i in [self.slider_E, self.slider_kx, self.slider_ky]:
            if i.value() > i.maximum():
                i.setValue(i.maximum())
        # Asign binnied value
        self.bin_e = bin_e
        self.bin_kx = bin_kx
        self.bin_ky = bin_ky
        if save:
            return arr
        else:
            return

    def on_bin_change(self):
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        self.apply_rotation(bin=True)
        self.refresh_slice()
        QApplication.restoreOverrideCursor()
    
    # def bin_data(self, arr, axis, bin_size):
    #     if bin_size <= 1:
    #         return arr
    #     shape = arr.shape
    #     new_shape = list(shape)
    #     new_shape[axis] = shape[axis] // bin_size
    #     arr = arr.take(indices=range(0, shape[axis] - shape[axis] % bin_size), axis=axis)
    #     arr = arr.reshape(*shape[:axis], -1, bin_size, *shape[axis+1:])
    #     arr = arr.mean(axis=axis+1)
    #     return arr
    
    def bin_data(self, odata, axis, bin_size, save=False):
        if bin_size <= 1:
            return odata
        # 計算可整除的長度
        length = (odata.shape[axis] // bin_size) * bin_size
        slicer = [slice(None)] * odata.ndim
        slicer[axis] = slice(0, length)
        data = odata[tuple(slicer)]
        odata = None
        # 重新 shape
        new_shape = list(data.shape)
        new_shape[axis] = length // bin_size
        new_shape.insert(axis + 1, bin_size)
        size = det_chunk(self.shape[(axis+1)%3], self.shape[(axis+2)%3], dtype=data.dtype)
        end = length
        if size/end > 1.2:
            odata = data.reshape(new_shape)
            if save:
                output = odata.mean(axis=axis + 1)
            else:
                output = odata.mean(axis=axis + 1)[:].astype(np.uint8)
            odata = None
        else:
            old_shape = list(data.shape)
            old_shape[axis] = self.shape[axis]//bin_size
            if save:
                bin_data = zarr.open(os.path.join(cdir, '.MDC_cut_DataViewer', 'bin'), mode='w', shape=old_shape, dtype=np.float32)
            else:
                bin_data = zarr.open(os.path.join(cdir, '.MDC_cut_DataViewer', 'bin'), mode='w', shape=old_shape, dtype=np.uint8)
            step = int(min(bin_size*size//bin_size, bin_size*(end//1.5)//bin_size))
            step = int(min(size//bin_size*bin_size, end//bin_size*bin_size))
            end = end-end%bin_size
            for i in range(0, end, step):
                ind = slice(i, min(i + step, end))
                dslicer = [slice(None)] * data.ndim
                dslicer[axis] = ind
                stop = i
                for j in range(i, min(i + step, end)):
                    stop+=1
                # print('dslicer: ', dslicer)
                bin_slicer = [slice(None)] * data.ndim
                bin_slicer[axis] = slice(i//bin_size, stop//bin_size)
                # print('bin_slicer: ', bin_slicer)
                mean_shape = list(data.shape)
                mean_shape[axis] = int(abs(i - stop)/bin_size)
                mean_shape.insert(axis + 1, bin_size)
                # print('mean_shape: ', mean_shape)
                bin_data[tuple(bin_slicer)] = data[tuple(dslicer)].reshape(mean_shape).mean(axis=axis+1)[:]
                print('Progress: %.2f%%'%(min(i + step, end)/end*100))
            data = None
            output = zarr.open(os.path.join(cdir, '.MDC_cut_DataViewer', 'bin'), mode='r')
        return output

    def update_E_slice(self, idx, init=False):
        self.current_mode = 'E'
        self.export_btn.hide()
        self.rotate_label.show()
        self.rotate_slider.show()
        self.rotate_edit.show()
        self.rotate_btn.show()
        self.xlow_edit.hide()
        self.xhigh_edit.hide()
        self.xlow_label.hide()
        self.xhigh_label.hide()
        arr = self.data_show[idx, :, :]
        kx_bin = self.kx
        ky_bin = self.ky
        # 旋轉
        if self.prload:
            self.angle = np.float32(self.rotate_edit.text()) - self.path_angle
        else:
            self.angle = np.float32(self.rotate_edit.text())
        if self.angle != 0:
            arr = self.rotate_array()
        kx_bin = self.kx[:arr.shape[1]]
        ky_bin = self.ky[:arr.shape[0]]
        self.xl, self.yl, self.xh, self.yh, self.dx, self.dy = kx_bin[0], ky_bin[0], kx_bin[-1], ky_bin[-1], kx_bin[-1] - kx_bin[0], ky_bin[-1] - ky_bin[0]
        
        if not init:
            if self.xRange != self.plot.getViewBox().viewRange()[0] or self.yRange != self.plot.getViewBox().viewRange()[1]:
                self.xRange, self.yRange = self.plot.getViewBox().viewRange()[0], self.plot.getViewBox().viewRange()[1]
        else:
            self.set_cmap()
            self.xRange, self.yRange = (self.xl, self.xh), (self.yl, self.yh)
        self.update_E_job(arr)

        self.v_cross = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen((255, 0, 0), width=2, style=Qt.DotLine))
        self.v_cross.setZValue(10)
        self.h_cross = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen((255, 0, 0), width=2, style=Qt.DotLine))
        self.h_cross.setZValue(10)
        self.v_cross.setValue(0)
        self.h_cross.setValue(0)
        self.plot.addItem(self.v_cross, ignoreBounds=True)
        self.plot.addItem(self.h_cross, ignoreBounds=True)
        self.v_cross.setVisible(False)
        self.h_cross.setVisible(False)

    def update_kx_slice(self, idx, setlim=False):
        self.current_mode = 'kx'
        self.export_btn.show()
        self.rotate_label.hide()
        self.rotate_slider.hide()
        self.rotate_edit.hide()
        self.rotate_btn.hide()
        self.xlow_edit.show()
        self.xhigh_edit.show()
        self.xlow_label.show()
        self.xhigh_label.show()
        self.xlow_label.setText("ky min")
        self.xhigh_label.setText("ky max")
        
        if not setlim:
            det = (self.data_show[:, :, idx].sum(axis=0) > 0)
            if det.any():
                xl, xh = self.det_lim(self.ky, det)
                d=abs(xl-xh)/20
                self.xlow_edit.setText(f'{xl-d:.3f}')
                self.xhigh_edit.setText(f'{xh+d:.3f}')
            else:
                xl = self.ky[0]
                xh = self.ky[-1]
                self.xlow_edit.setText(f'{xl:.3f}')
                self.xhigh_edit.setText(f'{xh:.3f}')
        
        xlow = float(self.xlow_edit.text())
        xhigh = float(self.xhigh_edit.text())
        try:
            li = np.argwhere(self.ky-xlow <= (self.ky[1]-self.ky[0])/2)[-1][0]
        except IndexError:
            li = 0
        try:
            hi = np.argwhere(self.ky-xhigh >= (self.ky[1]-self.ky[0])/2)[0][0]
        except IndexError:
            hi = len(self.ky) - 1
        arr = self.data_show[:, li:hi+1, idx]
        E_bin = self.E
        ky_bin = self.ky[li:hi+1]
        dx, dy = ky_bin[-1] - ky_bin[0], E_bin[-1] - E_bin[0]
        
        self.plot.clear()
        img_item = pg.ImageItem(arr.T)
        img_item.setLevels(np.min(arr), np.max(arr))
        self.hist.setImageItem(img_item)
        self.plot.setAspectLocked(False)  # 鎖定比例
        self.plot.setLimits(xMin=ky_bin[0]-dx/2, xMax=ky_bin[-1]+dx/2, yMin=E_bin[0]-dy/2, yMax=E_bin[-1]+dy/2)
        
        self.plot.setRange(xRange=(xlow, xhigh), yRange=(E_bin[0], E_bin[-1]), padding=0)
        # self.rescale(E_bin, ky_bin)
        rect = pg.QtCore.QRectF(ky_bin[0], E_bin[0], ky_bin[-1] - ky_bin[0], E_bin[-1] - E_bin[0])  # 真實位置
        img_item.setRect(rect)
        self.plot.addItem(img_item)
        self.xlabel.setPixmap(self.ky_pixmap_x)
        self.ylabel.setPixmap(self.E_pixmap_y)
        self.statusbar.setStyleSheet("font-size: 30px;")
        self.statusbar.showMessage(f"kx index: {idx} (kx={self.kx[idx]:.3f})")
        self.label_kx.setText(f'kx Slice (E-ky) Index: {idx} (kx={self.kx[idx]:.3f})')

    def update_ky_slice(self, idx, setlim=False):
        self.current_mode = 'ky'
        self.export_btn.show()
        self.rotate_label.hide()
        self.rotate_slider.hide()
        self.rotate_edit.hide()
        self.rotate_btn.hide()
        self.xlow_edit.show()
        self.xhigh_edit.show()
        self.xlow_label.show()
        self.xhigh_label.show()
        self.xlow_label.setText("kx min")
        self.xhigh_label.setText("kx max")
        
        if not setlim:
            det = (self.data_show[:, idx, :].sum(axis=0) > 0)
            if det.any():
                xl, xh = self.det_lim(self.kx, det)
                d=abs(xl-xh)/20
                self.xlow_edit.setText(f'{xl-d:.3f}')
                self.xhigh_edit.setText(f'{xh+d:.3f}')
            else:
                xl = self.kx[0]
                xh = self.kx[-1]
                self.xlow_edit.setText(f'{xl:.3f}')
                self.xhigh_edit.setText(f'{xh:.3f}')
        
        xlow = float(self.xlow_edit.text())
        xhigh = float(self.xhigh_edit.text())
        try:
            li = np.argwhere(self.kx-xlow <= (self.kx[1]-self.kx[0])/2)[-1][0]
        except IndexError:
            li = 0
        try:
            hi = np.argwhere(self.kx-xhigh >= (self.kx[1]-self.kx[0])/2)[0][0]
        except IndexError:
            hi = len(self.kx) - 1
        arr = self.data_show[:, idx, li:hi+1]
        E_bin = self.E
        kx_bin = self.kx[li:hi+1]
        dx, dy = kx_bin[-1] - kx_bin[0], E_bin[-1] - E_bin[0]

        self.plot.clear()
        img_item = pg.ImageItem(arr.T)
        img_item.setLevels(np.min(arr), np.max(arr))
        self.hist.setImageItem(img_item)
        self.plot.setAspectLocked(False)  # 鎖定比例
        self.plot.setLimits(xMin=kx_bin[0]-dx/2, xMax=kx_bin[-1]+dx/2, yMin=E_bin[0]-dy/2, yMax=E_bin[-1]+dy/2)
        
        self.plot.setRange(xRange=(xlow, xhigh), yRange=(E_bin[0], E_bin[-1]), padding=0)
        # self.rescale(E_bin, ky_bin)
        rect = pg.QtCore.QRectF(kx_bin[0], E_bin[0], kx_bin[-1] - kx_bin[0], E_bin[-1] - E_bin[0])  # 真實位置
        img_item.setRect(rect)
        self.plot.addItem(img_item)
        self.xlabel.setPixmap(self.kx_pixmap_x)
        self.ylabel.setPixmap(self.E_pixmap_y)
        self.statusbar.setStyleSheet("font-size: 30px;")
        self.statusbar.showMessage(f"ky index: {idx} (ky={self.ky[idx]:.3f})")
        self.label_ky.setText(f'ky Slice (E-kx) Index: {idx} (ky={self.ky[idx]:.3f})')

    def refresh_slice(self):
        if self.current_mode == 'E':
            self.update_E_slice(self.slider_E.value())
        elif self.current_mode == 'kx':
            self.update_kx_slice(self.slider_kx.value(), setlim=True)
        elif self.current_mode == 'ky':
            self.update_ky_slice(self.slider_ky.value(), setlim=True)

    def export_slice(self):
        # 匯出時也裁切 xlow/xhigh
        if self.path_angle == 0:
            if self.current_mode == 'kx':
                self.gen_E_ky()
            elif self.current_mode == 'ky':
                self.gen_E_kx()
        else:
            if self.mode == 'standard':
                self.save_as_zarr(h5=True)
            elif self.mode == 'display':
                self.save_as_zarr_disp(h5=True)
    
    def gen_E_kx(self, event=None):
        if self.hwnd:
            windll.user32.ShowWindow(self.hwnd, 9)
            windll.user32.SetForegroundWindow(self.hwnd)
        self.__md(axis="E_kx")
        v = self.ky[self.slider_ky.value()]
        self.name = f"{self.dirname}_ky_{v:.3f}"
        self.file = os.path.join(self.dir, f"{self.name}.h5")
        self.gen_h5(axis='kx')
        print(f"Save to {self.file}")
        self.statusbar.setStyleSheet("font-size: 15px;")
        self.statusbar.showMessage(f"Exported to {self.file}")

    def gen_E_ky(self, event=None):
        if self.hwnd:
            windll.user32.ShowWindow(self.hwnd, 9)
            windll.user32.SetForegroundWindow(self.hwnd)
        self.__md(axis="E_ky")
        v = self.kx[self.slider_kx.value()]
        self.name = f"{self.dirname}_kx_{v:.3f}"
        self.file = os.path.join(self.dir, f"{self.name}.h5")
        self.gen_h5(axis='ky')
        print(f"Save to {self.file}")
        self.statusbar.setStyleSheet("font-size: 15px;")
        self.statusbar.showMessage(f"Exported to {self.file}")
    
    def gen_h5(self, axis='kx'):
        with h5py.File(self.file, "w") as f:
            e_photon = 21.2  # deafault He I photon energy
            desc=["Sliced data"]
            y = self.E
            xlow, xhigh = self.plot.getViewBox().viewRange()[0]
            if axis == 'kx':
                try:
                    li = np.argwhere(self.kx-xlow <= (self.kx[1]-self.kx[0])/2)[-1][0]
                except IndexError:
                    li = 0
                try:
                    hi = np.argwhere(self.kx-xhigh >= (self.kx[1]-self.kx[0])/2)[0][0]
                except IndexError:
                    hi = len(self.kx) - 1
                x = self.kx[li:hi+1]
            else:
                try:
                    li = np.argwhere(self.ky-xlow <= (self.ky[1]-self.ky[0])/2)[-1][0]
                except IndexError:
                    li = 0
                try:
                    hi = np.argwhere(self.ky-xhigh >= (self.ky[1]-self.ky[0])/2)[0][0]
                except IndexError:
                    hi = len(self.ky) - 1
                x = self.ky[li:hi+1]
            xsize = np.array([len(y)], dtype=int)
            f.create_dataset('Data/XSize/Value', data=xsize, dtype=int)
            ysize = np.array([len(x)], dtype=int)
            f.create_dataset('Data/YSize/Value', data=ysize, dtype=int)
            
            acquisition = [bytes('DataCube', 'utf-8')]
            acquisition = np.array(acquisition, dtype=h5py.special_dtype(vlen=str))
            f.create_dataset('Region/Acquisition', data=acquisition, dtype=h5py.special_dtype(vlen=str))
            center_energy = np.array([(y[-1]+y[0])/2], dtype=float)
            f.create_dataset('Region/CenterEnergy/Value', data=center_energy, dtype=float)
            description = np.array([bytes(desc[0], 'utf-8')], dtype=h5py.special_dtype(vlen=str))
            f.create_dataset('Region/Description', data=description, dtype=h5py.special_dtype(vlen=str))
            dwell = np.array([bytes('Unknown', 'utf-8')], dtype=h5py.special_dtype(vlen=str))
            f.create_dataset('Region/Dwell/Value', data=dwell, dtype=h5py.special_dtype(vlen=str))
            
            energy_mode = [bytes('Kinetic', 'utf-8')]
            energy_mode = np.array(energy_mode, dtype=h5py.special_dtype(vlen=str))
            f.create_dataset('Region/EnergyMode', data=energy_mode, dtype=h5py.special_dtype(vlen=str))
            excitation_energy = np.array([e_photon], dtype=float)
            f.create_dataset('Region/ExcitationEnergy/Value', data=excitation_energy, dtype=float)
            high_energy = np.array([y[-1]], dtype=float)
            f.create_dataset('Region/HighEnergy/Value', data=high_energy, dtype=float)
            iterations = np.array([bytes('Unknown', 'utf-8')], dtype=h5py.special_dtype(vlen=str))
            f.create_dataset('Region/Iterations/Value', data=iterations, dtype=h5py.special_dtype(vlen=str))
            
            lens_mode = [bytes('Angular', 'utf-8')]
            lens_mode = np.array(lens_mode, dtype=h5py.special_dtype(vlen=str))
            f.create_dataset('Region/LensMode', data=lens_mode, dtype=h5py.special_dtype(vlen=str))
            low_energy = np.array([y[0]], dtype=float)
            f.create_dataset('Region/LowEnergy/Value', data=low_energy, dtype=float)
            name = np.array([bytes(self.name, 'utf-8')], dtype=h5py.special_dtype(vlen=str))
            f.create_dataset('Region/Name', data=name, dtype=h5py.special_dtype(vlen=str))
            y_scale_max = np.array([x[-1]], dtype=float)
            f.create_dataset('Region/YScaleMax/Value', data=y_scale_max, dtype=float)
            y_scale_min = np.array([x[0]], dtype=float)
            f.create_dataset('Region/YScaleMin/Value', data=y_scale_min, dtype=float)
            pass_energy = np.array([bytes('Unknown', 'utf-8')], dtype=h5py.special_dtype(vlen=str))
            f.create_dataset('Region/PassEnergy/Value', data=pass_energy, dtype=h5py.special_dtype(vlen=str))
            step = np.array([y[1]-y[0]], dtype=float)
            f.create_dataset('Region/Step/Value', data=step, dtype=float)
            
            slit = [bytes('Unknown', 'utf-8')]
            slit = np.array(slit, dtype=h5py.special_dtype(vlen=str))
            f.create_dataset('Region/Slit', data=slit, dtype=h5py.special_dtype(vlen=str))
            if axis == 'kx':
                arr = self.update_binned_data(save=True, indky=self.slider_ky.value(), indkx=slice(li,hi+1))
                f.create_dataset("Spectrum", data=np.array(arr))
            else:
                arr = self.update_binned_data(save=True, indky=slice(li,hi+1), indkx=self.slider_kx.value())
                f.create_dataset("Spectrum", data=np.array(arr))
            arr = None

    def __path_angle(self):
        while self.path_angle < 0:
            self.path_angle += 360
        while self.path_angle >= 360:
            self.path_angle -= 360
    
    def __md(self, axis="E_kx"):
        if self.hwnd:
            windll.user32.ShowWindow(self.hwnd, 9)
            windll.user32.SetForegroundWindow(self.hwnd)
        self.dirname = os.path.basename(self.path).removesuffix(".zarr")
        self.__path_angle()
        self.dir = os.path.join(self.path, f"Ang_{self.path_angle:.1f}_bin_{max(1, self.bin_e)}_{max(1, self.bin_kx)}_{max(1, self.bin_ky)}", axis)
        os.makedirs(self.dir, exist_ok=True)

    def set_exp_h5_btn(self):
        if self.path_angle != 0:
            self.export_btn.setToolTip("Original data should be rotated which is time consuming.\nSave as a new zarr and reload that file to export HDF5.")
            self.export_btn.setText("Save, reload, then export to HDF5")
        else:
            self.export_btn.setToolTip(None)
            self.export_btn.setText("Export to HDF5 File")
    
    def save_as_zarr(self, h5=False):
        self.apply_rotation(save=True)
        
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        pr_bar = ProgressDialog(4, self.icon)
        pr_bar.resize(self.w//3, self.h//4)
        pr_bar.show()
        pr_bar.increaseProgress('Preparing Main Data Array')
        
        xmin, xmax = self.raw_kx[0], self.raw_kx[-1]
        ymin, ymax = self.raw_ky[0], self.raw_ky[-1]
        ty = self.raw_E
        
        pr_bar.increaseProgress('Preparing Metadata for Attributes')
        pr_bar.increaseProgress('Combining Data Cube')
        size = det_chunk(self.shape[1], dtype=np.float32)
        end = self.shape[0]
        attr_array = np.zeros((self.shape[0], self.shape[1], 1))
        attr_array[:, 0, 0] = ty
        attr_array[0, 1, 0] = xmin
        attr_array[1, 1, 0] = xmax
        attr_array[2, 1, 0] = ymin
        attr_array[3, 1, 0] = ymax
        os.chdir(os.path.dirname(self.path))
        QApplication.restoreOverrideCursor()
        path, _ = QFileDialog.getSaveFileName(None, "Save Zarr File", f'data_cube_Ang_{self.path_angle:.1f}.zarr', "Zarr Files (*.zarr)")
        if path == "":
            attr_array, self.raw_data = None, None
            return

        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        pr_bar.increaseProgress('Combining Data Cube')
        if self.raw_data is not None:
            zdata = np.append(self.raw_data, attr_array, axis=2)
        pr_bar.increaseProgress('Done')
        pr_bar.close()
        
        pr_bar = ProgressDialog(3, self.icon)
        pr_bar.resize(self.w//3, self.h//4)
        pr_bar.show()
        pr_bar.increaseProgress('Saving data')
        if self.raw_data is not None:
            zarr.save(path, zdata)
            attr_array, zdata, self.raw_data = None, None, None
        else:
            zdata, self.raw_data = None, None
            rot_data = zarr.open(os.path.join(cdir, '.MDC_cut_DataViewer', 'rot_data'), mode='r')
            save_data = zarr.open(path, mode='w', shape=(self.shape[0], self.shape[1], self.shape[2]+1), dtype=np.float32)
            step = int(min(size, self.shape[0]//1.5))
            for i in range(0, end, step):
                ind = slice(i, min(i + step, end))
                odata = np.empty((abs(i-min(i + step, end)), self.shape[1], self.shape[2]+1), dtype=np.float32)
                for j in range(i, min(i + step, end)):
                    odata[j-i, :, :-1] = rot_data[j,...]
                    odata[j-i, :, -1] = attr_array[j, :, 0]
                    pr_bar.increaseProgress()
                save_data[ind,...] = odata
                odata = None
                print('Progress: %.2f%%'%(min(i + step, end)/end*100))
            attr_array = None
        
        pr_bar.increaseProgress('Setting file attributes')
        for name in os.listdir(path):
            item_path = os.path.join(path, name)
            if os.path.isfile(item_path):
                os.system(f'attrib +h +s "{item_path}"')
            elif os.path.isdir(item_path):
                os.system(f'attrib +h +s "{item_path}"')
        
        
        disp_path = os.path.join(path, '__disp__.zarr')
        size = det_chunk(self.shape[1], dtype=np.uint8)
        end = self.shape[0]
        if size/end >= 1.2:
            zdata = np.asarray(self.raw_data_show, dtype=np.uint8)
            zarr.save_group(disp_path, data=zdata, ang=np.array([0, 0],dtype=np.float32))
            zdata = None
            os.system(f'attrib +h +s "{disp_path}"')
            for name in os.listdir(disp_path):
                item_path = os.path.join(disp_path, name)
                if os.path.isfile(item_path):
                    os.system(f'attrib +h +s "{item_path}"')
                elif os.path.isdir(item_path):
                    os.system(f'attrib +h +s "{item_path}"')
        else:
            rot_data = zarr.open(os.path.join(cdir, '.MDC_cut_DataViewer', 'rot_data'), mode='r')
            save_data = zarr.open(os.path.join(disp_path, '__disp__.zarr', 'data'), mode='w', shape=self.shape, dtype=np.uint8)
            save_ang = zarr.open(os.path.join(disp_path, '__disp__.zarr', 'ang'), mode='w', shape=(2,), dtype=np.float32)
            save_ang[:] = np.array([0, 1], dtype=np.float32)
            step = int(min(size, self.shape[0]//1.5))
            for i in range(0, end, step):
                ind = slice(i, min(i + step, end))
                odata = np.empty((abs(i-min(i + step, end)), self.shape[1], self.shape[2]), dtype=np.uint8)
                for j in range(i, min(i + step, end)):
                    odata[j-i, :, :] = np.asarray(rot_data[j,...]/self.max_value*255, dtype=np.uint8)
                    pr_bar.increaseProgress()
                save_data[ind,...] = odata
                odata = None
                print('Progress: %.2f%%'%(min(i + step, end)/end*100))
            save_ang[1] = 0
            
        pr_bar.increaseProgress('Done')
        pr_bar.close()
        QApplication.restoreOverrideCursor()
        if h5:
            os.system(f'python -W ignore::SyntaxWarning -W ignore::UserWarning "{os.path.join(cdir, '.MDC_cut', 'tool', 'DataViewer.py')}" -f "{path}"')
        
    
    def save_as_zarr_disp(self, h5=False):
        self.apply_rotation()
        
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        pr_bar = ProgressDialog(3, self.icon)
        pr_bar.resize(self.w//3, self.h//4)
        pr_bar.show()
        pr_bar.increaseProgress('Preparing Main Data Array')
        
        size = det_chunk(self.shape[1], dtype=np.uint8)
        end = self.shape[0]
        if size/end >= 1.2:
            data = np.asarray(self.raw_data_show, dtype=np.uint8)
        xmin, xmax = self.raw_kx[0], self.raw_kx[-1]
        ymin, ymax = self.raw_ky[0], self.raw_ky[-1]
        ty = self.raw_E
        
        pr_bar.increaseProgress('Preparing Metadata for Attributes')
        attr_array = np.zeros((self.shape[0], 2))
        attr_array[:, 0] = ty
        attr_array[0, 1] = xmin
        attr_array[1, 1] = xmax
        attr_array[2, 1] = ymin
        attr_array[3, 1] = ymax
        pr_bar.increaseProgress('Done')
        pr_bar.close()
        os.chdir(os.path.dirname(self.path))
        QApplication.restoreOverrideCursor()
        path, _ = QFileDialog.getSaveFileName(None, "Save Zarr File", f'data_cube_Ang_{self.path_angle:.1f}_disp.zarr', "Zarr Files (*.zarr)")
        if path == "":
            data, attr_array = None, None
            return

        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        pr_bar = ProgressDialog(3, self.icon)
        pr_bar.resize(self.w//3, self.h//4)
        pr_bar.show()
        pr_bar.increaseProgress('Saving data')
        if size/end >= 1.2:
            zarr.save_group(path, data=data, attr_array=attr_array, ang=np.array([0, 0], dtype=np.float32))
            data, attr_array = None, None
        else:
            save_data = zarr.open(os.path.join(path, 'data'), mode='w', shape=self.shape, dtype=np.uint8)
            save_attr = zarr.open(os.path.join(path, 'attr_array'), mode='w', shape=(self.shape[0], 2), dtype=np.float32)
            save_attr[:] = attr_array
            attr_array = None
            save_ang = zarr.open(os.path.join(path, 'ang'), mode='w', shape=(2,), dtype=np.float32)
            save_ang[:] = np.array([0, 1], dtype=np.float32)
            step = int(min(size, self.shape[0]//1.5))
            for i in range(0, end, step):
                ind = slice(i, min(i + step, end))
                odata = np.empty((abs(i-min(i + step, end)), self.shape[1], self.shape[2]), dtype=np.uint8)
                for j in range(i, min(i + step, end)):
                    odata[j-i, :, :] = data[j,...]
                    pr_bar.increaseProgress()
                save_data[ind,...] = odata
                odata = None
                print('Progress: %.2f%%'%(min(i + step, end)/end*100))
            data = None
            save_ang[1] = 0
            
        pr_bar.increaseProgress('Setting file attributes')
        for name in os.listdir(path):
            item_path = os.path.join(path, name)
            if os.path.isfile(item_path):
                os.system(f'attrib +h +s "{item_path}"')
            elif os.path.isdir(item_path):
                os.system(f'attrib +h +s "{item_path}"')
        pr_bar.increaseProgress('Done')
        pr_bar.close()
        QApplication.restoreOverrideCursor()
        if h5:
            os.system(f'python -W ignore::SyntaxWarning -W ignore::UserWarning "{os.path.join(cdir, '.MDC_cut', 'tool', 'DataViewer.py')}" -f "{path}"')

    def update_E_job(self, arr):
        img_item = pg.ImageItem(arr.T)
        img_item.setLevels(np.min(arr), np.max(arr))
        self.hist.setImageItem(img_item)
        self.plot.clear()
        self.plot.getViewBox().state['limits']['xLimits'] = [self.xl - self.dx/2, self.xh + self.dx/2]
        self.plot.getViewBox().state['limits']['yLimits'] = [self.yl - self.dy/2, self.yh + self.dy/2]
        # self.plot.setLimits(xMin=self.xl-self.dx/2, xMax=self.xh+self.dx/2, yMin=self.yl-self.dy/2, yMax=self.yh+self.dy/2)   # ViewBox Line:776 RuntimeError
        self.plot.setRange(xRange=self.xRange, yRange=self.yRange, padding=0)

        # self.plot.enableAutoRange()
        self.plot.setAspectLocked(True)  # 鎖定比例

        # self.rescale(kx_bin, ky_bin)
        rect = pg.QtCore.QRectF(self.xl, self.yl, self.dx, self.dy)  # 真實位置
        img_item.setRect(rect)
        self.plot.addItem(img_item)
        
        self.xlabel.setPixmap(self.kx_pixmap_x)
        self.ylabel.setPixmap(self.ky_pixmap_y)
        idx = self.slider_E.value()
        self.statusbar.setStyleSheet("font-size: 30px;")
        self.statusbar.showMessage(f"E index: {idx} (E={self.E[idx]:.3f}), angle={self.angle:.1f}")
        self.label_E.setText(f'E Slice (ky-kx) Index: {idx} (E={self.E[idx]:.3f})')

    def sync_rotate_edit(self):
        value = self.rotate_slider.value()
        angle = value / 10.0
        if self.prload:
            self.angle = angle - self.path_angle
        else:
            self.angle = angle
        self.rotate_edit.setText(f"{angle:.1f}")
        self.v_cross.setVisible(True)
        self.h_cross.setVisible(True)
        arr = self.rotate_array()
        self.xRange, self.yRange = self.plot.getViewBox().viewRange()[0], self.plot.getViewBox().viewRange()[1]
        self.update_E_job(arr)
        

    def sync_rotate_slider(self):
        try:
            value = float(self.rotate_edit.text())
            while value < 0:
                value += 360
            while value >= 360:
                value -= 360
        except Exception:
            value = 0.0
        if self.prload:
            self.angle = value - self.path_angle
        else:
            self.angle = value
        self.rotate_slider.setValue(int(value * 10))
        self.v_cross.setVisible(True)
        self.h_cross.setVisible(True)
        arr = self.rotate_array()
        self.xRange, self.yRange = self.plot.getViewBox().viewRange()[0], self.plot.getViewBox().viewRange()[1]
        self.update_E_job(arr)

    def rot_raw_data(self, pr_bar):
        angle = self.path_angle
        path = os.path.join(self.path, '__disp__.zarr')
        size = det_chunk(self.shape[1], dtype=np.float32)
        end = self.shape[0]
        if self.raw_data is None:   # reflecting lack of memory
            #Saving raw data
            rot_data = zarr.open(os.path.join(cdir, '.MDC_cut_DataViewer', 'rot_data'), mode='w', shape=self.shape, dtype=np.float32)
            step = int(min(size, self.shape[0]//1.5))
            for i in range(0, end, step):
                ind = slice(i, min(i + step, end))
                odata = np.empty((abs(i-min(i + step, end)), self.shape[1], self.shape[2]), dtype=np.float32)
                for j in range(i, min(i + step, end)):
                    odata[j-i,...] = rotate(self.data(j), -angle, self.data(j).shape)
                    pr_bar.increaseProgress()
                rot_data[ind,...] = odata
                odata = None
                print('Progress: %.2f%%'%(min(i + step, end)/end*100))
            
            
            #Saving disp data
            print('Saving display data...')
            size = det_chunk(self.shape[1], dtype=np.float32)
            end = self.shape[0]
            if size/end <1.2:   # threshold: more than 1.2 times memory available
                raw_data_show = zarr.open(os.path.join(path, 'data'), mode='r+')
                angle = zarr.open(os.path.join(path, 'ang'), mode='r+')
                angle[0] = self.path_angle
                step = int(min(size, self.shape[0]//1.5))
                for i in range(0, end, step):
                    ind = slice(i, min(i + step, end))
                    odata = np.empty((abs(i-min(i + step, end)), self.shape[1], self.shape[2]), dtype=np.uint8)
                    for j in range(i, min(i + step, end)):
                        odata[j-i,...] = rotate(raw_data_show[j,...], -self.angle, raw_data_show[j,...].shape)
                        # pr_bar.increaseProgress()
                    raw_data_show[ind,...] = odata
                    odata = None
                    print('Progress: %.2f%%'%(min(i + step, end)/end*100))
                angle[-1] = self.path_angle
            else:
                raw_data_show = np.asarray(zarr.open_group(path, mode='r')['data'], dtype=np.uint8)
                for i in range(self.shape[0]):
                    surface = raw_data_show[i, :, :]
                    surface = rotate(surface, -self.angle, surface.shape)
                    raw_data_show[i, :, :] = surface
                    surface = None
                    # pr_bar.increaseProgress()
                os.chdir(self.path)
                if os.path.exists('__disp__.zarr'):
                    shutil.rmtree('__disp__.zarr')
                
                zarr.save_group(path, data=raw_data_show, ang=np.array([self.path_angle, self.path_angle], dtype=np.float32))
                raw_data_show = None
                os.system(f'attrib +h +s "{path}"')
                
                os.chdir(self.path)
                for name in os.listdir(path):
                    item_path = os.path.join(path, name)
                    if os.path.isfile(item_path):
                        os.system(f'attrib +h +s "{item_path}"')
                    elif os.path.isdir(item_path):
                        os.system(f'attrib +h +s "{item_path}"')
        ###new
        else:
            for i in range(self.shape[0]):
                surface = self.raw_data[i, :, :]
                surface = rotate(surface, -angle, surface.shape)
                self.raw_data[i, :, :] = surface
                surface = None
                pr_bar.increaseProgress()
            os.chdir(self.path)
            if os.path.exists('__disp__.zarr'):
                shutil.rmtree('__disp__.zarr')
            zdata = np.asarray(self.raw_data/self.max_value*255, dtype=np.uint8)
            zarr.save_group(path, data=zdata, ang=np.array([self.path_angle, self.path_angle], dtype=np.float32))
            zdata = None
            os.system(f'attrib +h +s "{path}"')
            for name in os.listdir(path):
                item_path = os.path.join(path, name)
                if os.path.isfile(item_path):
                    os.system(f'attrib +h +s "{item_path}"')
                elif os.path.isdir(item_path):
                    os.system(f'attrib +h +s "{item_path}"')
        pr_bar.increaseProgress('Updating Cube')
        self.raw_data_show = zarr.open_group(path, mode='r+')['data']
        self.prload = False
    
    def rot_raw_data_show(self, pr_bar):
        if self.mode == 'standard':
            path = os.path.join(self.path, '__disp__.zarr')
        elif self.mode == 'display':
            path = self.path
        ###new
        size = det_chunk(self.shape[1], dtype=np.uint8)
        end = self.shape[0]
        if size/end <1.2:   # threshold: more than 1.2 times memory available
            raw_data_show = zarr.open(os.path.join(path, 'data'), mode='r+')
            angle = zarr.open(os.path.join(path, 'ang'), mode='r+')
            angle[0] = self.path_angle
            step = int(min(size, self.shape[0]//1.5))
            for i in range(0, end, step):
                ind = slice(i, min(i + step, end))
                odata = np.empty((abs(i-min(i + step, end)), self.shape[1], self.shape[2]), dtype=np.uint8)
                for j in range(i, min(i + step, end)):
                    odata[j-i,...] = rotate(raw_data_show[j,...], -self.angle, raw_data_show[j,...].shape)
                    pr_bar.increaseProgress()
                raw_data_show[ind,...] = odata
                odata = None
                print('Progress: %.2f%%'%(min(i + step, end)/end*100))
            angle[-1] = self.path_angle
        ###new
        else:
            ###origin
            # heavy memory usage method
            raw_data_show = np.asarray(zarr.open_group(path, mode='r')['data'], dtype=np.uint8) #origin
            for i in range(self.shape[0]):
                surface = raw_data_show[i, :, :]
                surface = rotate(surface, -self.angle, surface.shape)
                raw_data_show[i, :, :] = surface
                surface = None
                pr_bar.increaseProgress()
            os.chdir(self.path)
            if self.mode == 'standard':
                if os.path.exists('__disp__.zarr'):
                    shutil.rmtree('__disp__.zarr')
            elif self.mode == 'display':
                os.chdir(os.path.dirname(self.path))
                for filename in os.listdir(path):
                    file_path = os.path.join(path, filename)
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # 刪除檔案或符號連結
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # 刪除子資料夾
            zdata = np.asarray(raw_data_show, dtype=np.uint8)
            if self.mode == 'standard':
                zarr.save_group(path, data=zdata, ang=np.array([self.path_angle, self.path_angle], dtype=np.float32))
                os.system(f'attrib +h +s "{path}"')
            elif self.mode == 'display':
                xmin, xmax = self.raw_kx[0], self.raw_kx[-1]
                ymin, ymax = self.raw_ky[0], self.raw_ky[-1]
                ty = self.raw_E
                
                attr_array = np.zeros((zdata.shape[0], 2))
                attr_array[:, 0] = ty
                attr_array[0, 1] = xmin
                attr_array[1, 1] = xmax
                attr_array[2, 1] = ymin
                attr_array[3, 1] = ymax
                zarr.save_group(path, data=zdata, attr_array=attr_array, ang=np.array([self.path_angle, self.path_angle], dtype=np.float32))
            os.chdir(self.path)
            for name in os.listdir(path):
                item_path = os.path.join(path, name)
                if os.path.isfile(item_path):
                    os.system(f'attrib +h +s "{item_path}"')
                elif os.path.isdir(item_path):
                    os.system(f'attrib +h +s "{item_path}"')
        ###origin
        pr_bar.increaseProgress('Updating Cube')
        self.raw_data_show = zarr.open_group(path, mode='r+')['data']
        self.prload = False
                
    def apply_rotation(self, save=False, bin=False):
        self.sync_rotate_slider()
        
        self.path_angle += self.angle
        self.__path_angle()
        self.set_exp_h5_btn()
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        pr_bar = ProgressDialog(self.shape[0]+3, self.icon)
        pr_bar.resize(self.w//3, self.h//4)
        pr_bar.show()
        pr_bar.increaseProgress('Preparing Data')
        if save:
            self.get_raw_data()
            if self.path_angle != 0:
                self.rot_raw_data(pr_bar)
            else:
                self.rot_raw_data_show(pr_bar)
        else:
            if self.angle != 0:
                self.rot_raw_data_show(pr_bar)
            
        pr_bar.increaseProgress('Initializing')
        self.rotate_edit.disconnect()
        self.rotate_slider.disconnect()
        self.rotate_edit.setText("0.0")
        self.rotate_slider.setValue(0)
        self.rotate_edit.editingFinished.connect(self.sync_rotate_slider)
        self.rotate_slider.valueChanged.connect(self.sync_rotate_edit)
        if not bin:
            self.bin_e_spin.setValue(1)
            self.bin_kx_spin.setValue(1)
            self.bin_ky_spin.setValue(1)
        self.update_binned_data()
        pr_bar.increaseProgress()
        QApplication.restoreOverrideCursor()

    def rotate_array(self):
        surface = self.data_show[self.slider_E.value(), :, :]
        if surface.shape[0] != surface.shape[1]:
            surface = cv2.resize(surface, (max(surface.shape), max(surface.shape)), interpolation=cv2.INTER_NEAREST)
        surface = rotate(surface, -self.angle, surface.shape)
        return surface

def get_hwnd():
    try:
        with open('hwnd', 'r') as f:
            hwnd = int(f.read().strip())
        return hwnd
    except Exception:
        return find_window()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    cdir = os.path.abspath(__file__)
    cdir = os.path.dirname(cdir)
    cdir = os.path.dirname(cdir)
    cdir = os.path.dirname(cdir)
    p = argparse.ArgumentParser(description="Open Zarr File")
    p.add_argument("-f", "--folder", help="Zarr folder path", type=str, required=False)
    args = p.parse_args()
    if args.folder:
        folder = args.folder
    else:
        folder = QFileDialog.getExistingDirectory(None, "Select Zarr Folder", cdir)
    if folder == "":
        sys.exit(0)
    hwnd = get_hwnd()
    win = SliceBrowser(folder, hwnd)
    win.show()
    sys.exit(app.exec())
