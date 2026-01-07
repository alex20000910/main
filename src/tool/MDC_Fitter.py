from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider, QLabel, QStatusBar,
    QSpinBox, QPushButton, QHBoxLayout, QLineEdit, QMenuBar, QAction, QRadioButton,
    QButtonGroup, QFileDialog, QProgressBar, QDialog, QTextEdit, QComboBox, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QPainter, QFont, QColor, QIcon, QCursor, QFontMetrics
import pyqtgraph as pg

import os, inspect, time, sys, argparse
from base64 import b64decode
from ctypes import windll
import copy, tqdm

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
from scipy.interpolate import griddata
from lmfit import Parameters, Minimizer
from lmfit.printfuncs import alphanumeric_sort, gformat, report_fit

cdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
if os.name == 'nt':
    cdir = cdir[0].upper() + cdir[1:]
app_name = os.path.basename(inspect.stack()[0].filename).removesuffix('.py')
mod_dir = '.MDC_cut'
for i in range(5):
    cdir = os.path.dirname(cdir)
    if mod_dir in os.listdir(cdir):
        break
sys.path.append(os.path.join(cdir, mod_dir))

from MDC_cut_utility import *
from tool.loader import loadfiles
from tool.util import IconManager

m = 9.110938356e-31  # electron mass
e = 1.602176634e-19  # elementary charge
h = 6.62607015e-34  # Planck's constant
hbar = h / (2 * np.pi)  # reduced Planck's constant

def get_hwnd():
    try:
        with open('hwnd', 'r') as f:
            hwnd = int(f.read().strip())
        return hwnd
    except Exception:
        return find_window()

def putfitpar(inpars, modelpars=None, show_correl=True, min_correl=0.1,
              sort_pars=False, correl_mode='list'):
    if isinstance(inpars, Parameters):
        result, params = None, inpars
    if hasattr(inpars, 'params'):
        result = inpars
        params = inpars.params

    if sort_pars:
        if callable(sort_pars):
            key = sort_pars
        else:
            key = alphanumeric_sort
        parnames = sorted(params, key=key)
    else:
        parnames = list(params.keys())

    buff = []
    add = buff.append
    namelen = max(len(n) for n in parnames)
    for name in parnames:
        if name != 'y1' and name != 'y2':
            par = params[name]
            space = ' '*(namelen-len(name))
            nout = f"{name}:{space}"
            inval = '(init = ?)'
            if par.init_value is not None:
                inval = f'(init = {par.init_value:.7g})'
            if modelpars is not None and name in modelpars:
                inval = f'{inval}, model_value = {modelpars[name].value:.7g}'
            try:
                sval = gformat(par.value)
            except (TypeError, ValueError):
                sval = ' Non Numeric Value?'
            if par.stderr is not None:
                serr = gformat(par.stderr)
                try:
                    spercent = f'({abs(par.stderr/par.value):.2%})'
                except ZeroDivisionError:
                    spercent = ''
                sval = f'{sval} +/-{serr} {spercent}'

            if par.vary:
                add(f"    {nout} {sval} {inval}")
            elif par.expr is not None:
                add(f"    {nout} {sval} == '{par.expr}'")
            else:
                add(f"    {nout} {par.value: .7g} (fixed)")
    return buff


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
            
class oklim():
    def __init__(self, npzf, ev, phi):
        if npzf:
            avg = np.mean(phi)
            l = max(phi) - min(phi)
            self.min = np.asarray([avg - l/40 for i in ev], dtype=float)
            self.max = np.asarray([avg + l/40 for i in ev], dtype=float)
        else:
            self.min = np.asarray((2*m*ev*1.602176634*10**-19)**0.5*np.sin(-0.5/180*np.pi)*10**-10/(h/2/np.pi), dtype=float)
            self.max = np.asarray((2*m*ev*1.602176634*10**-19)**0.5*np.sin(0.5/180*np.pi)*10**-10/(h/2/np.pi), dtype=float)

class mfit_data():
    def __init__(self):
        try:
            with np.load(os.path.join(cdir,'.MDC_cut', 'mfit.npz'), 'rb') as f:
                ko = str(f['ko'])
                fev, rpos, ophi, fwhm, pos = f['fev'], f['rpos'], f['ophi'], f['fwhm'], f['pos']
                kmin, kmax, skmin, skmax = f['kmin'], f['kmax'], f['skmin'], f['skmax']
                smaa1, smaa2, smfp, smfi = f['smaa1'], f['smaa2'], f['smfp'], f['smfi']
                smresult, smcst = [], []
                try:
                    mdet = f['mdet']
                except:
                    mdet = -1
                try:
                    smresult, smcst = f['smresult'], f['smcst']
                except:
                    pass
            self.status = 'prloaded'
            self.fpr = 1
        except:
            ko = '0'
            fev, rpos, ophi, fwhm, pos = [], [], [], [], []
            kmin, kmax, skmin, skmax = [], [], [], []
            smaa1, smaa2, smfp, smfi = [], [], [], []
            smresult, smcst = [], []
            mdet = -1
            self.status = 'no'
            self.fpr = 0
        self.ko = ko
        self.fev, self.rpos, self.ophi, self.fwhm, self.pos = fev, rpos, ophi, fwhm, pos
        self.kmin, self.kmax, self.skmin, self.skmax = kmin, kmax, skmin, skmax
        self.smaa1, self.smaa2, self.smfp, self.smfi = smaa1, smaa2, smfp, smfi
        self.smresult, self.smcst = smresult, smcst
        self.mdet = mdet
    def get(self):
        return self.ko, self.fev, self.rpos, self.ophi, self.fwhm, self.pos, self.kmin, self.kmax, self.skmin, self.skmax, self.smaa1, self.smaa2, self.smfp, self.smfi, self.smresult, self.smcst, self.fpr, self.mdet

class main(QMainWindow):
    def __init__(self, file, hwnd=None):
        self.lfs = loadfiles(file, name='external')
        self.mdata = mfit_data()    # pos 改為 mpos
        self.ko, self.fev, self.rpos, self.ophi, self.fwhm, self.mpos, self.kmin, self.kmax, self.skmin, self.skmax, self.smaa1, self.smaa2, self.smfp, self.smfi, self.smresult, self.smcst, self.fpr, self.mdet = self.mdata.get()
        self.smresult_original = copy.deepcopy(self.smresult)
        self.init_data()
        # 以上初始化皆待修
                
        icon = IconManager().icon
        pixmap = QPixmap()
        pixmap.loadFromData(b64decode(icon))
        qicon = QIcon(pixmap)
        self.icon = qicon
        super().__init__()
        self.hwnd=hwnd
        self.setWindowTitle("MDC Fitter")
        self.setAcceptDrops(True)
        self.resizeEvent = lambda event: self.update_indicator()
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
        """)
        
        self.setWindowIcon(qicon)
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        self.menu_bar = QMenuBar()
        self.setMenuBar(self.menu_bar)
        file_menu = self.menu_bar.addMenu("File")
        act_load = QAction("Load File", self)
        act_load.setShortcut("Ctrl+O")
        act_load.triggered.connect(self.load_file)
        file_menu.addAction(act_load)
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
        self.cmap_menu = view_menu.addMenu("Colormap")
        self.set_default_colormap()
        for cmap_name in self.cmap_colors_dict.keys():
            act_cmap = QAction(f"{cmap_name}", self)
            act_cmap.setCheckable(True)
            act_cmap.setChecked(cmap_name=='prevac_cmap')
            act_cmap.triggered.connect(lambda checked, name=cmap_name: self.set_cmap(name))
            self.cmap_menu.addAction(act_cmap)

        # Layouts
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        left_layout = QVBoxLayout()
        self.left_layout(left_layout)
        main_layout.addLayout(left_layout, stretch=1)
        
        mid_layout = QVBoxLayout()
        self.mid_layout(mid_layout)
        main_layout.addLayout(mid_layout, stretch=3)
        
        right_layout = QVBoxLayout()
        self.right_layout(right_layout)
        main_layout.addLayout(right_layout, stretch=2)
        
        self.update_plot_raw()
        
        

        # 狀態列
        self.statusbar = QStatusBar(self)
        self.setStatusBar(self.statusbar)
        # right_label = QLabel(f"{self.mode.capitalize()} Mode")
        right_label = QLabel("MDC Fitter")
        right_label.setStyleSheet("background-color: #D7D7D7; color: #000; font-weight: bold; font-size: 30px;")
        # right_label.setFocusPolicy(Qt.NoFocus)
        # right_label.setTextInteractionFlags(Qt.NoTextInteraction)
        self.statusbar.addPermanentWidget(right_label)
        
        self.showMaximized()
        
        self.fitm()
        #### mjob ####
        self.mresult = [[]for i in range(len(self.eV))]
        try:
            flsmresult = self.smresult
            flsmcst = self.smcst
        except:
            self.smcst=np.zeros(len(self.eV)*6).reshape(len(self.eV),6)
            self.smresult = [1]
        if self.mprfit == 1:
            self.fmfall()
        else:
            self.mfitplot()
        #### mjob ####
            
        self.keyPressEvent = self.on_key_press
    
    def init_data(self):
        self.data = self.lfs.get(0)
        self.kdata = self.data.data
        self.phi, self.eV = self.data.phi.values, self.data.eV.values
        self.xlim = [self.phi.min(), self.phi.max()]
        self.ylim = [self.eV.min(), self.eV.max()]
        self.k_offset = 0   #維持原始 不做偏移
        self.vfe = 21.2     #維持原始 不做偏移
        if not self.lfs.f_npz[0]:
            QMessageBox.warning(None, "Warning", "Please be noted that the raw data is in Energy-Angle space.\nThe k-range might be inaccurate.")
            self.kdata, self.xlim, self.ylim = self.warp_data(self.data.data, self.phi, self.eV)
        klim = oklim(self.lfs.f_npz[0], self.eV, self.phi)
        self.klim = klim
        shape=self.data.shape
        det=self.data.data[shape[0]//2, shape[1]//2]
        if self.mdet != det:
            self.fpr = 0
        self.scki = []
        if self.fpr == 1:
            try:
                self.kmin, self.kmax = self.skmin, self.skmax
            except NameError:
                self.kmin, self.kmax = klim.min.copy(), klim.max.copy()
            if len(self.scki) >= 2:
                self.cki = self.scki
        else:
            self.kmin, self.kmax = klim.min.copy(), klim.max.copy()
        
        self.mfi, self.mfi_err, self.mfi_x = [], [], []
        for i in range(len(self.eV)):
            if i not in self.mfi:
                self.mfi_x.append(i)
        self.mmof = 1   # clicked flag for mouse move
        
        self.fdo=0      # undo redo flag
        self.flmcgl2 = -1   # add 2 peaks flag
        self.mfp = [1 for i in range(len(self.eV))]
        try:
            if self.fpr == 1:
                self.mfp = list(self.smfp)
                self.mfi = list(self.smfi)
        except:
            pass
        self.mbgv = 0 # 基線微調
        self.mundo_stack = []
        self.mredo_stack = []
        self.mbase = [0 for i in range(len(self.eV))]
    
    def load_file(self):
        file = QFileDialog.getOpenFileName(self, "Open Data File", "", "HDF5 Files (*.h5 *.hdf5);;NPZ Files (*.npz);;JSON Files (*.json);;TXT Files (*.txt)")[0]
        if file:
            self.lfs = loadfiles(file, name='external')
            self.mdata = mfit_data()    # pos 改為 mpos
            self.ko, self.fev, self.rpos, self.ophi, self.fwhm, self.mpos, self.kmin, self.kmax, self.skmin, self.skmax, self.smaa1, self.smaa2, self.smfp, self.smfi, self.smresult, self.smcst, self.fpr, self.mdet = self.mdata.get()
            self.init_data()
            self.slider.setValue(0)
            self.fitm()
            #### mjob ####
            self.mresult = [[]for i in range(len(self.eV))]
            try:
                flsmresult = self.smresult
                flsmcst = self.smcst
            except:
                self.smcst=np.zeros(len(self.eV)*6).reshape(len(self.eV),6)
                self.smresult = [1]
            if self.mprfit == 1:
                self.fmfall()
            else:
                self.mfitplot()
            #### mjob ####
    
    def on_key_press(self, event):
        if event.key() == Qt.Key_Z and (event.modifiers() & Qt.ControlModifier):
            self.undo()
        elif event.key() == Qt.Key_Y and (event.modifiers() & Qt.ControlModifier):
            self.redo()
        elif event.key() == Qt.Key_Left:
            self.mflind()
        elif event.key() == Qt.Key_Right:
            self.mfrind()
        elif event.key() == Qt.Key_Up:
            self.mfbgu()
        elif event.key() == Qt.Key_Down:
            self.mfbgd()
        elif event.key() == Qt.Key_Enter or event.key() == Qt.Key_Return:
            self.mfit()
            self.mfitplot()
        
    def left_layout(self, left_layout):
        self.plot_pos = pg.PlotWidget()
        self.plot_pos.setLabel('bottom', 'k (2π/Å)')
        self.plot_pos.getAxis('bottom').setStyle(tickFont=pg.QtGui.QFont("Arial", 12))
        self.plot_pos.setLabel('left', 'Kinetic Energy (eV)')
        self.plot_pos.getAxis('left').setStyle(tickFont=pg.QtGui.QFont("Arial", 12))
        left_layout.addWidget(self.plot_pos, stretch=2)
        
        self.plot_fwhm1 = pg.PlotWidget()
        self.plot_fwhm1.setLabel('bottom', 'Binding Energy (eV)')
        self.plot_fwhm1.getAxis('bottom').setStyle(tickFont=pg.QtGui.QFont("Arial", 12))
        self.plot_fwhm1.invertX(True)
        self.plot_fwhm1.setLabel('left', 'FWHM (2π/Å)')
        self.plot_fwhm1.getAxis('left').setStyle(tickFont=pg.QtGui.QFont("Arial", 12))
        self.plot_fwhm1.setLimits(yMin=-10, yMax=10)
        left_layout.addWidget(self.plot_fwhm1, stretch=1)
        
        self.plot_fwhm2 = pg.PlotWidget()
        self.plot_fwhm2.setLabel('bottom', 'Binding Energy (eV)')
        self.plot_fwhm2.getAxis('bottom').setStyle(tickFont=pg.QtGui.QFont("Arial", 12))
        self.plot_fwhm2.invertX(True)
        self.plot_fwhm2.setLabel('left', 'FWHM (2π/Å)')
        self.plot_fwhm2.getAxis('left').setStyle(tickFont=pg.QtGui.QFont("Arial", 12))
        self.plot_fwhm2.setLimits(yMin=-10, yMax=10)
        left_layout.addWidget(self.plot_fwhm2, stretch=1)
        
        self.plot_raw = pg.PlotWidget()
        self.k_offset = 0
        self.hist = pg.HistogramLUTWidget()
        self.set_cmap()
        self.plot_raw.setLabel('bottom', 'k (2π/Å)')
        self.plot_raw.getAxis('bottom').setStyle(tickFont=pg.QtGui.QFont("Arial", 12))
        self.plot_raw.setLabel('left', 'Kinetic Energy (eV)')
        self.plot_raw.getAxis('left').setStyle(tickFont=pg.QtGui.QFont("Arial", 12))
        left_layout.addWidget(self.plot_raw, stretch=2)
    
    
    def mprend(self, p=0):
        # global rpos, pos, fwhm, fev, medxdata, medydata, medfitout, skmin, skmax, smaa1, smaa2, smfp, smfi, fpr, scki
        fev, pos, fwhm = [], [], []
        self.skmin, self.skmax, self.smaa1, self.smaa2 = self.kmin, self.kmax, self.maa1, self.maa2
        self.smfp = self.mfp
        smfi = self.mfi
        for i, v in enumerate(self.mfi):
            if self.mfp[v] == 1:
                fev.append(self.eV[v])
                pos.append(self.maa1[v, 0])
                fwhm.append(self.maa1[v, 2])
            elif self.mfp[v] == 2:
                if p == 1:
                    fev.append(self.eV[v])
                    pos.append(self.maa2[v, 0])
                    fwhm.append(self.maa2[v, 2])
                elif p == 2:
                    fev.append(self.eV[v])
                    pos.append(self.maa2[v, 4])
                    fwhm.append(self.maa2[v, 6])
                else:
                    fev.append(self.eV[v])
                    fev.append(self.eV[v])
                    pos.append(self.maa2[v, 0])
                    pos.append(self.maa2[v, 4])
                    fwhm.append(self.maa2[v, 2])
                    fwhm.append(self.maa2[v, 6])
                
        fwhm = res(fev, fwhm)
        pos = res(fev, pos)
        # skmin = res(smfi, skmin)
        # skmax = res(smfi, skmax)
        # smfp = res(smfi, smfp)
        fev = res(fev, fev)
        self.smfi = res(smfi, smfi)
                
        rpos, fev, pos, fwhm = np.asarray(pos, dtype=float), np.asarray(fev, dtype=float), np.asarray(pos, dtype=float), np.asarray(fwhm, dtype=float)
        self.rpos, self.fev, self.mpos, self.fwhm = rpos, fev, pos, fwhm
    
    def update_plot_pos(self):
        self.mprend()
        self.plot_pos.clear()
        self.plot_pos.plot(self.mpos, self.fev, pen=None, symbolPen=None, symbol='o', symbolSize=10, symbolBrush='w')
        self.plot_pos.plot(self.mpos - self.fwhm / 2, self.fev, pen=None, symbolPen=None, symbol='|', symbolSize=10, symbolBrush='r')
        self.plot_pos.plot(self.mpos + self.fwhm / 2, self.fev, pen=None, symbolPen=None, symbol='|', symbolSize=10, symbolBrush='r')
        self.plot_pos.plot(self.plot_pos.getViewBox().viewRange()[0], [self.eV[self.index], self.eV[self.index]], pen=pg.mkPen(color='g', width=1, style=Qt.SolidLine))
    
    def update_plot_fwhm(self):
        x1=[]
        x2=[]
        y1=[]
        y2=[]
        for i, v in enumerate(self.mfi):
            if self.mfp[v] == 1:
                x1.append(self.vfe-self.eV[v])
                y1.append(self.maa1[v, 2])
            elif self.mfp[v] == 2:
                x1.append(self.vfe-self.eV[v])
                x2.append(self.vfe-self.eV[v])
                y1.append(self.maa2[v, 2])
                y2.append(self.maa2[v, 6])
        y1 = res(x1, y1)
        y2 = res(x2, y2)
        x1 = res(x1, x1)
        x2 = res(x2, x2)
        self.plot_fwhm1.clear()
        self.plot_fwhm2.clear()
        self.plot_fwhm1.plot([self.vfe-self.eV[self.index], self.vfe-self.eV[self.index]], self.plot_fwhm1.getViewBox().viewRange()[1], pen=pg.mkPen(color='g', width=1, style=Qt.SolidLine))
        self.plot_fwhm2.plot([self.vfe-self.eV[self.index], self.vfe-self.eV[self.index]], self.plot_fwhm2.getViewBox().viewRange()[1], pen=pg.mkPen(color='r', width=1, style=Qt.SolidLine))
        self.plot_fwhm1.plot(x1, y1, symbol='o', symbolBrush='w', symbolPen=None, symbolSize=5, pen=pg.mkPen(color='g', width=2, style=Qt.SolidLine), name='Comp 1')    #plot
        self.plot_fwhm2.plot(x2, y2, symbol='o', symbolBrush='w', symbolPen=None, symbolSize=5, pen=pg.mkPen(color='r', width=2, style=Qt.SolidLine), name='Comp 2')    #plot
        legend1 = self.plot_fwhm1.addLegend(
                offset=(10, 10),                    # 距離右上角的偏移
                brush=pg.mkBrush(255, 255, 255, 50),    # 背景: 半透明黑色
                pen=pg.mkPen('w', width=2),         # 邊框: 白色, 2像素寬
                labelTextColor='w',                 # 文字顏色: 白色
            )
        legend1.setLabelTextSize('16pt')
        legend1.setLabelTextColor('w')
        legend2 = self.plot_fwhm2.addLegend(
                offset=(10, 10),                    # 距離右上角的偏移
                brush=pg.mkBrush(255, 255, 255, 50),    # 背景: 半透明黑色
                pen=pg.mkPen('w', width=2),         # 邊框: 白色, 2像素寬
                labelTextColor='w',                 # 文字顏色: 白色
            )
        legend2.setLabelTextSize('16pt')
        legend2.setLabelTextColor('w')
    
    def toggle_grid(self, checked):
        if checked:
            self.plot.showGrid(x=True, y=True)
        else:
            self.plot.showGrid(x=False, y=False)
        
    def setup_plot_raw_menu(self):
        """設定 plot_raw 的右鍵選單"""
        # 獲取 plot_raw 的右鍵選單
        view_box = self.plot_raw.getViewBox()
        menu = view_box.menu
        menu.clear()
        
        # 添加 Histogram LUT 選單
        hist_menu = menu.addMenu("Histogram LUT")
        
        # 顯示/隱藏 Histogram
        act_show_hist = QAction("Show/Hide Histogram", self)
        act_show_hist.triggered.connect(self.toggle_histogram)
        hist_menu.addAction(act_show_hist)
        
        hist_menu.addSeparator()
        
        # 重置和自動層級
        act_reset = QAction("Reset Levels", self)
        act_reset.triggered.connect(self.reset_histogram)
        hist_menu.addAction(act_reset)
        
        act_auto = QAction("Auto Levels", self)
        act_auto.triggered.connect(self.auto_level_histogram)
        hist_menu.addAction(act_auto)
        
        hist_menu.addSeparator()
        
        # 添加 colormap 選單
        self._cmap_menu = hist_menu.addMenu("Colormap")
        for cmap_name in self.cmap_colors_dict.keys():
            act_cmap = QAction(f"{cmap_name}", self)
            act_cmap.setCheckable(True)
            act_cmap.setChecked(cmap_name=='prevac_cmap')
            act_cmap.triggered.connect(lambda checked, name=cmap_name: self.set_cmap(name))
            self._cmap_menu.addAction(act_cmap)

    def toggle_histogram(self):
        """顯示/隱藏 Histogram LUT"""
        if not hasattr(self, 'hist_widget_container'):
            # 第一次顯示時創建容器
            self.hist_widget_container = QWidget()
            hist_layout = QVBoxLayout()
            hist_layout.addWidget(self.hist)
            self.hist_widget_container.setLayout(hist_layout)
            self.hist_widget_container.setWindowTitle("Histogram LUT")
            self.hist_widget_container.setWindowIcon(self.icon)
            self.hist_widget_container.resize(300, 400)
        
        if self.hist_widget_container.isVisible():
            self.hist_widget_container.hide()
        else:
            self.hist_widget_container.show()

    def reset_histogram(self):
        """重置 Histogram"""
        if hasattr(self, 'hist') and self.hist.item is not None:
            self.hist.setLevels(self.data.data.min(), self.data.data.max())
            self.statusbar.showMessage("Histogram levels reset", 2000)

    def auto_level_histogram(self):
        """自動調整 Histogram 層級"""
        if hasattr(self, 'hist') and self.hist.item is not None:
            self.hist.autoHistogramRange()
            self.statusbar.showMessage("Auto levels applied", 2000)
            
    def warp_data(self, data, phi, ev):
        k_offset = self.k_offset
        # 建立目標座標網格
        x_target, y_target = np.meshgrid(phi, ev)
        x_target = np.sqrt(2*m*e*y_target/hbar**2) * np.sin((x_target + k_offset)/180*np.pi) * 1e-10

        # 定義輸出圖像的尺寸和範圍
        h_out, w_out = 500, 500
        x_min, x_max = x_target.min(), x_target.max()
        y_min, y_max = y_target.min(), y_target.max()

        # 建立映射矩陣 (從輸出座標映射回原始索引)
        map_x = np.zeros((h_out, w_out), dtype=np.float32)
        map_y = np.zeros((h_out, w_out), dtype=np.float32)

        j_indices = np.arange(w_out)
        i_indices = np.arange(h_out)
        j_grid, i_grid = np.meshgrid(j_indices, i_indices)

        # 輸出空間的 x, y 座標 (向量化)
        x_out = x_min + (x_max - x_min) * j_grid / (w_out - 1)
        y_out = y_min + (y_max - y_min) * i_grid / (h_out - 1)

        # 反向計算原始 phi 和 ev (向量化)
        ev_val = y_out
        k_val = x_out / 1e-10
        phi_val = np.arcsin(k_val / np.sqrt(2*m*e*ev_val/hbar**2)) * 180/np.pi - k_offset

        # 轉換為陣列索引
        map_x = ((phi_val - phi.min()) / (phi.max() - phi.min()) * (len(phi) - 1)).astype(np.float32)
        map_y = ((ev_val - ev.min()) / (ev.max() - ev.min()) * (len(ev) - 1)).astype(np.float32)
        # 使用 cv2.remap 進行轉換
        z_remapped = cv2.remap(data.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR)

        # plt.imshow(z_remapped, extent=[x_min, x_max, y_min, y_max], aspect='auto', origin='lower')
        xlim = [x_min, x_max]
        ylim = [y_min, y_max]
        return z_remapped, xlim, ylim
    
    def update_plot_raw(self):
        self.plot_raw.clear()
        img = pg.ImageItem(self.kdata.T)
        self.hist.setImageItem(img)
        x, y = self.xlim, self.ylim
        dx, dy = x[-1]-x[0], y[-1]-y[0]
        self.plot_raw.setLimits(xMin=x[0]-dx/10, xMax=x[-1]+dx/10, yMin=y[0]-dy/10, yMax=y[-1]+dy/10)
        self.plot_raw.setRange(xRange=(self.xlim[0], self.xlim[-1]), yRange=(self.ylim[0], self.ylim[-1]), padding=0)
        rect = pg.QtCore.QRectF(x[0], y[0], x[-1] - x[0], y[-1] - y[0])  # 真實位置
        img.setRect(rect)
        self.plot_raw.addItem(img)
        
        xlim = self.plot.getViewBox().viewRange()[0]
        self.xlabel.setContentsMargins(int(self.plot.getAxis('left').width()+self.ylabel.pixmap().width()), 0, 0, 0)
        ylim = self.eV[self.index]
        self.plot_raw.plot(xlim, [ylim, ylim], pen=pg.mkPen(color='r', width=1, style=Qt.SolidLine))
        self.plot_raw.plot([xlim[0], xlim[0]], [ylim-dy/40, ylim+dy/40], pen=pg.mkPen(color='r', width=1, style=Qt.SolidLine))
        self.plot_raw.plot([xlim[1], xlim[1]], [ylim-dy/40, ylim+dy/40], pen=pg.mkPen(color='r', width=1, style=Qt.SolidLine))
        self.setup_plot_raw_menu()
        
    def mid_layout(self, mid_layout):
        
        self.indicator_layout = QVBoxLayout()
        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setStyleSheet("""
            QSlider {
                background: #222;
                height: 40px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #999;
                background: #333;
                height: 40px;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #DDD;
                border: 1px solid #555;
                width: 40px;
                margin: -2px 0;
                border-radius: 10px;
            }
            QSlider::handle:horizontal:hover {
                background: #FFF;
                border: 1px solid #222;
            }
        """)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.eV)-1)
        self.index = 0
        self.slider.valueChanged.connect(self.on_slider_value_changed)
        
        self.indicator_layout.addWidget(self.slider)
        
        self.ind_plot = pg.PlotWidget()
        self.ind_plot.setFixedHeight(50)
        # self.ind_plot.setMenuEnabled(False)  # 禁用右鍵選單
        # self.ind_plot.getViewBox().setMouseEnabled(x=False, y=False)
        self.ind_plot.getAxis('bottom').hide()
        self.ind_plot.getAxis('left').hide()
        
        # self.ind_plot.resizeEvent = lambda event: self.update_indicator()
        
        self.indicator_layout.addWidget(self.ind_plot)
        
        
        self.b_left = QPushButton("<<")
        self.b_left.clicked.connect(self.mflind)
        self.b_left.setFixedHeight(90)
        self.b_right = QPushButton(">>")
        self.b_right.clicked.connect(self.mfrind)
        self.b_right.setFixedHeight(90)
        hbox = QHBoxLayout()
        hbox.addWidget(self.b_left)
        hbox.addLayout(self.indicator_layout)
        hbox.addWidget(self.b_right)
        
        mid_layout.addLayout(hbox)
        
        
        plot_layout = QHBoxLayout()
        self.plot = pg.PlotWidget()
        
        self.KE_pixmap = self.make_axis_label("Kinetic Energy (eV)", font_size=18, vertical=False)
        self.KE_v_pixmap = self.make_axis_label("Kinetic Energy (eV)", font_size=18, vertical=True)
        self.BE_pixmap = self.make_axis_label("Binding Energy (eV)", font_size=18, vertical=False)
        self.mm_pixmap = self.make_axis_label("Position (mm)", font_size=18, vertical=False)
        self.k_pixmap = self.make_axis_label("k (2π/Å)", font_size=18, vertical=False)
        self.deg_pixmap = self.make_axis_label("Angle (deg)", font_size=18, vertical=False)
        self.intensity_pixmap = self.make_axis_label("Intensity (Counts)", font_size=18, vertical=True)
        self.fwhm_pixmap = self.make_axis_label("FWHM (2π/Å)", font_size=18, vertical=True)
        self.BE_m_pixmap = self.make_axis_label("Binding Energy (meV)", font_size=18, vertical=False)
        self.v_pixmap = self.make_axis_label("v (eV/Å)", font_size=18, vertical=True)
        self.im_pixmap = self.make_axis_label("Im Σ (meV)", font_size=18, vertical=True)
        
        self.xlabel = QLabel()
        self.ylabel = QLabel()
        self.xlabel.setPixmap(self.k_pixmap)
        self.ylabel.setPixmap(self.intensity_pixmap)
        self.xlabel.setContentsMargins(int(self.plot.getAxis('left').width()+self.ylabel.pixmap().width()), 0, 0, 0)
        self.xlabel.setAlignment(Qt.AlignCenter)
        self.ylabel.setAlignment(Qt.AlignCenter)
        # self.plot.setMinimumHeight(600)
        # self.plot.setMenuEnabled(False)  # 禁用右鍵選單
        # self.plot.setMouseEnabled(x=False, y=False)
        self.plot.setTitle("Kinetic Energy", color='#ffffff', size='18pt', family='Arial')
        self.plot.setLabel('bottom', '')
        self.plot.getAxis('bottom').setStyle(tickFont=pg.QtGui.QFont("Arial", 18))
        self.plot.setLabel('left', '')
        self.plot.getAxis('left').setStyle(tickFont=pg.QtGui.QFont("Arial", 18))
        self.plot.scene().sigMouseMoved.connect(self.mouse_moved_event)
        self.plot.sigRangeChanged.connect(self.update_plot_raw)
        self.plot.scene().sigMouseClicked.connect(self.mouse_clicked_event)
        self.move_flag = False
        
        self.flmreject = -1
        self.flmposcst = -1
        self.flmrmv = -1
        self.flmcomp1 = -1
        self.flmcomp2 = -1
        self.fit_warn = 0
        
        plot_layout.addWidget(self.ylabel)
        plot_layout.addWidget(self.plot)
        mid_layout.addLayout(plot_layout)
        mid_layout.addWidget(self.xlabel)
        # self.update_plot()
        
        b_fit_all = QPushButton("Fit All")
        b_fit_all.clicked.connect(self.fmfall)
        b_fit_all.setFixedWidth(300)
        mid_layout.addWidget(b_fit_all, alignment=Qt.AlignCenter)
        
        b_pr = QPushButton("Preview")
        b_pr.clicked.connect(self.fmpreview)
        b_pr.setFixedWidth(300)
        mid_layout.addWidget(b_pr, alignment=Qt.AlignCenter)
        
        b_exp = QPushButton("Export All")
        b_exp.clicked.connect(self.fmend)
        b_exp.setFixedWidth(300)
        mid_layout.addWidget(b_exp, alignment=Qt.AlignCenter)
        
        exp_container = QWidget()
        exp_layout = QHBoxLayout()
        b_exp1 = QPushButton("Export Comp 1")
        b_exp1.clicked.connect(lambda: self.fmend(1))
        b_exp1.setFixedWidth(300)
        exp_layout.addWidget(b_exp1)
        b_exp2 = QPushButton("Export Comp 2")
        b_exp2.clicked.connect(lambda: self.fmend(2))
        b_exp2.setFixedWidth(300)
        exp_layout.addWidget(b_exp2)
        exp_container.setLayout(exp_layout)
        mid_layout.addWidget(exp_container, alignment=Qt.AlignCenter)
    
    def fmresidual(self):
        if hasattr(self, 'g_residual'):
            if self.g_residual is not None:
                self.g_residual.show()
                self.g_residual.raise_()  # 將視窗提升到最上層
                self.g_residual.activateWindow()
                return
        mfi, mfi_err = self.mfi, self.mfi_err
        fmx, fmy = self.fmx, self.fmy
        kmin, kmax = self.kmin, self.kmax
        maa2 = self.maa2
        ev = self.eV
        # plt.figure()
        s3,s4=[],[]
        for i in range(len(ev)):
            if i in mfi_err or i in mfi:
                # x = fmxx[i, np.argwhere(fmxx[i, :] >= -20)].flatten()
                # y = fmyy[i, np.argwhere(fmxx[i, :] >= -20)].flatten()
                # lbg=lnr_bg(fmyy[i, :len(x)])
                x, x_arg = filter(fmx[i, :], kmin[i], kmax[i])
                y = fmy[i, x_arg]
                lbg = self.lnr_bg(y)
                s3.append(np.std(gl2(x, *maa2[i, :])+lbg-y))  # STD
                s4.append(np.sqrt(np.mean((gl2(x, *maa2[i, :])+lbg-y)**2)))  # RMS
            else:
                s3.append(0)
                s4.append(0)
        self.g_residual = QDialog(self)
        self.g_residual.closeEvent = lambda event: self.close_residual(event)
        tl = QVBoxLayout()
        plot_layout = QHBoxLayout()
        plot = pg.PlotWidget()
        xlabel = QLabel()
        ylabel = QLabel()
        xlabel.setPixmap(self.KE_pixmap)
        ylabel.setPixmap(self.intensity_pixmap)
        xlabel.setContentsMargins(int(plot.getAxis('left').width()+ylabel.pixmap().width()), 0, 0, 0)
        xlabel.setAlignment(Qt.AlignCenter)
        ylabel.setAlignment(Qt.AlignCenter)
        legend = plot.addLegend(
            offset=(10, 10),                    # 距離右上角的偏移
            brush=pg.mkBrush(255, 255, 255, 50),    # 背景: 半透明黑色
            pen=pg.mkPen('w', width=2),         # 邊框: 白色, 2像素寬
            labelTextColor='w',                 # 文字顏色: 白色
        )
        legend.setLabelTextSize('16pt') 
        legend.setLabelTextColor('w')
        
        plot.plot(ev,s3,pen=pg.mkPen(color='r', width=2), name='STD')
        plot.plot(ev,s4,pen=pg.mkPen(color='g', width=2), name='RMS')
        plot.setTitle('Residual', color='#ffffff', size='18pt', family='Arial')
        plot.setLabel('bottom', '')
        plot.getAxis('bottom').setStyle(tickFont=pg.QtGui.QFont("Arial", 18))
        plot.getAxis('left').setStyle(tickFont=pg.QtGui.QFont("Arial", 18))
        plot.setLabel('left', '')
        plot_layout.addWidget(ylabel)
        plot_layout.addWidget(plot)
        tl.addLayout(plot_layout)
        tl.addWidget(xlabel)
        self.g_residual.setLayout(tl)
        self.g_residual.setWindowTitle("Preview")
        self.g_residual.setWindowIcon(self.icon)
        self.g_residual.show()
    
    def close_residual(self, event):
        self.g_residual.close()
        self.g_residual = None
        event.accept()

    def fmarea(self):
        if hasattr(self, 'g_area'):
            if self.g_area is not None:
                self.g_area.show()
                self.g_area.raise_()  # 將視窗提升到最上層
                self.g_area.activateWindow()
                return
        mfi, mfi_err = self.mfi, self.mfi_err
        fmx, fmy = self.fmx, self.fmy
        kmin, kmax = self.kmin, self.kmax
        maa2 = self.maa2
        ev = self.eV
        # plt.figure()
        s1,s2=[],[]
        for i in range(len(ev)):
            if i in mfi_err or i in mfi:
                # x = fmxx[i, np.argwhere(fmxx[i, :] >= -20)].flatten()
                # y = fmyy[i, np.argwhere(fmxx[i, :] >= -20)].flatten()
                x, x_arg = filter(fmx[i, :], kmin[i], kmax[i])
                ty = gl1(x, *maa2[i, :4])
                s1.append(np.sum(np.array([((ty[i]+ty[i+1])/2)for i in range(len(x)-1)])
                            # Area 1
                            * np.array(([(x[i+1]-x[i])for i in range(len(x)-1)]))))
                ty = gl1(x, *maa2[i, -4:])
                s2.append(np.sum(np.array([((ty[i]+ty[i+1])/2)for i in range(len(x)-1)])
                            # Area 2
                            * np.array(([(x[i+1]-x[i])for i in range(len(x)-1)]))))
            else:
                s1.append(0)
                s2.append(0)
        self.g_area = QDialog(self)
        self.g_area.closeEvent = lambda event: self.close_area(event)
        tl = QVBoxLayout()
        plot_layout = QHBoxLayout()
        plot = pg.PlotWidget()
        xlabel = QLabel()
        ylabel = QLabel()
        xlabel.setPixmap(self.KE_pixmap)
        ylabel.setPixmap(self.intensity_pixmap)
        xlabel.setContentsMargins(int(plot.getAxis('left').width()+ylabel.pixmap().width()), 0, 0, 0)
        xlabel.setAlignment(Qt.AlignCenter)
        ylabel.setAlignment(Qt.AlignCenter)
        legend = plot.addLegend(
            offset=(10, 10),                    # 距離右上角的偏移
            brush=pg.mkBrush(255, 255, 255, 50),    # 背景: 半透明黑色
            pen=pg.mkPen('w', width=2),         # 邊框: 白色, 2像素寬
            labelTextColor='w',                 # 文字顏色: 白色
        )
        legend.setLabelTextSize('16pt') 
        legend.setLabelTextColor('w')
        
        plot.plot(ev,s1,pen=pg.mkPen(color='r', width=2), name='Area 1')
        plot.plot(ev,s2,pen=pg.mkPen(color='g', width=2), name='Area 2')
        plot.setTitle('Area', color='#ffffff', size='18pt', family='Arial')
        plot.setLabel('bottom', '')
        plot.getAxis('bottom').setStyle(tickFont=pg.QtGui.QFont("Arial", 18))
        plot.getAxis('left').setStyle(tickFont=pg.QtGui.QFont("Arial", 18))
        plot.setLabel('left', '')
        plot_layout.addWidget(ylabel)
        plot_layout.addWidget(plot)
        tl.addLayout(plot_layout)
        tl.addWidget(xlabel)
        self.g_area.setLayout(tl)
        self.g_area.setWindowTitle("Area")
        self.g_area.setWindowIcon(self.icon)
        self.g_area.show()
    
    def close_area(self, event):
        self.g_area.close()
        self.g_area = None
        event.accept()
        

    def fmfwhm(self):
        if hasattr(self, 'g_fwhm'):
            if self.g_fwhm is not None:
                self.g_fwhm.show()
                self.g_fwhm.raise_()  # 將視窗提升到最上層
                self.g_fwhm.activateWindow()
                return
        mfi, mfp = self.mfi, self.mfp
        maa1 = self.maa1
        maa2 = self.maa2
        ev = self.eV
        # global pos, fwhm, fev, rpos, ophi
        fev, pos, fwhm = [], [], []
        x1=[]
        x2=[]
        y1=[]
        y2=[]
        for i, v in enumerate(mfi):
            if mfp[v] == 1:
                fev.append(ev[v])
                pos.append(maa1[v, 0])
                fwhm.append(maa1[v, 2])
                x1.append(ev[v])
                y1.append(maa1[v, 2])
            elif mfp[v] == 2:
                x1.append(ev[v])
                x2.append(ev[v])
                y1.append(maa2[v, 2])
                y2.append(maa2[v, 6])
                
                fev.append(ev[v])
                fev.append(ev[v])
                pos.append(maa2[v, 0])
                pos.append(maa2[v, 4])
                fwhm.append(maa2[v, 2])
                fwhm.append(maa2[v, 6])
        fev = np.asarray(fev, dtype=float)
        rpos = np.asarray(pos, dtype=float)
        
        ophi = np.arcsin(rpos/(2*m*fev*1.602176634*10**-19)**0.5/10**-10*(h/2/np.pi))*180/np.pi
        pos = (2*m*fev*1.602176634*10**-19)**0.5 * np.sin((np.asarray(self.k_offset, dtype=float)+ophi)/180*np.pi)*10**-10/(h/2/np.pi)

        rpos = res(fev, rpos)
        ophi = res(fev, ophi)
        fwhm = res(fev, fwhm)
        pos = res(fev, pos)
        fev = res(fev, fev)
        
        self.g_fwhm = QDialog(self)
        self.g_fwhm.closeEvent = lambda event: self.close_fwhm(event)
        tl = QVBoxLayout()
        
        plot1 = pg.PlotWidget()
        plot2 = pg.PlotWidget()
        plot3 = pg.PlotWidget()
        for i in [plot1, plot2, plot3]:
            plot_layout = QHBoxLayout()
            xlabel = QLabel()
            ylabel = QLabel()
            xlabel.setPixmap(self.KE_pixmap)
            ylabel.setPixmap(self.fwhm_pixmap)
            xlabel.setContentsMargins(int(i.getAxis('left').width()+ylabel.pixmap().width()), 0, 0, 0)
            xlabel.setAlignment(Qt.AlignCenter)
            ylabel.setAlignment(Qt.AlignCenter)
            legend = i.addLegend(
                offset=(10, 10),                    # 距離右上角的偏移
                brush=pg.mkBrush(255, 255, 255, 50),    # 背景: 半透明黑色
                pen=pg.mkPen('w', width=2),         # 邊框: 白色, 2像素寬
                labelTextColor='w',                 # 文字顏色: 白色
            )
            legend.setLabelTextSize('16pt')
            legend.setLabelTextColor('w')
            i.setLabel('bottom', '')
            i.getAxis('bottom').setStyle(tickFont=pg.QtGui.QFont("Arial", 18))
            i.getAxis('left').setStyle(tickFont=pg.QtGui.QFont("Arial", 18))
            i.setLabel('left', '')
        
            plot_layout.addWidget(ylabel)
            plot_layout.addWidget(i)
            tl.addLayout(plot_layout)
            tl.addWidget(xlabel)
        
        self.g_fwhm.setLayout(tl)
        self.g_fwhm.setWindowTitle("FWHM")
        self.g_fwhm.setWindowIcon(self.icon)
        
        plot1.plot(x1, y1, pen=None, symbolPen=None, symbol='o', symbolSize=10, symbolBrush='r', name='Comp 1')
        plot2.plot(x2, y2, pen=None, symbolPen=None, symbol='o', symbolSize=10, symbolBrush='b', name='Comp 2')
        plot3.plot(x1, y1, pen=None, symbolPen=None, symbol='o', symbolSize=10, symbolBrush='r', name='Comp 1')
        plot3.plot(x2, y2, pen=None, symbolPen=None, symbol='o', symbolSize=10, symbolBrush='b', name='Comp 2')
        
        self.g_fwhm.show()

    def close_fwhm(self, event):
        self.g_fwhm.close()
        self.g_fwhm = None
        event.accept()

    def fmimse(self):
        if hasattr(self, 'g_imse'):
            if self.g_imse is not None:
                self.g_imse.show()
                self.g_imse.raise_()  # 將視窗提升到最上層
                self.g_imse.activateWindow()
                return
        mfi, mfp = self.mfi, self.mfp
        maa1 = self.maa1
        maa2 = self.maa2
        ev = self.eV
        
        # global pos, fwhm, fev, rpos, ophi
        fev, pos, fwhm = [], [], []
        y=[]
        pos1=[]
        pos2=[]
        fwhm1=[]
        fwhm2=[]
        for i, v in enumerate(mfi):
            if mfp[v] == 1:
                fev.append(ev[v])
                pos.append(maa1[v, 0])
                fwhm.append(maa1[v, 2])
            elif mfp[v] == 2:
                y.append(ev[v])
                pos1.append(maa2[v, 0])
                pos2.append(maa2[v, 4])
                fwhm1.append(maa2[v, 2])
                fwhm2.append(maa2[v, 6])
                
                fev.append(ev[v])
                fev.append(ev[v])
                pos.append(maa2[v, 0])
                pos.append(maa2[v, 4])
                fwhm.append(maa2[v, 2])
                fwhm.append(maa2[v, 6])
        y = np.asarray(y, dtype=float)
        fev = np.asarray(fev, dtype=float)
        rpos = np.asarray(pos, dtype=float)
        
        ophi = np.arcsin(rpos/np.sqrt(2*m*fev*1.602176634*10**-19)/10**-10*(h/2/np.pi))*180/np.pi
        pos = (2*m*fev*1.602176634*10**-19)**0.5 * np.sin((np.asarray(self.k_offset, dtype=float)+ophi)/180*np.pi)*10**-10/(h/2/np.pi)
        
        rpos = res(fev, rpos)
        ophi = res(fev, ophi)
        fwhm = res(fev, fwhm)
        pos = res(fev, pos)
        fev = res(fev, fev)
        
        pos1 = res(y, pos1)
        pos2 = res(y, pos2)
        fwhm1 = res(y, fwhm1)
        fwhm2 = res(y, fwhm2)
        y = res(y, y)
        
        xx = np.diff(y)
        yy1 = np.diff(pos1)
        yy2 = np.diff(pos2)
        
        # eliminate infinite vf
        for i in range(len(yy1)):
            if xx[i]/yy1[i] > 20000:
                yy1[i] = 0
        for i in range(len(yy2)):
            if xx[i]/yy2[i] > 20000:
                yy2[i] = 0
        
        v1 = xx/yy1
        v2 = xx/yy2
        yy1 = v1*fwhm1[1::]/2
        yy2 = v2*fwhm2[1::]/2
        xx/=2
        print(len(y))
        print(len(xx))
        x = ((y[-1:0:-1]+xx[::-1])-self.vfe)*1000
        print(len(x))
        
        self.g_imse = QDialog(self)
        self.g_imse.closeEvent = lambda event: self.close_imse(event)
        l_main = QVBoxLayout()
        l_up = QHBoxLayout()
        l_down = QHBoxLayout()
        plot1_u = pg.PlotWidget()
        plot2_u = pg.PlotWidget()
        plot1_d = pg.PlotWidget()
        plot2_d = pg.PlotWidget()
        for i in [plot1_u, plot2_u, plot1_d, plot2_d]:
            plot_layout = QVBoxLayout()
            xlabel = QLabel()
            ylabel = QLabel()
            xlabel.setPixmap(self.BE_m_pixmap)
            if i in [plot1_u, plot2_u]:
                ylabel.setPixmap(self.v_pixmap)
            else:
                ylabel.setPixmap(self.im_pixmap)
            xlabel.setContentsMargins(int(i.getAxis('left').width()+ylabel.pixmap().width()), 0, 0, 0)
            xlabel.setAlignment(Qt.AlignCenter)
            ylabel.setAlignment(Qt.AlignCenter)
            legend = i.addLegend(
                offset=(10, 10),                    # 距離右上角的偏移
                brush=pg.mkBrush(255, 255, 255, 50),    # 背景: 半透明黑色
                pen=pg.mkPen('w', width=2),         # 邊框: 白色, 2像素寬
                labelTextColor='w',                 # 文字顏色: 白色
            )
            legend.setLabelTextSize('16pt')
            legend.setLabelTextColor('w')
            i.setLabel('bottom', '')
            i.getAxis('bottom').setStyle(tickFont=pg.QtGui.QFont("Arial", 18))
            i.getAxis('left').setStyle(tickFont=pg.QtGui.QFont("Arial", 18))
            i.setLabel('left', '')
        
            plot_layout.addWidget(i)
            plot_layout.addWidget(xlabel)
            if i in [plot1_u, plot2_u]:
                i.setTitle('Group Velocity', color='#ffffff', size='18pt', family='Arial')
                l_up.addWidget(ylabel)
                l_up.addLayout(plot_layout)
            else:
                i.setTitle('Imaginary Part', color='#ffffff', size='18pt', family='Arial')
                l_down.addWidget(ylabel)
                l_down.addLayout(plot_layout)
            if i in [plot2_u]:
                l_main.addLayout(l_up)
            elif i in [plot2_d]:
                l_main.addLayout(l_down)
        
        plot1_u.plot(x,v1, pen=None, symbolPen=None, symbol='o', symbolSize=10, symbolBrush='r', name='Comp 1')
        plot2_u.plot(x,v2, pen=None, symbolPen=None, symbol='o', symbolSize=10, symbolBrush='b', name='Comp 2')
        plot1_d.plot(x,yy1*1000, pen=None, symbolPen=None, symbol='o', symbolSize=10, symbolBrush='r', name='Comp 1')
        plot2_d.plot(x,yy2*1000, pen=None, symbolPen=None, symbol='o', symbolSize=10, symbolBrush='b', name='Comp 2')
        
        self.g_imse.setLayout(l_main)
        self.g_imse.setWindowTitle("Imaginary Part")
        self.g_imse.setWindowIcon(self.icon)
        self.pos, self.fwhm, self.fev, self.rpos, self.ophi = pos, fwhm, fev, rpos, ophi
        
        self.g_imse.show()
        
    def close_imse(self, event):
        self.g_imse.close()
        self.g_imse = None
        event.accept()

    
    def fmpreview(self):
        if hasattr(self, 'tg'):
            self.tg.show()
            self.tg.raise_()
            self.tg.activateWindow()
            return
        self.tg = QDialog(self)
        self.tg.setWindowTitle('Preview MDC Result')
        self.tg.setWindowIcon(self.icon)
        self.tg.resize(400, 300)
        tg_layout = QVBoxLayout()
        self.tg.setLayout(tg_layout)
        bmresidual = QPushButton('Residual')
        bmresidual.setFixedHeight(60)
        bmresidual.clicked.connect(self.fmresidual)
        tg_layout.addWidget(bmresidual)
        bmarea = QPushButton('Area')
        bmarea.setFixedHeight(60)
        bmarea.clicked.connect(self.fmarea)
        tg_layout.addWidget(bmarea)
        bmfwhm = QPushButton('FWHM')
        bmfwhm.setFixedHeight(60)
        bmfwhm.clicked.connect(self.fmfwhm)
        tg_layout.addWidget(bmfwhm)
        bmimse = QPushButton('Imaginary Part')
        bmimse.setFixedHeight(60)
        bmimse.clicked.connect(self.fmimse)
        tg_layout.addWidget(bmimse)
        self.tg.show()
    
    def update_indicator(self):
        self.ind_plot.clear()
        x=np.arange(len(self.eV))
        width = self.slider.geometry().width()
        ratio = 20/width
        dx = x.max() - x.min()
        self.ind_plot.setLimits(xMin=x.min()-dx/5, xMax=x.max()+dx/5, yMin=-1, yMax=1)
        self.ind_plot.setRange(xRange=(x.min(), x.max()), yRange=(-0.6, 0.6), padding=ratio)
        self.gen_indicator()
        
    def gen_indicator(self):
        mfi = []
        mfi_err = []
        mfi_x = []
        for i in range(len(self.eV)):
            if i in self.mfi:
                mfi.append(i)
            elif i in self.mfi_x:
                mfi_x.append(i)
            elif i in self.mfi_err:
                mfi_err.append(i)
        
        data = np.zeros((20, len(self.eV)))
        oc = ['b', 'k', 'r']
        bc = ['#0000FF', '#A0A0A0', '#FF0000']
        c = [1, 0, 2]
        f=False
        for i, v in enumerate([mfi, mfi_x, mfi_err]):
            x = np.array(v, dtype=int)
            y = np.zeros_like(x)-0.2
            data[10:, x] = c[i]
            if self.index in v:
                self.ind_plot.plot([self.index], [0.4], pen=None, symbolPen=None, symbol='|', symbolBrush=oc[i], symbolSize=20)
                self.b_left.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {bc[i]};
                        color: #000;
                        font-family: Arial;
                        font-weight: bold;
                    }}
                    QPushButton:hover {{
                        background-color: #555;
                        color: #FFD700;
                    }}
                """)
                self.b_right.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {bc[i]};
                        color: #000;
                        font-family: Arial;
                        font-weight: bold;
                    }}
                    QPushButton:hover {{
                        background-color: #555;
                        color: #FFD700;
                    }}
                """)
                if i!=1:f=True
        if not f:
            self.ind_plot.plot([self.index], [0.4], pen=None, symbolPen=None, symbol='|', symbolBrush='w', symbolSize=20)
        
        data[0, 0] = 3
        cmap = mpl.colors.ListedColormap(['#000000', '#0000FF', '#FF0000', '#000000'])
        lut = (cmap(np.arange(0, cmap.N)) * 255).astype(np.uint8)
        img = pg.ImageItem(data.T)
        img.setLookupTable(lut)
        img.setRect(pg.QtCore.QRectF(0, -1.1, len(self.eV), 1.2))
        self.ind_plot.addItem(img)
    
    
    def update_plot(self):
        x = self.phi
        if not self.lfs.f_npz[0]:
            x = np.sqrt(2*m*e*self.eV[self.index])/hbar * np.sin((self.phi+self.k_offset)/180*np.pi) * 1e-10
        y = self.data.data[self.index, :]
        self.plot.clear()
        self.plot.setTitle(f"Kinetic Energy: {self.eV[self.index]:.3f} eV, {str(self.mfp[self.index])} Peak", color='#ffffff', size='18pt', family='Arial')
        self.c0 = self.plot.plot(x, y, pen=None, symbolPen=None, symbol='o', symbolBrush='w')
        self.reg = pg.LinearRegionItem(values=[self.kmin[self.index], self.kmax[self.index]], brush=pg.mkBrush(0, 0, 255, 25), pen=pg.mkPen(color='y', width=8, style=Qt.DotLine))
        self.reg.setMovable(True)
        self.plot.addItem(self.reg)
        self.reg.sigRegionChanged.connect(self.reg_change)
        self.reg.sigRegionChangeFinished.connect(self.reg_set)
        
        # if self.index in self.mfi:
        #     self.c1 = self.plot.plot(x, y, pen=pg.mkPen(None))
        #     self.base=self.plot.plot(x, np.zeros_like(x), pen=pg.mkPen(color='g', width=2, style=Qt.SolidLine))
        #     fill = pg.FillBetweenItem(self.c1, self.base, brush=pg.mkBrush(100, 100, 255, 100))
        #     self.plot.addItem(fill)
        
        
        dx, dy = x[-1]-x[0], y.max()-y.min()
        if y.max() == y.min():
            self.plot.getViewBox().setLimits(xMin=x[0]-dx/10, xMax=x[-1]+dx/10, yMin=-2, yMax=-1)
        else:
            self.plot.getViewBox().setLimits(xMin=x[0]-dx/10, xMax=x[-1]+dx/10, yMin=y.min()-dy/5, yMax=y.max()+dy/2)
        self.plot.setRange(xRange=(x[0]-dx/20, x[-1]+dx/20), yRange=(y.min()-dy/10, y.max()+dy/10), padding=0)
        
        self.update_indicator()
        self.update_plot_pos()
        self.update_plot_fwhm()
    
    def reg_change(self):
        v = self.reg.getRegion()
        self.kmin[self.index] = v[0]
        self.kmax[self.index] = v[1]
        
    def reg_set(self):
        self.func_cki()
        self.mfit()
        self.mfitplot()
    
    
    def mflind(self):
        # global mfiti
        ti = self.index
        if ti in self.mfi:
            for i in range(ti+1):
                if ti-i not in self.mfi:
                    self.slider.setValue(ti-i)
                    break
        elif ti in self.mfi_err:
            for i in range(ti+1):
                if ti-i not in self.mfi_err:
                    self.slider.setValue(ti-i)
                    break
        elif ti in self.mfi_x:
            for i in range(ti+1):
                if ti-i in self.mfi or ti-i in self.mfi_err:
                    self.slider.setValue(ti-i)
                    break
            if i == ti and ti != 0:
                self.slider.setValue(ti-1)



    def mfrind(self):
        # global mfiti
        ti = self.index
        if ti in self.mfi:
            for i in range(len(self.eV)-ti):
                if ti+i not in self.mfi:
                    self.slider.setValue(ti+i)
                    break
        elif ti in self.mfi_err:
            for i in range(len(self.eV)-ti):
                if ti+i not in self.mfi_err:
                    self.slider.setValue(ti+i)
                    break
        elif ti in self.mfi_x:
            for i in range(len(self.eV)-ti):
                if ti+i in self.mfi or ti+i in self.mfi_err:
                    self.slider.setValue(ti+i)
                    break
            if i == len(self.eV)-ti-1 and ti != len(self.eV)-1:
                self.slider.setValue(ti+1)

    
    def mfbgu(self):
        # global mbgv
        i=self.index
        # mbase[i] = int(base.get())  # 待調整
        self.mbase[i] = 0  # 待調整
        # fmxx[i, :] = fmxx[i, :]/fmxx[i, :]*-50
        # fmyy[i, :] = fmyy[i, :]/fmyy[i, :]*-50
        ecut = self.data.sel(eV=self.eV[i], method='nearest')
        if self.lfs.f_npz[0]:x = self.phi
        else:x = (2*m*self.eV[i]*1.602176634*10**-19)**0.5*np.sin(self.phi/180*np.pi)*10**-10/(h/2/np.pi)
        y = ecut.to_numpy().reshape(len(x))
        xx, x_arg = filter(x, self.kmin[i], self.kmax[i])
        # tx = x[np.argwhere(x >= kmin[i])].flatten()
        # xx = tx[np.argwhere(tx <= kmax[i])].flatten()
        # ty = y[np.argwhere(x >= kmin[i])].flatten()
        # yy = ty[np.argwhere(tx <= kmax[i])].flatten()
        yy = y[x_arg]
        yy = np.where(yy > self.mbase[i], yy, self.mbase[i])
        d = sorted(abs(np.diff(np.append(yy[0:5],yy[-6:-1]))))
        t=0
        ti=0
        while t==0:
            t=d[ti]
            ti+=1
            if ti==len(d):
                break
        print(t)
        try:
            self.mbgv+=t/2
            self.mfit()
            self.mfitplot()
        except:
            pass


    def mfbgd(self):
        # global mbgv
        i=self.index
        # mbase[i] = int(base.get())  # 待調整
        self.mbase[i] = 0  # 待調整
        # fmxx[i, :] = fmxx[i, :]/fmxx[i, :]*-50
        # fmyy[i, :] = fmyy[i, :]/fmyy[i, :]*-50
        ecut = self.data.sel(eV=self.eV[i], method='nearest')
        if self.lfs.f_npz[0]:x = self.phi
        else:x = (2*m*self.eV[i]*1.602176634*10**-19)**0.5*np.sin(self.phi/180*np.pi)*10**-10/(h/2/np.pi)
        y = ecut.to_numpy().reshape(len(x))
        xx, x_arg = filter(x, self.kmin[i], self.kmax[i])
        # tx = x[np.argwhere(x >= kmin[i])].flatten()
        # xx = tx[np.argwhere(tx <= kmax[i])].flatten()
        # ty = y[np.argwhere(x >= kmin[i])].flatten()
        # yy = ty[np.argwhere(tx <= kmax[i])].flatten()
        yy = y[x_arg]
        yy = np.where(yy > self.mbase[i], yy, self.mbase[i])
        d = sorted(abs(np.diff(np.append(yy[0:5],yy[-6:-1]))))
        t=0
        ti=0
        while t==0:
            t=d[ti]
            ti+=1
            if ti==len(d):
                break
        print(t)
        try:
            self.mbgv-=t/2
            self.mfit()
            self.mfitplot()
        except:
            pass

    def func_cki(self):
        # global cki, kmin, kmax
        if self.index not in self.cki:
            self.cki.append(self.index)
        if len(self.cki) >= 2:
            self.cki.sort()
            for i in range(len(self.cki)-1):
                self.kmin[self.cki[i]:self.cki[i+1] +
                    1] = np.linspace(self.kmin[self.cki[i]], self.kmin[self.cki[i+1]], self.cki[i+1]-self.cki[i]+1)
                self.kmax[self.cki[i]:self.cki[i+1] +
                    1] = np.linspace(self.kmax[self.cki[i]], self.kmax[self.cki[i+1]], self.cki[i+1]-self.cki[i]+1)
    
    def close_exp(self, event):
        self.g_exp.close()
        self.g_exp = None
        event.accept()
    
    def fmend(self, p=0):
        # global rpos, pos, fwhm, fev, medxdata, medydata, medfitout, skmin, skmax, smaa1, smaa2, smfp, smfi, fpr, scki, mendg
        self.mprend(p)
        self.scki = self.cki
        self.fpr = 1
        if hasattr(self, 'g_exp'):
            if self.g_exp is not None:
                if p==self.p:
                    self.g_exp.show()
                    self.g_exp.raise_()  # 將視窗提升到最上層
                    self.g_exp.activateWindow()
                    return
                else:
                    self.g_exp.close()
        self.p = p
        self.g_exp = QDialog(self)
        self.g_exp.closeEvent = lambda event: self.close_exp(event)
        tl = QVBoxLayout()
        plot_layout = QHBoxLayout()
        plot = pg.PlotWidget()
        xlabel = QLabel()
        ylabel = QLabel()
        xlabel.setPixmap(self.k_pixmap)
        ylabel.setPixmap(self.KE_v_pixmap)
        xlabel.setContentsMargins(int(plot.getAxis('left').width()+ylabel.pixmap().width()), 0, 0, 0)
        xlabel.setAlignment(Qt.AlignCenter)
        ylabel.setAlignment(Qt.AlignCenter)
        
        plot.plot(self.mpos+self.fwhm/2, self.fev, pen=None, symbolPen=None, symbol='o', symbolSize=10, symbolBrush='r')
        plot.plot(self.mpos-self.fwhm/2, self.fev, pen=None, symbolPen=None, symbol='o', symbolSize=10, symbolBrush='r')
        plot.plot(self.mpos, self.fev, pen=None, symbolPen=None, symbol='o', symbolSize=10, symbolBrush='w')
        plot.setTitle('MDC Lorentz Fit Result', color='#ffffff', size='18pt', family='Arial')
        plot.setLabel('bottom', '')
        plot.getAxis('bottom').setStyle(tickFont=pg.QtGui.QFont("Arial", 18))
        plot.getAxis('left').setStyle(tickFont=pg.QtGui.QFont("Arial", 18))
        plot.setLabel('left', '')
        
        plot_layout.addWidget(ylabel)
        plot_layout.addWidget(plot)
        tl.addLayout(plot_layout)
        tl.addWidget(xlabel)
        
        b_save = QPushButton('Save Fitted Data')
        b_save.clicked.connect(self.savemfit)
        tl.addWidget(b_save)
        
        self.g_exp.setLayout(tl)
        self.g_exp.setWindowTitle("Preview")
        self.g_exp.setWindowIcon(self.icon)
        self.g_exp.show()

    def savemfit(self):
        # global smresult, smcst, fev, fwhm, pos, skmin, skmax, smaa1, smaa2, smfp, smfi, mdet
        self.smresult = self.pack_fitpar(self.mresult)
        dpath = self.lfs.path[0]
        path, ftype = QFileDialog.getSaveFileName(self, "Save MDC Fitted Data", os.path.join(dpath, self.lfs.name[0]+"_mfit"), "NPZ files (*.npz)")
        try:
            self.raise_()
            self.activateWindow()
        except:
            pass
        if len(path) > 2:
            self.g_exp.close()
            self.g_exp = None
            data = self.data
            fev, fwhm, pos = self.fev, self.fwhm, self.mpos
            skmin, skmax = self.kmin, self.kmax
            smaa1, smaa2 = self.maa1, self.maa2
            smfp, smfi = self.mfp, self.mfi
            smresult = self.smresult
            smcst = self.smcst
            
            shape=data.shape
            mdet=data.data[shape[0]//2, shape[1]//2]
            np.savez(path, path=dpath, fev=fev, fwhm=fwhm, pos=pos, skmin=skmin,
                    skmax=skmax, smaa1=smaa1, smaa2=smaa2, smfp=smfp, smfi=smfi, smresult=smresult, smcst=smcst, mdet=mdet)
            self.close_flag = 0
        else:
            self.g_exp.show()
            self.g_exp.raise_()
            self.g_exp.activateWindow()
            self.close_flag = 1
    
    def fmfall(self):
        QTimer.singleShot(100, self.mfitjob)
    
    def fitm(self): # init fitter
        # global ev, phi, data, mvv, maa1, maa2, fmxx, fmyy, fmx, fmy, kmin, kmax, cki, mbase, mprfit, mf_prswap, smresult, klim, fpr
        self.mprfit = 0
        self.cki = []        
        self.mf_prswap = []
        npzf = self.lfs.f_npz[0]
        ev, phi = self.eV, self.phi
        # self.mbase = [0 for i in range(len(ev))]
        data = self.data
        # self.klim = oklim(npzf, ev, phi)
        # shape=data.shape
        # det=data.data[shape[0]//2, shape[1]//2]
        # if self.mdet != det:
        #     self.fpr = 0
        # if self.fpr == 1:
        #     try:
        #         kmin, kmax = self.skmin, self.skmax # 需要載入
        #     except NameError:
        #         kmin, kmax = self.klim.min.copy(), self.klim.max.copy()
            # if len(scki) >= 2:
            #     cki = scki
        # else:
        #     kmin, kmax = self.klim.min.copy(), self.klim.max.copy()
        kmin, kmax = self.kmin, self.kmax
        # fmxx = np.asarray((np.ones(len(phi)*len(ev)), dtype=float).reshape(len(ev), len(phi)))
        # fmyy = np.asarray((np.ones(len(phi)*len(ev)), dtype=float).reshape(len(ev), len(phi)))
        # fmxx *= -50
        # fmyy *= -50
        self.fmx = np.asarray(np.arange(len(phi)*len(ev), dtype=float).reshape(len(ev), len(phi)))
        self.fmy = np.asarray(np.arange(len(phi)*len(ev), dtype=float).reshape(len(ev), len(phi)))
        self.mvv = np.asarray(np.arange(len(ev)), dtype=float)
        self.maa1 = np.asarray(np.arange(4*len(ev)).reshape(len(ev), 4), dtype=float)
        self.maa2 = np.asarray(np.arange(8*len(ev)).reshape(len(ev), 8), dtype=float)
        # fmx, fmy = self.fmx, self.fmy
        # mvv = self.mvv
        # maa1 = self.maa1
        # maa2 = self.maa2
        
        pbar = tqdm.tqdm(total=len(ev), desc='MDC', colour='green')
        for i, v in enumerate(ev):
            ecut = data.sel(eV=v, method='nearest')
            if npzf:x = phi
            else:x = np.asarray((2*m*v*1.602176634*10**-19)**0.5*np.sin(phi/180*np.pi)*10**-10/(h/2/np.pi), dtype=float)
            y = ecut.to_numpy().reshape(len(x))
            try:
                xx, x_arg = filter(x, kmin[i], kmax[i])
            except IndexError:
                print("\033[31m\nCheck the Raw Data compatible with the current MDC Fitted File\n\033[0m")
                return
            except Exception as e:
                print("\nError occurred while filtering:", e, '\n')
                return
            # tx = x[np.argwhere(x >= kmin[i])].flatten()
            # xx = tx[np.argwhere(tx <= kmax[i])].flatten()
            # ty = y[np.argwhere(x >= kmin[i])].flatten()
            # yy = ty[np.argwhere(tx <= kmax[i])].flatten()
            # yy = y[x_arg]
            # yy = np.where(yy > int(base.get()), yy, int(base.get()))
            try:
                if i in self.smfi and self.fpr == 1:
                    a1 = self.smaa1[i, :]
                    a2 = self.smaa2[i, :]
                    smrx1 = self.smresult[i, 0]
                    smrx2 = self.smresult[i, 1]
                    smrh1 = self.smresult[i, 2]
                    smrh2 = self.smresult[i, 3]
                    smrw1 = self.smresult[i, 4]
                    smrw2 = self.smresult[i, 5]
                    if self.smaa1[i, 1] == 10 or self.smaa2[i, 1] == 10:
                        self.mprfit = 1
                    else:
                        # fmxx[i, :len(xx)] = xx
                        # tx = fmxx[i, np.argwhere(fmxx[i, :] >= -20)].flatten()
                        tx = xx
                        ty = gl1(tx, *a2[:4])
                        s1 = np.sum(np.array([((ty[i]+ty[i+1])/2)for i in range(len(tx)-1)])
                                # Area 1
                                * np.array(([(tx[i+1]-tx[i])for i in range(len(tx)-1)])))
                        ty = gl1(tx, *a2[-4:])
                        s2 = np.sum(np.array([((ty[i]+ty[i+1])/2)for i in range(len(tx)-1)])
                                # Area 2
                                * np.array(([(tx[i+1]-tx[i])for i in range(len(tx)-1)])))
                        if s1 < s2:
                            t1, t2 = a2[:4], a2[-4:]
                            a2 = np.array([t2, t1]).flatten()
                            self.mf_prswap.append(i)
                            smrx1 = smrx1.replace('x2', 'x1').replace('x1:', 'x2:')
                            smrx2 = smrx2.replace('x1', 'x2').replace('x2:', 'x1:')
                            smrh1 = smrh1.replace('h1:', 'h2:')
                            smrh2 = smrh2.replace('h2:', 'h1:')
                            smrw1 = smrw1.replace('w1:', 'w2:').replace('w2', 'w1')
                            smrw2 = smrw2.replace('w2:', 'w1:').replace('w1', 'w2')
                            smr = np.array([smrx2,smrx1,smrh2,smrh1,smrw2,smrw1]).flatten()
                        else:
                            smr = np.array([smrx1,smrx2,smrh1,smrh2,smrw1,smrw2]).flatten()
                else:
                    # a1 = [(kmin[i]+kmax[i])/2, (np.max(y) -
                    #                             int(base.get())), 0.5, int(base.get())]
                    # a2 = [(kmin[i]+kmax[i])/2, (np.max(y)-int(base.get())), 0.5, int(base.get()),
                    #       (kmin[i]+kmax[i])/2, (np.max(y)-int(base.get())), 5, int(base.get())]
                    a1 = [(kmin[i]+kmax[i])/2, (np.max(y)-0), 0.5, 0]
                    a2 = [(kmin[i]+kmax[i])/2, (np.max(y)-0), 0.5, 0,
                        (kmin[i]+kmax[i])/2, (np.max(y)-0), 5, 0]
                    smr = ['' for i in range(6)]
            except:
                # a1 = [(kmin[i]+kmax[i])/2, (np.max(y) -
                #                             int(base.get())), 0.5, int(base.get())]
                # a2 = [(kmin[i]+kmax[i])/2, (np.max(y)-int(base.get())), 0.5, int(base.get()),
                #       (kmin[i]+kmax[i])/2, (np.max(y)-int(base.get())), 0.5, int(base.get())]
                a1 = [(kmin[i]+kmax[i])/2, (np.max(y)-0), 0.5, 0]
                a2 = [(kmin[i]+kmax[i])/2, (np.max(y)-0), 0.5, 0,
                    (kmin[i]+kmax[i])/2, (np.max(y)-0), 0.5, 0]
                smr = ['' for i in range(6)]

            # fmxx[i, :len(xx)] = xx
            # fmyy[i, :len(yy)] = yy
            fmxx, fmyy = 1, 1 # 未使用 暫時保留
            
            self.fmx[i, :] = x
            self.fmy[i, :] = y
            self.mvv[i] = v
            self.maa1[i, :] = a1
            self.maa2[i, :] = a2
            try:
                self.smresult[i, :]=smr
            except:
                pass
            pbar.update(1)
        pbar.close()
        # threading.Thread(target=self.mjob, daemon=True).start()
    
    def fgl2_a(self, params, x, data):
        fa1, fa2 = float(self.maf1.text()), float(self.maf2.text())
        h1 = params['h1']
        h2 = params['h2']
        x1 = params['x1']
        x2 = params['x2']
        w1 = params['w1']
        w2 = params['w2']
        y1 = params['y1']
        y2 = params['y2']
        model = (gl1(x, x1, h1, w1, y1) +
                gl1(x, x2, h2, w2, y2))
        area1 = np.sum(gl1(x, x1, h1, w1, y1))
        area2 = np.sum(gl1(x, x2, h2, w2, y2))
        return model - data + area1/fa1 - area2/fa2

    def toa1(self):
        result = self.result
        a1 = []
        a1.append(result.params['x'].value)
        a1.append(result.params['h'].value)
        a1.append(result.params['w'].value)
        a1.append(result.params['y'].value)
        return a1

    def toa2(self, xx):
        result = self.result
        a2 = []
        a2.append(result.params['x1'].value)
        a2.append(result.params['h1'].value)
        a2.append(result.params['w1'].value)
        a2.append(result.params['y1'].value)
        a2.append(result.params['x2'].value)
        a2.append(result.params['h2'].value)
        a2.append(result.params['w2'].value)
        a2.append(result.params['y2'].value)
        
        self.fswa1a2 = 0
        # i = mfiti.get()
        
        # fmxx[i, :len(xx)] = xx
        # x = fmxx[i, np.argwhere(fmxx[i, :] >= -20)].flatten()
        x = xx
        ty = gl1(x, *a2[:4])
        s1 = np.sum(np.array([((ty[i]+ty[i+1])/2)for i in range(len(x)-1)])
                # Area 1
                * np.array(([(x[i+1]-x[i])for i in range(len(x)-1)])))
        ty = gl1(x, *a2[-4:])
        s2 = np.sum(np.array([((ty[i]+ty[i+1])/2)for i in range(len(x)-1)])
                # Area 2
                * np.array(([(x[i+1]-x[i])for i in range(len(x)-1)])))
        if s1 < s2:
            t1, t2 = a2[:4], a2[-4:]
            a2 = np.array([t2, t1]).flatten()
            self.fswa1a2 = 1
        return a2

    def swapc1c2(self):
        i = self.index
        result = self.result
        if self.mfp[i] == 2 and self.fswa1a2 == 1:
            o_result = copy.deepcopy(self.result)
            a1=['x1', 'h1', 'w1', 'y1']
            a2=['x2', 'h2', 'w2', 'y2']
            for i in range(4):
                if o_result.params[a2[i]].expr is not None:
                    if a1[i] in o_result.params[a2[i]].expr:
                        o_result.params[a2[i]].set(expr=o_result.params[a2[i]].expr.replace(a1[i], a2[i]))
                result.params[a1[i]].set(value=o_result.params[a2[i]].value, min=o_result.params[a2[i]].min, max=o_result.params[a2[i]].max, expr=o_result.params[a2[i]].expr, brute_step=o_result.params[a2[i]].brute_step, vary=o_result.params[a2[i]].vary)
                result.params[a2[i]].set(value=o_result.params[a1[i]].value, min=o_result.params[a1[i]].min, max=o_result.params[a1[i]].max, expr=o_result.params[a1[i]].expr, brute_step=o_result.params[a1[i]].brute_step, vary=o_result.params[a1[i]].vary)
        return result
    
    def mfit(self):
        self.b_comp1.setStyleSheet("")
        self.b_comp2.setStyleSheet("")
        self.mmof = 1
        klim = self.klim
        mfi, mfi_err, mfi_x = list(self.mfi), list(self.mfi_err), list(self.mfi_x)
        flmposcst = self.flmposcst
        flmcomp1 = self.flmcomp1
        flmcomp2 = self.flmcomp2
        smcst = self.smcst
        maa2 = self.maa2
        
        
        self.msave_state()
        i = self.index
        self.kmin[i], self.kmax[i] = self.reg.getRegion()
        kmin, kmax = self.kmin, self.kmax
        data = self.data
        # self.mbase[i] = int(base.get())  # 待調整
        self.mbase[i] = 0  # 待調整
        # fmxx[i, :] = fmxx[i, :]/fmxx[i, :]*-50
        # fmyy[i, :] = fmyy[i, :]/fmyy[i, :]*-50
        ecut = data.sel(eV=self.eV[i], method='nearest')
        if self.lfs.f_npz[0]:x = self.phi
        else:x = (2*m*self.eV[i]*1.602176634*10**-19)**0.5*np.sin(self.phi/180*np.pi)*10**-10/(h/2/np.pi)
        y = ecut.to_numpy().reshape(len(x))
        xx, x_arg = filter(x, self.kmin[i], self.kmax[i])
        # tx = x[np.argwhere(x >= kmin[i])].flatten()
        # xx = tx[np.argwhere(tx <= kmax[i])].flatten()
        # ty = y[np.argwhere(x >= kmin[i])].flatten()
        # yy = ty[np.argwhere(tx <= kmax[i])].flatten()
        yy = y[x_arg]
        yy = np.where(yy > self.mbase[i], yy, self.mbase[i])
        try:
            if self.mfp[i] == 1:
                self.smcst[i] = [0, 0, 0, 0, 0, 0]
                pars = Parameters()
                pars.add('x', value=kmin[i]+(kmax[i]-kmin[i])
                        * 0.2, min=kmin[i], max=kmax[i])
                pars.add('h', value=(
                    np.max(y)-self.mbase[i])+1, min=(np.max(y)-self.mbase[i])/10, max=np.max(y)-self.mbase[i]+1)
                pars.add('w', value=0.1, min=0.01, max=0.2)
                pars.add('y', value=0, vary=False)
                fitter = Minimizer(fgl1, pars, fcn_args=(xx, yy-self.lnr_bg(yy)))
                self.result = fitter.minimize()
                a1 = self.toa1()
                self.checkfit()
                if self.fit_warn == 1:
                    t = 5
                    while t > 0 and self.fit_warn == 1:
                        self.result = fitter.minimize()
                        a1 = self.toa1()
                        self.checkfit()
                        t -= 1
            elif self.mfp[i] == 2:
                flmcomp = self.flmcomp
                pars = Parameters()
                xr1, xr2 = float(self.mxf1.text()), float(self.mxf2.text())
                wr1, wr2 = float(self.mwf1.text()), float(self.mwf2.text())
                fa1, fa2 = float(self.maf1.text()), float(self.maf2.text())
                smcst[i] = [xr1, xr2, wr1, wr2, fa1, fa2]
                if flmcomp == 1:
                    if flmcomp1 == 1:
                        self.flmcomp1 = -1
                        pars.add('x1', value=maa2[i, 0], min=kmin[i], max=kmax[i])
                        if flmposcst == 1:
                            pars.add('xr1', value=xr1, vary=False)
                            pars.add('xr2', value=xr2, vary=False)
                            pars.add('x2', expr='x1*xr1+xr2')
                        else:
                            pars.add('x2', value=maa2[i, 4], min=kmin[i], max=kmax[i])
                    elif flmcomp2 == 1:
                        self.flmcomp2 = -1
                        pars.add('x2', value=maa2[i, 4], min=kmin[i], max=kmax[i])
                        if flmposcst == 1:
                            pars.add('xr1', value=xr1, vary=False)
                            pars.add('xr2', value=xr2, vary=False)
                            pars.add('x1', expr="(x2-xr2) / xr1")
                        else:
                            pars.add('x1', value=maa2[i, 0], min=kmin[i], max=kmax[i])
                            
                    
                    pars.add('h1', value=maa2[i, 1], min=(
                        np.max(y)-self.mbase[i])/10, max=np.max(y)-self.mbase[i]+1)
                    pars.add('h2', value=maa2[i, 5], min=(
                        np.max(y)-self.mbase[i])/10, max=np.max(y)-self.mbase[i]+1)
                    pars.add('w1', value=maa2[i, 2], min=0.01, max=0.2)
                    if wr1 != 0 and wr2 != 0:
                        pars.add('wr1', value=wr1, vary=False)
                        pars.add('wr2', value=wr2, vary=False)
                        pars.add('w2', expr='w1/wr1*wr2')
                    else:
                        pars.add('w2', value=maa2[i, 6], min=0.01, max=0.2)
                else:
                    pars.add('x1', value=kmin[i]+(kmax[i] -
                            kmin[i])*0.3, min=kmin[i], max=kmax[i])
                    if flmposcst == 1:
                        pars.add('xr1', value=xr1, vary=False)
                        pars.add('xr2', value=xr2, vary=False)
                        pars.add('x2', expr='x1*xr1+xr2')
                    else:
                        pars.add(
                            'x2', value=kmax[i]-(kmax[i]-kmin[i])*0.3, min=kmin[i], max=kmax[i])
                    pars.add('h1', value=(
                        np.max(y)-self.mbase[i])+1, min=(np.max(y)-self.mbase[i])/10, max=np.max(y)-self.mbase[i]+1)
                    pars.add('h2', value=(
                        np.max(y)-self.mbase[i])+1, min=(np.max(y)-self.mbase[i])/10, max=np.max(y)-self.mbase[i]+1)
                    pars.add('w1', value=0.02, min=0.01, max=0.2)
                    if wr1 != 0 and wr2 != 0:
                        pars.add('wr1', value=wr1, vary=False)
                        pars.add('wr2', value=wr2, vary=False)
                        pars.add('w2', expr='w1/wr1*wr2')
                    else:
                        pars.add('w2', value=0.02, min=0.01, max=0.2)

                pars.add('y1', value=0, vary=False)
                pars.add('y2', value=0, vary=False)
                if fa1 != 0 and fa2 != 0:
                    fitter = Minimizer(self.fgl2_a, pars, fcn_args=(xx, yy-self.lnr_bg(yy)))
                    self.result = fitter.minimize()
                else:
                    fitter = Minimizer(fgl2, pars, fcn_args=(xx, yy-self.lnr_bg(yy)))
                    self.result = fitter.minimize()
                a2 = self.toa2(xx)
                self.checkfit()
                if self.fit_warn == 1:
                    t = 5
                    while t > 0 and self.fit_warn == 1:
                        result = fitter.minimize()
                        a2 = self.toa2(xx)
                        self.checkfit()
                        t -= 1
            report_fit(self.result)
            self.result=self.swapc1c2()
            self.mresult[i] = self.result

            if (kmin[i], kmax[i]) == (klim.min[i], klim.max[i]):
                if i not in mfi_x:
                    mfi_x.append(i)
                if i in mfi:
                    mfi.remove(i)
                if i in mfi_err:
                    mfi_err.remove(i)
            elif (kmin[i], kmax[i]) != (klim.min[i], klim.max[i]):
                if self.fit_warn == 0:
                    if i not in mfi:
                        mfi.append(i)
                    if i in mfi_x:
                        mfi_x.remove(i)
                    if i in mfi_err:
                        mfi_err.remove(i)
                else:
                    if i not in mfi_err:
                        mfi_err.append(i)
                    if i in mfi_x:
                        mfi_x.remove(i)
                    if i in mfi:
                        mfi.remove(i)
        except RuntimeError:
            if i not in mfi_err:
                mfi_err.append(i)
            if i in mfi_x:
                mfi_x.remove(i)
            if i in mfi:
                mfi.remove(i)
            a1 = [(kmin[i]+kmax[i])/2, (np.max(y)-self.mbase[i]), 0.5, self.mbase[i]]
            a2 = [(kmin[i]+kmax[i])/2, (np.max(y)-self.mbase[i]), 0.5, self.mbase[i],
                (kmin[i]+kmax[i])/2, (np.max(y)-self.mbase[i]), 0.5, self.mbase[i]]
        except IndexError:
            if i not in mfi_err:
                mfi_err.append(i)
            if i in mfi_x:
                mfi_x.remove(i)
            if i in mfi:
                mfi.remove(i)
            a1 = [(kmin[i]+kmax[i])/2, (np.max(y)-self.mbase[i]), 0.5, self.mbase[i]]
            a2 = [(kmin[i]+kmax[i])/2, (np.max(y)-self.mbase[i]), 0.5, self.mbase[i],
                (kmin[i]+kmax[i])/2, (np.max(y)-self.mbase[i]), 0.5, self.mbase[i]]
            self.smcst=np.zeros(len(self.eV)*6).reshape(len(self.eV),6)

        # fmxx[i, :len(xx)] = xx
        # fmyy[i, :len(yy)] = yy
        self.fmx[i, :] = x
        self.fmy[i, :] = y
        self.mvv[i] = self.eV[i]
        if self.mfp[i] == 1:
            self.maa1[i, :] = a1
        elif self.mfp[i] == 2:
            self.maa2[i, :] = a2
        self.mfi, self.mfi_err, self.mfi_x = mfi, mfi_err, mfi_x
        
        
    def mfitjob(self):
        # global fmxx, fmyy, fmx, fmy, mvv, maa1, maa2, kmin, kmax, mfi, mfi_err, mfi_x, st, mst, result, fa1, fa2, fit_warn, wr1, wr2, mresult, xr1, xr2, smcst
        if len(self.mfi) < 1:
            mfi, mfi_err, mfi_x = [], [], []
        else:
            mfi, mfi_err, mfi_x = list(self.mfi), list(self.mfi_err), list(self.mfi_x)
        kmin, kmax = self.kmin, self.kmax
        data = self.data
        npzf = self.lfs.f_npz[0]
        phi = self.phi
        klim = self.klim
        mfp = self.mfp
        maa1, maa2 = self.maa1, self.maa2
        mresult = self.mresult
        flmposcst = self.flmposcst
        
        self.msave_state()
        pbar = tqdm.tqdm(total=len(self.eV), desc='Fitting MDC', colour='green')
        for i in range(len(self.eV)):
            # self.mbase[i] = int(base.get())  # 待調整
            self.mbase[i] = 0  # 待調整
            # fmxx[i, :] = fmxx[i, :]/fmxx[i, :]*-50
            # fmyy[i, :] = fmyy[i, :]/fmyy[i, :]*-50
            ecut = data.sel(eV=self.eV[i], method='nearest')
            if npzf:x = phi
            else:x = (2*m*self.eV[i]*1.602176634*10**-19)**0.5*np.sin(phi/180*np.pi)*10**-10/(h/2/np.pi)
            y = ecut.to_numpy().reshape(len(x))
            xx, x_arg = filter(x, kmin[i], kmax[i])
            # tx = x[np.argwhere(x >= kmin[i])].flatten()
            # xx = tx[np.argwhere(tx <= kmax[i])].flatten()
            # ty = y[np.argwhere(x >= kmin[i])].flatten()
            # yy = ty[np.argwhere(tx <= kmax[i])].flatten()
            yy = y[x_arg]
            yy = np.where(yy > self.mbase[i], yy, self.mbase[i])
            try:
                # if (kmin[i],kmax[i])==(klim.min[i], klim.max[i]) and i not in mfi:
                # if i not in mfi:
                #     if i not in mfi_x:
                #         mfi_x.append(i)
                #     # if i in mfi:
                #     #     mfi.remove(i)
                #     if i in mfi_err:
                #         mfi_err.remove(i)
                #     a1=[(kmin[i]+kmax[i])/2,(np.max(y)-self.mbase[i]),5,mbase[i]]
                #     a2=[(kmin[i]+kmax[i])/2,(np.max(y)-self.mbase[i]),5,mbase[i],(kmin[i]+kmax[i])/2,(np.max(y)-self.mbase[i]),5,mbase[i]]
                # elif (kmin[i],kmax[i])!=(klim.min[i], klim.max[i]):
                if mfp[i] == 1:
                    self.smcst[i] = [0, 0, 0, 0, 0, 0]
                    if i in mfi_err and (kmin[i], kmax[i]) != (klim.min[i], klim.max[i]):
                        pars = Parameters()
                        pars.add(
                            'x', value=kmin[i]+(kmax[i]-kmin[i])*0.3, min=kmin[i], max=kmax[i])
                        pars.add('h', value=(
                            np.max(y)-self.mbase[i])+1, min=(np.max(y)-self.mbase[i])/10, max=np.max(y)-self.mbase[i]+1)
                        pars.add('w', value=0.1, min=0.01, max=0.2)
                        pars.add('y', value=0, vary=False)
                        fitter = Minimizer(
                            fgl1, pars, fcn_args=(xx, yy-self.lnr_bg(yy)))
                        self.result = fitter.minimize()
                        a1 = self.toa1()
                        self.checkfit()
                        if self.fit_warn == 1:
                            t = 5
                            while t > 0 and self.fit_warn == 1:
                                self.result = fitter.minimize()
                                a1 = self.toa1()
                                self.checkfit()
                                t -= 1
                    else:
                        if i in mfi:
                            self.result = mresult[i]
                        a1 = maa1[i, :]
                        if (kmin[i], kmax[i]) == (klim.min[i], klim.max[i]):
                            self.fit_warn = 2
                        elif i not in mfi:
                            pars = Parameters()
                            pars.add(
                                'x', value=kmin[i]+(kmax[i]-kmin[i])*0.3, min=kmin[i], max=kmax[i])
                            pars.add('h', value=(
                                np.max(y)-self.mbase[i])+1, min=(np.max(y)-self.mbase[i])/10, max=np.max(y)-self.mbase[i]+1)
                            pars.add('w', value=0.1, min=0.01, max=0.2)
                            pars.add('y', value=0, vary=False)
                            fitter = Minimizer(
                                fgl1, pars, fcn_args=(xx, yy-self.lnr_bg(yy)))
                            self.result = fitter.minimize()
                            a1 = self.toa1()
                            self.checkfit()
                            if self.fit_warn == 1:
                                t = 5
                                while t > 0 and self.fit_warn == 1:
                                    self.result = fitter.minimize()
                                    a1 = self.toa1()
                                    self.checkfit()
                                    t -= 1
                        else:
                            self.fit_warn = 0
                elif mfp[i] == 2:
                    if i in mfi_err and (kmin[i], kmax[i]) != (klim.min[i], klim.max[i]):
                        pars = Parameters()
                        xr1, xr2 = float(self.mxf1.text()), float(self.mxf2.text())
                        wr1, wr2 = float(self.mwf1.text()), float(self.mwf2.text())
                        fa1, fa2 = float(self.maf1.text()), float(self.maf2.text())
                        self.smcst[i]=[xr1,xr2,wr1,wr2,fa1,fa2]
                        pars.add(
                            'x1', value=kmin[i]+(kmax[i]-kmin[i])*0.3, min=kmin[i], max=kmax[i])
                        if flmposcst == 1:
                            pars.add('xr1', value=xr1, vary=False)
                            pars.add('xr2', value=xr2, vary=False)
                            pars.add('x2', expr='x1*xr1+xr2')
                        else:
                            pars.add(
                                'x2', value=kmax[i]-(kmax[i]-kmin[i])*0.3, min=kmin[i], max=kmax[i])
                        pars.add('h1', value=(
                            np.max(y)-self.mbase[i])+1, min=(np.max(y)-self.mbase[i])/10, max=np.max(y)-self.mbase[i]+1)
                        pars.add('h2', value=(
                            np.max(y)-self.mbase[i])+1, min=(np.max(y)-self.mbase[i])/10, max=np.max(y)-self.mbase[i]+1)
                        pars.add('w1', value=0.02, min=0, max=0.2)
                        if wr1 != 0 and wr2 != 0:
                            pars.add('wr1', value=wr1, vary=False)
                            pars.add('wr2', value=wr2, vary=False)
                            pars.add('w2', expr='w1/wr1*wr2')
                        else:
                            pars.add('w2', value=0.02, min=0, max=0.2)
                        pars.add('y1', value=0, vary=False)
                        pars.add('y2', value=0, vary=False)
                        if fa1 != 0 and fa2 != 0:
                            fitter = Minimizer(
                                self.fgl2_a, pars, fcn_args=(xx, yy-self.lnr_bg(yy)))
                            self.result = fitter.minimize()
                        else:
                            fitter = Minimizer(
                                fgl2, pars, fcn_args=(xx, yy-self.lnr_bg(yy)))
                            self.result = fitter.minimize()
                        a2 = self.toa2(xx)
                        self.checkfit()
                        if self.fit_warn == 1:
                            t = 5
                            while t > 0 and self.fit_warn == 1:
                                self.result = fitter.minimize()
                                a2 = self.toa2(xx)
                                self.checkfit()
                                t -= 1
                    else:
                        if i in mfi:
                            self.result = mresult[i]
                        a2 = maa2[i, :]
                        if (kmin[i], kmax[i]) == (klim.min[i], klim.max[i]):
                            self.fit_warn = 2
                        elif i not in mfi:
                            pars = Parameters()
                            xr1, xr2 = float(self.mxf1.text()), float(self.mxf2.text())
                            wr1, wr2 = float(self.mwf1.text()), float(self.mwf2.text())
                            fa1, fa2 = float(self.maf1.text()), float(self.maf2.text())
                            self.smcst[i]=[xr1,xr2,wr1,wr2,fa1,fa2]
                            pars.add(
                                'x1', value=kmin[i]+(kmax[i]-kmin[i])*0.3, min=kmin[i], max=kmax[i])
                            if flmposcst == 1:
                                pars.add('xr1', value=xr1, vary=False)
                                pars.add('xr2', value=xr2, vary=False)
                                pars.add('x2', expr='x1*xr1+xr2')
                            else:
                                pars.add(
                                    'x2', value=kmax[i]-(kmax[i]-kmin[i])*0.3, min=kmin[i], max=kmax[i])
                            pars.add('h1', value=(
                                np.max(y)-self.mbase[i])+1, min=(np.max(y)-self.mbase[i])/10, max=np.max(y)-self.mbase[i]+1)
                            pars.add('h2', value=(
                                np.max(y)-self.mbase[i])+1, min=(np.max(y)-self.mbase[i])/10, max=np.max(y)-self.mbase[i]+1)
                            pars.add('w1', value=0.02, min=0.01, max=0.2)
                            if wr1 != 0 and wr2 != 0:
                                pars.add('wr1', value=wr1, vary=False)
                                pars.add('wr2', value=wr2, vary=False)
                                pars.add('w2', expr='w1/wr1*wr2')
                            else:
                                pars.add('w2', value=0.02, min=0.01, max=0.2)
                            pars.add('y1', value=0, vary=False)
                            pars.add('y2', value=0, vary=False)
                            if fa1 != 0 and fa2 != 0:
                                fitter = Minimizer(
                                    self.fgl2_a, pars, fcn_args=(xx, yy-self.lnr_bg(yy)))
                                self.result = fitter.minimize()
                            else:
                                fitter = Minimizer(
                                    fgl2, pars, fcn_args=(xx, yy-self.lnr_bg(yy)))
                                self.result = fitter.minimize()
                            a2 = self.toa2(xx)
                            self.checkfit()
                            if self.fit_warn == 1:
                                t = 5
                                while t > 0 and self.fit_warn == 1:
                                    self.result = fitter.minimize()
                                    a2 = self.toa2(xx)
                                    self.checkfit()
                                    t -= 1
                        else:
                            self.fit_warn = 0
                try:
                    '''using lmfit'''
                    result=self.swapc1c2()
                    mresult[i] = result
                    result = []
                except:
                    '''Casa Result'''
                    pass
                if self.fit_warn == 0:
                    if i not in mfi:
                        self.mfi.append(i)
                    if i in mfi_x:
                        self.mfi_x.remove(i)
                    if i in mfi_err:
                        self.mfi_err.remove(i)
                elif self.fit_warn == 2:
                    if i not in mfi_x:
                        self.mfi_x.append(i)
                    if i in mfi:
                        self.mfi.remove(i)
                    if i in mfi_err:
                        self.mfi_err.remove(i)
                else:
                    if i not in mfi_err:
                        self.mfi_err.append(i)
                    if i in mfi_x:
                        self.mfi_x.remove(i)
                    if i in mfi:
                        self.mfi.remove(i)
            except RuntimeError:
                print('runtime error')
                if i not in mfi_err:
                    self.mfi_err.append(i)
                if i in mfi_x:
                    self.mfi_x.remove(i)
                if i in mfi:
                    self.mfi.remove(i)
                a1 = [(kmin[i]+kmax[i])/2, (np.max(y)-self.mbase[i]), 5, self.mbase[i]]
                a2 = [(kmin[i]+kmax[i])/2, (np.max(y)-self.mbase[i]), 5, self.mbase[i],
                    (kmin[i]+kmax[i])/2, (np.max(y)-self.mbase[i]), 5, self.mbase[i]]
            except IndexError:
                if i not in mfi_err:
                    self.mfi_err.append(i)
                if i in mfi_x:
                    self.mfi_x.remove(i)
                if i in mfi:
                    self.mfi.remove(i)
                a1 = [(kmin[i]+kmax[i])/2, (np.max(y)-self.mbase[i]), 5, self.mbase[i]]
                a2 = [(kmin[i]+kmax[i])/2, (np.max(y)-self.mbase[i]), 5, self.mbase[i],
                    (kmin[i]+kmax[i])/2, (np.max(y)-self.mbase[i]), 5, self.mbase[i]]
                self.smcst=np.zeros(len(self.eV)*6).reshape(len(self.eV),6)
            # fmxx[i, :len(xx)] = xx
            # fmyy[i, :len(yy)] = yy
            self.fmx[i, :] = x
            self.fmy[i, :] = y
            self.mvv[i] = self.eV[i]
            if self.mfp[i] == 1:
                self.maa1[i, :] = a1
            elif self.mfp[i] == 2:
                self.maa2[i, :] = a2
            pbar.update(1)
        pbar.close()
        self.mfitplot()

    
    def checkfit(self):
        result = self.result
        self.fit_warn = 0
        t = 0
        for i in result.params:
            par = result.params[i]
            if par.value != 0:
                try:
                    if par.stderr/par.value > 0.2:  # uncertainty 20%
                        t += 1
                except TypeError:
                    self.fit_warn = 1
        if t >= 2:
            self.fit_warn = 1
    
    def mfcomp1(self):
        # global mbcomp1, flmcomp1, mbcomp2, flmcomp2
        if self.flmcomp2 == -1:
            self.flmcomp1 *= -1
            if self.flmcomp1 == 1:
                self.b_comp1.setText('Comp 1')
                self.b_comp1.setStyleSheet("background-color: green;color: #EEE;")
                self.reg.setMovable(False)
            else:
                self.b_comp1.setText('Comp 1')
                self.b_comp1.setStyleSheet("")
                self.reg.setMovable(True)
        else:
            self.flmcomp2 *= -1
            self.flmcomp1 *= -1
            self.b_comp1.setText('Comp 1')
            self.b_comp1.setStyleSheet("background-color: green;color: #EEE;")
            self.reg.setMovable(False)
            self.b_comp2.setText('Comp 2')
            self.b_comp2.setStyleSheet("")



    def mfcomp2(self):
        # global mbcomp1, flmcomp1, mbcomp2, flmcomp2
        if self.flmcomp1 == -1:
            self.flmcomp2 *= -1
            if self.flmcomp2 == 1:
                self.b_comp2.setText('Comp 2')
                self.b_comp2.setStyleSheet("background-color: purple;color: #EEE;")
                self.reg.setMovable(False)
            else:
                self.b_comp2.setText('Comp 2')
                self.b_comp2.setStyleSheet("")
                self.reg.setMovable(True)
        else:
            self.flmcomp1 *= -1
            self.flmcomp2 *= -1
            self.b_comp2.setText('Comp 2')
            self.b_comp2.setStyleSheet("background-color: purple;color: #EEE;")
            self.reg.setMovable(False)
            self.b_comp1.setText('Comp 1')
            self.b_comp1.setStyleSheet("")
        
    def mfitplot(self):  # mfiti Scale
        # global mfitax, mxl, myl, klmin, klmax, tmxl, kmin, kmax, maa2, flmcomp, lm1, lm2, lm3, lm4, lm5, lm6, mxf1, mxf2, mwf1, mwf2, maf1, maf2, mt1, mt2, mt3, mt4, mt5, fdo, mf_prswap
        mf_prswap = self.mf_prswap
        mresult = self.mresult
        fdo = self.fdo
        fmx, fmy = self.fmx, self.fmy
        mxf1, mxf2 = self.mxf1, self.mxf2
        mwf1, mwf2 = self.mwf1, self.mwf2
        maf1, maf2 = self.maf1, self.maf2
        kmin, kmax = self.kmin, self.kmax
        mfi, mfi_err, mfi_x = list(self.mfi), list(self.mfi_err), list(self.mfi_x)
        smcst = self.smcst
        maa1 = self.maa1
        maa2 = self.maa2
        mfp = self.mfp
        smresult = self.smresult
        lm1, lm2, lm3, lm4, lm5, lm6 = self.lm1, self.lm2, self.lm3, self.lm4, self.lm5, self.lm6
        
        
        
        i = self.index
        self.update_plot()
        txl = self.plot.viewRange()[0]
        tyl = self.plot.viewRange()[1]
        dy = (tyl[1]-tyl[0])/20
        dx = (txl[1]-txl[0])/50
        tymin = tyl[0]
        tymax = tyl[1]
        txmin = txl[0]
        txmax = txl[1]
        self.plot.plot([txmin-dx*10, txmax+dx*10], [tymax+dy, tymax+dy], pen=pg.mkPen(color='grey', width=1, style=Qt.SolidLine))
        self.plot.setRange(yRange=(tymin, tymax+3*dy))
        x, x_arg = filter(fmx[i, :], kmin[i], kmax[i])
        y = fmy[i, x_arg]
        lbg = self.lnr_bg(y)
        if i in mfi_x:
            for l, v in zip([lm1, lm2, lm3, lm4, lm5, lm6], ['', '', '', '', '', '']):
                l.setText(v)
            try:
                mxf1.setText(str(smcst[i][0]))
                mxf2.setText(str(smcst[i][1]))
                mwf1.setText(str(smcst[i][2]))
                mwf2.setText(str(smcst[i][3]))
                maf1.setText(str(smcst[i][4]))
                maf2.setText(str(smcst[i][5]))
            except:
                pass
        if mfp[i] == 1:
            try:
                mxf1.setText(str(smcst[i][0]))
                mxf2.setText(str(smcst[i][1]))
                mwf1.setText(str(smcst[i][2]))
                mwf2.setText(str(smcst[i][3]))
                maf1.setText(str(smcst[i][4]))
                maf2.setText(str(smcst[i][5]))
            except:
                pass
            if maa1[i, 0] == (kmin[i]+kmax[i])/2 and maa1[i, 2] == 0.5:
                self.fit_line = self.plot.plot(x, gl1(x, *maa1[i, :])+lbg, pen=pg.mkPen(color='r', width=5, style=Qt.DashLine))
            else:
                gl1_1 = gl1(x, *maa1[i, :])+lbg
                self.fit_line = self.plot.plot(x, gl1_1, pen=pg.mkPen(color='grey', width=5))
                self.base=self.plot.plot(x, lbg, pen=pg.mkPen(None))
                fill = pg.FillBetweenItem(self.fit_line, self.base, brush=pg.mkBrush(0, 0, 255, 128))
                self.plot.addItem(fill)
            if i in mfi_err or i in mfi:
                if i in mfi:
                    self.plot.plot(x, gl1(x, *maa1[i, :]) + lbg-y+tymax+dy, pen=pg.mkPen(color='gray', width=1, style=Qt.SolidLine))
                else:
                    self.plot.plot(x, gl1(x, *maa1[i, :]) + lbg-y+tymax+dy, pen=pg.mkPen(color='red', width=1, style=Qt.SolidLine))
                # s=(np.sum((gl1(x,*maa1[i,:])+lbg-y)**2)/(max(x)-min(x)))**0.5
                s = np.std(gl1(x, *maa1[i, :])+lbg-y)  # STD
                mt1 = pg.TextItem('Residual STD: '+str(round(s, 2)), color='w', anchor=(0, 0))
                mt1.setFont(QFont("Arial", 12))
                mt1.setPos(txmin+dx, tymax-dy)  # 設定位置
                self.plot.addItem(mt1)
                s = np.sqrt(np.mean((gl1(x, *maa1[i, :])+lbg-y)**2))  # RMS
                mt2 = pg.TextItem('Residual RMS: '+str(round(s, 2)), color='w', anchor=(0, 0))
                mt2.setFont(QFont("Arial", 12))
                mt2.setPos(txmin+dx, tymax-2*dy)  # 設定位置
                self.plot.addItem(mt2)
                ty = gl1(x, *maa1[i, :])
                s = np.sum(np.array([((ty[i]+ty[i+1])/2)for i in range(len(x)-1)])
                        # Area
                        * np.array(([(x[i+1]-x[i])for i in range(len(x)-1)])))
                mt3 = pg.TextItem('Area: '+str(round(s, 2)), color='w', anchor=(0, 0))
                mt3.setFont(QFont("Arial", 12))
                mt3.setPos(txmin+dx, tymax-3*dy)  # 設定位置
                self.plot.addItem(mt3)
                vv = []
                for ii in range(6):
                    if ii > 2:
                        vv.append(f"")
                    else:
                        vv.append(f"{gformat(maa1[i, ii])}")
                for l, n, v in zip([lm1, lm2, lm3, lm4, lm5, lm6], [f"x: ", f"h: ", f"w: ", f"", f"", f""], vv):
                    l.setText(n+v)
                    l.setAlignment(Qt.AlignCenter)
                try:
                    vv = smresult[i]
                    for l, v in zip([lm1, lm2, lm3, lm4, lm5, lm6], vv):
                        l.setText(v)
                        l.setAlignment(Qt.AlignLeft)
                except:
                    pass
                try:
                    self.fitpar1(mresult[i], lm1, lm2, lm3, lm4, lm5, lm6)
                except:
                    pass
        elif mfp[i] == 2:
            self.flmcomp = 0
            if maa2[i, 0] == (kmin[i]+kmax[i])/2 and maa2[i, 2] == 0.5:
                self.fit_line = self.plot.plot(x, gl2(x, *maa2[i, :])+lbg, pen=pg.mkPen(color='grey', width=5, style=Qt.DashLine))
            else:
                if self.flmcomp1 == 1:
                    if lbg[np.argwhere(abs(x-self.mcpx1) < 0.01)].flatten().size > 0:
                        maa2[i, :4] = [
                            self.mcpx1, self.mcpy1-lbg[np.argwhere(abs(x-self.mcpx1) < 0.01)].flatten()[0], 0.02, 0]
                        self.flmcomp = 1
                elif self.flmcomp2 == 1:
                    if lbg[np.argwhere(abs(x-self.mcpx2) < 0.01)].flatten().size > 0:
                        maa2[i, -4:] = [self.mcpx2, self.mcpy2 -
                                        lbg[np.argwhere(abs(x-self.mcpx2) < 0.01)].flatten()[0], 0.02, 0]
                        self.flmcomp = 1
                gl2_1 = gl1(x, *maa2[i, :4])+lbg
                gl2_2 = gl1(x, *maa2[i, -4:])+lbg
                self.fit_line = self.plot.plot(x, gl2_1+gl2_2 - lbg, pen=pg.mkPen(color='grey', width=5))
                self.base=self.plot.plot(x, lbg, pen=pg.mkPen(None))
                fill = pg.FillBetweenItem(self.plot.plot(x, gl2_1, pen=pg.mkPen(None)), self.base, brush=pg.mkBrush(0, 255, 0, 128))
                self.plot.addItem(fill)
                fill = pg.FillBetweenItem(self.plot.plot(x, gl2_2, pen=pg.mkPen(None)), self.base, brush=pg.mkBrush(128, 0, 128, 128))
                self.plot.addItem(fill)
            if i in mfi_err or i in mfi:
                if i in mfi:
                    self.plot.plot(x, gl2(x, *maa2[i, :]) + lbg-y+tymax+dy, pen=pg.mkPen(color='gray', width=1, style=Qt.SolidLine))
                else:
                    self.plot.plot(x, gl2(x, *maa2[i, :]) + lbg-y+tymax+dy, pen=pg.mkPen(color='red', width=1, style=Qt.SolidLine))
                # s=(np.sum((gl2(x,*maa2[i,:])+lbg-y)**2)/(max(x)-min(x)))**0.5
                s = np.std(gl2(x, *maa2[i, :])+lbg-y)  # STD
                mt1 = pg.TextItem('Residual STD: '+str(round(s, 2)), color='w', anchor=(0, 0))
                mt1.setFont(QFont("Arial", 12))
                mt1.setPos(txmin+dx, tymax-dy)  # 設定位置
                self.plot.addItem(mt1)
                s = np.sqrt(np.mean((gl2(x, *maa2[i, :])+lbg-y)**2))  # RMS
                mt2 = pg.TextItem('Residual RMS: '+str(round(s, 2)), color='w', anchor=(0, 0))
                mt2.setFont(QFont("Arial", 12))
                mt2.setPos(txmin+dx, tymax-2*dy)  # 設定位置
                self.plot.addItem(mt2)
                ty = gl1(x, *maa2[i, :4])
                s = np.sum(np.array([((ty[i]+ty[i+1])/2)for i in range(len(x)-1)])
                        # Area 1
                        * np.array(([(x[i+1]-x[i])for i in range(len(x)-1)])))
                mt3 = pg.TextItem('Area 1: '+str(round(s, 2)), color='w', anchor=(0, 0))
                mt3.setFont(QFont("Arial", 12))
                mt3.setPos(txmin+dx, tymax-3*dy)  # 設定位置
                self.plot.addItem(mt3)
                ty = gl1(x, *maa2[i, -4:])
                s = np.sum(np.array([((ty[i]+ty[i+1])/2)for i in range(len(x)-1)])
                        # Area 2
                        * np.array(([(x[i+1]-x[i])for i in range(len(x)-1)])))
                mt4 = pg.TextItem('Area 2: '+str(round(s, 2)), color='w', anchor=(0, 0))
                mt4.setFont(QFont("Arial", 12))
                mt4.setPos(txmin+dx, tymax-4*dy)  # 設定位置
                self.plot.addItem(mt4)
                try:
                    if smcst[i][4] != 0 and smcst[i][5] != 0:
                        mt5 = pg.TextItem('A1:A2='+str(smcst[i][4]) +':'+str(smcst[i][5]), color='w', anchor=(0, 0))
                        mt5.setFont(QFont("Arial", 12))
                        mt5.setPos(txmin+dx, tymax-5*dy)  # 設定位置
                        self.plot.addItem(mt5)
                    mxf1.setText(str(smcst[i][0]))
                    mxf2.setText(str(smcst[i][1]))
                    mwf1.setText(str(smcst[i][2]))
                    mwf2.setText(str(smcst[i][3]))
                    maf1.setText(str(smcst[i][4]))
                    maf2.setText(str(smcst[i][5]))
                except:
                    pass
                vv = []
                for ii in range(6):
                    if ii < 3:
                        vv.append(f"{gformat(maa2[i, ii])}")
                    else:
                        vv.append(f"{gformat(maa2[i, ii+1])}")

                for l, n, v in zip([lm1, lm3, lm5, lm2, lm4, lm6], [f"x1: ", f"h1: ", f"w1: ", f"x2: ", f"h2: ", f"w2: "], vv):
                    l.setText(n+v)
                    l.setAlignment(Qt.AlignCenter)
                try:
                    vv = smresult[i]
                    for l, v in zip([lm1, lm2, lm3, lm4, lm5, lm6], vv):
                        if 'nofit' not in v:
                            l.setText(v)
                            l.setAlignment(Qt.AlignLeft)
                except:
                    pass
                try:
                    if fdo==0 or i not in mf_prswap:
                        self.fitpar2(mresult[i], lm1, lm2, lm3, lm4, lm5, lm6)
                    else:
                        self.mresult[i]=smresult[i]
                        self.fdo=0
                        try:
                            if mf_prswap:
                                self.mf_prswap.remove(i)
                        except:
                            pass
                except:
                    pass
        x, x_arg, y, lbg, vv, ty, fl, txl, tyl, dx, dy = None, None, None, None, None, None, None, None, None, None, None

    
    def lnr_bg(self, x: np.ndarray, n_samples=5) -> np.ndarray:
        while len(x) < 2*n_samples:
            if len(x) < 2:
                o = np.array([])
            n_samples -= 1
        left, right = np.mean(x[:n_samples]), np.mean(x[-n_samples:])
        o = np.ones(len(x))*np.mean([left, right])
        return o+self.mbgv
        
    def right_layout(self, right_layout):
        do_layout = QHBoxLayout()
        b_undo = QPushButton("Undo")
        b_undo.clicked.connect(self.mundo)
        do_layout.addWidget(b_undo)
        b_redo = QPushButton("Redo")
        b_redo.clicked.connect(self.mredo)
        do_layout.addWidget(b_redo)
        right_layout.addLayout(do_layout)
        
        self.lm1 = QLabel("")
        self.lm2 = QLabel("")
        self.lm3 = QLabel("")
        self.lm4 = QLabel("")
        self.lm5 = QLabel("")
        self.lm6 = QLabel("")
        for i in [self.lm1, self.lm2, self.lm3, self.lm4, self.lm5, self.lm6]:
            i.setFont(QFont("Arial", 16, QFont.Bold))
            i.setFixedWidth(700)
            right_layout.addWidget(i, alignment=Qt.AlignCenter)
        
        confirm_layout = QHBoxLayout()
        b_accept = QPushButton("Accept")
        b_accept.clicked.connect(self.fmaccept)
        confirm_layout.addWidget(b_accept)
        self.b_reject = QPushButton("Reject")
        self.b_reject.clicked.connect(self.fmreject)
        confirm_layout.addWidget(self.b_reject)
        right_layout.addLayout(confirm_layout)
        
        l=QLabel("Index Operation")
        l.setFont(QFont("Arial", 12, QFont.Bold))
        right_layout.addWidget(l, alignment=Qt.AlignCenter)
        
        index_op_layout = QHBoxLayout()
        self.b_add2 = QPushButton("Add 2 Peaks")
        self.b_add2.clicked.connect(self.fmcgl2)
        index_op_layout.addWidget(self.b_add2)
        self.b_remove = QPushButton("Remove")
        self.b_remove.clicked.connect(self.fmrmv)
        index_op_layout.addWidget(self.b_remove)
        right_layout.addLayout(index_op_layout)
        
        comp_layout = QHBoxLayout()
        self.b_comp1 = QPushButton("Comp 1")
        self.b_comp1.clicked.connect(self.mfcomp1)
        self.b_comp1.setEnabled(False)
        comp_layout.addWidget(self.b_comp1)
        self.b_comp2 = QPushButton("Comp 2")
        self.b_comp2.clicked.connect(self.mfcomp2)
        self.b_comp2.setEnabled(False)
        comp_layout.addWidget(self.b_comp2)
        right_layout.addLayout(comp_layout)
        
        b_fit = QPushButton("Fit Components")
        b_fit.clicked.connect(self.ffitcp)
        right_layout.addWidget(b_fit)
        
        l=QLabel("FWHM Ratio Constraints")
        right_layout.addWidget(l, alignment=Qt.AlignCenter)
        
        wr_container = QWidget()
        wr_container.setFixedWidth(250)
        wr_layout = QHBoxLayout()
        self.mwf1 = QLineEdit("0.0")
        self.mwf1.textChanged.connect(self.fmwf1)
        self.mwf1.setFixedWidth(80)
        wr_layout.addWidget(self.mwf1)
        l=QLabel(":")
        l.setAlignment(Qt.AlignCenter)
        l.setFixedWidth(30)
        wr_layout.addWidget(l)
        self.mwf2 = QLineEdit("0.0")
        self.mwf2.textChanged.connect(self.fmwf2)
        self.mwf2.setFixedWidth(80)
        wr_layout.addWidget(self.mwf2)
        wr_container.setLayout(wr_layout)
        right_layout.addWidget(wr_container, alignment=Qt.AlignCenter)
        
        l=QLabel("Area Ratio Constraints")
        right_layout.addWidget(l, alignment=Qt.AlignCenter)
        
        ar_container = QWidget()
        ar_container.setFixedWidth(250)
        ar_layout = QHBoxLayout()
        self.maf1 = QLineEdit("0.0")
        self.maf1.textChanged.connect(self.fmaf1)
        self.maf1.setFixedWidth(80)
        ar_layout.addWidget(self.maf1)
        l=QLabel(":")
        l.setAlignment(Qt.AlignCenter)
        l.setFixedWidth(30)
        ar_layout.addWidget(l)
        self.maf2 = QLineEdit("0.0")
        self.maf2.textChanged.connect(self.fmaf2)
        self.maf2.setFixedWidth(80)
        ar_layout.addWidget(self.maf2)
        ar_container.setLayout(ar_layout)
        right_layout.addWidget(ar_container, alignment=Qt.AlignCenter)
        
        self.b_pos = QPushButton("Position Constraint")
        self.flmposcst = -1
        self.b_pos.pressed.connect(self.fmposcst)
        right_layout.addWidget(self.b_pos)
        
        pos_container = QWidget()
        pos_container.setFixedWidth(350)
        pos_layout = QHBoxLayout()
        l=QLabel("x2 = ")
        l.setAlignment(Qt.AlignCenter)
        pos_layout.addWidget(l)
        self.mxf1 = QLineEdit("1")
        self.mxf1.textChanged.connect(self.fmxf1)
        self.mxf1.setFixedWidth(80)
        self.mxf1.setEnabled(False)
        pos_layout.addWidget(self.mxf1)
        l=QLabel(" * x1 + ")
        l.setAlignment(Qt.AlignCenter)
        pos_layout.addWidget(l)
        self.mxf2 = QLineEdit("0")
        self.mxf2.textChanged.connect(self.fmxf2)
        self.mxf2.setFixedWidth(80)
        self.mxf2.setEnabled(False)
        pos_layout.addWidget(self.mxf2)
        pos_container.setLayout(pos_layout)
        right_layout.addWidget(pos_container, alignment=Qt.AlignCenter)
    
    def fmaccept(self):
        # global mfi, mfi_x, mfi_err
        self.msave_state()
        i = self.index
        if i not in self.mfi:
            self.mfi.append(i)
        if i in self.mfi_x:
            self.mfi_x.remove(i)
        if i in self.mfi_err:
            self.mfi_err.remove(i)
        self.mfitplot()



    def fmreject(self):
        # global mfi, mfi_x, mfi_err, mbreject, flmreject, mirej
        self.msave_state()
        i = self.index
        self.flmreject *= -1
        if self.flmreject == 1:
            self.mirej = i
            self.b_reject.setText('End Reject')
            self.b_reject.setStyleSheet("background-color: red;color: #EEE;")
        else:
            ti = sorted([i, self.mirej])
            for i in np.linspace(ti[0], ti[1], ti[1]-ti[0]+1, dtype=int):
                if i not in self.mfi_x:
                    self.mfi_x.append(i)
                if i in self.mfi:
                    self.mfi.remove(i)
                if i in self.mfi_err:
                    self.mfi_err.remove(i)
            self.b_reject.setText('Reject')
            self.b_reject.setStyleSheet("")
            self.mfitplot()
        
    def ffitcp(self):
        self.mfit()
        self.mfitplot()
    
    def fmwf1(self, event):
        if '' == self.mwf1.text():
            self.mwf1.setText('0.0')
            self.mwf1.selectAll()
        try:
            float(self.mwf1.text())
        except ValueError:
            self.mwf1.setText('0.0')
            self.mwf1.selectAll()
    
    def fmwf2(self, event):
        if '' == self.mwf2.text():
            self.mwf2.setText('0.0')
            self.mwf2.selectAll()
        try:
            float(self.mwf2.text())
        except ValueError:
            self.mwf2.setText('0.0')
            self.mwf2.selectAll()
        
    def fmaf1(self, event):
        if '' == self.maf1.text():
            self.maf1.setText('0.0')
            self.maf1.selectAll()
        try:
            float(self.maf1.text())
        except ValueError:
            self.maf1.setText('0.0')
            self.maf1.selectAll()
    
    def fmaf2(self, event):
        if '' == self.maf2.text():
            self.maf2.setText('0.0')
            self.maf2.selectAll()
        try:
            float(self.maf2.text())
        except ValueError:
            self.maf2.setText('0.0')
            self.maf2.selectAll()
    
    def fmxf1(self, event):
        if '' == self.mxf1.text() or '0.0' == self.mxf1.text():
            self.mxf1.setText('1')
            self.mxf1.selectAll()
        try:
            float(self.mxf1.text())
        except ValueError:
            self.mxf1.setText('1')
            self.mxf1.selectAll()
    
    def fmxf2(self, event):
        if '' == self.mxf2.text():
            self.mxf2.setText('0')
            self.mxf2.selectAll()
        try:
            float(self.mxf2.text())
        except ValueError:
            self.mxf2.setText('0')
            self.mxf2.selectAll()
    
    def fmposcst(self):
        self.flmposcst *= -1
        if self.flmposcst == 1:
            self.b_pos.setStyleSheet("background-color: purple;")  # 恢復預設樣式
            self.mxf1.setEnabled(True)
            self.mxf2.setEnabled(True)
        else:
            self.b_pos.setStyleSheet("")
            self.mxf1.setEnabled(False)
            self.mxf2.setEnabled(False)
    
    def fitpar1(self, result, lm1, lm2, lm3, lm4, lm5, lm6):
        s = putfitpar(result)
        x = s[0]
        h = s[1]
        w = s[2]
        for l, v in zip([lm1, lm2, lm3, lm4, lm5, lm6], [x, h, w, '', '', '']):
            # l.config(text=v)
            # l.config(anchor='w')
            l.setText(v)
            l.setAlignment(Qt.AlignLeft)
            

    def fitpar2(self, result, lm1, lm2, lm3, lm4, lm5, lm6):
        s = putfitpar(result)
        xr1, xr2 = float(self.mxf1.text()), float(self.mxf2.text())
        wr1, wr2 = float(self.mwf1.text()), float(self.mwf2.text())
        for i in s:
            '''preprocess the string to put values in the labels'''
            if 'x1*xr1+xr2' in i:
                if xr2>=0:
                    i = i.replace(' == \'x1*xr1+xr2\'', '='+str(xr1)+'*x1+'+str(xr2))
                else:
                    i = i.replace(' == \'x1*xr1+xr2\'', '='+str(xr1)+'*x1-'+str(-xr2))
            if 'x2*xr1+xr2' in i:
                if xr2>=0:
                    i = i.replace(' == \'x2*xr1+xr2\'', '='+str(xr1)+'*x2+'+str(xr2))
                else:
                    i = i.replace(' == \'x2*xr1+xr2\'', '='+str(xr1)+'*x2-'+str(-xr2))
            if "(x2-xr2) / xr1" in i:
                if xr2>=0:
                    i = i.replace(' == \'(x2-xr2) / xr1\'','=(x2-'+str(xr2) + ')/'+str(xr1))
                else:
                    i = i.replace(' == \'(x2-xr2) / xr1\'','=(x2+'+str(-xr2) + ')/'+str(xr1))
            if "(x1-xr2) / xr1" in i:
                if xr2>=0:
                    i = i.replace(' == \'(x1-xr2) / xr1\'','=(x1-'+str(xr2) + ')/'+str(xr1))
                else:
                    i = i.replace(' == \'(x1-xr2) / xr1\'','=(x1+'+str(-xr2) + ')/'+str(xr1))
            if 'w1/wr1*wr2' in i:
                i = i.replace(' == \'w1/wr1*wr2\'', '=w1/'+str(wr1)+'*'+str(wr2))
            if 'w2/wr1*wr2' in i:
                i = i.replace(' == \'w2/wr1*wr2\'', '=w2/'+str(wr1)+'*'+str(wr2))
            if 'x1:' in i:
                x1 = i
            if 'x2:' in i:
                x2 = i
            if 'h1:' in i:
                h1 = i
            if 'h2:' in i:
                h2 = i
            if 'w1:' in i:
                w1 = i
            if 'w2:' in i:
                w2 = i
        for l, v in zip([lm1, lm2, lm3, lm4, lm5, lm6], [x1, x2, h1, h2, w1, w2]):
            # l.config(text=v)
            # l.config(anchor='w')
            l.setText(v)
            l.setAlignment(Qt.AlignLeft)
    
    
    def fmrmv(self):
        # global mbrmv, flmrmv, mirmv, kmin, kmax, mfi, mfi_err, mfi_x, cki, mfp, mresult, smresult, smcst
        self.msave_state()
        i = self.index
        self.flmrmv *= -1
        if self.flmrmv == 1:
            self.mirmv = i
            self.b_remove.setText('End Remove')
            self.b_remove.setStyleSheet("background-color: red;")
        else:
            ti = sorted([i, self.mirmv])
            for i in np.linspace(ti[0], ti[1], ti[1]-ti[0]+1, dtype=int):
                self.mfp[i] = 1
                self.kmin[i], self.kmax[i] = self.klim.min[i], self.klim.max[i]
                if i not in self.mfi_x:
                    self.mfi_x.append(i)
                if i in self.mfi:
                    self.mfi.remove(i)
                if i in self.mfi_err:
                    self.mfi_err.remove(i)
                if i in self.cki:
                    self.cki.remove(i)
                self.mresult[i] = []
                try:
                    for j in range(6):
                        self.smresult[i][j] = 'nofit'
                        self.smcst[i][j] = 0
                except:
                    pass
            # mplfi()
            self.b_remove.setText('Remove')
            self.b_remove.setStyleSheet("")
            self.mfitplot()


    def fmcgl2(self):
        # global mbcgl2, kmin, kmax, flmcgl2, micgl2, mfp, mbcomp1, mbcomp2, flmcomp1, flmcomp2
        self.msave_state()
        self.b_comp1.setEnabled(True)
        self.b_comp2.setEnabled(True)
        self.flmcgl2 *= -1
        # mbcomp1.config(state='active')
        # mbcomp2.config(state='active')
        self.flmcomp1, self.flmcomp2 = -1, -1
        i = self.index
        if self.flmcgl2 == 1:
            self.micgl2 = i
            self.b_add2.setText('End Add 2 Peaks')
            self.b_add2.setStyleSheet("background-color: red;")
            # mbcgl2.config(text='End Add 2 Peaks', bg='red')
        else:
            ti = sorted([i, self.micgl2])
            for i in np.linspace(ti[0], ti[1], ti[1]-ti[0]+1, dtype=int):
                self.mfp[i] = 2
                if i not in self.mfi_x:
                    self.mfi_x.append(i)
                if i in self.mfi:
                    self.mfi.remove(i)
                if i in self.mfi_err:
                    self.mfi_err.remove(i)
            self.b_add2.setText('Add 2 Peaks')
            self.b_add2.setStyleSheet("")
            # mbcgl2.config(text='Add 2 Peaks', bg='white')
            self.mfitplot()

    def pack_fitpar(self, mresult):
        xr1, xr2 = float(self.mxf1.text()), float(self.mxf2.text())
        wr1, wr2 = float(self.mwf1.text()), float(self.mwf2.text())
        if len(self.smresult) > 1:
            o=self.smresult
            for ii, result in enumerate(mresult):
                try:
                    s = putfitpar(result)
                    for i in range(len(o[ii])):
                        o[ii][i]=""
                    for i in s:
                        '''preprocess the string to put values in the labels'''
                        if 'x1*xr1+xr2' in i:
                            if xr2>=0:
                                i = i.replace(' == \'x1*xr1+xr2\'', '='+str(xr1)+'*x1+'+str(xr2))
                            else:
                                i = i.replace(' == \'x1*xr1+xr2\'', '='+str(xr1)+'*x1-'+str(-xr2))
                        if 'x2*xr1+xr2' in i:
                            if xr2>=0:
                                i = i.replace(' == \'x2*xr1+xr2\'', '='+str(xr1)+'*x2+'+str(xr2))
                            else:
                                i = i.replace(' == \'x2*xr1+xr2\'', '='+str(xr1)+'*x2-'+str(-xr2))
                        if "(x2-xr2) / xr1" in i:
                            if xr2>=0:
                                i = i.replace(' == \'(x2-xr2) / xr1\'','=(x2-'+str(xr2) + ')/'+str(xr1))
                            else:
                                i = i.replace(' == \'(x2-xr2) / xr1\'','=(x2+'+str(-xr2) + ')/'+str(xr1))
                        if "(x1-xr2) / xr1" in i:
                            if xr2>=0:
                                i = i.replace(' == \'(x1-xr2) / xr1\'','=(x1-'+str(xr2) + ')/'+str(xr1))
                            else:
                                i = i.replace(' == \'(x1-xr2) / xr1\'','=(x1+'+str(-xr2) + ')/'+str(xr1))
                        if 'w1/wr1*wr2' in i:
                            i = i.replace(' == \'w1/wr1*wr2\'', '=w1/'+str(wr1)+'*'+str(wr2))
                        if 'w2/wr1*wr2' in i:
                            i = i.replace(' == \'w2/wr1*wr2\'', '=w2/'+str(wr1)+'*'+str(wr2))
                            
                        '''assign the values to the labels'''
                        if 'x:' in i:
                            o[ii][0]=i
                        if 'h:' in i:
                            o[ii][1]=i
                        if 'w:' in i:
                            o[ii][2]=i
                        if 'x1:' in i:
                            o[ii][0]=i
                        if 'x2:' in i:
                            o[ii][1]=i
                        if 'h1:' in i:
                            o[ii][2]=i
                        if 'h2:' in i:
                            o[ii][3]=i
                        if 'w1:' in i:
                            o[ii][4]=i
                        if 'w2:' in i:
                            o[ii][5]=i
                except:
                    pass
        else:
            o=[[]for i in range(len(mresult))]
            for ii,result in enumerate(mresult):
                try:
                    s = putfitpar(result)
                except:
                    s=[]
                    if self.mfp[ii]==2:
                        for i in ['x1: nofit','x2: nofit','h1: nofit','h2: nofit','w1: nofit','w2: nofit']:
                            s.append(i)
                    elif self.mfp[ii]==1:
                        for i in ['x: nofit','h: nofit','w: nofit','n1: nofit','n2: nofit','n3: nofit']:
                            s.append(i)
                for i in s:
                    if 'nofit' in i:
                        o[ii].append(i)
                    else:
                        '''preprocess the string to put values in the labels'''
                        if 'x1*xr1+xr2' in i:
                            if xr2>=0:
                                i = i.replace(' == \'x1*xr1+xr2\'', '='+str(xr1)+'*x1+'+str(xr2))
                            else:
                                i = i.replace(' == \'x1*xr1+xr2\'', '='+str(xr1)+'*x1-'+str(-xr2))
                        if 'x2*xr1+xr2' in i:
                            if xr2>=0:
                                i = i.replace(' == \'x2*xr1+xr2\'', '='+str(xr1)+'*x2+'+str(xr2))
                            else:
                                i = i.replace(' == \'x2*xr1+xr2\'', '='+str(xr1)+'*x2-'+str(-xr2))
                        if "(x2-xr2) / xr1" in i:
                            if xr2>=0:
                                i = i.replace(' == \'(x2-xr2) / xr1\'','=(x2-'+str(xr2) + ')/'+str(xr1))
                            else:
                                i = i.replace(' == \'(x2-xr2) / xr1\'','=(x2+'+str(-xr2) + ')/'+str(xr1))
                        if "(x1-xr2) / xr1" in i:
                            if xr2>=0:
                                i = i.replace(' == \'(x1-xr2) / xr1\'','=(x1-'+str(xr2) + ')/'+str(xr1))
                            else:
                                i = i.replace(' == \'(x1-xr2) / xr1\'','=(x1+'+str(-xr2) + ')/'+str(xr1))
                        if 'w1/wr1*wr2' in i:
                            i = i.replace(' == \'w1/wr1*wr2\'', '=w1/'+str(wr1)+'*'+str(wr2))
                        if 'w2/wr1*wr2' in i:
                            i = i.replace(' == \'w2/wr1*wr2\'', '=w2/'+str(wr1)+'*'+str(wr2))
                            
                        '''assign the values to the labels'''
                        if 'x:' in i:
                            o[ii].append(i)
                        if 'h:' in i:
                            o[ii].append(i)
                        if 'w:' in i:
                            o[ii].append(i)
                            o[ii].append('')
                            o[ii].append('')
                            o[ii].append('')
                        if 'x1:' in i:
                            o[ii].append(i)
                        if 'x2:' in i:
                            o[ii].append(i)
                        if 'h1:' in i:
                            o[ii].append(i)
                        if 'h2:' in i:
                            o[ii].append(i)
                        if 'w1:' in i:
                            o[ii].append(i)
                        if 'w2:' in i:
                            o[ii].append(i)
        return o
    
    def msave_state(self):
        # 保存當前狀態到撤銷堆疊，並清空重做堆疊
        self.smresult = self.pack_fitpar(self.mresult)
        state = {
            'mfi': self.mfi.copy(),
            'mfp': self.mfp.copy(),
            'kmin': self.kmin.copy(),
            'kmax': self.kmax.copy(),
            'maa1': self.maa1.copy(),
            'maa2': self.maa2.copy(),
            'smresult': copy.deepcopy(self.smresult),
            'smcst': self.smcst.copy(),
            'mfi_err': self.mfi_err.copy()
        }
        self.mundo_stack.append(state)
        self.mredo_stack.clear()


    def mundo(self):
        if self.mundo_stack:
            # global mfi, mfp, kmin, kmax, maa1, maa2, smresult, smcst, mfi_err, fdo
            # 從撤銷堆疊中彈出上一個狀態並恢復，並將當前狀態推入重做堆疊
            state = self.mundo_stack.pop()
            smresult = self.pack_fitpar(self.mresult)
            self.mredo_stack.append({
                'mfi': self.mfi.copy(),
                'mfp': self.mfp.copy(),
                'kmin': self.kmin.copy(),
                'kmax': self.kmax.copy(),
                'maa1': self.maa1.copy(),
                'maa2': self.maa2.copy(),
                'smresult': smresult.copy(),
                'smcst': self.smcst.copy(),
                'mfi_err': self.mfi_err.copy()
            })
            self.mfi = state['mfi']
            self.mfp = state['mfp']
            self.kmin = state['kmin']
            self.kmax = state['kmax']
            self.maa1 = state['maa1']
            self.maa2 = state['maa2']
            self.smresult = state['smresult']
            self.mresult = state['smresult']
            self.smcst = state['smcst']
            self.mfi_err = state['mfi_err']
            self.statusbar.showMessage("Undo")
            print("Undo")
            self.fdo=1
            self.mfitplot()
        else:
            self.statusbar.showMessage("No more actions to undo.")
            print("No more actions to undo.")
    
    def mredo(self):
        if self.mredo_stack:
            # global mfi, mfp, kmin, kmax, maa1, maa2, smresult, smcst, mfi_err, fdo
            # 從重做堆疊中彈出上一個狀態並恢復，並將當前狀態推入撤銷堆疊
            state = self.mredo_stack.pop()
            smresult = self.pack_fitpar(self.mresult)
            self.mundo_stack.append({
                'mfi': self.mfi.copy(),
                'mfp': self.mfp.copy(),
                'kmin': self.kmin.copy(),
                'kmax': self.kmax.copy(),
                'maa1': self.maa1.copy(),
                'maa2': self.maa2.copy(),
                'smresult': smresult.copy(),
                'smcst': self.smcst.copy(),
                'mfi_err': self.mfi_err.copy()
            })
            self.mfi = state['mfi']
            self.mfp = state['mfp']
            self.kmin = state['kmin']
            self.kmax = state['kmax']
            self.maa1 = state['maa1']
            self.maa2 = state['maa2']
            self.smresult = state['smresult']
            self.mresult = state['smresult']
            self.smcst = state['smcst']
            self.mfi_err = state['mfi_err']
            self.statusbar.showMessage("Redo")
            print("Redo")
            self.fdo=1
            self.mfitplot()
        else:
            self.statusbar.showMessage("No more actions to redo.")
            print("No more actions to redo.")

    def mouse_moved_event(self, event, *args):
        vb = self.plot.getViewBox()
        try:
            mouse_point = vb.mapSceneToView(event)
        except:
            mouse_point = vb.mapSceneToView(event.pos())
        # mouse_point = vb.mapSceneToView(event)
        # vb = self.plot.plotItem.vb
        
        # # 檢查事件位置是否在 ViewBox 內
        # if vb.sceneBoundingRect().contains(event.pos()):
        #     mouse_point = vb.mapSceneToView(event.pos())
        
        # if self.move_flag:
        #     self.reg.setRegion([mouse_point.x()-1, mouse_point.x()+1])
        self.mouse_x = mouse_point.x()
        if self.mmof == -1:
            if self.flmcomp1 == 1:
                self.mcpx1 = mouse_point.x()
                self.mcpy1 = mouse_point.y()
                self.mfitplot()
            elif self.flmcomp2 == 1:
                self.mcpx2 = mouse_point.x()
                self.mcpy2 = mouse_point.y()
                self.mfitplot()
        
        self.statusbar.setStyleSheet("font-size: 30px;")
        self.statusbar.showMessage(f"x={mouse_point.x():.2f}  y={mouse_point.y():.2f}")
        # pg.ViewBox.mouseMoveEvent(vb, event)

    def mouse_clicked_event(self, event):
        # vb = self.plot.plotItem.vb
        # mouse_point = vb.mapSceneToView(event.pos())
        # vb = self.plot.getViewBox()
        # mouse_point = vb.mapSceneToView(event.pos())
        # x = mouse_point.x()
        xlim = self.reg.getRegion()
        if self.mouse_x > xlim[0] and self.mouse_x < xlim[1]:
            self.mmof*=-1
        self.move_flag = False  # 可移除 move_flag 系列動作
        
        # self.statusbar.showMessage(f"Released at x={mouse_point.x():.2f}  y={mouse_point.y():.2f}", 3000)
        # pg.ViewBox.mouseReleaseEvent(vb, event)
        
    def on_slider_value_changed(self, value):
        self.index = value
        self.update_plot()
        self.update_plot_raw()
        self.statusbar.showMessage(f"Index: {self.index}")
        
        i = value
        self.mbgv = 0
        try:
            self.flmcomp1,self.flmcomp2 = -1, -1
            if self.mfp[i] == 2:
                self.b_comp1.setEnabled(True)
                self.b_comp2.setEnabled(True)
            else:
                self.b_comp1.setEnabled(False)
                self.b_comp2.setEnabled(False)
        except:
            pass
        self.mfitplot()
    
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
        for act in self.cmap_menu.actions():
            if act.text() == f"{cmap_name}":
                act.setChecked(True)
            else:
                act.setChecked(False)
                
        if hasattr(self, '_cmap_menu'):
            for act in self._cmap_menu.actions():
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
    
    def dragEnterEvent(self, event):
        # 檢查是否為檔案
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        # 獲取拖曳的檔案路徑
        files = [url.toLocalFile() for url in event.mimeData().urls()]
        if files:
            self.lfs = loadfiles(files, name='external')
            self.mdata = mfit_data()    # pos 改為 mpos
            self.ko, self.fev, self.rpos, self.ophi, self.fwhm, self.mpos, self.kmin, self.kmax, self.skmin, self.skmax, self.smaa1, self.smaa2, self.smfp, self.smfi, self.smresult, self.smcst, self.fpr, self.mdet = self.mdata.get()
            self.init_data()
            self.slider.setValue(0)
            self.fitm()
            #### mjob ####
            self.mresult = [[]for i in range(len(self.eV))]
            try:
                flsmresult = self.smresult
                flsmcst = self.smcst
            except:
                self.smcst=np.zeros(len(self.eV)*6).reshape(len(self.eV),6)
                self.smresult = [1]
            if self.mprfit == 1:
                self.fmfall()
            else:
                self.mfitplot()
            #### mjob ####
            
    def closeEvent(self, event):
        if hasattr(self, 'hist_widget_container'):
            if self.hist_widget_container.isVisible():
                self.hist_widget_container.close()
        flag = True
        smresult = self.pack_fitpar(self.mresult)
        for i, j in zip(smresult, self.smresult_original):
            if not np.array_equal(i, j):
                flag = False
                break
        if flag:
            pass
        else:
            self.mprend()
            if self.mpos.size == 0:
                event.accept()
                return
            reply = QMessageBox.question(
                self,
                'Confirm Exit',
                'Do you want to save before closing?',
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Cancel
            )        
            if reply == QMessageBox.Save:
                self.fmend()
                self.savemfit()
                if self.close_flag == 0:
                    event.accept()
                elif self.close_flag == 1:
                    event.ignore()
            elif reply == QMessageBox.Discard:
                # 不儲存直接關閉
                self.statusbar.showMessage("Closing without saving...")
                event.accept()
            else:
                # 取消關閉
                event.ignore()
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    hwnd = get_hwnd()
    p = argparse.ArgumentParser(description="Input Raw Data File Path")
    p.add_argument("-f", "--file", help="file path", type=str, nargs='+', required=False)
    args = p.parse_args()
    if args.file:
        file = args.file
    else:
        file = QFileDialog.getOpenFileName(None, "Open Data File", cdir, "HDF5 Files (*.h5 *.hdf5);;NPZ Files (*.npz);;JSON Files (*.json);;TXT Files (*.txt)")[0]
    if file:
        win = main(file, hwnd)
        win.show()
        sys.exit(app.exec_())
