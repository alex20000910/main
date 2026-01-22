import pytest
import os, inspect
import json
import tkinter as tk
from tkinter import filedialog as fd
import io
from base64 import b64decode
import queue
import threading
import warnings
import sys, shutil
from ctypes import windll
import copy
import gc
from tkinter import messagebox, colorchooser
# import ttkbootstrap as ttk
import subprocess
import argparse
import importlib
from typing import override, Literal
    
VERSION = sys.version.split()[0]
VERSION = int(''.join(VERSION.split('.')))

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
# if __name__ == '__main__':
from matplotlib.colors import LinearSegmentedColormap
# from matplotlib.widgets import SpanSelector
# from matplotlib.widgets import RectangleSelector
import matplotlib as mpl
# from matplotlib.widgets import Cursor
# from matplotlib.widgets import Slider
import numpy as np
import xarray as xr
import h5py
from PIL import Image, ImageTk
# if __name__ == '__main__':
# from scipy.optimize import curve_fit
from scipy.signal import hilbert
# from lmfit import Parameters, Minimizer
from lmfit.printfuncs import alphanumeric_sort, gformat, report_fit
import tqdm
import win32clipboard
# if __name__ == '__main__':
import originpro as op
from cv2 import Laplacian, GaussianBlur, CV_64F, CV_32F
import psutil
# if VERSION >= 3130:
import google_crc32c    # for numcodecs
# if __name__ == '__main__':
import cpuinfo
import zarr
import PyQt5
import pyqtgraph
from tkinterdnd2 import DND_FILES, TkinterDnD

tdir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')
sys.path.append(tdir)
sys.path.append(os.path.dirname(tdir))
# sys.path.append(os.path.join(os.path.dirname(os.path.dirname(tdir)), '.MDC_cut'))
from MDC_cut_utility import *
from tool.loader import loadfiles, mloader, eloader, tkDnD_loader, file_loader, data_loader, load_h5, load_json, load_npz, load_txt
from tool.spectrogram import spectrogram, lfs_exp_casa
from tool.util import laplacian_filter  # for originpro: from MDC_cut import *
# if __name__ == '__main__':
from tool.util import app_param, MDC_param, EDC_param, Button, MenuIconManager, ToolTip_util, IconManager, origin_util, motion, plots_util, exp_util
from tool.SO_Fitter import SO_Fitter
from tool.CEC import CEC, call_cec
from tool.VolumeSlicer import wait
from tool.window import AboutWindow, EmodeWindow, ColormapEditorWindow, c_attr_window, c_name_window, c_excitation_window, c_description_window, VersionCheckWindow, CalculatorWindow, Plot1Window, Plot1Window_MDC_curves, Plot1Window_Second_Derivative, Plot3Window

def test_loadfiles():
    path = []
    path.append(os.path.join(os.path.dirname(__file__), 'simulated_R1_15.0_R2_0.h5'))
    path.append(os.path.join(os.path.dirname(__file__), 'UPSPE20_2_test_1559#id#3cf2122d.json'))
    lfs = loadfiles(path)
    assert isinstance(lfs, FileSequence)
    assert isinstance(lfs.get(0), xr.DataArray)
    assert isinstance(lfs.get(1), xr.DataArray)

class tkDnD(tkDnD_loader):
    """
    A simple wrapper class to add drag-and-drop functionality to a Tkinter window using tkinterdnd2.
    
    Attributes:
        root (TkinterDnD.Tk): The main Tkinter window created by tkinterdnd2.
    """
    def __init__(self, root: tk.Misc):
        super().__init__(root)
    
    @override
    def load(self, drop: bool=True, files: tuple[str] | Literal[''] =''):
        pass

def test_tkDnD():
    path = []
    path.append(os.path.join(os.path.dirname(__file__), 'simulated_R1_15.0_R2_0.h5'))
    path.append(os.path.join(os.path.dirname(__file__), 'UPSPE20_2_test_1559#id#3cf2122d.json'))
    files = tkDnD.load_raw(path)
    assert isinstance(files, list)

@pytest.fixture
def tk_environment():
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    frame = tk.Frame(root)
    
    yield root, frame
    
    # 確保所有子視窗都被關閉
    try:
        for widget in root.winfo_children():
            try:
                widget.destroy()
            except:
                pass
    except:
        pass
    
    # 更新事件循環
    try:
        root.update_idletasks()
        root.update()
    except:
        pass
    
    # 最後銷毀 root
    try:
        root.quit()
        root.destroy()
    except:
        pass

def test_spectrogram(tk_environment):
    g, frame = tk_environment
    path = []
    path.append(os.path.join(os.path.dirname(__file__), 'simulated_R1_15.0_R2_0.h5'))
    path.append(os.path.join(os.path.dirname(__file__), 'UPSPE20_2_test_1559#id#3cf2122d.json'))
    app_pars = app_param(hwnd=None, scale=1, dpi=96, bar_pos='bottom', g_mem=8)
    s = spectrogram(path=path, app_pars=app_pars)
    s.plot(g)
    s.cf_up()
    s.cf_down()
    s.ups()
    assert isinstance(s.name, str)
    assert isinstance(s.data, xr.DataArray)

def test_VolumeSlicer(tk_environment):
    from tool.VolumeSlicer import VolumeSlicer
    g, frame = tk_environment
    odpi=g.winfo_fpixels('1i')
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src', 'odpi')
    with open(path, 'w') as f:
        f.write(f'{odpi}')  #for RestrictedToplevel
        f.close()
    path = os.path.join(os.path.dirname(__file__), 'simulated_R1_15.0_R2_0.h5')
    data = load_h5(path)
    odata = []
    r1 = np.linspace(10, 20, 11)
    for i in range(len(r1)):
        odata.append(data)
    odataframe = np.stack([i.transpose() for i in odata], axis=0, dtype=np.float32)
    
    ev, phi = odata[0].indexes.values()
    app_pars = app_param(hwnd=None, scale=1, dpi=96, bar_pos='bottom', g_mem=8)
    vs = VolumeSlicer(parent=frame, path=path, volume=odataframe, x=phi, y=r1, ev=ev, g=g, app_pars=app_pars)
    vs.change_mode()
    assert vs.surface.shape ==(vs.density, vs.density)
    assert vs.surface.dtype == np.float32
    assert vs.surface.flatten().max() > 0
    xlim=[10, 20]
    ylim=[-10, 10]
    vs.cdensity = int((vs.xmax-vs.xmin)//2e-3)
    r1 = np.linspace(xlim[0], xlim[1], int(vs.cdensity/180*(xlim[1]-xlim[0])+1))
    phi = np.linspace(ylim[0], ylim[1], int(vs.cdensity/180*(ylim[1]-ylim[0])))
    ev = ev[-1]
    x = np.sqrt(2*vs.m*vs.e*ev)/vs.hbar*10**-10*np.sin(r1[:, None]/180*np.pi) * np.cos(phi[None, :]/180*np.pi)  # x: r1, y: phi, at r2=0
    y = np.sqrt(2*vs.m*vs.e*ev)/vs.hbar*10**-10*np.sin(phi[None, :]/180*np.pi)
    txlim, tylim = [np.min(x), np.max(x)], [np.min(y), np.max(y)]
    data = vs.ovolume[:, vs.slim[0]:vs.slim[1]+1, -1]
    density = vs.cdensity
    data = vs.k_map(data, density=density, xlim=xlim, ylim=ylim, kxlim=txlim, kylim=tylim, ev=ev)
    shape = (int(density/(vs.xmax-vs.xmin)*(txlim[1]-txlim[0])), int(density/(vs.ymax-vs.ymin)*(tylim[1]-tylim[0])))
    assert data.dtype == np.float32
    assert data.T.shape == shape
    vs.det_core_num()
    vs.symmetry()
    vs.symmetry_(6)
    vs.t_cut_job_y()
    vs.t_cut_job_x()

def test_CEC(tk_environment):
    from tool.MDC_Fitter import get_file_from_github
    g, frame = tk_environment
    app_pars = app_param(hwnd=None, scale=1, dpi=96, bar_pos='bottom', g_mem=8)
    tg = wait(g, app_pars)
    tg.text('Preparing sample data...')
    path = rf"simulated_R1_15.0_R2_0.h5"
    tpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test_data_temp', rf"simulated_R1_15.0_R2_0.h5")
    if os.path.exists(tpath)==False:
        get_file_from_github(r"https://github.com/alex20000910/main/blob/main/test_data/"+path, tpath)
    files = []
    files.append(os.path.join(os.path.dirname(__file__), 'simulated_R1_15.0_R2_0.h5'))
    files.append(os.path.join(os.path.dirname(__file__), 'simulated_R1_15.1_R2_0.h5'))
    tg.done()
    tg = wait(g, app_pars)
    tg.text('Loading sample data...')
    lfs = loadfiles(files)
    tg.done()
    t_cec = CEC(g, lfs.path, cmap='viridis', app_pars=app_pars)

def test_interp():
    y = interp(1.5, [1, 2], [2, 3])
    assert y == 2.5
    y_array = interp([1.5, 2.5], [1, 2], [2, 3])
    assert np.allclose(y_array, [2.5, 3.5])

def test_get_bar_pos():
    pos = get_bar_pos()
    assert isinstance(pos, str)

def test_get_hwnd():
    from tool.MDC_Fitter import get_hwnd
    hwnd = get_hwnd()
    assert isinstance(hwnd, int)

