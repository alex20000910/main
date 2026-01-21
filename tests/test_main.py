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
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(tdir)), '.MDC_cut'))
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

def test_spectrogram():
    path = os.path.join(os.path.dirname(__file__), 'simulated_R1_15.0_R2_0.h5')
    s = spectrogram(path=path)
    assert s.name == 'simulated_R1_15.0_R2_0'
    assert isinstance(s.data, xr.DataArray)

def test_k_map():
    from tool.VolumeSlicer import k_map
    e = 1.602e-19
    hbar = 6.626e-34/2/np.pi
    phi = np.linspace(-30, 30, 21)
    r1 = np.linspace(60, 40, 7)
    ev = 21.2
    k = (ev*e*(2*9.11e-31))**0.5/hbar*1e-10
    x, y = np.meshgrid(r1, phi)
    x, y = k*np.sin(np.radians(x))*np.cos(np.radians(y)), k*np.sin(np.radians(y))
    
    density = 50
    kx = np.linspace(-2.5, 2.5, density, endpoint=False)
    ky = np.linspace(-2.5, 2.5, density, endpoint=False)
    kx += (kx[1]-kx[0])/2
    ky += (ky[1]-ky[0])/2
    kx, ky = np.meshgrid(kx, ky)
    data = x*0+1
    data = k_map(data, density=density+1, xlim=[40, 60], ylim=[-30, 30], kxlim=[x.min(), x.max()], kylim=[y.min(), y.max()], ev=ev)
    assert data.shape == (density+1, density+1)