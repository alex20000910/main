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
if VERSION < 3130:
    REQUIREMENTS = ["numpy==1.26.4",
    "opencv-python==4.10.0.84",
    "matplotlib==3.10.5",
    "xarray==2025.7.1",
    "h5py==3.14.0",
    "Pillow==11.3.0",
    "scipy==1.16.1",
    "lmfit==1.3.4",
    "tqdm==4.67.1",
    "pywin32==311",
    "originpro==1.1.13",
    "py-cpuinfo==9.0.0",
    "psutil==7.0.0",
    "zarr==3.1.1",
    "PyQt5==5.15.11",
    "pyqtgraph==0.13.7",
    "tkinterdnd2==0.4.3"
    ]
else:
    REQUIREMENTS = ["numpy==2.2.6",
    "opencv-python==4.12.0.88",
    "matplotlib==3.10.5",
    "xarray==2025.7.1",
    "h5py==3.14.0",
    "Pillow==11.3.0",
    "scipy==1.16.1",
    "lmfit==1.3.4",
    "tqdm==4.67.1",
    "pywin32==311",
    "originpro==1.1.13",
    "py-cpuinfo==9.0.0",
    "psutil==7.0.0",
    "zarr==3.1.1",
    "PyQt5==5.15.11",
    "pyqtgraph==0.13.7",
    "tkinterdnd2==0.4.3",
    "google-crc32c==1.8.0"  # for numcodecs
    ]

def restart():
    if os.name == 'nt':
        os.system('python -W ignore::SyntaxWarning -W ignore::UserWarning "'+os.path.abspath(inspect.getfile(inspect.currentframe()))+'"')
    elif os.name == 'posix':
        try:
            os.system('python3 -W ignore::SyntaxWarning -W ignore::UserWarning "'+os.path.abspath(inspect.getfile(inspect.currentframe()))+'"')
        except:
            os.system('python -W ignore::SyntaxWarning -W ignore::UserWarning "'+os.path.abspath(inspect.getfile(inspect.currentframe()))+'"')

def install(s: str = ''):
    print('Some Modules Not Found')
    a = input('pip install all the missing modules ???\nProceed (Y/n)? ')
    if a.lower() == 'y':
        if s == '':
            try:
                for i in REQUIREMENTS:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", i])
            except subprocess.CalledProcessError:
                for i in REQUIREMENTS:
                    subprocess.check_call([sys.executable, "-m", "pip3", "install", "--user", i])
        else:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", s])
            except subprocess.CalledProcessError:
                subprocess.check_call([sys.executable, "-m", "pip3", "install", "--user", s])
    else:
        quit()

try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    if __name__ == '__main__':
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
    if __name__ == '__main__':
        # from scipy.optimize import curve_fit
        from scipy.signal import hilbert
        # from lmfit import Parameters, Minimizer
        from lmfit.printfuncs import alphanumeric_sort, gformat, report_fit
    import tqdm
    import win32clipboard
    if __name__ == '__main__':
        import originpro as op
    from cv2 import Laplacian, GaussianBlur, CV_64F, CV_32F
    import psutil
    if VERSION >= 3130:
        import google_crc32c    # for numcodecs
    if __name__ == '__main__':
        import cpuinfo
        import zarr
        import PyQt5
        import pyqtgraph
        from tkinterdnd2 import DND_FILES, TkinterDnD
except ModuleNotFoundError:
    install()
    restart()
    quit()

try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
    from MDC_cut_utility import *
    from tool.loader import loadfiles, mloader, eloader, tkDnD_loader, file_loader, data_loader, load_h5, load_json, load_npz, load_txt
    from tool.spectrogram import spectrogram, lfs_exp_casa
    from tool.util import laplacian_filter  # for originpro: from MDC_cut import *
    if __name__ == '__main__':
        from tool.util import app_param, MDC_param, EDC_param, Button, MenuIconManager, ToolTip_util, IconManager, origin_util, motion, plots_util, exp_util
        from tool.SO_Fitter import SO_Fitter
        from tool.CEC import CEC, call_cec
        from tool.VolumeSlicer import wait
        from tool.window import AboutWindow, EmodeWindow, ColormapEditorWindow, c_attr_window, c_name_window, c_excitation_window, c_description_window, VersionCheckWindow, CalculatorWindow, Plot1Window, Plot1Window_MDC_curves, Plot1Window_Second_Derivative, Plot3Window
except ImportError as e:
    print(e)
    
path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_data', 'simulated_R1_5.0_R2_0.h5'))
lfs = loadfiles(path)
assert isinstance(lfs, FileSequence)