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
    path = os.path.join(os.path.dirname(__file__), 'simulated_R1_15.0_R2_0.h5')
    print('file check:',os.path.exists(path))
    # 檢查檔案大小（Git LFS 指標檔案通常很小，<1KB）
    file_size = os.path.getsize(path)
    if file_size < 1024:  # 小於 1KB
        with open(path, 'rb') as f:
            header = f.read(100)
            if b'version https://git-lfs.github.com' in header:
                raise ValueError(
                    f"H5 file appears to be a Git LFS pointer file.\n"
                    f"File size: {file_size} bytes\n"
                    f"Please run 'git lfs pull' to download the actual file."
                )

    # 驗證 HDF5 檔案簽名
    with open(path, 'rb') as f:
        signature = f.read(8)
        # HDF5 檔案應該以 \x89HDF\r\n\x1a\n 開頭
        if not signature.startswith(b'\x89HDF'):
            raise ValueError(
                f"Invalid HDF5 file signature.\n"
                f"Expected: \\x89HDF..., Got: {signature[:4]}\n"
                f"File may be corrupted or is a Git LFS pointer."
            )  
    lfs = loadfiles(f"{path}", mode ='eager')
    assert isinstance(lfs, FileSequence)
    assert isinstance(lfs.get(0), xr.DataArray)

def test_load_h5():
    path = os.path.join(os.path.dirname(__file__), 'simulated_R1_15.0_R2_0.h5')
    print('file check:',os.path.exists(path))
    data = load_h5(f"{path}")
    assert isinstance(data, xr.DataArray)

def test_load_json():
    path = os.path.join(os.path.dirname(__file__), 'UPSPE20_2_test_1559#id#3cf2122d.json')
    print('file check:',os.path.exists(path))
    data = load_json(f"{path}")
    assert isinstance(data, xr.DataArray)