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
    quit()

try:
    tdir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')
    sys.path.append(tdir)
    sys.path.append(os.path.dirname(tdir))
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(tdir)), '.MDC_cut'))
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

path = os.path.join('.', 'simulated_R1_15.0_R2_0.h5')

def test_loadfiles():
    lfs = loadfiles(["simulated_R1_15.0_R2_0.h5"], mode ='eager')
    assert isinstance(lfs, FileSequence)
    assert isinstance(lfs.get(0), xr.DataArray)

def test_load_h5():
    data = load_h5(path)
    assert isinstance(data, xr.DataArray)