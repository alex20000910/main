# MDC cut GUI
__version__ = "8.1"
__release_date__ = "2025-11-21"
# Name                     Version          Build               Channel
# asteval                   1.0.6                    pypi_0    pypi
# bzip2                     1.0.8                h2bbff1b_6  
# ca-certificates           2025.9.9             haa95532_0  
# colorama                  0.4.6                    pypi_0    pypi
# contourpy                 1.3.3                    pypi_0    pypi
# crc32c                    2.7.1                    pypi_0    pypi
# cycler                    0.12.1                   pypi_0    pypi
# dill                      0.4.0                    pypi_0    pypi
# donfig                    0.8.1.post1              pypi_0    pypi
# expat                     2.7.1                h8ddb27b_0
# fonttools                 4.60.1                   pypi_0    pypi
# h5py                      3.14.0                   pypi_0    pypi
# kiwisolver                1.4.9                    pypi_0    pypi
# libffi                    3.4.4                hd77b12b_1
# libmpdec                  4.0.0                h827c3e9_0
# libzlib                   1.3.1                h02ab6af_0
# lmfit                     1.3.4                    pypi_0    pypi
# matplotlib                3.10.5                   pypi_0    pypi
# numcodecs                 0.16.3                   pypi_0    pypi
# numpy                     2.2.6                    pypi_0    pypi
# opencv-python             4.12.0.88                pypi_0    pypi
# openssl                   3.0.18               h543e019_0
# originext                 1.2.4                    pypi_0    pypi
# originpro                 1.1.13                   pypi_0    pypi
# packaging                 25.0                     pypi_0    pypi
# pandas                    2.3.3                    pypi_0    pypi
# pillow                    11.3.0                   pypi_0    pypi
# pip                       25.2               pyhc872135_0
# psutil                    7.0.0                    pypi_0    pypi
# py-cpuinfo                9.0.0                    pypi_0    pypi
# pyparsing                 3.2.5                    pypi_0    pypi
# pyqt5                     5.15.11                  pypi_0    pypi
# pyqt5-qt5                 5.15.2                   pypi_0    pypi
# pyqt5-sip                 12.17.1                  pypi_0    pypi
# pyqtgraph                 0.13.7                   pypi_0    pypi
# python                    3.13.5          h286a616_100_cp313
# python-dateutil           2.9.0.post0              pypi_0    pypi
# python_abi                3.13                    1_cp313
# pytz                      2025.2                   pypi_0    pypi
# pywin32                   311                      pypi_0    pypi
# pyyaml                    6.0.3                    pypi_0    pypi
# scipy                     1.16.1                   pypi_0    pypi
# setuptools                78.1.1          py313haa95532_0
# six                       1.17.0                   pypi_0    pypi
# sqlite                    3.50.2               hda9a48d_1
# tk                        8.6.15               hf199647_0
# tkinterdnd2               0.4.3                    pypi_0    pypi
# tqdm                      4.67.1                   pypi_0    pypi
# typing-extensions         4.15.0                   pypi_0    pypi
# tzdata                    2025.2                   pypi_0    pypi
# ucrt                      10.0.22621.0         haa95532_0
# uncertainties             3.2.3                    pypi_0    pypi
# vc                        14.3                h2df5915_10
# vc14_runtime              14.44.35208         h4927774_10
# vs2015_runtime            14.44.35208         ha6b5a95_10
# wheel                     0.45.1          py313haa95532_0
# xarray                    2025.7.1                 pypi_0    pypi
# xz                        5.6.4                h4754444_1
# zarr                      3.1.1                    pypi_0    pypi
# zlib                      1.3.1                h02ab6af_0

# import tracemalloc
# tracemalloc.start()
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
    "pyqtgraph==0.13.7"
    "tkinterdnd2==0.4.3",
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
    "tkinterdnd2==0.4.3"
    ]

cdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
if os.name == 'nt':
    cdir = cdir[0].upper() + cdir[1:]
app_name = os.path.basename(inspect.stack()[0].filename).removesuffix('.py')

def get_bar_pos():
    if os.name == 'nt':  # only execute on Windows OS
        import ctypes
        from ctypes import wintypes
        class APPBARDATA(ctypes.Structure):
            _fields_ = [
                ('cbSize', wintypes.DWORD),
                ('hWnd', wintypes.HWND),
                ('uCallbackMessage', wintypes.UINT),
                ('uEdge', wintypes.UINT),
                ('rc', wintypes.RECT),
                ('lParam', wintypes.LPARAM),
            ]
        ABM_GETTASKBARPOS = 0x00000005
        def get_taskbar_position():
            appbar = APPBARDATA()
            appbar.cbSize = ctypes.sizeof(APPBARDATA)
            result = ctypes.windll.shell32.SHAppBarMessage(ABM_GETTASKBARPOS, ctypes.byref(appbar))
            if not result:
                return None
            edge = appbar.uEdge
            positions = {0: 'left', 1: 'top', 2: 'right', 3: 'bottom'}
            return positions.get(edge, 'unknown')
        
        bar_pos = get_taskbar_position()

    elif os.name == 'posix':
        def get_dock_position():
            script = 'tell application "System Events" to get the properties of the dock preferences'
            result = subprocess.check_output(['osascript', '-e', script])
            if b'left' in result:
                return 'left'
            elif b'bottom' in result:
                return 'bottom'
            elif b'right' in result:
                return 'right'
            else:
                return 'unknown'
        bar_pos = get_dock_position()
    return bar_pos

bar_pos = get_bar_pos()

def to_raw_url(url: str) -> str:
    # 將 github blob 連結轉成 raw 連結
    if "github.com" in url and "/blob/" in url:
        return url.replace("https://github.com/", "https://raw.githubusercontent.com/").replace("/blob/", "/")
    return url

def download(url: str, out_path: str, token: str = None) -> None:
    try:
        import requests
    except Exception:
        requests = None

    headers = {}
    if token:
        headers["Authorization"] = f"token {token}"

    url = to_raw_url(url)

    if requests:
        with requests.get(url, headers=headers, stream=True) as r:
            r.raise_for_status()
            total = r.headers.get("Content-Length")
            if total and total.isdigit():
                total = int(total)
            with open(out_path, "wb") as f:
                downloaded = 0
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                # optionally you could print progress here
    else:
        # fallback to standard library
        import urllib.request
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as resp, open(out_path, "wb") as out:
            out.write(resp.read())

def get_file_from_github(url: str, out_path: str, token: str = None):
    # p = argparse.ArgumentParser(description="Download a file from a GitHub URL")
    # p.add_argument("url", help="GitHub file URL (github.com/.../blob/... or raw.githubusercontent.com/... )")
    # p.add_argument("-o", "--output", help="Output filename or path. If omitted, uses the filename from the URL.")
    # p.add_argument("--token", help="GitHub token for private repos (optional)", default=None)
    # args = p.parse_args()

    # url = args.url.strip()
    args = argparse.Namespace(output=out_path, token=None)
    raw = to_raw_url(url)
    if not args.output:
        # try to extract filename
        parts = raw.rstrip("/").split("/")
        if len(parts) >= 1:
            filename = parts[-1]
        else:
            print("Please use -o to specify the output filename.", file=sys.stderr)
        out_path = filename
    else:
        out_path = args.output

    # ensure output dir exists
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    try:
        download(url, out_path, args.token)
    except Exception as e:
        print("Failed to download source file:", e, file=sys.stderr)
        print("\033[35mPlease ensure the Network is connected. \033[0m", file=sys.stderr)

def get_src(ver=False):
    url = [r"https://github.com/alex20000910/main/blob/update/MDC_cut.py",
           r"https://github.com/alex20000910/main/blob/update/src/viridis_2D.otp",
           r"https://github.com/alex20000910/main/blob/update/src/MDC_cut_utility.py",
           r"https://github.com/alex20000910/main/blob/update/src/tool/__init__.py",
           r"https://github.com/alex20000910/main/blob/update/src/tool/loader.py",
           r"https://github.com/alex20000910/main/blob/update/src/tool/spectrogram.py",
           r"https://github.com/alex20000910/main/blob/update/src/tool/SO_Fitter.py",
           r"https://github.com/alex20000910/main/blob/update/src/tool/VolumeSlicer.py",
           r"https://github.com/alex20000910/main/blob/update/src/tool/CEC.py",
           r"https://github.com/alex20000910/main/blob/update/src/tool/DataViewer.py",
           r"https://github.com/alex20000910/main/blob/update/src/tool/MDC_Fitter.py",
           r"https://github.com/alex20000910/main/blob/update/src/tool/EDC_Fitter.py",
           r"https://github.com/alex20000910/main/blob/update/src/tool/window.py"]
    for i, v in enumerate(url):
        if i < 3:
            out_path = os.path.join(cdir, '.MDC_cut', os.path.basename(v))
        else:
            out_path = os.path.join(cdir, '.MDC_cut', 'tool', os.path.basename(v))
        get_file_from_github(v, out_path)
        if ver:
            break

# set up .MDC_cut folder
os.chdir(cdir)
if not os.path.exists('.MDC_cut'):
    os.makedirs('.MDC_cut')
os.system('attrib +h +s .MDC_cut')
sys.path.append(os.path.join(cdir, '.MDC_cut'))

# upgrade check
v_check_path = os.path.join(cdir, '.MDC_cut', 'version.check')
if os.path.exists(v_check_path):
    with open(v_check_path, mode='r') as f:
        ver = f.read().strip()
    if ver != __version__:
        get_src()
        with open(v_check_path, mode='w') as f:
            f.write(__version__)
else:
    get_src()
    with open(v_check_path, mode='w') as f:
        f.write(__version__)

# clean temp folders
if __name__ == '__main__':
    if os.path.exists(os.path.join(cdir, '.MDC_cut', 'cut_temp_save')):
        shutil.rmtree(os.path.join(cdir, '.MDC_cut', 'cut_temp_save'))
    if os.path.exists(os.path.join(cdir, '.MDC_cut', 'cube_temp_save')):
        shutil.rmtree(os.path.join(cdir, '.MDC_cut', 'cube_temp_save'))

# make sure pip is installed
try:
    os.chdir(os.path.join(cdir, '.MDC_cut'))
    if os.path.exists('pip_check.txt')==0:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "--version"])
            with open('pip_check.txt', 'w', encoding = 'utf-8') as f:
                f.write('pip found')
                f.close()
        except subprocess.CalledProcessError:
            try:
                if os.name == 'nt':
                    print('pip not found\nOS: Windows\nInstalling pip...')
                    os.system('python -m ensurepip')    #install pip
                    os.system('python -W ignore::SyntaxWarning -W ignore::UserWarning "'+os.path.abspath(inspect.getfile(inspect.currentframe()))+'"')  #restart the script to ensure pip works without potential errors
                elif os.name == 'posix':
                    print('pip not found\nOS: Linux or MacOS\nInstalling pip...')
                    try:    #python3 if installed
                        os.system('python3 -m ensurepip')   #install pip
                        os.system('python3 -W ignore::SyntaxWarning -W ignore::UserWarning "'+os.path.abspath(inspect.getfile(inspect.currentframe()))+'"')   #restart the script to ensure pip works without potential errors
                    except: #python2.7(default in MacOS)
                        os.system('python -m ensurepip')
                        os.system('python -W ignore::SyntaxWarning -W ignore::UserWarning "'+os.path.abspath(inspect.getfile(inspect.currentframe()))+'"')
                with open('pip_check.txt', 'w', encoding = 'utf-8') as f:
                    f.write('pip found')
                    f.close()
            except Exception as e:
                print(f"An error occurred: {e}")
            quit()  #end the current script
except:
    pass

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
        try:
            for i in REQUIREMENTS:
                subprocess.check_call([sys.executable, "-m", "pip", "install", i])
        except subprocess.CalledProcessError:
            for i in REQUIREMENTS:
                subprocess.check_call([sys.executable, "-m", "pip3", "install", i])
    else:
        quit()

def pool_protect(func):
    def wrapper(*args, **kwargs):
        if __name__ == "__main__":
            func(*args, **kwargs)
    return wrapper

def set_globals(var, glob):
    if var is not None:
        globals()[glob] = var

def init_globals(var):
    try: globals()[var]
    except KeyError:
        globals()[var] = None

try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    if __name__ == '__main__':
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.widgets import SpanSelector
        from matplotlib.widgets import RectangleSelector
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
    from cv2 import Laplacian, GaussianBlur, CV_64F
    import psutil
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
    from MDC_cut_utility import *
    from tool.loader import loadfiles, mloader, eloader, tkDnD_loader, file_loader, data_loader, load_h5, load_json, load_npz, load_txt
    from tool.spectrogram import spectrogram, lfs_exp_casa
    if __name__ == '__main__':
        from tool.SO_Fitter import SO_Fitter
        from tool.CEC import CEC, call_cec
        from tool.window import AboutWindow, EmodeWindow, ColormapEditorWindow, c_attr_window, c_name_window, c_excitation_window, c_description_window, VersionCheckWindow, CalculatorWindow, Plot1Window, Plot1Window_MDC_curves, Plot1Window_Second_Derivative, Plot3Window
except ImportError:
    print('Some source files missing. Downloading...')
    get_src()

if __name__ == '__main__':
    pid = os.getpid()
    # g_mem = psutil.virtual_memory().available
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)
h=6.62607015*10**-34
m=9.10938356*10**-31
mp, ep, mf, ef = 1, 1, 1, 1
fk = []
fev = []
fit_so = None   # for SO_Fitter instance checking

if __name__ == '__main__':
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
            load(drop, files)
    
    class pr_load(data_loader):
        def __init__(self, data: xr.DataArray):
            super().__init__(menu1, menu2, menu3, in_fit, b_fit, l_path, info, cdir, lfs)
            self.pr_load(data)

        @override
        def pars(self):
            for i, j in zip(['name','dvalue','e_photon','description','dpath', 'ev', 'phi'], [self.name, self.dvalue, self.e_photon, self.description, self.dpath, self.ev, self.phi]):
                set_globals(j, i)
            
    class main_loader(file_loader):
        def __init__(self, files: tuple[str]|Literal['']):
            super().__init__(files, path, value3.get(), lfs, g, app_pars, st, limg, img, b_name, b_excitation, b_desc, koffset, k_offset, fr_tool, b_tools, l_name, scale)
        
        @property
        @override
        def name(self) -> str:
            return name
        
        @override
        def call_cec(self, g, lfs) -> FileSequence:
            return call_cec(g, lfs)

        @override
        def pr_load(self, data):
            pr_load(data)
        
        @override
        def change_file(self, *args):
            change_file()
            
        @override
        def tools(self, *args):
            tools()
        
        @override
        def set_k_offset(self):
            if 'ko' in globals():
                self.k_offset.set(ko)
            else:
                self.k_offset.set('0')
        
        @override
        def pars(self):
            for i, j in zip(['data', 'rdd', 'fpr', 'lfs', 'npzf', 'b_tools', 'l_name', 'nlist', 'namevar'], [self.data, self.rdd, self.fpr, self.lfs, self.npzf, self.b_tools, self.l_name, self.nlist, self.namevar]):
                set_globals(j, i)
    
    class G_emode(EmodeWindow):
        def __init__(self):
            super().__init__(g, vfe, scale)
            
        @override
        def save_fe(self, *args):
            global gfe, vfe
            try:
                vfe=float(self.fe_in.get())
                self.destroy()
                plot1()
                plot2()
                plot3()
            except:
                messagebox.showwarning("Warning","Invalid Input\n"+str(sys.exc_info()[1]))
                self.destroy()
                gfe = G_emode()
    
    class c_attr(c_attr_window):
        def __init__(self, parent: tk.Misc | None = None, dpath: str='', attr: float|str='', scale: float=1.0):
            '''
            Parameters
            ----------
            attr : float | str
                Attribute value to be set (e.g., e_photon, name, description).
            '''
            super().__init__(parent, dpath, attr, scale)
        
        @override
        def pars(self):
            set_globals(self.data, 'data')
        
        @override
        def pr_load(self, data):
            pr_load(data)
        
        @override
        def load_h5(self, path: str):
            return load_h5(path)
        
        @override
        def load_json(self, path: str):
            return load_json(path)
        
        @override
        def load_npz(self, path: str):
            return load_npz(path)

    class c_excitation(c_excitation_window, c_attr):
        def __init__(self):
            super().__init__(g, dpath, e_photon, scale)
        
        @override
        def check_string(self, s:str) -> str:
            try:
                float(s)
                return s
            except Exception as e:
                messagebox.showerror("Error", f"{e}\nPlease enter a valid number for Excitation Energy.")
                c_excitation()
                return ''

    class c_name(c_name_window, c_attr):
        def __init__(self):
            super().__init__(g, dpath, name, scale)
            
        @override
        def pars(self):
            for i, j in zip(['data', 'dpath'], [self.data, self.dpath]):
                set_globals(j, i)

    class c_description(c_description_window, c_attr):
        def __init__(self):
            super().__init__(g, dpath, description, scale)

    class ColormapEditor(ColormapEditorWindow):
        def __init__(self):
            super().__init__(g, scale, optionList3, value3, setcmap, cmlf, cdir)
    
    class version_check(VersionCheckWindow):
        def __init__(self):
            super().__init__(g, scale, cdir, app_name, __version__, hwnd)
        
        @override
        def get_src(self, ver: bool = False):
            get_src(ver)
    
    class plot1_window(Plot1Window):
        def __init__(self, parent: tk.Misc | None, scale: float):
            super().__init__(parent, scale)

    class plot1_window_MDC_curves(Plot1Window_MDC_curves, plot1_window):
        def __init__(self, d: int, l: int, p: int):
            super().__init__(g, scale, d, l, p)

        @override
        def chf(self):
            global d, l, p
            try:
                d = int(self.v_d.get())
                l = int(self.v_l.get())
                p = int(self.v_p.get())
                if p < l:
                    threading.Thread(target=o_plot1, daemon=True).start()
                    self.destroy()
                else:
                    messagebox.showwarning("Warning","Invalid Input\n"+"Polyorder must be less than window_length")
                    self.destroy()
                    plot1()
            except:
                self.destroy()
                plot1()

    class plot1_window_Second_Derivative(Plot1Window_Second_Derivative, plot1_window):
        def __init__(self, im_kernel: int):
            super().__init__(g, scale, im_kernel)
        
        @override
        def chf(self):
            global im_kernel
            try:
                if int(self.v_k.get())%2==1:
                    im_kernel = int(self.v_k.get())
                    threading.Thread(target=o_plot1, daemon=True).start()
                    self.destroy()
                else:
                    messagebox.showwarning("Warning","Invalid Input\n"+"Kernel size must be an odd number")
                    self.destroy()
                    plot1()
            except:
                self.destroy()
                plot1()
    
    class plot3_window(Plot3Window):
        def __init__(self, fev: list, fk: list):
            super().__init__(g, scale, fev, fk)
        
        @override
        def ini(self):
            global mp, ep, mf, ef
            if len(self.fev) <= 0:
                mp, mf = 0, 0
                for i in [self.mpos, self.mfwhm]:
                    i.deselect()
                    i.config(state='disabled')
            if len(self.fk) <= 0:
                ep, ef = 0, 0
                for i in [self.epos, self.efwhm]:
                    i.deselect()
                    i.config(state='disabled')
                
        @override
        def chf(self):
            global mp, ep, mf, ef
            mp, ep, mf, ef = self.v_mpos.get(), self.v_epos.get(), self.v_mfwhm.get(), self.v_efwhm.get()
            threading.Thread(target=o_plot3, daemon=True).start()
            self.destroy()
    
    class MainMotion(motion):
        def __init__(self):
            var_list = ['scale', 'value', 'value1', 'value2', 'k_offset', 'be', 'k', 'bb_offset', 'bbk_offset', 'ao', 'bo', 'out', 'figy', 'rcx', 'rcy', 'xdata', 'ydata', 'emf', 'data', 'vfe', 'ev', 'phi', 'pos', 'fwhm', 'rpos', 'ophi', 'fev', 'epos', 'efwhm', 'fk', 'ffphi', 'fphi', 'mp', 'ep', 'mf', 'ef', 'xl', 'yl', 'tb0', 'tb0_', 'tb1', 'tb1_', 'tb2']
            for i in var_list:
                init_globals(i)
            super().__init__(scale, value, value1, value2, k_offset,
                 be, k, bb_offset, bbk_offset,
                 ao, bo, out, figy,
                 rcx, rcy, xdata, ydata,
                 emf, data, vfe, ev, phi,
                 pos, fwhm, rpos, ophi, fev,
                 epos, efwhm, fk, ffphi, fphi,
                 mp, ep, mf, ef, xl, yl,
                 tb0, tb0_, tb1, tb1_, tb2)

@pool_protect
def f_help(*e):
    import webbrowser
    url = r"https://github.com/alex20000910/main"
    webbrowser.open(url)

@pool_protect
def about(*e):
    AboutWindow(master=g, scale=scale, version=__version__, release_date=__release_date__)

@pool_protect
def fit_so_app(*args):
    global fit_so
    try:
        fit_so.lift()
    except TypeError:
        fit_so = SO_Fitter(g, app_pars)
    except AttributeError:
        fit_so = SO_Fitter(g, app_pars)

@pool_protect
def emode():
    global gfe,fe_in,b_emode,emf,v_fe,mfpr
    if 'gfe' in globals():
        gfe.destroy()
        clear(gfe)
    mfpr=0
    if emf=='KE':
        emf='BE'
        b_emode.config(text='B.E.')
        gfe = G_emode()
    else:
        emf='KE'
        b_emode.config(text='K.E.')
        gfe = G_emode()

def patch_origin():
    threading.Thread(target=f_patch_origin,daemon=True).start()

def f_patch_origin():
    limg.config(image=img[np.random.randint(len(img))])
########################### patching ############################
    print('Patching OriginPro...')
    st.put('Patching OriginPro...')
    exe=rf"\Origin.exe" # OriginPro Patching file
    cmd=f'start "" cmd /C "dir "{exe}" /s"'
    result = os.popen(cmd) # 返回的結果是一個<class 'os._wrap_close'>對象，需要讀取後才能處理
    context = result.read()
    for line in context.splitlines():
        if '的目錄' in line or 'Directory of' in line:
            path = line.removeprefix('Directory of ')
            path = line.removesuffix(' 的目錄')
            # print(line)
            # print(path)
            path = path.removeprefix(" ")
            path = rf"{path}"
            path = rf"{path}{exe}"
            if path.split(os.sep)[-2] != 'Crack':
                ori_temp_path = path.removesuffix(os.sep+path.split(os.sep)[-1])
                print('Origin Path: '+ori_temp_path)
                os.system(f"\"{path}\"")
    result.close()
    print('Patching OriginPro...Done')
    st.put('Patching OriginPro...Done')
########################### patching ############################
def ch_suffix(*e):
    origin_func.ch_suffix(dpath, l1, b3)

def pre_process(input):
    return str(input).replace(' ',', ').replace(', , , , ,',',').replace(', , , ,',',').replace(', , ,',',').replace(', ,',',').replace('[, ','[').replace(', ]',']')

def gui_exp_origin(*e):
    global gori,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,l1,b3,origin_func
    if data is None:
        st.put('No data loaded!')
        messagebox.showwarning("Warning","No data loaded!")
        return
    limg.config(image=img[np.random.randint(len(img))])
    if 'gori' in globals():
        gori.destroy()
    origin_func = origin()
    gori=RestrictedToplevel(g,bg='white')
    gori.title('Export to Origin')
    l1=tk.Label(gori,text=f"{dpath.removesuffix('.h5').removesuffix('.json').removesuffix('.txt')}.{origin_func.suffix}",font=('Arial', size(10), "bold"),bg='white',wraplength=600)
    l1.grid(row=0,column=0)
    b1=tk.Button(gori,text='Patch Origin',command=patch_origin, width=15, height=1, font=('Arial', size(18), "bold"), bg='white', bd=5)
    # b1.grid(row=1,column=0)
    fr=tk.Frame(gori,bg='white')
    fr.grid(row=2,column=0)
    pr_exp_origin()
    v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11=tk.IntVar(),tk.IntVar(),tk.IntVar(),tk.IntVar(),tk.IntVar(),tk.IntVar(),tk.IntVar(),tk.IntVar(),tk.IntVar(),tk.IntVar(),tk.IntVar()
    c1=tk.Checkbutton(fr,text='E-Phi (Raw Data)',variable=v1,font=('Arial', size(18), "bold"),bg='white')
    if npzf:c1.config(text='E-k (Sliced Data)')
    c1.grid(row=0,column=0,sticky='w')
    c2=tk.Checkbutton(fr,text='E-k (Processed Data)',variable=v2,font=('Arial', size(18), "bold"),bg='white')
    c2.grid(row=1,column=0,sticky='w')
    c3=tk.Checkbutton(fr,text='MDC Fit Position',variable=v3,font=('Arial', size(18), "bold"),bg='white')
    c3.grid(row=2,column=0,sticky='w')
    c4=tk.Checkbutton(fr,text='MDC Fit FWHM',variable=v4,font=('Arial', size(18), "bold"),bg='white')
    c4.grid(row=3,column=0,sticky='w')
    c5=tk.Checkbutton(fr,text='EDC Fit Position',variable=v5,font=('Arial', size(18), "bold"),bg='white')
    c5.grid(row=4,column=0,sticky='w')
    c6=tk.Checkbutton(fr,text='EDC Fit FWHM',variable=v6,font=('Arial', size(18), "bold"),bg='white')
    c6.grid(row=5,column=0,sticky='w')
    c7=tk.Checkbutton(fr,text='Self Energy Real Part',variable=v7,font=('Arial', size(18), "bold"),bg='white')
    c7.grid(row=6,column=0,sticky='w')
    c8=tk.Checkbutton(fr,text='Self Energy Imaginary Part',variable=v8,font=('Arial', size(18), "bold"),bg='white')
    c8.grid(row=7,column=0,sticky='w')
    c9=tk.Checkbutton(fr,text='Data plot with pos',variable=v9,font=('Arial', size(18), "bold"),bg='white')
    c9.grid(row=8,column=0,sticky='w')
    c10=tk.Checkbutton(fr,text='Data plot with pos & bare band',variable=v10,font=('Arial', size(18), "bold"),bg='white')
    c10.grid(row=9,column=0,sticky='w')
    c11=tk.Checkbutton(fr,text='Second Derivative',variable=v11,font=('Arial', size(18), "bold"),bg='white')
    c11.grid(row=10,column=0,sticky='w')
    fr_exp=tk.Frame(fr,bg='white')
    fr_exp.grid(row=11,column=0)
    b2=tk.Button(fr_exp,text='Export',command=exp_origin, width=15, height=1, font=('Arial', size(18), "bold"), bg='white', bd=5)
    b2.pack(side='left')
    b3=tk.Button(fr_exp,text=f'(.{origin_func.suffix})',command=ch_suffix, width=15, height=1, font=('Arial', size(18), "bold"), bg='white', bd=5)
    b3.pack(side='right')
    cl=[c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11]
    for i in range(len(cl)):
        if i in no:
            cl[i].deselect()
            cl[i].config(state='disabled')
        else:
            cl[i].config(state='normal')
            cl[i].select()
    if npzf:
        c2.deselect()
        c2.config(state='disabled')
    gori.bind('<Return>', exp_origin)
    set_center(g, gori, 0, 0)
    gori.focus_set()
    gori.limit_bind()
    
def pr_exp_origin():
    global cmdlist, no
    ex_raw,ex_ek,ex_mp,ex_mf,ex_ep,ex_ef,ex_ser,ex_sei,ex_dpp,ex_dppbb,ex_sd='','','','','','','','','','',''
    cmdlist=dict({0:f'{ex_raw}',1:f'{ex_ek}',2:f'{ex_mp}',3:f'{ex_mf}',4:f'{ex_ep}',5:f'{ex_ef}',6:f'{ex_ser}',7:f'{ex_sei}',8:f'{ex_dpp}',9:f'{ex_dppbb}',10:f'{ex_sd}'})
    no=[]
    try:
        cmdlist[0]=f'''plot2d()\n'''
    except:
        no.append(0)
    try:
        cmdlist[1]=f'''plot2d(title='E-k (Processed Data)')\n'''
    except:
        no.append(1)
    try:
        ophi = np.arcsin(rpos/(2*m*fev*1.602176634*10**-19)**0.5 /
                        10**-10*(h/2/np.pi))*180/np.pi
        pos = (2*m*fev*1.602176634*10**-19)**0.5 * \
            np.sin((np.float64(k_offset.get())+ophi)/180*np.pi)*10**-10/(h/2/np.pi)
        cmdlist[2]=rf'''plot1d(x={pre_process((vfe-fev)*1000)}, y1={pre_process(pos)}, title='MDC Fit Position', xlabel='Binding Energy', ylabel='k', xunit='meV', yunit=r"2\g(p)Å\+(-1)")
'''
    except:
        no.append(2)
    try:
        cmdlist[3]=rf'''plot1d(x={pre_process((vfe-fev)*1000)}, y1={pre_process(fwhm)}, title='MDC Fit FWHM', xlabel='Binding Energy', ylabel='k', xunit='meV', yunit=r"2\g(p)Å\+(-1)")
'''
    except:
        no.append(3)
    try:
        cmdlist[4]=rf'''plot1d(x={pre_process(fk)}, y1={pre_process((vfe-epos)*1000)}, title='EDC Fit Position', xlabel='k', ylabel='Binding Energy', xunit=r"2\g(p)Å\+(-1)", yunit='meV')
'''
    except:
        no.append(4)
    try:
        cmdlist[5]=rf'''plot1d(x={pre_process(fk)}, y1={pre_process(efwhm)}, title='EDC Fit FWHM', xlabel='k', ylabel='Binding Energy', xunit=r"2\g(p)Å\+(-1)", yunit='meV')
'''
    except:
        no.append(5)
    try:
        ophi = np.arcsin(rpos/(2*m*fev*1.602176634*10**-19)**0.5 /
                        10**-10*(h/2/np.pi))*180/np.pi
        pos = (2*m*fev*1.602176634*10**-19)**0.5 * \
            np.sin((np.float64(k_offset.get())+ophi)/180*np.pi)*10**-10/(h/2/np.pi)
        x = (vfe-fev)*1000
        y = pos
        yy = interp(pos, k*np.float64(bbk_offset.get()), be -
                    # interp x into be,k set
                    np.float64(bb_offset.get()))
        x = (vfe-fev)*1000
        rx = x
        ry = -(x+yy)
        tbe = (vfe-fev)*1000
        x = interp(tbe, -be+np.float64(bb_offset.get()),
                    k*np.float64(bbk_offset.get()))
        y = interp(x, k*np.float64(bbk_offset.get()),
                    -be+np.float64(bb_offset.get()))
        xx = np.diff(x)
        yy = np.diff(y)

        # eliminate vf in gap
        for i in range(len(yy)):
            if yy[i]/xx[i] > 20000:
                yy[i] = 0
        v = yy/xx
        # v = np.append(v, v[-1])  # fermi velocity
        v=interp(pos,x[0:-1]+xx/2,v)
        yy = np.abs(v*fwhm/2)
        xx = tbe

        ix = xx
        iy = yy
        ix=(tbe-tbe[-1])*-1
        cix=np.append(ix+ix[0],ix)
        tix=cix[0:len(cix)-1]*-1
        # kx=ix
        kx = np.append(cix,tix[::-1])
        ky = np.linspace(0, 1, len(kx))
        ciy=np.append(iy*0+np.mean(iy),iy)
        tiy=ciy[0:len(ciy)-1]
        ciy = np.append(ciy,tiy[::-1])

        #for imaginary part
        ix=(tbe-tbe[-1])*-1
        cix=np.append(ix+ix[0],ix)
        tix=cix[0:len(cix)-1]*-1
        kx = np.append(cix,tix[::-1])
        ky = np.linspace(0, 1, len(kx))
        cry=np.append(ry*0,ry)
        tcry=cry[0:len(cry)-1]*-1
        cry = np.append(cry,tcry[::-1])

        # Hilbert transform
        analytic_signal_r = hilbert(cry)
        analytic_signal_i = hilbert(ciy)
        # Reconstructed real and imaginary parts
        reconstructed_real = np.imag(analytic_signal_i)
        reconstructed_imag = -np.imag(analytic_signal_r)
        cmdlist[6]=rf'''plot1d(x={pre_process((vfe-fev)*1000)}, y1={pre_process(-1*((vfe-fev)*1000+interp(pos, k*np.float64(bbk_offset.get()), be - np.float64(bb_offset.get()))))}, y2={pre_process(reconstructed_real[len(ix):2*len(ix)]+(ry-np.mean(reconstructed_real[len(ix):2*len(ix)])))}, title='Self Energy Real Part', xlabel='Binding Energy', ylabel=r"Re \g(S)", ylabel1=r"Re \g(S)", ylabel2=r"Re \g(S)\-(KK)=KK(Im \g(S))", xunit='meV', yunit='meV')
'''
    except:
        no.append(6)
    try:
        ophi = np.arcsin(rpos/(2*m*fev*1.602176634*10**-19)**0.5 /
                        10**-10*(h/2/np.pi))*180/np.pi
        pos = (2*m*fev*1.602176634*10**-19)**0.5 * \
            np.sin((np.float64(k_offset.get())+ophi)/180*np.pi)*10**-10/(h/2/np.pi)
        x = (vfe-fev)*1000
        y = pos
        yy = interp(pos, k*np.float64(bbk_offset.get()), be -
                    # interp x into be,k set
                    np.float64(bb_offset.get()))
        x = (vfe-fev)*1000
        rx = x
        ry = -(x+yy)
        tbe = (vfe-fev)*1000
        x = interp(tbe, -be+np.float64(bb_offset.get()),
                    k*np.float64(bbk_offset.get()))
        y = interp(x, k*np.float64(bbk_offset.get()),
                    -be+np.float64(bb_offset.get()))
        xx = np.diff(x)
        yy = np.diff(y)

        # eliminate vf in gap
        for i in range(len(yy)):
            if yy[i]/xx[i] > 20000:
                yy[i] = 0
        v = yy/xx
        # v = np.append(v, v[-1])  # fermi velocity
        v=interp(pos,x[0:-1]+xx/2,v)
        yy = np.abs(v*fwhm/2)
        xx = tbe

        ix = xx
        iy = yy
        ix=(tbe-tbe[-1])*-1
        cix=np.append(ix+ix[0],ix)
        tix=cix[0:len(cix)-1]*-1
        # kx=ix
        kx = np.append(cix,tix[::-1])
        ky = np.linspace(0, 1, len(kx))
        ciy=np.append(iy*0+np.mean(iy),iy)
        tiy=ciy[0:len(ciy)-1]
        ciy = np.append(ciy,tiy[::-1])

        #for imaginary part
        ix=(tbe-tbe[-1])*-1
        cix=np.append(ix+ix[0],ix)
        tix=cix[0:len(cix)-1]*-1
        kx = np.append(cix,tix[::-1])
        ky = np.linspace(0, 1, len(kx))
        cry=np.append(ry*0,ry)
        tcry=cry[0:len(cry)-1]*-1
        cry = np.append(cry,tcry[::-1])

        # Hilbert transform
        analytic_signal_r = hilbert(cry)
        analytic_signal_i = hilbert(ciy)
        # Reconstructed real and imaginary parts
        reconstructed_real = np.imag(analytic_signal_i)
        reconstructed_imag = -np.imag(analytic_signal_r)
        cmdlist[7]=rf'''plot1d(x={pre_process((vfe-fev)*1000)}, y1={pre_process(iy)}, y2={pre_process(reconstructed_imag[len(ix):2*len(ix)]+(iy-np.mean(reconstructed_imag[len(ix):2*len(ix)])))}, title='Self Energy Imaginary Part', xlabel='Binding Energy', ylabel=r"Im \g(S)", ylabel1=r"Im \g(S)", ylabel2=r"Im \g(S)\-(KK)=KK(Re \g(S))", xunit='meV', yunit='meV')
'''
    except:
        no.append(7)
    try:
        x1 = pos
        if emf=='KE':
            y1=np.float64(fev)
        else:
            y1= vfe-np.float64(fev)
        cmdlist[8]=f'''plot2d(title='Data plot with pos')\n'''
    except:
        no.append(8)
    try:
        x2 = k*float(bbk_offset.get())
        if emf=='KE':
            y2 = (be - float(bb_offset.get()))/1000+vfe
        else:
            y2 = (-be + float(bb_offset.get()))/1000
        cmdlist[9]=f'''plot2d(title='Data plot with pos & bare band')\n'''
    except:
        no.append(9)
    try:
        cmdlist[10]=f'''plot2d(title='Second Derivative (Processed Data)')\n'''
    except:
        no.append(10)
        
def exp_origin(*e):
    global origin_func
    origin_temp_var = f'''from {app_name} import *
import originpro as op

cdir = r"{cdir}"
npzf = {npzf}
dpath = r"{dpath}"      # Data Path
emf = r"{emf}"             # Energy Mode: KE or BE
ko = {k_offset.get()}
bbo = {bb_offset.get()}
bbk = {bbk_offset.get()}
vfe = {vfe}
im_kernel = {im_kernel}     # Gaussian Filter Kernel Size
nan = np.nan
'''
    try:
        origin_temp_var += f'''
bpath = r"{bpath}"         # Bare Band Path
be = np.float64({pre_process(be)})
k = np.float64({pre_process(k)})
'''
    except:
        origin_temp_var += f'''
bpath = None
'''
    try:
        ophi = np.arcsin(rpos/(2*m*fev*1.602176634*10**-19)**0.5 /
                        10**-10*(h/2/np.pi))*180/np.pi
        pos = (2*m*fev*1.602176634*10**-19)**0.5 * \
            np.sin((np.float64(k_offset.get())+ophi)/180*np.pi)*10**-10/(h/2/np.pi)
        origin_temp_var += f'''
fev = np.float64({pre_process(np.asarray(fev, dtype=np.float64))})
pos = np.float64({pre_process(np.asarray(pos, dtype=np.float64))})
fwhm = np.float64({pre_process(np.asarray(fwhm, dtype=np.float64))})
'''
    except: pass
    try:
        ffphi = np.float64(k_offset.get())+fphi
        fk = (2*m*epos*1.602176634*10**-19)**0.5 * \
            np.sin(ffphi/180*np.pi)*10**-10/(h/2/np.pi)
        origin_temp_var += f'''
fk = np.float64({pre_process(np.asarray(fk, dtype=np.float64))})
epos = np.float64({pre_process(np.asarray(epos, dtype=np.float64))})
efwhm = np.float64({pre_process(np.asarray(efwhm, dtype=np.float64))})
'''
    except: pass
    if '.h5' in os.path.basename(dpath):
        tload = f'''
data = load_h5(dpath)        
'''
    elif '.json' in os.path.basename(dpath):
        tload = f'''
data = load_json(dpath)
'''
    elif '.txt' in os.path.basename(dpath):
        tload = f'''
data = load_txt(dpath)
'''
    elif '.npz' in os.path.basename(dpath):
        tload = f'''
data = load_npz(dpath)
'''
    origin_temp_var += tload
    origin_temp_var += f'''
dvalue = list(data.attrs.values())
dkey = list(data.attrs.keys())
ev, phi = data.indexes.values()
ev, phi = np.float64(ev), np.float64(phi)

if emf=='KE':
    le_mode='Kinetic Energy'
    tx, ty = np.meshgrid(phi, ev)
    tev = ty.copy()
else:
    le_mode='Binding Energy'
    tx, ty = np.meshgrid(phi, vfe-ev)
    tev = vfe-ty.copy()
tz = data.to_numpy()
sdz = laplacian_filter(data.to_numpy(), im_kernel)
'''
    origin_temp_exec = r'''
op.new()
op.set_show(True)

'''
    origin_temp_save = r'''
note()
save()
'''
    cl=[v1.get(),v2.get(),v3.get(),v4.get(),v5.get(),v6.get(),v7.get(),v8.get(),v9.get(),v10.get(),v11.get()]
    gori.destroy()
    for i in cmdlist.keys():
        if cl[i]==1:
            origin_temp_exec+=cmdlist[i]
        
    with open(cdir+os.sep+'origin_temp.py', 'w', encoding='utf-8') as f:
        f.write(origin_temp_var+origin_func.func+origin_temp_exec+origin_temp_save)
    f.close()
    origin_func = None
    def j():
        # os.system(f'code {cdir+r"\origin_temp.py"}')
        limg.config(image=img[np.random.randint(len(img))])
        print('Exporting to Origin...')
        st.put('Exporting to Origin...')
        temp=os.sep+"origin_temp.py"
        os.system(f'python -W ignore::SyntaxWarning -W ignore::UserWarning "{cdir+temp}"')
        os.system(f'del "{cdir+temp}"')
        limg.config(image=img[np.random.randint(len(img))])
        print('Exported to Origin')
        st.put('Exported to Origin')
    threading.Thread(target=j,daemon=True).start()

def rplot(f, canvas):
    """
    Plot the raw data on a given canvas.

    Parameters
    -----
        f (Figure object): The figure object on which the plot will be created.
        canvas (Canvas object): The canvas object on which the plot will be drawn.

    Returns
    -----
        None
    """
    global data, ev, phi, value3, h0, ao, xl, yl, rcx, rcy, acb
    ao = f.add_axes([0.13, 0.1, 0.6, 0.65])
    rcx = f.add_axes([0.13, 0.78, 0.6, 0.15])
    rcy = f.add_axes([0.75, 0.1, 0.12, 0.65])
    acb = f.add_axes([0.9, 0.1, 0.02, 0.65])
    rcx.set_xticks([])
    rcx.set_yticks([])
    rcy.set_xticks([])
    rcy.set_yticks([])
    if emf=='KE':
        tx, ty = np.meshgrid(phi, ev)
    else:
        tx, ty = np.meshgrid(phi, vfe-ev)
    tz = data.to_numpy()
    # h1 = a.scatter(tx,ty,c=tz,marker='o',s=scale*scale*0.9,cmap=value3.get());
    h0 = ao.pcolormesh(tx, ty, tz, cmap=value3.get())
    f.colorbar(h0, cax=acb, orientation='vertical')
    # a.set_title('Raw Data',font='Arial',fontsize=size(16))
    rcx.set_title('            Raw Data', font='Arial', fontsize=size(16))
    if npzf:ao.set_xlabel(r'k ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=size(12))
    else:ao.set_xlabel('Angle (deg)', font='Arial', fontsize=size(12))
    if emf=='KE':
        ao.set_ylabel('Kinetic Energy (eV)', font='Arial', fontsize=size(12))
    else:
        ao.set_ylabel('Binding Energy (eV)', font='Arial', fontsize=size(12))
        ao.invert_yaxis()
    xl = ao.get_xlim()
    yl = ao.get_ylim()
    np.save('raw_data.npy',tz.T/np.max(tz))
    # a.set_xticklabels(labels=a.get_xticklabels(),font='Arial',fontsize=size(10));
    # a.set_yticklabels(labels=a.get_yticklabels(),font='Arial',fontsize=size(10));
    canvas.draw()

@pool_protect
def cexcitation():
    global gcestr
    messagebox.showwarning("Warning","Floats Input Only")
    if 'gcestr' in globals():
        gcestr.destroy()
        clear(gcestr)
    gcestr = c_excitation()

@pool_protect
def cname():
    global gcstr
    messagebox.showwarning("Warning","允許中文、符號")
    if 'gcstr' in globals():
        gcstr.destroy()
    gcstr = c_name()

@pool_protect
def desc():
    global gstr
    messagebox.showwarning("Warning","允許中文、符號")
    if 'gstr' in globals():
        gstr.destroy()
    gstr=c_description()

def view_3d(*e):
    DataViewer_PyQt5()

def DataViewer_PyQt5():
    def j():
        os.system(f'python -W ignore::SyntaxWarning -W ignore::UserWarning "{os.path.join(cdir, '.MDC_cut', 'tool', 'DataViewer.py')}"')
    threading.Thread(target=j,daemon=True).start()

@pool_protect
def set_window_background(g):
    """設定視窗背景圖片"""
    try:
        # 載入圖片
        bg_image = Image.open(io.BytesIO(b64decode(icon.icon)))
        bg_image = bg_image.resize((800, 600), Image.Resampling.LANCZOS)
        bg_photo = ImageTk.PhotoImage(bg_image)
        
        # 創建背景 Label
        bg_label = tk.Label(g, image=bg_photo)
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        bg_label.image = bg_photo  # 保持引用
        
        return bg_label
    except Exception as e:
        print(f"Fail to load background image: {e}")
        return None

class Button(tk.Button):
    """自定義按鈕類別，增加背景顏色"""
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.config(bg="white")

def poly_smooth(x, y, order=6,xx=None):
    """
    Polynomial fitting and smoothing.
    
    Parameters
    ----------
        x (array-like): 1D array
        y (array-like): 1D array
        order (int): default=6
        xx (array-like): 1D array, interpolation points, default: None
    Returns
    -------
        y (np.ndarray) : Smoothed or interpolated y values.
    """
    coeffs = np.polyfit(x, y, order)
    if xx is None:
        y = np.polyval(coeffs, x)
    else:
        y = np.polyval(coeffs, xx)
    return y

@pool_protect
def show_info():
    # 創建自定義窗口
    info_window = tk.Toplevel()
    info_window.title("Information")
    
    # 添加信息標籤
    l = tk.Label(info_window, text="Graph copied to clipboard", font=("Arial", size(30), "bold"),fg='red')
    l.pack(pady=5)
    label = tk.Label(info_window, text="window closed in 3 second", font=("Arial", size(20)))
    label.pack(pady=5)

    info_window.update()
    w= info_window.winfo_reqwidth()
    h= info_window.winfo_reqheight()
    info_window.geometry(f"{w}x{h}+{screen_width//2-w//2}+{screen_height//2-h//2}")
    
    # 設置計時器，3 秒後自動關閉窗口
    info_window.update()
    info_window.after(1000, label.config(text="window closed in 2 second"))
    info_window.update()
    info_window.after(1000, label.config(text="window closed in 1 second"))
    info_window.update()
    info_window.after(1000, label.config(text="window closed in 0 second"))
    info_window.update()
    info_window.destroy()
    
@pool_protect
def f_copy_to_clipboard():
    copy_to_clipboard(ff=fig)
    if value.get() != '---Plot1---' or value1.get() != '---Plot2---' or value2.get() != '---Plot3---':
        st.put('Copied to clipboard')
        
@pool_protect
def copy_to_clipboard(ff: Figure) -> None:
    """
    Copies the given figure to the clipboard as a bitmap image.
    
    Parameters
    -----------
        ff (matplotlib.figure.Figure) : The figure to be copied to the clipboard.
    
    Returns
    -----------
        None
    """
    try:
        limg.config(image=img[np.random.randint(len(img))])
    except:
        pass
    buf = io.BytesIO()
    ff.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    output = io.BytesIO()
    
    image.convert("RGB").save(output, "BMP")
    data = output.getvalue()[14:]
    output.close()
    send_to_clipboard(win32clipboard.CF_DIB, data)
    
@pool_protect
def send_to_clipboard(clip_type, data):
    win32clipboard.OpenClipboard()
    win32clipboard.EmptyClipboard()
    win32clipboard.SetClipboardData(clip_type, data)
    win32clipboard.CloseClipboard()
    
# spectrogram
            
@pool_protect
def trans_plot(*e):
    if data is None:
        st.put('No data loaded!')
        messagebox.showwarning("Warning","No data loaded!")
        return
    global gtp
    gtp=RestrictedToplevel(g)
    gtp.title('Spectrogram')
    b_raw = tk.Button(gtp, text='Raw', command=raw_plot, width=15, height=1, font=('Arial', size(14), "bold"), bg='white', bd=5)
    b_raw.grid(row=0, column=0)
    b_smooth = tk.Button(gtp, text='Smooth', command=smooth_plot, width=15, height=1, font=('Arial', size(14), "bold"), bg='white', bd=5)
    b_smooth.grid(row=0, column=1)
    b_fd = tk.Button(gtp, text='First Derivative', command=fd_plot, width=15, height=1, font=('Arial', size(14), "bold"), bg='white', bd=5)
    b_fd.grid(row=0, column=2)
    set_center(g, gtp, 0, 0)
    gtp.focus_set()
    gtp.bind('<Return>', raw_plot)
    gtp.limit_bind()

@pool_protect
def raw_plot(*args):
    gtp.destroy()
    cmap=value3.get()
    s=spectrogram(data, name='internal', app_pars=lfs.app_pars)
    s.plot(g, cmap)

@pool_protect
def smooth_plot():
    gtp.destroy()
    cmap=value3.get()
    y=smooth(np.sum(data.to_numpy().transpose(),axis=0),l=13)
    s=spectrogram(data, name='internal', app_pars=lfs.app_pars)
    s.setdata(ev, y, dtype='smooth', unit='Counts')
    s.plot(g, cmap)

@pool_protect
def fd_plot():
    gtp.destroy()
    cmap=value3.get()
    y=smooth(np.sum(data.to_numpy().transpose(),axis=0),l=13)
    s=spectrogram(data, name='internal', app_pars=lfs.app_pars)
    s.setdata(ev[0:-1]+(ev[1]-ev[0])/2, np.diff(y)/np.diff(ev), dtype='fd', unit='dN/dE')
    s.plot(g, cmap)

@pool_protect
def calculator(*e):
    global calf
    try:
        calf.destroy()
        clear(calf)
    except:
        pass
    calf = CalculatorWindow(g, scale)

@pool_protect
def scroll(event):
    if lfs is not None:
        if len(lfs.name) >1:
            if event.delta>0:
                cf_up()
            elif event.delta<0:
                cf_down()

@pool_protect
def cf_up(*args):
    global namevar
    now = namevar.get()
    for i, j in enumerate(lfs.name):
        if now == j:
            if i == 0:
                namevar.set(lfs.name[-1])
            else:
                namevar.set(lfs.name[i-1])
    change_file()

@pool_protect
def cf_down(*args): 
    global namevar
    now = namevar.get()
    for i, j in enumerate(lfs.name):
        if now == j:
            if i == len(lfs.name)-1:
                namevar.set(lfs.name[0])
            else:
                namevar.set(lfs.name[i+1])
    change_file()

npzf = False
@pool_protect
def change_file(*args):
    global data, rdd, npzf
    name = namevar.get()
    if len(name) >30:
        l_name.config(font=('Arial', size(11), "bold"))
    elif len(name) >20:
        l_name.config(font=('Arial', size(12), "bold"))
    else:
        l_name.config(font=('Arial', size(14), "bold"))
    for i, j, k, l in zip(lfs.name, lfs.data, lfs.path, lfs.f_npz):
        if name == i:
            data = lfs.get(j)
            pr_load(lfs.get(j))
            rdd = k
            if l:
                npzf = True
                koffset.config(state='normal')
                k_offset.set('0')
                koffset.config(state='disabled')
            else:
                npzf = False
                koffset.config(state='normal')
                try:
                    k_offset.set(ko)
                except:
                    k_offset.set('0')
    st.put(name)
    if value.get() != '---Plot1---':
        o_plot1()
    
@pool_protect
def tools(*args):
    def spec(*args):
        s = spectrogram(path=lfs.path, name='internal', app_pars=lfs.app_pars)
        s.plot(g, value3.get())
        toolg.destroy()
        return
        
    def exp_casa():
        explfs = lfs_exp_casa(lfs)
        explfs.export_casa()
        toolg.destroy()
        return
        
    def kplane():
        CEC(g, lfs.path, cmap=value3.get(), app_pars=lfs.app_pars)
        toolg.destroy()
        return
    
    global toolg
    if 'toolg' in globals():
        toolg.destroy()
    toolg = RestrictedToplevel(g)
    toolg.title('Batch Master')
    b_spec = tk.Button(toolg, text='Spectrogram', command=spec, width=15, height=1, font=('Arial', size(14), "bold"), bg='white', bd=5)
    b_spec.grid(row=0, column=0)
    try:
        flag = False
        t_, t__ = lfs.r1.copy(), lfs.r2.copy()
        flag = True
        t_, t__ = None, None
    except:
        pass
    if lfs.sort != 'no' and flag:
        b_kplane = tk.Button(toolg, text='k-Plane', command=kplane, width=15, height=1, font=('Arial', size(14), "bold"), bg='white', bd=5)
        b_kplane.grid(row=0, column=1)
    b_exp_casa = tk.Button(toolg, text='Export to Casa', command=exp_casa, width=15, height=1, font=('Arial', size(14), "bold"), bg='white', bd=5)
    b_exp_casa.grid(row=0, column=2)
    toolg.bind('<Return>', spec)
    set_center(g, toolg, 0, 0)
    toolg.focus_set()
    toolg.limit_bind()
    return

@pool_protect
def def_cmap():
    global CE
    if 'CE' in globals():
        CE.destroy()
        clear(CE)
    CE = ColormapEditor()

fpr = 0

@pool_protect
def o_load(drop=False, files=''):
    if not drop:
        files = fd.askopenfilenames(title="Select Raw Data", filetypes=(
        ("HDF5 files", "*.h5"), ("NPZ files", "*.npz"), ("JSON files", "*.json"), ("TXT files", "*.txt")))
    st.put('Loading...')
    files = tkDnD.load_raw(files)
    main_loader(files)

@pool_protect
def o_ecut():
    global data, ev, phi, mfpath, limg, img, name, rdd, st
    if data is None:
        st.put('No data loaded!')
        messagebox.showwarning("Warning","No data loaded!")
        return
    limg.config(image=img[np.random.randint(len(img))])
    mfpath = ''
    os.chdir(os.path.dirname(rdd))
    try:
        ndir = os.path.dirname(rdd)
        if ndir.split(os.sep)[-1] == name+'_MDC_'+lowlim.get():
            os.chdir('../')
        os.chdir(ndir)
        os.makedirs(name+'_MDC_'+lowlim.get())
    except:
        pass
    os.chdir(name+'_MDC_'+lowlim.get())
    pbar = tqdm.tqdm(total=len(ev), desc='MDC', colour='green')
    for n in range(len(ev)):
        ecut = data.sel(eV=ev[n], method='nearest')
        if npzf:x = phi
        else:x = (2*m*ev[n]*1.602176634*10**-19)**0.5*np.sin(phi/180*np.pi)*10**-10/(h/2/np.pi)
        y = ecut.to_numpy().reshape(len(x))
        y = np.where(y > int(lowlim.get()), y, int(lowlim.get()))
        path = 'ecut_%.3f.txt' % ev[n]
        mfpath += path
        pbar.update(1)
        if (n+1) % (len(ev)//100) == 0:
            st.put(str(round((n+1)/len(ev)*100))+'%'+' ('+str(len(ev))+')')
        f = open(path, 'w', encoding='utf-8')  # tab 必須使用 '\t' 不可"\t"
        f.write('#Wave Vector'+'\t'+'#Intensity'+'\n')
        for i in range(len(x)-1, -1, -1):
            f.write('%-6e' % x[i]+'\t'+'%-6e' % y[i]+'\n')
        f.close()
    os.chdir(cdir)
    np.savez(os.path.join(cdir, '.MDC_cut', 'mfpath.npz'), mfpath=mfpath)
    os.chdir(os.path.dirname(rdd))
    pbar.close()
    print('Done')
    st.put('Done')


@pool_protect
def o_angcut():
    global data, ev, phi, efpath, limg, img, name, rdd, st
    if data is None:
        st.put('No data loaded!')
        messagebox.showwarning("Warning","No data loaded!")
        return
    limg.config(image=img[np.random.randint(len(img))])
    efpath = ''
    os.chdir(os.path.dirname(rdd))
    try:
        ndir = os.path.dirname(rdd)
        if ndir.split(os.sep)[-1] == name+'_EDC'+lowlim.get():
            os.chdir('../')
        os.chdir(ndir)
        os.makedirs(name+'_EDC'+lowlim.get())
    except:
        pass
    os.chdir(name+'_EDC'+lowlim.get())
    pbar = tqdm.tqdm(total=len(phi), desc='EDC', colour='blue')
    for n in range(len(phi)):
        angcut = data.sel(phi=phi[n], method='nearest')
        x = ev
        y = angcut.to_numpy().reshape(len(x))
        y = np.where(y > int(lowlim.get()), y, int(lowlim.get()))
        path = 'angcut_%.5d.txt' % (phi[n]*1000)
        efpath += path
        pbar.update(1)
        if (n+1) % (len(phi)//100) == 0:
            st.put(str(round((n+1)/len(phi)*100))+'%'+' ('+str(len(phi))+')')
        f = open(path, 'w', encoding='utf-8')  # tab 必須使用 '\t' 不可"\t"
        f.write('#Wave Vector'+'\t'+'#Intensity'+'\n')
        for i in range(len(x)-1, -1, -1):
            f.write('%-6e' % x[i]+'\t'+'%-6e' % y[i]+'\n')
        f.close()
    os.chdir(cdir)
    np.savez(os.path.join(cdir, '.MDC_cut', 'efpath.npz'), efpath=efpath)
    os.chdir(os.path.dirname(rdd))
    pbar.close()
    print('Done')
    st.put('Done')

@pool_protect
def cmfit(*e):
    if data is None:
        st.put('No data loaded!')
        messagebox.showwarning("Warning","No data loaded!")
        return
    import tool.MDC_Fitter
    from tool.MDC_Fitter import mgg as mgg
    if mgg is None:
        mdc_pars = MDC_param(ScaleFactor=ScaleFactor, sc_y=sc_y, g=g, scale=scale, npzf=npzf, vfe=vfe, emf=emf, st=st, dpath=dpath, name=name, k_offset=k_offset, value3=value3, ev=ev, phi=phi, data=data, base=base, fpr=fpr, skmin=skmin, skmax=skmax, smfp=smfp, smfi=smfi, smaa1=smaa1, smaa2=smaa2, smresult=smresult, smcst=smcst)
        threading.Thread(target=tool.MDC_Fitter.fitm, args=(mdc_pars,)).start()
        clear(mdc_pars)
    elif isinstance(mgg, tk.Toplevel):
        mgg.lift()
    elif mgg == True:
        importlib.reload(tool.MDC_Fitter)
        mdc_pars = MDC_param(ScaleFactor=ScaleFactor, sc_y=sc_y, g=g, scale=scale, npzf=npzf, vfe=vfe, emf=emf, st=st, dpath=dpath, name=name, k_offset=k_offset, value3=value3, ev=ev, phi=phi, data=data, base=base, fpr=fpr, skmin=skmin, skmax=skmax, smfp=smfp, smfi=smfi, smaa1=smaa1, smaa2=smaa2, smresult=smresult, smcst=smcst)
        threading.Thread(target=tool.MDC_Fitter.fitm, args=(mdc_pars,)).start()
        clear(mdc_pars)

@pool_protect
def cefit(*e):
    if data is None:
        st.put('No data loaded!')
        messagebox.showwarning("Warning","No data loaded!")
        return
    import tool.EDC_Fitter
    from tool.EDC_Fitter import egg as egg
    if egg is None:
        edc_pars = EDC_param(ScaleFactor=ScaleFactor, sc_y=sc_y, g=g, scale=scale, npzf=npzf, vfe=vfe, emf=emf, st=st, dpath=dpath, name=name, k_offset=k_offset, value3=value3, ev=ev, phi=phi, data=data, base=base, fpr=fpr, semin=semin, semax=semax, sefp=sefp, sefi=sefi, seaa1=seaa1, seaa2=seaa2)
        threading.Thread(target=tool.EDC_Fitter.fite, args=(edc_pars,)).start()
        clear(edc_pars)
    elif isinstance(egg, tk.Toplevel):
        egg.lift()
    elif egg == True:
        importlib.reload(tool.EDC_Fitter)
        edc_pars = EDC_param(ScaleFactor=ScaleFactor, sc_y=sc_y, g=g, scale=scale, npzf=npzf, vfe=vfe, emf=emf, st=st, dpath=dpath, name=name, k_offset=k_offset, value3=value3, ev=ev, phi=phi, data=data, base=base, fpr=fpr, semin=semin, semax=semax, sefp=sefp, sefi=sefi, seaa1=seaa1, seaa2=seaa2)
        threading.Thread(target=tool.EDC_Fitter.fite, args=(edc_pars,)).start()
        clear(edc_pars)

###########################decprecated##############################
# def o_fitgl():
#     try:
#         # global pos,fwhm,epos,efwhm,base,k_offset,st,evv,eaa,fexx,feyy,fex,fey,mvv,maa,fmxx,fmyy,fmx,fmy
#         global st
#         print('fitting')
#         st.put('fitting')
#         t1 = threading.Thread(target=fitm)
#         t2 = threading.Thread(target=fite)
#         t1.start()
#         t2.start()
#         t1.join()
#         t2.join()
#         print('Done')
#         st.put('Done')
#     except:
#         pass
###########################decprecated##############################

def clmfit():
    global rpos, pos, fwhm, fev, ophi
    rpos = []
    pos = []
    fwhm = []
    fev = []
    ophi = []


def clefit():
    global fphi, epos, ffphi, efwhm, fk
    fphi = []
    epos = []
    ffphi = []
    efwhm = []
    fk = []


def cminrange(*e):
    if vcmax.get()-vcmin.get() < 1:
        try:
            vcmax.set(vcmin.get())
        except:
            pass
    try:
        h0.set_clim([vcmin.get(), vcmax.get()])
        out.draw()
    except:
        pass


def cmaxrange(*e):
    if vcmax.get()-vcmin.get() < 1:
        try:
            vcmin.set(vcmax.get())
        except:
            pass
    try:
        h0.set_clim([vcmin.get(), vcmax.get()])
        out.draw()
    except:
        pass


def o_fbb_offset(*e):
    global bb_offset
    if '' == bb_offset.get():
        bb_offset.set('0')
        bboffset.select_range(0, 1)
    os.chdir(cdir)
    np.savez(os.path.join(cdir, '.MDC_cut', 'bb.npz'), path=bpath, be=be, k=k, bbo=float(bb_offset.get()), bbk=float(bbk_offset.get()))


def fbb_offset(*e):
    t = threading.Thread(target=o_fbb_offset)
    t.daemon = True
    t.start()


def o_fbbk_offset(*e):
    global bbk_offset
    if '' == bbk_offset.get():
        bbk_offset.set('1')
        bbkoffset.select_range(0, 1)
    os.chdir(cdir)
    np.savez(os.path.join(cdir, '.MDC_cut', 'bb.npz'), path=bpath, be=be, k=k, bbo=float(bb_offset.get()), bbk=float(bbk_offset.get()))


def fbbk_offset(*e):
    t = threading.Thread(target=o_fbbk_offset)
    t.daemon = True
    t.start()


def o_fbase(*e):
    global base
    if '' == base.get():
        base.set('0')
        in_fit.select_range(0, 1)


def fbase(*e):
    t = threading.Thread(target=o_fbase)
    t.daemon = True
    t.start()


def o_flowlim(*e):
    global lowlim
    if '' == lowlim.get():
        lowlim.set('0')
        in_lowlim.select_range(0, 1)


def flowlim(*e):
    t = threading.Thread(target=o_flowlim)
    t.daemon = True
    t.start()


@pool_protect
def o_reload(*e):
    global k_offset, fev, ophi, rpos, pos, ffphi, fwhm, fk, st, kmin, kmax, smresult, smcst, smaa1, smaa2, smfp, smfi, skmin, skmax, epos, efwhm, ffphi, fk, emin, emax, seaa1, seaa2, sefp, sefi, semin, semax
    if '' == k_offset.get():
        k_offset.set('0')
        koffset.select_range(0, 1)
    try:
        ophi = np.arcsin(rpos/np.sqrt(2*m*fev*1.602176634*10**-19)/10**-10*(h/2/np.pi))*180/np.pi
    except NameError:
        return
    except TypeError:
        return
    pos = np.sqrt(2*m*fev*1.602176634*10**-19)*np.sin((np.float64(k_offset.get())+ophi)/180*np.pi)*10**-10/(h/2/np.pi)
    ophimin = np.arcsin((rpos-fwhm/2)/np.sqrt(2*m*fev*1.602176634*10**-19)/10**-10*(h/2/np.pi))*180/np.pi
    ophimax = np.arcsin((rpos+fwhm/2)/np.sqrt(2*m*fev*1.602176634*10**-19)/10**-10*(h/2/np.pi))*180/np.pi
    kmin = np.sqrt(2*m*fev*1.602176634*10**-19)*np.sin((np.float64(k_offset.get())+ophimin)/180*np.pi)*10**-10/(h/2/np.pi)
    kmax = np.sqrt(2*m*fev*1.602176634*10**-19)*np.sin((np.float64(k_offset.get())+ophimax)/180*np.pi)*10**-10/(h/2/np.pi)
    # okmphi = np.arcsin(kmin/np.sqrt(2*m*fev*1.602176634*10**-19) /
    #                    10**-10*(h/2/np.pi))*180/np.pi
    # kmin = np.sqrt(2*m*fev*1.602176634*10**-19) * \
    #     np.sin((np.float64(k_offset.get())+okmphi) /
    #            180*np.pi)*10**-10/(h/2/np.pi)
    # okMphi = np.arcsin(kmax/np.sqrt(2*m*fev*1.602176634*10**-19) /
    #                    10**-10*(h/2/np.pi))*180/np.pi
    # kmax = np.sqrt(2*m*fev*1.602176634*10**-19) * \
    #     np.sin((np.float64(k_offset.get())+okMphi) /
    #            180*np.pi)*10**-10/(h/2/np.pi)
    os.chdir(cdir)
    try:
        np.savez(os.path.join(cdir, '.MDC_cut', 'mfit.npz'), ko=k_offset.get(), fev=fev, rpos=rpos, ophi=ophi, fwhm=fwhm, pos=pos, kmin=kmin,
                 kmax=kmax, skmin=skmin, skmax=skmax, smaa1=smaa1, smaa2=smaa2, smfp=smfp, smfi=smfi)
        np.savez(os.path.join(cdir, '.MDC_cut', 'mfit.npz'), ko=k_offset.get(), fev=fev, rpos=rpos, ophi=ophi, fwhm=fwhm, pos=pos, kmin=kmin,
                 kmax=kmax, skmin=skmin, skmax=skmax, smaa1=smaa1, smaa2=smaa2, smfp=smfp, smfi=smfi, smresult=smresult, smcst=smcst)
    except:
        try:
            ffphi = np.float64(k_offset.get())+fphi
            fk = (2*m*epos*1.602176634*10**-19)**0.5 * \
                np.sin(ffphi/180*np.pi)*10**-10/(h/2/np.pi)
            np.savez(os.path.join(cdir, '.MDC_cut', 'efit.npz'), ko=k_offset.get(), fphi=fphi, epos=epos, ffphi=ffphi, efwhm=efwhm, fk=fk,
                 emin=emin, emax=emax, semin=semin, semax=semax, seaa1=seaa1, seaa2=seaa2, sefp=sefp, sefi=sefi)
        except:
            pass
        pass

    print('k_offset changed')
    st.put('k_offset changed')


@pool_protect
def climon():
    cm.set(h0.get_clim()[0])
    cM.set(h0.get_clim()[1])
    lcmax.config(fg='black')
    lcmin.config(fg='black')
    Cmax.config(from_=cm.get(), to=cM.get(), state='active', fg='black')
    Cmin.config(from_=cm.get(), to=cM.get(), state='active', fg='black')
    vcmin.set(cm.get())
    vcmax.set(cM.get())


@pool_protect
def climoff():
    cm.set(-10000)
    cM.set(10000)
    lcmax.config(fg='white')
    lcmin.config(fg='white')
    Cmax.config(from_=cm.get(), to=cM.get(), state='disabled', fg='white')
    Cmin.config(from_=cm.get(), to=cM.get(), state='disabled', fg='white')
    vcmin.set(cm.get())
    vcmax.set(cM.get())

@pool_protect
def chcmp(*e):
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    a = lcmpd.subplots()
    h = lcmpd.colorbar(mpl.cm.ScalarMappable(
        norm=norm, cmap=value3.get()), cax=a, orientation='vertical', label='')
    h.set_ticks(h.get_ticks())
    h.set_ticklabels(h.get_ticks(), font='Arial')
    cmpg.draw()


@pool_protect
def Chcmp(*e):
    global st, f, out, h0, h1, h2, f0
    limg.config(image=img[np.random.randint(len(img))])
    try:
        if value.get() == 'MDC Normalized':
            plot1()
            print('Colormap changed')
            st.put('Colormap changed')
        else:
            h0.set_cmap(value3.get())
            h0.set_clim([vcmin.get(), vcmax.get()])
            try:
                h1.set_cmap(value3.get())
                h1.set_clim([vcmin.get(), vcmax.get()])
                h2.set_cmap(value3.get())
                h2.set_clim([vcmin.get(), vcmax.get()])
                f.canvas.draw_idle()
                f0.canvas.draw_idle()
            except:
                pass
            out.draw()
            print('Colormap changed')
            st.put('Colormap changed')
    except:
        print('Fail to execute')
        st.put('Fail to execute')


@pool_protect
def o_exptm():
    global name, pos, fwhm, fev, st
    print('Processing...')
    st.put('Processing...')
    # os.chdir(rdd.removesuffix(rdd.split('/')[-1]))
    os.chdir(os.path.dirname(rdd))
    print('export to ',os.path.dirname(rdd))
    ff = open(name+'_mdc_fitted_data.txt', 'w',
              encoding='utf-8')  # tab 必須使用 '\t' 不可"\t"
    ff.write('K.E. (eV)'+'\t'+'FWHM (k)'+'\t'+'Position (k)'+'\n')
    for i in range(len(fev)):
        ff.write(str(fev[i])+'\t'+str(fwhm[i])+'\t'+str(pos[i])+'\n')
    ff.close()
    print('Done')
    st.put('Done')


@pool_protect
def o_expte():
    global name, epos, efwhm, ffphi, st
    print('Processing...')
    st.put('Processing...')
    # os.chdir(rdd.removesuffix(rdd.split('/')[-1]))
    os.chdir(os.path.dirname(rdd))
    print('export to ',os.path.dirname(rdd))
    ff = open(name+'_edc_fitted_data.txt', 'w',
              encoding='utf-8')  # tab 必須使用 '\t' 不可"\t"
    if npzf:ff.write('k (2pi/A)'+'\t'+'FWHM (eV)'+'\t'+'Position (eV)'+'\n')
    else:ff.write('Angle (deg)'+'\t'+'FWHM (eV)'+'\t'+'Position (eV)'+'\n')
    for i in range(len(ffphi)):
        ff.write(str(ffphi[i])+'\t'+str(efwhm[i])+'\t'+str(epos[i])+'\n')
    ff.close()
    print('Done')
    st.put('Done')

@pool_protect
def o_bareband():
    file = fd.askopenfilename(title="Select TXT file",
                              filetypes=(("TXT files", "*.txt"),))
    # global be,k,rx,ry,ix,iy,limg,img
    global be, k, limg, img, st, bpath
    if len(file) > 0:
        bpath = file
        print('Loading...')
        st.put('Loading...')
        # t_k = []
        # t_ke = []
        # with open(file) as f:
        #     for i, line in enumerate(f):
        #         if i != 0:  # ignore 1st row data (index = 0)
        #             t_k.append(line.split('\t')[0])
        #             t_ke.append(line.split('\t')[1].replace('\n', ''))
        try:
            d=np.loadtxt(file,delimiter='\t',encoding='utf-8',dtype=float,skiprows=1,usecols=(0,1))
        except UnicodeError:
            d=np.loadtxt(file,delimiter='\t',encoding='utf-16',dtype=float,skiprows=1,usecols=(0,1))
        t_k = d[:,0]
        t_ke = d[:,1]
        # [::-1] inverse the order for np.interp (xp values should be increasing)
        be = np.float64(t_ke)*1000
        # [::-1] inverse the order for np.interp (xp values should be increasing)
        k = np.float64(t_k)
        os.chdir(cdir)
        np.savez(os.path.join(cdir, '.MDC_cut', 'bb.npz'), path=bpath, be=be, k=k, bbo=float(bb_offset.get()), bbk=float(bbk_offset.get()))
        limg.config(image=img[np.random.randint(len(img))])
        print('Done')
        st.put('Done')
    else:
        limg.config(image=img[np.random.randint(len(img))])
        print('No file selected')
        st.put('No file selected')
        
def im_smooth(data, kernel_size=17):
    return GaussianBlur(data, (kernel_size, kernel_size), 0)

def laplacian_operation(data):
    return -Laplacian(data, CV_64F)

def laplacian_filter(data, kernel_size=17):
    im=im_smooth(data, kernel_size)
    laplacian=laplacian_operation(im)
    return laplacian

@pool_protect
def sdgd_filter(data, phi, ev):
    # not used
    
    # 計算數據的梯度
    grad_phi = np.diff(smooth(data))/np.diff(phi)
    grad_ev = np.diff(smooth(data.transpose(),l=40))/np.diff(ev)
    grad_ev = grad_ev.transpose()

    # 計算梯度方向
    # magnitude = np.sqrt(grad_phi**2 + grad_ev**2)
    # direction = np.arctan2(grad_ev, grad_phi)

    # 計算梯度方向上的二階導數
    grad_phi_phi = np.diff(smooth(grad_phi))/np.diff(phi[0:-1])
    grad_ev_ev = np.diff(smooth(grad_ev.transpose(),l=40))/np.diff(ev[0:-1])
    grad_ev_ev = grad_ev_ev.transpose()
    grad_phi_ev = np.diff(smooth(grad_phi.transpose(),l=40))/np.diff(ev)
    grad_phi_ev = grad_phi_ev.transpose()
    a=grad_phi_phi[0:-2,:]*grad_phi[0:-2,0:-1]**2
    b=2*grad_phi_ev[0:-1,0:-1]*grad_phi[0:-2,0:-1]*grad_ev[0:-1,0:-2]
    c=grad_ev_ev[:,0:-2]*grad_ev[0:-1,0:-2]**2
    # 計算 SDGD
    sdgd = -(a + b + c)/(grad_phi[0:-2,0:-1]**2 + grad_ev[0:-1,0:-2]**2)
    # sdgd = grad_phi_phi[0:-2,:]
    # sdgd = -grad_ev_ev[:,0:-2]
    return sdgd

@pool_protect
def o_plot1(*e):
    global value, value1, value2, data, ev, phi, mfpath, fig, out, pflag, k_offset, value3, limg, img, optionList, h0, ao, xl, yl, st
    if value.get() in optionList:
        try:
            b_sw.grid_remove()
        except:
            pass
        limg.config(image=img[np.random.randint(len(img))])
        print('Plotting...')
        st.put('Plotting...')
        pflag = 1
        value1.set('---Plot2---')
        value2.set('---Plot3---')
        fig.clear()
        try:
            ev
        except:
            print('Please load Raw Data')
            st.put('Please load Raw Data')
        if value.get() == 'Raw Data':
            rplot(fig, out)
        else:
            if value.get() == 'First Derivative':   #axis: phi
                ao = fig.subplots()
                pz = np.diff(smooth(data.to_numpy()))/np.diff(phi)
                if emf=='KE':
                    px, py = np.meshgrid(phi[0:-1]+np.diff(phi)/2, ev)
                    tev = py.copy()
                else:
                    px, py = np.meshgrid(phi[0:-1]+np.diff(phi)/2, vfe-ev)
                    tev = vfe-py.copy()
                if npzf:
                    px = phi[0:-1]+np.diff(phi)/2
                else:
                    px = (2*m*tev*1.602176634*10**-19)**0.5*np.sin((np.float64(k_offset.get())+px+np.diff(phi)/2)/180*np.pi)*10**-10/(h/2/np.pi)
                h0 = ao.pcolormesh(px, py, pz, cmap=value3.get())
                cb = fig.colorbar(h0)
                cb.set_ticklabels(cb.get_ticks(), font='Arial')
                
            # if value.get() == 'First Derivative':    #axis: eV
            #     ao = fig.subplots()
            #     pz = np.diff(smooth(data.to_numpy().transpose()))/np.diff(ev)
            #     pz = pz.transpose()
            #     if emf=='KE':
            #         px, py = np.meshgrid(phi, ev[0:-1]+np.diff(ev)/2)
            #         tev = py.copy()
            #     else:
            #         px, py = np.meshgrid(phi, vfe-ev[0:-1]-np.diff(ev)/2)
            #         tev = vfe-py.copy()
            #     px = (2*m*tev*1.602176634*10**-19)**0.5*np.sin((np.float64(k_offset.get())+px)/180*np.pi)*10**-10/(h/2/np.pi)
            #     h0 = ao.pcolormesh(px, py, pz, cmap=value3.get())
            #     cb = fig.colorbar(h0)
            #     cb.set_ticklabels(cb.get_ticks(), font='Arial')
            
            elif value.get() == 'Second Derivative':    #axis: phi, eV
                ao = fig.subplots()                
                pz = laplacian_filter(data.to_numpy(), im_kernel)
                if emf=='KE':
                    px, py = np.meshgrid(phi, ev)
                    tev = py.copy()
                else:
                    px, py = np.meshgrid(phi, vfe-ev)
                    tev = vfe-py.copy()
                if npzf:
                    px = phi
                else:
                    px = (2*m*tev*1.602176634*10**-19)**0.5*np.sin((np.float64(k_offset.get())+px)/180*np.pi)*10**-10/(h/2/np.pi)
                
                h0 = ao.pcolormesh(px, py, pz, cmap=value3.get())
                cb = fig.colorbar(h0)
                cb.set_ticklabels(cb.get_ticks(), font='Arial')
            else:
                if 'MDC Curves' not in value.get():
                    fig.clear()
                    ao = fig.subplots()
                elif value.get() == 'MDC Curves':
                    fig.clear()
                    ao = fig.add_axes([0.2, 0.13, 0.5, 0.8])
                else:
                    fig.clear()
                    at = fig.add_axes([0.25, 0.13, 0.5, 0.8])
                    at.set_xticks([])
                    at.set_yticks([])
                    ao = fig.add_axes([0.1, 0.13, 0.4, 0.8])
                    ao1 = fig.add_axes([0.5, 0.13, 0.4, 0.8])
                if value.get() == 'E-k Diagram':
                    # h1=a.scatter(mx,my,c=mz,marker='o',s=scale*scale*0.9,cmap=value3.get());
                    if emf=='KE':
                        px, py = np.meshgrid(phi, ev)
                        tev = py.copy()
                    else:
                        px, py = np.meshgrid(phi, vfe-ev)
                        tev = vfe-py.copy()
                    if npzf:
                        px = phi
                    else:
                        px = (2*m*tev*1.602176634*10**-19)**0.5*np.sin((np.float64(k_offset.get())+px)/180*np.pi)*10**-10/(h/2/np.pi)
                    pz = data.to_numpy()
                    h0 = ao.pcolormesh(px, py, pz, cmap=value3.get())
                    cb = fig.colorbar(h0)
                    cb.set_ticklabels(cb.get_ticks(), font='Arial')
                    
                elif value.get() == 'MDC Normalized':
                    pbar = tqdm.tqdm(
                        total=len(ev)-1, desc='MDC Normalized', colour='red')
                    for n in range(len(ev)-1):
                        ecut = data.sel(eV=ev[n], method='nearest')
                        if npzf:
                            x = phi
                        else:
                            x = (2*m*ev[n]*1.602176634*10**-19)**0.5*np.sin(
                            (np.float64(k_offset.get())+phi)/180*np.pi)*10**-10/(h/2/np.pi)
                        y = ecut.to_numpy().reshape(len(ecut))
                        # mz[len(phi)*n:len(phi)*(n+1)]=np.array(y,dtype=float)
                        # mx[len(phi)*n:len(phi)*(n+1)]=x
                        # ty=np.arange(len(x), dtype=float)
                        # my[len(phi)*n:len(phi)*(n+1)]=np.full_like(ty, ev[n])
                        # a.scatter(x,np.full_like(ty, ev[n]),c=np.array(y,dtype=int),marker='o',s=scale*scale*0.9,cmap=value3.get());
                        if emf=='KE':
                            px, py = np.meshgrid(x, ev[n:(n+2)])
                        else:
                            px, py = np.meshgrid(x, vfe-ev[n:(n+2)])
                        ao.pcolormesh(px, py, np.full_like(
                            np.zeros([2, len(phi)], dtype=float), y), cmap=value3.get())
                        pbar.update(1)
                        # print(str(round((n+1)/(len(ev)-1)*100))+'%'+' ('+str(len(ev)-1)+')')
                        st.put(str(round((n+1)/(len(ev)-1)*100)) +
                               '%'+' ('+str(len(ev)-1)+')')
                    pbar.close()
                elif value.get() == 'MDC Curves':
                    pbar = tqdm.tqdm(
                        total=len(ev)//d, desc='MDC', colour='red')
                    y = np.zeros([len(ev),len(phi)],dtype=float)
                    for n in range(len(ev)):
                        ecut = data.sel(eV=ev[n], method='nearest')
                        if npzf:
                            x = phi
                        else:
                            x = (2*m*ev[n]*1.602176634*10**-19)**0.5*np.sin(
                            (np.float64(k_offset.get())+phi)/180*np.pi)*10**-10/(h/2/np.pi)
                        y[n][:] = ecut.to_numpy().reshape(len(ecut))
                    for n in range(len(ev)//d):
                        yy=y[n*d][:]+n*np.max(y)/d
                        yy=smooth(yy,l,p)
                        ao.plot(x, yy, c='black')
                        pbar.update(1)
                        # print(str(round((n+1)/(len(ev))*100))+'%'+' ('+str(len(ev))+')')
                        st.put(str(round((n+1)/(len(ev)//d)*100)) +
                               '%'+' ('+str(len(ev)//d)+')')
                    pbar.close()
                elif value.get() == 'E-k with MDC Curves':
                    pbar = tqdm.tqdm(
                        total=len(ev)//d, desc='MDC', colour='red')
                    y = np.zeros([len(ev),len(phi)],dtype=float)
                    for n in range(len(ev)):
                        ecut = data.sel(eV=ev[n], method='nearest')
                        if npzf:
                            x = phi
                        else:
                            x = (2*m*ev[n]*1.602176634*10**-19)**0.5*np.sin(
                            (np.float64(k_offset.get())+phi)/180*np.pi)*10**-10/(h/2/np.pi)
                        y[n][:] = ecut.to_numpy().reshape(len(ecut))
                    for n in range(len(ev)//d):
                        yy=y[n*d][:]+n*np.max(y)/d
                        yy=smooth(yy,l,p)
                        ao1.plot(x, yy, c='black')
                        pbar.update(1)
                        # print(str(round((n+1)/(len(ev))*100))+'%'+' ('+str(len(ev))+')')
                        st.put(str(round((n+1)/(len(ev)//d)*100)) +
                               '%'+' ('+str(len(ev)//d)+')')
                    pbar.close()
                    if emf=='KE':
                        px, py = np.meshgrid(phi, ev)
                        tev = py.copy()
                    else:
                        px, py = np.meshgrid(phi, vfe-ev)
                        tev = vfe-py.copy()
                    if npzf:
                        px = phi
                    else:
                        px = (2*m*tev*1.602176634*10**-19)**0.5*np.sin((np.float64(k_offset.get())+px)/180*np.pi)*10**-10/(h/2/np.pi)
                    pz = data.to_numpy()
                    h0 = ao.pcolormesh(px, py, pz, cmap=value3.get())
                    ylb=ao1.twinx()
                    ylb.set_ylabel('Intensity (a.u.)', font='Arial', fontsize=size(14))
                    ylb.set_yticklabels([])
                    # cb = fig.colorbar(h0, ax=ao1)
                    # cb.set_ticklabels(cb.get_ticks(), font='Arial')
            if 'E-k with' not in value.get():
                ao.set_title(value.get(), font='Arial', fontsize=size(16))
            else:
                at.set_title(value.get(), font='Arial', fontsize=size(18))
            ao.set_xlabel(r'k ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=size(14))
            if 'MDC Curves' not in value.get():
                if emf=='KE':
                    ao.set_ylabel('Kinetic Energy (eV)', font='Arial', fontsize=size(14))
                else:
                    ao.set_ylabel('Binding Energy (eV)', font='Arial', fontsize=size(14))
                    ao.invert_yaxis()
            else:
                if 'E-k with' in value.get():
                    if emf=='KE':
                        ao.set_ylabel('Kinetic Energy (eV)', font='Arial', fontsize=size(14))
                        ao.set_ylim([ev[0], ev[n*d]])
                    else:
                        ao.set_ylabel('Binding Energy (eV)', font='Arial', fontsize=size(14))
                        ao.invert_yaxis()
                        ao.set_ylim([vfe-ev[0], vfe-ev[n*d]])
                    ao1.set_xlabel(r'k ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=size(14))
                    ao1.set_yticklabels([])
                    ao1.set_xlim([min(x), max(x)])
                    ao1.set_ylim([0, np.max(n*np.max(y)/d)])
                else:
                    ylr=ao.twinx()
                    ao.set_yticklabels([])
                    ao.set_ylabel('Intensity (a.u.)', font='Arial', fontsize=size(14))
                    ylr.set_ylabel(r'$\longleftarrow$ Binding Energy', font='Arial', fontsize=size(14))
                    ylr.set_yticklabels([])
                    ao.set_xlim([min(x), max(x)])
                    ao.set_ylim([0, np.max(n*np.max(y)/d)])
                
            xl = ao.get_xlim()
            yl = ao.get_ylim()
        try:
            if value.get() != 'MDC Normalized' and value.get() != 'MDC Curves':
                climon()
                out.draw()
            else:
                climoff()
                out.draw()
        except:
            pass
        print('Done')
        st.put('Done')
        main_plot_bind()
        gc.collect()


@pool_protect
def o_plot2(*e):
    global fig, out, fwhm, fev, pos, value, value1, value2, k, be, rx, ry, ix, iy, pflag, limg, img, bb_offset, bbk_offset, optionList1, st
    if 'gg' in globals():
        gg.destroy()
    if value1.get() in optionList1:
        try:
            b_sw.grid_remove()
        except:
            pass
        limg.config(image=img[np.random.randint(len(img))])
        print('Plotting...')
        st.put('Plotting...')
        pflag = 2
        value.set('---Plot1---')
        value2.set('---Plot3---')
        fig.clear()
        climoff()
        if value1.get() == 'MDC fitted Data':
            try:
                x = (vfe-fev)*1000
                # y = (fwhm*6.626*10**-34/2/3.1415926/(10**-10))**2/2/(9.11*10**-31)/(1.602176634*10**-19)*1000
            except:
                print(r'Please Load MDC fitted file')
                st.put(r'Please Load MDC fitted file')
            try:
                a = fig.subplots(2, 1)
                a[0].set_title('MDC Fitting Result', font='Arial', fontsize=size(18))
                a[0].set_xlabel('Binding Energy (meV)',
                                font='Arial', fontsize=size(14))
                a[0].set_ylabel(
                    r'Position ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=size(14))
                a[0].tick_params(direction='in')
                a[0].scatter(x, pos, c='black', s=scale*scale*5)

                a[1].set_xlabel('Binding Energy (meV)',
                                font='Arial', fontsize=size(14))
                a[1].set_ylabel(
                    r'FWHM ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=size(14))
                a[1].tick_params(direction='in')
                a[1].scatter(x, fwhm, c='black', s=scale*scale*5)
                
                a[0].invert_xaxis()
                a[1].invert_xaxis()
            except:
                print('Please load MDC fitted file')
                st.put('Please load MDC fitted file')
        elif value1.get() == 'EDC fitted Data':
            try:
                x = fk
            except:
                print(r'Please Load EDC fitted file')
                st.put(r'Please Load EDC fitted file')
            try:
                a = fig.subplots(2, 1)
                a[0].set_title('EDC Fitting Result', font='Arial', fontsize=size(18))
                a[0].set_xlabel(
                    r'Position ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=size(14))
                a[0].set_ylabel('Binding Energy (meV)',
                                font='Arial', fontsize=size(14))
                a[0].tick_params(direction='in')
                a[0].scatter(x, (vfe-epos)*1000, c='black', s=scale*scale*5)

                a[1].set_xlabel(
                    r'Position ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=size(14))
                a[1].set_ylabel('FWHM (meV)', font='Arial', fontsize=size(14))
                a[1].tick_params(direction='in')
                a[1].scatter(x, efwhm*1000, c='black', s=scale*scale*5)
                
                a[0].invert_yaxis()
            except:
                print('Please load EDC fitted file')
                st.put('Please load EDC fitted file')
        elif value1.get() == 'Real Part':
            try:
                x = (vfe-fev)*1000
                y = pos
            except:
                print('Please load MDC fitted file')
                st.put('Please load MDC fitted file')
            try:
                yy = interp(y, k*np.float64(bbk_offset.get()), be -
                            # interp x into be,k set
                            np.float64(bb_offset.get()))
            except:
                print('Please load Bare Band file')
                st.put('Please load Bare Band file')
            a = fig.subplots(2, 1)
            a[0].set_title('Real Part', font='Arial', fontsize=size(18))
            a[0].plot(x, -(x+yy), c='black', linestyle='-', marker='.')

            rx = x
            ry = -(x+yy)
            a[0].tick_params(direction='in')
            a[0].set_xlabel('Binding Energy (meV)', font='Arial', fontsize=size(14))
            a[0].set_ylabel(r'Re $\Sigma$ (meV)', font='Arial', fontsize=size(14))

            h1 = a[1].scatter(y, x, c='black', s=scale*scale*5)
            h2 = a[1].scatter(k*np.float64(bbk_offset.get()),
                              -be+np.float64(bb_offset.get()), c='red', s=scale*scale*5)

            a[1].legend([h1, h2], ['fitted data', 'bare band'])
            a[1].tick_params(direction='in')
            a[1].set_ylabel('Binding Energy (meV)', font='Arial', fontsize=size(14))
            a[1].set_xlabel(
                r'Pos ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=size(14))
            
            a[0].invert_xaxis()
            a[1].invert_yaxis()

            # a[0].set_xlim([-1000,50])
            # a[0].set_ylim([-100,500])
            # a[1].set_ylim([-600,200])
            # a[1].set_xlim([-0.05,0.05])
        elif value1.get() == 'Imaginary Part':
            try:
                tbe = (vfe-fev)*1000
            except:
                print(r'Please Load MDC fitted file')
                st.put(r'Please Load MDC fitted file')
            try:
                x = interp(tbe, -be+np.float64(bb_offset.get()),
                           k*np.float64(bbk_offset.get()))
                y = interp(x, k*np.float64(bbk_offset.get()),
                           -be+np.float64(bb_offset.get()))
            except:
                print('Please load Bare Band file')
                st.put('Please load Bare Band file')
            xx = np.diff(x)
            yy = np.diff(y)

            # eliminate vf in gap
            for i in range(len(yy)):
                if yy[i]/xx[i] > 20000:
                    yy[i] = 0
            v = yy/xx
            # v = np.append(v, v[-1])  # fermi velocity
            try:
                v=interp(pos,x[0:-1]+xx/2,v)
                yy = np.abs(v*fwhm/2)
            except:
                print('Please load MDC fitted file')
                st.put('Please load MDC fitted file')
            xx = tbe
            ax = fig.subplots(2, 1)
            a = ax[0]
            b = ax[1]
            a.set_title('Imaginary Part', font='Arial', fontsize=size(18))
            a.plot(xx, yy, c='black', linestyle='-', marker='.')

            ix = xx
            iy = yy
            a.tick_params(direction='in')
            a.set_xlabel('Binding Energy (eV)', font='Arial', fontsize=size(14))
            a.set_ylabel(r'Im $\Sigma$ (meV)', font='Arial', fontsize=size(14))

            x = (vfe-fev)*1000
            y = fwhm
            b.plot(x, y, c='black', linestyle='-', marker='.')
            b.tick_params(direction='in')
            b.set_xlabel('Binding Energy (eV)', font='Arial', fontsize=size(14))
            b.set_ylabel(r'FWHM ($\frac{2\pi}{\AA}$)',
                         font='Arial', fontsize=size(14))
            
            a.invert_xaxis()
            b.invert_xaxis()
        out.draw()
        print('Done')
        st.put('Done')
        main_plot_bind()
        gc.collect()


@pool_protect
def o_plot3(*e):
    global fig, out, rx, ry, ix, iy, fwhm, pos, value, value1, value2, pflag, k, be, k_offset, value3, limg, img, bb_offset, bbk_offset, optionList2, h0, bo, xl, yl, posmin, posmax, eposmin, eposmax, tb0, tb0_, tb1, tb1_, tb2, st, dl, b_sw
    if value2.get() in optionList2:
        limg.config(image=img[np.random.randint(len(img))])
        print('Plotting...')
        st.put('Plotting...')
        pflag = 3
        value.set('---Plot1---')
        value1.set('---Plot2---')
        fig.clear()
        ophi = np.arcsin(rpos/(2*m*fev*1.602176634*10**-19)**0.5 /
                        10**-10*(h/2/np.pi))*180/np.pi
        pos = (2*m*fev*1.602176634*10**-19)**0.5 * \
            np.sin((np.float64(k_offset.get())+ophi)/180*np.pi)*10**-10/(h/2/np.pi)
        try:
            x = (vfe-fev)*1000
            y = pos
        except:
            print('Please load MDC fitted file')
            st.put('Please load MDC fitted file')
        if 'Data Plot with Pos' in value2.get():
            try:
                b_sw.grid_remove()
            except:
                pass
        else:
            try:
                b_sw.grid(row=0, column=4)
            except:
                pass    
        if value2.get() != 'Data Plot with Pos':
            try:
                yy = interp(y, k*np.float64(bbk_offset.get()), be -
                            # interp x into be,k set
                            np.float64(bb_offset.get()))
                rx = x
                ry = -(x+yy)
                tbe = (vfe-fev)*1000
                x = interp(tbe, -be+np.float64(bb_offset.get()),
                           k*np.float64(bbk_offset.get()))
                y = interp(x, k*np.float64(bbk_offset.get()),
                           -be+np.float64(bb_offset.get()))
                xx = np.diff(x)
                yy = np.diff(y)

                # eliminate vf in gap
                for i in range(len(yy)):
                    if yy[i]/xx[i] > 20000:
                        yy[i] = 0
                v = yy/xx
                # v = np.append(v, v[-1])  # fermi velocity
                v=interp(pos,x[0:-1]+xx/2,v)
                yy = np.abs(v*fwhm/2)
                xx = tbe
                ix = xx
                iy = yy
            except:
                print('Please load Bare Band file')
                st.put('Please load Bare Band file')
        if value2.get() == 'Real & Imaginary':
            a = fig.subplots(2, 1)
            a[0].set_title(r'Self Energy $\Sigma$', font='Arial', fontsize=size(18))
            if dl==0:
                a[0].scatter(rx, ry, edgecolors='black', c='w')
            elif dl==1:
                a[0].plot(rx, ry, c='black')
            elif dl==2:
                a[0].plot(rx, ry, c='black', linestyle='-', marker='.')
            a[0].tick_params(direction='in')
            a[0].set_xlabel('Binding Energy (meV)', font='Arial', fontsize=size(14))
            a[0].set_ylabel(r'Re $\Sigma$ (meV)', font='Arial', fontsize=size(14))
            if dl==0:
                a[1].scatter(ix, iy, edgecolors='black', c='w')
            elif dl==1:
                a[1].plot(ix, iy, c='black')
            elif dl==2:
                a[1].plot(ix, iy, c='black', linestyle='-', marker='.')
            a[1].tick_params(direction='in')
            a[1].set_xlabel('Binding Energy (meV)', font='Arial', fontsize=size(14))
            a[1].set_ylabel(r'Im $\Sigma$ (meV)', font='Arial', fontsize=size(14))
            a[0].invert_xaxis()
            a[1].invert_xaxis()
        elif 'KK Transform' in value2.get():
            ################################################################################## Hilbert Transform
            ##################################################################################
            tbe = (vfe-fev)*1000
            
            ix=(tbe-tbe[-1])*-1
            cix=np.append(ix+ix[0],ix)
            tix=cix[0:len(cix)-1]*-1
            # kx=ix
            kx = np.append(cix,tix[::-1])
            ky = np.linspace(0, 1, len(kx))
            ciy=np.append(iy*0+np.mean(iy),iy)
            tiy=ciy[0:len(ciy)-1]
            ciy = np.append(ciy,tiy[::-1])

            #for imaginary part
            ix=(tbe-tbe[-1])*-1
            cix=np.append(ix+ix[0],ix)
            tix=cix[0:len(cix)-1]*-1
            kx = np.append(cix,tix[::-1])
            ky = np.linspace(0, 1, len(kx))
            cry=np.append(ry*0,ry)
            tcry=cry[0:len(cry)-1]*-1
            cry = np.append(cry,tcry[::-1])

            # Hilbert transform
            analytic_signal_r = hilbert(cry)
            amplitude_envelope_r = np.abs(analytic_signal_r)
            instantaneous_phase_r = np.unwrap(np.angle(analytic_signal_r))
            instantaneous_frequency_r = np.diff(instantaneous_phase_r) / (2.0 * np.pi)

            analytic_signal_i = hilbert(ciy)
            amplitude_envelope_i = np.abs(analytic_signal_i)
            instantaneous_phase_i = np.unwrap(np.angle(analytic_signal_i))
            instantaneous_frequency_i = np.diff(instantaneous_phase_i) / (2.0 * np.pi)

            # Reconstructed real and imaginary parts
            reconstructed_real = np.imag(analytic_signal_i)
            reconstructed_imag = -np.imag(analytic_signal_r)
            ################################################################################## # Export data points as txt files
            ##################################################################################
            
            # np.savetxt('re_sigma.txt', np.column_stack((tbe, ry)), delimiter='\t', header='Binding Energy (meV)\tRe Sigma (meV)', comments='')
            # np.savetxt('kk_re_sigma.txt', np.column_stack((tbe, reconstructed_real[len(ix):2*len(ix)])), delimiter='\t', header='Binding Energy (meV)\tRe Sigma KK (meV)', comments='')
            # np.savetxt('im_sigma.txt', np.column_stack((tbe, iy)), delimiter='\t', header='Binding Energy (meV)\tIm Sigma (meV)', comments='')
            # np.savetxt('kk_im_sigma.txt', np.column_stack((tbe, reconstructed_imag[len(ix):2*len(ix)])), delimiter='\t', header='Binding Energy (meV)\tIm Sigma KK (meV)', comments='')
            
            ##################################################################################
            ################################################################################## # Export data points as txt files
                # Plot
            if 'Real Part' not in value2.get() and 'Imaginary Part' not in value2.get():
                ax = fig.subplots(2, 1)
                a = ax[0]
                b = ax[1]
                # Plot imaginary data and its Hilbert transformation
                a.set_title(r'Self Energy $\Sigma$', font='Arial', fontsize=size(18))
                if dl==0:
                    a.scatter(tbe, ry, edgecolors='black', c='w', label=r'Re $\Sigma$')
                    a.scatter(tbe, reconstructed_real[len(ix):2*len(ix)]+(ry-np.mean(reconstructed_real[len(ix):2*len(ix)])), edgecolors='red', c='w', label=r'Re $\Sigma_{KK}$=KK(Im $\Sigma$)')
                elif dl==1:
                    a.plot(tbe, ry, c='black', label=r'Re $\Sigma$')
                    a.plot(tbe, reconstructed_real[len(ix):2*len(ix)]+(ry-np.mean(reconstructed_real[len(ix):2*len(ix)])), c='red', label=r'Re $\Sigma_{KK}$=KK(Im $\Sigma$)')
                elif dl==2:
                    a.plot(tbe, ry, c='black', linestyle='-', marker='.', label=r'Re $\Sigma$')
                    a.plot(tbe, reconstructed_real[len(ix):2*len(ix)]+(ry-np.mean(reconstructed_real[len(ix):2*len(ix)])), c='red', linestyle='-', marker='.', label=r'Re $\Sigma_{KK}$=KK(Im $\Sigma$)')
                a.set_xlabel('Binding Energy (eV)', font='Arial', fontsize=size(14))
                a.set_ylabel(r'Re $\Sigma$ (meV)', font='Arial', fontsize=size(14))
                a.legend()
                if dl==0:
                    b.scatter(tbe, iy, edgecolors='black', c='w', label=r'Im $\Sigma$')
                    b.scatter(tbe, reconstructed_imag[len(ix):2*len(ix)]+(iy-np.mean(reconstructed_imag[len(ix):2*len(ix)])), edgecolors='red', c='w', label=r'Im $\Sigma_{KK}$=KK(Re $\Sigma$)')
                elif dl==1:
                    b.plot(tbe, iy, c='black', label=r'Im $\Sigma$')
                    b.plot(tbe, reconstructed_imag[len(ix):2*len(ix)]+(iy-np.mean(reconstructed_imag[len(ix):2*len(ix)])), c='red', label=r'Im $\Sigma_{KK}$=KK(Re $\Sigma$)')
                elif dl==2:
                    b.plot(tbe, iy, c='black', linestyle='-', marker='.', label=r'Im $\Sigma$')
                    b.plot(tbe, reconstructed_imag[len(ix):2*len(ix)]+(iy-np.mean(reconstructed_imag[len(ix):2*len(ix)])), c='red', linestyle='-', marker='.', label=r'Im $\Sigma_{KK}$=KK(Re $\Sigma$)')
                b.set_xlabel('Binding Energy (meV)', font='Arial', fontsize=size(14))
                b.set_ylabel(r'Im $\Sigma$ (meV)', font='Arial', fontsize=size(14))
                b.legend()
                a.invert_xaxis()
                b.invert_xaxis()
            elif 'Real Part' in value2.get():
                ax = fig.subplots()
                ttbe=tbe/1000
                if 'nd' in value2.get():
                    ax.set_title(r'Self Energy $\Sigma$ Real Part', font='Arial', fontsize=size(20))
                    ty=np.diff(smooth(ry,20,3))/np.diff(ttbe)
                    np.save(name+'_re_sigma.npy', np.column_stack((ttbe[0:-1], ty)))
                    if dl==0:
                        ax.scatter(ttbe[0:-1], ty, edgecolors='black', c='w', label=r'Re $\Sigma$')
                    elif dl==1:
                        ax.plot(ttbe[0:-1], ty, c='black', label=r'Re $\Sigma$')
                    elif dl==2:
                        ax.plot(ttbe[0:-1], ty, c='black', linestyle='-', marker='.', label=r'Re $\Sigma$')
                    ax.set_xlabel('Binding Energy (eV)', font='Arial', fontsize=size(18))
                    ax.set_ylabel(r'$2^{nd} der. Re \Sigma$', font='Arial', fontsize=size(18))
                    ax.set_xticklabels(ax.get_xticklabels(),fontsize=size(16))
                    ax.set_yticks([0])
                    ax.set_yticklabels(ax.get_yticklabels(),fontsize=size(16))
                else:
                    ax.set_title(r'Self Energy $\Sigma$ Real Part', font='Arial', fontsize=size(20))
                    if dl==0:
                        ax.scatter(ttbe, ry, edgecolors='black', c='w', label=r'Re $\Sigma$')
                        ax.scatter(ttbe, reconstructed_real[len(ix):2*len(ix)]+(ry-np.mean(reconstructed_real[len(ix):2*len(ix)])), edgecolors='red', c='w', label=r'Re $\Sigma_{KK}$=KK(Im $\Sigma$)')
                    elif dl==1:
                        ax.plot(ttbe, ry, c='black', label=r'Re $\Sigma$')
                        ax.plot(ttbe, reconstructed_real[len(ix):2*len(ix)]+(ry-np.mean(reconstructed_real[len(ix):2*len(ix)])), c='red', label=r'Re $\Sigma_{KK}$=KK(Im $\Sigma$)')
                    elif dl==2:
                        ax.plot(ttbe, ry, c='black', linestyle='-', marker='.', label=r'Re $\Sigma$')
                        ax.plot(ttbe, reconstructed_real[len(ix):2*len(ix)]+(ry-np.mean(reconstructed_real[len(ix):2*len(ix)])), c='red', linestyle='-', marker='.', label=r'Re $\Sigma_{KK}$=KK(Im $\Sigma$)')
                    ax.set_xlabel('Binding Energy (eV)', font='Arial', fontsize=size(18))
                    ax.set_ylabel(r'Re $\Sigma$ (meV)', font='Arial', fontsize=size(18))
                    ax.set_xticklabels(ax.get_xticklabels(),fontsize=size(16))
                    ax.set_yticklabels(ax.get_yticklabels(),fontsize=size(16))
                    l=ax.legend(fontsize=size(16))
                    l.draw_frame(False)
                ax.invert_xaxis()
            elif 'Imaginary Part' in value2.get():
                ax = fig.subplots()
                ttbe=tbe/1000
                if 'st' in value2.get():
                    ax.set_title(r'Self Energy $\Sigma$ Imaginary Part', font='Arial', fontsize=size(20))
                    ty=np.diff(smooth(iy,20,3))/np.diff(ttbe)
                    np.save(name+'_im_sigma.npy', np.column_stack((ttbe[0:-1], ty)))
                    if dl==0:
                        ax.scatter(ttbe[0:-1], ty, edgecolors='black', c='w', label=r'Im $\Sigma$')
                    elif dl==1:
                        ax.plot(ttbe[0:-1], ty, c='black', label=r'Im $\Sigma$')
                    elif dl==2:
                        ax.plot(ttbe[0:-1], ty, c='black', linestyle='-', marker='.', label=r'Im $\Sigma$')
                    ax.set_xlabel('Binding Energy (eV)', font='Arial', fontsize=size(18))
                    ax.set_ylabel(r'$1^{st} der. Im \Sigma$', font='Arial', fontsize=size(18))
                    ax.set_xticklabels(ax.get_xticklabels(),fontsize=size(16))
                    ax.set_yticks([0])
                    ax.set_yticklabels(ax.get_yticklabels(),fontsize=size(16))
                else:
                    ax.set_title(r'Self Energy $\Sigma$ Imaginary Part', font='Arial', fontsize=size(20))
                    if dl==0:
                        ax.scatter(ttbe, iy, edgecolors='black', c='w', label=r'Im $\Sigma$')
                        ax.scatter(ttbe, reconstructed_imag[len(ix):2*len(ix)]+(iy-np.mean(reconstructed_imag[len(ix):2*len(ix)])), edgecolors='red', c='w', label=r'Im $\Sigma_{KK}$=KK(Re $\Sigma$)')
                    elif dl==1:
                        ax.plot(ttbe, iy, c='black', label=r'Im $\Sigma$')
                        ax.plot(ttbe, reconstructed_imag[len(ix):2*len(ix)]+(iy-np.mean(reconstructed_imag[len(ix):2*len(ix)])), c='red', label=r'Im $\Sigma_{KK}$=KK(Re $\Sigma$)')
                    elif dl==2:
                        ax.plot(ttbe, iy, c='black', linestyle='-', marker='.', label=r'Im $\Sigma$')
                        ax.plot(ttbe, reconstructed_imag[len(ix):2*len(ix)]+(iy-np.mean(reconstructed_imag[len(ix):2*len(ix)])), c='red', linestyle='-', marker='.', label=r'Im $\Sigma_{KK}$=KK(Re $\Sigma$)')
                    ax.set_xlabel('Binding Energy (eV)', font='Arial', fontsize=size(18))
                    ax.set_ylabel(r'Im $\Sigma$ (meV)', font='Arial', fontsize=size(18))
                    ax.set_xticklabels(ax.get_xticklabels(),fontsize=size(16))
                    ax.set_yticklabels(ax.get_yticklabels(),fontsize=size(16))
                    l=ax.legend(fontsize=size(16))
                    l.draw_frame(False)
                ax.invert_xaxis()
            ##################################################################################
            ################################################################################## Hilbert Transform

        elif value2.get() == 'Data Plot with Pos' or value2.get() == 'Data Plot with Pos and Bare Band':
            bo = fig.subplots()
            if emf=='KE':
                px, py = np.meshgrid(phi, ev)
                tev = py.copy()
            else:
                px, py =np.meshgrid(phi, vfe-ev)
                tev = vfe-py.copy()
            if npzf:
                px = phi
            else:
                px = (2*m*tev*1.602176634*10**-19)**0.5*np.sin((np.float64(k_offset.get())+px)/180*np.pi)*10**-10/(h/2/np.pi)
            pz = data.to_numpy()
            h0 = bo.pcolormesh(px, py, pz, cmap=value3.get())
            txl = bo.get_xlim()
            tyl = bo.get_ylim()
            cb = fig.colorbar(h0)
            # cb.set_ticklabels(cb.get_ticks(), font='Arial', fontsize=size(14), minor=False)
            cb.set_ticklabels(cb.get_ticks(), font='Arial')
            
            #   MDC Norm
            # for i in range(len(ev)):
            #     b.scatter(mx[len(phi)*i:len(phi)*(i+1)],my[len(phi)*i:len(phi)*(i+1)],c=mz[len(phi)*i:len(phi)*(i+1)],marker='o',s=scale*scale*0.9,cmap='viridis',alpha=0.3)
            # a.set_title('MDC Normalized')
            bo.set_title(value2.get(), font='Arial', fontsize=size(18))
            # a.set_xlabel(r'k ($\frac{2\pi}{\AA}$)',fontsize=size(14))
            # a.set_ylabel('Kinetic Energy (eV)',fontsize=size(14))
            bo.set_xlabel(r'k ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=size(16))
            if emf=='KE':
                bo.set_ylabel('Kinetic Energy (eV)', font='Arial', fontsize=size(16))
            else:
                bo.set_ylabel('Binding Energy (eV)', font='Arial', fontsize=size(16))
            # b.set_xticklabels(labels=b.get_xticklabels(),fontsize=size(14))
            # b.set_yticklabels(labels=b.get_yticklabels(),fontsize=size(14))
            try:
                if mp == 1:
                    if emf=='KE':
                        tb0 = bo.scatter(pos, fev, marker='.', s=scale*scale*0.3, c='black')
                    else:
                        tb0 = bo.scatter(pos, vfe-fev, marker='.', s=scale*scale*0.3, c='black')
                if mf == 1:
                    ophimin = np.arcsin(
                        (rpos-fwhm/2)/np.sqrt(2*m*fev*1.602176634*10**-19)/10**-10*(h/2/np.pi))*180/np.pi
                    ophimax = np.arcsin(
                        (rpos+fwhm/2)/np.sqrt(2*m*fev*1.602176634*10**-19)/10**-10*(h/2/np.pi))*180/np.pi
                    posmin = np.sqrt(2*m*fev*1.602176634*10**-19)*np.sin(
                        (np.float64(k_offset.get())+ophimin)/180*np.pi)*10**-10/(h/2/np.pi)
                    posmax = np.sqrt(2*m*fev*1.602176634*10**-19)*np.sin(
                        (np.float64(k_offset.get())+ophimax)/180*np.pi)*10**-10/(h/2/np.pi)
                    if emf=='KE':
                        tb0_ = bo.scatter([posmin, posmax], [
                                        fev, fev], marker='|', c='grey', s=scale*scale*10, alpha=0.8)
                    else:
                        tb0_ = bo.scatter([posmin, posmax], [vfe-fev, vfe-fev], marker='|', c='grey', s=scale*scale*10, alpha=0.8)    
            except:
                pass
            try:
                if ep == 1:
                    if emf=='KE':
                        tb1 = bo.scatter(fk, epos, marker='.', s=scale*scale*0.3, c='black')
                    else:
                        tb1 = bo.scatter(fk, vfe-epos, marker='.', s=scale*scale*0.3, c='black')
                if ef == 1:
                    eposmin = epos-efwhm/2
                    eposmax = epos+efwhm/2
                    if emf=='KE':
                        tb1_ = bo.scatter(
                            [fk, fk], [eposmin, eposmax], marker='_', c='grey', s=scale*scale*10, alpha=0.8)
                    else:
                        tb1_ = bo.scatter(
                            [fk, fk], [vfe-eposmin, vfe-eposmax], marker='_', c='grey', s=scale*scale*10, alpha=0.8)
                    
            except:
                pass
            try:
                if value2.get() == 'Data Plot with Pos and Bare Band':
                    if emf=='KE':
                        tb2, = bo.plot(k*np.float64(bbk_offset.get()), (be -
                                    np.float64(bb_offset.get()))/1000+vfe, linewidth=scale*0.3, c='red', linestyle='--')
                    else:
                        tb2, = bo.plot(k*np.float64(bbk_offset.get()), (-be +
                                np.float64(bb_offset.get()))/1000, linewidth=scale*0.3, c='red', linestyle='--')
                    bo.set_xlim(txl)
                    bo.set_ylim(tyl)
            except:
                bo.set_title('Data Plot with Pos w/o Bare Band',
                             font='Arial', fontsize=size(18))
                print('Please load Bare Band file')
                st.put('Please load Bare Band file')
            if emf=='BE':
                bo.invert_yaxis()
        try:
            if value2.get() != 'Real & Imaginary' and 'KK Transform' not in value2.get():
                xl = bo.get_xlim()
                yl = bo.get_ylim()
                climon()
                out.draw()
            else:
                climoff()
                out.draw()
        except:
            pass
        print('Done')
        st.put('Done')
        main_plot_bind()
        gc.collect()


props = dict(facecolor='green', alpha=0.3)


def select_callback(eclick, erelease):
    """
    Callback for line selection.

    *eclick* and *erelease* are the press and release events.
    """
    global ta0, ta0_, ta1, ta1_, ta2, a, f
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    if eclick.button == 1:
        a.set_xlim(sorted([x1, x2]))
        if emf=='KE':
            a.set_ylim(sorted([y1, y2]))
        else:
            a.set_ylim(sorted([y1, y2], reverse=True))
        f.show()
        if abs(x1-x2) < (xl[1]-xl[0])/3*2 or abs(y1-y2) < (yl[1]-yl[0])/3*2:
            try:
                if mp == 1:
                    ta0.remove()
                if mf == 1:
                    ta0_.remove()
            except:
                pass
            try:
                if ep == 1:
                    ta1.remove()
                if ef == 1:
                    ta1_.remove()
            except:
                pass
            try:
                ta2.remove()
            except:
                pass
            if value2.get() == 'Data Plot with Pos and Bare Band' or value2.get() == 'Data Plot with Pos':
                try:
                    if mp == 1:
                        if emf=='KE':
                            ta0 = a.scatter(pos, fev, marker='.', s=scale*scale*30, c='black')
                        else:
                            ta0 = a.scatter(pos, vfe-fev, marker='.', s=scale*scale*30, c='black')
                    if mf == 1:
                        if emf=='KE':
                            ta0_ = a.scatter([posmin, posmax], [
                                         fev, fev], marker='|', c='grey', s=scale*scale*50, alpha=0.8)
                        else:
                            ta0_ = a.scatter([posmin, posmax], [vfe-fev, vfe-fev], marker='|', c='grey', s=scale*scale*50, alpha=0.8)
                except:
                    pass
                try:
                    if ep == 1:
                        if emf=='KE':
                            ta1 = a.scatter(fk, epos, marker='.', s=scale*scale*30, c='black')
                        else:
                            ta1 = a.scatter(fk, vfe-epos, marker='.', s=scale*scale*30, c='black')
                            
                    if ef == 1:
                        if emf=='KE':
                            ta1_ = a.scatter(
                                [fk, fk], [eposmin, eposmax], marker='_', c='grey', s=scale*scale*50, alpha=0.8)
                        else:
                            ta1_ = a.scatter(
                                [fk, fk], [vfe-eposmin, vfe-eposmax], marker='_', c='grey', s=scale*scale*50, alpha=0.8)
                except:
                    pass

                if value2.get() == 'Data Plot with Pos and Bare Band':
                    if emf=='KE':
                        ta2, = a.plot(k*np.float64(bbk_offset.get()), (be -
                                    np.float64(bb_offset.get()))/1000+vfe, linewidth=scale*2, c='red', linestyle='--')
                    else:
                        ta2, = a.plot(k*np.float64(bbk_offset.get()), (-be +
                                    np.float64(bb_offset.get()))/1000, linewidth=scale*2, c='red', linestyle='--')
            f.show()
        else:
            try:
                if mp == 1:
                    ta0.remove()
                    if emf=='KE':
                        ta0 = a.scatter(pos, fev, marker='.', s=scale*scale*0.3, c='black')
                    else:
                        ta0 = a.scatter(pos, vfe-fev, marker='.', s=scale*scale*0.3, c='black')
                        
                if mf == 1:
                    ta0_.remove()
                    if emf=='KE':
                        ta0_ = a.scatter([posmin, posmax], [fev, fev],
                                        marker='|', c='grey', s=scale*scale*10, alpha=0.8)
                    else:
                        ta0_ = a.scatter([posmin, posmax], [vfe-fev, vfe-fev],
                                        marker='|', c='grey', s=scale*scale*10, alpha=0.8)
            except:
                pass
            try:
                if ep == 1:
                    ta1.remove()
                    if emf=='KE':
                        ta1 = a.scatter(fk, epos, marker='.', s=scale*scale*0.3, c='black')
                    else:
                        ta1 = a.scatter(fk, vfe-epos, marker='.', s=scale*scale*0.3, c='black')
                        
                if ef == 1:
                    ta1_.remove()
                    if emf=='KE':
                        ta1_ = a.scatter([fk, fk], [eposmin, eposmax],
                                     marker='_', c='grey', s=scale*scale*10, alpha=0.8)
                    else:
                        ta1_ = a.scatter([fk, fk], [vfe-eposmin, vfe-eposmax],
                                     marker='_', c='grey', s=scale*scale*10, alpha=0.8)
            except:
                pass
            try:
                if value2.get() == 'Data Plot with Pos and Bare Band':
                    ta2.remove()
                    if emf =='KE':
                        ta2, = a.plot(k*np.float64(bbk_offset.get()), (be -
                                  np.float64(bb_offset.get()))/1000+vfe, linewidth=scale*0.3, c='red', linestyle='--')
                    else:
                        ta2, = a.plot(k*np.float64(bbk_offset.get()), (-be +
                                  np.float64(bb_offset.get()))/1000, linewidth=scale*0.3, c='red', linestyle='--')
            except:
                pass
            f.show()
    else:
        a.set_xlim(xl)
        a.set_ylim(yl)
        try:
            if mp == 1:
                ta0.remove()
                if emf=='KE':
                    ta0 = a.scatter(pos, fev, marker='.', s=scale*scale*0.3, c='black')
                else:
                    ta0 = a.scatter(pos, vfe-fev, marker='.', s=scale*scale*0.3, c='black')
                    
            if mf == 1:
                ta0_.remove()
                if emf=='KE':
                    ta0_ = a.scatter([posmin, posmax], [fev, fev],
                                 marker='|', c='grey', s=scale*scale*10, alpha=0.8)
                else:
                    ta0_ = a.scatter([posmin, posmax], [vfe-fev, vfe-fev],
                                 marker='|', c='grey', s=scale*scale*10, alpha=0.8)
        except:
            pass
        try:
            if ep == 1:
                ta1.remove()
                if emf=='KE':
                    ta1 = a.scatter(fk, epos, marker='.', s=scale*scale*0.3, c='black')
                else:
                    ta1 = a.scatter(fk, vfe-epos, marker='.', s=scale*scale*0.3, c='black')
                    
            if ef == 1:
                ta1_.remove()
                if emf=='KE':
                    ta1_ = a.scatter([fk, fk], [eposmin, eposmax],
                                 marker='_', c='grey', s=scale*scale*10, alpha=0.8)
                else:
                    ta1_ = a.scatter([fk, fk], [vfe-eposmin, vfe-eposmax],
                                 marker='_', c='grey', s=scale*scale*10, alpha=0.8)
        except:
            pass
        try:
            if value2.get() == 'Data Plot with Pos and Bare Band':
                ta2.remove()
                if emf=='KE':
                    ta2, = a.plot(k*np.float64(bbk_offset.get()), (be -
                                np.float64(bb_offset.get()))/1000+vfe, linewidth=scale*0.3, c='red', linestyle='--')
                else:
                    ta2, = a.plot(k*np.float64(bbk_offset.get()), (-be +
                              np.float64(bb_offset.get()))/1000, linewidth=scale*0.3, c='red', linestyle='--')
        except:
            pass
        f.show()
    # print(f"({x1:3.2f}, {y1:3.2f}) --> ({x2:3.2f}, {y2:3.2f})")
    # print(f"The buttons you used were: {eclick.button} {erelease.button}")
# def toggle_selector(event):
#     print('Key pressed.')
#     if event.key == 't':
#         for selector in selectors:
#             name = type(selector).__name__
#             if selector.active:
#                 print(f'{name} deactivated.')
#                 selector.set_active(False)
#             else:
#                 print(f'{name} activated.')
#                 selector.set_active(True)


def cur_move(event):
    global f, a, xx, yy
    if event.inaxes == a and event.xdata is not None and event.ydata is not None:
        f.canvas.get_tk_widget().config(cursor="crosshair")
        try:
            xx.remove()
            yy.remove()
        except:
            pass
        xx=a.axvline(event.xdata, color='red')
        yy=a.axhline(event.ydata, color='red')
    f.show()
    

def cur_on_move(event):
    if event.inaxes == a and event.xdata is not None and event.ydata is not None:
        annot.xy = (event.xdata, event.ydata)
        text = f"x={event.xdata:.3f}\ny={event.ydata:.3f}"
        annot.set_text(text)
        # 取得座標軸範圍
        xlim = a.get_xlim()
        ylim = a.get_ylim()
        # 設定 annotation 方向
        offset_x, offset_y = 20, 20
        # 靠近右邊界
        if event.xdata > xlim[1] - (xlim[1]-xlim[0])*0.15:
            offset_x = -60
        # 靠近左邊界
        elif event.xdata < xlim[0] + (xlim[1]-xlim[0])*0.15:
            offset_x = 20
        # 靠近上邊界
        if event.ydata > ylim[1] - (ylim[1]-ylim[0])*0.15:
            offset_y = -40
        # 靠近下邊界
        elif event.ydata < ylim[0] + (ylim[1]-ylim[0])*0.15:
            offset_y = 20
        annot.set_position((offset_x, offset_y))
        annot.set_visible(True)
        f.canvas.draw_idle()
    else:
        annot.set_visible(False)
        f.canvas.draw_idle()

def onselect(xmin, xmax):
    global f, f0, h1, h2
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    # vcmin.set(xmin)
    # vcmax.set(xmax)
    h2.set_clim(xmin, xmax)
    # f0.canvas.draw_idle()
    f0.show()
    h1.set_clim(xmin, xmax)
    # f.canvas.draw_idle()
    f.show()


def onmove_callback(xmin, xmax):
    global f, f0, h1, h2
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    # vcmin.set(xmin)
    # vcmax.set(xmax)
    h2.set_clim(xmin, xmax)
    # f0.canvas.draw_idle()
    f0.show()
    h1.set_clim(xmin, xmax)
    # f.canvas.draw_idle()
    f.show()

cf = True

def cut_move(event):
    global cxdata, cydata, acx, acy, a, f, xx ,yy
    # ,x,y
    f.canvas.get_tk_widget().config(cursor="")
    if event.inaxes:
        cxdata = event.xdata
        cydata = event.ydata
        xf = (cxdata >= a.get_xlim()[0] and cxdata <= a.get_xlim()[1])
        if emf=='KE':
            yf = (cydata >= a.get_ylim()[0] and cydata <= a.get_ylim()[1])
        else:
            yf = (cydata <= a.get_ylim()[0] and cydata >= a.get_ylim()[1])
        if xf and yf:
            f.canvas.get_tk_widget().config(cursor="crosshair")
            try:
                xx.remove()
                yy.remove()
            except:
                pass
            xx=a.axvline(cxdata,color='r')
            yy=a.axhline(cydata,color='r')
            if cf:
                if emf=='KE':
                    dx = data.sel(
                        eV=cydata, method='nearest').to_numpy().reshape(len(phi))
                else:
                    dx = data.sel(eV=vfe-cydata, method='nearest').to_numpy().reshape(len(phi))
                dy = data.sel(
                    phi=cxdata, method='nearest').to_numpy().reshape(len(ev))
                acx.clear()
                acy.clear()
                acx.set_title('                Raw Data', font='Arial', fontsize=size(18))
                acx.plot(phi, dx, c='black')
                if emf=='KE':
                    acy.plot(dy, ev, c='black')
                else:
                    acy.plot(dy, vfe-ev, c='black')
                acx.set_xticks([])
                acy.set_yticks([])
                acx.set_xlim(a.get_xlim())
                acy.set_ylim(a.get_ylim())
                # f.canvas.draw_idle()
    else:
        try:
            if cf:
                acx.clear()
                acy.clear()
                acx.set_title('                Raw Data', font='Arial', fontsize=size(18))
                acx.set_xticks([])
                acx.set_yticks([])
                acy.set_xticks([])
                acy.set_yticks([])
            xx.remove()
            yy.remove()
        except:
            pass
    f.show()

def cut_select(event):
    global cf, a, f, x, y, acx, acy
    if event.button == 1 and cf:
        cf = False
        x = a.axvline(event.xdata, color='red')
        y = a.axhline(event.ydata, color='red')
    elif event.button == 1 and not cf:
        x.remove()
        y.remove()
        x = a.axvline(event.xdata, color='red')
        y = a.axhline(event.ydata, color='red')
        if emf=='KE':
            dx = data.sel(eV=event.ydata,
                        method='nearest').to_numpy().reshape(len(phi))
        else:
            dx = data.sel(eV=vfe-event.ydata,
                        method='nearest').to_numpy().reshape(len(phi))
        dy = data.sel(phi=event.xdata,
                      method='nearest').to_numpy().reshape(len(ev))
        acx.clear()
        acy.clear()
        acx.set_title('                Raw Data', font='Arial', fontsize=size(18))
        acx.plot(phi, dx, c='black')
        if emf=='KE':
            acy.plot(dy, ev, c='black')
        else:
            acy.plot(dy, vfe-ev, c='black')
        acx.set_xticks([])
        acy.set_yticks([])
        acx.set_xlim(a.get_xlim())
        acy.set_ylim(a.get_ylim())

    elif event.button == 3:
        cf = True
        x.remove()
        y.remove()
    # f.canvas.draw_idle()
    copy_to_clipboard(ff=f)
    f.show()

def exp(*e):
    global value, value1, value2, value3, data, ev, phi, mx, my, mz, mfpath, fev, fwhm, pos, k, be, rx, ry, ix, iy, pflag, k_offset, limg, img, bb_offset, bbk_offset, h1, h2, a0, a, b, f0, f, selectors, acx, acy, posmin, posmax, eposmin, eposmax, annot
    if data is None:
        st.put('No data loaded!')
        messagebox.showwarning("Warning","No data loaded!")
        return
    limg.config(image=img[np.random.randint(len(img))])
    selectors = []
    cursor = []
    h1 = []
    h2 = []
    f = []
    f0 = []
    try:
        if pflag:
            pass
    except NameError:
        print('Choose a plot type first')
        st.put('Choose a plot type first')
        return
    if pflag == 1:
        if 'MDC Curves' not in value.get():
            mz = data.to_numpy()
            f0 = plt.figure(figsize=(8*scale, 7*scale), layout='constrained')
            a0 = plt.axes([0.13, 0.45, 0.8, 0.5])
            a1 = plt.axes([0.13, 0.08, 0.8, 0.2])
            a0.set_title('Drag to select specific region', font='Arial', fontsize=size(18))
            selectors.append(RectangleSelector(
                a0, select_callback,
                useblit=True,
                button=[1, 3],  # disable middle button
                minspanx=5, minspany=5,
                spancoords='pixels',
                interactive=True,
                props=props))
            # f0.canvas.mpl_connect('key_press_event',toggle_selector)
        if value.get() != 'Raw Data' and 'MDC Curves' not in value.get():
            f, a = plt.subplots(dpi=150)
        elif value.get() == 'MDC Curves':
            f=plt.figure(figsize=(4*scale, 6*scale),dpi=150)
            a = f.subplots()
        elif value.get() == 'E-k with MDC Curves':
            f = plt.figure(figsize=(9*scale, 7*scale), layout='constrained')
            at_ = plt.axes([0.28, 0.15, 0.5, 0.75])
            at_.set_xticks([])
            at_.set_yticks([])
            a = plt.axes([0.13, 0.15, 0.4, 0.75])
            a1_ = plt.axes([0.53, 0.15, 0.4, 0.75])
        if value.get() == 'Raw Data':
            f = plt.figure(figsize=(9*scale, 7*scale), layout='constrained')
            a = plt.axes([0.13, 0.1, 0.55, 0.6])
            acx = plt.axes([0.13, 0.73, 0.55, 0.18])
            acy = plt.axes([0.7, 0.1, 0.15, 0.6])
            eacb = plt.axes([0.87, 0.1, 0.02, 0.6])
            plt.connect('motion_notify_event', cut_move)
            plt.connect('button_press_event', cut_select)
            if emf=='KE':
                mx, my = np.meshgrid(phi, ev)
            else:
                mx, my = np.meshgrid(phi, vfe-ev)
            # h1 = a.scatter(mx,my,c=mz,marker='o',s=scale*scale*0.9,cmap=value3.get());
            h1 = a.pcolormesh(mx, my, mz, cmap=value3.get())
            annot = a.annotate(
                "", xy=(0,0), xytext=(20,20), textcoords="offset points",
                bbox=dict(boxstyle="round", fc="w", alpha=0.6),
                fontsize=size(14)
                # fontsize=size(12),
                # arrowprops=dict(arrowstyle="->")
            )
            annot.set_visible(False)
            f.canvas.mpl_connect('motion_notify_event', cur_on_move)
            if emf=='KE':
                yl = a.get_ylim()
            else:
                yl = sorted(a.get_ylim(), reverse=True)
            cb = f.colorbar(h1, cax=eacb, orientation='vertical')
            # cb.set_ticklabels(cb.get_ticks(), font='Arial', fontsize=size(14))
            
            h2 = a0.pcolormesh(mx, my, mz, cmap=value3.get())
            cb1 = f0.colorbar(h2)
            # cb1.set_ticklabels(cb.get_ticks(), font='Arial', fontsize=size(14))

            acx.set_xticks([])
            acx.set_yticks([])
            acy.set_xticks([])
            acy.set_yticks([])
            
            n = a1.hist(mz.flatten(), bins=np.linspace(
                min(mz.flatten()), max(mz.flatten()), 50), color='green')
            a1.set_xlabel('Intensity')
            a1.set_ylabel('Counts')
            a1.set_title('Drag to Select the range of Intensity ')
            selectors.append(SpanSelector(
                a1,
                onselect,
                "horizontal",
                useblit=True,
                props=dict(alpha=0.3, facecolor="tab:blue"),
                onmove_callback=onmove_callback,
                interactive=True,
                drag_from_anywhere=True,
                snap_values=n[1]
            ))
        elif value.get() == 'First Derivative':
            pz = np.diff(smooth(data.to_numpy()))/np.diff(phi)
            if emf=='KE':
                px, py = np.meshgrid(phi[0:-1], ev)
                tev = py.copy()
            else:
                px, py = np.meshgrid(phi[0:-1], vfe-ev)
                tev = vfe-py.copy()
            if npzf:
                px = phi[0:-1]+np.diff(phi)/2
            else:
                px = (2*m*tev*1.602176634*10**-19)**0.5*np.sin((np.float64(k_offset.get())+px+np.diff(phi)/2)/180*np.pi)*10**-10/(h/2/np.pi)
            h1 = a.pcolormesh(px, py, pz, cmap=value3.get())
            # cursor = Cursor(a, useblit=True, color='red', linewidth=scale*1)
            annot = a.annotate(
                "", xy=(0,0), xytext=(20,20), textcoords="offset points",
                bbox=dict(boxstyle="round", fc="w", alpha=0.6),
                fontsize=size(12)
                # fontsize=size(12),
                # arrowprops=dict(arrowstyle="->")
            )
            annot.set_visible(False)
            f.canvas.mpl_connect('motion_notify_event', cur_move)
            f.canvas.mpl_connect('motion_notify_event', cur_on_move)
            if emf=='KE':
                yl = a.get_ylim()
            else:
                yl = sorted(a.get_ylim(), reverse=True)
            h2 = a0.pcolormesh(px, py, pz, cmap=value3.get())
            cb = f.colorbar(h1)
            # cb.set_ticklabels(cb.get_ticks(), font='Arial', fontsize=size(14))
            
            cb1 = f0.colorbar(h2)
            # cb1.set_ticklabels(cb1.get_ticks(), font='Arial', fontsize=size(14))

            n = a1.hist(pz.flatten(), bins=np.linspace(
                min(pz.flatten()), max(pz.flatten()), 50), color='green')
            a1.set_xlabel('Intensity')
            a1.set_ylabel('Counts')
            a1.set_title('Drag to Select the range of Intensity ')
            selectors.append(SpanSelector(
                a1,
                onselect,
                "horizontal",
                useblit=True,
                props=dict(alpha=0.3, facecolor="tab:blue"),
                onmove_callback=onmove_callback,
                interactive=True,
                drag_from_anywhere=True,
                snap_values=n[1]
            ))
        elif value.get() == 'Second Derivative':            
            pz = laplacian_filter(data.to_numpy(), im_kernel)
            if emf=='KE':
                px, py = np.meshgrid(phi, ev)
                tev = py.copy()
            else:
                px, py = np.meshgrid(phi, vfe-ev)
                tev = vfe-py.copy()
            if npzf:
                px = phi
            else:
                px = (2*m*tev*1.602176634*10**-19)**0.5*np.sin((np.float64(k_offset.get())+px)/180*np.pi)*10**-10/(h/2/np.pi)
            
            h1 = a.pcolormesh(px, py, pz, cmap=value3.get())
            # cursor = Cursor(a, useblit=True, color='red', linewidth=scale*1)
            annot = a.annotate(
                "", xy=(0,0), xytext=(20,20), textcoords="offset points",
                bbox=dict(boxstyle="round", fc="w", alpha=0.6),
                fontsize=size(12)
                # fontsize=size(12),
                # arrowprops=dict(arrowstyle="->")
            )
            annot.set_visible(False)
            f.canvas.mpl_connect('motion_notify_event', cur_move)
            f.canvas.mpl_connect('motion_notify_event', cur_on_move)
            if emf=='KE':
                yl = a.get_ylim()
            else:
                yl = sorted(a.get_ylim(), reverse=True)
            h2 = a0.pcolormesh(px, py, pz, cmap=value3.get())
            cb = f.colorbar(h1)
            # cb.set_ticklabels(cb.get_ticks(), font='Arial', fontsize=size(14))
            
            cb1 = f0.colorbar(h2)
            # cb1.set_ticklabels(cb1.get_ticks(), font='Arial', fontsize=size(14))

            n = a1.hist(pz.flatten(), bins=np.linspace(
                min(pz.flatten()), max(pz.flatten()), 50), color='green')
            a1.set_xlabel('Intensity')
            a1.set_ylabel('Counts')
            a1.set_title('Drag to Select the range of Intensity ')
            selectors.append(SpanSelector(
                a1,
                onselect,
                "horizontal",
                useblit=True,
                props=dict(alpha=0.3, facecolor="tab:blue"),
                onmove_callback=onmove_callback,
                interactive=True,
                drag_from_anywhere=True,
                snap_values=n[1]
            ))
        else:
            if value.get() == 'E-k Diagram':
                # h1=a.scatter(mx,my,c=mz,marker='o',s=scale*scale*0.9,cmap=value3.get());
                if emf=='KE':
                    px, py = np.meshgrid(phi, ev)
                    tev = py.copy()
                else:
                    px, py = np.meshgrid(phi, vfe-ev)
                    tev = vfe-py.copy()
                if npzf:
                    px = phi
                else:
                    px = (2*m*tev*1.602176634*10**-19)**0.5*np.sin((np.float64(k_offset.get())+px)/180*np.pi)*10**-10/(h/2/np.pi)
                pz = data.to_numpy()
                h1 = a.pcolormesh(px, py, pz, cmap=value3.get())
                # cursor = Cursor(a, useblit=True, color='red', linewidth=scale*1)
                annot = a.annotate(
                    "", xy=(0,0), xytext=(20,20), textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w", alpha=0.6),
                    fontsize=size(12)
                    # fontsize=size(12),
                    # arrowprops=dict(arrowstyle="->")
                )
                annot.set_visible(False)
                f.canvas.mpl_connect('motion_notify_event', cur_move)
                f.canvas.mpl_connect('motion_notify_event', cur_on_move)
                if emf=='KE':
                    yl = a.get_ylim()
                else:
                    yl = sorted(a.get_ylim(), reverse=True)
                h2 = a0.pcolormesh(px, py, pz, cmap=value3.get())
                cb = f.colorbar(h1)
                # cb.set_ticklabels(cb.get_ticks(), font='Arial', fontsize=size(14))
                
                cb1 = f0.colorbar(h2)
                # cb1.set_ticklabels(cb1.get_ticks(), font='Arial', fontsize=size(14))
                

                n = a1.hist(pz.flatten(), bins=np.linspace(
                    min(pz.flatten()), max(pz.flatten()), 50), color='green')
                a1.set_xlabel('Intensity')
                a1.set_ylabel('Counts')
                a1.set_title('Drag to Select the range of Intensity ')
                selectors.append(SpanSelector(
                    a1,
                    onselect,
                    "horizontal",
                    useblit=True,
                    props=dict(alpha=0.3, facecolor="tab:blue"),
                    onmove_callback=onmove_callback,
                    interactive=True,
                    drag_from_anywhere=True,
                    snap_values=n[1]
                ))
            elif value.get() == 'MDC Normalized':
                for n in range(len(ev)-1):
                    ecut = data.sel(eV=ev[n], method='nearest')
                    if npzf:
                        x = phi
                    else:
                        x = (2*m*ev[n]*1.602176634*10**-19)**0.5*np.sin(
                        (np.float64(k_offset.get())+phi)/180*np.pi)*10**-10/(h/2/np.pi)
                    y = ecut.to_numpy().reshape(len(ecut))
                    # mz[len(phi)*n:len(phi)*(n+1)]=np.array(y,dtype=float)
                    # mx[len(phi)*n:len(phi)*(n+1)]=x
                    # ty=np.arange(len(x), dtype=float)
                    # my[len(phi)*n:len(phi)*(n+1)]=np.full_like(ty, ev[n])
                    # a.scatter(x,np.full_like(ty, ev[n]),c=np.array(y,dtype=int),marker='o',s=scale*scale*0.9,cmap=value3.get());
                    if emf=='KE':
                        px, py = np.meshgrid(x, ev[n:n+2])
                    else:
                        px, py = np.meshgrid(x, vfe-ev[n:n+2])
                    a.pcolormesh(px, py, np.full_like(
                        np.zeros([2, len(phi)], dtype=float), y), cmap=value3.get())
                    if emf=='KE':
                        yl = a.get_ylim()
                    else:
                        yl = sorted(a.get_ylim(), reverse=True)
                    a0.pcolormesh(px, py, np.full_like(
                        np.zeros([2, len(phi)], dtype=float), y), cmap=value3.get())
                # cursor = Cursor(a, useblit=True, color='red', linewidth=scale*1)
                annot = a.annotate(
                    "", xy=(0,0), xytext=(20,20), textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w", alpha=0.6),
                    fontsize=size(12)
                    # fontsize=size(12),
                    # arrowprops=dict(arrowstyle="->")
                )
                annot.set_visible(False)
                f.canvas.mpl_connect('motion_notify_event', cur_move)
                f.canvas.mpl_connect('motion_notify_event', cur_on_move)
            elif value.get() == 'MDC Curves':
                y = np.zeros([len(ev),len(phi)],dtype=float)
                for n in range(len(ev)):
                    ecut = data.sel(eV=ev[n], method='nearest')
                    if npzf:
                        x = phi
                    else:
                        x = (2*m*ev[n]*1.602176634*10**-19)**0.5*np.sin(
                        (np.float64(k_offset.get())+phi)/180*np.pi)*10**-10/(h/2/np.pi)
                    y[n][:] = ecut.to_numpy().reshape(len(ecut))
                for n in range(len(ev)//d):
                    yy=y[n*d][:]+n*np.max(y)/d
                    yy=smooth(yy,l,p)
                    a.plot(x, yy, c='black')
            elif value.get() == 'E-k with MDC Curves':
                    y = np.zeros([len(ev),len(phi)],dtype=float)
                    for n in range(len(ev)):
                        ecut = data.sel(eV=ev[n], method='nearest')
                        if npzf:
                            x = phi
                        else:
                            x = (2*m*ev[n]*1.602176634*10**-19)**0.5*np.sin(
                            (np.float64(k_offset.get())+phi)/180*np.pi)*10**-10/(h/2/np.pi)
                        y[n][:] = ecut.to_numpy().reshape(len(ecut))
                    for n in range(len(ev)//d):
                        yy=y[n*d][:]+n*np.max(y)/d
                        yy=smooth(yy,l,p)
                        a1_.plot(x, yy, c='black')
                    if emf=='KE':
                        px, py = np.meshgrid(phi, ev)
                        tev = py.copy()
                    else:
                        px, py = np.meshgrid(phi, vfe-ev)
                        tev = vfe-py.copy()
                    if npzf:
                        px = phi
                    else:
                        px = (2*m*tev*1.602176634*10**-19)**0.5*np.sin((np.float64(k_offset.get())+px)/180*np.pi)*10**-10/(h/2/np.pi)
                    pz = data.to_numpy()
                    h1 = a.pcolormesh(px, py, pz, cmap=value3.get())
                    ylb=a1_.twinx()
                    ylb.set_ylabel('Intensity (a.u.)', font='Arial', fontsize=size(22))
                    ylb.set_yticklabels([])
                    # cb = fig.colorbar(h1, ax=a1_)
                    # cb.set_ticklabels(cb.get_ticks(), font='Arial', fontsize=size(20))
        if 'E-k with' not in value.get():
            if  value.get() != 'Raw Data':
                a.set_title(value.get(), font='Arial', fontsize=size(18))
        else:
            at_.set_title(value.get(), font='Arial', fontsize=size(24))
        a.set_xlabel(r'k ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=size(16))
        # a.set_xticklabels(labels=a.get_xticklabels(), fontsize=size(20))
        if 'MDC Curves' not in value.get():
            a0.set_xlabel(r'k ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=size(16))
            # a0.set_xticklabels(labels=a0.get_xticklabels(), fontsize=size(14))
            if emf=='KE':
                a.set_ylabel('Kinetic Energy (eV)', font='Arial', fontsize=size(16))
                # a.set_yticklabels(labels=a.get_yticklabels(), fontsize=size(20))
                a0.set_ylabel('Kinetic Energy (eV)', font='Arial', fontsize=size(16))
                # a0.set_yticklabels(labels=a0.get_yticklabels(), fontsize=size(14))
                if value.get() == 'Raw Data':
                    a.set_ylabel('Kinetic Energy (eV)', font='Arial', fontsize=size(16))
                    # a.set_yticklabels(labels=a.get_yticklabels(), fontsize=size(14))
                    a0.set_ylabel('Kinetic Energy (eV)', font='Arial', fontsize=size(16))
                    # a0.set_yticklabels(labels=a0.get_yticklabels(), fontsize=size(14))
            else:
                a.set_ylabel('Binding Energy (eV)', font='Arial', fontsize=size(16))
                # a.set_yticklabels(labels=a.get_yticklabels(), fontsize=size(20))
                a0.set_ylabel('Binding Energy (eV)', font='Arial', fontsize=size(16))
                # a0.set_yticklabels(labels=a0.get_yticklabels(), fontsize=size(14))
                if value.get() == 'Raw Data':
                    a.set_ylabel('Binding Energy (eV)', font='Arial', fontsize=size(16))
                    # a.set_yticklabels(labels=a.get_yticklabels(), fontsize=size(14))
                    a0.set_ylabel('Binding Energy (eV)', font='Arial', fontsize=size(16))
                    # a0.set_yticklabels(labels=a0.get_yticklabels(), fontsize=size(14))
                a.invert_yaxis()
                a0.invert_yaxis()
        else:
            if 'E-k with' in value.get():
                if emf=='KE':
                    a.set_ylabel('Kinetic Energy (eV)', font='Arial', fontsize=size(22))
                    a.set_yticklabels(labels=a.get_yticklabels(), fontsize=size(20))
                    a.set_ylim([ev[0], ev[n*d]])
                else:
                    a.set_ylabel('Binding Energy (eV)', font='Arial', fontsize=size(22))
                    a.set_yticklabels(labels=a.get_yticklabels(), fontsize=size(20))
                    a.invert_yaxis()
                    a.set_ylim([vfe-ev[0], vfe-ev[n*d]])
                a.set_xlabel(r'k ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=size(22))
                a.set_xticklabels(labels=a.get_xticklabels(), fontsize=size(20))
                a1_.set_xlabel(r'k ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=size(22))
                a1_.set_xticklabels(labels=a1_.get_xticklabels(), fontsize=size(20))
                a1_.set_yticklabels([])
                a1_.set_xlim([min(x), max(x)])
                a1_.set_ylim([0, np.max(n*np.max(y)/d)])
            else:
                ylr=a.twinx()
                a.set_ylabel('Intensity (a.u.)', font='Arial', fontsize=size(22))
                a.set_yticklabels([])
                ylr.set_ylabel(r'$\longleftarrow$ Binding Energy', font='Arial', fontsize=size(22))
                ylr.set_yticklabels([])
                a.set_xlim([min(x), max(x)])
                a.set_ylim([0, np.max(n*np.max(y)/d)])
        if value.get() == 'Raw Data':
            acx.set_title('                Raw Data', font='Arial', fontsize=size(18))
            # cb.set_ticklabels(cb.get_ticks(), font='Arial', fontsize=size(14))
            if npzf:
                a.set_xlabel(r'k ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=size(16))
                a0.set_xlabel(r'k ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=size(16))
            else:
                a.set_xlabel('Angle (deg)', font='Arial', fontsize=size(16))
                a0.set_xlabel('Angle (deg)', font='Arial', fontsize=size(16))
            # a.set_xticklabels(labels=a.get_xticklabels(), fontsize=size(14))
            # a0.set_xticklabels(labels=a0.get_xticklabels(), fontsize=size(14))
        # a.set_xticklabels(labels=a.get_xticklabels(),fontsize=size(10))
        # a.set_yticklabels(labels=a.get_yticklabels(),fontsize=size(10))
    if pflag == 2:
        f, a = plt.subplots(2, 1, dpi=150)
        if value1.get() == 'MDC fitted Data':
            x = (vfe-fev)*1000

            a[0].set_title('MDC Fitting Result', font='Arial', fontsize=size(24))
            a[0].set_xlabel('Binding Energy (meV)', font='Arial', fontsize=size(22))
            a[0].set_xticklabels(labels=a[0].get_xticklabels(), fontsize=size(20))
            a[0].set_ylabel(
                r'Position ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=size(22))
            a[0].set_yticklabels(labels=a[0].get_yticklabels(), fontsize=size(20))
            a[0].tick_params(direction='in')
            a[0].scatter(x, pos, c='black', s=scale*scale*5)

            a[1].set_xlabel('Binding Energy (meV)', font='Arial', fontsize=size(22))
            a[1].set_xticklabels(labels=a[1].get_xticklabels(), fontsize=size(20))
            a[1].set_ylabel(
                r'FWHM ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=size(22))
            a[1].set_yticklabels(labels=a[1].get_yticklabels(), fontsize=size(20))
            a[1].tick_params(direction='in')
            a[1].scatter(x, fwhm, c='black', s=scale*scale*5)
            
            a[0].invert_xaxis()
            a[1].invert_xaxis()
        elif value1.get() == 'EDC fitted Data':
            x = fk

            a[0].set_title('EDC Fitting Result', font='Arial', fontsize=size(24))
            a[0].set_xlabel(
                r'Position ($\frac{2\pi}{\AA}$', font='Arial', fontsize=size(22))
            a[0].set_xticklabels(labels=a[0].get_xticklabels(), fontsize=size(20))
            a[0].set_ylabel('Binding Energy (meV))', font='Arial', fontsize=size(22))
            a[0].set_yticklabels(labels=a[0].get_yticklabels(), fontsize=size(20))
            a[0].tick_params(direction='in')
            a[0].scatter(x, (vfe-epos)*1000, c='black', s=scale*scale*5)

            a[1].set_xlabel(
                r'Position ($\frac{2\pi}{\AA}$', font='Arial', fontsize=size(22))
            a[1].set_xticklabels(labels=a[1].get_xticklabels(), fontsize=size(20))
            a[1].set_ylabel('FWHM (meV)', font='Arial', fontsize=size(22))
            a[1].set_yticklabels(labels=a[1].get_yticklabels(), fontsize=size(20))
            a[1].tick_params(direction='in')
            a[1].scatter(x, efwhm*1000, c='black', s=scale*scale*5)
            
            a[0].invert_yaxis()
            
        elif value1.get() == 'Real Part':
            x = (vfe-fev)*1000
            y = pos
            a[0].set_title('Real Part', font='Arial', fontsize=size(24))
            a[0].plot(rx, ry, c='black', linestyle='-', marker='.')

            a[0].tick_params(direction='in')
            a[0].set_xlabel('Binding Energy (meV)', font='Arial', fontsize=size(22))
            a[0].set_xticklabels(labels=a[0].get_xticklabels(), fontsize=size(20))
            a[0].set_ylabel(r'Re $\Sigma$ (meV)', font='Arial', fontsize=size(22))
            a[0].set_yticklabels(labels=a[0].get_yticklabels(), fontsize=size(20))

            h1 = a[1].scatter(y, x, c='black', s=scale*scale*5)
            h2 = a[1].scatter(k*np.float64(bbk_offset.get()),
                              -be+np.float64(bb_offset.get()), c='red', s=scale*scale*5)

            a[1].legend([h1, h2], ['fitted data', 'bare band'],fontsize=size(20))
            a[1].tick_params(direction='in')
            a[1].set_ylabel('Binding Energy (meV)', font='Arial', fontsize=size(22))
            a[1].set_yticklabels(labels=a[1].get_yticklabels(), fontsize=size(20))
            a[1].set_xlabel(
                r'Pos ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=size(22))
            a[1].set_xticklabels(labels=a[1].get_xticklabels(), fontsize=size(20))
            
            a[0].invert_xaxis()
            a[1].invert_yaxis()

            # a[0].set_xlim([-1000,50])
            # a[0].set_ylim([-100,500])
            # a[1].set_ylim([-600,200])
            # a[1].set_xlim([-0.05,0.05])
        elif value1.get() == 'Imaginary Part':

            tbe = (vfe-fev)*1000

            x = interp(tbe, -be+np.float64(bb_offset.get()),
                       k*np.float64(bbk_offset.get()))
            y = interp(x, k*np.float64(bbk_offset.get()),
                       -be+np.float64(bb_offset.get()))
            xx = np.diff(x)
            yy = np.diff(y)

            # eliminate vf in gap
            for i in range(len(yy)):
                if yy[i]/xx[i] > 20000:
                    yy[i] = 0
            v = yy/xx
            # v = np.append(v, v[-1])  # fermi velocity
            v=interp(pos,x[0:-1]+xx/2,v)
            yy = np.abs(v*fwhm/2)
            xx = tbe
            ax = a
            a = ax[0]
            b = ax[1]
            a.set_title('Imaginary Part', font='Arial', fontsize=size(24))
            a.plot(xx, yy, c='black', linestyle='-', marker='.')

            ix = xx
            iy = yy
            a.tick_params(direction='in')
            a.set_xlabel('Binding Energy (eV)', font='Arial', fontsize=size(22))
            a.set_xticklabels(labels=a.get_xticklabels(), fontsize=size(20))
            a.set_ylabel(r'Im $\Sigma$ (meV)', font='Arial', fontsize=size(22))
            a.set_yticklabels(labels=a.get_yticklabels(), fontsize=size(20))

            x = (vfe-fev)*1000
            y = fwhm
            b.plot(x, y, c='black', linestyle='-', marker='.')
            b.tick_params(direction='in')
            b.set_xlabel('Binding Energy (eV)', font='Arial', fontsize=size(22))
            b.set_xticklabels(labels=b.get_xticklabels(), fontsize=size(20))
            b.set_ylabel(r'FWHM ($\frac{2\pi}{\AA}$)',
                         font='Arial', fontsize=size(22))
            b.set_yticklabels(labels=b.get_yticklabels(), fontsize=size(20))

            x = (vfe-fev)*1000
            y = pos
            yy = interp(y, k*np.float64(bbk_offset.get()), be -
                        np.float64(bb_offset.get()))  # interp x into be,k set
            
            a.invert_xaxis()
            b.invert_xaxis()
    if pflag == 3:
        if value2.get() == 'Real & Imaginary':
            f, a = plt.subplots(2, 1, dpi=150)
            a[0].set_title(r'Self Energy $\Sigma$', font='Arial', fontsize=size(24))
            if dl==0:
                a[0].scatter(rx, ry, edgecolors='black', c='w')
            elif dl==1:
                a[0].plot(rx, ry, c='black')
            elif dl==2:
                a[0].plot(rx, ry, c='black', linestyle='-', marker='.')
            a[0].tick_params(direction='in')
            a[0].set_xlabel('Binding Energy (meV)', font='Arial', fontsize=size(22))
            a[0].set_xticklabels(labels=a[0].get_xticklabels(), fontsize=size(20))
            a[0].set_ylabel(r'Re $\Sigma$ (meV)', font='Arial', fontsize=size(22))
            a[0].set_yticklabels(labels=a[0].get_yticklabels(), fontsize=size(20))
            if dl==0:
                a[1].scatter(ix, iy, edgecolors='black', c='w')
            elif dl==1:
                a[1].plot(ix, iy, c='black')
            elif dl==2:
                a[1].plot(ix, iy, c='black', linestyle='-', marker='.')
            a[1].tick_params(direction='in')
            a[1].set_xlabel('Binding Energy (meV)', font='Arial', fontsize=size(22))
            a[1].set_xticklabels(labels=a[1].get_xticklabels(), fontsize=size(20))
            a[1].set_ylabel(r'Im $\Sigma$ (meV)', font='Arial', fontsize=size(22))
            a[1].set_yticklabels(labels=a[1].get_yticklabels(), fontsize=size(20))
            
            a[0].invert_xaxis()
            a[1].invert_xaxis()
        elif 'KK Transform' in value2.get():
            
            tbe = (vfe-fev)*1000
            ix=(tbe-tbe[-1])*-1
            cix=np.append(ix+ix[0],ix)
            tix=cix[0:len(cix)-1]*-1
            # kx=ix
            kx = np.append(cix,tix[::-1])
            ky = np.linspace(0, 1, len(kx))
            ciy=np.append(iy*0+np.mean(iy),iy)
            tiy=ciy[0:len(ciy)-1]
            ciy = np.append(ciy,tiy[::-1])

            #for imaginary part
            ix=(tbe-tbe[-1])*-1
            cix=np.append(ix+ix[0],ix)
            tix=cix[0:len(cix)-1]*-1
            kx = np.append(cix,tix[::-1])
            ky = np.linspace(0, 1, len(kx))
            cry=np.append(ry*0,ry)
            tcry=cry[0:len(cry)-1]*-1
            cry = np.append(cry,tcry[::-1])

            # Hilbert transform
            analytic_signal_r = hilbert(cry)
            amplitude_envelope_r = np.abs(analytic_signal_r)
            instantaneous_phase_r = np.unwrap(np.angle(analytic_signal_r))
            instantaneous_frequency_r = np.diff(instantaneous_phase_r) / (2.0 * np.pi)

            analytic_signal_i = hilbert(ciy)
            amplitude_envelope_i = np.abs(analytic_signal_i)
            instantaneous_phase_i = np.unwrap(np.angle(analytic_signal_i))
            instantaneous_frequency_i = np.diff(instantaneous_phase_i) / (2.0 * np.pi)

            # Reconstructed real and imaginary parts
            reconstructed_real = np.imag(analytic_signal_i)
            reconstructed_imag = -np.imag(analytic_signal_r)

                # Plot
            if 'Real Part' not in value2.get() and 'Imaginary Part' not in value2.get():
                f, a = plt.subplots(2, 1, dpi=150)
                # Plot imaginary data and its Hilbert transformation
                a[0].set_title(r'Self Energy $\Sigma$', font='Arial', fontsize=size(24))
                if dl==0:
                    a[0].scatter(tbe, ry, edgecolors='black', c='w', label=r'Re $\Sigma$')
                    a[0].scatter(tbe, reconstructed_real[len(ix):2*len(ix)]+(ry-np.mean(reconstructed_real[len(ix):2*len(ix)])), edgecolors='red', c='w', label=r'Re $\Sigma_{KK}$=KK(Im $\Sigma$)')
                elif dl==1:
                    a[0].plot(tbe, ry, c='black', label=r'Re $\Sigma$')
                    a[0].plot(tbe, reconstructed_real[len(ix):2*len(ix)]+(ry-np.mean(reconstructed_real[len(ix):2*len(ix)])), c='red', label=r'Re $\Sigma_{KK}$=KK(Im $\Sigma$)')
                elif dl==2:
                    a[0].plot(tbe, ry, c='black', linestyle='-', marker='.', label=r'Re $\Sigma$')
                    a[0].plot(tbe, reconstructed_real[len(ix):2*len(ix)]+(ry-np.mean(reconstructed_real[len(ix):2*len(ix)])), c='red', linestyle='-', marker='.', label=r'Re $\Sigma_{KK}$=KK(Im $\Sigma$)')
                a[0].set_xlabel('Binding Energy (meV)', font='Arial', fontsize=size(22))
                a[0].set_xticklabels(a[0].get_xticklabels(), fontsize=size(20))
                a[0].set_ylabel(r'Re $\Sigma$ (meV)', font='Arial', fontsize=size(22))
                a[0].set_yticklabels(a[0].get_yticklabels(), fontsize=size(20))
                a[0].legend(fontsize=size(20))
                if dl==0:
                    a[1].scatter(tbe, iy, edgecolors='black', c='w', label=r'Im $\Sigma$')
                    a[1].scatter(tbe, reconstructed_imag[len(ix):2*len(ix)]+(iy-np.mean(reconstructed_imag[len(ix):2*len(ix)])), edgecolors='red', c='w', label=r'Im $\Sigma_{KK}$=KK(Re $\Sigma$)')
                elif dl==1:
                    a[1].plot(tbe, iy, c='black', label=r'Im $\Sigma$')
                    a[1].plot(tbe, reconstructed_imag[len(ix):2*len(ix)]+(iy-np.mean(reconstructed_imag[len(ix):2*len(ix)])), c='red', label=r'Im $\Sigma_{KK}$=KK(Re $\Sigma$)')
                elif dl==2:
                    a[1].plot(tbe, iy, c='black', linestyle='-', marker='.', label=r'Im $\Sigma$')
                    a[1].plot(tbe, reconstructed_imag[len(ix):2*len(ix)]+(iy-np.mean(reconstructed_imag[len(ix):2*len(ix)])), c='red', linestyle='-', marker='.', label=r'Im $\Sigma_{KK}$=KK(Re $\Sigma$)')
                a[1].set_xlabel('Binding Energy (meV)', font='Arial', fontsize=size(22))
                a[1].set_xticklabels(a[1].get_xticklabels(), fontsize=size(20))
                a[1].set_ylabel(r'Im $\Sigma$ (meV)', font='Arial', fontsize=size(22))
                a[1].set_yticklabels(a[1].get_yticklabels(), fontsize=size(20))
                a[1].legend(fontsize=size(20))
                a[0].invert_xaxis()
                a[1].invert_xaxis()
            elif 'Real Part' in value2.get():
                f = plt.figure(figsize=(8*scale, 7*scale),layout='constrained')
                a=plt.axes([0.2,0.12,0.7,0.8])
                ttbe=tbe/1000
                if 'nd' in value2.get():
                    a.set_title(r'Self Energy $\Sigma$ Real Part', font='Arial', fontsize=size(24))
                    ty=np.diff(smooth(ry,20,3))/np.diff(ttbe)
                    if dl==0:
                        a.scatter(ttbe[0:-1], ty, edgecolors='black', c='w', label=r'Re $\Sigma$')
                    elif dl==1:
                        a.plot(ttbe[0:-1], ty, c='black', label=r'Re $\Sigma$')
                    elif dl==2:
                        a.plot(ttbe[0:-1], ty, c='black', linestyle='-', marker='.', label=r'Re $\Sigma$')
                    a.set_xlabel('Binding Energy (eV)', font='Arial', fontsize=size(22))
                    a.set_ylabel(r'$2^{nd} der. Re \Sigma$', font='Arial', fontsize=size(22))
                    a.set_xticklabels(a.get_xticklabels(),fontsize=size(20))
                    a.set_yticks([0])
                    a.set_yticklabels(a.get_yticklabels(),fontsize=size(20))
                else:
                    a.set_title(r'Self Energy $\Sigma$ Real Part', font='Arial', fontsize=size(24))
                    if dl==0:
                        a.scatter(ttbe, ry, edgecolors='black', c='w', label=r'Re $\Sigma$')
                        a.scatter(ttbe, reconstructed_real[len(ix):2*len(ix)]+(ry-np.mean(reconstructed_real[len(ix):2*len(ix)])), edgecolors='red', c='w', label=r'Re $\Sigma_{KK}$=KK(Im $\Sigma$)')
                    elif dl==1:
                        a.plot(ttbe, ry, c='black', label=r'Re $\Sigma$')
                        a.plot(ttbe, reconstructed_real[len(ix):2*len(ix)]+(ry-np.mean(reconstructed_real[len(ix):2*len(ix)])), c='red', label=r'Re $\Sigma_{KK}$=KK(Im $\Sigma$)')
                    elif dl==2:
                        a.plot(ttbe, ry, c='black', linestyle='-', marker='.', label=r'Re $\Sigma$')
                        a.plot(ttbe, reconstructed_real[len(ix):2*len(ix)]+(ry-np.mean(reconstructed_real[len(ix):2*len(ix)])), c='red', linestyle='-', marker='.', label=r'Re $\Sigma_{KK}$=KK(Im $\Sigma$)')
                    a.set_xlabel('Binding Energy (eV)', font='Arial', fontsize=size(22))
                    a.set_ylabel(r'Re $\Sigma$ (meV)', font='Arial', fontsize=size(22))
                    a.set_xticklabels(a.get_xticklabels(),fontsize=size(20))
                    a.set_yticklabels(a.get_yticklabels(),fontsize=size(20))
                    ll=a.legend(fontsize=size(20))
                    ll.draw_frame(False)
                a.invert_xaxis()
            elif 'Imaginary Part' in value2.get():
                f = plt.figure(figsize=(8*scale, 7*scale),layout='constrained')
                a=plt.axes([0.2,0.12,0.7,0.8])
                ttbe=tbe/1000
                if 'st' in value2.get():
                    a.set_title(r'Self Energy $\Sigma$ Imaginary Part', font='Arial', fontsize=size(24))
                    ty=np.diff(smooth(iy,20,3))/np.diff(ttbe)
                    if dl==0:
                        a.scatter(ttbe[0:-1], ty, edgecolors='black', c='w', label=r'Im $\Sigma$')
                    elif dl==1:
                        a.plot(ttbe[0:-1], ty, c='black', label=r'Im $\Sigma$')
                    elif dl==2:
                        a.plot(ttbe[0:-1], ty, c='black', linestyle='-', marker='.', label=r'Im $\Sigma$')
                    a.set_xlabel('Binding Energy (eV)', font='Arial', fontsize=size(22))
                    a.set_ylabel(r'$1^{st} der. Im \Sigma$', font='Arial', fontsize=size(22))
                    a.set_xticklabels(a.get_xticklabels(),fontsize=size(20))
                    a.set_yticks([0])
                    a.set_yticklabels(a.get_yticklabels(),fontsize=size(20))
                else:
                    a.set_title(r'Self Energy $\Sigma$ Imaginary Part', font='Arial', fontsize=size(24))
                    if dl==0:
                        a.scatter(ttbe, iy, edgecolors='black', c='w', label=r'Im $\Sigma$')
                        a.scatter(ttbe, reconstructed_imag[len(ix):2*len(ix)]+(iy-np.mean(reconstructed_imag[len(ix):2*len(ix)])), edgecolors='red', c='w', label=r'Im $\Sigma_{KK}$=KK(Re $\Sigma$)')
                    elif dl==1:
                        a.plot(ttbe, iy, c='black', label=r'Im $\Sigma$')
                        a.plot(ttbe, reconstructed_imag[len(ix):2*len(ix)]+(iy-np.mean(reconstructed_imag[len(ix):2*len(ix)])), c='red', label=r'Im $\Sigma_{KK}$=KK(Re $\Sigma$)')
                    elif dl==2:
                        a.plot(ttbe, iy, c='black', linestyle='-', marker='.', label=r'Im $\Sigma$')
                        a.plot(ttbe, reconstructed_imag[len(ix):2*len(ix)]+(iy-np.mean(reconstructed_imag[len(ix):2*len(ix)])), c='red', linestyle='-', marker='.', label=r'Im $\Sigma_{KK}$=KK(Re $\Sigma$)')
                    a.set_xlabel('Binding Energy (eV)', font='Arial', fontsize=size(22))
                    a.set_ylabel(r'Im $\Sigma$ (meV)', font='Arial', fontsize=size(22))
                    a.set_xticklabels(a.get_xticklabels(),fontsize=size(20))
                    a.set_yticklabels(a.get_yticklabels(),fontsize=size(20))
                    ll=a.legend(fontsize=size(20))
                    ll.draw_frame(False)
                a.invert_xaxis()
            
        elif value2.get() == 'Data Plot with Pos' or value2.get() == 'Data Plot with Pos and Bare Band':
            f0 = plt.figure(figsize=(8*scale, 7*scale), layout='constrained')
            a0 = plt.axes([0.13, 0.45, 0.8, 0.5])
            a1 = plt.axes([0.13, 0.08, 0.8, 0.2])
            a0.set_title('Drag to select specific region', font='Arial', fontsize=size(18))
            selectors.append(RectangleSelector(
                a0, select_callback,
                useblit=True,
                button=[1, 3],  # disable middle button
                minspanx=5, minspany=5,
                spancoords='pixels',
                interactive=True,
                props=props))
            # f0.canvas.mpl_connect('key_press_event',toggle_selector)
            f, a = plt.subplots(dpi=150)
            if emf=='KE':
                px, py = np.meshgrid(phi, ev)
                tev = py.copy()
            else:
                px, py = np.meshgrid(phi, vfe-ev)
                tev = vfe - py.copy()
            if npzf:
                px = phi
            else:
                px = (2*m*tev*1.602176634*10**-19)**0.5*np.sin((np.float64(k_offset.get())+px)/180*np.pi)*10**-10/(h/2/np.pi)
            pz = data.to_numpy()
            h1 = a.pcolormesh(px, py, pz, cmap=value3.get())
            if emf=='KE':
                yl = a.get_ylim()
            else:
                yl = sorted(a.get_ylim(), reverse=True)
            cb = f.colorbar(h1)
            # cb.set_ticklabels(cb.get_ticks(), font='Arial', fontsize=size(14))
            
            a.set_title(value2.get(), font='Arial', fontsize=size(18))
            a.set_xlabel(r'k ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=size(16))
            # a.set_xticklabels(labels=a.get_xticklabels(), fontsize=size(20))
            if emf=='KE':
                a.set_ylabel('Kinetic Energy (eV)', font='Arial', fontsize=size(16))
            else:
                a.set_ylabel('Binding Energy (eV)', font='Arial', fontsize=size(16))
            # a.set_yticklabels(labels=a.get_yticklabels(), fontsize=size(20))
            try:
                if mp == 1:
                    if emf=='KE':
                        a.scatter(pos, fev, marker='.', s=scale*scale*0.3, c='black')
                    else:
                        a.scatter(pos, vfe-fev, marker='.', s=scale*scale*0.3, c='black')
                        
                if mf == 1:
                    ophimin = np.arcsin(
                        (rpos-fwhm/2)/(2*m*fev*1.602176634*10**-19)**0.5/10**-10*(h/2/np.pi))*180/np.pi
                    ophimax = np.arcsin(
                        (rpos+fwhm/2)/(2*m*fev*1.602176634*10**-19)**0.5/10**-10*(h/2/np.pi))*180/np.pi
                    posmin = (2*m*fev*1.602176634*10**-19)**0.5*np.sin(
                        (np.float64(k_offset.get())+ophimin)/180*np.pi)*10**-10/(h/2/np.pi)
                    posmax = (2*m*fev*1.602176634*10**-19)**0.5*np.sin(
                        (np.float64(k_offset.get())+ophimax)/180*np.pi)*10**-10/(h/2/np.pi)
                    if emf=='KE':
                        a.scatter([posmin, posmax], [fev, fev],
                                marker='|', c='grey', s=scale*scale*10, alpha=0.8)
                    else:
                        a.scatter([posmin, posmax], [vfe-fev, vfe-fev],
                                marker='|', c='grey', s=scale*scale*10, alpha=0.8)
            except:
                pass
            try:
                if ep == 1:
                    if emf=='KE':
                        a.scatter(fk, epos, marker='.', s=scale*scale*0.3, c='black')
                    else:
                        a.scatter(fk, vfe-epos, marker='.', s=scale*scale*0.3, c='black')
                            
                if ef == 1:
                    eposmin = epos-efwhm/2
                    eposmax = epos+efwhm/2
                    if emf=='KE':
                        a.scatter([fk, fk], [eposmin, eposmax],
                                marker='_', c='grey', s=scale*scale*10, alpha=0.8)
                    else:
                        a.scatter([fk, fk], [vfe-eposmin, vfe-eposmax],
                                marker='_', c='grey', s=scale*scale*10, alpha=0.8)
                        
            except:
                pass
            h2 = a0.pcolormesh(px, py, pz, cmap=value3.get())
            cb1 = f0.colorbar(h2)
            cb1.set_ticks(cb1.get_ticks())
            cb1.set_ticklabels(cb1.get_ticks(), font='Arial',
                               fontsize=size(14), minor=False)
            a0.set_xlabel(r'k ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=size(16))
            if emf=='KE':
                a0.set_ylabel('Kinetic Energy (eV)', font='Arial', fontsize=size(16))
            else:
                a0.set_ylabel('Binding Energy (eV)', font='Arial', fontsize=size(16))
                
            try:
                if mp == 1:
                    if emf=='KE':
                        a0.scatter(pos, fev, marker='.', s=scale*scale*0.3, c='black')
                    else:
                        a0.scatter(pos, vfe-fev, marker='.', s=scale*scale*0.3, c='black')
                        
                if mf == 1:
                    ophimin = np.arcsin(
                        (rpos-fwhm/2)/(2*m*fev*1.602176634*10**-19)**0.5/10**-10*(h/2/np.pi))*180/np.pi
                    ophimax = np.arcsin(
                        (rpos+fwhm/2)/(2*m*fev*1.602176634*10**-19)**0.5/10**-10*(h/2/np.pi))*180/np.pi
                    posmin = (2*m*fev*1.602176634*10**-19)**0.5*np.sin(
                        (np.float64(k_offset.get())+ophimin)/180*np.pi)*10**-10/(h/2/np.pi)
                    posmax = (2*m*fev*1.602176634*10**-19)**0.5*np.sin(
                        (np.float64(k_offset.get())+ophimax)/180*np.pi)*10**-10/(h/2/np.pi)
                    if emf=='KE':
                        a0.scatter([posmin, posmax], [fev, fev],
                                marker='|', c='grey', s=scale*scale*10, alpha=0.8)
                    else:
                        a0.scatter([posmin, posmax], [vfe-fev, vfe-fev],
                                marker='|', c='grey', s=scale*scale*10, alpha=0.8)
            except:
                pass
            try:
                if ep == 1:
                    if emf=='KE':
                        a0.scatter(fk, epos, marker='.', s=scale*scale*0.3, c='black')
                    else:
                        a0.scatter(fk, vfe-epos, marker='.', s=scale*scale*0.3, c='black')
                        
                if ef == 1:
                    eposmin = epos-efwhm/2
                    eposmax = epos+efwhm/2
                    if emf=='KE':
                        a0.scatter([fk, fk], [eposmin, eposmax],
                                marker='_', c='grey', s=scale*scale*10, alpha=0.8)
                    else:
                        a0.scatter([fk, fk], [vfe-eposmin, vfe-eposmax],
                                marker='_', c='grey', s=scale*scale*10, alpha=0.8)
            except:
                pass
            # b.set_xticklabels(labels=b.get_xticklabels(),font='Arial',fontsize=size(20))
            # b.set_yticklabels(labels=b.get_yticklabels(),font='Arial',fontsize=size(20))

            n = a1.hist(pz.flatten(), bins=np.linspace(
                min(pz.flatten()), max(pz.flatten()), 50), color='green')
            a1.set_xlabel('Intensity')
            a1.set_ylabel('Counts')
            a1.set_title('Drag to Select the range of Intensity ')
            selectors.append(SpanSelector(
                a1,
                onselect,
                "horizontal",
                useblit=True,
                props=dict(alpha=0.3, facecolor="tab:blue"),
                onmove_callback=onmove_callback,
                interactive=True,
                drag_from_anywhere=True,
                snap_values=n[1]
            ))
            try:
                if value2.get() == 'Data Plot with Pos and Bare Band':
                    if emf=='KE':
                        a.plot(k*np.float64(bbk_offset.get()), (be -
                            np.float64(bb_offset.get()))/1000+vfe, linewidth=scale*0.3, c='red', linestyle='--')
                        a0.plot(k*np.float64(bbk_offset.get()), (be -
                                np.float64(bb_offset.get()))/1000+vfe, linewidth=scale*0.3, c='red', linestyle='--')
                    else:
                        a.plot(k*np.float64(bbk_offset.get()), (-be +
                            np.float64(bb_offset.get()))/1000, linewidth=scale*0.3, c='red', linestyle='--')
                        a0.plot(k*np.float64(bbk_offset.get()), (-be +
                                np.float64(bb_offset.get()))/1000, linewidth=scale*0.3, c='red', linestyle='--')
            except:
                pass
            if emf=='BE':
                a.invert_yaxis()
                a0.invert_yaxis()
            # cursor = Cursor(a, useblit=True, color='red', linewidth=scale*1)
            annot = a.annotate(
                "", xy=(0,0), xytext=(20,20), textcoords="offset points",
                bbox=dict(boxstyle="round", fc="w", alpha=0.6),
                fontsize=size(12)
                # fontsize=size(12),
                # arrowprops=dict(arrowstyle="->")
            )
            annot.set_visible(False)
            f.canvas.mpl_connect('motion_notify_event', cur_move)
            f.canvas.mpl_connect('motion_notify_event', cur_on_move)
    try:
        if value1.get() == '---Plot2---' and value2.get() != 'Real & Imaginary' and 'KK Transform' not in value2.get() and 'MDC Curves' != value.get():
            try:
                h1.set_clim([vcmin.get(), vcmax.get()])
                h2.set_clim([vcmin.get(), vcmax.get()])
            except:
                pass
            try:    # ignore the problem occurred in E-k with MDC curves
                a0.set_xlim(xl)
                a0.set_ylim(yl)
                a.set_xlim(xl)
                a.set_ylim(yl)
            except:
                pass
            if value.get() != 'Raw Data':
                plt.tight_layout()
            # if value.get()=='Raw Data':
            #     plt.connect('motion_notify_event', cut_move)
            copy_to_clipboard(f)
            st.put('graph copied to clipboard')
            if value.get() != 'Raw Data':
                threading.Thread(target=show_info,daemon=True).start()
            plt.show()
            try:
                h1.set_clim([cm.get(), cM.get()])
                h2.set_clim([cm.get(), cM.get()])
            except:
                pass
        else:
            plt.tight_layout()
            copy_to_clipboard(f)
            st.put('graph copied to clipboard')
            threading.Thread(target=show_info,daemon=True).start()
            plt.show()
        # f.ion()
        # f0.ion()
    except:
        print('fail to export graph')
        pass

    # fp=fd.asksaveasfilename(filetypes=(("PNG files", "*.png"),))
    # f.savefig(fname=fp)

@pool_protect
def angcut(*e):
    t0 = threading.Thread(target=o_angcut)
    t0.daemon = True
    t0.start()


@pool_protect
def ecut(*e):
    t1 = threading.Thread(target=o_ecut)
    t1.daemon = True
    t1.start()


@pool_protect
def loadmfit(*e):
    t2 = threading.Thread(target=o_loadmfit)
    t2.daemon = True
    t2.start()


@pool_protect
def loadefit(*e):
    global rdd, efi_x, data, fpr
    global fphi, epos, ffphi, efwhm, fk, emin, emax, semin, semax, seaa1, seaa2, sefp, sefi
    el = eloader(st, data, ev, phi, rdd, cdir, lowlim.get())
    el.loadparam(k_offset.get(), base.get(), npzf, fpr)
    file = fd.askopenfilename(title="Select EDC Fitted file", filetypes=(("NPZ files", "*.npz"), ("VMS files", "*.vms"),))
    t3 = threading.Thread(target=el.loadefit, args=(file,))
    t3.daemon = True
    t3.start()
    t3.join()
    rdd, efi_x, data, fpr = el.rdd, el.efi_x, el.data, el.fpr
    fphi, epos, ffphi, efwhm, fk, emin, emax, semin, semax, seaa1, seaa2, sefp, sefi = el.fphi, el.epos, el.ffphi, el.efwhm, el.fk, el.emin, el.emax, el.semin, el.semax, el.seaa1, el.seaa2, el.sefp, el.sefi
    limg.config(image=img[np.random.randint(len(img))])
    if el.fload:
        try:
            tbasename = os.path.basename(rdd)
            if '.h5' in tbasename:
                data = load_h5(rdd)
                pr_load(data)
            elif '.json' in tbasename:
                data = load_json(rdd)
                pr_load(data)
            elif '.txt' in tbasename:
                data = load_txt(rdd)
                pr_load(data)
        except FileNotFoundError:
            print(f'{rdd} File path not found, skip loading raw data.')
        except Exception as e:
            print(f'Error loading raw data from {rdd}: {e}')
    clear(el)

@pool_protect
def reload(*e):
    t4 = threading.Thread(target=o_reload)
    t4.daemon = True
    t4.start()


@pool_protect
def expte():
    t5 = threading.Thread(target=o_expte)
    t5.daemon = True
    t5.start()


@pool_protect
def exptm():
    t6 = threading.Thread(target=o_exptm)
    t6.daemon = True
    t6.start()


@pool_protect
def bareband(*e):
    t7 = threading.Thread(target=o_bareband)
    t7.daemon = True
    t7.start()

@pool_protect
def main_plot_bind():
    global main_motion, out, main_notify_cid, main_press_cid, main_release_cid
    out.mpl_disconnect(main_notify_cid)
    out.mpl_disconnect(main_press_cid)
    out.mpl_disconnect(main_release_cid)
    clear(main_motion)
    main_motion = MainMotion()
    main_notify_cid = out.mpl_connect('motion_notify_event', main_motion.move)
    main_press_cid = out.mpl_connect('button_press_event', main_motion.press)
    main_release_cid = out.mpl_connect('button_release_event', main_motion.release)

@pool_protect
def plot(event):
    if value.get() == '---Plot1---' and value1.get() == '---Plot2---' and value2.get() == '---Plot3---':
        trans_plot()
    else:        
        plot1()
        plot2()
        plot3()        
    
im_kernel = 17
d,l,p = 8,20,3
@pool_protect
def plot1(*e):
    global gg
    if 'gg' in globals():
        gg.destroy()
    if 'MDC Curves' in value.get():
        gg = plot1_window_MDC_curves(d, l, p)
    elif value.get() == 'Second Derivative':
        gg = plot1_window_Second_Derivative(im_kernel)
    else:
        t8 = threading.Thread(target=o_plot1)
        t8.daemon = True
        t8.start()

@pool_protect
def plot2(*e):
    t9 = threading.Thread(target=o_plot2)
    t9.daemon = True
    t9.start()


@pool_protect
def plot3(*e):
    global gg
    if 'gg' in globals():
        gg.destroy()
    if value2.get() == 'Data Plot with Pos' or value2.get() == 'Data Plot with Pos and Bare Band':
        gg = plot3_window(fev, fk)
    else:
        t10 = threading.Thread(target=o_plot3)
        t10.daemon = True
        t10.start()


def load(drop=False, files='', *args):
    if 'KeyPress' in str(drop):
        drop = False
    t11 = threading.Thread(target=o_load, args=(drop, files))
    t11.daemon = True
    t11.start()


def fitgl():
    pass
    # t12 = threading.Thread(target=o_fitgl)
    # t12.daemon = True
    # t12.start()


@pool_protect
def tstate():
    try:
        while True:
            state.config(text=str(st.get()))
    except:
        pass

@pool_protect
def lm2p():
    lmgg.destroy()
    global rdd
    ml = mloader(st, data, ev, phi, rdd, cdir, lowlim.get())
    file = fd.askopenfilename(title="Select MDC Fitted file", filetypes=(("VMS files", "*.vms"),))
    t = threading.Thread(target=ml.loadmfit_2p, args=(file,))
    t.daemon = True
    t.start()
    t.join()
    rdd = ml.rdd
    clear(ml)


@pool_protect
def lmre():
    lmgg.destroy()
    global rdd
    ml = mloader(st, data, ev, phi, rdd, cdir, lowlim.get())
    file = fd.askopenfilename(title="Select MDC Fitted file", filetypes=(("VMS files", "*.vms"),))
    t = threading.Thread(target=ml.loadmfit_re, args=(file,))
    t.daemon = True
    t.start()
    t.join()
    rdd = ml.rdd
    clear(ml)

@pool_protect
def lm():
    lmgg.destroy()
    global rdd, mfi_x, data, fpr
    global fev, rpos, ophi, fwhm, pos, kmin, kmax, skmin, skmax, smaa1, smaa2, smfp, smfi, smresult, smcst
    ml = mloader(st, data, ev, phi, rdd, cdir, lowlim.get())
    ml.loadparam(k_offset.get(), base.get(), npzf, fpr)
    file = fd.askopenfilename(title="Select MDC Fitted file", filetypes=(("NPZ files", "*.npz"), ("VMS files", "*.vms"),))
    t = threading.Thread(target=ml.loadmfit_, args=(file,))
    t.daemon = True
    t.start()
    t.join()
    rdd, mfi_x, data, fpr = ml.rdd, ml.mfi_x, ml.data, ml.fpr
    fev, rpos, ophi, fwhm, pos, kmin, kmax, skmin, skmax, smaa1, smaa2, smfp, smfi, smresult, smcst = ml.fev, ml.rpos, ml.ophi, ml.fwhm, ml.pos, ml.kmin, ml.kmax, ml.skmin, ml.skmax, ml.smaa1, ml.smaa2, ml.smfp, ml.smfi, ml.smresult, ml.smcst
    limg.config(image=img[np.random.randint(len(img))])
    if ml.fload:
        try:
            tbasename = os.path.basename(rdd)
            if '.h5' in tbasename:
                data = load_h5(rdd)
                pr_load(data)
            elif '.json' in tbasename:
                data = load_json(rdd)
                pr_load(data)
            elif '.txt' in tbasename:
                data = load_txt(rdd)
                pr_load(data)
        except FileNotFoundError:
            print(f'{rdd} File path not found, skip loading raw data.')
        except Exception as e:
            print(f'Error loading raw data from {rdd}: {e}')
    clear(ml)


@pool_protect
def o_loadmfit():
    global g, st, lmgg
    if 'lmgg' in globals():
        lmgg.destroy()
    lmgg = RestrictedToplevel(g)
    lmgg.title('Load MDC fitted File')
    lmgg.geometry('400x200')  # format:'1400x800'
    b1 = tk.Button(lmgg, command=lm2p, text='vms 1 peak to 2 peaks', font=(
        "Arial", size(12), "bold"), fg='red', width=30, height='1', bd=10)
    b1.pack()
    b2 = tk.Button(lmgg, command=lmre, text='reverse vms axis', font=(
        "Arial", size(12), "bold"), fg='red', width=30, height='1', bd=10)
    b2.pack()
    b3 = tk.Button(lmgg, command=lm, text='load MDC fitted File', font=(
        "Arial", size(12), "bold"), fg='red', width=30, height='1', bd=10)
    b3.pack()
    lmgg.update()
    w=lmgg.winfo_reqwidth()
    h=lmgg.winfo_reqheight()
    lmgg.geometry(f'{w}x{h}')
    set_center(g, lmgg, 0, 0)
    lmgg.focus_set()
    lmgg.limit_bind()

def dl_sw():
    global dl, b_sw
    s=['dot','line','dot-line']
    dl=(dl+1)%3
    b_sw.config(text=s[dl])
    t = threading.Thread(target=o_plot3)
    t.daemon = True
    t.start()

def plot1_set(opt):
    global value, value1, value2
    value.set(opt)
    value1.set('---Plot2---')
    value2.set('---Plot3---')
    
def plot2_set(opt):
    global value, value1, value2
    value.set('---Plot1---')
    value1.set(opt)
    value2.set('---Plot3---')
    
def plot3_set(opt):
    global value, value1, value2
    value.set('---Plot1---')
    value1.set('---Plot2---')
    value2.set(opt)
    
def size(s: int) -> int:
    return int(s * scale)

if __name__ == '__main__':
    os.chdir(cdir)
    if os.path.exists('open_check_MDC_cut.txt')==0:
        with open('open_check_MDC_cut.txt', 'w', encoding = 'utf-8') as f:
            f.write('1')
            f.close()
        if os.name == 'nt':
            os.system(rf'start "" cmd /C "chcp 65001 > nul && python -W ignore::SyntaxWarning -W ignore::UserWarning "{app_name}.py""')
        elif os.name == 'posix':
            try:
                os.system(rf'start "" cmd /C "chcp 65001 > nul && python3 -W ignore::SyntaxWarning -W ignore::UserWarning "{app_name}.py""')
            except:
                os.system(rf'start "" cmd /C "chcp 65001 > nul && python -W ignore::SyntaxWarning -W ignore::UserWarning "{app_name}.py""')
        quit()
    else:
        os.remove('open_check_MDC_cut.txt')
        
    hwnd = find_window()
    path = os.path.join(cdir, '.MDC_cut', 'hwnd')
    with open(path, 'w') as f:
        f.write(f'{hwnd}')  #for DataViewer Qt GUI
        f.close()
    ScaleFactor = windll.shcore.GetScaleFactorForDevice(0)
    osf = windll.shcore.GetScaleFactorForDevice(0)
    # print('ScaleFactor:',ScaleFactor)
    t_sc_w, t_sc_h = windll.user32.GetSystemMetrics(0), windll.user32.GetSystemMetrics(1)   # Screen width and height
    t_sc_h-=int(40*ScaleFactor/100)
    if bar_pos == 'top':    #taskbar on top
        sc_y = int(40*ScaleFactor/100)
    else:
        sc_y = 0
    # w 1920 1374 (96 dpi)
    # h 1080 748 (96 dpi)
    g = TkinterDnD.Tk()
    tkDnD(g)    #bind whole window to Drag-and-drop function
    # g = ttk.Window(themename='darkly')
    odpi=g.winfo_fpixels('1i')
    path = os.path.join(cdir, '.MDC_cut', 'odpi')
    with open(path, 'w') as f:
        f.write(f'{odpi}')  #for RestrictedToplevel
        f.close()
    # print('odpi:',odpi)
    # prfactor = 1 if ScaleFactor <= 150 else 1.03
    # prfactor = 1.03 if ScaleFactor <= 100 else 0.9 if ScaleFactor <= 125 else 0.8 if ScaleFactor <= 150 else 0.5
    prfactor = 1
    ScaleFactor /= prfactor*(ScaleFactor/100*1880/96*odpi/t_sc_w) if 1880/t_sc_w >= (950)/t_sc_h else prfactor*(ScaleFactor/100*(950)/96*odpi/t_sc_h)
    g.tk.call('tk', 'scaling', ScaleFactor/100)
    dpi=g.winfo_fpixels('1i')
    # print('dpi:',dpi)
    windll.shcore.SetProcessDpiAwareness(1)
    scale = odpi / dpi
    base_font_size = 16
    scaled_font_size = int(base_font_size * scale)
    
    g_mem = psutil.Process(pid).memory_info().rss/1024**3
    app_pars = app_param(hwnd=hwnd, scale=scale, dpi=dpi, bar_pos=bar_pos, g_mem=g_mem)

    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = int(plt.rcParams['font.size'] * scale)
    plt.rcParams['lines.linewidth'] = plt.rcParams['lines.linewidth'] * scale
    plt.rcParams['lines.markersize'] = plt.rcParams['lines.markersize'] * scale
    plt.rcParams['figure.figsize'] = (plt.rcParams['figure.figsize'][0] * scale, plt.rcParams['figure.figsize'][1] * scale)
    # plt.rcParams['figure.facecolor'] = '#222222'
    # plt.rcParams['axes.facecolor'] = '#333'
    # plt.rcParams['axes.titlecolor'] = 'white'
    # plt.rcParams['axes.edgecolor'] = 'white'
    # plt.rcParams['axes.labelcolor'] = 'white'
    # plt.rcParams['axes.titlecolor'] = 'white'
    # plt.rcParams['xtick.color'] = 'white'
    # plt.rcParams['ytick.color'] = 'white'
    # plt.rcParams['lines.color'] = 'white'
    # plt.rcParams['lines.markeredgecolor'] = 'white'
    # plt.rcParams['lines.markerfacecolor'] = 'white'
    # plt.rcParams['scatter.edgecolors'] = 'white'
    # print('scale:', scale)
        
    # 設定預設字體
    default_font = ('Arial', scaled_font_size)
    g.option_add('*Font', default_font)
    icon_manager = MenuIconManager(scale=scale, ScaleFactor=ScaleFactor, odpi=odpi, dpi=dpi)
    ToolTip.icon = icon_manager
    ToolTip.scaled_font_size = scaled_font_size
    
    g.geometry(f'1900x1080+0+{sc_y}')
    g.title('MDC cut')
    g.config(bg='white')
    # g.geometry('1920x980')  # format:'1400x800'
    g.resizable(True, True)
    
    menubar = tk.Menu(g, tearoff=0, bg="white")

    # File Menu
    filemenu = tk.Menu(menubar, tearoff=0, bg="white")
    
    filemenu1 = tk.Menu(filemenu, tearoff=0, bg="white")
    filemenu1.add_command(label="MDC Fitted File", command=loadmfit, image=icon_manager.get_mini_icon('mdc_fitted_file'), compound='left', accelerator="F1")
    filemenu1.add_command(label="EDC Fitted File", image=icon_manager.get_mini_icon('edc_fitted_file'), command=loadefit, compound='left', accelerator="F2")
    
    filemenu2 = tk.Menu(filemenu, tearoff=0, bg="white")
    filemenu2.add_command(label="Export Graph", command=exp, image=icon_manager.get_mini_icon('exp_graph'), compound='left', accelerator="F10")
    filemenu2.add_command(label="Export to Origin", command=gui_exp_origin, image=icon_manager.get_mini_icon('exp_origin'), compound='left', accelerator="F11")
    filemenu2.add_command(label="Export MDC Fitted Data (k offset)", command=exptm)
    filemenu2.add_command(label="Export EDC Fitted Data (k offset)", command=expte)
    
    
    menubar.add_cascade(label="File", menu=filemenu)
    filemenu.add_command(label="Load Raw Data", image=icon_manager.get_mini_icon('raw_data'), command=load, accelerator="Ctrl+O", compound='left')
    filemenu.add_cascade(label="Load fitted File", menu=filemenu1)
    filemenu.add_command(label="Load Bare Band File", image=icon_manager.get_mini_icon('bare_band'), command=bareband, compound='left', accelerator="F3")
    filemenu.add_separator()
    filemenu.add_cascade(label="Export Data", menu=filemenu2)
    filemenu.add_command(label="Exit", command=g.quit)
    
    # filemenu.entryconfig("Load Raw Data", state='disabled')
    
    # Plot Menu
    plotmenu = tk.Menu(menubar, tearoff=0, bg="white")
    
    optionList = ['Raw Data', 'E-k Diagram', 'MDC Normalized', 'First Derivative', 'Second Derivative', 'MDC Curves', 'E-k with MDC Curves']
    pltmenu1 = tk.Menu(plotmenu, tearoff=0, bg="white")
    for opt in optionList:
        pltmenu1.add_command(label=opt, command=lambda opt=opt: plot1_set(opt))

    optionList1 = ['MDC fitted Data', 'EDC fitted Data', 'Real Part', 'Imaginary Part']
    pltmenu2 = tk.Menu(plotmenu, tearoff=0, bg="white")
    for opt in optionList1:
        pltmenu2.add_command(label=opt, command=lambda opt=opt: plot2_set(opt))
    
    optionList2 = ['Real & Imaginary', 'KK Transform Real & Imaginary', 'KK Transform Real Part', 'KK Transform Imaginary Part', 'KK Transform Real Part 2nd Derivative', 'KK Transform Imaginary Part 1st Derivative', 'Data Plot with Pos', 'Data Plot with Pos and Bare Band']
    pltmenu3 = tk.Menu(plotmenu, tearoff=0, bg="white")
    for opt in optionList2:
        pltmenu3.add_command(label=opt, command=lambda opt=opt: plot3_set(opt))

    menubar.add_cascade(label="Plot", menu=plotmenu)
    plotmenu.add_cascade(label="Raw", menu=pltmenu1)
    plotmenu.add_cascade(label="Fit", menu=pltmenu2)
    plotmenu.add_cascade(label="Transform", menu=pltmenu3)
    plotmenu.add_separator()
    plotmenu.add_command(label="Clear MDC Fitted Data", command=clmfit)
    plotmenu.add_command(label="Clear EDC Fitted Data", command=clefit)
    
    # Tools Menu
    toolmenu = tk.Menu(menubar, tearoff=0, bg="white")
    
    toolmenu1 = tk.Menu(toolmenu, tearoff=0, bg="white")
    toolmenu1.add_command(label="MDC Fitter", command=cmfit, image=icon_manager.get_mini_icon('mdc_fitter'), compound='left', accelerator="F5")
    toolmenu1.add_command(label="EDC Fitter", command=cefit, image=icon_manager.get_mini_icon('edc_fitter'), compound='left', accelerator="F6")
    
    toolmenu2 = tk.Menu(toolmenu, tearoff=0, bg="white")
    toolmenu2.add_command(label="MDC Cutter", command=ecut, image=icon_manager.get_mini_icon('mdc_cutter'), compound='left', accelerator="F7")    #重定義ecut 包括lower limit
    toolmenu2.add_command(label="EDC Cutter", command=angcut, image=icon_manager.get_mini_icon('edc_cutter'), compound='left', accelerator="F8")
    
    menubar.add_cascade(label="Tools", menu=toolmenu)
    toolmenu.add_command(label="Spectrogram", command=trans_plot, image=icon_manager.get_mini_icon('spectrogram'), compound='left', accelerator="F4")
    toolmenu.add_cascade(label="Fitter", menu=toolmenu1)
    toolmenu.add_cascade(label="Cutter", menu=toolmenu2)
    toolmenu.add_command(label="E-k Angle Converter", command=calculator, image=icon_manager.get_mini_icon('calculator'), compound='left', accelerator="F9")
    toolmenu.add_command(label="Volume Viewer", command=view_3d, image=icon_manager.get_mini_icon('view_3d'), compound='left', accelerator="F12")
    toolmenu.add_command(label="Sample Offset Fitter", command=fit_so_app, image=icon_manager.get_mini_icon('so_fit'), compound='left', accelerator="Ctrl+P")
    
    helpmenu = tk.Menu(menubar, tearoff=0, bg="white")
    menubar.add_cascade(label="Help", menu=helpmenu)
    helpmenu.add_command(label="About MDC_Cut...", command=about)
    helpmenu.add_command(label="Help", command=f_help)
    
    g.config(menu=menubar)
    
    
    fr_toolbar = tk.Frame(g, bg="white")
    fr_toolbar.pack(anchor=tk.W)
    b_load = Button(fr_toolbar, text="Load Raw Data", image=icon_manager.get_icon('raw_data'), command=load)
    b_load.pack(side=tk.LEFT)
    b_loadmfit = Button(fr_toolbar, text="Load MDC Fitted File", image=icon_manager.get_icon('mdc_fitted_file'), command=loadmfit)
    b_loadmfit.pack(side=tk.LEFT)
    b_loadefit = Button(fr_toolbar, text="Load EDC Fitted File", image=icon_manager.get_icon('edc_fitted_file'), command=loadefit)
    b_loadefit.pack(side=tk.LEFT)
    b_loadbb = Button(fr_toolbar, text="Load Bare Band File", image=icon_manager.get_icon('bare_band'), command=bareband)
    b_loadbb.pack(side=tk.LEFT)
    
    b_spec = Button(fr_toolbar, text="Spectrogram", image=icon_manager.get_icon('spectrogram'), command=trans_plot)
    b_spec.pack(side=tk.LEFT)
    b_mfit = Button(fr_toolbar, text="MDC Fitter", image=icon_manager.get_icon('mdc_fitter'), command=cmfit)
    b_mfit.pack(side=tk.LEFT)
    b_efit = Button(fr_toolbar, text="EDC Fitter", image=icon_manager.get_icon('edc_fitter'), command=cefit)
    b_efit.pack(side=tk.LEFT)
    b_mcut = Button(fr_toolbar, text="MDC Cutter", image=icon_manager.get_icon('mdc_cutter'), command=ecut)
    b_mcut.pack(side=tk.LEFT)
    b_ecut = Button(fr_toolbar, text="EDC Cutter", image=icon_manager.get_icon('edc_cutter'), command=angcut)
    b_ecut.pack(side=tk.LEFT)
    b_kcal = Button(fr_toolbar, text="E-k Angle Converter", image=icon_manager.get_icon('calculator'), command=calculator)
    b_kcal.pack(side=tk.LEFT)
    
    b_exp_graph = Button(fr_toolbar, text="Export Graph", image=icon_manager.get_icon('exp_graph'), command=exp)
    b_exp_graph.pack(side=tk.LEFT)
    b_exp_origin = Button(fr_toolbar, text="Export to Origin", image=icon_manager.get_icon('exp_origin'), command=gui_exp_origin)
    b_exp_origin.pack(side=tk.LEFT)
    
    b_view_3d = Button(fr_toolbar, text="Volume Viewer", image=icon_manager.get_icon('view_3d'), command=view_3d)
    b_view_3d.pack(side=tk.LEFT)
    b_so_fit = Button(fr_toolbar, text="Sample Offset Fitter", image=icon_manager.get_icon('so_fit'), command=fit_so_app)
    b_so_fit.pack(side=tk.LEFT)
    
    
    # 建立tooltip
    ToolTip(b_load, "Select and load your raw data files - supports H5, JSON, NPZ, and TXT formats. You can choose multiple files at once.", "Ctrl+O")
    ToolTip(b_loadmfit, "Select the MDC fitted file in VMS or NPZ formats. Note that the VMS file should only contain two peak information.", "F1")
    ToolTip(b_loadefit, "Select the EDC fitted file in VMS or NPZ formats. Note that the VMS file should only contain two peak information. This feature is not well-developed yet.", "F2")
    ToolTip(b_loadbb, "Import the bare band file in TXT format. Please check the user manual for detailed file format specifications.", "F3")
    ToolTip(b_spec, "View the current data using Spectrogram Interface.", "F4")
    ToolTip(b_mfit, "Utilize the embedded MDC Fitter Interface to perform fitting operations on or to visualize the MDC slices. The corresponding raw data will be automatically loaded if the file path is valid.", "F5")
    ToolTip(b_efit, "Utilize the embedded EDC Fitter Interface to perform fitting operations on or to visualize the EDC slices. The corresponding raw data will be automatically loaded if the file path is valid. Note that this feature is not fully implemented yet.", "F6")
    ToolTip(b_mcut, "Slice the dataset along the angular axis to extract momentum distribution curves (MDCs) and export the data series to VMS format.", "F7")
    ToolTip(b_ecut, "Perform energy-axis slicing of the dataset to extract energy distribution curves (EDCs) and export the resulting data series in VMS format.", "F8")
    ToolTip(b_kcal, "A calculator tool that converts k-values at a specified energy into corresponding angle values expressed in degrees.", "F9")
    ToolTip(b_exp_graph, "Display and save the graph with Matplotlib window.", "F10")
    ToolTip(b_exp_origin, "Select the required data and import it into OriginPro to enable advanced data processing.", "F11")
    ToolTip(b_view_3d, "Open a 3D data viewer to visualize the data cube in three dimensions. Rotation and HDF5 file export are supported.", "F12")
    ToolTip(b_so_fit, "Add data points to correct for sample angle offset. The measured points on the Energy-Angle diagram show asymmetry instead of the expected symmetry at specific kinetic energies, indicating slight sample/sample holder misalignment that can be corrected through fitting.", "Ctrl+P")
    
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
    
    value3 = tk.StringVar()
    value3.set('prevac_cmap')
    value3.trace_add('write', chcmp)

    try:
        with np.load(os.path.join('.MDC_cut', 'rd.npz'), 'rb') as ff:
            path = str(ff['path'])
            lpath = ff['lpath']
            print('\n\033[90mRaw Data preloaded:\033[0m\n\n')
            lfs = loadfiles(lpath, init=True, name='internal', cmap=value3.get(), app_pars=app_pars)
            if lfs.cec_pars:
                lfs = call_cec(g, lfs)
            data = lfs.get(0)
            for _ in data.attrs.keys():
                if _ != 'Description':
                    print(_,':', data.attrs[_])
                else:
                    print(_,':', data.attrs[_].replace('\n','\n              '))
            dvalue = list(data.attrs.values())
            rdd = path  # old version data path
            dpath = path    # new version data path
            name, e_photon, description = dvalue[0], float(dvalue[3].split()[0]), dvalue[13]
    except:
        data, lfs, name, description, e_photon = None, None, None, None, None
        print('\033[90mNo Raw Data preloaded\033[0m')

    try:
        with np.load(os.path.join('.MDC_cut', 'bb.npz'), 'rb') as f:
            bpath = str(f['path'])
            be = f['be']
            k = f['k']
            bbo = f['bbo']
            bbk = f['bbk']
            print('\033[90mBare Band file preloaded:')
            print(bpath+'\n\033[0m')
    except:
        bpath = ''
        print('\033[90mNo Bare Band file preloaded\033[0m')

    try:
        with np.load(os.path.join('.MDC_cut', 'efpath.npz'), 'rb') as f:
            efpath = str(f['efpath'])
            print('\033[90mEDC Fitted path preloaded\033[0m')
    except:
        print('\033[90mNo EDC Fitted path preloaded\033[0m')

    try:
        with np.load(os.path.join('.MDC_cut', 'mfpath.npz'), 'rb') as f:
            mfpath = str(f['mfpath'])
            print('\033[90mMDC Fitted path preloaded\033[0m')
    except:
        print('\033[90mNo MDC Fitted path preloaded\033[0m')

    try:
        with np.load(os.path.join('.MDC_cut', 'efit.npz'), 'rb') as f:
            ko = str(f['ko'])
            fphi = f['fphi']
            epos = f['epos']
            ffphi = f['ffphi']
            efwhm = f['efwhm']
            fk = f['fk']
            emin = f['emin']
            emax = f['emax']
            semin = f['semin']
            semax = f['semax']
            seaa1 = f['seaa1']
            seaa2 = f['seaa2']
            sefp = f['sefp']
            sefi = f['sefi']
            print('\033[90mEDC Fitted Data preloaded (Casa)\033[0m')
        fpr = 1
    except:
        ko = ''
        fphi = []
        epos = []
        ffphi = []
        efwhm = []
        fk = []
        emin = []
        emax = []
        semin = []
        semax = []
        seaa1 = []
        seaa2 = []
        sefp = []
        sefi = []
        # seresult = []
        # secst = []
        print('\033[90mNo EDC fitted data preloaded (Casa)\033[0m')

    try:
        with np.load(os.path.join('.MDC_cut', 'mfit.npz'), 'rb') as f:
            ko = str(f['ko'])
            fev = f['fev']
            rpos = f['rpos']
            ophi = f['ophi']
            fwhm = f['fwhm']
            pos = f['pos']
            kmin = f['kmin']
            kmax = f['kmax']
            skmin = f['skmin']
            skmax = f['skmax']
            smaa1 = f['smaa1']
            smaa2 = f['smaa2']
            smfp = f['smfp']
            smfi = f['smfi']
            try:
                smresult = f['smresult']
                smcst = f['smcst']
                print('\033[90mMDC Fitted Data preloaded (lmfit)\033[0m')
            except:
                print('\033[90mMDC Fitted Data preloaded (Casa)\033[0m')
        fpr = 1
    except:
        ko = ''
        fev = []
        rpos = []
        ophi = []
        fwhm = []
        pos = []
        kmin = []
        kmax = []
        skmin = []
        skmax = []
        smaa1 = []
        smaa2 = []
        smfp = []
        smfi = []
        smresult = []
        smcst = []
        print('\033[90mNo MDC fitted data preloaded (Casa)\033[0m')

    try:
        cdata = np.load(os.path.join('.MDC_cut', 'colormaps.npz'), allow_pickle=True)
        colors = list(cdata["colors"])
        scales = list(cdata["scales"])
        vmin = float(cdata["vmin"])
        vmax = float(cdata["vmax"])
        colormap_name = str(cdata["name"])
        pr_cmap = LinearSegmentedColormap.from_list(colormap_name, list(zip(scales, colors)))
        mpl.colormaps.register(pr_cmap, force=True)
        print('\033[90mLast User Defined Colormap preloaded\033[0m')
    except:
        pr_cmap, colormap_name = None, ''
        pass

    emf='KE'
    try:
        vfe=e_photon
    except:
        vfe=21.2

    '''
    try:
        with np.load(os.path.join('.MDC_cut', 'efpara.npz'),'rb') as f:
            rdd=f['path']
            fphi=f['fphi']
            efwhm=f['efwhm']
            epos=f['epos']
            semin=f['semin']
            semax=f['semax']
            seaa1=f['seaa1']
            seaa2=f['seaa2']
            sefp=f['sefp']
            sefi=f['sefi']
            print('EDC Fitted Data preloaded')
    except:
        print('No EDC fitted data preloaded')
        
    try:
        with np.load(os.path.join('.MDC_cut', 'mfpara.npz'),'rb') as f:
            rdd=f['path']
            fev=f['fev']
            fwhm=f['fwhm']
            pos=f['pos']
            skmin=f['skmin']
            skmax=f['skmax']
            smaa1=f['smaa1']
            smaa2=f['smaa2']
            smfp=f['smfp']
            smfi=f['smfi']
            print('MDC Fitted Data preloaded')
    except:
        print('No MDC fitted data preloaded')
    '''

    icon = IconManager()
    g.iconphoto(True, tk.PhotoImage(data=b64decode(icon.icon)))

    fr_main = tk.Frame(g, bg="white")
    fr_main.pack(side=tk.TOP, fill='both', expand=True)
    
    fr = tk.Frame(fr_main, bg='white')
    fr.grid(row=0, column=0)
    fr_info = tk.Frame(fr,bg='white')
    fr_info.pack(side=tk.TOP)
    fr_tool = tk.Frame(fr_info,bg='white',width=25)
    fr_tool.pack(fill='x')
    l_path = tk.Text(fr_info, wrap='word', font=("Arial", size(12), "bold"), bg="white", fg="black", state='disabled',height=3,width=25)
    l_path.pack(fill='x')
    # info = tk.Label(fr_main,text='                                   \n\n\n\n\n\n\n\n\n\n\n\n\n', font=("Arial", size(14), "bold"), bg="white", fg="black",padx = 30,pady=30)
    xscroll = tk.Scrollbar(fr_info, orient='horizontal')
    xscroll.pack(side='bottom', fill='x')
    yscroll = tk.Scrollbar(fr_info, orient='vertical')
    yscroll.pack(side='right', fill='y')
    info = tk.Text(fr_info, wrap='none', font=("Arial", size(14), "bold"), bg="white", fg="black", state='disabled',
                height=10, width=25, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
    info.pack()
    xscroll.config(command=info.xview)
    yscroll.config(command=info.yview)
    fr_mod = tk.Frame(fr,bg='white')
    fr_mod.pack(side=tk.TOP)
    b_name = tk.Button(fr_mod, text='Modify Name', font=('Arial', size(10), 'bold'),command=cname)
    b_name.grid(row=0,column=0)
    b_excitation = tk.Button(fr_mod, text='Modify Excitation Energy', font=('Arial', size(10), 'bold'),command=cexcitation)
    b_excitation.grid(row=0,column=1)
    b_desc = tk.Button(fr_mod, text='Modify Description', font=('Arial', size(10), 'bold'),command=desc)
    b_desc.grid(row=0,column=2)  
    
    
    # lfit = tk.Frame(step, bg='white')
    # lfit.grid(row=3, column=1)
    # lmfit = tk.Button(lfit, text='Load MDC fitted File', font=(
    #     "Arial", size(12), "bold"), width=16, height='1', command=loadmfit, bd=5, fg='blue')
    # lmfit.grid(row=0, column=0)
    # lefit = tk.Button(lfit, text='Load EDC fitted File', font=(
    #     "Arial", size(12), "bold"), width=16, height='1', command=loadefit, bd=5, fg='black')
    # lefit.grid(row=0, column=1)

    # cfit = tk.Frame(step, bg='white')
    # cfit.grid(row=4, column=1)
    # b_cmfit = tk.Button(cfit, text='Clear MDC fitted File', font=(
    #     "Arial", size(12), "bold"), width=16, height='1', command=clmfit, bd=5, fg='blue')
    # b_cmfit.grid(row=0, column=0)
    # b_cefit = tk.Button(cfit, text='Clear EDC fitted File', font=(
    #     "Arial", size(12), "bold"), width=16, height='1', command=clefit, bd=5, fg='black')
    # b_cefit.grid(row=0, column=1)


    # lbb = tk.Button(step, text='Load Bare Band File', font=(
    #     "Arial", size(12), "bold"), width=16, height='1', command=bareband, bd=5, fg='blue')
    # lbb.grid(row=5, column=1)
    

    plots = tk.Frame(fr, bg='white')
    plots.pack(side=tk.TOP)
    
    cmf = tk.Frame(plots, bg='white')
    cmf.grid(row=0, column=1)

    cmbf = tk.Frame(cmf, bg='white')
    cmbf.grid(row=0, column=0)
    
    bchcmp = tk.Button(cmbf, text='Change cmap', font=(
        "Arial", size(12), "bold"), height='1', command=Chcmp, border=2)
    bchcmp.pack(side='left', padx=2, pady=2)
    bdefcmp = tk.Button(cmbf, text='User Defined cmap', font=(
        "Arial", size(12), "bold"), height='1', command=def_cmap, border=2)
    bdefcmp.pack(side='left', padx=2, pady=2)

    cmlf = tk.Frame(cmf, bg='white')
    cmlf.grid(row=1, column=0)
    if pr_cmap is not None:
        optionList3 = ['prevac_cmap', colormap_name, 'terrain', 'custom_cmap1', 'custom_cmap2', 'custom_cmap3', 'custom_cmap4', 'viridis', 'turbo', 'inferno', 'plasma', 'copper', 'grey', 'bwr']
    else:
        optionList3 = ['prevac_cmap', 'terrain', 'custom_cmap1', 'custom_cmap2', 'custom_cmap3', 'custom_cmap4', 'viridis', 'turbo', 'inferno', 'plasma', 'copper', 'grey', 'bwr']
    setcmap = tk.OptionMenu(cmlf, value3, *optionList3)
    setcmap.grid(row=0, column=1)
    cmp = plt.colormaps()
    cm = tk.OptionMenu(cmlf, value3, *cmp)
    cm.grid(row=1, column=1)

    c1 = tk.Label(cmlf, text='Commonly Used:', font=(
        "Arial", size(12)), bg="white", height='1')
    c1.grid(row=0, column=0)
    c2 = tk.Label(cmlf, text='All:', font=("Arial", size(12)), bg="white", height='1')
    c2.grid(row=1, column=0)
    
    frraw = tk.Frame(fr, bg='white')
    frraw.pack(side=tk.TOP)

    optionList = ['Raw Data', 'E-k Diagram', 'MDC Normalized',
                'First Derivative', 'Second Derivative', 'MDC Curves', 'E-k with MDC Curves']   # 選項
    value = tk.StringVar()                                        # 取值
    value.set('---Plot1---')
    # 第二個參數是取值，第三個開始是選項，使用星號展開
    menu1 = tk.OptionMenu(frraw, value, *optionList)
    menu1.grid(row=0, column=1)
    value.trace_add('write', plot1)

    b_spec = tk.Button(frraw, text='Spectrogram', fg='red', font=("Arial", size(12), "bold"),width=10, height='1', command=trans_plot, bd=5)
    # b_spec.grid(row=0, column=1)


    frfit = tk.Frame(plots, bg='white')
    # frfit.grid(row=2, column=1)
    optionList1 = ['MDC fitted Data', 'EDC fitted Data',
                'Real Part', 'Imaginary Part']   # 選項
    value1 = tk.StringVar()                                        # 取值
    value1.set('---Plot2---')
    # 第二個參數是取值，第三個開始是選項，使用星號展開
    menu2 = tk.OptionMenu(frraw, value1, *optionList1)
    menu2.grid(row=1, column=1)
    value1.trace_add('write', plot2)
    
    ##### Base and FWHM not packing to frfit #####
    l_fit = tk.Label(frfit, text='Base counts:', font=(
        "Arial", size(10), "bold"), bg="white", height='1', bd=5)
    # l_fit.grid(row=0, column=1)
    base = tk.StringVar()
    base.set('0')
    base.trace_add('write', fbase)
    in_fit = tk.Entry(frfit, font=("Arial", size(10)), width=5, textvariable=base, bd=5)
    # in_fit.grid(row=0, column=2)
    b_fit = tk.Button(frfit, text='Fit FWHM', font=(
        "Arial", size(10), "bold"), bg="white", height='1', bd=5, command=fitgl)
    # b_fit.grid(row=0, column=3)
    ##### Base and FWHM not packing to frfit #####

    optionList2 = ['Real & Imaginary', 'KK Transform Real & Imaginary', 'KK Transform Real Part', 'KK Transform Imaginary Part', 'KK Transform Real Part 2nd Derivative', 'KK Transform Imaginary Part 1st Derivative', 'Data Plot with Pos', 'Data Plot with Pos and Bare Band']   # 選項
    value2 = tk.StringVar()                                        # 取值
    value2.set('---Plot3---')
    # 第二個參數是取值，第三個開始是選項，使用星號展開
    menu3 = tk.OptionMenu(frraw, value2, *optionList2)
    menu3.grid(row=2, column=1)
    value2.trace_add('write', plot3)


    m1 = tk.Label(frraw, text='Raw', font=(
        "Arial", size(12), "bold"), bg="white", fg='red')
    m1.grid(row=0, column=0)
    m2 = tk.Label(frraw, text='Fit', font=(
        "Arial", size(12), "bold"), bg="white", fg='blue')
    m2.grid(row=1, column=0)
    m3 = tk.Label(frraw, text='Transform', font=(
        "Arial", size(12), "bold"), bg="white", fg="blue")
    m3.grid(row=2, column=0)

    
    fr_state = tk.Frame(fr_main, bg='white')
    fr_state.grid(row=0, column=2)

    st = queue.Queue(maxsize=0)
    state = tk.Label(fr_state, text=f"Version: {__version__}", font=(
        "Arial", size(14), "bold"), bg="white", fg="black", wraplength=250, justify='center')
    state.grid(row=0, column=0)

    Icon = [icon.icon1, icon.icon2, icon.icon3, icon.icon4, icon.icon5, icon.icon6, icon.icon7, icon.icon8, icon.icon9, icon.icon10, icon.icon11, icon.icon12, icon.icon13, icon.icon14, icon.icon15, icon.icon16, icon.icon17, icon.icon18, icon.icon19, icon.icon20]
    img = []
    tdata = []
    for _ in Icon:
        if _:
            tdata.append(io.BytesIO(b64decode(_)))
            timg = Image.open(io.BytesIO(b64decode(_))).resize([250, 250])
            tk_img = ImageTk.PhotoImage(timg)
            img.append(tk_img)
    trd = np.random.randint(len(img))
    timg = ImageTk.PhotoImage(Image.open(tdata[trd]).resize([250, 250]))
    tdata = tdata[trd]
    limg = tk.Label(fr_state, image=timg, width='250', height='250', bg='white')
    limg.grid(row=1, column=0)

    
    exf = tk.Frame(fr_state, bg='white')
    exf.grid(row=2, column=0)

    clim = tk.Frame(exf, bg='white')
    clim.grid(row=0, column=0)
    lcmax = tk.Label(clim, text='Maximum', font=(
        "Arial", size(12)), bg='white', fg='white')
    lcmax.grid(row=0, column=0)
    lcmin = tk.Label(clim, text='Minimum', font=(
        "Arial", size(12)), bg='white', fg='white')
    lcmin.grid(row=1, column=0)
    cmax = tk.Frame(clim, bg='white', width=15, bd=5)
    cmax.grid(row=0, column=1)
    cmin = tk.Frame(clim, bg='white', width=15, bd=5)
    cmin.grid(row=1, column=1)


    cM = tk.DoubleVar()
    cm = tk.DoubleVar()
    cM.set(10000)
    cm.set(-10000)
    vcmax = tk.DoubleVar()
    vcmax.set(10000)
    vcmax.trace_add('write', cmaxrange)
    Cmax = tk.Scale(cmax, from_=cm.get(), to=cM.get(), orient='horizontal',
                    variable=vcmax, state='disabled', bg='white', fg='white')
    Cmax.pack()
    vcmin = tk.DoubleVar()
    vcmin.set(-10000)
    vcmin.trace_add('write', cminrange)
    Cmin = tk.Scale(cmin, from_=cm.get(), to=cM.get(), orient='horizontal',
                    variable=vcmin, state='disabled', bg='white', fg='white')
    Cmin.pack()
    
    step = tk.Frame(fr_state, bg='white')
    step.grid(row=3, column=0)

    # l1 = tk.Label(step, text='Step 1', font=(
    #     "Arial", size(12), "bold"), bg="white", fg='red')
    # l1.grid(row=0, column=0)
    # l2 = tk.Label(step, text='Step 2', font=(
    #     "Arial", size(12), "bold"), bg="white", fg='blue')
    # l2.grid(row=1, column=0)
    l3 = tk.Label(step, text='k offset (deg)', font=(
        "Arial", size(12), "bold"), bg="white", fg="black")
    l3.grid(row=2, column=0)
    # l4 = tk.Label(step, text='Step 3', font=(
    #     "Arial", size(12), "bold"), bg="white", fg='blue')
    # l4.grid(row=3, column=0)
    # l5 = tk.Label(step, text='Step 4', font=("Arial", size(12), "bold"),
    #             bg="white", fg="blue", height=1)
    # l5.grid(row=5, column=0)

    fremfit = tk.Frame(master=step)
    fremfit.grid(row=0, column=1)
    # lf = tk.Button(fremfit, text='Load Raw Data', font=(
    #     "Arial", size(12), "bold"), fg='red', width=15, height='1', command=load, bd=9)
    # lf.grid(row=0, column=0)
    # bmfit = tk.Button(fremfit, text='MDC Fit', font=(
    #     "Arial", size(12), "bold"), fg='red', width=8, height='1', command=cmfit, bd=9)
    # bmfit.grid(row=0, column=1)
    # befit = tk.Button(fremfit, text='EDC Fit', font=(
    #     "Arial", size(12), "bold"), fg='red', width=8, height='1', command=cefit, bd=9)
    # befit.grid(row=0, column=2)


    cut = tk.Frame(step, bg='white')
    # cut.grid(row=1, column=1)
    # mdccut = tk.Button(cut, text='MDC cut', font=(
    #     "Arial", size(12), "bold"), width=8, height='1', command=ecut, bd=5, fg='blue')
    # mdccut.grid(row=0, column=0)
    # edccut = tk.Button(cut, text='EDC cut', font=(
    #     "Arial", size(12), "bold"), width=8, height='1', command=angcut, bd=5, fg='black')
    # edccut.grid(row=0, column=1)
    l_lowlim = tk.Label(cut, text='Lower Limit', font=(
        "Arial", size(10), "bold"), bg="white", fg="black", height=1)
    l_lowlim.grid(row=0, column=2)
    lowlim = tk.StringVar()
    lowlim.set('0')
    lowlim.trace_add('write', flowlim)
    in_lowlim = tk.Entry(cut, font=("Arial", size(10), "bold"),
                        width=7, textvariable=lowlim, bd=5)
    in_lowlim.grid(row=0, column=3)


    k_offset = tk.StringVar()
    try:
        k_offset.set(ko)
    except:
        k_offset.set('0')
    k_offset.trace_add('write', reload)
    koffset = tk.Entry(step, font=("Arial", size(12), "bold"),
                    width=12, textvariable=k_offset, bd=9)
    koffset.grid(row=2, column=1)
    
    bb_offset = tk.StringVar()
    try:
        bb_offset.set(bbo)
    except:
        bb_offset.set('0')
    bb_offset.trace_add('write', fbb_offset)
    bboffset = tk.Entry(step, font=("Arial", size(12), "bold"),
                        width=12, textvariable=bb_offset, bd=9)
    bboffset.grid(row=3, column=1)
    bbk_offset = tk.StringVar()
    try:
        bbk_offset.set(bbk)
    except:
        bbk_offset.set('1')
    bbk_offset.trace_add('write', fbbk_offset)
    bbkoffset = tk.Entry(step, font=("Arial", size(12), "bold"),
                        width=12, textvariable=bbk_offset, bd=9)
    bbkoffset.grid(row=4, column=1)
    l6 = tk.Label(step, text='Bare band E offset (meV)', font=(
        "Arial", size(12), "bold"), bg="white", fg="black", height=1)
    l6.grid(row=3, column=0)
    l7 = tk.Label(step, text='Bare band k ratio', font=(
        "Arial", size(12), "bold"), bg="white", fg="black", height=1)
    l7.grid(row=4, column=0)

    figfr = tk.Frame(fr_main, bg='white')
    figfr.grid(row=0, column=1, sticky='nsew')
    global figy
    figy = 8.5 if osf<=100 else 8.25 if osf<=150 else 8
    figx = 11.5 if osf<=100 else 11.25 if osf<=150 else 11
    fig = Figure(figsize=(figx*scale, figy*scale), layout='constrained')
    out = FigureCanvasTkAgg(fig, master=figfr)
    out.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    ao = None
    ax= fig.subplots()
    tim = np.asarray(Image.open(tdata), dtype=np.uint8)
    ax.imshow(tim, aspect='equal', alpha=0.4)
    fontdict = {
    'fontsize': size(40),
    'fontweight': 'bold',
    'fontname': 'Arial'
}
    ax.text(tim.shape[0]/2, tim.shape[1]/2, f"Version: {__version__}\n\nRelease Date: {__release_date__}", fontdict=fontdict, color='black', ha='center', va='center')
    ax.axis('off')
    out.draw()

    xydata = tk.Frame(figfr, bg='white')
    xydata.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    xdata = tk.Label(xydata, text='xdata:', font=(
        "Arial", size(16), "bold"), width='15', height='1', bd=9, bg='white')
    xdata.grid(row=0, column=0)
    ydata = tk.Label(xydata, text='ydata:', font=(
        "Arial", size(16), "bold"), width='15', height='1', bd=9, bg='white')
    ydata.grid(row=0, column=1)
    
    v_fe = tk.StringVar()
    v_fe.set(str(vfe))
    b_emode = tk.Button(xydata, text='K.E.', fg='blue', font=("Arial", size(16), "bold"), width=5, height='1', command=emode, bd=9)
    b_emode.grid(row=0, column=2)
    b_copyimg = tk.Button(xydata, fg='red', text='Copy Image to Clipboard', font=('Arial', size(16), 'bold'), command=f_copy_to_clipboard, bd=9)
    b_copyimg.grid(row=0, column=3)
    
    
    dl=0
    b_sw = tk.Button(xydata, text='dot', font=('Arial', size(16), 'bold'), command=dl_sw, bd=9)

    lcmp = tk.Frame(plots, bg='white')
    lcmp.grid(row=0, column=0)

    lcmpd = Figure(figsize=(0.75*scale, 1*scale), layout='constrained')
    cmpg = FigureCanvasTkAgg(lcmpd, master=lcmp)
    cmpg.get_tk_widget().grid(row=0, column=1)
    lsetcmap = tk.Label(lcmp, text='Colormap:', font=(
        "Arial", size(12), "bold"), bg="white", height='1', bd=9)
    lsetcmap.grid(row=0, column=0)
    chcmp()

    # expf = tk.Frame(exf, bg='white')
    # expf.grid(row=1,column=0)
    # ex = tk.Button(expf, fg='red', text='Export Graph', font=(
    #     "Arial", size(12), "bold"), height='1', command=exp, bd=9)
    # ex.grid(row=0, column=0)
    # exo = tk.Button(expf, fg='blue', text='Export to Origin', font=(
    #     "Arial", size(12), "bold"), height='1', command=gui_exp_origin, bd=9)
    # exo.grid(row=0, column=1)
    # extm = tk.Button(exf, text='Export MDC Fitted Data (k offset)', font=(
    #     "Arial", size(12), "bold"), height='1', command=exptm, bd=9)
    # extm.grid(row=2, column=0)
    # exte = tk.Button(exf, text='Export EDC Fitted Data (k offset)', font=(
    #     "Arial", size(12), "bold"), height='1', command=expte, bd=9)
    # exte.grid(row=3, column=0)
    
    
    tt = threading.Thread(target=tstate)
    tt.daemon = True
    tt.start()
    if lfs is None:
        b_name.config(state='disable')
        b_excitation.config(state='disable')
        b_desc.config(state='disable')
    if data is not None:
        pr_load(data)
        b_tools = tk.Button(fr_tool, text='Batch Master', command=tools, width=12, height=1, font=('Arial', size(12), "bold"), bg='white')
        nlist = lfs.name
        namevar = tk.StringVar(value=nlist[0])
        l_name = tk.OptionMenu(fr_tool, namevar, *nlist, command=change_file)
        if len(lfs.name) > 1:
            if len(lfs.n)>0:lfs.sort='no'
            b_tools.grid(row=0, column=0)
            if len(namevar.get()) >30:
                l_name.config(font=('Arial', size(11), "bold"))
            elif len(namevar.get()) >20:
                l_name.config(font=('Arial', size(12), "bold"))
            else:
                l_name.config(font=('Arial', size(14), "bold"))
            l_name.grid(row=0, column=1)
        if lfs.f_npz[0]:
            npzf = True
            koffset.config(state='normal')
            k_offset.set('0')
            koffset.config(state='disabled')
    else:
        b_tools, l_name = None, None
    # from tool.MDC_Fitter import gl1, gl2, fgl2
    print(f"\033[36mVersion: {__version__}")
    print(f"Release Date: {__release_date__}\n\033[0m")
    g.bind('<Configure>', lambda event: on_configure(g, event))
    ###### hotkey ######
    g.bind('<Return>', plot)
    g.bind('<Up>', cf_up)
    g.bind('<Down>', cf_down)
    g.bind('<MouseWheel>', scroll)
    g.bind('<MouseWheel>', scroll)
    g.bind('<Control-o>', load)
    g.bind("<F1>", loadmfit)
    g.bind("<F2>", loadefit)
    g.bind("<F3>", bareband)
    g.bind("<F4>", trans_plot)
    g.bind("<F5>", cmfit)
    g.bind("<F6>", cefit)
    g.bind("<F7>", ecut)
    g.bind("<F8>", angcut)
    g.bind("<F9>", calculator)
    g.bind("<F10>", exp)
    g.bind("<F11>", gui_exp_origin)
    g.bind("<F12>", view_3d)
    g.bind('<Control-p>', fit_so_app)
    main_motion = MainMotion()
    main_notify_cid = out.mpl_connect('motion_notify_event', main_motion.move)
    main_press_cid = out.mpl_connect('button_press_event', main_motion.press)
    main_release_cid = out.mpl_connect('button_release_event', main_motion.release)
    g.update()
    screen_width = g.winfo_reqwidth()
    screen_height = g.winfo_reqheight()
    # print(f"Screen Width: {screen_width}, Screen Height: {screen_height}")
    g.geometry(f"{screen_width}x{screen_height}+0+{sc_y}")
    # g.protocol("WM_DELETE_WINDOW", quit)
    g.update()
    if lfs is not None: # CEC loaded old data to show the cutting rectangle
        if lfs.cec is not None:
            lfs.cec.tlg.lift()
            lfs.cec.tlg.focus_force()
    version_check()
    # g_mem = (g_mem - psutil.virtual_memory().available)/1024**3   # Main GUI memory in GB
    g_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**3  # Main GUI memory in GB
    app_pars.g_mem = g_mem
    # print(f"Main GUI memory usage: {g_mem:.2f} GB")
    # snapshot = tracemalloc.take_snapshot()
    # top_stats = snapshot.statistics('lineno')
    
    # print("記憶體使用量 Top 10:")
    # for index, stat in enumerate(top_stats[:10], 1):
    #     print(f"{index}. {stat}")
    g.mainloop()
