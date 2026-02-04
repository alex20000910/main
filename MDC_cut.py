# MDC cut GUI
__version__ = "9.1"
__release_date__ = "2026-02-05"
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
if os.name == 'nt':
    from ctypes import windll
else:
    import tempfile
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
    "tkinterdnd2==0.4.3",
    "markdown==3.10.1",
    "tkhtmlview==0.3.1"
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
    "google-crc32c==1.8.0",  # for numcodecs
    "markdown==3.10.1",
    "tkhtmlview==0.3.1"
    ]
if os.name == 'posix':
    REQUIREMENTS.remove(REQUIREMENTS[9])  # no pywin32 in Linux or MacOS
    REQUIREMENTS.remove(REQUIREMENTS[9]) # no originpro in Linux or MacOS

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
        return 0
    except Exception as e:
        print("Failed to download source file:", e, file=sys.stderr)
        print("\033[35mPlease ensure the Network is connected. \033[0m", file=sys.stderr)
        return -1

def get_src(ver=False):
    branch = 'update'
    url = [rf"https://github.com/alex20000910/main/blob/{branch}/MDC_cut.py",
           rf"https://github.com/alex20000910/main/blob/{branch}/release_note.md",
           rf"https://github.com/alex20000910/main/blob/{branch}/src/viridis_2D.otp",
           rf"https://github.com/alex20000910/main/blob/{branch}/src/MDC_cut_utility.py",
           rf"https://github.com/alex20000910/main/blob/{branch}/src/tool/__init__.py",
           rf"https://github.com/alex20000910/main/blob/{branch}/src/tool/util.py",
           rf"https://github.com/alex20000910/main/blob/{branch}/src/tool/loader.py",
           rf"https://github.com/alex20000910/main/blob/{branch}/src/tool/spectrogram.py",
           rf"https://github.com/alex20000910/main/blob/{branch}/src/tool/SO_Fitter.py",
           rf"https://github.com/alex20000910/main/blob/{branch}/src/tool/VolumeSlicer.py",
           rf"https://github.com/alex20000910/main/blob/{branch}/src/tool/CEC.py",
           rf"https://github.com/alex20000910/main/blob/{branch}/src/tool/DataViewer.py",
           rf"https://github.com/alex20000910/main/blob/{branch}/src/tool/MDC_Fitter.py",
           rf"https://github.com/alex20000910/main/blob/{branch}/src/tool/EDC_Fitter.py",
           rf"https://github.com/alex20000910/main/blob/{branch}/src/tool/window.py",
           rf"https://github.com/alex20000910/main/blob/{branch}/src/tool/RawDataViewer.py",
           rf"https://github.com/alex20000910/main/blob/{branch}/src/tool/qt_util.py"]
    for i, v in enumerate(url):
        if i < 4:
            out_path = os.path.join(cdir, '.MDC_cut', os.path.basename(v))
        else:
            out_path = os.path.join(cdir, '.MDC_cut', 'tool', os.path.basename(v))
        status = get_file_from_github(v, out_path)
        if ver and i == 1:
            break
    return status

def cal_ver(ver: str) -> int:
    '''
    Version string to integer for comparison
    e.g. '8.3.1' -> 80301\n
    # Parameters
    - ver: str\n
    'major.minor.patch' or 'major.minor', 0-99 for each part
    # Returns
    output: int
    '''
    ver = [int(i) for i in ver.split('.')]
    if len(ver) != 3:
        ver.append(0)
    ver = ver[0]*10000 + ver[1]*100 + ver[2]
    return ver


def check_github_connection() -> bool:
    try:
        import requests
    except ImportError:
        requests = None
        
    url = "https://github.com"
    if requests:
        try:
            response = requests.get(url, timeout=5)  # 設定 5 秒的超時時間
            if response.status_code == 200:
                return True
            else:
                return False
        except requests.ConnectionError:
            return False
        except requests.Timeout:
            return False
        except Exception:
            return False
    else:
        import urllib.request
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return True
                else:
                    return False
        except Exception:
            return False

def force_update():
    f = check_github_connection()
    if not f:
        print('\n\033[31mPlease check your network connection!\033[0m\n')
        return
    get_src(ver=True)
    path = os.path.join(cdir, '.MDC_cut', 'MDC_cut.py')
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('__version__ =') or line.startswith("__version__="):
                remote_ver = line.split('=')[1].strip().strip('"').strip("'")
                break
    if cal_ver(remote_ver) > cal_ver(__version__):
        src = path
        dst = os.path.join(cdir, f'{app_name}.py')
        os.chdir(cdir)
        if os.name == 'nt':
            os.system(f'copy "{src}" "{dst}" > nul')
            os.system(rf'start "" cmd /C "chcp 65001 > nul && {sys.executable} -W ignore::SyntaxWarning -W ignore::UserWarning "{app_name}.py""')
        elif os.name == 'posix':
            try:
                os.system(f'cp "{src}" "{dst}"')
                os.system(f'{sys.executable} -W ignore::SyntaxWarning -W ignore::UserWarning "{dst}" &')
            except:
                os.system(f'cp "{src}" "{dst}"')
                os.system(f'{sys.executable} -W ignore::SyntaxWarning -W ignore::UserWarning "{dst}" &')
        os.remove(src)
        quit()
    if os.name == 'nt':
        os.system(f'del {path}')
    elif os.name == 'posix':
        os.system(f'rm -rf {path}')

# set up .MDC_cut folder
os.chdir(cdir)
if not os.path.exists('.MDC_cut'):
    os.makedirs('.MDC_cut')
    force_update()
if os.name == 'nt':
    os.system('attrib +h +s .MDC_cut')
sys.path.append(os.path.join(cdir, '.MDC_cut'))
sys.path.append(os.path.join(cdir, 'src'))

# upgrade check
v_check_path = os.path.join(cdir, '.MDC_cut', 'version.check')
if os.path.exists(v_check_path):
    with open(v_check_path, mode='r') as f:
        ver = f.read().strip()
    if cal_ver(ver) < cal_ver('8.5.1'):
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
        if os.name == 'nt':
            shutil.rmtree(os.path.join(cdir, '.MDC_cut', 'cut_temp_save'))
            # os.system(f'rmdir /s /q "{os.path.join(cdir, ".MDC_cut", "cut_temp_save")}"')
        elif os.name == 'posix':
            os.system(f'rm -rf "{os.path.join(cdir, ".MDC_cut", "cut_temp_save")}"')
    if os.path.exists(os.path.join(cdir, '.MDC_cut', 'cube_temp_save')):
        if os.name == 'nt':
            shutil.rmtree(os.path.join(cdir, '.MDC_cut', 'cube_temp_save'))
            os.system(f'rmdir /s /q "{os.path.join(cdir, ".MDC_cut", "cube_temp_save")}"')
        elif os.name == 'posix':
            os.system(f'rm -rf "{os.path.join(cdir, ".MDC_cut", "cube_temp_save")}"')

# make sure pip is installed
try:
    os.chdir(os.path.join(cdir, '.MDC_cut'))
    if os.path.exists('pip_check.txt')==0:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "--version"])
            with open('pip_check.txt', 'w', encoding = 'utf-8') as f:
                f.write('pip found')
                f.close()
            os.system('cls' if os.name == 'nt' else 'clear')
        except subprocess.CalledProcessError:
            try:
                if os.name == 'nt':
                    print('pip not found\nOS: Windows\nInstalling pip...')
                    os.system(f'{sys.executable} -m ensurepip')    #install pip
                    os.system(f'{sys.executable} -W ignore::SyntaxWarning -W ignore::UserWarning "'+os.path.abspath(inspect.getfile(inspect.currentframe()))+'"')  #restart the script to ensure pip works without potential errors
                elif os.name == 'posix':
                    print('pip not found\nOS: Linux or MacOS\nInstalling pip...')
                    try:    #python3 if installed
                        os.system(f'{sys.executable} -m ensurepip')   #install pip
                        os.system(f'{sys.executable} -W ignore::SyntaxWarning -W ignore::UserWarning "'+os.path.abspath(inspect.getfile(inspect.currentframe()))+'"')   #restart the script to ensure pip works without potential errors
                    except: #python2.7(default in MacOS)
                        os.system(f'{sys.executable} -m ensurepip')
                        os.system(f'{sys.executable} -W ignore::SyntaxWarning -W ignore::UserWarning "'+os.path.abspath(inspect.getfile(inspect.currentframe()))+'"')
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
        os.chdir(cdir)
        os.system(rf'start "" cmd /C "chcp 65001 > nul && {sys.executable} -W ignore::SyntaxWarning -W ignore::UserWarning "{app_name}.py""')
    elif os.name == 'posix':
        script = rf'''
        tell application "Terminal"
            activate
            do script "cd {cdir} && {sys.executable} -W ignore::SyntaxWarning -W ignore::UserWarning {app_name}.py"
        end tell
        '''
        os.system('clear')
        try:
            subprocess.run(['osascript', '-e', script])
        except:
            subprocess.run(['osascript', '-e', script])

def install(s: str = ''):
    print('Some Modules Not Found')
    try:
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
    except EOFError:
        restart()
        os.system('cls' if os.name == 'nt' else 'clear')
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
    if os.name == 'nt':
        import win32clipboard
    if __name__ == '__main__':
        if os.name == 'nt':
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
        import markdown
        from tkhtmlview import HTMLLabel
except ModuleNotFoundError:
    install()
    restart()
    quit()

try:
    from MDC_cut_utility import *
    from tool.loader import loadfiles, mloader, eloader, tkDnD_loader, file_loader, data_loader, load_h5, load_json, load_npz, load_txt
    from tool.spectrogram import spectrogram, lfs_exp_casa
    if __name__ == '__main__':
        from tool.util import MDC_param, EDC_param, origin_util, motion, plots_util, exp_util
        from tool.SO_Fitter import SO_Fitter
        from tool.CEC import CEC, call_cec
        from tool.window import AboutWindow, EmodeWindow, ColormapEditorWindow, c_attr_window, c_name_window, c_excitation_window, c_description_window, VersionCheckWindow, CalculatorWindow, Plot1Window, Plot1Window_MDC_curves, Plot1Window_Second_Derivative, Plot3Window
except ImportError as e:
    print(e)
    print('Some source files missing. Downloading...')
    status = get_src()
    if status == 0:
        restart()
        quit()
    else:
        input('\n\033[31mPlease check your network connection!\033[0m\n')

if __name__ == '__main__':
    pid = os.getpid()
    # g_mem = psutil.virtual_memory().available
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
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
            super().__init__(menu1, menu2, menu3, in_fit, b_fit, l_path, info, cdir, lfs, scale)
            self.pr_load(data)

        @override
        def pars(self):
            for i, j in zip(['name','dvalue','e_photon','description','dpath', 'ev', 'phi'], [self.name, self.dvalue, self.e_photon, self.description, self.dpath, self.ev, self.phi]):
                set_globals(j, i)
            
    class main_loader(file_loader):
        def __init__(self, files: tuple[str]|Literal['']):
            super().__init__(files, dpath, value3.get(), lfs, g, app_pars, st, limg, img, b_name, b_excitation, b_desc, koffset, k_offset, fr_tool, b_tools, l_name, scale)
        
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
    
    class ToolTip(ToolTip_util):
        def __init__(self, widget: tk.Widget, text: str, accelerator=None):
            super().__init__(widget, text, accelerator,
                             icon_manager, scaled_font_size)
        
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
            for i, j in zip(['dpath', 'rdd'], [self.dpath, self.dpath]):
                set_globals(j, i)
            main_loader(lfs.opath)
        
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
    
    class origin(origin_util):
        def __init__(self):
            var_list = ['be', 'k', 'k_offset', 'bb_offset', 'bbk_offset',
                        'limg', 'img', 'st', 'im_kernel', 'emf', 'data', 'vfe', 'ev', 'phi',
                        'pos', 'fwhm', 'rpos', 'ophi', 'fev',
                        'epos', 'efwhm', 'fk', 'ffphi', 'fphi',
                        'cdir', 'dpath', 'bpath', 'app_name', 'npzf',
                        'pos_err', 'fwhm_err',
                        'g', 'gori', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11']
            for i in var_list:
                init_globals(i)
            super().__init__(be, k,
                 k_offset, bb_offset, bbk_offset,
                 limg, img, st, im_kernel,
                 emf, data, vfe, ev, phi,
                 pos, fwhm, rpos, ophi, fev,
                 epos, efwhm, fk, ffphi, fphi,
                 cdir, dpath, bpath, app_name, npzf,
                 pos_err, fwhm_err,
                 g, gori, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11)
    
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
            var_list = ['scale', 'value', 'value1', 'value2', 'k_offset', 'be', 'k', 'bb_offset', 'bbk_offset', 'ao', 'out', 'figy', 'rcx', 'rcy', 'xdata', 'ydata', 'emf', 'data', 'vfe', 'ev', 'phi', 'pos', 'fwhm', 'rpos', 'ophi', 'fev', 'epos', 'efwhm', 'fk', 'ffphi', 'fphi', 'mp', 'ep', 'mf', 'ef', 'xl', 'yl', 'tb0', 'tb0_', 'tb1', 'tb1_', 'tb2']
            for i in var_list:
                init_globals(i)
            super().__init__(scale, value, value1, value2, k_offset,
                 be, k, bb_offset, bbk_offset,
                 ao, out, figy,
                 rcx, rcy, xdata, ydata,
                 emf, data, vfe, ev, phi,
                 pos, fwhm, rpos, ophi, fev,
                 epos, efwhm, fk, ffphi, fphi,
                 mp, ep, mf, ef, xl, yl,
                 tb0, tb0_, tb1, tb1_, tb2)

    class ExpUtil(exp_util):
        def __init__(self):
            var_list = ['scale', 'value', 'value1', 'value2', 'value3', 'k_offset', 'be', 'k', 'bb_offset', 'bbk_offset',
                        'emf', 'data', 'vfe', 'ev', 'phi',
                        'pos', 'fwhm', 'rpos', 'ophi', 'fev',
                        'epos', 'efwhm', 'fk', 'ffphi', 'fphi',
                        'mp', 'ep', 'mf', 'ef', 'xl', 'yl', 'cm', 'cM', 'vcmin', 'vcmax', 'dl', 'st',
                        'pflag', 'limg', 'img', 'd', 'l', 'p', 'npzf', 'im_kernel',
                        'rx', 'ry', 'ix', 'iy']
            for i in var_list:
                init_globals(i)
            super().__init__(scale, value, value1, value2, value3, k_offset,
                             be, k, bb_offset, bbk_offset,
                             emf, data, vfe, ev, phi,
                             pos, fwhm, rpos, ophi, fev,
                             epos, efwhm, fk, ffphi, fphi,
                             mp, ep, mf, ef, xl, yl, cm, cM, vcmin, vcmax, dl, st,
                             pflag, limg, img, d, l, p, npzf, im_kernel,
                             rx, ry, ix, iy)
        
        @override
        def pars(self):
            var_list = ['f', 'f0', 'h1', 'h2']
            for i, j in zip(var_list, [self.f, self.f0, self.h1, self.h2]):
                set_globals(j, i)
        
        @override
        def show_info(self):
            show_info()
            
        @override
        def show_version(self):
            show_version()
    
    class PlotsUtil(plots_util):
        def __init__(self):
            var_list = ['scale', 'value', 'value1', 'value2', 'value3', 'be', 'k', 'k_offset', 'bb_offset', 'bbk_offset', 'b_sw',
                        'limg', 'img', 'st', 'im_kernel', 'optionList', 'optionList1', 'optionList2', 'emf', 'data', 'vfe', 'ev', 'phi',
                        'pos', 'fwhm', 'rpos', 'ophi', 'fev', 'epos', 'efwhm', 'fk', 'ffphi', 'fphi', 'mp', 'ep', 'mf', 'ef', 'npzf', 'fig', 'out',
                        'd', 'l', 'p', 'dl', 'rx', 'ry', 'ix', 'iy']
            for i in var_list:
                init_globals(i)
            super().__init__(scale, value, value1, value2, value3, be, k, k_offset, bb_offset, bbk_offset, b_sw,
                             limg, img, st, im_kernel, optionList, optionList1, optionList2,
                             emf, data, vfe, ev, phi,
                             pos, fwhm, rpos, ophi, fev,
                             epos, efwhm, fk, ffphi, fphi,
                             mp, ep, mf, ef, npzf, fig, out, d, l, p, dl, rx, ry, ix, iy)

        @override
        def pars_warn(self):
            set_globals(self.warn_str, 'warn_str')

        @override
        def pars1(self):
            var_list = ['h0', 'ao', 'xl', 'yl', 'pflag', 'rcx', 'rcy', 'acb', 'warn_str']
            for i, j in zip(var_list, [self.h0, self.ao, self.xl, self.yl, self.pflag, self.rcx, self.rcy, self.acb, self.warn_str]):
                set_globals(j, i)
                
        @override
        def pars2(self):
            var_list = ['rx', 'ry', 'ix', 'iy', 'pflag', 'warn_str']
            for i, j in zip(var_list, [self.rx, self.ry, self.ix, self.iy, self.pflag, self.warn_str]):
                set_globals(j, i)
                
        @override
        def pars3(self):
            var_list = ['h0', 'ao', 'xl', 'yl', 'pflag', 'tb0', 'tb0_', 'tb1', 'tb1_', 'tb2', 'rx', 'ry', 'ix', 'iy', 'warn_str']
            for i, j in zip(var_list, [self.h0, self.ao, self.xl, self.yl, self.pflag, self.tb0, self.tb0_, self.tb1, self.tb1_, self.tb2, self.rx, self.ry, self.ix, self.iy, self.warn_str]):
                set_globals(j, i)
            
        @override
        def show_version(self):
            show_version()
        
        @override
        def main_plot_bind(self):
            main_plot_bind()
        
        @override
        def climon(self):
            climon()
        
        @override
        def climoff(self):
            climoff()

@pool_protect
def suggest():
    global b_suggest
    if value.get() == 'Raw Data':
        b_suggest.config(text='Raw Data Viewer', bg='#aa0000', fg='white', font=('Arial', size(18), "bold"))
        ToolTip(b_suggest, "Use Qt Window to view raw data with high performance")
        b_suggest.config(command=lambda: qt_app(lfs.path))
        b_suggest.grid(row=1, column=0, pady=20)
        def job():
            import time
            for i in range(10):
                b_suggest.config(bg='white', fg='red')
                g.update_idletasks()
                time.sleep(0.5)
                b_suggest.config(bg='#aa0000', fg='white')
                g.update_idletasks()
                time.sleep(0.5)
        threading.Thread(target=job, daemon=True).start()
    else:
        b_suggest.grid_forget()

@pool_protect
def g_close(*e):
    try:
        g.destroy()
        plt.close('all')
        if os.name == 'posix':
            subprocess.run(['open', '-a', 'Terminal'])
            os.system(rf'''osascript -e 'tell application "Terminal" to quit' ''')
            # os.system('killall Terminal')
        quit()
    except:
        pass
    
@pool_protect
def f_help(*e):
    import webbrowser
    url = r"https://github.com/alex20000910/main"
    webbrowser.open(url)

@pool_protect
def about(*e):
    AboutWindow(master=g, scale=scale, version=__version__, release_date=__release_date__)

@pool_protect
def sample_data(*e):
    tg = wait(g, app_pars)
    tg.text('Preparing sample data...')
    R1 = np.linspace(5, 25, 201)
    if not os.path.exists(os.path.join(cdir, 'test_data')):
        os.makedirs(os.path.join(cdir, 'test_data'))
    files=[]
    for r1 in R1:
        path = rf"simulated_R1_{r1:.1f}_R2_0.h5"
        tpath = os.path.join(cdir, 'test_data', rf"simulated_R1_{r1:.1f}_R2_0.h5")
        files.append(tpath)
        if os.path.exists(tpath)==False:
            url=r"https://github.com/alex20000910/main/raw/refs/heads/main/test_data/"+path
            download(url, tpath)
    tg.done()
    tg = wait(g, app_pars)
    tg.text('Loading sample data...')
    o_load(drop=True, files=files)
    tg.done()
    g.after(500, lambda: print('Batch Master, k-Plane'))
    t_cec = CEC(g, lfs.path, cmap=value3.get(), app_pars=lfs.app_pars)
    g.after(2000, lambda: print('Transmission Mode --> Reciprocal Mode'))
    t_cec.view.change_mode()
    t_cec.view.text_e.set('21.200')
    t_cec.view.set_slim()
    g.after(1000, lambda: print('Symmetrical Extend'))
    t_cec.view.symmetry()
    g.after(1000, lambda: print('6-Fold Rotation'))
    t_cec.view.symmetry_(6)
    t_cec.view.grab_set()
    t_cec.view.focus_set()
    tg = wait(g, app_pars)
    tg.label_wait.pack_forget()
    b = tk.Button(tg, text='OK', command=tg.done)
    b.pack()
    tg.bind('<Return>', lambda e: tg.done())
    tg.text('Press Export to generate data cube.(May take few minutes)')
    
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
    global gfe,emf
    if 'gfe' in globals():
        gfe.destroy()
        clear(gfe)
    if emf=='KE':
        emf='BE'
        b_emode.config(text='B.E.')
        gfe = G_emode()
    else:
        emf='KE'
        b_emode.config(text='K.E.')
        gfe = G_emode()

@pool_protect
def gui_exp_origin(*e):
    global gori,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,l1,b3,origin_func
    if data is None:
        st.put('No data loaded!')
        messagebox.showwarning("Warning","No data loaded!")
        return
    def set_Checkbutton(frame, text, variable, row):
        Checkbutton = tk.Checkbutton(frame,text=text,variable=variable,font=('Arial', size(18), "bold"),bg='white')
        Checkbutton.grid(row=row,column=0,sticky='w')
        return Checkbutton
    limg.config(image=img[np.random.randint(len(img))])
    if 'gori' in globals():
        gori.destroy()
    gori=RestrictedToplevel(g,bg='white')
    v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11=tk.IntVar(),tk.IntVar(),tk.IntVar(),tk.IntVar(),tk.IntVar(),tk.IntVar(),tk.IntVar(),tk.IntVar(),tk.IntVar(),tk.IntVar(),tk.IntVar()
    var=[v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11]
    text=['E-Phi (Raw Data)','E-k (Processed Data)','MDC Fit Position','MDC Fit FWHM','EDC Fit Position','EDC Fit FWHM','Self Energy Real Part','Self Energy Imaginary Part','Data plot with pos','Data plot with pos & bare band','Second Derivative']
    origin_func = origin()
    gori.title('Export to Origin')
    l1=tk.Label(gori,text=f"{dpath.removesuffix('.h5').removesuffix('.json').removesuffix('.txt')}.{origin_func.suffix}",font=('Arial', size(10), "bold"),bg='white',wraplength=600)
    l1.grid(row=0,column=0)
    b1=tk.Button(gori,text='Patch Origin',command=origin_func.patch_origin, width=15, height=1, font=('Arial', size(18), "bold"), bg='white', bd=5)
    # b1.grid(row=1,column=0)
    fr=tk.Frame(gori,bg='white')
    fr.grid(row=2,column=0)
    origin_func.pr_exp_origin()
    cl=[]
    for i in range(len(var)):
        cl.append(set_Checkbutton(fr, text[i], var[i], i))
    if npzf:cl[0].config(text='E-k (Sliced Data)')
    fr_exp=tk.Frame(fr,bg='white')
    fr_exp.grid(row=11,column=0)
    b2=tk.Button(fr_exp,text='Export',command=origin_func.exp_origin, width=15, height=1, font=('Arial', size(18), "bold"), bg='white', bd=5)
    b2.pack(side='left')
    b3=tk.Button(fr_exp,text=f'(.{origin_func.suffix})', width=15, height=1, font=('Arial', size(18), "bold"), bg='white', bd=5)
    b3.config(command=lambda: origin_func.ch_suffix(dpath, l1,  b3))
    b3.pack(side='right')
    for i in range(len(cl)):
        if i in origin_func.no:
            cl[i].deselect()
            cl[i].config(state='disabled')
        else:
            cl[i].config(state='normal')
            cl[i].select()
    if npzf:
        cl[1].deselect()
        cl[1].config(state='disabled')
    gori.bind('<Return>', origin_func.exp_origin)
    set_center(g, gori, 0, 0)
    gori.focus_set()
    gori.limit_bind()

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

@pool_protect
def view_3d(*e):
    DataViewer_PyQt5()

@pool_protect
def DataViewer_PyQt5():
    def j():
        if os.name == 'nt':
            os.system(f'{sys.executable} -W ignore::SyntaxWarning -W ignore::UserWarning "{os.path.join(cdir, '.MDC_cut', 'tool', 'DataViewer.py')}"')
        elif os.name == 'posix':
            os.system(f'''{sys.executable} -W ignore::SyntaxWarning -W ignore::UserWarning "{os.path.join(cdir, '.MDC_cut', 'tool', 'DataViewer.py')}" &''')
    threading.Thread(target=j,daemon=True).start()

@pool_protect
def show_version():
    global ax
    ax = fig.subplots()
    tim = np.asarray(Image.open(tdata), dtype=np.uint8)
    ax.imshow(tim, aspect='equal', alpha=0.4)
    if os.name == 'nt':
        ver_size = 40
    elif os.name == 'posix':
        ver_size = 30
    fontdict = {
    'fontsize': size(ver_size),
    'fontweight': 'bold',
    'fontname': 'Arial'
    }
    ax.text(tim.shape[0]/2, tim.shape[1]/2, f"Version: {__version__}\n\nRelease Date: {__release_date__}", fontdict=fontdict, color='black', ha='center', va='center')
    ax.axis('off')
    out.draw()

@pool_protect
def show_info():
    info_window = RestrictedToplevel(g)
    info_window.title("Information")
    tk.Label(info_window, text="Graph copied to clipboard", font=("Arial", size(30), "bold"),fg='red').pack(pady=5)
    label = tk.Label(info_window, text="window closed in 3 second", font=("Arial", size(20)))
    label.pack(pady=5)
    set_center(g, info_window, 0, 0)
    info_window.update()
    for i in range(3, 0, -1):
        info_window.after(1000, label.config(text=f"window closed in {i-1} second"))
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
    
    if os.name == 'nt':
        image.convert("RGB").save(output, "BMP")
        data = output.getvalue()[14:]
        output.close()
        send_to_clipboard(win32clipboard.CF_DIB, data)
    elif os.name == 'posix':
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp.write(buf.getvalue())
            temp_path = tmp.name

        try:
            # 使用 AppleScript 將圖片複製到剪貼簿
            applescript = f'''
            set imagePath to POSIX file "{temp_path}"
            set the clipboard to (read imagePath as «class PNGf»)
            '''
            
            process = subprocess.Popen(['osascript', '-'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, error = process.communicate(input=applescript.encode('utf-8'))
            
            if error:
                print(f"Error copying to clipboard: {error.decode()}")
            else:
                print("Copied to clipboard successfully.")
        finally:
            # 清理臨時檔案
            os.unlink(temp_path)
    
@pool_protect
def send_to_clipboard(clip_type, data):
    win32clipboard.OpenClipboard()
    win32clipboard.EmptyClipboard()
    win32clipboard.SetClipboardData(clip_type, data)
    win32clipboard.CloseClipboard()

@pool_protect
def trans_plot(*e):
    if data is None:
        st.put('No data loaded!')
        messagebox.showwarning("Warning","No data loaded!")
        return
    global gtp
    if 'gtp' in globals():
        gtp.destroy()
    gtp=RestrictedToplevel(g, bg='white')
    gtp.title('Spectrogram')
    b_raw = tk.Button(gtp, text='Raw', command=raw_plot, width=15, height=2, font=('Arial', size(14), "bold"), bg='white', fg='black', bd=5)
    b_raw.grid(row=0, column=0)
    b_smooth = tk.Button(gtp, text='Smooth', command=smooth_plot, width=15, height=2, font=('Arial', size(14), "bold"), bg='white', fg='black', bd=5)
    b_smooth.grid(row=0, column=1)
    b_fd = tk.Button(gtp, text='First Derivative', command=fd_plot, width=15, height=2, font=('Arial', size(14), "bold"), bg='white', fg='black', bd=5)
    b_fd.grid(row=0, column=2)
    set_center(g, gtp, 0, 0)
    gtp.bind('<Return>', raw_plot)
    gtp.limit_bind()
    g.focus_set()
    gtp.focus_set()

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
    if len(name) > 30:
        l_name.config(font=('Arial', size(f10), "bold"), width=lfs.max_name_len)
    elif len(name) > 20:
        l_name.config(font=('Arial', size(f12), "bold"), width=len(namevar.get()))
    else:
        l_name.config(font=('Arial', size(f14), "bold"), width=len(namevar.get()))
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
def qt_app(path: list[str]):
    def job():
        if os.name == 'nt':
            subprocess.call([f'{sys.executable}', '-W', 'ignore::SyntaxWarning', '-W', 'ignore::UserWarning', f'{os.path.join(cdir, '.MDC_cut', 'tool', 'RawDataViewer.py')}', '-f'] + list(path))
        elif os.name == 'posix':
            tpath = []
            for i in path:
                tpath.append(str(i))
            spath = str(tpath).removeprefix('[').removesuffix(']').replace(', ', ' ')
            os.system(f'''{sys.executable} -W ignore::SyntaxWarning -W ignore::UserWarning "{os.path.join(cdir, '.MDC_cut', 'tool', 'RawDataViewer.py')}" -f {spath} &''')
    threading.Thread(target=job, daemon=True).start()
    if os.name == 'nt' and hwnd:
        windll.user32.ShowWindow(hwnd, 9)
        windll.user32.SetForegroundWindow(hwnd)
    elif os.name == 'posix':
        subprocess.run(['open', '-a', 'Terminal'])
    print('\033[36m\nTransfering Data...\nPlease wait...\033[0m')

@pool_protect
def tools(*args):
    def raw_data_viewer(*args):
        qt_app(lfs.path)
        toolg.destroy()
        
    def spec(*args):
        s = spectrogram(path=lfs.path, name='internal', app_pars=lfs.app_pars)
        s.plot(g, value3.get())
        toolg.destroy()
        
    def exp_casa():
        explfs = lfs_exp_casa(lfs)
        explfs.export_casa()
        toolg.destroy()
        
    def kplane():
        CEC(g, lfs.path, cmap=value3.get(), app_pars=lfs.app_pars)
        toolg.destroy()
    
    global toolg
    if 'toolg' in globals():
        toolg.destroy()
    toolg = RestrictedToplevel(g)
    toolg.title('Batch Master')
    b_raw_viewer = tk.Button(toolg, text='Raw Data Viewer', command=raw_data_viewer, width=15, height=2, font=('Arial', size(14), "bold"), bg='white', fg='black', bd=5)
    b_raw_viewer.grid(row=0, column=0)
    b_spec = tk.Button(toolg, text='Spectrogram', command=spec, width=15, height=2, font=('Arial', size(14), "bold"), bg='white', fg='black', bd=5)
    b_spec.grid(row=0, column=1)
    try:
        flag = False
        t_, t__ = lfs.r1.copy(), lfs.r2.copy()
        flag = True
        t_, t__ = None, None
    except:
        pass
    if lfs.sort != 'no' and flag:
        b_kplane = tk.Button(toolg, text='k-Plane', command=kplane, width=15, height=2, font=('Arial', size(14), "bold"), bg='white', fg='black', bd=5)
        b_kplane.grid(row=0, column=2)
    b_exp_casa = tk.Button(toolg, text='Export to Casa', command=exp_casa, width=15, height=2, font=('Arial', size(14), "bold"), bg='white', fg='black', bd=5)
    b_exp_casa.grid(row=0, column=3)
    toolg.bind('<Return>', spec)
    set_center(g, toolg, 0, 0)
    toolg.limit_bind()
    g.focus_set()
    toolg.focus_set()

@pool_protect
def def_cmap():
    global CE
    if 'CE' in globals():
        CE.destroy()
        clear(CE)
    CE = ColormapEditor()

@pool_protect
def o_load(drop=False, files=''):
    if not drop:
        files = fd.askopenfilenames(title="Select Raw Data", filetypes=(
        ("HDF5 files", "*.h5"), ("NPZ files", "*.npz"), ("JSON files", "*.json"), ("TXT files", "*.txt")))
    st.put('Loading...')
    tg = wait(g, app_pars)
    tg.text('Loading data...')
    files = tkDnD.load_raw(files)
    main_loader(files)
    tg.done()

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
    def job():
        src = '.MDC_cut'
        if os.name == 'nt':
            subprocess.call([f'{sys.executable}', '-W', 'ignore::SyntaxWarning', '-W', 'ignore::UserWarning', f'{os.path.join(cdir, src, "tool", "MDC_Fitter.py")}', '-f', data.attrs['Path']])
        elif os.name == 'posix':
            os.system(f'''{sys.executable} -W ignore::SyntaxWarning -W ignore::UserWarning "{os.path.join(cdir, src, "tool", "MDC_Fitter.py")}" -f "{data.attrs['Path']}" &''')
    threading.Thread(target=job, daemon=True).start()
    if os.name == 'nt' and hwnd:
        windll.user32.ShowWindow(hwnd, 9)
        windll.user32.SetForegroundWindow(hwnd)
    elif os.name == 'posix':
        subprocess.run(['open', '-a', 'Terminal'])
    print('\033[36m\nTransfering Data...\nPlease wait...\033[0m')

@pool_protect
def cefit(*e):
    messagebox.showwarning("Warning", "Temporarily Disabled")
    return
    if data is None:
        st.put('No data loaded!')
        messagebox.showwarning("Warning","No data loaded!")
        return
    import tool.EDC_Fitter
    from tool.EDC_Fitter import egg as egg
    if egg is None:
        edc_pars = EDC_param(ScaleFactor=ScaleFactor, sc_y=sc_y, g=g, scale=scale, npzf=npzf, vfe=vfe, emf=emf, st=st, dpath=dpath, name=name, k_offset=k_offset, value3=value3, ev=ev, phi=phi, data=data, base=base, fpr=fpr, semin=semin, semax=semax, sefp=sefp, sefi=sefi, seaa1=seaa1, seaa2=seaa2, edet=edet)
        threading.Thread(target=tool.EDC_Fitter.fite, args=(edc_pars,)).start()
        clear(edc_pars)
    elif isinstance(egg, tk.Toplevel):
        egg.lift()
    elif egg == True:
        importlib.reload(tool.EDC_Fitter)
        edc_pars = EDC_param(ScaleFactor=ScaleFactor, sc_y=sc_y, g=g, scale=scale, npzf=npzf, vfe=vfe, emf=emf, st=st, dpath=dpath, name=name, k_offset=k_offset, value3=value3, ev=ev, phi=phi, data=data, base=base, fpr=fpr, semin=semin, semax=semax, sefp=sefp, sefi=sefi, seaa1=seaa1, seaa2=seaa2, edet=edet)
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
@pool_protect
def clmfit():
    global rpos, pos, fwhm, fev, ophi
    rpos = []
    pos = []
    fwhm = []
    fev = []
    ophi = []

@pool_protect
def clefit():
    global fphi, epos, ffphi, efwhm, fk
    fphi = []
    epos = []
    ffphi = []
    efwhm = []
    fk = []

@pool_protect
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

@pool_protect
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


@pool_protect
def o_fbb_offset(*e):
    global bb_offset
    if '' == bb_offset.get():
        bb_offset.set('0')
        bboffset.select_range(0, 1)
    os.chdir(cdir)
    np.savez(os.path.join(cdir, '.MDC_cut', 'bb.npz'), path=bpath, be=be, k=k, bbo=float(bb_offset.get()), bbk=float(bbk_offset.get()))


@pool_protect
def fbb_offset(*e):
    threading.Thread(target=o_fbb_offset, daemon=True).start()


@pool_protect
def o_fbbk_offset(*e):
    global bbk_offset
    if '' == bbk_offset.get():
        bbk_offset.set('1')
        bbkoffset.select_range(0, 1)
    os.chdir(cdir)
    np.savez(os.path.join(cdir, '.MDC_cut', 'bb.npz'), path=bpath, be=be, k=k, bbo=float(bb_offset.get()), bbk=float(bbk_offset.get()))

@pool_protect
def fbbk_offset(*e):
    threading.Thread(target=o_fbbk_offset, daemon=True).start()

@pool_protect
def o_fbase(*e):
    global base
    if '' == base.get():
        base.set('0')
        in_fit.select_range(0, 1)

@pool_protect
def fbase(*e):
    threading.Thread(target=o_fbase, daemon=True).start()

@pool_protect
def o_flowlim(*e):
    global lowlim
    if '' == lowlim.get():
        lowlim.set('0')
        in_lowlim.select_range(0, 1)

@pool_protect
def flowlim(*e):
    threading.Thread(target=o_flowlim, daemon=True).start()

@pool_protect
def o_reload(*e):
    global k_offset, fev, ophi, rpos, pos, ffphi, fwhm, fk, st, kmin, kmax, smresult, smcst, smaa1, smaa2, smfp, smfi, skmin, skmax, epos, efwhm, ffphi, fk, emin, emax, seaa1, seaa2, sefp, sefi, semin, semax
    try:
        if '' == k_offset.get():
            k_offset.set('0')
            koffset.select_range(0, 1)
    except RuntimeError:
        return
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
    os.chdir(cdir)
    try:
        np.savez(os.path.join(cdir, '.MDC_cut', 'mfit.npz'), ko=k_offset.get(), fev=fev, rpos=rpos, ophi=ophi, fwhm=fwhm, pos=pos, kmin=kmin,
                 kmax=kmax, skmin=skmin, skmax=skmax, smaa1=smaa1, smaa2=smaa2, smfp=smfp, smfi=smfi)
        np.savez(os.path.join(cdir, '.MDC_cut', 'mfit.npz'), ko=k_offset.get(), fev=fev, rpos=rpos, ophi=ophi, fwhm=fwhm, pos=pos, kmin=kmin,
                 kmax=kmax, skmin=skmin, skmax=skmax, smaa1=smaa1, smaa2=smaa2, smfp=smfp, smfi=smfi, smresult=smresult, smcst=smcst, mdet=mdet)
    except:
        try:
            ffphi = np.float64(k_offset.get())+fphi
            fk = (2*m*epos*1.602176634*10**-19)**0.5 * \
                np.sin(ffphi/180*np.pi)*10**-10/(h/2/np.pi)
            np.savez(os.path.join(cdir, '.MDC_cut', 'efit.npz'), ko=k_offset.get(), fphi=fphi, epos=epos, ffphi=ffphi, efwhm=efwhm, fk=fk,
                 emin=emin, emax=emax, semin=semin, semax=semax, seaa1=seaa1, seaa2=seaa2, sefp=sefp, sefi=sefi, edet=edet)
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
    file = fd.askopenfilename(title="Select TXT file", filetypes=(("TXT files", "*.txt"),))
    global be, k, limg, img, st, bpath
    if len(file) > 0:
        bpath = file
        print('Loading...')
        st.put('Loading...')
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

@pool_protect
def o_plot1():
    PlotsUtil().o_plot1()

@pool_protect
def o_plot2():
    PlotsUtil().o_plot2()

@pool_protect
def o_plot3():
    PlotsUtil().o_plot3()

props = dict(facecolor='green', alpha=0.3)
warn_str = ''
@pool_protect
def exp(*e):
    if data is None:
        st.put('No data loaded!')
        messagebox.showwarning("Warning", "No data loaded!")
        return
    if warn_str != '':
        messagebox.showwarning("Warning", warn_str)
        return
    ExpUtil().exp()

@pool_protect
def angcut(*e):
    threading.Thread(target=o_angcut, daemon=True).start()

@pool_protect
def ecut(*e):
    threading.Thread(target=o_ecut, daemon=True).start()

@pool_protect
def loadmfit(*e):
    threading.Thread(target=o_loadmfit, daemon=True).start()

@pool_protect
def loadefit(*e):
    global rdd, efi_x, data, fpr, lfs, npzf
    global fphi, epos, ffphi, efwhm, fk, emin, emax, semin, semax, seaa1, seaa2, sefp, sefi, edet
    el = eloader(st, data, ev, phi, rdd, cdir, lowlim.get())
    el.loadparam(k_offset.get(), base.get(), npzf, fpr)
    file = fd.askopenfilename(title="Select EDC Fitted file", filetypes=(("NPZ files", "*.npz"), ("VMS files", "*.vms"),))
    t3 = threading.Thread(target=el.loadefit, args=(file,), daemon=True)
    t3.start()
    t3.join()
    rdd, efi_x, data, fpr = el.rdd, el.efi_x, el.data, el.fpr
    fphi, epos, ffphi, efwhm, fk, emin, emax, semin, semax, seaa1, seaa2, sefp, sefi, edet = el.fphi, el.epos, el.ffphi, el.efwhm, el.fk, el.emin, el.emax, el.semin, el.semax, el.seaa1, el.seaa2, el.sefp, el.sefi, el.edet
    limg.config(image=img[np.random.randint(len(img))])
    if el.fload:
        try:
            lfs = loadfiles([rdd], init=False, name='internal', cmap=value3.get(), app_pars=app_pars)
            data = lfs.get(0)
            pr_load(data)
            npzf = lfs.f_npz[0]
        except FileNotFoundError:
            print(f'{rdd} File path not found, skip loading raw data.')
        except Exception as e:
            print(f'Error loading raw data from {rdd}: {e}')
    clear(el)

@pool_protect
def reload(*e):
    threading.Thread(target=o_reload, daemon=True).start()

@pool_protect
def expte():
    threading.Thread(target=o_expte, daemon=True).start()

@pool_protect
def exptm():
    threading.Thread(target=o_exptm, daemon=True).start()

@pool_protect
def bareband(*e):
    threading.Thread(target=o_bareband, daemon=True).start()

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
    suggest()
    global gg
    if 'gg' in globals():
        gg.destroy()
    if 'MDC Curves' in value.get():
        gg = plot1_window_MDC_curves(d, l, p)
    elif value.get() == 'Second Derivative':
        gg = plot1_window_Second_Derivative(im_kernel)
    else:
        threading.Thread(target=o_plot1, daemon=True).start()

@pool_protect
def plot2(*e):
    suggest()
    threading.Thread(target=o_plot2, daemon=True).start()

@pool_protect
def plot3(*e):
    global gg
    suggest()
    if 'gg' in globals():
        gg.destroy()
    if value2.get() == 'Data Plot with Pos' or value2.get() == 'Data Plot with Pos and Bare Band':
        gg = plot3_window(fev, fk)
    else:
        threading.Thread(target=o_plot3, daemon=True).start()

@pool_protect
def load(drop=False, files='', *args):
    if 'KeyPress' in str(drop):
        drop = False
    threading.Thread(target=o_load, args=(drop, files), daemon=True).start()

def fitgl():
    pass
    # threading.Thread(target=o_fitgl, daemon=True).start()

@pool_protect
def tstate():
    try:
        while True:
            state.config(text=str(st.get()))
    except:
        pass

@pool_protect
def get_yerr():
    global pos_err, fwhm_err
    pos_err, fwhm_err = [], []
    if smresult is not None:
        for i, v in enumerate(smresult):
            if i in smfi:
                try:
                    res = v[0].split('+/- ')[1].split(' (')[0]
                    pos_err.append(res)
                except:
                    pass
                try:
                    res = v[4].split('+/- ')[1].split(' (')[0]
                    fwhm_err.append(res)
                except:
                    pass            
        pos_err = np.array(pos_err, dtype=float)
        fwhm_err = np.array(fwhm_err, dtype=float)
        
@pool_protect
def lm2p():
    lmgg.destroy()
    global rdd
    ml = mloader(st, data, ev, phi, rdd, cdir, lowlim.get())
    file = fd.askopenfilename(title="Select MDC Fitted file", filetypes=(("VMS files", "*.vms"),))
    t = threading.Thread(target=ml.loadmfit_2p, args=(file,), daemon=True)
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
    t = threading.Thread(target=ml.loadmfit_re, args=(file,), daemon=True)
    t.start()
    t.join()
    rdd = ml.rdd
    clear(ml)

@pool_protect
def lm():
    lmgg.destroy()
    global rdd, mfi_x, data, fpr, lfs, npzf
    global fev, rpos, ophi, fwhm, pos, kmin, kmax, skmin, skmax, smaa1, smaa2, smfp, smfi, smresult, smcst, mdet
    ml = mloader(st, data, ev, phi, rdd, cdir, lowlim.get())
    ml.loadparam(k_offset.get(), base.get(), npzf, fpr)
    file = fd.askopenfilename(title="Select MDC Fitted file", filetypes=(("NPZ files", "*.npz"), ("VMS files", "*.vms"),))
    t = threading.Thread(target=ml.loadmfit_, args=(file,), daemon=True)
    t.start()
    t.join()
    rdd, mfi_x, data, fpr = ml.rdd, ml.mfi_x, ml.data, ml.fpr
    fev, rpos, ophi, fwhm, pos, kmin, kmax, skmin, skmax, smaa1, smaa2, smfp, smfi, smresult, smcst, mdet = ml.fev, ml.rpos, ml.ophi, ml.fwhm, ml.pos, ml.kmin, ml.kmax, ml.skmin, ml.skmax, ml.smaa1, ml.smaa2, ml.smfp, ml.smfi, ml.smresult, ml.smcst, ml.mdet
    get_yerr()
    limg.config(image=img[np.random.randint(len(img))])
    if ml.fload:
        try:
            lfs = loadfiles([rdd], init=False, name='internal', cmap=value3.get(), app_pars=app_pars)
            data = lfs.get(0)
            pr_load(data)
            npzf = lfs.f_npz[0]
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
        "Arial", size(12), "bold"), fg='red', width=30, height=2, bd=2)
    b1.pack()
    b2 = tk.Button(lmgg, command=lmre, text='reverse vms axis', font=(
        "Arial", size(12), "bold"), fg='red', width=30, height=2, bd=2)
    b2.pack()
    b3 = tk.Button(lmgg, command=lm, text='load MDC fitted File', font=(
        "Arial", size(12), "bold"), fg='red', width=30, height=2, bd=2)
    b3.pack()
    lmgg.update()
    w=lmgg.winfo_reqwidth()
    h=lmgg.winfo_reqheight()
    lmgg.geometry(f'{w}x{h}')
    set_center(g, lmgg, 0, 0)
    lmgg.focus_set()
    lmgg.limit_bind()

@pool_protect
def dl_sw():
    global dl, b_sw
    s=['dot','line','dot-line']
    dl=(dl+1)%3
    b_sw.config(text=s[dl])
    threading.Thread(target=o_plot3, daemon=True).start()

@pool_protect
def plot1_set(opt):
    global value, value1, value2
    value.set(opt)
    value1.set('---Plot2---')
    value2.set('---Plot3---')
    
@pool_protect
def plot2_set(opt):
    global value, value1, value2
    value.set('---Plot1---')
    value1.set(opt)
    value2.set('---Plot3---')
    
@pool_protect
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
        restart()
        quit()
    else:
        os.remove('open_check_MDC_cut.txt')
        
    hwnd = find_window()
    path = os.path.join(cdir, '.MDC_cut', 'hwnd')
    with open(path, 'w') as f:
        f.write(f'{hwnd}')  #for DataViewer Qt GUI
        f.close()
    if os.name == 'nt':
        ScaleFactor = windll.shcore.GetScaleFactorForDevice(0)
        osf = windll.shcore.GetScaleFactorForDevice(0)
        # print('ScaleFactor:',ScaleFactor)
        t_sc_w, t_sc_h = windll.user32.GetSystemMetrics(0), windll.user32.GetSystemMetrics(1)   # Screen width and height
        t_sc_h-=int(40*ScaleFactor/100)
    elif os.name == 'posix':
        temp_root = tk.Tk()
        temp_root.withdraw()
        t_sc_w = temp_root.winfo_screenwidth()
        t_sc_h = temp_root.winfo_screenheight()
        dpi = temp_root.winfo_fpixels('1i')
        temp_root.destroy()
        
        ScaleFactor = int((72.054 / dpi) * 100)
        osf = ScaleFactor   # ~72
    if bar_pos == 'top':    #taskbar on top
        sc_y = int(40*ScaleFactor/100)
    else:
        sc_y = 0
    # w 1920 1374 (96 dpi)
    # h 1080 748 (96 dpi)
    g = TkinterDnD.Tk()
    # g.withdraw()
    # # 最小化視窗
    # g.iconify()
    # # 恢復視窗
    # g.deiconify()
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
    if os.name == 'nt':
        ScaleFactor /= prfactor*(ScaleFactor/100*1880/96*odpi/t_sc_w) if 1880/t_sc_w >= (950)/t_sc_h else prfactor*(ScaleFactor/100*(950)/96*odpi/t_sc_h)
        g.tk.call('tk', 'scaling', ScaleFactor/100)
    elif os.name == 'posix':
        ScaleFactor /= prfactor*(ScaleFactor/100*1512/72.054*odpi/t_sc_w) if 1512/t_sc_w >= (851)/t_sc_h else prfactor*(ScaleFactor/100*(851)/72.054*odpi/t_sc_h)
        g.tk.call('tk', 'scaling', ScaleFactor/100)
    dpi=g.winfo_fpixels('1i')
    # print('dpi:',dpi)
    if os.name == 'nt':
        windll.shcore.SetProcessDpiAwareness(1)
    scale = odpi / dpi
    base_font_size = 16
    scaled_font_size = int(base_font_size * scale)-1
    
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
    g.option_add('*Foreground', 'black')
    g.option_add('*Background', 'white')
    icon_manager = MenuIconManager(scale=scale, ScaleFactor=ScaleFactor, odpi=odpi, dpi=dpi)
    
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
    filemenu.add_command(label="Exit", command=g_close, accelerator="Ctrl+Q")
    
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
    helpmenu.add_command(label="Sample Data Demo", command=sample_data)
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
    
    os.system('cls' if os.name == 'nt' else 'clear')
    try:
        with np.load(os.path.join('.MDC_cut', 'rd.npz'), 'rb') as ff:
            path = str(ff['path'])
            lpath = ff['lpath']
            print('\n\033[90mRaw Data preloaded:\033[0m\n\n')
            lfs = loadfiles(lpath, init=True, name='internal', cmap=value3.get(), app_pars=app_pars)
            print(lfs)
            if lfs.cec_pars:
                lfs = call_cec(g, lfs)
            data = lfs.get(0)
            dvalue = list(data.attrs.values())
            rdd = path  # old version data path
            dpath = path    # new version data path
            name, e_photon, description = dvalue[0], float(dvalue[3].split()[0]), dvalue[13]
    except:
        data, lfs, name, description, e_photon, rdd, dpath = None, None, None, None, None, '', ''
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

    fpr = 0
    try:
        with np.load(os.path.join('.MDC_cut', 'efit.npz'), 'rb') as f:
            ko = str(f['ko'])
            fphi, epos, ffphi, efwhm, fk = f['fphi'], f['epos'], f['ffphi'], f['efwhm'], f['fk']
            emin, emax, semin, semax = f['emin'], f['emax'], f['semin'], f['semax']
            seaa1, seaa2, sefp, sefi = f['seaa1'], f['seaa2'], f['sefp'], f['sefi']
            try:
                edet = f['edet']
            except:
                edet = -1
            print('\033[90mEDC Fitted Data preloaded (Casa)\033[0m')
        fpr = 1
    except:
        ko = '0'
        fphi, epos, ffphi, efwhm, fk = [], [], [], [], []
        emin, emax, semin, semax = [], [], [], []
        seaa1, seaa2, sefp, sefi = [], [], [], []
        # seresult, secst = [], []
        edet = -1
        print('\033[90mNo EDC fitted data preloaded (Casa)\033[0m')

    try:
        with np.load(os.path.join('.MDC_cut', 'mfit.npz'), 'rb') as f:
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
                get_yerr()
                print('\033[90mMDC Fitted Data preloaded (lmfit)\033[0m')
            except:
                print('\033[90mMDC Fitted Data preloaded (Casa)\033[0m')
        fpr = 1
    except:
        ko = '0'
        fev, rpos, ophi, fwhm, pos = [], [], [], [], []
        kmin, kmax, skmin, skmax = [], [], [], []
        smaa1, smaa2, smfp, smfi = [], [], [], []
        smresult, smcst = [], []
        mdet = -1
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

    icon = IconManager()
    g.iconphoto(True, tk.PhotoImage(data=b64decode(icon.icon)))

    f14 = 14 if os.name == 'nt' else 11
    f12 = 12 if os.name == 'nt' else 10
    f10 = 10 if os.name == 'nt' else 8
    
    fr_main = tk.Frame(g, bg="white")
    fr_main.pack(side=tk.TOP, fill='both', expand=True)
    
    fr = tk.Frame(fr_main, bg='white')
    fr.grid(row=0, column=0, sticky='n', pady=10)
    fr_info = tk.Frame(fr,bg='white')
    fr_info.pack(side=tk.TOP)
    w_info = 25
    fr_tool = tk.Frame(fr_info,bg='white',width=w_info)
    fr_tool.pack(fill='x')
    l_path = tk.Text(fr_info, wrap='word', font=("Arial", size(f12), "bold"), bg="white", fg="black", state='disabled',height=3,width=w_info)
    l_path.pack(fill='x')
    # info = tk.Label(fr_main,text='                                   \n\n\n\n\n\n\n\n\n\n\n\n\n', font=("Arial", size(14), "bold"), bg="white", fg="black",padx = 30,pady=30)
    xscroll = tk.Scrollbar(fr_info, orient='horizontal')
    xscroll.pack(side='bottom', fill='x')
    yscroll = tk.Scrollbar(fr_info, orient='vertical')
    yscroll.pack(side='right', fill='y')
    info = tk.Text(fr_info, wrap='none', font=("Arial", size(f14), "bold"), bg="white", fg="black", state='disabled',
                height=10, width=w_info, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
    info.pack(anchor='w')
    xscroll.config(command=info.xview)
    yscroll.config(command=info.yview)
    fr_mod = tk.Frame(fr,bg='white')
    fr_mod.pack(side=tk.TOP)
    b_name = tk.Button(fr_mod, text='Modify\nName', font=('Arial', size(f12), 'bold'), height=2, command=cname)
    b_name.grid(row=0,column=0)
    b_excitation = tk.Button(fr_mod, text='Modify\nExcitation Energy', font=('Arial', size(f12), 'bold'), height=2, command=cexcitation)
    b_excitation.grid(row=0,column=1)
    b_desc = tk.Button(fr_mod, text='Modify\nDescription', font=('Arial', size(f12), 'bold'), height=2, command=desc)
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
    
    bchcmp = tk.Button(cmbf, text='Change\nColormap', font=(
        "Arial", size(f12), "bold"), height=2, command=Chcmp, border=2)
    bchcmp.pack(side='left', padx=2, pady=2)
    bdefcmp = tk.Button(cmbf, text='User Defined\nColormap', font=(
        "Arial", size(f12), "bold"), height=2, command=def_cmap, border=2)
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
        "Arial", size(f12)), bg="white", height='1')
    c1.grid(row=0, column=0)
    c2 = tk.Label(cmlf, text='All:', font=("Arial", size(f12)), bg="white", height='1')
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
        "Arial", size(f14), "bold"), bg="white", fg='red')
    m1.grid(row=0, column=0)
    m2 = tk.Label(frraw, text='Fit', font=(
        "Arial", size(f14), "bold"), bg="white", fg='blue')
    m2.grid(row=1, column=0)
    m3 = tk.Label(frraw, text='Transform', font=(
        "Arial", size(f14), "bold"), bg="white", fg="blue")
    m3.grid(row=2, column=0)

    
    fr_state = tk.Frame(fr_main, bg='white')
    fr_state.grid(row=0, column=2, sticky='n')

    st = queue.Queue(maxsize=0)
    state = tk.Label(fr_state, text=f"Version: {__version__}", font=(
        "Arial", size(f14), "bold"), bg="white", fg="black", wraplength=250, justify='center')
    state.grid(row=0, column=0, pady=20)

    # Icon = [icon.icon1, icon.icon2, icon.icon3, icon.icon4, icon.icon5, icon.icon6, icon.icon7, icon.icon8, icon.icon9, icon.icon10, icon.icon11, icon.icon12, icon.icon13, icon.icon14, icon.icon15, icon.icon16, icon.icon17, icon.icon18, icon.icon19, icon.icon20]
    Icon = [icon.icon0]
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
    # limg.grid(row=1, column=0)
    b_suggest = tk.Button(fr_state, text='Suggestion', font=(
        "Arial", size(14), "bold"), bg="white", height='1', bd=5)
    # b_suggest.grid(row=1, column=0)

    
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
    w_15 = 15 if os.name == 'nt' else 12
    cmax = tk.Frame(clim, bg='white', width=w_15, bd=5)
    cmax.grid(row=0, column=1)
    cmin = tk.Frame(clim, bg='white', width=w_15, bd=5)
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
        "Arial", size(f12), "bold"), bg="white", fg="black")
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
        "Arial", size(f10), "bold"), bg="white", fg="black", height=1)
    l_lowlim.grid(row=0, column=2)
    lowlim = tk.StringVar()
    lowlim.set('0')
    lowlim.trace_add('write', flowlim)
    in_lowlim = tk.Entry(cut, font=("Arial", size(f10), "bold"),
                        width=7, textvariable=lowlim, bd=5)
    in_lowlim.grid(row=0, column=3)


    k_offset = tk.StringVar()
    try:
        k_offset.set(ko)
    except:
        k_offset.set('0')
    k_offset.trace_add('write', reload)
    koffset = tk.Entry(step, font=("Arial", size(f12), "bold"),
                    width=f12, textvariable=k_offset, bd=2)
    koffset.grid(row=2, column=1)
    
    bb_offset = tk.StringVar()
    try:
        bb_offset.set(bbo)
    except:
        bb_offset.set('0')
    bb_offset.trace_add('write', fbb_offset)
    bboffset = tk.Entry(step, font=("Arial", size(f12), "bold"),
                        width=f12, textvariable=bb_offset, bd=2)
    bboffset.grid(row=3, column=1)
    bbk_offset = tk.StringVar()
    try:
        bbk_offset.set(bbk)
    except:
        bbk_offset.set('1')
    bbk_offset.trace_add('write', fbbk_offset)
    bbkoffset = tk.Entry(step, font=("Arial", size(f12), "bold"),
                        width=f12, textvariable=bbk_offset, bd=2)
    bbkoffset.grid(row=4, column=1)
    l6 = tk.Label(step, text='Bare band E offset (meV)', font=(
        "Arial", size(f12), "bold"), bg="white", fg="black", height=1)
    l6.grid(row=3, column=0)
    l7 = tk.Label(step, text='Bare band k ratio', font=(
        "Arial", size(f12), "bold"), bg="white", fg="black", height=1)
    l7.grid(row=4, column=0)

    figfr = tk.Frame(fr_main, bg='white')
    figfr.grid(row=0, column=1, sticky='nsew')
    if os.name == 'nt':
        figy = 8.5 if osf<=100 else 8.25 if osf<=150 else 8
        figx = 11.5 if osf<=100 else 11.25 if osf<=150 else 11
        label_size = 16
    elif os.name == 'posix':
        # print('scale:', scale)
        figy = 8 if scale<=1 else 6.3
        figx = 9.5 if scale<=1 else 6.8
        label_size = 16
    fig = Figure(figsize=(figx*scale, figy*scale), layout='constrained')
    out = FigureCanvasTkAgg(fig, master=figfr)
    out.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    ao = None
    tb0, tb0_, tb1, tb1_, tb2 = None, None, None, None, None
    pflag = 0
    xl, yl, rcx, rcy = None, None, None, None
    rx, ry, ix, iy = None, None, None, None
    show_version()

    xydata = tk.Frame(figfr, bg='white')
    xydata.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    xdata = tk.Label(xydata, text='xdata:', font=(
        "Arial", size(label_size), "bold"), width='15', height='1', bd=9, bg='white')
    xdata.grid(row=0, column=0)
    ydata = tk.Label(xydata, text='ydata:', font=(
        "Arial", size(label_size), "bold"), width='15', height='1', bd=9, bg='white')
    ydata.grid(row=0, column=1)
    
    v_fe = tk.StringVar()
    v_fe.set(str(vfe))
    b_emode = tk.Button(xydata, text='K.E.', fg='blue', font=("Arial", size(label_size), "bold"), width=5, height='1', command=emode, bd=2)
    b_emode.grid(row=0, column=2)
    b_copyimg = tk.Button(xydata, fg='red', text='Copy Image to Clipboard', font=('Arial', size(label_size), 'bold'), command=f_copy_to_clipboard, bd=2)
    b_copyimg.grid(row=0, column=3)
    
    
    dl=0
    b_sw = tk.Button(xydata, text='dot', font=('Arial', size(label_size), 'bold'), command=dl_sw, bd=2)

    lcmp = tk.Frame(plots, bg='white')
    lcmp.grid(row=0, column=0)

    w, h = (0.75, 1) if os.name == 'nt' else (0.2, 0.7)
    lcmpd = Figure(figsize=(w*scale, h*scale), layout='constrained')
    cmpg = FigureCanvasTkAgg(lcmpd, master=lcmp)
    cmpg.get_tk_widget().grid(row=0, column=1)
    lsetcmap = tk.Label(lcmp, text='Colormap:', font=(
        "Arial", size(f12), "bold"), bg="white", height='1', bd=1)
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
    
    
    threading.Thread(target=tstate, daemon=True).start()
    if lfs is None:
        b_name.config(state='disable')
        b_excitation.config(state='disable')
        b_desc.config(state='disable')
    if data is not None:
        pr_load(data)
        b_tools = tk.Button(fr_tool, text='Batch\nMaster', command=tools, height=2, font=('Arial', size(14), "bold"), bg='white')
        nlist = lfs.name
        namevar = tk.StringVar(value=nlist[0])
        l_name = tk.OptionMenu(fr_tool, namevar, *nlist, command=change_file)
        if len(lfs.name) > 1:
            if len(lfs.n)>0:lfs.sort='no'
            b_tools.grid(row=0, column=0)
            if len(namevar.get()) >30:
                l_name.config(font=('Arial', size(10), "bold"), width=lfs.max_name_len)
            elif len(namevar.get()) >20:
                l_name.config(font=('Arial', size(12), "bold"), width=len(namevar.get()))
            else:
                l_name.config(font=('Arial', size(14), "bold"), width=len(namevar.get()))
            l_name.grid(row=0, column=1)
        if lfs.f_npz[0]:
            npzf = True
            koffset.config(state='normal')
            k_offset.set('0')
            koffset.config(state='disabled')
    else:
        b_tools, l_name, ev, phi = None, None, None, None
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
    g.bind('<Control-q>', g_close)
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
    g.protocol("WM_DELETE_WINDOW", g_close)
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
