import os, io
import tkinter as tk
import threading
from ctypes import windll
from abc import ABC, abstractmethod
import numpy as np
import xarray as xr
from PIL import Image
import win32clipboard
import cv2
import psutil

class CEC_Object(ABC):
    @abstractmethod
    def info(self):
        pass
    @abstractmethod
    def load(self, angle, cx, cy, cdx, cdy, phi_offset, r1_offset, phi1_offset, r11_offset, slim, sym, name, path):
        pass
    @abstractmethod
    def on_closing(self):
        pass

class FileSequence(ABC):
    @abstractmethod
    def get(self, index: int) -> xr.DataArray:
        pass
    @abstractmethod
    def check_repeat(self, name: list[str]) -> list[str]:
        pass
    @abstractmethod
    def gen_r1(self, name: list[str], r1_splitter: list[str], r2_splitter: list[str]) -> np.float64 | list[str]:
        pass
    @abstractmethod
    def gen_r2(self, name: list[str], r1_splitter: list[str], r2_splitter: list[str]) -> np.float64 | list[str]:
        pass

class MDC_param:
    def __init__(self, ScaleFactor, sc_y, g, scale, npzf, vfe, emf, st, dpath, name, k_offset, value3, ev, phi, data, base, fpr, skmin, skmax, smfp, smfi, smaa1, smaa2, smresult, smcst):
        self.ScaleFactor = ScaleFactor
        self.sc_y = sc_y
        self.g = g
        self.scale = scale
        self.npzf = npzf
        self.vfe = vfe
        self.emf = emf
        self.st = st
        self.dpath = dpath
        self.name = name
        self.k_offset = k_offset
        self.value3 = value3
        self.ev = ev
        self.phi = phi
        self.data = data
        self.base = base
        self.fpr = fpr
        self.skmin = skmin
        self.skmax = skmax
        self.smfp = smfp
        self.smfi = smfi
        self.smaa1 = smaa1
        self.smaa2 = smaa2
        self.smresult = smresult
        self.smcst = smcst
        

class cec_param:
    def __init__(self, path_to_file: str=None, name: str=None, lf_path: list[str]=None, tlfpath: list[str]=None, cmap: str=None):
        self.path_to_file = path_to_file
        self.name = name
        self.lf_path = lf_path
        self.tlfpath = tlfpath
        self.cmap = cmap

class app_param:
    def __init__(self, hwnd=None, scale=None, dpi=None, bar_pos=None, g_mem=None):
        self.hwnd = hwnd
        self.scale = scale
        self.dpi = dpi
        self.bar_pos = bar_pos
        self.g_mem = g_mem

class RestrictedToplevel(tk.Toplevel):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        set_center(parent, self, 0, 0)
        self.parent = parent
        self.width = parent.winfo_reqwidth()
        self.height = parent.winfo_reqheight()
        
    def limit_bind(self):
        # 綁定配置變化事件
        self.bind('<Configure>', self.on_configure)
        
    def on_configure(self, event):
        # 只處理視窗位置變化（不是大小變化）
        if event.widget == self:
            x = self.winfo_x()
            y = self.winfo_y()
            self.x_min = int(self.parent.geometry().split('+')[1])
            self.y_min = int(self.parent.geometry().split('+')[2])
            self.x_max = self.x_min + self.width or self.winfo_screenwidth()
            self.y_max = self.y_min + self.height or self.winfo_screenheight()
            # 限制移動範圍
            new_x = max(self.x_min, min(x, self.x_max - self.winfo_width()))
            new_y = max(self.y_min, min(y, self.y_max - self.winfo_height()))
            
            # 如果位置超出範圍，重新設定
            if x != new_x or y != new_y:
                self.geometry(f"+{new_x}+{new_y}")

def interp(x: float | list[float] | np.ndarray, xp: list[float] | np.ndarray, fp: list[float] | np.ndarray) -> np.ndarray:
    """
    Interpolates a 1-D function.
    Given the data points (xp, fp), this function returns the interpolated values at the points x.
    If the values in x are outside the range of xp, linear extrapolation is used.
    A more general version of np.interp, which can handle decreasing x-coordinates.
    
    Args
    ----------
        x (float): The x-coordinates at which to evaluate the interpolated values.
        xp (float): The x-coordinates of the data points.
        fp (float): The y-coordinates of the data points.

    Returns
    ----------
        out (ndarray) : The interpolated values, same shape as x.
    
    Example
    ----------
        >>> interp(1.5, [1, 2], [2, 3])
        2.5
        >>> interp([1.5, 2.5], [1, 2], [2, 3])
        array([2.5, 3.5])
    """
    if xp[1] >= xp[0]:
        y=np.interp(x,xp,fp)
        try:
            if len(np.array(x))>1:
                for i,v in enumerate(x):
                    if v < xp[0]:
                        y[i]=(v-xp[0])/(xp[1]-xp[0])*(fp[1]-fp[0])+fp[0]
                    elif v > xp[-1]:
                        y[i]=(v-xp[-1])/(xp[-2]-xp[-1])*(fp[-2]-fp[-1])+fp[-1]
        except:
            v=x
            if v < xp[0]:
                y=(v-xp[0])/(xp[1]-xp[0])*(fp[1]-fp[0])+fp[0]
            elif v > xp[-1]:
                y=(v-xp[-1])/(xp[-2]-xp[-1])*(fp[-2]-fp[-1])+fp[-1]
    else:
        xp,fp=xp[::-1],fp[::-1]
        y=np.interp(x,xp,fp)
        try:
            if len(np.array(x))>1:
                for i,v in enumerate(x):
                    if v < xp[0]:
                        y[i]=(v-xp[0])/(xp[1]-xp[0])*(fp[1]-fp[0])+fp[0]
                    elif v > xp[-1]:
                        y[i]=(v-xp[-1])/(xp[-2]-xp[-1])*(fp[-2]-fp[-1])+fp[-1]
        except:
            v=x
            if v < xp[0]:
                y=(v-xp[0])/(xp[1]-xp[0])*(fp[1]-fp[0])+fp[0]
            elif v > xp[-1]:
                y=(v-xp[-1])/(xp[-2]-xp[-1])*(fp[-2]-fp[-1])+fp[-1]
    return y

def copy_to_clipboard(ff) -> None:
    """
    Copies the given figure to the clipboard as a bitmap image.
    
    Parameters
    -----------
        ff (matplotlib.figure.Figure) : The figure to be copied to the clipboard.
    
    Returns
    -----------
        None
    """
    buf = io.BytesIO()
    ff.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    output = io.BytesIO()
    
    image.convert("RGB").save(output, "BMP")
    data = output.getvalue()[14:]
    output.close()
    send_to_clipboard(win32clipboard.CF_DIB, data)
    
def send_to_clipboard(clip_type, data):
    win32clipboard.OpenClipboard()
    win32clipboard.EmptyClipboard()
    win32clipboard.SetClipboardData(clip_type, data)
    win32clipboard.CloseClipboard()

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

def smooth(x,l=20,p=3):
    """
    Using Savitzky-Golay filter to smooth the data.
    
    Parameters
    ------
    x (array_like)
        1D array data to be smoothed
    l : int, default: 20
        window length
    p : int, default: 3
        polynomial order
    """
    from scipy.signal import savgol_filter
    x=savgol_filter(x, l, p)
    # for i in range(len(x)):
    #     if i>=l//2 and i+1<len(x)-l//2:
    #         x[i]=np.mean(x[i-l//2:i+l//2])
    return x

# def res(a, b):
#     a = np.array(a)
#     det = [1 for i in range(len(a)-1)]
#     while sum(det) != 0:
#         for i in range(len(a)-1):
#             if a[i+1] < a[i]:
#                 det[i] = 1
#                 a[i+1], a[i] = a[i], a[i+1]
#                 b[i+1], b[i] = b[i], b[i+1]
#             else:
#                 det[i] = 0
#     return np.array(b)

def res(a: np.ndarray | list[float], b: np.ndarray | list[float]) -> np.ndarray:
    return np.array([b[i] for i in np.argsort(a)])

def hidden_job(path):
    os.system(f'attrib +h +s "{path}"')
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            os.system(f'attrib +h +s "{file_path}"')
        for dirname in dirnames:
            dir_path = os.path.join(dirpath, dirname)
            os.system(f'attrib +h +s "{dir_path}"')
            
def set_hidden(path):
    t = threading.Thread(target=hidden_job, args=(path,))
    t.daemon = True
    t.start()

def find_window():
    # Windows系統中 可能的終端機視窗名稱
    hwnd = windll.user32.FindWindowW(None, "命令提示字元")
    if not hwnd:
        hwnd = windll.user32.FindWindowW(None, "Command Prompt")
    if not hwnd:
        hwnd = windll.user32.FindWindowW(None, "cmd")
    return hwnd

def set_entry_value(entry: tk.Entry, value: str) -> None:
    entry.delete(0, tk.END)
    entry.insert(0, value)

def mesh(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a meshgrid from x and y arrays.
    
    Args
    ------
        x (np.ndarray) : 2D array for x-coordinates.
        y (np.ndarray) : 2D array for y-coordinates.
        
    Returns
    -------
        (X1, Y1, X2, Y2) (tuple) : Meshgrid arrays for x and y coordinates.
    """
    x1, y1 = x.T.copy(), y.T.copy()
    for i in range(x.shape[0]):
        if i % 2 == 0:
            x[i] = x[i][::-1]
            y[i] = y[i][::-1]
    for i in range(y1.shape[0]):
        if i % 2 == 0:
            x1[i] = x1[i][::-1]
            y1[i] = y1[i][::-1]
    return x, y, x1, y1

def rotate(data: cv2.typing.MatLike, angle: float, size: tuple[int, int]) -> cv2.typing.MatLike:
    """
    for square data
    """
    mat = cv2.getRotationMatrix2D((size[1]/2, size[0]/2), angle, 1)
    data = cv2.warpAffine(data, mat, (size[1], size[0]), flags=cv2.INTER_NEAREST)
    return data

def set_center(parent: tk.Tk, child: tk.Toplevel, w_extend: int | None = None, h_extend: int | None = None):
    """
    
    Set the position of child window to the center of parent window.
    The parent window should have been set with a certain geometry.
    
    Args
    ------
        parent : tk.Tk or tk.Toplevel
        child : tk.Toplevel
    Returns
    -------
        None
    """
    if not isinstance(parent, tk.Tk) and not isinstance(parent, tk.Toplevel):
        raise TypeError("Parent must be a Tk or Toplevel instance.")
    if not isinstance(child, tk.Toplevel):
        raise TypeError("Child must be a Toplevel instance.")
    child.update()
    if w_extend is None and h_extend is None:
        w_extend = round(child.winfo_reqwidth()/5)
        h_extend = round(child.winfo_reqheight()/5)
    if w_extend is None:
        w_extend = 0
    if h_extend is None:
        h_extend = 0
    if not isinstance(w_extend, int) or not isinstance(h_extend, int):
        raise TypeError("w_extend and h_extend must be integers.")
    w_parent, h_parent = parent.winfo_reqwidth(), parent.winfo_reqheight()
    w_child, h_child = child.winfo_reqwidth(), child.winfo_reqheight()
    px = parent.winfo_x() + w_parent // 2 - w_child // 2
    py = parent.winfo_y() + h_parent // 2 - h_child // 2
    child.geometry(f'{w_child+w_extend}x{h_child+h_extend}+{px}+{py}')
    return

def det_chunk(cdensity: int, dtype: np.dtype=np.float32) -> int:
    current_mem = psutil.virtual_memory().available/1024**3
    use_mem = current_mem*0.8  # 80%
    print(f"Memory available: {current_mem:.2f} GB, 80% Upper Limit: {use_mem:.2f} GB")
    mem = np.empty((cdensity, cdensity), dtype=dtype).nbytes/1024**3
    chunk_size = int(use_mem / mem)
    mem = None
    return chunk_size

def clear(obj):
    try:
        for i in obj.__dir__():
            try:
                setattr(obj, i, None)
            except: # property does not have setter
                try:
                    obj.i = None
                except:
                    pass
    except:
        pass
