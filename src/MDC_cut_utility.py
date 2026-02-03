import os, io, subprocess
import tkinter as tk
import threading
from abc import ABC, abstractmethod
import numpy as np
import xarray as xr
from PIL import Image
if os.name == 'nt':
    from ctypes import windll
    import win32clipboard
else:
    import tempfile
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

class cec_param:
    def __init__(self, path_to_file: str=None, name: str=None, lf_path: list[str]=None, tlfpath: list[str]=None, cmap: str=None):
        self.path_to_file = path_to_file
        self.name = name
        self.lf_path = lf_path
        self.tlfpath = tlfpath
        self.cmap = cmap

class RestrictedToplevel(tk.Toplevel):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        set_center(parent, self, 0, 0)
        self.parent = parent
        dpi=parent.winfo_fpixels('1i')
        cdir = os.path.dirname(os.path.abspath(__file__))   # .MDC_cut
        path = os.path.join(cdir,'odpi')
        with open(path, 'r') as f:
            s=f.read()
            odpi=float(s)
        try:
            size = int(str(tk.Menu(parent).cget('font')).split(' ')[1])
        except ValueError:
            size = int(str(tk.Menu(parent).cget('font')).split(' ')[-1])
        bd = int(tk.Menu(parent).cget('bd'))
        if os.name != 'nt':
            self.menusize=int((size*dpi/odpi+bd*2*2)*dpi/72)
        else:
            self.menusize=int((size*dpi/odpi+bd*2*2)*windll.shcore.GetScaleFactorForDevice(0)/100)
        
    @property
    def width(self):
        return self.parent.winfo_width()
    
    @property
    def height(self):
        return self.parent.winfo_height()+self.menusize
    
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

class CanvasButton:
    """
    自訂 Canvas 按鈕類別,支援顏色變化和點擊事件
    
    Parameters:
        parent: 父容器 (tk.Tk 或 tk.Frame)
        text (str): 按鈕文字
        command (callable): 點擊時執行的函數
        width (int): 寬度,預設 200
        height (int): 高度,預設 50
        bg_color (str): 背景色,預設 '#00ff00'
        hover_color (str): 懸停色,預設 auto lighten
        text_color (str): 文字色,預設 'black'
        font (tuple): 字型,預設 ('Arial', 14)
        outline_color (str): 邊框色,預設 'black'
        outline_width (int): 邊框寬度,預設 2
    """
    
    def __init__(self, parent, text='Button', command=None, width=200, height=50,
                 bg_color='#00ff00', hover_color=None, text_color='black',
                 font=('Arial', 14), outline_color='black', outline_width=2):
        
        self.command = command
        self.bg_color = bg_color
        self.hover_color = hover_color if hover_color is not None else None
        
        # 建立 Canvas
        self.canvas = tk.Canvas(
            parent, 
            width=width, 
            height=height, 
            bg='white', 
            highlightthickness=0
        )
        
        # 計算矩形座標 (邊距 10 pixels)
        margin = 10
        self.rect = self.canvas.create_rectangle(
            margin, margin, 
            width - margin, height - margin,
            fill=bg_color,
            outline=outline_color,
            width=outline_width
        )
        
        # 建立文字
        self.text = self.canvas.create_text(
            width // 2, 
            height // 2,
            text=text,
            font=font,
            fill=text_color
        )
        
        # 綁定事件
        self.canvas.bind('<Enter>', self._on_enter)
        self.canvas.bind('<Leave>', self._on_leave)
        self.canvas.bind('<Button-1>', self._on_click)
    
    def config(self, **kwargs):
        if 'bg' or 'background' in kwargs:
            self.bg_color = kwargs.get('bg', kwargs.get('background', self.bg_color))
            self.canvas.itemconfig(self.rect, fill=self.bg_color)
    
    @staticmethod
    def hex_to_rgb(hex_color):
        """將十六進位色碼轉換為 RGB"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    @staticmethod
    def rgb_to_hex(rgb):
        """將 RGB 轉換為十六進位色碼"""
        return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
    
    @staticmethod
    def lighten_color(hex_color, factor=0.7):
        """使顏色變淡 (用於懸停效果)"""
        r, g, b = CanvasButton.hex_to_rgb(hex_color)
        r = int(r + (255 - r) * (1 - factor))
        g = int(g + (255 - g) * (1 - factor))
        b = int(b + (255 - b) * (1 - factor))
        return CanvasButton.rgb_to_hex((r, g, b))
    
    def _on_enter(self, event):
        """滑鼠進入時改變顏色"""
        hover_color = self.hover_color if self.hover_color is not None else self.lighten_color(self.bg_color, 0.4)
        self.canvas.itemconfig(self.rect, fill=hover_color)
    
    def _on_leave(self, event):
        """滑鼠離開時恢復顏色"""
        self.canvas.itemconfig(self.rect, fill=self.bg_color)
    
    def _on_click(self, event):
        """點擊時執行指令"""
        if self.command:
            self.command()
    
    def pack(self, **kwargs):
        """包裝 Canvas (傳遞給 pack 方法)"""
        self.canvas.pack(**kwargs)
    
    def grid(self, **kwargs):
        """網格佈局 Canvas (傳遞給 grid 方法)"""
        self.canvas.grid(**kwargs)
    
    def place(self, **kwargs):
        """位置佈局 Canvas (傳遞給 place 方法)"""
        self.canvas.place(**kwargs)

def on_configure(g, *e):
    if g.winfo_width() < g.winfo_reqwidth() or g.winfo_height() < g.winfo_reqheight():
        g.geometry(f"{g.winfo_reqwidth()}x{g.winfo_reqheight()}")

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
    if os.name == 'nt':
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
    if os.name == 'nt':
        hwnd = windll.user32.FindWindowW(None, "命令提示字元")
        if not hwnd:
            hwnd = windll.user32.FindWindowW(None, "Command Prompt")
        if not hwnd:
            hwnd = windll.user32.FindWindowW(None, "cmd")
        return hwnd
    else:
        return 0

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
    w_parent, h_parent = parent.winfo_width(), parent.winfo_height()
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

def file_walk(path=[], file_type='.h5'):
    out = []
    if path != []:
        if os.path.isdir(path):
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    if filename.endswith(file_type) or filename.endswith(file_type.upper()):
                        full_path = os.path.join(dirpath, filename)
                        out.append(full_path)
        elif os.path.isfile(path):
            if path.endswith(file_type) or path.endswith(file_type.upper()):
                out.append(path)
    return out

def gl1(x, x0, a, w, y0):
    """
    Calculate the value of a Lorentzian function at a given x-coordinate.

    Parameters:
        x (float): The x-coordinate at which to evaluate the function.
        x0 (float): The center of the Lorentzian function.
        a (float): The amplitude of the Lorentzian function.
        w (float): The full width at half maximum (FWHM) of the Lorentzian function.
        y0 (float): The y-offset of the Lorentzian function.

    Returns:
        float: The value of the Lorentzian function at the given x-coordinate.
    """
    v = a/(1+(x-x0)**2/(1/2*w)**2)+y0
    return v

def gl2(x, x1, h1, w1, y1, x2, h2, w2, y2):
    """
    Calculates the sum of two Lorentzian functions.

    Parameters:
        x (float): The input value.
        x1 (float): The center of the first Lorentzian function.
        h1 (float): The height of the first Lorentzian function.
        w1 (float): The width of the first Lorentzian function.
        y1 (float): The y-offset of the first Lorentzian function.
        x2 (float): The center of the second Lorentzian function.
        h2 (float): The height of the second Lorentzian function.
        w2 (float): The width of the second Lorentzian function.
        y2 (float): The y-offset of the second Lorentzian function.

    Returns:
        float: The sum of the two Lorentzian functions.
    """
    v1 = h1/(1+(x-x1)**2/(1/2*w1)**2)+y1
    v2 = h2/(1+(x-x2)**2/(1/2*w2)**2)+y2
    return v1+v2

def fgl2(params, x, data):
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
    return model - data

def fgl1(params, xx, data):
    h = params['h']
    x = params['x']
    w = params['w']
    y = params['y']
    model = gl1(xx, x, h, w, y)
    return model - data

def filter(y, a, b):
    """
    Filters the input array y based on the conditions defined by a and b.
    Returns two arrays: one containing the filtered values and another containing
    the indices of the filtered values in the original array.
    If a is greater than b, it swaps them to ensure a is always less than or equal to b.
    If no values in y meet the condition, it returns empty arrays.
    
    Parameters:
        y (array-like): The input array to be filtered.
        a (float): The lower bound for filtering.
        b (float): The upper bound for filtering.
    Returns:
        (filtered y, index of filtered y) (tuple): A tuple of two ndarrays.
    
    Example:
        >>> y = [1, 2, 3, 4, 5]
        >>> a = 2
        >>> b = 4
        >>> filter(y, a, b)
        (array([2, 3, 4]), array([1, 2, 3]))
    """
    if a > b:
        a, b = b, a  # Ensure a is less than or equal to b
    return np.array([x for x in y if a <= x <= b]), np.array([i for i, x in enumerate(y) if a <= x <= b])

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

def cal_ver(ver):
    ver = [int(i) for i in ver.split('.')]
    if len(ver) != 3:
        ver.append(0)
    ver = ver[0]*10000 + ver[1]*100 + ver[2]
    return ver
