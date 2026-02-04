import pytest
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QPointF, QPoint, Qt
from PyQt5.QtGui import QWheelEvent
import os, sys
import shutil, inspect
import time
import queue
from typing import Literal, override
tdir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')
sys.path.insert(0, tdir)
sys.path.insert(0, os.path.dirname(tdir))
cdir = os.path.dirname(os.path.dirname(__file__))
if not os.path.exists(os.path.join(cdir, '.MDC_cut')):
    os.mkdir(os.path.join(cdir, '.MDC_cut'))
    url = [r"https://github.com/alex20000910/main/blob/main/MDC_cut.py",
           r"https://github.com/alex20000910/main/blob/main/src/viridis_2D.otp",
           r"https://github.com/alex20000910/main/blob/main/src/MDC_cut_utility.py",
           r"https://github.com/alex20000910/main/blob/main/src/tool/__init__.py",
           r"https://github.com/alex20000910/main/blob/main/src/tool/util.py",
           r"https://github.com/alex20000910/main/blob/main/src/tool/loader.py",
           r"https://github.com/alex20000910/main/blob/main/src/tool/spectrogram.py",
           r"https://github.com/alex20000910/main/blob/main/src/tool/SO_Fitter.py",
           r"https://github.com/alex20000910/main/blob/main/src/tool/VolumeSlicer.py",
           r"https://github.com/alex20000910/main/blob/main/src/tool/CEC.py",
           r"https://github.com/alex20000910/main/blob/main/src/tool/DataViewer.py",
           r"https://github.com/alex20000910/main/blob/main/src/tool/MDC_Fitter.py",
           r"https://github.com/alex20000910/main/blob/main/src/tool/EDC_Fitter.py",
           r"https://github.com/alex20000910/main/blob/main/src/tool/window.py",
           r"https://github.com/alex20000910/main/blob/main/src/tool/RawDataViewer.py",
           r"https://github.com/alex20000910/main/blob/main/src/tool/qt_util.py"]
    for i, v in enumerate(url):
        if i < 3:
            src = os.path.join(cdir, 'src', os.path.basename(v))
            dst = os.path.join(cdir, '.MDC_cut', os.path.basename(v))
        else:
            src = os.path.join(cdir, 'src', 'tool', os.path.basename(v))
            dst = os.path.join(cdir, '.MDC_cut', 'tool', os.path.basename(v))
        os.system(f'copy "{src}" "{dst}" > nul')
os.system('attrib +h +s .MDC_cut')
sys.path.append(os.path.join(cdir, '.MDC_cut'))
sys.path.append(os.path.join(cdir, '.MDC_cut', 'tool'))
from MDC_cut_utility import *
from tool.loader import loadfiles, tkDnD_loader, load_h5
from tool.spectrogram import spectrogram, lfs_exp_casa
from tool.SO_Fitter import SO_Fitter
from tool.CEC import CEC, call_cec, CEC_Object

def test_loadfiles():
    path = []
    path.append(os.path.join(os.path.dirname(__file__), 'simulated_R1_15.0_R2_0#id#0d758f03.h5'))
    path.append(os.path.join(os.path.dirname(__file__), 'UPSPE20_2_test_1559#id#3cf2122d.json'))
    lfs = loadfiles(path)
    print(lfs)
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
    path.append(os.path.join(os.path.dirname(__file__), 'simulated_R1_15.0_R2_0#id#0d758f03.h5'))
    path.append(os.path.join(os.path.dirname(__file__), 'UPSPE20_2_test_1559#id#3cf2122d.json'))
    files = tkDnD.load_raw(path)
    assert isinstance(files, list)

@pytest.fixture(scope="module")
def tk_root():
    """模組級別的 Tk root,確保整個測試過程只有一個實例"""
    import tkinter as tk
    
    # 確保沒有其他 Tk 實例
    try:
        root = tk.Tk()
    except tk.TclError:
        # 如果失敗,嘗試清理並重試
        import gc
        gc.collect()
        root = tk.Tk()
    
    root.withdraw()  # 隱藏主視窗
    
    yield root
    
    # 清理
    try:
        for widget in root.winfo_children():
            try:
                widget.destroy()
            except:
                pass
        root.quit()
        root.destroy()
    except:
        pass

@pytest.fixture
def tk_environment(tk_root):
    """使用共用的 root,每個測試創建獨立的 frame"""
    frame = tk.Frame(tk_root)
    
    yield tk_root, frame
    
    # 清理 frame
    try:
        for widget in frame.winfo_children():
            try:
                widget.destroy()
            except:
                pass
        frame.destroy()
    except:
        pass
    
    # 更新事件循環
    try:
        tk_root.update_idletasks()
        tk_root.update()
    except:
        pass

def set_globals(var, glob):
    if var is not None:
        globals()[glob] = var

def test_data_loader(tk_environment):
    g, frame = tk_environment
    from tool.loader import data_loader, file_loader
    from base64 import b64decode
    from PIL import Image, ImageTk
    menu1 = tk.OptionMenu(frame, tk.StringVar(value='Option1'), 'Option1', 'Option2')
    menu2 = tk.OptionMenu(frame, tk.StringVar(value='OptionA'), 'OptionA', 'OptionB')
    menu3 = tk.OptionMenu(frame, tk.StringVar(value='ChoiceX'), 'ChoiceX', 'ChoiceY')
    in_fit = tk.Entry(frame)
    b_fit = tk.Button(frame, text='Fit')
    l_path = tk.Text(frame)
    info = tk.Text(frame)
    cdir = os.path.dirname(os.path.dirname(__file__))
    path = []
    path.append(os.path.join(os.path.dirname(__file__), 'data_cut.npz'))
    path.append(os.path.join(os.path.dirname(__file__), 'simulated_R1_15.0_R2_0#id#0d758f03.h5'))
    path.append(os.path.join(os.path.dirname(__file__), 'UPSPE20_2_test_1559#id#3cf2122d.json'))
    app_pars = app_param(hwnd=None, scale=1, dpi=96, bar_pos='bottom', g_mem=0.25)
    lfs = loadfiles(path, init=True, name='internal', cmap='viridis', app_pars=app_pars)
    scale = 1.0
    
    
    dpath = lfs.path[0]
    st = queue.Queue()
    limg = tk.Label(frame)
    icon = IconManager()
    img = []
    Icon = [icon.icon0]
    timg = Image.open(io.BytesIO(b64decode(Icon[0]))).resize([250, 250])
    tk_img = ImageTk.PhotoImage(timg)
    img.append(tk_img)
    b_name = tk.Button(frame, text='Set Name')
    b_excitation = tk.Button(frame, text='Set Excitation')
    b_desc = tk.Button(frame, text='Set Description')
    koffset = tk.Entry(frame)
    k_offset = tk.StringVar(value='0')
    fr_tool = tk.Frame(frame)
    b_tools = tk.Button(frame, text='Tools')
    l_name = tk.OptionMenu(frame, tk.StringVar(value='Name1'), 'Name1', 'Name2')
    scale = 1.0
    class pr_load(data_loader):
        def __init__(self, data: xr.DataArray):
            super().__init__(menu1, menu2, menu3, in_fit, b_fit, l_path, info, cdir, lfs, scale)
            self.pr_load(data)

        @override
        def pars(self):
            pass
        
    for i in range(len(lfs.name)):
        pr_load(lfs.get(i))
    
    class main_loader(file_loader):
        def __init__(self, files: tuple[str]|Literal['']):
            super().__init__(files, dpath, 'viridis', lfs, g, app_pars, st, limg, img, b_name, b_excitation, b_desc, koffset, k_offset, fr_tool, b_tools, l_name, scale, test=True)
        
        @override
        def call_cec(self, g, lfs) -> FileSequence:
            return call_cec(g, lfs, test=True)

        @override
        def pr_load(self, data):
            pr_load(data)
        
        @override
        def change_file(self, *args):
            pass
            
        @override
        def tools(self, *args):
            pass
        
        @override
        def set_k_offset(self):
            pass
        
        @override
        def pars(self):
            pass
    main_loader('')
    main_loader(lfs.path)

def test_spectrogram(tk_environment):
    g, frame = tk_environment
    path = []
    path.append(os.path.join(os.path.dirname(__file__), 'simulated_R1_15.0_R2_0#id#0d758f03.h5'))
    path.append(os.path.join(os.path.dirname(__file__), 'UPSPE20_2_test_1559#id#3cf2122d.json'))
    data = load_h5(path[0])
    ev, phi = data.indexes.values()
    app_pars = app_param(hwnd=None, scale=1, dpi=96, bar_pos='bottom', g_mem=0.25)
    y=smooth(np.sum(data.to_numpy().transpose(),axis=0),l=13)
    s = spectrogram(data, name='internal', app_pars=app_pars)
    s.setdata(ev, y, dtype='smooth', unit='Counts')
    s.plot(g)
    s.b_exp.invoke()
    s.copy_button.invoke()
    s.info.event_generate('<FocusIn>')
    for i in [s.rpo, s.tpo]:
        i.get_tk_widget().event_generate('<Button-1>', x=400, y=250)
        i.get_tk_widget().event_generate('<Motion>', x=450, y=250)
        i.get_tk_widget().event_generate('<Motion>', x=480, y=250)
        i.get_tk_widget().event_generate('<ButtonRelease-1>', x=500, y=250)
        i.get_tk_widget().event_generate('<Button-3>', x=400, y=250)
        i.get_tk_widget().event_generate('<ButtonRelease-3>', x=400, y=250)
        i.get_tk_widget().event_generate('<Button-1>', x=450, y=250)
        i.get_tk_widget().event_generate('<ButtonRelease-1>', x=450, y=250)
    s.rgo.get_tk_widget().event_generate('<Button-1>', x=13, y=436)
    s.rgo.get_tk_widget().event_generate('<Motion>', x=13, y=300)
    s.rgo.get_tk_widget().event_generate('<ButtonRelease-1>', x=13, y=300)
    s.rgo.get_tk_widget().event_generate('<Button-1>', x=13, y=40)
    s.rgo.get_tk_widget().event_generate('<Motion>', x=13, y=150)
    s.rgo.get_tk_widget().event_generate('<ButtonRelease-1>', x=13, y=150)
    s.rgo.get_tk_widget().event_generate('<Button-1>', x=13, y=200)
    s.rgo.get_tk_widget().event_generate('<Motion>', x=13, y=250)
    s.rgo.get_tk_widget().event_generate('<ButtonRelease-1>', x=13, y=250)
    s.tpg.event_generate('<Return>')
    t=time.time()
    while time.time()-t<60:
        try:
            if s.grg.winfo_exists():
                break
        except:
            pass
    if time.time()-t>=60:
        print('timeout 60s')
    else:
        s.grg.event_generate('<Return>')
    s.closing()
    s = spectrogram(path=path, name='external', app_pars=app_pars)
    s.plot(g)
    s.cf_up()
    s.cf_down()
    assert isinstance(s.name, str)
    assert isinstance(s.data, xr.DataArray)
    s.ups()
    for option in s.fit_options:
        s.selected_fit.set(option)
        s.update_plot()
        if option in ["Fermi-Dirac Fitting", "ERFC Fit"]:
            s.update_fit()
            s.canvas.get_tk_widget().event_generate('<Motion>', x=765, y=300)
            s.canvas.get_tk_widget().event_generate('<Button-1>', x=780, y=300)
            s.canvas.get_tk_widget().event_generate('<ButtonRelease-1>', x=780, y=300)
            s.canvas.get_tk_widget().event_generate('<Motion>', x=950, y=300)
            s.canvas.get_tk_widget().event_generate('<Button-1>', x=860, y=300)
            s.canvas.get_tk_widget().event_generate('<ButtonRelease-1>', x=860, y=300)

def test_lfs_exp_casa():
    opath = []
    path = os.path.join(os.path.dirname(__file__), 'simulated_R1_15.0_R2_0#id#0d758f03.h5')
    opath.append(path)
    path = os.path.join(os.path.dirname(__file__), 'simulated_R1_15.1_R2_0#id#d7bebfaa.h5')
    opath.append(path)
    path = os.path.join(os.path.dirname(__file__), 'simulated_R1_15.5_R2_0#id#1ee3c8fd.h5')
    opath.append(path)
    path = os.path.join(os.path.dirname(__file__), "simulated_R1_15.0_R2_60#id#67245b5a.h5")
    opath.append(path)
    path = os.path.join(os.path.dirname(__file__), "simulated_R1_15.1_R2_60#id#1e8223d1.h5")
    opath.append(path)
    path = os.path.join(os.path.dirname(__file__), "simulated_R1_15.5_R2_60#id#56c06b00.h5")
    opath.append(path)
    app_pars = app_param(hwnd=None, scale=1, dpi=96, bar_pos='bottom', g_mem=0.25)
    lfs = loadfiles(opath, name='internal', cmap='viridis', app_pars=app_pars)
    explfs = lfs_exp_casa(lfs)
    path = os.path.join(os.path.dirname(__file__), 'exp_casa.vms')
    explfs.export_casa(path)

def init_tempdir():
    tempdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    tempdir = os.path.dirname(tempdir)
    os.chdir(os.path.join(tempdir))
    if os.path.exists(os.path.join(tempdir, 'stop_signal')):
        os.remove(os.path.join(tempdir, 'stop_signal'))
    if os.path.exists('cut_temp_save'):
        shutil.rmtree('cut_temp_save')
    os.mkdir('cut_temp_save')
    if os.path.exists('cube_temp_save'):
        shutil.rmtree('cube_temp_save')
    os.mkdir('cube_temp_save')

def test_VolumeSlicer(tk_environment):
    g, frame = tk_environment
    from tool.VolumeSlicer import VolumeSlicer, g_cut_plot, cut_job_x, cut_job_y
    import cpuinfo
    import psutil
    import zarr
    odpi=g.winfo_fpixels('1i')
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src', 'odpi')
    with open(path, 'w') as f:
        f.write(f'{odpi}')  #for RestrictedToplevel
        f.close()
    odata = []
    opath = []
    path = os.path.join(os.path.dirname(__file__), 'simulated_R1_15.0_R2_0#id#0d758f03.h5')
    odata.append(load_h5(path))
    opath.append(path)
    path = os.path.join(os.path.dirname(__file__), 'simulated_R1_15.1_R2_0#id#d7bebfaa.h5')
    odata.append(load_h5(path))
    opath.append(path)
    path = os.path.join(os.path.dirname(__file__), 'simulated_R1_15.5_R2_0#id#1ee3c8fd.h5')
    odata.append(load_h5(path))
    opath.append(path)
    path = os.path.join(os.path.dirname(__file__), "simulated_R1_15.0_R2_60#id#67245b5a.h5")
    odata.append(load_h5(path))
    opath.append(path)
    path = os.path.join(os.path.dirname(__file__), "simulated_R1_15.1_R2_60#id#1e8223d1.h5")
    odata.append(load_h5(path))
    opath.append(path)
    path = os.path.join(os.path.dirname(__file__), "simulated_R1_15.5_R2_60#id#56c06b00.h5")
    odata.append(load_h5(path))
    opath.append(path)
    odataframe = np.stack([i.transpose() for i in odata], axis=0, dtype=np.float32)
    r1 = np.array([15.0, 15.1, 15.5, 15.0, 15.1, 15.5])
    r2 = np.array([0.0, 0.0, 0.0, 60.0, 60.0, 60.0])
    
    ev, phi = odata[0].indexes.values()
    app_pars = app_param(hwnd=None, scale=1, dpi=96, bar_pos='bottom', g_mem=0.25)
    e_photon = float(odata[0].attrs['ExcitationEnergy'].removesuffix(' eV'))
    vs = VolumeSlicer(parent=frame, path=opath, volume=odataframe, x=phi, y=r1, z=r2, ev=ev, e_photon=e_photon, g=g, app_pars=app_pars)
    vs.test = True
    vs.change_mode()
    assert vs.surface.shape ==(vs.density, vs.density)
    assert vs.surface.dtype == np.float32
    assert vs.surface.flatten().max() > 0
    xlim=[15.0, 15.5]
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
    vs.fit_so_app()
    vs.set_density()
    vs.set_window()
    vs.text_e.set(str(f'%.3f'%vs.ev[400]))
    vs.set_slim()
    vs.text_a.set(str(30))
    vs.angle_slider.set_val(-30)
    vs.symmetry()
    vs.symmetry_(6)
    vs.r1_offset = 15.25 # for test boost the speed
    set_entry_value(vs.entry_r1_offset, str(vs.r1_offset))
    set_entry_value(vs.cut_xy_x_entry, '0.35')
    set_entry_value(vs.cut_xy_y_entry, '0.35')
    set_entry_value(vs.cut_xy_dx_entry, '0.2')
    set_entry_value(vs.cut_xy_dy_entry, '0.2')
    vs.stop_event = threading.Event()
    vs.set_xy_lim()
    vs.cdensity = int((vs.xmax-vs.xmin)//2e-3)
    print('\nSampling Density: \033[31m500 per 2pi/Angstrom')
    print('\033[0mProcessing \033[32m%d x %d x %d \033[0msize data cube'%(vs.cdensity, vs.cdensity, len(vs.ev)))
    print('\n\033[33mProcessor:\033[36m',cpuinfo.get_cpu_info()['brand_raw'])
    print('\033[33mPhysical CPU cores:\033[36m', psutil.cpu_count(logical=False))
    vs.det_core_num()
    print('\033[0mPlease wait...\n')
    print('\nThe following shows the progress bar and the estimation of the processing time')
    angle = vs.angle
    vs.cx_cut = vs.cx
    vs.cy_cut = vs.cy
    vs.cdx_cut = vs.cdx
    vs.cdy_cut = vs.cdy
    x = [vs.cx-vs.cdx/2, vs.cx+vs.cdx/2]
    z = [vs.cy-vs.cdy/2, vs.cy+vs.cdy/2]
    ty = vs.ev
    vs.data_cut = np.zeros((len(ty), vs.cdensity), dtype=np.float32)
    # vs.data_cube = np.zeros((len(ty), vs.cdensity, vs.cdensity), dtype=np.float32)
    vs.data_cube = np.empty((len(ty), vs.cdensity, vs.cdensity), dtype=np.uint8)
    phi_offset = vs.phi_offset
    r1_offset = vs.r1_offset
    phi1_offset = vs.phi1_offset
    r11_offset = vs.r11_offset
    vs.slim_cut = vs.slim.copy()
    vs.sym_cut = vs.sym
    self_x = vs.ox[vs.slim[0]:vs.slim[1]+1]
    self_volume = vs.ovolume[:, vs.slim[0]:vs.slim[1]+1, :]
    i = 200
    args = (i, angle, phi_offset, r1_offset, phi1_offset, r11_offset, self_x, self_volume[:, :, i], vs.cdensity, vs.xmax, vs.xmin, vs.ymax, vs.ymin, z, x, vs.z, vs.y, vs.ev, vs.e_photon, vs.sym)
    init_tempdir()
    cut_job_y(args)
    init_tempdir()
    cut_job_x(args)
    vs.t_cut_job_y()
    vs.t_cut_job_x()
    tempdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    tempdir = os.path.dirname(tempdir)
    for i in range(len(ty)):
        try:
            vs.data_cut[i] = zarr.open_array(os.path.join(tempdir, 'cut_temp_save', f'cut_{i}.zarr'))
            # vs.data_cube[i] = zarr.open_array(os.path.join(tempdir, 'cube_temp_save', f'cube_{i}.zarr'))
        except FileNotFoundError:
            pass
    gcp = g_cut_plot(vs, vs.data_cut, vs.cx, vs.cy, vs.cdx, vs.cdy, vs.cdensity, ty, z, x, angle, phi_offset, r1_offset, phi1_offset, r11_offset, vs.stop_event, vs.pool, vs.path, vs.e_photon, vs.slim_cut, vs.sym_cut, vs.xmin, vs.xmax, vs.ymin, vs.ymax, vs.data_cube, vs.app_pars, test=True)
    gcp.save_cut(path=os.path.join(os.path.dirname(__file__), 'test_cut.h5'))
    gcp.save_cube(path=os.path.join(os.path.dirname(__file__), 'test_cube.zarr'))
    gcp.on_closing()
    vs.change_mode()

def test_CEC(tk_environment):
    g, frame = tk_environment
    from tool.MDC_Fitter import get_file_from_github
    from MDC_cut_utility import file_walk
    app_pars = app_param(hwnd=None, scale=1, dpi=96, bar_pos='bottom', g_mem=0.25)
    tg = wait(g, app_pars)
    tg.text('Preparing sample data...')
    path = rf"simulated_R1_15.0_R2_0#id#0d758f03.h5"
    tpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test_data_temp', rf"simulated_R1_15.0_R2_0#id#0d758f03.h5")
    if os.path.exists(tpath)==False:
        get_file_from_github(r"https://github.com/alex20000910/main/blob/main/test_data/"+path, tpath)
    
    r1files = []
    r1files.append(os.path.join(os.path.dirname(__file__), 'simulated_R1_15.0#id#86e3731a.h5'))
    r1files.append(os.path.join(os.path.dirname(__file__), 'simulated_R1_15.1#id#a57f6928.h5'))
    r1files.append(os.path.join(os.path.dirname(__file__), 'simulated_R1_15.5#id#e19d4403.h5'))
    r1files.append(os.path.join(os.path.dirname(__file__), 'simulated_R1_15.0#id#86e3731a#d#20260127_161305.h5'))
    path = os.path.dirname(__file__)
    files = file_walk(path)
    for file in files:
        if 'simulated' not in file:
            files.remove(file)
    for i in r1files:
        if i in files:
            files.remove(i)
    tg.done()
    tg = wait(g, app_pars)
    tg.text('Loading sample data...')
    lfs = loadfiles(files)
    tg.done()
    t_cec = CEC(g, lfs.path, cmap='viridis', app_pars=app_pars)
    time.sleep(2)
    if t_cec.gg.winfo_exists():
        t_cec.check()
    if t_cec.gg.winfo_exists():
        t_cec.check()
    t_cec.info()
    lfs = loadfiles(r1files)
    t_cec = CEC(g, lfs.path, cmap='viridis', app_pars=app_pars)
    time.sleep(2)
    if t_cec.gg.winfo_exists():
        t_cec.check()
    t_cec.info()
    
def test_call_cec(tk_environment):
    g, frame = tk_environment
    app_pars = app_param(hwnd=None, scale=1, dpi=96, bar_pos='bottom', g_mem=0.25)
    lfs = loadfiles(os.path.join(os.path.dirname(__file__), 'test_cut.h5'), init=True, mode='eager', name='internal', cmap='viridis', app_pars=app_pars)
    lfs = call_cec(g, lfs, test=True)
    assert isinstance(lfs.cec, CEC_Object)
    lfs.cec.info()
    lfs.cec.on_closing()
    lfs = loadfiles(os.path.join(os.path.dirname(__file__), 'data_cut.h5'), init=True, mode='eager', name='internal', cmap='viridis', app_pars=app_pars)
    lfs = call_cec(g, lfs, test=True)
    assert lfs.cec is None

def test_interp():
    y = interp(0, [1, 2], [2, 3])
    assert y == 1
    y = interp(2.5, [2, 1], [3, 2])
    assert y == 3.5
    y_array = interp([0, 1.5, 2.5], [1, 2], [2, 3])
    assert np.allclose(y_array, [1, 2.5, 3.5])
    y_array = interp(np.array([0, 1.5, 2.5]), np.array([2, 1]), np.array([3, 2]))
    assert np.allclose(y_array, [1, 2.5, 3.5])


def test_get_bar_pos():
    pos = get_bar_pos()
    assert isinstance(pos, str)

def test_get_hwnd():
    from tool.MDC_Fitter import get_hwnd
    hwnd = get_hwnd()
    assert isinstance(hwnd, int)

def set_point(so_fitter: SO_Fitter, r2: list=[20, 80, 206]):
    so_fitter.v_r2.set(r2[0])
    so_fitter.v_r1.set(20)
    so_fitter.v_phi.set(0)
    so_fitter.add_point()
    so_fitter.v_r2.set(r2[1])
    so_fitter.v_r1.set(22.5)
    so_fitter.v_phi.set(-0.2)
    so_fitter.add_point()
    so_fitter.v_r2.set(r2[2])
    so_fitter.v_r1.set(19.5)
    so_fitter.v_phi.set(0)
    so_fitter.add_point()
    
def test_SO_Fitter(tk_environment):
    g, frame = tk_environment
    app_pars = app_param(hwnd=None, scale=1, dpi=96, bar_pos='bottom', g_mem=0.25)
    so_fitter = SO_Fitter(g, app_pars)
    so_fitter.add_point()
    set_point(so_fitter)
    so_fitter.set_tolerance()
    so_fitter._fit()
    so_fitter.clear_points()
    set_point(so_fitter, r2=[0, 31, 119])
    so_fitter._fit()
    so_fitter.clear_points()
    set_point(so_fitter, r2=[0, 89, 182])
    so_fitter._fit()
    so_fitter.clear_points()
    set_point(so_fitter, r2=[0, 121, 242])
    so_fitter._fit()
    so_fitter.clear_points()
    so_fitter.v_r2.set(0)
    so_fitter.v_r1.set(20)
    so_fitter.v_phi.set(0)
    so_fitter.add_point()
    so_fitter.v_r2.set(181)
    so_fitter.v_r1.set(22.5)
    so_fitter.v_phi.set(-0.2)
    so_fitter.add_point()
    so_fitter._fit()
    so_fitter.clear_points()
    so_fitter.on_closing()

def test_mfit_data():
    from tool.MDC_Fitter import mfit_data
    from tool.loader import mloader
    st = queue.Queue()
    path = os.path.join(os.path.dirname(__file__), 'simulated_R1_15.0_R2_0#id#0d758f03.h5')
    data = load_h5(path)
    ev, phi = data.indexes.values()
    rdd = path
    cdir = os.path.dirname(os.path.dirname(__file__))
    lowlim = '0'
    ml = mloader(st, data, ev, phi, rdd, cdir, lowlim)
    ml.loadparam('0', '0', True, 1)
    ml.loadmfit_(os.path.join(cdir, 'tests', 'rev_PE10A20f -170 VIV_pass.vms'))
    ml.loadmfit_re(os.path.join(cdir, 'tests', 'PE10A20f -170 VIV.vms'))
    ml.loadmfit_2p(os.path.join(cdir, 'tests', 'rev_PE10A20f -170 VIV.vms'))
    ml.loadmfit_(os.path.join(cdir, 'tests', 'simulated_R1_14.0_R2_0_mfit_1peak.npz'))
    ml.loadmfit_(os.path.join(cdir, 'tests', 'simulated_R1_14.0_R2_0_mfit.npz'))
    mdata = mfit_data(cdir=cdir)
    ko, fev, rpos, ophi, fwhm, mpos, kmin, kmax, skmin, skmax, smaa1, smaa2, smfp, smfi, smresult, smcst, fpr, mdet = mdata.get()
    assert isinstance(ko, str)
    info = ['    x1: -0.04088383 +/- 3.2355e-04 (0.79%) (init = -0.06451523)',
            '    x2:  0.03889559 +/- 5.5296e-04 (1.42%) (init = 0.06826833)',
            '    h1:  8229.22392 +/- 164.752546 (2.00%) (init = 8913)',
            '    h2:  5771.04494 +/- 1456.15965 (25.23%) (init = 8913)',
            '    w1:  0.02869309 +/- 9.8838e-04 (3.44%) (init = 0.02)',
            '    w2:  0.04064557 +/- 0.00169512 (4.17%) (init = 0.02)']
    for i, j in zip(info, smresult[510]):
        assert i == str(j)

def test_Icon():
    icon_manager = IconManager()

def test_ToolTip(tk_environment):
    g, frame = tk_environment
    scaled_font_size = 1
    icon_manager = MenuIconManager()
    class ToolTip(ToolTip_util):
        def __init__(self, widget: tk.Widget, text: str, accelerator=None):
            super().__init__(widget, text, accelerator,
                             icon_manager, scaled_font_size)
    button = Button(frame, text="Load Raw Data", image=icon_manager.get_icon('raw_data'))
    button.pack()
    tt = ToolTip(button, "This is a tooltip", "Ctrl+O")
    event = tk.Event()
    event.x = 10
    event.y = 10
    event.x_root = 100
    event.y_root = 100
    tt.show_tooltip(event)
    tt.hide_tooltip(event)
    tt.update_position(event=event)

def test_qt_widget(qtbot):
    """測試 Qt widget"""
    widget = QtWidgets.QPushButton("Click me")
    qtbot.addWidget(widget)
    
    # 模擬點擊
    qtbot.mouseClick(widget, QtCore.Qt.LeftButton)

def drag_bl1(qtbot, plot_widget):
    qtbot.mouseMove(plot_widget, pos=QPoint(400, 300))
    qtbot.wait(50)
    qtbot.mousePress(plot_widget, Qt.LeftButton, pos=QPoint(433, 300))
    qtbot.wait(50)
    qtbot.mouseMove(plot_widget, pos=QPoint(400, 300))
    qtbot.wait(50)
    qtbot.mouseMove(plot_widget, pos=QPoint(270, 300))
    qtbot.wait(50)
    qtbot.mouseRelease(plot_widget, Qt.LeftButton, pos=QPoint(270, 300))
    qtbot.wait(50)
    qtbot.mouseMove(plot_widget, pos=QPoint(470, 300))
    qtbot.wait(50)
    qtbot.mousePress(plot_widget, Qt.LeftButton, pos=QPoint(470, 300))
    qtbot.wait(50)
    qtbot.mouseMove(plot_widget, pos=QPoint(470, 300))
    qtbot.wait(50)
    qtbot.mouseMove(plot_widget, pos=QPoint(650, 300))
    qtbot.wait(50)
    qtbot.mouseRelease(plot_widget, Qt.LeftButton, pos=QPoint(650, 300))
    qtbot.wait(50)

def drag_bl2(qtbot, plot_widget):
    qtbot.mouseMove(plot_widget, pos=QPoint(400, 300))
    qtbot.wait(50)
    qtbot.mousePress(plot_widget, Qt.LeftButton, pos=QPoint(439, 300))
    qtbot.wait(50)
    qtbot.mouseMove(plot_widget, pos=QPoint(400, 300))
    qtbot.wait(50)
    qtbot.mouseMove(plot_widget, pos=QPoint(270, 300))
    qtbot.wait(50)
    qtbot.mouseRelease(plot_widget, Qt.LeftButton, pos=QPoint(270, 300))
    qtbot.wait(50)
    qtbot.mouseMove(plot_widget, pos=QPoint(475, 300))
    qtbot.wait(50)
    qtbot.mousePress(plot_widget, Qt.LeftButton, pos=QPoint(475, 300))
    qtbot.wait(50)
    qtbot.mouseMove(plot_widget, pos=QPoint(480, 300))
    qtbot.wait(50)
    qtbot.mouseMove(plot_widget, pos=QPoint(650, 300))
    qtbot.wait(50)
    qtbot.mouseRelease(plot_widget, Qt.LeftButton, pos=QPoint(650, 300))
    qtbot.wait(50)
    
def test_MDC_Fitter(qtbot, monkeypatch):
    from tool.MDC_Fitter import main
    from PyQt5.QtWidgets import QMessageBox
    # 模擬 QMessageBox.information 自動回傳 QMessageBox.Ok
    monkeypatch.setattr(QMessageBox, 'information', lambda *args, **kwargs: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args, **kwargs: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, 'critical', lambda *args, **kwargs: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, 'question', lambda *args, **kwargs: QMessageBox.Yes)
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getOpenFileName', lambda *args, **kwargs: (os.path.join(os.path.dirname(__file__), 'simulated_R1_15.0_R2_0#id#0d758f03.h5'), ''))
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getSaveFileName', lambda *args, **kwargs: (os.path.join(os.path.dirname(__file__), 'simulated_R1_15.0_R2_0_mfit.npz'), ''))
    
    file = os.path.join(os.path.dirname(__file__), 'simulated_R1_15.0_R2_0#id#0d758f03.h5')
    win = main(file=file)
    qtbot.waitExposed(win)
    win.load_file()
    qtbot.wait(100)
    qtbot.keyClick(win, QtCore.Qt.Key_Right)
    qtbot.keyClick(win, QtCore.Qt.Key_Left)
    qtbot.wait(100)
    
    win.slider.setValue(200)
    win.fmcgl2()
    plot_widget = win.plot.viewport()
    center = plot_widget.rect().center()

    drag_bl1(qtbot, plot_widget)
    win.slider.setValue(520)
    win.fmcgl2()
    qtbot.wait(100)
    drag_bl2(qtbot, plot_widget)
    qtbot.wait(100)
    win.fmfall()
    qtbot.wait(2)
    
    win.fmreject()
    qtbot.wait(100)
    win.fmreject()
    qtbot.wait(100)
    win.fmaccept()
    qtbot.wait(100)
    
    qtbot.mouseClick(win.b_pos, Qt.LeftButton)
    qtbot.wait(100)
    qtbot.mouseClick(win.b_pos, Qt.LeftButton)
    qtbot.wait(100)
        
    qtbot.mouseMove(plot_widget, pos=QPoint(450, 300))
    qtbot.wait(50)
    qtbot.mousePress(plot_widget, Qt.LeftButton, pos=center)
    qtbot.wait(50)
    qtbot.mouseMove(plot_widget, pos=QPoint(400, 300))
    qtbot.wait(50)
    qtbot.mouseRelease(plot_widget, Qt.LeftButton, pos=QPoint(400, 300))
    qtbot.wait(50)
    qtbot.mouseMove(plot_widget, pos=QPoint(400, 300))
    qtbot.wait(50)
    qtbot.mousePress(plot_widget, Qt.LeftButton, pos=QPoint(400, 300))
    qtbot.wait(50)
    qtbot.mouseMove(plot_widget, pos=QPoint(450, 300))
    qtbot.wait(50)
    qtbot.mouseRelease(plot_widget, Qt.LeftButton, pos=center)
    qtbot.wait(50)
    qtbot.mouseMove(plot_widget, pos=QPoint(450, 300))
    qtbot.keyClick(win, QtCore.Qt.Key_Up)
    qtbot.wait(500)
    qtbot.keyClick(win, QtCore.Qt.Key_Down)
    qtbot.wait(500)
    qtbot.keyClick(win, QtCore.Qt.Key_Enter)
    qtbot.wait(500)
    qtbot.keyClick(win, QtCore.Qt.Key_Z, Qt.ControlModifier)
    qtbot.wait(500)
    qtbot.keyClick(win, QtCore.Qt.Key_Y, Qt.ControlModifier)
    qtbot.wait(500)
    
    
    win.mflind()
    qtbot.wait(100)
    win.mflind()
    qtbot.wait(100)
    win.mfrind()
    qtbot.wait(100)
    win.mfrind()
    qtbot.wait(100)
    
    win.fmpreview()
    qtbot.waitExposed(win.tg)
    win.fmpreview()
    qtbot.waitExposed(win.tg)
    win.fmresidual()
    win.fmarea()
    win.fmfwhm()
    win.fmimse()
    
    
    win.fmend()
    qtbot.waitExposed(win.g_exp)
    win.fmend(1)
    qtbot.waitExposed(win.g_exp)
    win.fmend(2)
    qtbot.waitExposed(win.g_exp)
    win.savemfit()
    qtbot.wait(100)
    
    
    win.slider.setValue(200)
    qtbot.wait(100)
    win.fmrmv(test=True)
    qtbot.wait(100)
    win.slider.setValue(520)
    qtbot.wait(100)
    win.fmrmv(test=True)
    qtbot.wait(2)
    
    win.toggle_grid(checked=True)
    qtbot.wait(100)
    win.toggle_grid(checked=False)
    qtbot.wait(100)
    win.toggle_histogram()
    qtbot.wait(100)
    win.reset_histogram()
    qtbot.wait(100)
    win.auto_level_histogram()
    qtbot.wait(100)
    
    win.help_window()
    
    win.show_shortcuts()
    win.close()
    
    os.system(f"del {os.path.join(os.path.dirname(__file__), 'mfit.npz')}")  # 删除保存的拟合结果文件
    win = main(file=file)
    qtbot.waitExposed(win)
    win.close()

def test_DataViewer(qtbot, monkeypatch):
    from tool.DataViewer import SliceBrowser, get_hwnd, disp_zarr_save, load_zarr
    from PyQt5.QtWidgets import QMessageBox
    # 模擬 QMessageBox.information 自動回傳 QMessageBox.Ok
    monkeypatch.setattr(QMessageBox, 'information', lambda *args, **kwargs: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args, **kwargs: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, 'critical', lambda *args, **kwargs: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, 'question', lambda *args, **kwargs: QMessageBox.Yes)
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getSaveFileName', lambda *args, **kwargs: (os.path.join(os.path.dirname(__file__), 'test_save.zarr'), ''))
    
    hwnd = get_hwnd()
    assert isinstance(hwnd, int)
    path = os.path.join(os.path.dirname(__file__), 'test_cube.zarr')
    output = os.path.join(os.path.dirname(__file__), 'test_cube_disp.zarr')
    mode, shape, xmin, xmax, ymin, ymax, E = load_zarr(path)
    disp_zarr_save(path, output, shape, max_val=10750)
    
    path = os.path.join(os.path.dirname(__file__), 'test_cube.zarr')
    win = SliceBrowser(path=path, hwnd=hwnd)
    qtbot.waitExposed(win)
    win.on_radio_button_changed("E")
    qtbot.wait(100)
    win.on_radio_button_changed("kx")
    qtbot.wait(100)
    win.export_slice()
    qtbot.wait(100)
    win.on_radio_button_changed("ky")
    qtbot.wait(100)
    win.export_slice()
    qtbot.wait(100)
    win.rotate_slider.setValue(90)
    win.sync_rotate_edit()
    qtbot.wait(100)
    win.apply_rotation()
    shutil.rmtree(os.path.join(os.path.dirname(__file__), 'test_save.zarr'), ignore_errors=True)
    win.export_slice()
    win.on_radio_button_changed("kx")
    qtbot.wait(100)
    shutil.rmtree(os.path.join(os.path.dirname(__file__), 'test_save.zarr'), ignore_errors=True)
    win.export_slice()
    shutil.rmtree(os.path.join(os.path.dirname(__file__), 'test_save.zarr'), ignore_errors=True)
    win.save_as_zarr_disp()
    win.close()
    
    win = SliceBrowser(os.path.join(os.path.dirname(__file__), 'test_save.zarr'), hwnd)
    qtbot.waitExposed(win)
    win.bin_e_spin.setValue(5)
    win.on_bin_change()
    win.on_radio_button_changed("kx")
    win.on_bin_change()
    win.on_radio_button_changed("ky")
    win.on_bin_change()
    win.close()
    
    shutil.rmtree(os.path.join(path, '__disp__.zarr'), ignore_errors=True)
    win = SliceBrowser(path=path, hwnd=hwnd)
    qtbot.waitExposed(win)
    win.close()

def test_RawDataViewer(qtbot, monkeypatch):
    from tool.RawDataViewer import main
    from PyQt5.QtWidgets import QMessageBox
    # 模擬 QMessageBox.information 自動回傳 QMessageBox.Ok
    monkeypatch.setattr(QMessageBox, 'information', lambda *args, **kwargs: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args, **kwargs: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, 'critical', lambda *args, **kwargs: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, 'question', lambda *args, **kwargs: QMessageBox.Yes)
    monkeypatch.setattr(QtWidgets.QDialog, 'exec_', lambda self: QMessageBox.Ok)
    path = []
    path.append(os.path.join(os.path.dirname(__file__), 'simulated_R1_15.0_R2_0#id#0d758f03.h5'))
    path.append(os.path.join(os.path.dirname(__file__), 'UPSPE20_2_test_1559#id#3cf2122d.json'))
    lfs = loadfiles(path, name='internal')
    win = main(lfs, test=True)
    qtbot.waitExposed(win)
    event = QtCore.QEvent(QtCore.QEvent.MouseButtonPress)
    win.energy_mode(event)
    qtbot.wait(100)
    
    # 模擬鼠標移動到 plot widget
    plot_widget = win.plot.viewport()
    center = plot_widget.rect().center()
    plot_widgetx = win.plotx.viewport()
    centerx = plot_widgetx.rect().center()
    
    qtbot.mouseMove(plot_widget, pos=QPoint(250, 230))
    qtbot.wait(100)
    
    # 模擬鼠標移動到 plot widget
    qtbot.mouseMove(plot_widget, pos=center)
    qtbot.wait(100)
    
    # 模擬鼠標點擊
    qtbot.mouseClick(plot_widget, Qt.LeftButton, pos=center)
    qtbot.wait(100)
    
    # 模擬鼠標移動
    qtbot.mouseMove(plot_widgetx, pos=centerx)
    qtbot.wait(100)
    
    win.range_changed()
    qtbot.wait(100)
    
    
    # 模擬鼠標移動到 plot widget
    qtbot.mouseMove(plot_widget, pos=QPoint(250, 230))
    qtbot.wait(100)
    
    # 模擬鼠標點擊
    qtbot.mouseClick(plot_widget, Qt.LeftButton, pos=QPoint(250, 230))
    qtbot.wait(100)
    
    # 模擬鼠標移動到 plot widget
    qtbot.mouseMove(plot_widget, pos=QPoint(255, 235))
    qtbot.wait(100)
    
    # 模擬鼠標移動
    qtbot.mouseMove(plot_widgetx, pos=centerx)
    qtbot.wait(100)
    
    win.range_changed()
    qtbot.wait(100)
    
    wheel_event = QWheelEvent(
        QPointF(center),  # pos (滑鼠位置)
        QPointF(center),  # globalPos (全局位置)
        QPoint(0, 0),     # pixelDelta
        QPoint(0, 120),   # angleDelta (正值向上滾動,負值向下滾動)
        Qt.NoButton,      # buttons
        Qt.NoModifier,    # modifiers
        Qt.ScrollUpdate,  # phase
        False            # inverted
    )
    win.text_display.wheelEvent(wheel_event)
    qtbot.wait(100)
    
    wheel_event = QWheelEvent(
        QPointF(center),  # pos (滑鼠位置)
        QPointF(center),  # globalPos (全局位置)
        QPoint(0, 0),     # pixelDelta
        QPoint(0, -120),   # angleDelta (正值向上滾動,負值向下滾動)
        Qt.NoButton,      # buttons
        Qt.NoModifier,    # modifiers
        Qt.ScrollUpdate,  # phase
        False            # inverted
    )
    win.text_display.wheelEvent(wheel_event)
    qtbot.wait(100)

    win.load_file(path)
    qtbot.wait(100)
