import pytest
import os, sys
from typing import Literal, override
tdir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')
sys.path.append(tdir)
sys.path.append(os.path.dirname(tdir))
from MDC_cut_utility import *
from tool.loader import loadfiles, tkDnD_loader, load_h5
from tool.spectrogram import spectrogram, lfs_exp_casa
from tool.util import app_param
from tool.SO_Fitter import SO_Fitter
from tool.CEC import CEC, call_cec
from tool.VolumeSlicer import wait

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

def test_lfs_exp_casa():
    path = []
    path.append(os.path.join(os.path.dirname(__file__), 'simulated_R1_15.0_R2_0#id#0d758f03.h5'))
    path.append(os.path.join(os.path.dirname(__file__), 'UPSPE20_2_test_1559#id#3cf2122d.json'))
    lfs = loadfiles(path, init=True, mode='eager', name='external', spectrogram=True)
    explfs = lfs_exp_casa(lfs)
    path = os.path.join(os.path.dirname(__file__), 'exp_casa.vms')
    explfs.export_casa(path)

def init_tempdir():
    import inspect, shutil
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
    import inspect
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
    
    ev, phi = odata[0].indexes.values()
    app_pars = app_param(hwnd=None, scale=1, dpi=96, bar_pos='bottom', g_mem=0.25)
    e_photon = float(odata[0].attrs['ExcitationEnergy'].removesuffix(' eV'))
    vs = VolumeSlicer(parent=frame, path=opath, volume=odataframe, x=phi, y=r1, ev=ev, e_photon=e_photon, g=g, app_pars=app_pars)
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
    vs.set_window()
    vs.set_slim()
    vs.symmetry()
    vs.symmetry_(6)
    vs.r1_offset = 15.25 # for test boost the speed
    set_entry_value(vs.entry_r1_offset, str(vs.r1_offset))
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
    vs.change_mode()  # back to real mode

def test_CEC(tk_environment):
    g, frame = tk_environment
    from tool.MDC_Fitter import get_file_from_github
    from MDC_cut_utility import file_walk
    import time
    app_pars = app_param(hwnd=None, scale=1, dpi=96, bar_pos='bottom', g_mem=0.25)
    tg = wait(g, app_pars)
    tg.text('Preparing sample data...')
    path = rf"simulated_R1_15.0_R2_0#id#0d758f03.h5"
    tpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test_data_temp', rf"simulated_R1_15.0_R2_0#id#0d758f03.h5")
    if os.path.exists(tpath)==False:
        get_file_from_github(r"https://github.com/alex20000910/main/blob/main/test_data/"+path, tpath)
    path = os.path.dirname(__file__)
    files = file_walk(path)
    for file in files:
        if 'simulated' not in file:
            files.remove(file)
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
    
def test_call_cec(tk_environment):
    g, frame = tk_environment
    app_pars = app_param(hwnd=None, scale=1, dpi=96, bar_pos='bottom', g_mem=0.25)
    lfs = loadfiles([os.path.join(os.path.dirname(__file__), 'test_cut.h5')], init=True, mode='eager', name='internal', cmap='viridis', app_pars=app_pars)
    call_cec(g, lfs)

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

def test_SO_Fitter(tk_environment):
    g, frame = tk_environment
    app_pars = app_param(hwnd=None, scale=1, dpi=96, bar_pos='bottom', g_mem=0.25)
    so_fitter = SO_Fitter(g, app_pars)
    so_fitter.on_closing()

def test_mfit_data():
    from tool.MDC_Fitter import mfit_data
    from tool.loader import mloader
    import queue
    st = queue.Queue()
    path = os.path.join(os.path.dirname(__file__), 'simulated_R1_15.0_R2_0#id#0d758f03.h5')
    data = load_h5(path)
    ev, phi = data.indexes.values()
    rdd = path
    cdir = os.path.dirname(os.path.dirname(__file__))
    lowlim = '0'
    ml = mloader(st, data, ev, phi, rdd, cdir, lowlim)
    ml.loadparam('0', '0', True, 1)
    ml.loadmfit_(os.path.join(cdir, 'tests', 'simulated_R1_14.0_R2_0_mfit.npz'), src='tests')
    mdata = mfit_data(cdir=cdir, src='tests')
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
