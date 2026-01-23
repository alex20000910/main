from MDC_cut_utility import *
from .util import app_param
from .SO_Fitter import SO_Fitter
import os, inspect
import sys, shutil
from multiprocessing import Pool
import threading
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog as fd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import Slider
from ctypes import windll
import h5py
import tqdm
import time
import gc
import cv2
import cpuinfo
import psutil
import zarr

def cut_job_y(args):
    if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))), 'stop_signal')):
        return -1
    i, angle, phi_offset, r1_offset, phi1_offset, r11_offset, self_x, self_volume, cdensity, xmax, xmin, ymax, ymin, z, x, self_z, self_y, ev, e_photon, sym = args
    g=VolumeSlicer()
    g.ev = ev
    g.y = self_y
    g.z = self_z
    g.cdensity = cdensity
    g.type = 'reciprocal'
    g.xmin = xmin
    g.xmax = xmax
    g.ymin = ymin
    g.ymax = ymax
    g.ox = self_x
    g.phi_offset = phi_offset
    g.r1_offset = r1_offset
    g.phi1_offset = phi1_offset
    g.r11_offset = r11_offset
    g.e_photon = e_photon
    g.angle = angle
    g.slice_index = i
    g.sym = sym
    surface = g.slice_data(i, angle, self_x, self_volume, x, z)
    angle, self_x, self_y, self_volume, ev = None, None, None, None, None
    td = surface[int(cdensity/(xmax-xmin)*(min(z)-xmin)):int(cdensity/(xmax-xmin)*(max(z)-xmin)), int(cdensity/(ymax-ymin)*(min(x)-ymin)):int(cdensity/(ymax-ymin)*(max(x)-ymin))]
    td = cv2.resize(td, (cdensity, td.shape[1]), interpolation=cv2.INTER_CUBIC)
    result = td.mean(axis=0)
    td = None
    path_cut = os.path.join('cut_temp_save', f'cut_{i}.zarr')
    path_cube = os.path.join('cube_temp_save', f'cube_{i}.zarr')
    zarr.save(path_cut, result)
    result = None
    zarr.save(path_cube, surface)
    surface = None
    return i

def cut_job_x(args):
    if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))), 'stop_signal')):
        return -1
    i, angle, phi_offset, r1_offset, phi1_offset, r11_offset, self_x, self_volume, cdensity, xmax, xmin, ymax, ymin, z, x, self_z, self_y, ev, e_photon, sym = args
    g=VolumeSlicer()
    g.ev = ev
    g.y = self_y
    g.z = self_z
    g.cdensity = cdensity
    g.type = 'reciprocal'
    g.xmin = xmin
    g.xmax = xmax
    g.ymin = ymin
    g.ymax = ymax
    g.ox = self_x
    g.phi_offset = phi_offset
    g.r1_offset = r1_offset
    g.phi1_offset = phi1_offset
    g.r11_offset = r11_offset
    g.e_photon = e_photon
    g.angle = angle
    g.slice_index = i
    g.sym = sym
    surface = g.slice_data(i, angle, self_x, self_volume, x, z)
    angle, self_x, self_y, self_volume, ev = None, None, None, None, None
    td = surface[int(cdensity/(xmax-xmin)*(min(z)-xmin)):int(cdensity/(xmax-xmin)*(max(z)-xmin)), int(cdensity/(ymax-ymin)*(min(x)-ymin)):int(cdensity/(ymax-ymin)*(max(x)-ymin))]
    td = cv2.resize(td, (td.shape[0], cdensity), interpolation=cv2.INTER_CUBIC)
    result = td.mean(axis=1)
    td = None
    path_cut = os.path.join('cut_temp_save', f'cut_{i}.zarr')
    path_cube = os.path.join('cube_temp_save', f'cube_{i}.zarr')
    zarr.save(path_cut, result)
    result = None
    zarr.save(path_cube, surface)
    surface = None
    return i


class g_cut_plot(tk.Toplevel):
    def __init__(self, master, data_cut, cx, cy, cdx, cdy, cdensity, ty, z, x, angle, phi_offset, r1_offset, phi1_offset, r11_offset, stop_event, pool, path, e_photon, slim, sym, xmin, xmax, ymin, ymax, cube, app_pars, test=False):
    # def __init__(self, master, data_cut, cx, cy, cdx, cdy, cdensity, ty, z, x, angle, phi_offset, r1_offset, phi1_offset, r11_offset, stop_event, pool, path, e_photon, slim, sym, xmin, xmax, ymin, ymax):
        super().__init__(master, background='white')
        self.cx_cut = cx
        self.cy_cut = cy
        self.cdx_cut = cdx
        self.cdy_cut = cdy
        self.phi_offset = phi_offset
        self.r1_offset = r1_offset
        self.phi1_offset = phi1_offset
        self.r11_offset = r11_offset
        self.slim_cut = slim
        self.sym_cut = sym
        self.data_cut = data_cut
        self.cdx = cdx
        self.cdy = cdy
        self.cdensity = cdensity
        self.ty = ty
        self.z = z
        self.x = x
        self.angle = angle
        self.e_photon = e_photon
        self.stop_event = stop_event
        self.pool = pool
        self.path = path
        self.cube = cube
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.app_pars = app_pars
        if test is False:
            self.save_cube()
        self.create_window()

    def size(self, s: int) -> int:
        return int(s * self.app_pars.scale)

    def create_window(self):
        self.title('Cut Plot')
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        fig = plt.Figure(figsize=(8*self.app_pars.scale, 8*self.app_pars.scale), constrained_layout=True)
        ax = fig.add_subplot(111)
        ax.set_xlabel(r'k ($\frac{2\pi}{\AA}$)', fontsize=self.size(16))
        ax.set_ylabel('Kinetic Energy (eV)', fontsize=self.size(16))

        if self.cdx <= self.cdy:
            tx = np.linspace(min(self.z), max(self.z), self.cdensity)
        else:
            tx = np.linspace(min(self.x), max(self.x), self.cdensity)
        x, y = np.meshgrid(tx, self.ty)
        ax.pcolormesh(x, y, self.data_cut, cmap='gray')

        fr_fig = tk.Frame(self, bg='white')
        fr_fig.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas = FigureCanvasTkAgg(fig, master=fr_fig)
        canvas.draw()
        self.x_cut = tx
        self.y_cut = self.ty
        self.angle_cut = self.angle
        self.phi_offset_cut = self.phi_offset
        self.r1_offset_cut = self.r1_offset
        self.phi1_offset_cut = self.phi1_offset
        self.r11_offset_cut = self.r11_offset

        fig, ax, x, y, tx = None, None, None, None, None

        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        toolbar = NavigationToolbar2Tk(canvas, fr_fig)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)

        fr_save = tk.Frame(self, bg='white')
        fr_save.pack(side=tk.TOP, anchor='center')
        
        save_button = tk.Button(fr_save, text='Save', command=self.save_cut, bg='white', font=('Arial', self.size(16), "bold"))
        save_button.pack(side=tk.LEFT)
        save_cube_button = tk.Button(fr_save, text='Save Data Cube', command=self.save_cube, bg='white', font=('Arial', self.size(16), "bold"))
        save_cube_button.pack(side=tk.LEFT)
        
        self.bind("<Return>", self.save_cut)
        self.focus_set()
        return
    
    @staticmethod
    def zarr_chunk_save(path: str, data: np.ndarray = np.empty((10, 100, 100)), attr_array: np.ndarray = np.empty((10, 100, 1))):
        """
        Using zarr to save large array data in chunks to avoid memory issues.
        Parameters:
            path (str): The file path to save the zarr data.
            data (np.ndarray): The large array data (3D) to be saved.
            attr_array (np.ndarray): The attribute array (3D) to be appended to data.
        """
        size = det_chunk(data.shape[1])
        step = int(min(size, data.shape[0]//1.1))
        merged_shape = data.shape[:-1] + (data.shape[-1] + attr_array.shape[-1],)
        # chunk_shape =  (step,) + (data.shape[1],) + (data.shape[2] + attr_array.shape[2],)
        output_zarr = zarr.open(path, mode='w',
                            shape=merged_shape,
                            dtype=np.float32)
        end = data.shape[0]
        max_val = 0
        tempdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        tempdir = os.path.dirname(tempdir)
        for i in range(0, end, step):
            ind = slice(i, min(i + step, end))
            odata = np.empty((abs(i-min(i + step, end)), data.shape[1], data.shape[2]+attr_array.shape[2]), dtype=np.float32)
            for j in range(i, min(i + step, end)):
                odata[j-i,:,:-1] = zarr.open_array(os.path.join(tempdir, 'cube_temp_save', f'cube_{j}.zarr'))[:]
                if np.max(odata[j-i,:,:-1]) > max_val:
                    max_val = np.max(odata[j-i,:,:-1])
            odata[:, :2, -1] = attr_array[ind, :2, 0]
            output_zarr[ind,...] = odata
            odata = None
            print('Progress: %.2f%%'%(min(i + step, end)/end*100))
            
        return max_val.astype(np.float16)
    
    @staticmethod
    def disp_zarr_save(input_path, output_path, data, max_val):
        zarr.save_group(output_path, ang=np.array([0, 0], dtype=np.float32))
        end = data.shape[0]
        size = det_chunk(data.shape[1], dtype=np.uint8)
        step = int(min(size, end//1.1))
        path = os.path.join(output_path, 'data')
        savez = zarr.open(path, mode='w', shape=data.shape, dtype=np.uint8)
        for i in range(0, end, step):
            ind = slice(i, min(i + step, end))
            savez[ind,...] = np.asarray(zarr.open(input_path, mode='r')[ind, :, :-1]/max_val*255, dtype=np.uint8)
            print('Progress: %.2f%%'%(min(i + step, end)/end*100))
    
    def save_zarr(self, path=None, data=None, xmin=None, xmax=None, ymin=None, ymax=None, ev=None):
        attr_array = np.zeros((data.shape[0], data.shape[1], 1), dtype=np.float32)  # Example attribute array
        attr_array[:, 0, 0] = ev
        attr_array[0, 1, 0] = xmin
        attr_array[1, 1, 0] = xmax
        attr_array[2, 1, 0] = ymin
        attr_array[3, 1, 0] = ymax
        # current_mem = psutil.virtual_memory().available
        # zdata = np.append(data, attr_array, axis=2)
        # zarr.save(path, zdata)
        if self.app_pars:
            windll.user32.ShowWindow(self.app_pars.hwnd, 9)
            windll.user32.SetForegroundWindow(self.app_pars.hwnd)
        max_val = self.zarr_chunk_save(path, data=data, attr_array=attr_array)
        for name in os.listdir(path):
            item_path = os.path.join(path, name)
            if os.path.isfile(item_path):
                os.system(f'attrib +h +s "{item_path}"')
            elif os.path.isdir(item_path):
                os.system(f'attrib +h +s "{item_path}"')
        disp_path = os.path.join(path, '__disp__.zarr')
        self.disp_zarr_save(path, disp_path, data, max_val)
        os.system(f'attrib +h +s "{disp_path}"')
        for name in os.listdir(disp_path):
            item_path = os.path.join(disp_path, name)
            if os.path.isfile(item_path):
                os.system(f'attrib +h +s "{item_path}"')
            elif os.path.isdir(item_path):
                os.system(f'attrib +h +s "{item_path}"')

    def save_cube(self, event=None, path=None):
        try:
            if not path:
                path = fd.asksaveasfilename(title="Save Data Cube as Zarr Folder", filetypes=(("Zarr folders", "*.zarr"),), initialdir=self.path[0], initialfile='data_cube', defaultextension=".zarr")
            if not path:
                print('Save operation cancelled')
            else:
                self.save_zarr(path, data=self.cube, xmin=self.xmin, xmax=self.xmax, ymin=self.ymin, ymax=self.ymax, ev=self.ty)
                # self.save_zarr(path, xmin=self.xmin, xmax=self.xmax, ymin=self.ymin, ymax=self.ymax, ev=self.ty)
                print(f'Data cube saved to {path}')
        except Exception as e:
            print(f"An error occurred while saving the data cube: {e}")
        return

    def save_cut(self, event=None, path=None):
        try:
            if not path:
                path = fd.asksaveasfilename(title="Save as", filetypes=(("HDF5 files", "*.h5"), ("NPZ files", "*.npz"),), initialdir=self.path[0], initialfile='data_cut', defaultextension=".h5")
            if not path:
                print('Save operation cancelled')
                return
            elif path.split('.')[-1] == 'npz':
                np.savez(path, path=self.path, data=self.data_cut, x=self.x_cut, y=self.y_cut, angle=self.angle_cut, cx=self.cx_cut, cy=self.cy_cut, cdx=self.cdx_cut, cdy=self.cdy_cut, phi_offset=self.phi_offset_cut, r1_offset=self.r1_offset_cut, phi1_offset=self.phi1_offset_cut, r11_offset=self.r11_offset_cut, e_photon=self.e_photon, slim=self.slim_cut, sym=self.sym_cut, desc=["Sliced data"])
            elif path.split('.')[-1] == 'h5':
                self.saveh5(path, path=self.path, data=self.data_cut, x=self.x_cut, y=self.y_cut, angle=self.angle_cut, cx=self.cx_cut, cy=self.cy_cut, cdx=self.cdx_cut, cdy=self.cdy_cut, phi_offset=self.phi_offset_cut, r1_offset=self.r1_offset_cut, phi1_offset=self.phi1_offset_cut, r11_offset=self.r11_offset_cut, e_photon=self.e_photon, slim=self.slim_cut, sym=self.sym_cut, desc=["Sliced data"])
            print('Data saved to %s'%path)
        except Exception as e:
            print(f"An error occurred: {e}")
        return

    def on_closing(self):
        self.cube, self.data_cut = None, None
        self.stop_event.clear()
        if self.pool:
            self.pool.terminate()
            print('Pool Terminated')
            self.pool.join()
            print('Pool Joined')
        self.destroy()
        clear(self)
        gc.collect()
        return
    
    def saveh5(self, tpath, path, data, x, y, angle, cx, cy, cdx, cdy, phi_offset, r1_offset, phi1_offset, r11_offset, e_photon, slim, sym, desc):
        with h5py.File(tpath, 'w') as f:
            xsize = np.array([len(y)], dtype=int)
            f.create_dataset('Data/XSize/Value', data=xsize, dtype=int)
            ysize = np.array([len(x)], dtype=int)
            f.create_dataset('Data/YSize/Value', data=ysize, dtype=int)
            
            acquisition = [bytes('VolumeSlicer', 'utf-8')]
            acquisition = np.array(acquisition, dtype=h5py.special_dtype(vlen=str))
            f.create_dataset('Region/Acquisition', data=acquisition, dtype=h5py.special_dtype(vlen=str))
            center_energy = np.array([(y[-1]+y[0])/2], dtype=float)
            f.create_dataset('Region/CenterEnergy/Value', data=center_energy, dtype=float)
            description = np.array([bytes(desc[0], 'utf-8')], dtype=h5py.special_dtype(vlen=str))
            f.create_dataset('Region/Description', data=description, dtype=h5py.special_dtype(vlen=str))
            dwell = np.array([bytes('Unknown', 'utf-8')], dtype=h5py.special_dtype(vlen=str))
            f.create_dataset('Region/Dwell/Value', data=dwell, dtype=h5py.special_dtype(vlen=str))
            
            energy_mode = [bytes('Kinetic', 'utf-8')]
            energy_mode = np.array(energy_mode, dtype=h5py.special_dtype(vlen=str))
            f.create_dataset('Region/EnergyMode', data=energy_mode, dtype=h5py.special_dtype(vlen=str))
            excitation_energy = np.array([e_photon], dtype=float)
            f.create_dataset('Region/ExcitationEnergy/Value', data=excitation_energy, dtype=float)
            high_energy = np.array([y[-1]], dtype=float)
            f.create_dataset('Region/HighEnergy/Value', data=high_energy, dtype=float)
            iterations = np.array([bytes('Unknown', 'utf-8')], dtype=h5py.special_dtype(vlen=str))
            f.create_dataset('Region/Iterations/Value', data=iterations, dtype=h5py.special_dtype(vlen=str))
            
            lens_mode = [bytes('Angular', 'utf-8')]
            lens_mode = np.array(lens_mode, dtype=h5py.special_dtype(vlen=str))
            f.create_dataset('Region/LensMode', data=lens_mode, dtype=h5py.special_dtype(vlen=str))
            low_energy = np.array([y[0]], dtype=float)
            f.create_dataset('Region/LowEnergy/Value', data=low_energy, dtype=float)
            name = np.array([bytes(os.path.basename(tpath).removesuffix('.h5'), 'utf-8')], dtype=h5py.special_dtype(vlen=str))
            f.create_dataset('Region/Name', data=name, dtype=h5py.special_dtype(vlen=str))
            y_scale_max = np.array([x[-1]], dtype=float)
            f.create_dataset('Region/YScaleMax/Value', data=y_scale_max, dtype=float)
            y_scale_min = np.array([x[0]], dtype=float)
            f.create_dataset('Region/YScaleMin/Value', data=y_scale_min, dtype=float)
            pass_energy = np.array([bytes('Unknown', 'utf-8')], dtype=h5py.special_dtype(vlen=str))
            f.create_dataset('Region/PassEnergy/Value', data=pass_energy, dtype=h5py.special_dtype(vlen=str))
            step = np.array([y[1]-y[0]], dtype=float)
            f.create_dataset('Region/Step/Value', data=step, dtype=float)
            
            slit = [bytes('Unknown', 'utf-8')]
            slit = np.array(slit, dtype=h5py.special_dtype(vlen=str))
            f.create_dataset('Region/Slit', data=slit, dtype=h5py.special_dtype(vlen=str))
            
            # additional data
            path = np.array([bytes(i, 'utf-8') for i in path], dtype=h5py.special_dtype(vlen=str))
            f.create_dataset('VolumeSlicer/path', data=path)
            angle = np.array([angle], dtype=float)
            f.create_dataset('VolumeSlicer/angle', data=angle)
            cx = np.array([cx], dtype=float)
            f.create_dataset('VolumeSlicer/cx', data=cx)
            cy = np.array([cy], dtype=float)
            f.create_dataset('VolumeSlicer/cy', data=cy)
            cdx = np.array([cdx], dtype=float)
            f.create_dataset('VolumeSlicer/cdx', data=cdx)
            cdy = np.array([cdy], dtype=float)
            f.create_dataset('VolumeSlicer/cdy', data=cdy)
            phi_offset = np.array([phi_offset], dtype=float)
            f.create_dataset('VolumeSlicer/phi_offset', data=phi_offset)
            r1_offset = np.array([r1_offset], dtype=float)
            f.create_dataset('VolumeSlicer/r1_offset', data=r1_offset)
            phi1_offset = np.array([phi1_offset], dtype=float)
            f.create_dataset('VolumeSlicer/phi1_offset', data=phi1_offset)
            r11_offset = np.array([r11_offset], dtype=float)
            f.create_dataset('VolumeSlicer/r11_offset', data=r11_offset)
            slim = np.array([slim], dtype=int)
            f.create_dataset('VolumeSlicer/slim', data=slim)
            sym = np.array([sym], dtype=int)
            f.create_dataset('VolumeSlicer/sym', data=sym)
            
            f.create_dataset('Spectrum', data=np.array(data))
        return


class wait(tk.Toplevel):
    def __init__(self, master, app_pars: app_param):
        self.g = master
        self.app_pars = app_pars
        super().__init__(master, background='white')
        set_center(self.g, self)
        self.title('Info')
        self.label_wait = tk.Label(self, bg='white', text='Please wait...', font=('Arial', self.size(16), "bold"))
        self.label_wait.pack(side=tk.TOP, pady=20)
        self.label_info = tk.Label(self, bg='white', text='', font=('Arial', self.size(14)))
        self.label_info.pack(side=tk.TOP, pady=20)
        self.grab_set()
        self.focus_set()
    
    def size(self, s: int) -> int:
        return int(s * self.app_pars.scale)
    
    def text(self, text):
        self.label_info.config(text=text)
        set_center(self.g, self)
        self.update()
    
    def done(self):
        self.grab_release()
        self.destroy()
        return

class VolumeSlicer(tk.Frame):
    def __init__(self, parent=None, path=None, volume=np.empty((5,5,5), dtype=np.float32), cmap='gray', x=None, y=None, z=None, ev=None, e_photon=21.2, density=600, g=None, app_pars: app_param=None):
        '''
        Args
        ------
            parent (tkinker-master object or None) : If the master given, the plot will be embedded in the master window.
            path (list) : `FileSequence.path`
            volume (np.ndarray) : shape=(r1, phi, ev)
            cmap (str) : Colormap to be used for the plot, default is 'gray'.
            x (array-like) : 1-D array representing phi values of the raw data in reciprocal mode and z values in real mode.
            y (array-like) : 1-D array representing r1 positions of the data cube in reciprocal mode and x positions in real mode.
            z (array-like) : 1-D array representing r2 positions of the data cube in reciprocal mode, or z positions in real mode.
            ev (array-like) : 1-D array representing kinetic energy values of the raw data.
            e_photon (float) : Photon energy in eV, default is 21.2 eV.
            density (int) : Density of the plot, default is 600.
            g (tkinter main window) : Used to define other tk.Toplevel windows, such as wait window and symmetry choosing window.
        
        Returns
        ------
            Frame (tk.Frame) : A tkinter Frame object containing the plot and controls.
        '''
        if parent is not None:
            super().__init__(parent, bg='white')
        if path is not None:
            self.path = path
        self.cmap=cmap
        # self.volume = volume    # data cube stored as a 3D numpy array
        self.slice_index = volume.shape[2] // 2
        self.angle = 0
        self.density = density
        self.ovolume = volume
        self.phi_offset = 48 # mm / 0 degree
        self.r1_offset = 11.5 # mm / -31 degree
        self.phi1_offset = 0
        self.r11_offset = 0
        self.e_photon = e_photon
        self.z = None
        # base dimensions
        
        # temperaly window range set
        self.m = 9.10938356e-31
        self.hbar = 1.0545718e-34
        self.e = 1.60217662e-19
        self.type = 'real'   # directly stack  'real', 'reciprocal'
        self.mode = 'normal'
        self.sym = 1
        self.g = g
        self.app_pars = app_pars
        self.test = False
        
        if x is not None and y is not None:
            # if __name__ != '__main__':
            #     plt.rcParams['font.family'] = 'Arial'
            #     plt.rcParams['font.size'] = int(plt.rcParams['font.size'] * self.app_pars.scale)
            #     plt.rcParams['lines.linewidth'] = plt.rcParams['lines.linewidth'] * self.app_pars.scale
            #     plt.rcParams['lines.markersize'] = plt.rcParams['lines.markersize'] * self.app_pars.scale
            #     plt.rcParams['figure.figsize'] = (plt.rcParams['figure.figsize'][0] * self.app_pars.scale, plt.rcParams['figure.figsize'][1] * self.app_pars.scale)
            self.ox = np.float32(x)
            self.y = np.float32(y)
            if z is not None:
                self.z = np.float32(z)
            self.ev = np.float64(ev)
            self.slim = [0, 493]    # init phi slice range -10~10 degree or -2.5~2.5 mm
            # Create a figure and axis
            self.fig = plt.Figure(figsize=(9*self.app_pars.scale, 9*self.app_pars.scale),constrained_layout=True)
            self.ax = self.fig.add_subplot(111)
            self.ax.set_aspect('equal')
            # self.ax.set_xticks([])
            # self.ax.set_yticks([])
            self.fig.subplots_adjust(bottom=0.25)
            
            self.fig_region = plt.Figure(figsize=(4*self.app_pars.scale, 4*self.app_pars.scale),constrained_layout=True)
            self.ax_region = self.fig_region.add_subplot(111)
            self.fig_region.subplots_adjust(bottom=0.25)
            
            if self.type == 'real':
                self.ax.set_xlabel('x (mm)', fontsize=self.size(16))
                self.ax.set_ylabel('z (mm)', fontsize=self.size(16))
                if z is not None:
                    self.xmin = np.min(np.min(x)+np.min(z))
                    self.xmax = np.max(np.max(x)+np.max(z))
                else:
                    self.xmin = np.min(x)
                    self.xmax = np.max(x)
                self.ymin = np.min(y)
                self.ymax = np.max(y)
                if self.xmin+self.xmax > 2*self.phi_offset:
                    self.xmin = self.phi_offset-(self.xmax-self.phi_offset)
                if self.xmax+self.xmin < 2*self.phi_offset:
                    self.xmax = self.phi_offset-(self.xmin-self.phi_offset)
                if self.ymin+self.ymax > 2*self.r1_offset:
                    self.ymin = self.r1_offset-(self.ymax-self.r1_offset)
                if self.ymax+self.ymin < 2*self.r1_offset:
                    self.ymax = self.r1_offset-(self.ymin-self.r1_offset)
            elif self.type == 'reciprocal':
                self.ax.set_xlabel(r'kx ($\frac{2\pi}{\AA}$)', fontsize=self.size(20))
                self.ax.set_ylabel(r'ky ($\frac{2\pi}{\AA}$)', fontsize=self.size(20))
                self.set_xy_lim()
            
            self.fl_show = False
            # self.interpolate_slice(self.slice_index)
            self.surface = np.zeros((self.density, self.density), dtype=np.float32)
            self.img = self.ax.imshow(self.surface, cmap=cmap, extent=[-1, 1, -1, 1], origin='lower')
            self.hl, = self.ax.plot([0, 0], [0, 0], color='green', linestyle='--')
            self.vl, = self.ax.plot([0, 0], [0, 0], color='green', linestyle='--')
            self.cut_l, = self.ax.plot([0, 0], [0, 0], color='red', linestyle='-')
            self.hl.set_data([],[])
            self.vl.set_data([],[])
            self.cut_l.set_data([], [])
            
            if parent is not None:
                self.fl_show = True     #flag to allow self.label_info window updating
                self.fit_so = None
                
                frame1 = tk.Frame(self, bg='white')
                frame1.grid(row=0, column=0)
                
                # Create a canvas and add it to the frame
                self.canvas = FigureCanvasTkAgg(self.fig, master=frame1)
                self.canvas.draw()
                self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
                # self.canvas.get_tk_widget().grid(row=0, column=1)
                
                # Create a toolbar and add it to the frame
                self.toolbar = NavigationToolbar2Tk(self.canvas, frame1)
                self.toolbar.update()
                self.toolbar.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)
                # self.toolbar.grid(row=1, column=0)
                
                frame2 = tk.Frame(self, bg='white')
                frame2.grid(row=0, column=1)
                
                frame_mode = tk.Frame(frame2, bg='white')
                frame_mode.pack(side=tk.TOP)
                self.b_mode = tk.Button(frame_mode, text='Transmission Mode', command=self.change_mode, bg='white', font=('Arial', self.size(16), "bold"))
                self.b_mode.pack(side=tk.LEFT)
                label_d = tk.Label(frame_mode, text='Density:', bg='white', font=('Arial', self.size(16), "bold")) 
                label_d.pack(side=tk.LEFT)
                self.entry_d = tk.Entry(frame_mode, bg='white', font=('Arial', self.size(16), "bold"))
                self.entry_d.pack(side=tk.LEFT)
                self.entry_d.insert(0, str(self.density))
                self.b_d = tk.Button(frame_mode, text='Set Density', command=self.set_density, bg='white', font=('Arial', self.size(16), "bold"))
                self.b_d.pack(side=tk.LEFT)
                
                
                frame_entry1 = tk.Frame(frame2, bg='white')
                frame_entry1.pack(side=tk.TOP)
                label_info = tk.Label(frame_entry1, text="Set Slit Slice Range (0-493 for initial range)", bg='white', font=('Arial', self.size(14), "bold"))
                label_info.pack(side=tk.TOP)
                
                # Create entries and button to set self.slim
                label_min = tk.Label(frame_entry1, text="Min:", bg='white', font=('Arial', self.size(14), "bold"))
                label_min.pack(side=tk.LEFT)
                self.entry_min = tk.Entry(frame_entry1, bg='white', font=('Arial', self.size(14), "bold"))
                self.entry_min.pack(side=tk.LEFT)
                self.entry_min.insert(0, str(self.slim[0]))

                label_max = tk.Label(frame_entry1, text="Max:", bg='white', font=('Arial', self.size(14), "bold"))
                label_max.pack(side=tk.LEFT)
                self.entry_max = tk.Entry(frame_entry1, bg='white', font=('Arial', self.size(14), "bold"))
                self.entry_max.pack(side=tk.LEFT)
                self.entry_max.insert(0, str(self.slim[1]))

                self.set_slim_button = tk.Button(frame_entry1, text="Set Limit", command=self.set_slim, font=('Arial', self.size(14), "bold"), bg='white')
                self.set_slim_button.pack(side=tk.LEFT)
                
                frame_entry2 = tk.Frame(frame2, bg='white')
                frame_entry2.pack(side=tk.TOP)
                # Create labels and entries for window range
                label_xmin = tk.Label(frame_entry2, text="X Min:", bg='white', font=('Arial', self.size(14), "bold"))
                label_xmin.pack(side=tk.LEFT)
                self.entry_xmin = tk.Entry(frame_entry2, bg='white', font=('Arial', self.size(14), "bold"))
                self.entry_xmin.pack(side=tk.LEFT)
                self.entry_xmin.insert(0, str(self.ymin))
                self.entry_xmin.config(state='disabled')

                label_xmax = tk.Label(frame_entry2, text="X Max:", bg='white', font=('Arial', self.size(14), "bold"))
                label_xmax.pack(side=tk.LEFT)
                self.entry_xmax = tk.Entry(frame_entry2, bg='white', font=('Arial', self.size(14), "bold"))
                self.entry_xmax.pack(side=tk.LEFT)
                self.entry_xmax.insert(0, str(self.ymax))
                self.entry_xmax.config(state='disabled')

                frame_entry3 = tk.Frame(frame2, bg='white')
                frame_entry3.pack(side=tk.TOP)

                label_ymin = tk.Label(frame_entry3, text="Y Min:", bg='white', font=('Arial', self.size(14), "bold"))
                label_ymin.pack(side=tk.LEFT)
                self.entry_ymin = tk.Entry(frame_entry3, bg='white', font=('Arial', self.size(14), "bold"))
                self.entry_ymin.pack(side=tk.LEFT)
                self.entry_ymin.insert(0, str(self.xmin))
                self.entry_ymin.config(state='disabled')

                label_ymax = tk.Label(frame_entry3, text="Y Max:", bg='white', font=('Arial', self.size(14), "bold"))
                label_ymax.pack(side=tk.LEFT)
                self.entry_ymax = tk.Entry(frame_entry3, bg='white', font=('Arial', self.size(14), "bold"))
                self.entry_ymax.pack(side=tk.LEFT)
                self.entry_ymax.insert(0, str(self.xmax))
                self.entry_ymax.config(state='disabled')

                self.win_sym_frame = tk.Frame(frame2, bg='white')
                self.win_sym_frame.pack(side=tk.TOP)
                self.set_window_button = tk.Button(self.win_sym_frame, text="Set Window Range", command=self.set_window, font=('Arial', self.size(14), "bold"), bg='white')
                self.set_window_button.pack(side=tk.LEFT)
                self.set_sym_button = tk.Button(self.win_sym_frame, text="Symmetrical extend", command=self.symmetry, font=('Arial', self.size(14), "bold"), bg='white')
                
                self.frame_region = tk.Frame(frame2, bg='white')
                # self.frame_region.pack(side=tk.TOP)
                self.canvas_region = FigureCanvasTkAgg(self.fig_region, master=self.frame_region)
                self.canvas_region.draw()
                self.canvas_region.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
                self.pr_disp_region()
                
                frame_entry4 = tk.Frame(frame2, bg='white')
                frame_entry4.pack(side=tk.TOP)
                self.label_phi_offset = tk.Label(frame_entry4, text="Set Z Offset (mm):", bg='white', font=('Arial', self.size(14), "bold"))
                self.label_phi_offset.pack(side=tk.LEFT)
                self.entry_phi_offset = tk.Entry(frame_entry4, bg='white', font=('Arial', self.size(14), "bold"), state='normal')
                self.entry_phi_offset.pack(side=tk.LEFT)
                self.entry_phi_offset.insert(0, str(self.phi_offset))
                
                self.frame_entry5 = tk.Frame(frame2, bg='white')
                self.frame_entry5.pack(side=tk.TOP)
                self.label_r1_offset = tk.Label(self.frame_entry5, text="Set X Offset (mm):", bg='white', font=('Arial', self.size(14), "bold"))
                self.label_r1_offset.pack(side=tk.LEFT)
                self.entry_r1_offset = tk.Entry(self.frame_entry5, bg='white', font=('Arial', self.size(14), "bold"), state='normal')
                self.entry_r1_offset.pack(side=tk.LEFT)
                self.entry_r1_offset.insert(0, str(self.r1_offset))
                
                self.frame_entry6 = tk.Frame(frame2, bg='white')
                self.frame_entry6.pack(side=tk.TOP)
                self.label_phi1_offset = tk.Label(self.frame_entry6, text="Set Sample Phi Offset (deg):", bg='white', font=('Arial', self.size(14), "bold"))
                self.label_phi1_offset.pack(side=tk.LEFT)
                self.entry_phi1_offset = tk.Entry(self.frame_entry6, bg='white', font=('Arial', self.size(14), "bold"), state='normal')
                self.entry_phi1_offset.pack(side=tk.LEFT)
                self.entry_phi1_offset.insert(0, str(self.phi1_offset))
                
                self.frame_entry7 = tk.Frame(frame2, bg='white')
                self.frame_entry7.pack(side=tk.TOP)
                self.label_r11_offset = tk.Label(self.frame_entry7, text="Set Sample R1 Offset (deg):", bg='white', font=('Arial', self.size(14), "bold"))
                self.label_r11_offset.pack(side=tk.LEFT)
                self.entry_r11_offset = tk.Entry(self.frame_entry7, bg='white', font=('Arial', self.size(14), "bold"), state='normal')
                self.entry_r11_offset.pack(side=tk.LEFT)
                self.entry_r11_offset.insert(0, str(self.r11_offset))
                
                self.fit_so_button = tk.Button(frame2, text="Fit Sample Offsets", command=self.fit_so_app, font=('Arial', self.size(14), "bold"), bg='white')
                self.fit_so_button.pack(side=tk.TOP)

                self.fig1 = plt.Figure(figsize=(5*self.app_pars.scale, 0.5*self.app_pars.scale),constrained_layout=True)
                self.ax_slider = self.fig1.add_axes([0.2, 0.6, 0.8, 0.3])
                self.slider = Slider(self.ax_slider, 'Energy', self.ev[0], self.ev[-1], valinit=self.ev[self.slice_index], valstep=self.ev[1]-self.ev[0])
                self.slider.on_changed(self.set_sl)

                self.ax_angle_slider = self.fig1.add_axes([0.2, 0.1, 0.8, 0.3])
                self.angle_slider = Slider(self.ax_angle_slider, 'Angle', 0, 360, valinit=self.angle, valstep=0.001)
                self.angle_slider.on_changed(self.set_angle_sl)                
                
                self.ea_frame = tk.Frame(frame2, bg='white')
                self.ea_frame.pack(side=tk.TOP, after=self.fit_so_button)
                self.ea_text_frame = tk.Frame(self.ea_frame, bg='white')
                self.ea_text_frame.pack(side=tk.LEFT)
                self.text_e = tk.StringVar()
                self.text_e.set(str(f'%.3f'%self.ev[self.slice_index]))
                self.text_e.trace_add('write', self.set_tx)
                self.text = tk.Entry(self.ea_text_frame, bg='white', textvariable=self.text_e, font=('Arial', self.size(12), "bold"), state='normal', width=7).pack(side=tk.TOP)
                self.text_a = tk.StringVar()
                self.text_a.set(str(self.angle))
                self.text_a.trace_add('write', self.set_angle_tx)
                self.text_ang = tk.Entry(self.ea_text_frame, bg='white', textvariable=self.text_a, font=('Arial', self.size(12), "bold"), state='normal', width=7).pack(side=tk.TOP)
                
                self.canvas1 = FigureCanvasTkAgg(self.fig1, master=self.ea_frame)
                self.canvas1.draw()
                self.canvas1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1, before=self.ea_text_frame)
                
                self.frame_cut_xy = tk.Frame(frame2, bg='white')
                frame_xy = tk.Frame(self.frame_cut_xy, bg='white')
                frame_xy.pack(side=tk.TOP)
                self.cut_xy_x_label = tk.Label(frame_xy, text="kx:", bg='white', font=('Arial', self.size(14), "bold"))
                self.cut_xy_x_label.pack(side=tk.LEFT)
                self.cut_xy_x_entry = tk.Entry(frame_xy, bg='white', font=('Arial', self.size(14), "bold"))
                self.cut_xy_x_entry.pack(side=tk.LEFT)
                self.cut_xy_x_entry.insert(0, '0')
                self.cut_xy_y_label = tk.Label(frame_xy, text="ky:", bg='white', font=('Arial', self.size(14), "bold"))
                self.cut_xy_y_label.pack(side=tk.LEFT)
                self.cut_xy_y_entry = tk.Entry(frame_xy, bg='white', font=('Arial', self.size(14), "bold"))
                self.cut_xy_y_entry.pack(side=tk.LEFT)
                self.cut_xy_y_entry.insert(0, '0')
                
                frame_dxy = tk.Frame(self.frame_cut_xy, bg='white')
                frame_dxy.pack(side=tk.TOP)
                self.cut_xy_dx_label = tk.Label(frame_dxy, text="kx bin:", bg='white', font=('Arial', self.size(14), "bold"))
                self.cut_xy_dx_label.pack(side=tk.LEFT)
                self.cut_xy_dx_entry = tk.Entry(frame_dxy, bg='white', font=('Arial', self.size(14), "bold"))
                self.cut_xy_dx_entry.pack(side=tk.LEFT)
                self.cut_xy_dx_entry.insert(0, '0.05')
                self.cut_xy_dy_label = tk.Label(frame_dxy, text="ky bin:", bg='white', font=('Arial', self.size(14), "bold"))
                self.cut_xy_dy_label.pack(side=tk.LEFT)
                self.cut_xy_dy_entry = tk.Entry(frame_dxy, bg='white', font=('Arial', self.size(14), "bold"))
                self.cut_xy_dy_entry.pack(side=tk.LEFT)
                self.cut_xy_dy_entry.insert(0, '0.4')
                
                frame_cut_button = tk.Frame(self.frame_cut_xy, bg='white')
                frame_cut_button.pack(side=tk.TOP,anchor='center')
                b_cut = tk.Button(frame_cut_button, text='Cut', command=self.cut_xy, bg='white', font=('Arial', self.size(14), "bold"))
                b_cut.pack(side=tk.LEFT)
                
                b_cut_plot = tk.Button(frame_cut_button, text='Export', command=self.pr_cut_plot, bg='white', font=('Arial', self.size(14), "bold"))
                b_cut_plot.pack(side=tk.RIGHT)
    
    def size(self, s: int) -> int:
        return int(s * self.app_pars.scale)
    
    def fit_so_app(self, *args):
        try:
            self.fit_so.lift()
        except TypeError:
            self.fit_so = SO_Fitter(self.g, self.app_pars)
        except AttributeError:
            self.fit_so = SO_Fitter(self.g, self.app_pars)
    
    def cal_r1_phi_offset(self, r2=None):
        if self.z is not None and r2 is not None:
            r11_offset, phi1_offset = self.rot(self.r11_offset, self.phi1_offset, angle=-(r2-self.z[0]))
        else:
            r11_offset, phi1_offset = self.rot(self.r11_offset, self.phi1_offset)
        r1_offset = self.r1_offset + r11_offset.astype(np.float32)
        phi_offset = self.phi_offset + phi1_offset.astype(np.float32)
        return r1_offset, phi_offset
    
    def symmetry(self):
        self.sym_g = RestrictedToplevel(self.g, background='white')
        self.sym_g.title('Symmetry')
        self.sym_g.resizable(False, False)
        tk.Button(self.sym_g, text='2-fold symmetry', command=lambda: self.symmetry_(2), bg='white', font=('Arial', self.size(16), "bold")).pack(side=tk.TOP, pady=5)
        tk.Button(self.sym_g, text='3-fold symmetry', command=lambda: self.symmetry_(3), bg='white', font=('Arial', self.size(16), "bold")).pack(side=tk.TOP, pady=5)
        tk.Button(self.sym_g, text='4-fold symmetry', command=lambda: self.symmetry_(4), bg='white', font=('Arial', self.size(16), "bold")).pack(side=tk.TOP, pady=5)
        tk.Button(self.sym_g, text='6-fold symmetry', command=lambda: self.symmetry_(6), bg='white', font=('Arial', self.size(16), "bold")).pack(side=tk.TOP, pady=5)
        set_center(self.g, self.sym_g, 50)
        self.sym_g.update()
        self.sym_g.limit_bind()
    
    def gen_sym(self, n):
        try:
            self.sym_g.destroy()
        except:
            pass
        if n != 1:
            self.sym = n
            osurface = self.surface.copy()
            tmin = np.min(osurface[osurface>0])
            osurface[osurface < tmin-tmin/3] = np.nan
            for i in range(n-1):
                surface = rotate(self.surface, 360//n*(i+1), self.surface.shape)
                surface[surface < tmin-tmin/3] = np.nan
                osurface = np.nanmean([osurface, surface], axis=0)
            self.surface = np.nan_to_num(osurface)
            rotated_surface = rotate(self.surface, -self.angle, self.surface.shape)
            self.img.set_data(rotated_surface)
            copy_to_clipboard(self.fig)
            self.canvas.draw()
            
    def symmetry_(self, n):
        self.gen_sym(n)
        
    def set_xy_lim(self):
        '''
        Input: self.x, self.y, self.r1_offset, self.phi_offset, self.e_photon
        Return: self.xmin, self.xmax, self.ymin, self.ymax
        '''
        if self.z is None: # for 1 cube
            r1_offset, phi_offset = self.cal_r1_phi_offset()
        elif self.z is not None: # for multiple cubes
            tr11o, tphi1o = 0, 0
            for r2 in self.z:
                r1_offset, phi_offset = self.cal_r1_phi_offset(r2)
                if abs(r1_offset) > abs(tr11o):
                    tr11o, tphi1o = r1_offset, phi_offset
            r1_offset, phi_offset = tr11o, tphi1o
        r1 = self.y - r1_offset
        phi = self.x - phi_offset
        tr1 = np.array([np.min(r1), np.max(r1), np.max(r1), np.min(r1)])
        tphi = np.array([np.min(phi), np.min(phi), np.max(phi), np.max(phi)])
        tx = np.sqrt(2*self.m*self.e*self.e_photon)/self.hbar*10**-10*np.sin(tr1/180*np.pi) * np.cos(tphi/180*np.pi)
        ty = np.sqrt(2*self.m*self.e*self.e_photon)/self.hbar*10**-10*np.sin(tphi/180*np.pi)
        r = np.max(np.sqrt(tx**2 + ty**2))
        self.xmin, self.xmax = -r, r
        self.ymin, self.ymax = -r, r
        
    def cut_xy(self, init=False):
        cx = np.float64(self.cut_xy_x_entry.get())
        cy = np.float64(self.cut_xy_y_entry.get())
        cdx = np.float64(self.cut_xy_dx_entry.get())
        cdy = np.float64(self.cut_xy_dy_entry.get())
        if cx-cdx/2 < self.ymin or cx+cdx/2 > self.ymax or cy-cdy/2 < self.xmin or cy+cdy/2 > self.xmax:
            if cx-cdx/2 < self.ymin:
                cdx = (cx-self.ymin)*2
            elif cx+cdx/2 > self.ymax:
                cdx = (self.ymax-cx)*2
            if cy-cdy/2 < self.xmin:
                cdy = (cy-self.xmin)*2
            elif cy+cdy/2 > self.xmax:
                cdy = (self.xmax-cy)*2
        self.cut_xy_dx_entry.delete(0, tk.END)
        self.cut_xy_dx_entry.insert(0, str(cdx))
        self.cut_xy_dy_entry.delete(0, tk.END)
        self.cut_xy_dy_entry.insert(0, str(cdy))
        self.cx = cx
        self.cy = cy
        self.cdx = cdx
        self.cdy = cdy
        if not init:
            self.cut_l.set_data([self.cx-self.cdx/2, self.cx+self.cdx/2, self.cx+self.cdx/2, self.cx-self.cdx/2, self.cx-self.cdx/2], [self.cy-self.cdy/2, self.cy-self.cdy/2, self.cy+self.cdy/2, self.cy+self.cdy/2, self.cy-self.cdy/2])
            self.canvas.draw()
    
    def k_map(self, data, density, xlim, ylim, kxlim, kylim, ev):
        kx_grid, ky_grid = np.meshgrid(
        np.linspace(kxlim[0], kxlim[1], int(density/180*(xlim[1]-xlim[0]))*4, dtype=np.float32),
        np.linspace(kylim[0], kylim[1], int(density/180*(xlim[1]-xlim[0]))*4, dtype=np.float32))
        k = np.float32(np.sqrt(2*self.m*self.e*ev)/self.hbar*1e-10)
        Phi_target = np.arcsin(np.clip(ky_grid / k, -1, 1)) * 180 / np.pi
        cos_phi = np.cos(np.deg2rad(Phi_target))
        cos_phi[cos_phi == 0] = np.float32(1e-8)
        R1_target = np.arcsin(np.clip(kx_grid / (k * cos_phi), -1, 1)) * 180 / np.pi
        kx_grid, ky_grid, cos_phi = None, None, None
        map_x = (R1_target - xlim[0]) / (xlim[1] - xlim[0]) * (data.shape[0] - 1)
        map_y = (Phi_target - ylim[0]) / (ylim[1] - ylim[0]) * (data.shape[1] - 1)
        valid_mask = (
            (R1_target >= xlim[0]) & (R1_target <= xlim[1]) &
            (Phi_target >= ylim[0]) & (Phi_target <= ylim[1]) &
            np.isfinite(map_x) & np.isfinite(map_y)
        )
        R1_target, Phi_target = None, None
        map_x[~np.isfinite(map_x)] = 0
        map_y[~np.isfinite(map_y)] = 0
        map_x = np.clip(map_x, 0, data.shape[0] - 1)
        map_y = np.clip(map_y, 0, data.shape[1] - 1)
        data = cv2.remap(data.T, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        map_x, map_y = None, None
        data[~valid_mask] = 0
        data = cv2.resize(data, (int(density/(self.xmax-self.xmin)*(kxlim[1]-kxlim[0])), int(density/(self.ymax-self.ymin)*(kylim[1]-kylim[0]))), interpolation=cv2.INTER_CUBIC)
        valid_mask = None
        # kx_grid, ky_grid, R1_target, Phi_target, map_x, map_y, cos_phi, valid_mask = None, None, None, None, None, None, None, None
        return data
    
    def combine_slice(self, data, xlim, ylim, r2=None, ev=None, step=0.25):  # complete
        '''
        Args:
        ---
        data: raw image (shape:(len(r1), len(phi)))
        xlim: [min, max]  x: r1
        ylim: [min, max]  y: phi
        step: width for only one r1 cube (default: 0.25 degree/mm)
        
        return:
        ---
        image
        shape: (density, density)
        '''
        xlim, ylim = sorted(xlim), sorted(ylim)
        fr2 = True
        if self.type == 'real':
            if int(self.cdensity/(self.ymax-self.ymin)*(xlim[1]-xlim[0])) ==0:
                xlim[1] += step
                xlim[0] -= step
            data = cv2.resize(data, (int(self.cdensity/(self.ymax-self.ymin)*(ylim[1]-ylim[0])), int(self.cdensity/(self.xmax-self.xmin)*(xlim[1]-xlim[0]))), interpolation=cv2.INTER_CUBIC)
            base = np.zeros((self.cdensity, self.cdensity), dtype=np.float32)
            base[0:data.shape[0], 0:data.shape[1]] = data
            data = np.roll(base.T, (int((ylim[0]-self.xmin)/(self.xmax-self.xmin)*self.cdensity), int((xlim[0]-self.ymin)/(self.ymax-self.ymin)*self.cdensity)), axis=(0, 1))
            base = None
            data = data
        elif self.type == 'reciprocal':
            if r2 is None:
                fr2=False
                r2, self.z = 0, [0, 0]
            if int(self.cdensity/180*(xlim[1]-xlim[0])) == 0: # at least 1 pixel
                d = 1/self.cdensity*180/2
                c = (xlim[0]+xlim[1])/2
                xlim = [float(c-d), float(c+d)]
                if self.cut_show:
                    print(f'Warning: R1-axis density is too low (R2=%.2f)'%r2)
                    print('in combine_slice')
            r1 = np.linspace(xlim[0], xlim[1], int(self.cdensity/180*(xlim[1]-xlim[0])+1))
            phi = np.linspace(ylim[0], ylim[1], int(self.cdensity/180*(ylim[1]-ylim[0])))
            # r1, phi = np.broadcast_arrays(r1, phi)
            x = np.sqrt(2*self.m*self.e*ev)/self.hbar*10**-10*np.sin(r1[:, None]/180*np.pi) * np.cos(phi[None, :]/180*np.pi)  # x: r1, y: phi, at r2=0
            y = np.sqrt(2*self.m*self.e*ev)/self.hbar*10**-10*np.sin(phi[None, :]/180*np.pi)
            r1, phi = None, None
            txlim, tylim = [np.min(x), np.max(x)], [np.min(y), np.max(y)]
            x, y = None, None
            ####### new method start
            data = self.k_map(data, self.cdensity, xlim, ylim, txlim, tylim, ev)
            ####################### new method end
            
            ############## original method start
            # tmax = np.max(data)
            # fig, ax = plt.subplots(dpi=150)
            # fig.patch.set_facecolor('black')
            # ax.set_facecolor('black')
            # data = cv2.resize(data, (int(self.cdensity/180*(ylim[1]-ylim[0])), int(self.cdensity/180*(xlim[1]-xlim[0])+1)), interpolation=cv2.INTER_CUBIC)
            # ax.pcolormesh(x, y, data.T, cmap='gray')    # compatible with r2=0, phi: yaxis, r1: xaxis in transmission mode
            # ax.set_position([0, 0, 1, 1])
            # ax.axis('off')
            # fig.canvas.draw()
            # image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            # image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            # plt.close()
            # del fig, ax
            # image_gray = cv2.cvtColor(image_from_plot, cv2.COLOR_RGB2GRAY)/255*tmax
            # image_gray = cv2.resize(image_gray[::-1,:], (data.shape[1],data.shape[0]), interpolation=cv2.INTER_CUBIC)
            # data = cv2.resize(image_gray, (int(self.cdensity/(self.xmax-self.xmin)*(txlim[1]-txlim[0])), int(self.cdensity/(self.ymax-self.ymin)*(tylim[1]-tylim[0]))), interpolation=cv2.INTER_CUBIC)
            ########### original method end
            base = np.zeros((self.cdensity, self.cdensity), dtype=np.float32)
            base[0:data.shape[0], 0:data.shape[1]] = data
            # del image_from_plot, image_gray, data
            data = None
            data = np.roll(base, (int((tylim[0]-self.ymin)/(self.ymax-self.ymin)*self.cdensity), int((txlim[0]-self.xmin)/(self.xmax-self.xmin)*self.cdensity)), axis=(0, 1))
            txlim, tylim = None, None
            # data = data[::-1, :]
            base = None
            data = rotate(data, r2-self.z[0], data.shape)
            
            if not fr2:
                self.z = None
        if self.z is not None and fr2==True:  # for multiple cubes need np.nanmean
            msk = data<self.tmin-self.tmin/3
            data[msk] = np.nan
            msk = None
            
        return data
    
    def slice_data(self, i, angle=0, self_x=None, self_volume=None, xlim=None, zlim=None):
        """
        Args
        ------
        self_x : phi-axis, slit limit
        self_volume : 3D numpy array, shape: (r1, phi, ev)
        angle : rotation angle
        xlim : cutting kx lim after rotation, [min, max]
        zlim : cutting ky lim after rotation, [min, max]
        """
        # def filter(ind, ii, r2=None):
        #     r1 = np.linspace(min(self.y[ind]), max(self.y[ind]), len(self.y[ind]))[:, None]
        #     phi = np.linspace(min(self_x), max(self_x), len(self_x))[None, :]
        #     r1, phi = np.broadcast_arrays(r1, phi)
        #     for i in range(self.sym):
        #         r1_offset, phi_offset = self.cal_r1_phi_offset()
        #         if r2 is None:
        #             r1, phi = self.rot(r1, phi, r1_offset, phi_offset, angle-360//self.sym*i)
        #         else:
        #             r1, phi = self.rot(r1, phi, r1_offset, phi_offset, angle-(r2-self.z[0])-360//self.sym*i)
        #         if i == 0:
        #             x = np.sqrt(2*self.m*self.e*self.ev[ii])/self.hbar*10**-10*np.sin(r1/180*np.pi) * np.cos(phi/180*np.pi)  # x: r1, y: phi, at r2=0
        #             y = np.sqrt(2*self.m*self.e*self.ev[ii])/self.hbar*10**-10*np.sin(phi/180*np.pi)
        #         else:
        #             x = np.append(x, np.sqrt(2*self.m*self.e*self.ev[ii])/self.hbar*10**-10*np.sin(r1/180*np.pi) * np.cos(phi/180*np.pi), axis=0)
        #             y = np.append(y, np.sqrt(2*self.m*self.e*self.ev[ii])/self.hbar*10**-10*np.sin(phi/180*np.pi), axis=0)
        #     ti=[]
        #     for i in range(r1.shape[1]):
        #         if any(xlim[0]<x[:,i]) and any(zlim[0]<y[:,i]) and any(x[:,i]<xlim[1]) and any(y[:,i]<zlim[1]):
        #             ti.append(i)
        #     if len(ti) != 0:
        #         if min(ti)>0:
        #             ti.insert(0, min(ti)-1)
        #         if max(ti)<len(self.y[ind])-1:
        #             ti.append(max(ti)+1)
        #     return ind[ti]
        self.cut_show = False
        if i == 100:
            self.cut_show = True
        xlim, zlim = sorted(xlim), sorted(zlim)
        if self.z is None: # for 1 cube
            # ind = filter(np.arange(len(self.y)), i)     #取消filter 篩選加速功能 一律算全範圍 保留完整Data Cube
            # if len(ind) != 0:       #取消filter 篩選加速功能 一律算全範圍 保留完整Data Cube
            #     surface = self.combine_slice(self_volume[ind, :], xlim = [min(self.y[ind])-r1_offset, max(self.y[ind])-r1_offset], ylim = [min(self_x)-phi_offset, max(self_x)-phi_offset], ev=self.ev[i])       #取消filter 篩選加速功能 一律算全範圍 保留完整Data Cube
            # else:
            r1_offset, phi_offset = self.cal_r1_phi_offset()
            surface = self.combine_slice(self_volume, xlim = [min(self.y)-r1_offset, max(self.y)-r1_offset], ylim = [min(self_x)-phi_offset, max(self_x)-phi_offset], ev=self.ev[i])
        elif self.z is not None: # for multiple cubes
            img = self_volume
            try:
                self.tmin = np.min(img[img>0])
                r2 = sorted(set(self.z))
                surface = np.full((self.cdensity, self.cdensity), np.nan, dtype=np.float32)
                for z in r2:
                    ind = np.array(np.argwhere(self.z==z), dtype=np.int64).flatten()
                    # ind = filter(ind, i, r2=z)        #取消filter 篩選加速功能 一律算全範圍 保留完整Data Cube
                    # if len(ind) != 0:     #取消filter 篩選加速功能 一律算全範圍 保留完整Data Cube
                    r1_offset, phi_offset = self.cal_r1_phi_offset(r2=z)
                    surface = np.nanmean([surface, self.combine_slice(self_volume[ind, :], xlim = [min(self.y[ind])-r1_offset, max(self.y[ind])-r1_offset], ylim = [min(self_x)-phi_offset, max(self_x)-phi_offset], r2=z, ev=self.ev[i])], axis=0)
                surface = np.nan_to_num(surface)
            except:
                surface = np.zeros((self.cdensity, self.cdensity), dtype=np.float32)
        img, self_volume, self_x = None, None, None
        surface = rotate(surface, -angle, surface.shape)
        osurface = surface.copy()
        try:
            tmin = np.min(osurface[osurface>0])
            osurface[osurface < tmin - tmin / 3] = np.nan
            for ii in range(self.sym-1):
                rotated_surface = rotate(surface, 360//self.sym*(ii+1), surface.shape)
                rotated_surface[rotated_surface < tmin-tmin/3] = np.nan
                osurface = np.nanmean([osurface, rotated_surface], axis=0)
                rotated_surface = None
            surface = None
            osurface[osurface < tmin-tmin/3] = np.nan
            surface = np.nan_to_num(osurface)
        except:
            surface = np.zeros((self.cdensity, self.cdensity), dtype=np.float32)
        osurface = None
        return surface
    
    def listen_for_stop_command(self):
        command = input()
        if command.strip().lower() == '':
            self.stop_event.set()
            tempdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
            tempdir = os.path.dirname(tempdir)
            with open(os.path.join(tempdir, 'stop_signal'), 'w') as f:
                f.write('stop')
            print("\033[35mStopping the process...\033[0m")
    
    def monitor(self):
        tempdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        tempdir = os.path.dirname(tempdir)
        path = os.path.join(tempdir, 'cube_temp_save')
        total = len(self.ev)
        pbar = tqdm.tqdm(total=len(self.ev), desc='Processing', colour='blue', file=sys.stdout)
        while not self.stop_event.is_set():
            time.sleep(0.5)
            files = os.listdir(path)
            completed = len(files)
            if completed > pbar.n:
                pbar.update(completed - pbar.n)
            if completed >= total:
                break
    
    def t_cut_job_y(self):
        angle = self.angle
        x = [self.cx-self.cdx/2, self.cx+self.cdx/2]
        z = [self.cy-self.cdy/2, self.cy+self.cdy/2]
        phi_offset = self.phi_offset
        r1_offset = self.r1_offset
        phi1_offset, r11_offset = self.phi1_offset, self.r11_offset
        self_x = self.ox[self.slim[0]:self.slim[1]+1]
        self_volume = self.ovolume[:, self.slim[0]:self.slim[1]+1, :]
        self.set_xy_lim()
        tempdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        tempdir = os.path.dirname(tempdir)
        try:
            os.chdir(os.path.join(tempdir))
            if os.path.exists(os.path.join(tempdir, 'stop_signal')):
                os.remove(os.path.join(tempdir, 'stop_signal'))
            if os.path.exists('cut_temp_save'):
                shutil.rmtree('cut_temp_save')
            os.mkdir('cut_temp_save')
            if os.path.exists('cube_temp_save'):
                shutil.rmtree('cube_temp_save')
            os.mkdir('cube_temp_save')
            with Pool(self.pool_size) as self.pool:
                args = [(i, angle, phi_offset, r1_offset, phi1_offset, r11_offset, self_x, self_volume[:, :, i], self.cdensity, self.xmax, self.xmin, self.ymax, self.ymin, z, x, self.z, self.y, self.ev, self.e_photon, self.sym) for i in range(len(self.ev))]
                # threading.Thread(target=self.monitor).start()
                # self.pool.map(cut_job_y, args)
                for i, result in enumerate(tqdm.tqdm(self.pool.imap(cut_job_y, args), total=len(self.ev), desc="Processing", file=sys.stdout, colour='blue')):
                    if self.stop_event.is_set():
                        break
                if not self.stop_event.is_set():
                    print("\n\033[32mPress \033[31m'Enter' \033[32mto coninue...\033[0m")
                args = None
        except Exception as e:
            args = None
            print('t_cut_job_y')
            print(f"An error occurred: {e}")
                    
    def t_cut_job_x(self):
        angle = self.angle
        x = [self.cx-self.cdx/2, self.cx+self.cdx/2]
        z = [self.cy-self.cdy/2, self.cy+self.cdy/2]
        phi_offset = self.phi_offset
        r1_offset = self.r1_offset
        phi1_offset, r11_offset = self.phi1_offset, self.r11_offset
        self_x = self.ox[self.slim[0]:self.slim[1]+1]
        self_volume = self.ovolume[:, self.slim[0]:self.slim[1]+1, :]
        self.set_xy_lim()
        tempdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        tempdir = os.path.dirname(tempdir)
        try:
            os.chdir(os.path.join(tempdir))
            if os.path.exists('cut_temp_save'):
                shutil.rmtree('cut_temp_save')
            os.mkdir('cut_temp_save')
            if os.path.exists('cube_temp_save'):
                shutil.rmtree('cube_temp_save')
            os.mkdir('cube_temp_save')
            with Pool(self.pool_size) as self.pool:
                args = [(i, angle, phi_offset, r1_offset, phi1_offset, r11_offset, self_x, self_volume[:, :, i], self.cdensity, self.xmax, self.xmin, self.ymax, self.ymin, z, x, self.z, self.y, self.ev, self.e_photon, self.sym) for i in range(len(self.ev))]
                # threading.Thread(target=self.monitor).start()
                # self.pool.map(cut_job_x, args)
                for i, result in enumerate(tqdm.tqdm(self.pool.imap(cut_job_x, args), total=len(self.ev), desc="Processing", file=sys.stdout, colour='blue')):
                    if self.stop_event.is_set():
                        break
                if not self.stop_event.is_set():
                    print("\n\033[32mPress \033[31m'Enter' \033[32mto coninue...\033[0m")
                args = None
        except Exception as e:
            args = None
            print('t_cut_job_x')
            print(f"An error occurred: {e}")
            
    def confirm_cut(self, *args):
        self.pr_cut_g.destroy()
        self.cut_plot()
    
    def pr_cut_plot(self):
        if self.sym != 1:
            self.pr_cut_g = RestrictedToplevel(self.g, background='white')
            self.pr_cut_g.title('Info')
            self.pr_cut_g.resizable(False, False)
            tk.Label(self.pr_cut_g, bg='white', text=f'Using {self.sym}-fold symmetry', font=('Arial', self.size(16), "bold")).pack(side=tk.TOP, padx=10, pady=10)
            tk.Button(self.pr_cut_g, text='OK', command=self.confirm_cut, bg='white', font=('Arial', self.size(16), "bold")).pack(side=tk.TOP, pady=10)
            self.pr_cut_g.bind("<Return>", self.confirm_cut)
            set_center(self.g, self.pr_cut_g, 100)
            self.pr_cut_g.focus_set()
            self.pr_cut_g.limit_bind()
        else:
            self.cut_plot()
            
    
    def det_core_num(self):
        '''
        Memory is used to calculate a safe amount of cores by concering the least used memory size per worker.
        The actual memory size that the pool would use may be a little bit larger (1x ~ 1.4x).
        '''
        use_core = max(1, int(psutil.cpu_count(logical=False)/4*3))
        mem_max = self.ovolume[:, self.slim[0]:self.slim[1]+1, :].nbytes/1024**3    # GB
        # print(mem_max, 'mem_max GB')
        mem = self.app_pars.g_mem+mem_max
        # print(mem, 'mem GB')
        current_mem = psutil.virtual_memory().available/1024**3
        num = current_mem/mem
        if num < use_core:
            use_core = int(num)
            print('\033[33mPlease note that the number of cores is set according to the memory limit for stability reasons.\033[0m')
        if use_core < 1:
            use_core = 1
            print('\033[33mLack of memory.\033[0m')
        self.pool_size = use_core
        print('\033[33mUsing \033[36m%d \033[33mcores\033[0m'%self.pool_size)
    
    def cut_plot(self):
        if self.app_pars:
            windll.user32.ShowWindow(self.app_pars.hwnd, 9)
            windll.user32.SetForegroundWindow(self.app_pars.hwnd)
        self.stop_event = threading.Event()
        t1 = threading.Thread(target=self.listen_for_stop_command, daemon=True)
        t1.start()
        self.set_xy_lim()
        self.cdensity = int((self.xmax-self.xmin)//2e-3)
        print('\nSampling Density: \033[31m500 per 2pi/Angstrom')
        print('\033[0mProcessing \033[32m%d x %d x %d \033[0msize data cube'%(self.cdensity, self.cdensity, len(self.ev)))
        print('\n\033[33mProcessor:\033[36m',cpuinfo.get_cpu_info()['brand_raw'])
        print('\033[33mPhysical CPU cores:\033[36m', psutil.cpu_count(logical=False))
        self.det_core_num()
        print('\033[0mPlease wait...\n')
        print('\nThe following shows the progress bar and the estimation of the processing time')
        angle = self.angle
        self.cx_cut = self.cx
        self.cy_cut = self.cy
        self.cdx_cut = self.cdx
        self.cdy_cut = self.cdy
        x = [self.cx-self.cdx/2, self.cx+self.cdx/2]
        z = [self.cy-self.cdy/2, self.cy+self.cdy/2]
        ty = self.ev
        self.data_cut = np.zeros((len(ty), self.cdensity), dtype=np.float32)
        # self.data_cube = np.zeros((len(ty), self.cdensity, self.cdensity), dtype=np.float32)
        self.data_cube = np.empty((len(ty), self.cdensity, self.cdensity), dtype=np.uint8)
        phi_offset = self.phi_offset
        r1_offset = self.r1_offset
        phi1_offset = self.phi1_offset
        r11_offset = self.r11_offset
        self.slim_cut = self.slim.copy()
        self.sym_cut = self.sym
        self_x = self.ox[self.slim[0]:self.slim[1]+1]   # -----stable version no multiprocessing-----
        self_volume = self.ovolume[:, self.slim[0]:self.slim[1]+1, :]   # -----stable version no multiprocessing-----
        if self.cdx<=self.cdy:  # cut along ky
            
            # -----stable version no multiprocessing-----
            # for i in range(len(y)):
            #     surface = self.slice_data(i, angle, phi_offset, r1_offset, self_x, self_volume)
            #     td = surface[int(self.cdensity/(self.xmax-self.xmin)*(min(z)-self.xmin)):int(self.cdensity/(self.xmax-self.xmin)*(max(z)-self.xmin)), int(self.cdensity/(self.ymax-self.ymin)*(min(x)-self.ymin)):int(self.cdensity/(self.ymax-self.ymin)*(max(x)-self.ymin))]
            #     del surface
            #     td = cv2.resize(td, (td.shape[0], self.cdensity), interpolation=cv2.INTER_CUBIC)
            #     data[i,:] = td.mean(axis=1)
            #     td = None
            # -----stable version no multiprocessing-----
            
            self.t = threading.Thread(target=self.t_cut_job_x, daemon=True)
            self.t.start()
        else:   # cut along kx
            self.t = threading.Thread(target=self.t_cut_job_y, daemon=True)
            self.t.start()
        print("\n\033[32m-----Press \033[31m'Enter' \033[32mto terminate the pool-----\033[0m\n")
        t1.join()
        self.t.join()
        print('Proccess finished\nWait a moment...')
        tempdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        tempdir = os.path.dirname(tempdir)
        for i in range(len(ty)):
            try:
                self.data_cut[i] = zarr.open_array(os.path.join(tempdir, 'cut_temp_save', f'cut_{i}.zarr'))
                # self.data_cube[i] = zarr.open_array(os.path.join(tempdir, 'cube_temp_save', f'cube_{i}.zarr'))
            except FileNotFoundError:
                pass
        g_cut_plot(self, self.data_cut, self.cx, self.cy, self.cdx, self.cdy, self.cdensity, ty, z, x, angle, phi_offset, r1_offset, phi1_offset, r11_offset, self.stop_event, self.pool, self.path, self.e_photon, self.slim_cut, self.sym_cut, self.xmin, self.xmax, self.ymin, self.ymax, self.data_cube, self.app_pars)
        # if os.path.exists(os.path.join(tempdir, 'cut_temp_save')):
        #     shutil.rmtree(os.path.join(tempdir, 'cut_temp_save'))
        # if os.path.exists(os.path.join(tempdir, 'cube_temp_save')):
        #     shutil.rmtree(os.path.join(tempdir, 'cube_temp_save'))
        self.data_cube, self.data_cut = None, None
        return
        
    def set_density(self, *args):
        try:
            self.density = int(self.entry_d.get())
            self.update()
        except ValueError:
            print("Invalid input for density value")
    
    def refresh_geometry(self):
        self.g.update_idletasks()
        w = self.g.winfo_reqwidth()
        h = self.g.winfo_reqheight()
        t_sc_w = windll.user32.GetSystemMetrics(0)
        tx = t_sc_w if self.g.winfo_x()+self.g.winfo_width()/2 > t_sc_w else 0
        ScaleFactor = windll.shcore.GetScaleFactorForDevice(0)
        if self.app_pars.bar_pos == 'top':    #taskbar on top
            sc_y = int(40*ScaleFactor/100)
        else:
            sc_y = 0
        self.g.geometry(f'{w}x{h}+{tx}+{sc_y}')
        self.g.update()
    
    def change_mode(self, mode='normal'):
        self.mode = mode
        self.__get_slim()
        self.x = self.ox[self.slim[0]:self.slim[1]+1]
        if self.type == 'real':
            try:
                self.type = 'reciprocal'
                self.phi_offset = 0
                self.r1_offset = -31
                r1 = self.y - self.r1_offset
                phi = self.x - self.phi_offset
                tr1 = np.array([np.min(r1), np.max(r1), np.max(r1), np.min(r1)])
                tphi = np.array([np.min(phi), np.min(phi), np.max(phi), np.max(phi)])
                tx = np.sqrt(2*self.m*self.e*self.ev[self.slice_index])/self.hbar*10**-10*np.sin(tr1/180*np.pi) * np.cos(tphi/180*np.pi)
                ty = np.sqrt(2*self.m*self.e*self.ev[self.slice_index])/self.hbar*10**-10*np.sin(tphi/180*np.pi)
                r = np.max(np.sqrt(tx**2 + ty**2))
                self.xmin, self.xmax = -r, r
                self.ymin, self.ymax = -r, r
                self.txlim, self.tylim = [-r, r], [-r, r]
                self.b_mode.config(text='Reciprocal Mode')
                self.label_phi_offset.config(text="Set Manipulator Phi Offset (deg):")
                self.label_r1_offset.config(text="Set Manipulator R1 Offset (deg):")
                self.frame_entry6.pack(side=tk.TOP, after=self.frame_entry5)
                self.frame_entry7.pack(side=tk.TOP, after=self.frame_entry6)
                self.frame_region.pack(side=tk.TOP)
                self.frame_cut_xy.pack(side=tk.TOP)
                self.set_sym_button.pack(side=tk.LEFT, after=self.set_window_button)
                if mode == 'normal':
                    self.cut_xy(init=True)   # init cut params
                    self.update_window()
                self.refresh_geometry()
            except Exception as e:
                print(e)
                self.type = 'real'
                if self.z is not None:
                    self.xmin = np.min(np.min(self.x)+np.min(self.z))
                    self.xmax = np.max(np.max(self.x)+np.max(self.z))
                else:
                    self.xmin = np.min(self.x)
                    self.xmax = np.max(self.x)
                self.ymin = np.min(self.y)
                self.ymax = np.max(self.y)
                self.phi_offset = 48
                self.r1_offset = 11.5
                if self.xmin+self.xmax > 2*self.phi_offset:
                    self.xmin = self.phi_offset-(self.xmax-self.phi_offset)
                if self.xmax+self.xmin < 2*self.phi_offset:
                    self.xmax = self.phi_offset-(self.xmin-self.phi_offset)
                if self.ymin+self.ymax > 2*self.r1_offset:
                    self.ymin = self.r1_offset-(self.ymax-self.r1_offset)
                if self.ymax+self.ymin < 2*self.r1_offset:
                    self.ymax = self.r1_offset-(self.ymin-self.r1_offset)
                self.ax.set_xlabel('x (mm)', fontsize=self.size(20))
                self.ax.set_ylabel('z (mm)', fontsize=self.size(20))
                self.b_mode.config(text='Transmission Mode')
                self.label_phi_offset.config(text="Set Z Offset (mm):")
                self.label_r1_offset.config(text="Set X Offset (mm):")
                self.frame_entry6.pack_forget()
                self.frame_entry7.pack_forget()
                
        elif self.type == 'reciprocal':
            try:
                self.type = 'real'
                if self.z is not None:
                    self.xmin = np.min(np.min(self.x)+np.min(self.z))
                    self.xmax = np.max(np.max(self.x)+np.max(self.z))
                else:
                    self.xmin = np.min(self.x)
                    self.xmax = np.max(self.x)
                self.ymin = np.min(self.y)
                self.ymax = np.max(self.y)
                self.phi_offset = 48
                self.r1_offset = 11.5
                if self.xmin+self.xmax > 2*self.phi_offset:
                    self.xmin = self.phi_offset-(self.xmax-self.phi_offset)
                if self.xmax+self.xmin < 2*self.phi_offset:
                    self.xmax = self.phi_offset-(self.xmin-self.phi_offset)
                if self.ymin+self.ymax > 2*self.r1_offset:
                    self.ymin = self.r1_offset-(self.ymax-self.r1_offset)
                if self.ymax+self.ymin < 2*self.r1_offset:
                    self.ymax = self.r1_offset-(self.ymin-self.r1_offset)
                self.b_mode.config(text='Transmission Mode')
                self.label_phi_offset.config(text="Set Z Offset (mm):")
                self.label_r1_offset.config(text="Set X Offset (mm):")
                self.frame_entry6.pack_forget()
                self.frame_entry7.pack_forget()
                self.frame_cut_xy.pack_forget()
                self.frame_region.pack_forget()
                self.set_sym_button.pack_forget()
                if mode == 'normal':
                    self.update_window()
                self.refresh_geometry()
            except Exception as e:
                print(e)
                self.type = 'reciprocal'
                self.phi_offset = 0
                self.r1_offset = -31
                r1 = self.y - self.r1_offset
                phi = self.x - self.phi_offset
                tr1 = np.array([np.min(r1), np.max(r1), np.max(r1), np.min(r1)])
                tphi = np.array([np.min(phi), np.min(phi), np.max(phi), np.max(phi)])
                tx = np.sqrt(2*self.m*self.e*self.ev[self.slice_index])/self.hbar*10**-10*np.sin(tr1/180*np.pi) * np.cos(tphi/180*np.pi)
                ty = np.sqrt(2*self.m*self.e*self.ev[self.slice_index])/self.hbar*10**-10*np.sin(tphi/180*np.pi)
                r = np.max(np.sqrt(tx**2 + ty**2))
                self.xmin, self.xmax = -r, r
                self.ymin, self.ymax = -r, r
                self.txlim, self.tylim = [-r, r], [-r, r]
                self.ax.set_xlabel(r'$k_x$ ($\frac{2\pi}{\AA}$)', fontsize=self.size(20))
                self.ax.set_ylabel(r'$k_y$ ($\frac{2\pi}{\AA}$)', fontsize=self.size(20))
                self.b_mode.config(text='Reciprocal Mode')
                self.label_phi_offset.config(text="Set Manipulator Phi Offset (deg):")
                self.label_r1_offset.config(text="Set Manipulator R1 Offset (deg):")
                self.frame_entry6.pack(side=tk.TOP, after=self.frame_entry5)
                self.frame_entry7.pack(side=tk.TOP, after=self.frame_entry6)
                self.frame_region.pack(side=tk.TOP)
                self.frame_cut_xy.pack(side=tk.TOP)
                self.set_sym_button.pack(side=tk.LEFT, after=self.set_window_button)
    
    def __get_slim(self):
        min_val = int(float(self.entry_min.get()))
        max_val = int(float(self.entry_max.get()))
        self.slim = sorted([min_val, max_val])
        if self.slim[0] < 0:
            self.slim[0] = 0
        if self.slim[1] > 493:
            self.slim[1] = 493
        set_entry_value(self.entry_min, str(self.slim[0]))
        set_entry_value(self.entry_max, str(self.slim[1]))
        
    def set_slim(self, *args):
        self.sym = 1
        try:
            self.__get_slim()
            self.x, self.volume = [], []
            self.slice_index = np.argwhere(np.abs(self.ev-self.slider.val)<(self.ev[1]-self.ev[0])/2)[0][0]
            self.wait = wait(self.g, app_pars=self.app_pars)
            self.interpolate_slice(self.slice_index)
            self.disp_region()
            rotated_volume = rotate(self.surface, -self.angle, self.surface.shape)
            self.ax.clear()
            self.hl, = self.ax.plot([0, 0], [0, 0], color='green', linestyle='--')
            self.vl, = self.ax.plot([0, 0], [0, 0], color='green', linestyle='--')
            self.hl.set_data([],[])
            self.vl.set_data([],[])
            self.img = self.ax.imshow(rotated_volume, cmap=self.cmap, extent=[self.ymin, self.ymax, self.xmin, self.xmax], origin='lower')
            self.fig.canvas.draw_idle()
            self.wait.done()
            # setting entry
            self.entry_xmin.config(state='normal')
            self.entry_xmax.config(state='normal')
            self.entry_ymin.config(state='normal')
            self.entry_ymax.config(state='normal')
            if self.type == 'real':
                set_entry_value(self.entry_xmin, str(self.ymin))
                set_entry_value(self.entry_xmax, str(self.ymax))
                set_entry_value(self.entry_ymin, str(self.xmin))
                set_entry_value(self.entry_ymax, str(self.xmax))
                self.ax.set_xlabel('x (mm)', fontsize=self.size(20))
                self.ax.set_ylabel('z (mm)', fontsize=self.size(20))
                self.entry_xmin.config(state='disabled')
                self.entry_xmax.config(state='disabled')
                self.entry_ymin.config(state='disabled')
                self.entry_ymax.config(state='disabled')
            elif self.type == 'reciprocal':
                self.cut_l, = self.ax.plot([0, 0], [0, 0], color='red', linestyle='-')
                self.cut_l.set_data([], [])
                set_entry_value(self.entry_xmin, str(self.ymin))
                set_entry_value(self.entry_xmax, str(self.ymax))
                set_entry_value(self.entry_ymin, str(self.xmin))
                set_entry_value(self.entry_ymax, str(self.xmax))
                self.ax.set_xlabel(r'$k_x$ ($\frac{2\pi}{\AA}$)', fontsize=self.size(20))
                self.ax.set_ylabel(r'$k_y$ ($\frac{2\pi}{\AA}$)', fontsize=self.size(20))
            copy_to_clipboard(self.fig)
            gc.collect()
        except ValueError:
            self.wait.done()
            print("Invalid input for slim values")
        except Exception as e:
            self.wait.done()
            print(f"An error occurred: {e}\nVolumeSlicer: set_slim()")

    def update_window(self):
        try:
            self.entry_xmin.config(state='normal')
            self.entry_xmax.config(state='normal')
            self.entry_ymin.config(state='normal')
            self.entry_ymax.config(state='normal')
            if self.type == 'real':
                set_entry_value(self.entry_xmin, str(self.ymin))
                set_entry_value(self.entry_xmax, str(self.ymax))
                set_entry_value(self.entry_ymin, str(self.xmin))
                set_entry_value(self.entry_ymax, str(self.xmax))
            elif self.type == 'reciprocal':
                set_entry_value(self.entry_xmin, str(self.tylim[0]))
                set_entry_value(self.entry_xmax, str(self.tylim[1]))
                set_entry_value(self.entry_ymin, str(self.txlim[0]))
                set_entry_value(self.entry_ymax, str(self.txlim[1]))
            set_entry_value(self.entry_phi_offset, str(self.phi_offset))
            set_entry_value(self.entry_r1_offset, str(self.r1_offset))
            self.x, self.volume = [], []
            self.wait = wait(self.g, app_pars=self.app_pars)
            self.interpolate_slice(self.slice_index)
            self.disp_region()
            rotated_volume = rotate(self.surface, -self.angle, self.surface.shape)
            self.ax.clear()
            self.img = self.ax.imshow(rotated_volume, cmap=self.cmap, extent=[self.ymin, self.ymax, self.xmin, self.xmax], origin='lower')
            self.hl, = self.ax.plot([0, 0], [0, 0], color='green', linestyle='--')
            self.vl, = self.ax.plot([0, 0], [0, 0], color='green', linestyle='--')
            self.hl.set_data([],[])
            self.vl.set_data([],[])
            if self.type == 'reciprocal':
                self.cut_l, = self.ax.plot([0, 0], [0, 0], color='red', linestyle='-')
                self.cut_l.set_data([], [])
                self.ax.set_xlim([self.tylim[0], self.tylim[1]])
                self.ax.set_ylim([self.txlim[0], self.txlim[1]])
                self.ax.set_xlabel(r'$k_x$ ($\frac{2\pi}{\AA}$)', fontsize=self.size(20))
                self.ax.set_ylabel(r'$k_y$ ($\frac{2\pi}{\AA}$)', fontsize=self.size(20))
            elif self.type == 'real':
                self.ax.set_xlim([self.ymin, self.ymax])
                self.ax.set_ylim([self.xmin, self.xmax])
                self.ax.set_xlabel('x (mm)', fontsize=self.size(20))
                self.ax.set_ylabel('z (mm)', fontsize=self.size(20))
            copy_to_clipboard(self.fig)
            self.canvas.draw()
            self.wait.done()
            if self.type == 'real':
                self.entry_xmin.config(state='disabled')
                self.entry_xmax.config(state='disabled')
                self.entry_ymin.config(state='disabled')
                self.entry_ymax.config(state='disabled')
        except ValueError:
            self.wait.done()
            print("Range is not compatible with the current mode.")
        except Exception as e:
            self.wait.done()
            print(f"An error occurred: {e}\nVolumeSlicer: update_window()")
        
    def set_window(self):
        try:
            if self.type == 'reciprocal':
                self.txlim = sorted([float(self.entry_ymin.get()), float(self.entry_ymax.get())])
                self.tylim = sorted([float(self.entry_xmin.get()), float(self.entry_xmax.get())])
                self.update_window()
        except ValueError:
            print("Invalid input for window range values")
    
    def interpolate_slice(self, i):
        # self.xmin, self.xmax range should be larger than txlim, tylim in combine, and so as y
        try:
            self.phi_offset = np.float32(self.entry_phi_offset.get())
            self.r1_offset = np.float32(self.entry_r1_offset.get())
            self.phi1_offset = np.float32(self.entry_phi1_offset.get())
            self.r11_offset = np.float32(self.entry_r11_offset.get())
        except:pass
        self.x = self.ox[self.slim[0]:self.slim[1]+1]
        self.volume = self.ovolume[:, self.slim[0]:self.slim[1]+1, :]
        # def filter(ind, i, r2=None):    #test the filtering process in slice_data function
        #     r1 = np.linspace(min(self.y[ind]), max(self.y[ind]), len(self.y[ind]))[:, None]
        #     phi = np.linspace(min(self.x), max(self.x), len(self.x))[None, :]
        #     r1, phi = np.broadcast_arrays(r1, phi)
        #     if r2 is None:
        #         r1_offset, phi_offset = self.cal_r1_phi_offset()
        #         r1, phi = self.rot(r1, phi, r1_offset, phi_offset, self.angle)
        #     else:
        #         r1_offset, phi_offset = self.cal_r1_phi_offset(r2)
        #         r1, phi = self.rot(r1, phi, r1_offset, phi_offset, self.angle-(r2-self.z[0]))
        #     x = np.sqrt(2*self.m*self.e*self.ev[i])/self.hbar*10**-10*np.sin(r1/180*np.pi) * np.cos(phi/180*np.pi)  # x: r1, y: phi, at r2=0
        #     y = np.sqrt(2*self.m*self.e*self.ev[i])/self.hbar*10**-10*np.sin(phi/180*np.pi)
        #     ti=[]
        #     for i in range(r1.shape[1]):
        #         if any(-0.1<x[:,i]) and any(-0.2<y[:,i]) and any(x[:,i]<0.1) and any(y[:,i]<0.2):
        #             ti.append(i)
        #     if len(ti) != 0:
        #         if min(ti)>0:
        #             ti.insert(0, min(ti)-1)
        #         if max(ti)<len(self.y[ind])-1:
        #             ti.append(max(ti)+1)
        #     return ind[ti]
        if self.type == 'real':
            if self.z is not None:
                self.xmin = np.min(np.min(self.x)+np.min(self.z))
                self.xmax = np.max(np.max(self.x)+np.max(self.z))
            else:
                self.xmin = np.min(self.x)
                self.xmax = np.max(self.x)
            self.ymin = np.min(self.y)
            self.ymax = np.max(self.y)
            if self.xmin+self.xmax > 2*self.phi_offset:
                self.xmin = self.phi_offset-(self.xmax-self.phi_offset)
            elif self.xmax+self.xmin < 2*self.phi_offset:
                self.xmax = self.phi_offset-(self.xmin-self.phi_offset)
            if self.ymin+self.ymax > 2*self.r1_offset:
                self.ymin = self.r1_offset-(self.ymax-self.r1_offset)
            elif self.ymax+self.ymin < 2*self.r1_offset:
                self.ymax = self.r1_offset-(self.ymin-self.r1_offset)
            r = np.sqrt(((self.xmax-self.xmin)/2)**2+((self.ymax-self.ymin)/2)**2)
            self.xmin, self.xmax = self.phi_offset-r, self.phi_offset+r
            self.ymin, self.ymax = self.r1_offset-r, self.r1_offset+r
        elif self.type == 'reciprocal':
            if self.z is None: # for 1 cube
                r1_offset, phi_offset = self.cal_r1_phi_offset()
            elif self.z is not None: # for multiple cubes
                tr11o, tphi1o = 0, 0
                for r2 in self.z:
                    r1_offset, phi_offset = self.cal_r1_phi_offset(r2)
                    if abs(r1_offset) > abs(tr11o):
                        tr11o, tphi1o = r1_offset, phi_offset
                r1_offset, phi_offset = tr11o, tphi1o
            r1 = self.y - r1_offset
            phi = self.x - phi_offset
            tr1 = np.array([np.min(r1), np.max(r1), np.max(r1), np.min(r1)])
            tphi = np.array([np.min(phi), np.min(phi), np.max(phi), np.max(phi)])
            tx = np.float32(np.sqrt(2*self.m*self.e*self.ev[i])/self.hbar*10**-10*np.sin(tr1/180*np.pi) * np.cos(tphi/180*np.pi))
            ty = np.float32(np.sqrt(2*self.m*self.e*self.ev[i])/self.hbar*10**-10*np.sin(tphi/180*np.pi))
            # tx = np.sqrt(2*self.m*self.e*self.e_photon)/self.hbar*10**-10*np.sin(tr1/180*np.pi) * np.cos(tphi/180*np.pi)
            # ty = np.sqrt(2*self.m*self.e*self.e_photon)/self.hbar*10**-10*np.sin(tphi/180*np.pi)
            r = np.max(np.sqrt(tx**2 + ty**2))
            self.xmin, self.xmax = -r, r
            self.ymin, self.ymax = -r, r
            
        if self.z is None: # for 1 cube
            if self.type == 'real':
                self.surface = self.combine(self.volume[:, :, i], xlim = [min(self.y), max(self.y)], ylim = [min(self.x), max(self.x)])
            elif self.type == 'reciprocal':
                r1_offset, phi_offset = self.cal_r1_phi_offset()
                self.surface = self.combine(self.volume[:, :, i], xlim = [min(self.y)-r1_offset, max(self.y)-r1_offset], ylim = [min(self.x)-phi_offset, max(self.x)-phi_offset], ev=self.ev[i])
        elif self.z is not None: # for multiple cubes
            img = self.volume[:, :, i]
            try:
                self.tmin = np.min(img[img>0])
                r2 = sorted(set(self.z))
                self.surface = np.full((self.density, self.density), np.nan, dtype=np.float32)
                if self.type == 'real':
                    tt=0
                    for z in r2:
                        t1=time.perf_counter()
                        ind = np.array(np.argwhere(self.z==z), dtype=np.int64).flatten()
                        self.surface = np.nanmean([self.surface, self.combine(data = self.volume[ind, :, i], xlim = [min(self.y[ind]), max(self.y[ind])], ylim = [min(self.x)+z, max(self.x)+z])], axis=0)
                        if self.fl_show:
                            tt+=1
                            self.wait.text(f'R2 = {z}: {time.perf_counter()-t1:.3f}s ({tt}/{len(r2)})')
                    self.surface = np.nan_to_num(self.surface)
                elif self.type == 'reciprocal':
                    tt=0
                    for z in r2:
                        t1=time.perf_counter()
                        ind = np.array(np.argwhere(self.z==z), dtype=np.int64).flatten()
                        # ind = filter(ind, i, r2=z)
                        # if len(ind) != 0:
                        r1_offset, phi_offset = self.cal_r1_phi_offset(r2=z)
                        self.surface = np.nanmean([self.surface, self.combine(self.volume[ind, :, i], xlim = [min(self.y[ind])-r1_offset, max(self.y[ind])-r1_offset], ylim = [min(self.x)-phi_offset, max(self.x)-phi_offset], r2=z, ev=self.ev[i])], axis=0)
                        if self.fl_show:
                            tt+=1
                            self.wait.text(f'R2 = {z}: {time.perf_counter()-t1:.3f}s ({tt}/{len(r2)})')
                    self.surface = np.nan_to_num(self.surface)
            except:
                self.surface = np.zeros((self.density, self.density), dtype=np.float32)
        return self.surface
        
    def update(self, *args):
        self.hl.set_data([],[])
        self.vl.set_data([],[])
        self.slice_index = np.argwhere(np.abs(self.ev-self.slider.val)<(self.ev[1]-self.ev[0])/2)[0][0]
        self.wait = wait(self.g, app_pars=self.app_pars)
        self.interpolate_slice(self.slice_index)
        rotated_volume = rotate(self.surface, -self.angle, self.surface.shape)
        self.img.set_data(rotated_volume)
        copy_to_clipboard(self.fig)
        self.canvas.draw()
        self.wait.done()
        return

    def set_sl(self, *args):
        try:
            self.text_e.set(f'%.3f'%self.slider.val)
            # self.set_slim()
        except Exception as e:
            print(f"error: {e}\nVolumeSlicer: set_sl()")

    def set_tx(self, *args):
        try:
            self.slider.set_val(np.float64(self.text_e.get()))
            # self.set_slim()
        except Exception as e:
            print(f"error: {e}\nVolumeSlicer: set_tx()")
        
    def set_angle_sl(self, *args):
        try:
            self.angle = np.float64(self.angle_slider.val)
            self.text_a.set(f'%.3f'%self.angle)
            self.rotate_image()
        except ValueError:
            print("Invalid input for angle value")
    
    def set_angle_tx(self, *args):
        try:
            self.angle = np.float64(self.text_a.get())
            self.angle_slider.set_val(self.angle)
            self.rotate_image()
        except ValueError:
            print("Invalid input for angle value")
        
    def rotate_image(self):
        if self.mode == 'normal':
            self.disp_region()
            if self.type == 'real':
                self.hl.set_data([self.r1_offset, self.r1_offset], [self.xmin, self.xmax])
                self.vl.set_data([self.ymin, self.ymax], [self.phi_offset, self.phi_offset])
            elif self.type == 'reciprocal':
                self.hl.set_data([0, 0], [self.xmin, self.xmax])
                self.vl.set_data([self.ymin, self.ymax], [0, 0])
            rotated_volume = rotate(self.surface, -self.angle, self.surface.shape)
            self.img.set_data(rotated_volume)
            self.canvas.draw()

    def rot(self, x, y, r1_offset=0, phi_offset=0, angle=0):
        '''
        rotate the image with the given angle under the offset
        '''
        angle *= np.pi / 180
        c, s = np.cos(angle), np.sin(angle)
        x = x - r1_offset   # ndimage.shift in the process
        y = y - phi_offset
        x_rot = x * c - y * s   # ndimage.rotate in the process
        y_rot = x * s + y * c
        return x_rot, y_rot


    
    def pr_disp_region(self):
        c = ['black', 'blue', 'green', 'magenta', 'purple', 'orange', 'pink', 'cyan', 'brown', 'gray', 'gold', 'lime', 'navy', 'teal', 'coral', 'salmon', 'yellow']
        self.reg_l1 = []
        self.reg_l2 = []
        if self.z is None:
            self.reg_l1.append(self.ax_region.plot([], [], color=c[0%len(c)], linewidth=self.app_pars.scale*0.5))
            self.reg_l2.append(self.ax_region.plot([], [], color=c[0%len(c)], linewidth=self.app_pars.scale*0.5))
        else:
            for i in range(len(set(self.z))):
                self.reg_l1.append(self.ax_region.plot([], [], color=c[i%len(c)], linewidth=self.app_pars.scale*0.5))
                self.reg_l2.append(self.ax_region.plot([], [], color=c[i%len(c)], linewidth=self.app_pars.scale*0.5))
        return
    
    def disp_region(self):
        if self.type == 'reciprocal':
            tr1 = [0, 0]
            tphi = [0, 0]
            if self.z is None:
                r1 = self.y[:, None]
                phi = np.linspace(min(self.x), max(self.x), int(max(self.x)-min(self.x)))[None, :]
                r1, phi = np.broadcast_arrays(r1, phi)
                r1_offset, phi_offset = self.cal_r1_phi_offset()
                r1, phi = self.rot(r1, phi, r1_offset, phi_offset, self.angle)
                r1, phi, r1_, phi_ = mesh(r1, phi)
                self.reg_l1[0][0].set_data(r1, phi)
                self.reg_l2[0][0].set_data(r1_, phi_)
                if np.min(r1)<tr1[0]:
                    tr1[0] = np.min(r1)
                if np.max(r1)>tr1[1]:
                    tr1[1] = np.max(r1)
                if np.min(phi)<tphi[0]:
                    tphi[0] = np.min(phi)
                if np.max(phi)>tphi[1]:
                    tphi[1] = np.max(phi)
            else:
                r2 = sorted(set(self.z))
                for i, z in enumerate(r2):
                    ind = np.array(np.argwhere(self.z==z), dtype=np.int64).flatten()
                    if len(self.y[ind]) > 1:
                        r1 = self.y[ind][:, None]
                    else:
                        r1 = np.linspace(self.y[ind]-0.25, self.y[ind]+0.25, 2)[:, None]
                    phi = np.linspace(min(self.x), max(self.x), int(max(self.x)-min(self.x)))[None, :]
                    r1, phi = np.broadcast_arrays(r1, phi)
                    r1_offset, phi_offset = self.cal_r1_phi_offset(z)
                    r1, phi = self.rot(r1, phi, r1_offset, phi_offset, self.angle-(z-r2[0]))
                    r1, phi, r1_, phi_ = mesh(r1, phi)
                    self.reg_l1[i][0].set_data(r1, phi)
                    self.reg_l2[i][0].set_data(r1_, phi_)
                    if np.min(r1)<tr1[0]:
                        tr1[0] = np.min(r1)
                    if np.max(r1)>tr1[1]:
                        tr1[1] = np.max(r1)
                    if np.min(phi)<tphi[0]:
                        tphi[0] = np.min(phi)
                    if np.max(phi)>tphi[1]:
                        tphi[1] = np.max(phi)
            self.ax_region.set_aspect('equal')
            txlim, tylim = tr1, tphi
            txlim = -np.max([np.abs(txlim), np.abs(tylim)]), np.max([np.abs(txlim), np.abs(tylim)])
            tylim = txlim
            self.ax_region.set_xlim(txlim)
            self.ax_region.set_ylim(tylim)
            self.ax_region.set_xlabel(r'$R1$ (deg)', fontsize=self.size(20))
            self.ax_region.set_ylabel(r'$Phi$ (deg)', fontsize=self.size(20))
            self.canvas_region.draw()
        return

    def combine(self, data, xlim, ylim, r2=None, ev=None, step=0.25):  # complete
        '''
        Args:
        ---
        data: raw image (shape:(len(r1), len(phi)))
        xlim: [min, max]  x: r1
        ylim: [min, max]  y: phi
        step: width for only one r1 cube (default: 0.25 degree/mm)
        
        return:
        ---
            image
            shape: (density, density)
        '''
        xlim, ylim = sorted(xlim), sorted(ylim)
        fr2 = True
        if self.type == 'real':
            if int(self.density/(self.ymax-self.ymin)*(xlim[1]-xlim[0])) ==0:
                xlim[1] += step
                xlim[0] -= step
            data = cv2.resize(data, (int(self.density/(self.ymax-self.ymin)*(ylim[1]-ylim[0])), int(self.density/(self.xmax-self.xmin)*(xlim[1]-xlim[0]))), interpolation=cv2.INTER_CUBIC)
            base = np.zeros((self.density, self.density), dtype=np.float32)
            base[0:data.shape[0], 0:data.shape[1]] = data
            data = np.roll(base.T, (int((ylim[0]-self.xmin)/(self.xmax-self.xmin)*self.density), int((xlim[0]-self.ymin)/(self.ymax-self.ymin)*self.density)), axis=(0, 1))
            base = None
            data = data
        elif self.type == 'reciprocal':
            if r2 is None:
                fr2=False
                r2, self.z = 0, [0, 0]
            if int(self.density/180*(xlim[1]-xlim[0])) == 0: # at least 1 pixel
                d = 1/self.density*180/2
                c = (xlim[0]+xlim[1])/2
                xlim = [float(c-d), float(c+d)]
                print(f'Warning: R1-axis density is too low (R2=%.2f)'%r2)
                if self.test is False:
                    messagebox.showwarning("Warning",f'Warning: R1-axis density is too low (R2=%.2f)'%r2)
                self.focus_set()
            r1 = np.linspace(xlim[0], xlim[1], int(self.density/180*(xlim[1]-xlim[0]))*4)
            phi = np.linspace(ylim[0], ylim[1], int(self.density/180*(ylim[1]-ylim[0]))*4)
            # r1, phi = np.broadcast_arrays(r1, phi)
            x = np.sqrt(2*self.m*self.e*ev)/self.hbar*10**-10*np.sin(r1[:, None]/180*np.pi) * np.cos(phi[None, :]/180*np.pi)  # x: r1, y: phi, at r2=0
            y = np.sqrt(2*self.m*self.e*ev)/self.hbar*10**-10*np.sin(phi[None, :]/180*np.pi)
            r1, phi = None, None
            txlim, tylim = [np.min(x), np.max(x)], [np.min(y), np.max(y)]
            x, y = None, None
            ####### new method start
            # t=time.perf_counter()
            data = self.k_map(data, self.density, xlim, ylim, txlim, tylim, ev)
            ####################### new method end
            
            ####################### original plot start
            # tmax = np.max(data)
            # fig, ax = plt.subplots(dpi=150)
            # fig.patch.set_facecolor('black')
            # ax.set_facecolor('black')
            # data = cv2.resize(data, (int(self.density/180*(ylim[1]-ylim[0]))*4, int(self.density/180*(xlim[1]-xlim[0]))*4), interpolation=cv2.INTER_CUBIC)
            # ax.pcolormesh(x, y, data.T, cmap='gray')    # compatible with r2=0, phi: yaxis, r1: xaxis in transmission mode
            # ax.set_position([0, 0, 1, 1])
            # ax.axis('off')
            # fig.canvas.draw()
            # image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            # image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            # plt.close()
            # del fig, ax
            # image_gray = cv2.cvtColor(image_from_plot, cv2.COLOR_RGB2GRAY)/255*tmax
            # image_gray = cv2.resize(image_gray[::-1,:], (data.shape[1],data.shape[0]), interpolation=cv2.INTER_CUBIC)
            # data = cv2.resize(image_gray, (int(self.density/(self.xmax-self.xmin)*(txlim[1]-txlim[0])), int(self.density/(self.ymax-self.ymin)*(tylim[1]-tylim[0]))), interpolation=cv2.INTER_CUBIC)
            ########################### original plot end
            # print('1, resize+draw:', time.perf_counter()-t)
            # t = time.perf_counter()
            # print('2, resize*2:', time.perf_counter()-t)
            # t = time.perf_counter()
            base = np.zeros((self.density, self.density), dtype=np.float32)
            base[0:data.shape[0], 0:data.shape[1]] = data
            # del data, image_gray, image_from_plot
            data = None
            data = np.roll(base, (int((tylim[0]-self.ymin)/(self.ymax-self.ymin)*self.density), int((txlim[0]-self.xmin)/(self.xmax-self.xmin)*self.density)), axis=(0, 1))
            txlim, tylim = None, None
            # print('3, shift:', time.perf_counter()-t)
            # data = data[::-1, :]
            base = None
            # t = time.perf_counter()
            data = rotate(data, r2-self.z[0], data.shape)
            # print('4, rotate:', time.perf_counter()-t)
            if not fr2:
                self.z = None
        if self.z is not None and fr2==True:  # for multiple cubes need np.nanmean
            msk = data<self.tmin-self.tmin/3
            data[msk] = np.nan
            msk = None
            
        return data

    # def cal_xz
    
    def show(self):
        self.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
