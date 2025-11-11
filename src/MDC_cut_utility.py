import os, inspect
import tkinter as tk
from tkinter import filedialog as fd
import threading
import sys, shutil
from ctypes import windll
import gc
from multiprocessing import Pool
import time
from typing import Literal, Any
from abc import ABC, abstractmethod

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import xarray as xr
import h5py
import tqdm
import cv2
import cpuinfo
import psutil
import zarr
from scipy import special
from scipy.optimize import curve_fit
from MDC_cut import loadfiles, copy_to_clipboard, get_bar_pos

tempdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# deprecated
# def init_pars(app_pars):
#     global hwnd, ScaleFactor, scaled_font_size, scale, bar_pos, g_mem
#     hwnd = app_pars.hwnd
#     ScaleFactor = app_pars.ScaleFactor
#     scaled_font_size = app_pars.scaled_font_size
#     scale = app_pars.scale
#     bar_pos = app_pars.bar_pos
#     g_mem = app_pars.g_mem
    # print('memory: ',g_mem)

# deprecated
# def size(s: int) -> int:
#     return int(s * scale)

def smooth(x,l=20,p=3):
    from scipy.signal import savgol_filter
    """
    Using Savitzky-Golay filter to smooth the data.
    
    Parameters
    ------
    x : 1D array
        data to be smoothed
    l : int, default: 20
        window length
    p : int, default: 3
        polynomial order
    """
    x=savgol_filter(x, l, p)
    # for i in range(len(x)):
    #     if i>=l//2 and i+1<len(x)-l//2:
    #         x[i]=np.mean(x[i-l//2:i+l//2])
    return x

class CEC_Object(ABC):
    @abstractmethod
    def info(self):
        pass
    @abstractmethod
    def load(self):
        pass
    @abstractmethod
    def on_closing(self):
        pass

class app_param:
    def __init__(self, hwnd=None, ScaleFactor=None, scaled_font_size=None, scale=None, dpi=None, bar_pos=None, g_mem=None, g=None):
        self.hwnd = hwnd
        self.ScaleFactor = ScaleFactor
        self.scaled_font_size = scaled_font_size
        self.scale = scale
        self.dpi = dpi
        self.bar_pos = bar_pos
        self.g_mem = g_mem
        self.g = g


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

class spectrogram:
    """A class to plot the spectrogram data in a new Tkinter Toplevel window.
    This class creates a new Tkinter Toplevel window and populates it with
    various widgets to display the spectrogram data and related information.
    It includes two matplotlib figures for plotting, text widgets for displaying
    file paths and additional information, labels for displaying energy, cursor,
    and data values, and buttons for exporting data and copying images to the clipboard.
    
    Attributes
    ------------
        g (Tk): The Tkinter root window.
        data (xarray.DataArray): The spectrogram data.
        cmap (str): The colormap for plotting.
        tp_cf (bool): The flag to check if the top window is closed.
        dvalue (list): The list of data attributes.
        tst (str): The string of data attributes.
        lst (list): The list of lengths of the data attributes.
        x (array-like): The x-axis data.
        y (array-like): The y-axis data.
        s_exp (str): The name of the exported data file.
        s_exp_casa (str): The name of the exported data file for CASA.
        s_yl (str): The y-axis label for the exported data file.
        type (str): The type of data.
    """
    def __init__(self, data: xr.DataArray = [], path: list[str] | tuple[str, ...] | str = [], **kwargs) -> None:   # should input path in main function
        self.lfs = None
        self.npzf = False
        if len(path) > 0:
            if 'name' in kwargs:
                if kwargs['name'] == 'internal':
                    self.app_pars = kwargs['app_pars']
                    self.lfs = loadfiles(path, init=True, spectrogram=True)
                elif kwargs['name'] == 'external':
                    self.lfs = loadfiles(path, spectrogram=True)
            else:
                self.lfs = loadfiles(path, spectrogram=True)
            self.data = self.lfs.get(0)
            if self.lfs.f_npz[0]:self.npzf = True
        else:
            if 'app_pars' in kwargs:
                self.app_pars = kwargs['app_pars']
            self.data = data
        self.__preload(self.data)
        self.rr1 = self.phi[0]
        self.rr2 = self.phi[-1]
    
    def size(self, s: int) -> int:
        return int(s * self.scale)
    
    def __preload(self, data=[]) -> None:
        """Initialize the spectrogram class.
        
        Args
        -----------
        g : object
            A graphical user interface object.
        data : xr.DataArray
            The spectrogram data.
        cmap : str
            The colormap used for plotting.
        
        Returns
        -----------
        None
        """
        self.data = data
        self.tp_cf = True
        dvalue = list(self.data.attrs.values())
        self.dvalue = dvalue
        st=''
        lst=[]
        for _ in self.data.attrs.keys():
            if _ == 'Description':
                ts=str(self.data.attrs[_])
                ts=ts.replace('\n\n\n','\n')
                ts=ts.replace('\n\n','\n')
                t=ts.split('\n')
                st+=str(_)+' : '+str(self.data.attrs[_]).replace('\n','\n                     ')
                # st+=str(_)+' : '+str(self.data.attrs[_]).replace('\n','\n                         ')
                lst.append(len(' : '+t[0]))
                for i in range(1,len(t)):
                    lst.append(len('              '+t[i]))
            elif _ == 'Path':
                pass
            else:
                st+=str(_)+' : '+str(self.data.attrs[_])+'\n'
                lst.append(len(str(_)+' : '+str(self.data.attrs[_])))
        tst=st
        ev, phi = self.data.indexes.values()
        self.ev = np.float64(ev)
        self.phi = np.float64(phi)
        self.name = dvalue[0]
        self.e_photon = np.float64(dvalue[3].split(' ')[0])
        self.lensmode = dvalue[8]
        self.e_mode = dvalue[2]
        self.rdd = dvalue[14]
        self.desc = dvalue[13]
        self.desc=self.desc.replace('\n\n\n\n\n','\n')
        self.desc=self.desc.replace('\n\n\n\n','\n')
        self.desc=self.desc.replace('\n\n\n','\n')
        self.desc=self.desc.replace('\n\n','\n')
        self.desc=self.desc.replace('\n','; ')
        self.tst = tst
        self.lst = lst
        self.x = ev
        self.y = np.sum(self.data.to_numpy().transpose(),axis=0)
        if os.path.basename(self.rdd).split('.')[-1] != 'txt':
            self.s_exp=self.name+'.txt'
        else:
            self.s_exp=self.name+'_txt.txt'
        self.s_exp_casa=self.name+'_Casa.vms'
        self.s_yl='Intensity (Counts)'
        self.type='raw'
        self.fr1 = False
        self.fr2 = False
        self.fr3 = False
        self.fx1 = False
        self.fx2 = False
        self.fx3 = False
        dvalue, tst, lst = None, None, None
    
    def __change_file(self, *args):
        name = self.namevar.get()
        for i, j, k in zip(self.lfs.name, self.lfs.data, self.lfs.f_npz):
            if name == i:
                self.data = self.lfs.get(j)
                if k:self.npzf = True
                else:self.npzf = False
        self.__preload(self.data)
        self.l_path.config(width=max(self.lst)+2, state='normal')
        self.l_path.delete(1.0, tk.END)
        self.l_path.insert(tk.END,self.rdd)
        self.l_path.see(1.0)
        self.l_path.config(state='disabled')
        self.info.config(height=len(self.tst.split('\n')), width=max(self.lst)+2, state='normal')
        self.info.delete(1.0, tk.END)
        self.info.insert(tk.END, self.tst)
        self.info.see(1.0)
        self.info.config(state='disabled')
        try:self.s3.remove()
        except: pass
        self.__tp_a1_plot()
        self.__tp_a2_plot(self.oxl[0],self.oxl[1])
        self.__tp_rga_plot()
        self.rpo.draw()
        self.tpo.draw()
        self.rgo.draw()
    
    def __sel_y(self):
        phi_max = max([self.rr1, self.rr2])
        phi_min = min([self.rr1, self.rr2])
        i = (self.phi<=phi_max) & (self.phi>=phi_min)
        x = self.x
        if self.type=='raw':
            y = np.sum(self.data.to_numpy()[:,i], 1)
        elif self.type=='smooth':
            y=smooth(np.sum(self.data.to_numpy()[:,i].transpose(),axis=0),l=13)
        elif self.type=='fd':
            y=smooth(np.sum(self.data.to_numpy()[:,i].transpose(),axis=0),l=13)
            y=np.diff(y)/np.diff(self.ev)
        else:
            y = self.y
        return x, y
    
    def near(self, data, value):
        if len(data) == 1:
            return data[0]
        else:
            if max(data) >= value >= min(data):
                return data[np.argwhere(abs(data-value)<=abs(data[1]-data[0])/2)[0][0]]
            elif value < min(data):
                return min(data)
            elif value > max(data):
                return max(data)
                
    def update_plot(self, *args):
        self.plot_spectrum(self.selected_fit.get())
        self.update_input_fields(self.selected_fit.get())
    
    def create_input_row(self, label_text, variable, row, frame, col_start=0):
        label = tk.Label(frame, text=label_text, font=('Arial', self.size(18), 'bold'))
        label.grid(row=row, column=col_start, sticky='e')
        entry = tk.Entry(frame, textvariable=variable, font=('Arial', self.size(18), 'bold'))
        entry.grid(row=row, column=col_start + 1, sticky='w')
        return label, entry
    
    def update_input_fields(self, fit_type):
        # 隱藏所有輸入框
        try:
            for widget in self.inputs_frame.winfo_children():
                widget.grid_remove()
        except:
            pass
        
        # 根據選擇的擬合方式顯示相應的輸入框
        if fit_type == "Fermi-Dirac Fitting":
            self.emin_label.grid()
            self.emax_label.grid()
            self.emin_entry.grid()
            self.emax_entry.grid()
        elif fit_type == "Linear Fits":
            self.fL_min_label.grid()
            self.fL_max_label.grid()
            self.fF_min_label.grid()
            self.fF_max_label.grid()
            self.fR_min_label.grid()
            self.fR_max_label.grid()
            self.fL_min_entry.grid()
            self.fL_max_entry.grid()
            self.fF_min_entry.grid()
            self.fF_max_entry.grid()
            self.fR_min_entry.grid()
            self.fR_max_entry.grid()
        elif fit_type == "ERFC Fit":
            self.eminc_label.grid()
            self.emaxc_label.grid()
            self.eminc_entry.grid()
            self.emaxc_entry.grid()
        self.root.update()
        w = self.root.winfo_reqwidth()
        h = self.root.winfo_reqheight()
        w = int(12*self.dpi*self.scale+w)
        h = int(6*self.dpi*self.scale+h)
        self.root.geometry(f'{w}x{h}')
        self.root.update()
    
    def fit_press(self, event):
        if event.button == 1 and event.inaxes:
            self.fx1 = False
            self.fx2 = False
            self.fx3 = False
            self.fox = event.xdata
            if self.selected_fit.get() == "ERFC Fit":
                self.omin = self.eminc_val.get()
                self.omax = self.emaxc_val.get()
                if abs(self.eminc_val.get()-event.xdata) < abs(self.tr_a1.get_xlim()[1]-self.tr_a1.get_xlim()[0])*1/100:
                    self.fx1 = True
                    self.eminc_val.set(event.xdata)
                elif abs(self.emaxc_val.get()-event.xdata) < abs(self.tr_a1.get_xlim()[1]-self.tr_a1.get_xlim()[0])*1/100:
                    self.fx2 = True
                    self.emaxc_val.set(event.xdata)
                elif self.eminc_val.get() < event.xdata < self.emaxc_val.get():
                    self.fx3 = True
            elif self.selected_fit.get() == "Fermi-Dirac Fitting":
                self.omin = self.emin_val.get()
                self.omax = self.emax_val.get()
                if abs(self.emin_val.get()-event.xdata) < abs(self.tr_a1.get_xlim()[1]-self.tr_a1.get_xlim()[0])*1/100:
                    self.fx1 = True
                    self.emin_val.set(event.xdata)
                elif abs(self.emax_val.get()-event.xdata) < abs(self.tr_a1.get_xlim()[1]-self.tr_a1.get_xlim()[0])*1/100:
                    self.fx2 = True
                    self.emax_val.set(event.xdata)
                elif self.emin_val.get() < event.xdata < self.emax_val.get():
                    self.fx3 = True
            self.update_fit()
                
    def fit_move(self, event):
        if self.fx1 or self.fx2 or self.fx3:
            if self.selected_fit.get() == "ERFC Fit":
                if self.fx1:
                    self.eminc_val.set(event.xdata)
                elif self.fx2:
                    self.emaxc_val.set(event.xdata)
                elif self.fx3:
                    self.eminc_val.set(self.omin+(event.xdata-self.fox))
                    self.emaxc_val.set(self.omax+(event.xdata-self.fox))
            elif self.selected_fit.get() == "Fermi-Dirac Fitting":
                if self.fx1:
                    self.emin_val.set(event.xdata)
                elif self.fx2:
                    self.emax_val.set(event.xdata)
                elif self.fx3:
                    self.emin_val.set(self.omin+(event.xdata-self.fox))
                    self.emax_val.set(self.omax+(event.xdata-self.fox))
            self.update_fit()
        
            
    def fit_release(self, event):
        self.fx1, self.fx2, self.fx3 = False, False, False
        self.update_fit()
    
    def update_fit(self, *args):
        e = self.ev
        x, ss = self.__sel_y()
        fit_type = self.selected_fit.get()
        if fit_type == "Fermi-Dirac Fitting":
            try:
                self.fl1.remove()
                self.fl2.remove()
                self.fl3.remove()
                self.flg.remove()
            except:
                pass
            emin = self.emin_val.get()
            emax = self.emax_val.get()
            self.fl1 = self.a1.axvline(emin, color='r', linestyle='--')
            self.fl2 = self.a1.axvline(emax, color='r', linestyle='--')
            mask = (e > emin) & (e < emax)
            
            x = e[mask]
            y = ss[mask]
            
            def fermi_dirac(E, EF, T, A, B):
                k_B = 8.617333262145e-5  # Boltzmann constant in eV/K
                return A / (np.exp((E - EF) / (k_B * T)) + 1) + B
            
            try:
                initial_guess = [self.e_photon, 300.0, np.max(y), np.min(y)]
                popt, pcov = curve_fit(fermi_dirac, x, y, p0=initial_guess)
                k_B = 8.617333262145e-5
                EF = popt[0]
                T = popt[1]
                self.fl3, = self.a1.plot(x, fermi_dirac(x, *popt), 'r-', label=f'Fermi-Dirac Fit: EF = {EF:.2f} eV, T = {T:.2f} K, $k_bT={k_B*T:.2f}$')
                self.flg = self.a1.legend()
            except:
                EF = None
                T = None
                pass
        
        elif fit_type == "ERFC Fit":
            try:
                self.fl1.remove()
                self.fl2.remove()
                self.fl3.remove()
                self.flg.remove()
            except:
                pass
            eminc = self.eminc_val.get()
            emaxc = self.emaxc_val.get()
            self.fl1 = self.a2.axvline(eminc, color='r', linestyle='--')
            self.fl2 = self.a2.axvline(emaxc, color='r', linestyle='--')
            mask = (e > eminc) & (e < emaxc)
            
            x = e[mask]
            y = ss[mask]
            
            def erfc_fit(E, E0, sigma, A, B):
                return A * special.erfc((E - E0) / sigma) + B
            try:
                initial_guess = [self.e_photon, 0.1, np.max(y), np.min(y)]
                popt, pcov = curve_fit(erfc_fit, x, y, p0=initial_guess)
                
                EF = popt[0]
                E0 = popt[0]
                sigma = popt[1]
                
                self.fl3, = self.a2.plot(x, erfc_fit(x, *popt), 'r-', label=f'ERFC Fit: E0 = {E0:.2f} eV, sigma = {sigma:.2f}')
                self.flg = self.a2.legend()
            except:
                EF = None
                E0 = None
                sigma = None
                pass
        try:
            if EF is not None:
                self.ef_label.config(text=f"Fermi Level (EF): {EF:.2f} eV")
            else:
                self.ef_label.config(text="Fermi Level (EF): N/A")
        except:
            pass
        self.canvas.draw()
    
    
    def plot_spectrum(self, fit_type):
        from scipy.ndimage import gaussian_filter1d
        e = self.ev
        x, ss = self.__sel_y()
        # Smooth the data using Gaussian smoothing
        smoothed_ss = gaussian_filter1d(ss, sigma=2)

        # 清空先前的畫布
        try:
            for widget in self.frame.winfo_children():
                widget.destroy()
        except:
            pass
        
        def plot_base_spectrum(ax, x, y, title, xlabel, ylabel):
            ax.scatter(x, y, s= self.scale*self.scale*1, c='k', alpha=0.8)
            ax.set_title(title, fontsize=self.size(20))
            ax.set_xlabel(xlabel, fontsize=self.size(18))
            ax.set_ylabel(ylabel, fontsize=self.size(18))
            ax.set_xlim(self.tr_a2.get_xlim())

        def add_canvas(fig):
            try:
                self.canvas = FigureCanvasTkAgg(fig, master=self.frame)
                self.canvas.draw()
                self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
                if fit_type in ["Fermi-Dirac Fitting", "ERFC Fit"]:
                    self.canvas.mpl_connect('button_press_event', self.fit_press)
                    self.canvas.mpl_connect('motion_notify_event', self.fit_move)
                    self.canvas.mpl_connect('button_release_event', self.fit_release)
                self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame)
                self.toolbar.update()
                self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
                self.root.update()
            except:
                pass

        EF = None

        if fit_type == "Raw Data":
            fig = plt.Figure(figsize=(12*self.scale, 6*self.scale))
            ax = fig.add_subplot(111)
            plot_base_spectrum(ax, e, ss, 'Raw Data', 'Kinetic Energy (eV)', 'Intensity (counts)')
            add_canvas(fig)
        
        elif fit_type == "Smooth Data":
            fig = plt.Figure(figsize=(12*self.scale, 6*self.scale))
            ax = fig.add_subplot(111)
            plot_base_spectrum(ax, e, smoothed_ss, 'Smooth Data', 'Kinetic Energy (eV)', 'Intensity (counts)')
            add_canvas(fig)

        elif fit_type == "Fermi-Dirac Fitting":
            fig1 = plt.Figure(figsize=(12*self.scale, 6*self.scale))
            self.a1 = fig1.add_subplot(111)
            plot_base_spectrum(self.a1, e, ss, 'Raw Data with Fermi-Dirac Fitting', 'Kinetic Energy (eV)', 'Intensity (counts)')

            emin = self.emin_val.get()
            emax = self.emax_val.get()
            self.fl1 = self.a1.axvline(emin, color='r', linestyle='--')
            self.fl2 = self.a1.axvline(emax, color='r', linestyle='--')
            mask = (e > emin) & (e < emax)
            
            x = e[mask]
            y = ss[mask]
            
            def fermi_dirac(E, EF, T, A, B):
                k_B = 8.617333262145e-5  # Boltzmann constant in eV/K
                return A / (np.exp((E - EF) / (k_B * T)) + 1) + B
            
            try:
                initial_guess = [self.e_photon, 300.0, np.max(y), np.min(y)]
                popt, pcov = curve_fit(fermi_dirac, x, y, p0=initial_guess)
                k_B = 8.617333262145e-5
                EF = popt[0]
                T = popt[1]
                self.fl3, = self.a1.plot(x, fermi_dirac(x, *popt), 'r-', label=f'Fermi-Dirac Fit: EF = {EF:.2f} eV, T = {T:.2f} K, $k_bT={k_B*T:.2f}$')
                self.flg = self.a1.legend()
            except:
                EF = None
                T = None
                pass
            add_canvas(fig1)
        
        elif fit_type == "ERFC Fit":
            fig2 = plt.Figure(figsize=(12*self.scale, 6*self.scale))
            self.a2 = fig2.add_subplot(111)
            plot_base_spectrum(self.a2, e, ss, 'Raw Data with ERFC Fit', 'Kinetic Energy (eV)', 'Intensity (counts)')
            
            eminc = self.eminc_val.get()
            emaxc = self.emaxc_val.get()
            self.fl1 = self.a2.axvline(eminc, color='r', linestyle='--')
            self.fl2 = self.a2.axvline(emaxc, color='r', linestyle='--')
            mask = (e > eminc) & (e < emaxc)
            
            x = e[mask]
            y = ss[mask]
            
            def erfc_fit(E, E0, sigma, A, B):
                return A * special.erfc((E - E0) / sigma) + B
            try:
                initial_guess = [self.e_photon, 0.1, np.max(y), np.min(y)]
                popt, pcov = curve_fit(erfc_fit, x, y, p0=initial_guess)
                
                EF = popt[0]
                E0 = popt[0]
                sigma = popt[1]
                
                self.fl3, = self.a2.plot(x, erfc_fit(x, *popt), 'r-', label=f'ERFC Fit: E0 = {E0:.2f} eV, sigma = {sigma:.2f}')
                self.flg = self.a2.legend()
            except:
                EF = None
                E0 = None
                sigma = None
                pass
            add_canvas(fig2)

        elif fit_type == "Linear Fits":
            fig3 = plt.Figure(figsize=(12*self.scale, 6*self.scale))
            a3 = fig3.add_subplot(111)
            plot_base_spectrum(a3, e, ss, 'Raw Data with Linear Fits', 'Kinetic Energy (eV)', 'Intensity (counts)')

            fL_min = self.fL_min_val.get()
            fL_max = self.fL_max_val.get()
            fF_min = self.fF_min_val.get()
            fF_max = self.fF_max_val.get()
            fR_min = self.fR_min_val.get()
            fR_max = self.fR_max_val.get()

            mask_L = (e > fL_min) & (e < fL_max)
            mask_F = (e > fF_min) & (e < fF_max)
            mask_R = (e > fR_min) & (e < fR_max)

            xL, yL = e[mask_L], ss[mask_L]
            xF, yF = e[mask_F], ss[mask_F]
            xR, yR = e[mask_R], ss[mask_R]

            # 进行线性拟合
            def linear_fit(x, a, b):
                return a * x + b
            try:
                popt_L, _ = curve_fit(linear_fit, xL, yL)
                popt_F, _ = curve_fit(linear_fit, xF, yF)
                popt_R, _ = curve_fit(linear_fit, xR, yR)

                a3.plot(xL, linear_fit(xL, *popt_L), 'r-', label='Left Fit')
                a3.plot(xF, linear_fit(xF, *popt_F), 'g-', label='Flat Fit')
                a3.plot(xR, linear_fit(xR, *popt_R), 'b-', label='Right Fit')

                a3.legend()
            except:
                pass
            add_canvas(fig3)

        elif fit_type == "First Derivative":
            fig4 = plt.Figure(figsize=(12*self.scale, 6*self.scale))
            a4 = fig4.add_subplot(111)
            plot_base_spectrum(a4, e, np.gradient(ss), 'First Derivative', 'Kinetic Energy (eV)', 'dIntensity / dE')
            add_canvas(fig4)

        elif fit_type == "Second Derivative":
            fig5 = plt.Figure(figsize=(12*self.scale, 6*self.scale))
            a5 = fig5.add_subplot(111)
            plot_base_spectrum(a5, e, np.gradient(np.gradient(ss)), 'Second Derivative', 'Kinetic Energy (eV)', 'd²Intensity / dE²')
            add_canvas(fig5)
            
        elif fit_type == "Smooth Data with First Derivative":
            fig6 = plt.Figure(figsize=(12*self.scale, 6*self.scale))
            a6 = fig6.add_subplot(111)
            plot_base_spectrum(a6, e, np.gradient(smoothed_ss), 'Smoothed Data with First Derivative', 'Kinetic Energy (eV)', 'd(Smoothed Intensity) / dE')
            add_canvas(fig6)
            
        elif fit_type == "Segmented Tangents":
            fig7 = plt.Figure(figsize=(12*self.scale, 6*self.scale))
            a7 = fig7.add_subplot(111)
            plot_base_spectrum(a7, e, smoothed_ss, 'Smooth Data with Segmented Tangents', 'Kinetic Energy (eV)', 'Intensity (counts)')

            diff = np.gradient(smoothed_ss, e)
            for i in range(len(e) - 1):
                x_segment = [e[i], e[i+1]]
                y_segment = [smoothed_ss[i], smoothed_ss[i] + diff[i] * (e[i+1] - e[i])]
                a7.plot(x_segment, y_segment, 'r-')

            add_canvas(fig7)
        try:
            if EF is not None:
                self.ef_label.config(text=f"Fermi Level (EF): {EF:.2f} eV")
            else:
                self.ef_label.config(text="Fermi Level (EF): N/A")
        except:
            pass
        
    def __ups(self):
        self.root = tk.Toplevel(self.g,bg='white')
        self.root.title('UPS spectrum')
        
        # OptionMenu 設定
        fit_options = ["Raw Data", "Smooth Data", "Fermi-Dirac Fitting", "Linear Fits", "ERFC Fit", "First Derivative", "Second Derivative", "Smooth Data with First Derivative", "Segmented Tangents"]
        self.selected_fit = tk.StringVar(self.root)
        self.selected_fit.set(fit_options[4])  # 初始選項

        option_menu = tk.OptionMenu(self.root, self.selected_fit, *fit_options, command=self.update_plot)
        option_menu.config(font=('Arial', self.size(18), 'bold'))
        option_menu.grid(row=0, column=0)

        # emax 和 emin 的初始值
        self.emin_val = tk.DoubleVar(value=self.e_photon-0.2)
        self.emax_val = tk.DoubleVar(value=self.e_photon+0.3)

        # emaxc 和 eminc 的初始值
        self.eminc_val = tk.DoubleVar(value=self.e_photon-0.2)
        self.emaxc_val = tk.DoubleVar(value=self.e_photon+0.3)

        # Linear Fits 的上下界初始值
        self.fL_min_val = tk.DoubleVar(value=self.e_photon-0.4)
        self.fL_max_val = tk.DoubleVar(value=self.e_photon-0.35)
        self.fF_min_val = tk.DoubleVar(value=self.e_photon-0.2)
        self.fF_max_val = tk.DoubleVar(value=self.e_photon-0.02)
        self.fR_min_val = tk.DoubleVar(value=self.e_photon)
        self.fR_max_val = tk.DoubleVar(value=self.e_photon+0.8)

        # 創建輸入框
        self.inputs_frame = tk.Frame(self.root)
        self.inputs_frame.grid(row=1, column=0, columnspan=5, sticky='w')

        # 創建輸入框，並預設隱藏
        self.emin_label, self.emin_entry = self.create_input_row("emin:", self.emin_val, 0, self.inputs_frame)
        self.emax_label, self.emax_entry = self.create_input_row("emax:", self.emax_val, 0, self.inputs_frame, col_start=2)

        self.eminc_label, self.eminc_entry = self.create_input_row("eminc:", self.eminc_val, 1, self.inputs_frame)
        self.emaxc_label, self.emaxc_entry = self.create_input_row("emaxc:", self.emaxc_val, 1, self.inputs_frame, col_start=2)

        self.fL_min_label, self.fL_min_entry = self.create_input_row("fL min:", self.fL_min_val, 2, self.inputs_frame)
        self.fL_max_label, self.fL_max_entry = self.create_input_row("fL max:", self.fL_max_val, 2, self.inputs_frame, col_start=2)

        self.fF_min_label, self.fF_min_entry = self.create_input_row("fF min:", self.fF_min_val, 3, self.inputs_frame)
        self.fF_max_label, self.fF_max_entry = self.create_input_row("fF max:", self.fF_max_val, 3, self.inputs_frame, col_start=2)

        self.fR_min_label, self.fR_min_entry = self.create_input_row("fR min:", self.fR_min_val, 4, self.inputs_frame)
        self.fR_max_label, self.fR_max_entry = self.create_input_row("fR max:", self.fR_max_val, 4, self.inputs_frame, col_start=2)

        # 添加顯示EF值的區塊
        self.ef_label = tk.Label(self.root, text="Fermi Level (EF): N/A", font=('Arial', self.size(18), 'bold'))
        self.ef_label.grid(row=2, column=4, rowspan=2, padx=20)

        # 創建可滾動畫布
        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.grid(row=8, column=0, columnspan=5, sticky='nsew')

        self.canvas = tk.Canvas(self.canvas_frame)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.frame, anchor='nw')

        # 讓主窗口和 canvas_frame 自動擴展
        self.root.grid_rowconfigure(8, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)

        self.update_input_fields(self.selected_fit.get())  # 初始化輸入框顯示
        self.update_plot()
        self.root.bind("<Return>", self.update_plot)
        self.root.update()
    
    def _select_all(self, event):
        event.widget.tag_add(tk.SEL, "1.0", tk.END)
        event.widget.mark_set(tk.INSERT, "1.0")
        event.widget.see(tk.INSERT)
        return 'break'
    
    def __copy_to_clipboard(self):
        buf1 = io.BytesIO()
        buf2 = io.BytesIO()
        self.rpf.savefig(buf1, format='png')
        self.tpf.savefig(buf2, format='png')
        buf1.seek(0)
        buf2.seek(0)
        image1 = Image.open(buf1)
        Image2 = Image.open(buf2)
        image = Image.new('RGB', (image1.width, image1.height + Image2.height))
        image.paste(image1, (0, 0))
        image.paste(Image2, (0, image1.height))
        output = io.BytesIO()
        
        image.convert("RGB").save(output, "BMP")
        data = output.getvalue()[14:]
        output.close()
        send_to_clipboard(win32clipboard.CF_DIB, data)
        
        # image.save(output, format='PNG')
        # data = output.getvalue()[14:]
        # output.close()

        # self.g.clipboard_clear()
        # self.g.clipboard_append(data)
        # self.g.update()  # now it stays on the clipboard after the window is closed
    
    def setdata(self, x, y, dtype='raw', unit='Counts'):
        """Set the data for plotting.
        
        Args
        -----------
            x (list or array-like): The x-axis data.
            y (list or array-like): The y-axis data.
            dtype (str, optional): The type of data. Defaults to 'raw'.
            unit (str, optional): The unit of the y-axis data. Defaults to 'Counts'.
            
        Raises
        -----------
            ValueError : If the length of x and y are not the same.
        """
        self.x=x
        self.y=y
        self.type=dtype
        if len(x)!=len(y):
            print('len(x):',len(x),'len(y):',len(y))
            raise ValueError('The length of x and y must be the same.')
        if dtype=='raw':
            self.s_yl='Intensity (Counts)'
            if os.path.basename(self.rdd).split('.')[-1]!='txt':
                self.s_exp=self.name+'.txt'
            else:
                self.s_exp=self.name+'_txt'+'.txt'
            self.s_exp_casa=self.name+'_Casa.vms'
        else:
            self.s_yl='Intensity ('+unit+')'
            if os.path.basename(self.rdd).split('.')[-1]!='txt':
                self.s_exp=self.name+'.txt'
            else:
                self.s_exp=self.name+'_txt'+'.txt'
            self.s_exp_casa=self.name+'_'+dtype+'_Casa.vms'
    
    def __scroll(self, event):
        if event.delta>0:
            self.__cf_up()
        elif event.delta<0:
            self.__cf_down()
    
    def __cf_up(self, *args):
        now = self.namevar.get()
        for i, j in enumerate(self.lfs.name):
            if now == j:
                if i == 0:
                    self.namevar.set(self.lfs.name[-1])
                else:
                    self.namevar.set(self.lfs.name[i-1])
        self.__change_file()

    def __cf_down(self, *args):
        now = self.namevar.get()
        for i, j in enumerate(self.lfs.name):
            if now == j:
                if i == len(self.lfs.name)-1:
                    self.namevar.set(self.lfs.name[0])
                else:
                    self.namevar.set(self.lfs.name[i+1])
        self.__change_file()
    
    
    def plot(self, g=None, cmap='viridis'):
        """Plot the spectrogram data in a new Tkinter Toplevel window.
        This method creates a new Tkinter Toplevel window and populates it with
        various widgets to display the spectrogram data and related information.
        It includes two matplotlib figures for plotting, text widgets for displaying
        file paths and additional information, labels for displaying energy, cursor,
        and data values, and buttons for exporting data and copying images to the clipboard.
        
        .. Widgets:
        --------
            - Toplevel window with title 'Spectrogram: <name>'
            - Fitting utility button for Fermi level fitting
            - Two matplotlib figures for plotting spectrogram data
            - Text widget for displaying the file path
            - Text widget for displaying additional information
            - Labels for displaying energy, cursor, and data values
            - Buttons for exporting raw data and copying images to the clipboard
            
        .. Event Bindings:
        --------
            - Motion notify event for matplotlib figures
            - Button press event for matplotlib figures
            - Button release event for matplotlib figures
            - FocusIn event for the additional information text widget
            
        .. Methods Called:
        --------
            - __ups
            - __export
            - __export_casa
            - __copy_to_clipboard
            - __trans_plot_job
            
        Note:
        --------
            The method uses the Tkinter library for GUI components and matplotlib for plotting.
        
        """
        # global tpf,tpo,rpf,rpo,l_cx,l_cy,l_dy
        self.cmap = cmap
        if g is None:
            ScaleFactor = windll.shcore.GetScaleFactorForDevice(0)
            t_sc_w, t_sc_h = windll.user32.GetSystemMetrics(0), windll.user32.GetSystemMetrics(1)   # Screen width and height
            t_sc_h-=int(40*ScaleFactor/100)
            bar_pos = get_bar_pos()
            if bar_pos == 'top':    #taskbar on top
                sc_y = int(40*ScaleFactor/100)
            else:
                sc_y = 0
            self.tpg = tk.Tk()
            self.g = self.tpg
            odpi=self.tpg.winfo_fpixels('1i')
            # prfactor = 1.03 if ScaleFactor <= 100 else 1.2 if ScaleFactor <= 125 else 0.9 if ScaleFactor <= 150 else 0.55
            prfactor = 1
            ScaleFactor /= prfactor*(ScaleFactor/100*1890/96*odpi/t_sc_w) if 1890/t_sc_w >= (954)/t_sc_h else prfactor*(ScaleFactor/100*(954)/96*odpi/t_sc_h)
            self.tpg.tk.call('tk', 'scaling', ScaleFactor/100)
            # global scale, dpi
            self.dpi = self.tpg.winfo_fpixels('1i')
            windll.shcore.SetProcessDpiAwareness(1)
            self.scale = odpi / self.dpi * ScaleFactor / 100
            self.tpg.config(bg='white')
            base_font_size = 14
            scaled_font_size = int(base_font_size * self.scale)

            plt.rcParams['font.family'] = 'Arial'
            plt.rcParams['font.size'] = int(plt.rcParams['font.size'] * self.scale)
            plt.rcParams['lines.linewidth'] = plt.rcParams['lines.linewidth'] * self.scale
            plt.rcParams['lines.markersize'] = plt.rcParams['lines.markersize'] * self.scale * self.scale
            plt.rcParams['figure.figsize'] = (plt.rcParams['figure.figsize'][0] * self.scale, plt.rcParams['figure.figsize'][1] * self.scale)

            # 設定預設字體
            default_font = ('Arial', scaled_font_size)
            self.tpg.option_add('*Font', default_font)
        else:
            self.g = g
            ScaleFactor = windll.shcore.GetScaleFactorForDevice(0)
            t_sc_w, t_sc_h = windll.user32.GetSystemMetrics(0), windll.user32.GetSystemMetrics(1)   # Screen width and height
            t_sc_h-=int(40*ScaleFactor/100)
            if self.app_pars.bar_pos == 'top':    #taskbar on top
                sc_y = int(40*ScaleFactor/100)
            else:
                sc_y = 0
            self.scale, self.dpi = self.app_pars.scale, self.app_pars.dpi
            self.tpg = tk.Toplevel(g, bg='white')
            tx = int(t_sc_w*ScaleFactor/100) if g.winfo_x()+g.winfo_width()/2 > t_sc_w else 0
            self.tpg.geometry(f"1900x1000+{tx}+{sc_y}")
        self.tpg.title('Spectrogram: '+self.name)
        
        fr_fig=tk.Frame(self.tpg,bg='white',bd=0)
        fr_fig.grid(row=0,column=0,sticky='nsew')
        
        self.rpf = plt.Figure(figsize=(15*self.scale, 4.75*self.scale), layout='constrained')
        self.rpo = FigureCanvasTkAgg(self.rpf, master=fr_fig)
        self.rpo.get_tk_widget().grid(row=0, column=0)
        self.rpo.mpl_connect('motion_notify_event', self.__rp_move)
        self.rpo.mpl_connect('button_press_event', self.__rp_press)
        self.rpo.mpl_connect('button_release_event', self.__rp_release)
        
        self.tpf = plt.Figure(figsize=(15*self.scale, 4.75*self.scale), layout='constrained')
        self.tpo = FigureCanvasTkAgg(self.tpf, master=fr_fig)
        self.tpo.get_tk_widget().grid(row=1, column=0)
        self.tpo.mpl_connect('motion_notify_event', self.__tp_move)
        self.tpo.mpl_connect('button_press_event', self.__tp_press)
        self.tpo.mpl_connect('button_release_event', self.__tp_release)
        
        self.rgf = plt.Figure(figsize=(0.25*self.scale, 4.75*self.scale), layout='constrained')
        self.rgo = FigureCanvasTkAgg(self.rgf, master=fr_fig)
        self.rgo.get_tk_widget().grid(row=0, column=1)
        self.rgo.mpl_connect('motion_notify_event', self.__rg_move)
        self.rgo.mpl_connect('button_press_event', self.__rg_press)
        self.rgo.mpl_connect('button_release_event', self.__rg_release)
        
        self.fr_info=tk.Frame(self.tpg,bg='white',bd=5)
        self.fr_info.grid(row=0,column=1)
        try:
            if len(self.lfs.name)>1:
                nlist = self.lfs.name
                self.namevar = tk.StringVar(value=nlist[0])
                self.l_name = tk.OptionMenu(self.fr_info, self.namevar, *nlist, command=self.__change_file)
                self.l_name.config(font=('Arial', self.size(13), 'bold'))
                self.l_name.grid(row=0, column=0, sticky='ew')
        except:
            pass
        self.l_path = tk.Text(self.fr_info, wrap='word', font=("Arial", self.size(11), "bold"), bg="white", fg="black", state='disabled',height=3,width=30)
        self.l_path.grid(row=1, column=0)
        self.l_path.config(width=max(self.lst)+2, state='normal')
        self.l_path.delete(1.0, tk.END)
        self.l_path.insert(tk.END,self.rdd)
        self.l_path.see(1.0)
        self.l_path.config(state='disabled')
        
        self.info = tk.Text(self.fr_info, wrap='none', font=("Arial", self.size(11), "bold"), bg="white", fg="black", state='disabled', height=10, width=30)
        self.info.grid(row=2, column=0)
        self.info.bind("<FocusIn>", self._select_all)
        self.info.config(height=len(self.tst.split('\n')), width=max(self.lst)+2, state='normal')
        self.info.insert(tk.END, self.tst)
        self.info.see(1.0)
        self.info.config(state='disabled')
        
        self.l_cx=tk.Label(self.fr_info,text='%9s'%'Energy : ',fg='green',font=('Arial', self.size(18)),bg='white',width=20,anchor='w')
        self.l_cx.grid(row=3,column=0)
        
        self.l_cy=tk.Label(self.fr_info,text='%10s'%'Cursor : ',font=('Arial', self.size(18)),bg='white',width=20,anchor='w')
        self.l_cy.grid(row=4,column=0)
        
        self.l_dy=tk.Label(self.fr_info,text='%11s'%'Data : ',fg='red',font=('Arial', self.size(18)),bg='white',width=20,anchor='w')
        self.l_dy.grid(row=5,column=0)
        
        if self.type != 'fd':
            self.b_ups = tk.Button(self.fr_info, text='Fermi Level Fitting', command=self.__ups, width=30, height=1, font=('Arial', self.size(12), "bold"), bg='white', bd=5)
            self.b_ups.grid(row=6, column=0)
        
        self.b_exp = tk.Button(self.fr_info, text='Export Data ( .txt )', command=self.__export, width=30, height=1, font=('Arial', self.size(12), "bold"), bg='white', bd=5)
        self.b_exp.grid(row=7, column=0)
        
        if self.lfs is not None:
            text_casa = 'Export All Data ( _Casa.vms )'
        else:
            text_casa = 'Export Data ( _Casa.vms )'
        self.b_exp_casa = tk.Button(self.fr_info, text=text_casa, command=self.__export_casa, width=30, height=1, font=('Arial', self.size(12), "bold"), bg='white', bd=5)
        self.b_exp_casa.grid(row=8, column=0)
        
        self.copy_button = tk.Button(self.fr_info, text="Copy Image to Clipboard", width=30, height=1, font=('Arial', self.size(12), "bold"), bg='white', fg='red', bd=5, command=self.__copy_to_clipboard)
        self.copy_button.grid(row=9, column=0)
        
        self.__trans_plot_job()
        # self.tpg.update()
        self.tpg.bind("<Return>", self.__rg_entry)
        if self.lfs is not None:
            self.tpg.bind('<Up>', self.__cf_up)
            self.tpg.bind('<Down>', self.__cf_down)
            self.tpg.bind('<MouseWheel>', self.__scroll)
        if g is not None:
            self.tpg.update()
            screen_width = self.tpg.winfo_reqwidth()
            screen_height = self.tpg.winfo_reqheight()
            t_sc_w = windll.user32.GetSystemMetrics(0)
            tx = int(t_sc_w*ScaleFactor/100) if g.winfo_x()+g.winfo_width()/2 > t_sc_w else 0
            self.tpg.geometry(f"{screen_width}x{screen_height}+{tx}+{sc_y}")
            self.tpg.protocol("WM_DELETE_WINDOW", self.closing)
            self.tpg.focus_force()
        else:
            self.tpg.update()
            screen_width = self.tpg.winfo_reqwidth()
            screen_height = self.tpg.winfo_reqheight()
            self.tpg.geometry(f"{screen_width}x{screen_height}+{0}+{sc_y}")
            self.tpg.mainloop()
    
    def closing(self):
        self.tpg.destroy()
        clear(self.lfs)
        clear(self)
        gc.collect()
    
    def __export(self):
        # os.chdir(self.rdd.removesuffix(self.rdd.split('/')[-1]))
        os.chdir(os.path.dirname(self.rdd))
        x, y = self.__sel_y()
        f = open(self.s_exp, 'w', encoding='utf-8')  # tab 必須使用 '\t' 不可"\t"
        f.write('Kinetic Energy'+'\t'+'Intensity'+'\n')
        for i in range(len(x)):
            f.write('%-6e' % x[i]+'\t'+'%-6e' % y[i]+'\n')
        f.close()
    
    # def __export_casa(self):
    # Casa.txt format simple version
    #     os.chdir(self.rdd.removesuffix(self.rdd.split('/')[-1]))
    #     x,y=self.e_photon-self.x,self.y
    #     f = open(self.s_exp_casa, 'w', encoding='utf-8')  # tab 必須使用 '\t' 不可"\t"
    #     f.write('#Wave Vector'+'\t'+'#Intensity'+'\n')
    #     for i in range(len(x)):
    #         f.write('%-6e' % x[i]+'\t'+'%-6e' % y[i]+'\n')
    #     f.close()
    def gen_casa_body(self):
        from datetime import datetime
        x, y = self.__sel_y()
        name = f'''{self.name}
Spectrum
'''
        current_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_")
        date = current_date.replace('_','\n')+'1\n'+str(len(self.tst.split('\n'))+5)+'\n'
        info = rf'''Casa Info Follows
0
0
0
0
{self.tst}'''
        if self.e_photon == 21.2:
            source = 'He I'
        elif self.e_photon == 40.8:
            source = 'He II'
        elif self.e_photon == 1253.6:
            source = 'Mg'
        elif self.e_photon == 1486.6:
            source = 'Al'
        elif self.e_photon == 3000.0:
            source = 'ES40'
        else:
            source = 'Sync'
        if '_' in self.name:
            n1, n2 = self.name.split('_')[0], ''
        else:
            n1, n2 = self.name, ''
        params = rf'''
XPS
0
{source}
{self.e_photon}
0
0
0
54
0
FAT
{self.dvalue[9].replace(' eV','')}
1e+037
0
0
1e+037
1e+037
1e+037
1e+037
{n1}
{n2}
-1
Kinetic Energy
eV
{np.max(self.ev)}
{self.dvalue[7].replace(' eV','')}
1
Intensity
d
pulse counting
{self.dvalue[11].replace(' s','')}
{self.dvalue[12]}
0
0.0
0.0
0.0
0
{len(x)}
0
1
'''
        data = ''
        for i in range(len(x)):
            data += f'{int(y[i])}\n'
        return name+date+info+params+data
        
    def __export_casa(self):
    # Casa.vms format complete version
        if self.lfs is not None:
            path = fd.asksaveasfilename(title="Save as", filetypes=(("VMS files", "*.vms"),), initialdir=self.lfs.path[0], initialfile=self.lfs.name[0], defaultextension='.vms')
            if path.split('.')[-1] != 'vms':
                path += '.vms'
            if path != '':
                # os.chdir(path.removesuffix(path.split('/')[-1]))
                os.chdir(os.path.dirname(path))
                f = open(path, 'w', encoding='utf-8')
                head = rf'''VAMAS Surface Chemical Analysis Standard Data Transfer Format 1988 May 4
    Not Specified
    PREVAC EA15
    2D Lab
    Not Specified
    3
    Casa Info Follows CasaXPS Version 2.3.18PR1.0
    0
    Number of Regions={len(self.lfs.name)}
    NORM
    REGULAR
    0
    1
    Data Set
    d
    0
    0
    0
    0
    {len(self.lfs.name)}
    '''     
                body = ''
                for i in self.lfs.data:
                    s=spectrogram(self.lfs.get(i), name='internal')
                    s.rr1, s.rr2 = self.rr1, self.rr2
                    body+=s.gen_casa_body()
                f.write(head+body+'end of experiment\n')
        else:
            # os.chdir(self.rdd.removesuffix(self.rdd.split('/')[-1]))
            os.chdir(os.path.dirname(self.rdd))
            f = open(self.s_exp_casa, 'w', encoding='utf-8')  # tab 必須使用 '\t' 不可"\t"
            head = r'''VAMAS Surface Chemical Analysis Standard Data Transfer Format 1988 May 4
Not Specified
PREVAC EA15
2D Lab
Not Specified
3
Casa Info Follows CasaXPS Version 2.3.18PR1.0
0
Number of Regions=1
NORM
REGULAR
0
1
Data Set
d
0
0
0
0
1
'''
# 1: Number of Regions

            f.write(head+self.gen_casa_body()+'end of experiment\n')
        f.close()
    
    # def __export_casa(self):
    # # Casa.txt format more complete version
    #     os.chdir(self.rdd.removesuffix(self.rdd.split('/')[-1]))
    #     x, y = self.__sel_y()
    #     f = open(self.s_exp_casa, 'w', encoding='utf-8')  # tab 必須使用 '\t' 不可"\t"
    #     f.write(f'[Info]\n'+
    #         f'Number of Regions=1\n'+
    #         f'[Region 1]\n'+
    #         f'Region Name={self.name}\n'+
    #         f'Dimension 1 name=Kinetic Energy [eV]\n'+
    #         f'Dimension 1 size={len(x)}\n'+
    #         f'Dimension 1 scale=')
    #     for i,v in enumerate(x):
    #         if i!=len(x)-1:
    #             f.write(f'{v} ')
    #         else:
    #             f.write(f'{v}\n')
    #     f.write(f'[Info 1]\n')
    #     key=['Region Name','Acquisition Mode','Energy Scale','Excitation Energy','Center Energy','High Energy','Low Energy','Energy Step','Lens Mode','Pass Energy','Slit','Step Time','Number of Sweeps','Description']
    #     for i in range(len(key)):
    #         if i<len(key)-1:
    #             if key[i]=='Step Time':
    #                 f.write(f"{key[i]}={int(float(self.dvalue[i].replace(' s',''))*1000)}\n")
    #             elif key[i]=='Pass Energy':
    #                 f.write(f"{key[i]}={int(float(self.dvalue[i].replace(' eV','')))}\n")
    #             elif key[i]=='Energy Scale' and self.e_mode=='Binding':
    #                 f.write(f"{key[i]}=Kinetic\n")
    #             elif key[i]=='Low Energy' and self.e_mode=='Binding':
    #                 f.write(f"{key[i]}={21.2-float(self.dvalue[5].replace(' eV','').replace(' (B.E.)',''))}\n")
    #             elif key[i]=='High Energy' and self.e_mode=='Binding':
    #                 f.write(f"{key[i]}={21.2-float(self.dvalue[6].replace(' eV','').replace(' (B.E.)',''))}\n")
    #             else:
    #                 f.write(f"{key[i]}={self.dvalue[i].replace(' eV','').replace(' (K.E.)','')}\n")
    #         else:
    #             f.write(f"{key[i]}={self.dvalue[i]}\n") 
    #     f.write(f'Detector First X-Channel=0\n'+
    #             f'Detector Last X-Channel=0\n'+
    #             f'Detector First Y-Channel=0\n'+
    #             f'Detector Last Y-Channel=0\n'+
    #             f'Number of Slices={len(self.phi)}\n'+
    #             f'spectrum Name={self.name}\n'+
    #             f'Comments={self.desc}; Slit: {self.dvalue[10]};\n')
    #     f.write(f'[Run Mode Information 1]\n'+
    #             f'Name=Normal\n')
    #     f.write(f'[Data1]\n')
    #     for i in range(len(x)):
    #         f.write('%-6e' % i+' '+'%-6e' % y[i]+'\n')
    #     f.close()
    
    def __rg_entry(self, *args):
        self.grg=RestrictedToplevel(self.tpg, bg='white')
        self.grg.title('Data Range')
        
        fr=tk.Frame(self.grg,bg='white')
        fr.pack(side=tk.TOP, padx=5, pady=5)
        
        self.v_r1=tk.DoubleVar(value=self.rr1)
        l_rr1in1 = tk.Label(fr, text='From', font=('Arial', self.size(16), "bold"), bg='white')
        self.rr1_in = tk.Entry(fr, font=("Arial", self.size(16), "bold"), width=10, textvariable=self.v_r1, bd=5)
        if self.lensmode == 'Transmission':
            l_rr1in2 = tk.Label(fr, text='mm', font=('Arial', self.size(16), "bold"), bg='white')
        else:
            l_rr1in2 = tk.Label(fr, text='deg', font=('Arial', self.size(16), "bold"), bg='white')
        l_rr1in1.grid(row=0,column=0)
        self.rr1_in.grid(row=0,column=1)
        l_rr1in2.grid(row=0,column=2)
        
        self.v_r2=tk.DoubleVar(value=self.rr2)
        l_rr2in1 = tk.Label(fr, text='To', font=('Arial', self.size(16), "bold"), bg='white')
        self.rr2_in = tk.Entry(fr, font=("Arial", self.size(16), "bold"), width=10, textvariable=self.v_r2, bd=5)
        if self.lensmode == 'Transmission':
            l_rr2in2 = tk.Label(fr, text='mm', font=('Arial', self.size(16), "bold"), bg='white')
        else:
            l_rr2in2 = tk.Label(fr, text='deg', font=('Arial', self.size(16), "bold"), bg='white')
        l_rr2in1.grid(row=1,column=0)
        self.rr2_in.grid(row=1,column=1)
        l_rr2in2.grid(row=1,column=2)
        
        fr1 = tk.Frame(self.grg,bg='white')
        fr1.pack(side=tk.TOP, padx=5, pady=5)
        b1=tk.Button(self.grg,text='Confirm',command=self.__save_rg, width=15, height=1, font=('Arial', self.size(14), "bold"), bg='white', bd=5)
        b1.pack(side=tk.TOP, padx=5, pady=5)
        
        self.grg.bind('<Return>', self.__save_rg)
        self.grg.focus_set()
        self.rr1_in.focus_set()
        self.rr1_in.select_range(0,tk.END)
        self.rr1_in.icursor(tk.END)
        self.grg.update()
        w = self.grg.winfo_reqwidth()
        h = self.grg.winfo_reqheight()
        self.grg.geometry(f"{int(w*1.4)}x{h}")  # Adjust height to fit the buttons
        set_center(self.tpg, self.grg, 0, 0)
        self.grg.update()
        self.grg.limit_bind()

    def __save_rg(self, *args):
        try:
            tmax = max([self.v_r1.get(), self.v_r2.get()])
            tmin = min([self.v_r1.get(), self.v_r2.get()])
            if tmin < min(self.phi) or tmax > max(self.phi):
                tk.messagebox.showwarning("Warning","Invalid Input\nThe range must be within the data range.")
                self.tpg.focus_set()
                self.grg.destroy()
                self.__rg_entry()
            else:
                self.rr1, self.rr2 = tmin, tmax
                self.grg.destroy()
                try:
                    self.r1.remove()
                    self.r2.remove()
                    self.s3.remove()
                except: pass
                self.s3,=self.tr_rga.plot([0, 0],[self.rr1, self.rr2],c='lightgreen',marker='<',markersize=self.scale*20,markerfacecolor='r',linewidth=self.scale*20)
                self.r1 = self.tr_a1.axhline(self.rr1, c='r')
                self.r2 = self.tr_a1.axhline(self.rr2, c='r')
                self.__tp_a2_plot(self.tr_a1.get_xlim()[0],self.tr_a1.get_xlim()[1])
                self.tpo.draw()
                self.rgo.draw()
                self.rpo.draw()
        except:
            tk.messagebox.showwarning("Warning","Invalid Input\n"+str(sys.exc_info()[1]))
            self.tpg.focus_set()
            self.grg.destroy()
            self.__rg_entry()
    
    def __rg_move(self, event):
        if event.inaxes:
            y = self.near(self.phi, event.ydata)
            if self.fr1==True:
                try:
                    # self.r1.remove()
                    self.s3.remove()
                except: pass
                self.rr1 = y
                self.s3,=self.tr_rga.plot([0, 0],[self.rr1, self.rr2],c='lightgreen',marker='<',markersize=self.scale*20,markerfacecolor='r',linewidth=self.scale*20)
                # self.r1 = self.tr_a1.axhline(self.rr1, c='r')
            elif self.fr2==True:
                try:
                    # self.r2.remove()
                    self.s3.remove()
                except: pass
                self.rr2 = y
                self.s3,=self.tr_rga.plot([0, 0],[self.rr1, self.rr2],c='lightgreen',marker='<',markersize=self.scale*20,markerfacecolor='r',linewidth=self.scale*20)
                # self.r2 = self.tr_a1.axhline(self.rr2, c='r')
            elif self.fr3==True:
                try:
                    # self.r1.remove()
                    # self.r2.remove()
                    self.s3.remove()
                except: pass
                if y-self.roy+self.romin < min(self.phi):
                    self.rr1 = min(self.phi)
                    self.rr2 = min(self.phi)+(self.romax-self.romin)
                elif y-self.roy+self.romax > max(self.phi):
                    self.rr2 = max(self.phi)
                    self.rr1 = max(self.phi)-(self.romax-self.romin)
                else:
                    self.rr1 = y-self.roy+self.romin
                    self.rr2 = y-self.roy+self.romax
                
                # self.r1 = self.tr_a1.axhline(self.rr1, c='r')
                # self.r2 = self.tr_a1.axhline(self.rr2, c='r')
                self.s3,=self.tr_rga.plot([0, 0],[self.rr1, self.rr2],c='lightgreen',marker='<',markersize=self.scale*20,markerfacecolor='r',linewidth=self.scale*20)
        self.__tp_a2_plot(self.tr_a1.get_xlim()[0],self.tr_a1.get_xlim()[1])
        self.tpo.draw()
        # self.rpo.draw()
        self.rgo.draw()
    
    def __rg_press(self, event):
        if event.button == 1 and event.inaxes:
            y = self.near(self.phi, event.ydata)
            self.fr1 = False
            self.fr2 = False
            self.fr3 = False
            self.roy = self.near(self.phi, event.ydata)
            self.rr1, self.rr2 = sorted([self.near(self.phi, self.rr1), self.near(self.phi, self.rr2)])
            self.romin = self.rr1
            self.romax = self.rr2
            if abs(self.rr1-y) < (self.phi[1]-self.phi[0])*len(self.phi)*1/40:
                try:
                    # self.r1.remove()
                    self.s3.remove()
                except: pass
                self.fr1 = True
                self.rr1 = y
                
            elif abs(self.rr2-y) < (self.phi[1]-self.phi[0])*len(self.phi)*1/40:
                try:
                    # self.r2.remove()
                    self.s3.remove()
                except: pass
                self.fr2 = True
                self.rr2 = y
            elif self.rr1 < y < self.rr2:
                try:
                    # self.r1.remove()
                    # self.r2.remove()
                    self.s3.remove()
                except: pass
                self.fr3 = True
        elif event.button == 3 and event.inaxes:
            self.__rg_entry()
        self.rgo.draw()
        self.rpo.draw()
        
    def __rg_release(self, event):
        self.fr1 = False
        self.fr2 = False
        self.fr3 = False
        self.__re_tr_a1_plot(self.tr_a1.get_xlim()[0],self.tr_a1.get_xlim()[1])
        self.rpo.draw()
        
    
    def __rp_move(self, event):
        # global rpf, rpo, tpf, tpo, tr_a1, tr_a2, xx2, yy2, aa1, aa2, cur, l_cx, l_cy, l_dy
        self.rpf.canvas.get_tk_widget().config(cursor="")
        if event.inaxes:
            self.rpf.canvas.get_tk_widget().config(cursor="tcross")
            self.out = False
            try:
                # self.xx1.remove()
                self.xx2.remove()
                self.yy2.remove()
                self.cur.remove()
                # self.aa1.remove()
                self.aa2.remove()
            except:
                pass
            if self.lensmode == 'Transmission':
                unit=' mm'
            else:
                if self.npzf:unit=' 2pi/A'
                else:unit = ' deg'
            if event.xdata>self.ev[-1]:
                cxdata = self.ev[-1]
            elif event.xdata<self.ev[0]:
                cxdata = self.ev[0]
            else:
                cxdata = event.xdata
            cydata = event.ydata
            self.tx = cxdata
            xf = (cxdata > self.oxl[0] and cxdata < self.oxl[1])
            yf = (cydata > self.tr_a1.get_ylim()[0] and cydata < self.tr_a1.get_ylim()[1])
            if xf and yf:
                tz = self.data.to_numpy().transpose()
                x = self.x
                y = self.y
                yy = self.phi
                xi = 0
                yi = 0
                
                if cxdata < x[0]:
                    xi=0
                elif cxdata > x[-1]:
                    xi=len(x)-1
                else:
                    xi=np.argwhere(abs(x-cxdata) <= (x[1]-x[0])/2)[0][0]
                if cydata < yy[0]:
                    yi=0
                elif cydata > yy[-1]:
                    yi=len(yy)-1
                else:
                    yi=np.argwhere(abs(yy-cydata) <= (yy[1]-yy[0])/2)[0][0]
                    
                try:
                    self.l_cx.config(text='%9s%8.3f%3s'%('Energy : ',cxdata,' eV'))
                    self.l_cy.config(text='%10s%11.4g%4s'%('Cursor : ',cydata,unit))
                    self.l_dy.config(text='%11s%11.4g'%('Data : ',tz[yi][xi]))
                except:
                    pass
                # self.xx1.set_data([cxdata,cxdata], self.oy1)
                # self.xx1=self.tr_a1.axvline(cxdata,color='g')
                self.xx2=self.tr_a2.axvline(cxdata,color='g')
                self.yy2=self.tr_a2.axhline(-max(y),color='grey')
                
                x, y = self.__sel_y()
                    
                x,y=x[xi],y[xi]
                self.cur=self.tr_a2.scatter(x,y,c='r',marker='o',s=self.scale*self.scale*30)
                self.tr_a2.set_ylim(self.oy2)
                if not self.tp_cf:
                    # self.aa1=self.tr_a1.fill_between([self.ox,cxdata],self.oy1[0],self.oy1[1],color='g',alpha=0.2)
                    self.aa2=self.tr_a2.fill_between([self.ox,cxdata],self.oy2[0],self.oy2[1],color='g',alpha=0.2)
                    self.tr_a2.set_ylim(self.oy2)
        else:
            try:
                self.l_cx.config(text='%9s'%'Energy : ')
                self.l_cy.config(text='%10s'%'Cursor : ')
                self.l_dy.config(text='%11s'%'Data : ')
                # self.xx1.remove()
                self.xx2.remove()
                self.yy2.remove()
                self.cur.remove()
                # self.xx2.remove()
                # self.yy2.remove()
                # self.cur.remove()
                # self.aa1.remove()
                # self.aa2.remove()
            except:
                pass
        # self.rpo.draw()
        self.tpo.draw()

    def __rp_press(self, event):
        # global tp_cf, rpf, rpo ,tpf, tpo , tr_a1, tr_a2 , x1 , x2 , ox, aa1, aa2
        if event.button == 1 and self.tp_cf:
            self.tp_cf = False
            self.out=True
            # self.x1 = self.tr_a1.axvline(event.xdata, color='g')
            # self.x1.set_data([event.xdata,event.xdata], self.oy1)
            self.x2 = self.tr_a2.axvline(event.xdata, color='g')
            self.tr_a2.set_ylim(self.oy2)
            self.ox=event.xdata

        elif event.button == 3:
            self.rpf.canvas.get_tk_widget().config(cursor="watch")
            self.tp_cf = True
            self.__tp_a1_plot()
            self.__tp_a2_plot(self.oxl[0],self.oxl[1])
            self.rpo.draw()
            self.rpf.canvas.get_tk_widget().config(cursor="tcross")
        
        self.tpo.draw()
        

    def __rp_release(self, event):
        # global tp_cf, rpf, rpo ,tpf, tpo , tr_a1, tr_a2, x1, x2 , ox, aa1, aa2
        if event.button == 1 and not self.tp_cf:
            self.rpf.canvas.get_tk_widget().config(cursor="watch")
            self.tp_cf = True
            try:
                # self.x1.remove()
                # self.x1.set_data([],[])
                self.x2.remove()
                # self.aa1.remove()
                self.aa2.remove()
            except:
                pass
            if self.out == False:
                self.__re_tr_a1_plot(sorted([self.ox, self.tx])[0],sorted([self.ox, self.tx])[1])
                self.__tp_a2_plot(sorted([self.ox, self.tx])[0],sorted([self.ox, self.tx])[1])
            else:
                self.__re_tr_a1_plot(self.oxl[0],self.oxl[1])
                self.__tp_a2_plot(self.oxl[0],self.oxl[1])
            self.rpo.draw()
            self.tpo.draw()
            self.rpf.canvas.get_tk_widget().config(cursor="tcross")
        
    def __tp_move(self, event):
        # global tpf, tpo, tr_a1, tr_a2, tpf, xx2, yy2, aa1, aa2, cur, l_cx, l_cy, l_dy
        self.tpf.canvas.get_tk_widget().config(cursor="")
        if event.inaxes:
            self.out = False
            try:
                self.xx2.remove()
                self.yy2.remove()
                self.cur.remove()
                # self.aa1.remove()
                self.aa2.remove()
            except:
                pass
            self.tpf.canvas.get_tk_widget().config(cursor="tcross")
            if event.xdata>self.ev[-1]:
                cxdata = self.ev[-1]
            elif event.xdata<self.ev[0]:
                cxdata = self.ev[0]
            else:
                cxdata = event.xdata
            cydata = event.ydata
            self.tx = cxdata
            xf = (cxdata >= self.oxl[0] and cxdata <= self.oxl[1])
            yf = (cydata >= self.tr_a2.get_ylim()[0] and cydata <= self.tr_a2.get_ylim()[1])
            if xf and yf:
                y = self.y
                x = self.x
                xi = 0
                
                if cxdata < x[0]:
                    xi=0
                elif cxdata > x[-1]:
                    xi=len(x)-1
                else:
                    xi=np.argwhere(abs(x-cxdata) <= (x[1]-x[0])/2)[0][0]
                
                x, y = self.__sel_y()
                    
                x,y=x[xi],y[xi]
                try:
                    self.l_cx.config(text='%9s%8.3f%3s'%('Energy : ',cxdata,' eV'))
                    self.l_cy.config(text='%10s%11.4g'%('Cursor : ',cydata))
                    self.l_dy.config(text='%11s%11.4g'%('Data : ',y))
                except:
                    pass
                self.xx2=self.tr_a2.axvline(cxdata,color='g')
                self.yy2=self.tr_a2.axhline(cydata,color='grey')
                self.cur=self.tr_a2.scatter(x,y,c='r',marker='o',s=self.scale*self.scale*30)
                self.tr_a2.set_ylim(self.oy2)
                if not self.tp_cf:
                    # self.aa1=self.tr_a1.fill_between([self.ox,cxdata],self.oy1[0],self.oy1[1],color='g',alpha=0.2)
                    self.aa2=self.tr_a2.fill_between([self.ox,cxdata],self.oy2[0],self.oy2[1],color='g',alpha=0.2)
                    self.tr_a2.set_ylim(self.oy2)
        else:
            try:
                self.l_cx.config(text='%9s'%'Energy : ')
                self.l_cy.config(text='%10s'%'Cursor : ')
                self.l_dy.config(text='%11s'%'Data : ')
                self.xx2.remove()
                self.yy2.remove()
                self.cur.remove()
                # self.xx2.remove()
                # self.yy2.remove()
                # self.cur.remove()
                # self.aa1.remove()
                # self.aa2.remove()
            except:
                pass
        # self.rpo.draw()
        self.tpo.draw()


    def __tp_press(self, event):
        # global tp_cf, rpf, rpo ,tpf, tpo , tr_a1, tr_a2 , x1 , x2 , ox, aa1, aa2
        if event.button == 1 and self.tp_cf:
            self.tp_cf = False
            self.out=True
            # self.x1 = self.tr_a1.axvline(event.xdata, color='g')
            self.x2 = self.tr_a2.axvline(event.xdata, color='g')
            self.tr_a2.set_ylim(self.oy2)
            self.ox=event.xdata

        elif event.button == 3:
            self.tpf.canvas.get_tk_widget().config(cursor="watch")
            self.tp_cf = True
            self.__tp_a1_plot()
            self.__tp_a2_plot(self.oxl[0],self.oxl[1])
            self.rpo.draw()
            self.tpf.canvas.get_tk_widget().config(cursor="tcross")
        
        self.tpo.draw()
        
        
    def __tp_release(self, event):
        # global tp_cf, rpf, rpo ,tpf, tpo , tr_a1, tr_a2, x1, x2 , ox, aa1, aa2
        if event.button == 1 and not self.tp_cf:
            self.tpf.canvas.get_tk_widget().config(cursor="watch")
            self.tp_cf = True
            try:
                # self.x1.remove()
                self.x2.remove()
                # self.aa1.remove()
                self.aa2.remove()
            except:
                pass
            if self.out == False:
                self.__re_tr_a1_plot(sorted([self.ox, self.tx])[0],sorted([self.ox, self.tx])[1])
                self.__tp_a2_plot(sorted([self.ox, self.tx])[0],sorted([self.ox, self.tx])[1])
            else:
                self.__re_tr_a1_plot(self.oxl[0],self.oxl[1])
                self.__tp_a2_plot(self.oxl[0],self.oxl[1])
            self.rpo.draw()
            self.tpo.draw()
            self.tpf.canvas.get_tk_widget().config(cursor="tcross")
            
    def __re_tr_a1_plot(self,xx1,xx2):
        z = self.data.to_numpy().transpose()
        # self.tr_a1.scatter(self.ev, np.sum(tz,axis=0), c='k', marker='o', s=self.scale*self.scale*0.9)
        x = self.ev
        xi=[]
        x1, x2 = sorted([xx1, xx2])
        xx1, xx2 = self.near(x, xx1), self.near(x, xx2)
        for i,v in enumerate(x):
            if v>=xx1 and v<=xx2:
                xi.append(i)
        x = x[xi]
        tx, ty = np.meshgrid(x, self.phi)
        tz = tx*0
        tz[0:, 0:] = z[0:, xi]
        # ttx = np.linspace(min(x),max(x),len(x)*4)
        # tx, ty = np.meshgrid(ttx, self.phi)
        # x, y = np.meshgrid(x, self.phi)
        # tz = griddata((x.flatten(), y.flatten()), tz.flatten(), (tx, ty), method='cubic')
        self.tr_a1.clear()
        self.tr_a1.pcolormesh(tx,ty,tz,cmap=self.cmap)
        tx, ty, tz = None, None, None
        self.r1=self.tr_a1.axhline(self.rr1, c='r')
        self.r2=self.tr_a1.axhline(self.rr2, c='r')
        if self.lensmode=='Transmission':
            self.tr_a1.set_ylabel('Position (mm)', font='Arial', fontsize=self.size(16))
        else:
            if self.npzf:self.tr_a1.set_ylabel(r'k ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=self.size(16))
            else:self.tr_a1.set_ylabel('Angle (deg)', font='Arial', fontsize=self.size(16))
                
        self.tr_a1.set_xticks([])
        self.tr_a1.set_yticklabels(labels=self.tr_a1.get_yticklabels(), font='Arial', fontsize=self.size(14))
        self.tr_a1.set_xlim([x1, x2])
        self.tr_a1.set_ylim(self.oy1)
        
    def __tp_a1_plot(self):
        # global tr_a2, oy2
        tx, ty = np.meshgrid(self.ev, self.phi)
        tz = self.data.to_numpy().transpose()
        # self.tr_a1.scatter(self.ev, np.sum(tz,axis=0), c='k', marker='o', s=self.scale*self.scale*0.9)
        self.tr_a1.clear()
        self.tr_a1.pcolormesh(tx,ty,tz,cmap=self.cmap)
        tx, ty, tz = None, None, None
        self.r1=self.tr_a1.axhline(self.rr1, c='r')
        self.r2=self.tr_a1.axhline(self.rr2, c='r')
        if self.lensmode=='Transmission':
            self.tr_a1.set_ylabel('Position (mm)', font='Arial', fontsize=self.size(16))
        else:
            if self.npzf:self.tr_a1.set_ylabel(r'k ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=self.size(16))
            else:self.tr_a1.set_ylabel('Angle (deg)', font='Arial', fontsize=self.size(16))
        self.tr_a1.set_xticks([])
        self.tr_a1.set_yticklabels(labels=self.tr_a1.get_yticklabels(), font='Arial', fontsize=self.size(14))
        # self.tr_a1.set_xlim(self.tr_a1.get_xlim())
        # self.x1, = self.tr_a1.plot([],[],'g-')
        # self.xx1, = self.tr_a1.plot([],[],'g-')
        self.tr_a1.set_xlim([sorted([self.ev[0], self.ev[-1]])[0]-abs(self.ev[-1]-self.ev[0])/50, sorted([self.ev[0], self.ev[-1]])[1]+abs(self.ev[-1]-self.ev[0])/50])
        self.tr_a1.set_ylim([sorted([self.phi[0], self.phi[-1]])[0]-abs(self.phi[-1]-self.phi[0])/20, sorted([self.phi[0], self.phi[-1]])[1]+abs(self.phi[-1]-self.phi[0])/20])
        self.oxl=self.tr_a1.get_xlim()
        self.oy1=self.tr_a1.get_ylim()
    
    def __tp_a2_plot(self,xx1,xx2):
        # global tr_a2, oy2
        x, y = self.__sel_y()
        xi=[]
        for i,v in enumerate(x):
            if v>=xx1 and v<=xx2:
                xi.append(i)
        x = x[xi]
        y = y[xi]
        self.tr_a2.clear()
        if self.type=='fd':
            self.tr_a2.plot(x,y, color='k')
        else:
            if abs(xx1-xx2)>abs(self.oxl[1]-self.oxl[0])/2:
                self.tr_a2.scatter(x,y, c='k', marker='o', s=self.scale*self.scale*0.9)
            elif abs(xx1-xx2)>abs(self.oxl[1]-self.oxl[0])/4:
                self.tr_a2.scatter(x,y, c='k', marker='o', s=self.scale*self.scale*10)
            else:
                self.tr_a2.scatter(x,y, c='k', marker='o', s=self.scale*self.scale*30)
        self.tr_a2.ticklabel_format(style='plain', axis='y', scilimits=(0,0))
        self.tr_a2.set_xlim(self.tr_a1.get_xlim())
        self.tr_a2.set_xlabel('Kinetic Energy (eV)', font='Arial', fontsize=self.size(16))
        self.tr_a2.set_ylabel(self.s_yl, font='Arial', fontsize=self.size(16))
        self.tr_a2.set_xticklabels(labels=self.tr_a2.get_xticklabels(), font='Arial', fontsize=self.size(14))
        self.tr_a2.set_yticklabels(labels=self.tr_a2.get_yticklabels(), font='Arial', fontsize=self.size(14))
        self.oy2=self.tr_a2.get_ylim()

    def __tp_rga_plot(self):
        self.s3,=self.tr_rga.plot([0, 0],[self.rr1, self.rr2],c='lightgreen',marker='<',markersize=self.scale*20,markerfacecolor='r',linewidth=self.scale*20)
        self.tr_rga.set_ylim([sorted([self.phi[0], self.phi[-1]])[0]-abs(self.phi[-1]-self.phi[0])/20, sorted([self.phi[0], self.phi[-1]])[1]+abs(self.phi[-1]-self.phi[0])/20])
        self.tr_rga.set_xticks([])
        self.tr_rga.set_yticks([])
    
    def __trans_plot_job(self):
        # global rpf,rpo,tpf,tpo,tr_a1,tr_a2,oxl,oy1
        self.tr_a1=self.rpf.add_axes([0.1, 0.05, 0.88, 0.9])
        self.tr_a1.set_facecolor('lightblue')
        self.tr_a2=self.tpf.add_axes([0.1, 0.15, 0.88, 0.82])
        self.tr_rga=self.rgf.add_axes([0, 0.05, 1, 0.9])
        self.tr_rga.set_facecolor('lightblue')
        self.__tp_a1_plot()
        self.__tp_a2_plot(self.oxl[0],self.oxl[1])
        self.__tp_rga_plot()
        # self.rpf.tight_layout()
        # self.tpf.tight_layout()
        self.rpo.draw()
        self.tpo.draw()
        self.rgo.draw()

def cut_job_y(args):
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

def set_entry_value(entry, value):
    entry.delete(0, tk.END)
    entry.insert(0, value)

def mesh(x, y):
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

def rotate(data, angle, size):
    """
    for square data
    """
    mat = cv2.getRotationMatrix2D((size[1]/2, size[0]/2), angle, 1)
    data = cv2.warpAffine(data, mat, (size[1], size[0]), flags=cv2.INTER_NEAREST)
    return data

h=6.62607015e-34  # J·s
m=9.10938356e-31  # kg
class SO_Fitter(tk.Toplevel):
    def __init__(self, master, app_pars: app_param=None):
        super().__init__(master, background='white')
        self.title('Sample Offset Fitter')
        self.ev=1.602176634e-19  # eV=1.602176634e-19 J
        self.hbar=h/2/np.pi
        self.e=21.2   # need an entry
        # self.e=20.781
        self.r10=-31
        self.phi0=0
        self.r11=0
        self.phi1=0
        self.tolerance = 0.03   # tolerance for dot product (dominance term) # need an entry
        self.app_pars = app_pars
        self.size = lambda s: int(s * self.app_pars.scale)
        
        self.p_list = []    # R2 orderd list
        # Trial data points
        # p1 = [20.2-20.2, 20, 0]
        # p2 = [80.2-20.2, 22.5, -0.2]
        # p3 = [206-20.2, 19.5, 0]
        # p1 = [20.2, 20, 0]
        # p2 = [80.2, 22.5, -0.2]
        # p3 = [206, 19.5, 0]
        # self.p_list.append(p1)
        # self.p_list.append(p2)
        # self.p_list.append(p3)
        self.k_vec = np.sqrt(2*m*self.e*self.ev)/self.hbar*1e-10
        
        self.layout()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.update()
        
    def on_closing(self):
        self.destroy()
        clear(self)
        gc.collect()
        
    
    @staticmethod
    def rot(x, y, angle):
        '''
        rotate the image with the given angle under the offset
        '''
        angle *= np.pi / 180
        c, s = np.cos(angle), np.sin(angle)
        x_rot = x * c - y * s   # ndimage.rotate in the process
        y_rot = x * s + y * c
        return x_rot, y_rot
    
    def kxy(self, r2, r1, phi, r10=-31, phi0=0, r11=0, phi1=0):
        r11, phi1 = self.rot(r11, phi1, -r2)
        kx = self.k_vec * np.sin(np.deg2rad(r1-r10-r11)) * np.cos(np.deg2rad(phi-phi0-phi1))
        ky = self.k_vec * np.sin(np.deg2rad(phi-phi0-phi1))
        kx, ky = self.rot(kx, ky, -r2)
        return kx, ky

    @staticmethod
    def cal_r(kx,ky):
        kx, ky = np.array(kx), np.array(ky)
        r = np.sqrt(kx**2 + ky**2)
        return r

    @staticmethod
    def cal_dot(kx1,ky1,kx2,ky2):
        r1, r2 = SO_Fitter.cal_r(kx1, ky1), SO_Fitter.cal_r(kx2, ky2)
        kx1, ky1 = np.array(kx1)/r1, np.array(ky1)/r1
        kx2, ky2 = np.array(kx2)/r2, np.array(ky2)/r2
        dot = kx1 * kx2 + ky1 * ky2
        return dot

    @staticmethod
    def find_min_distance(r11, phi1, r12, phi2):
        min_r = np.inf
        tr12 = 0
        tphi2 = 0
        for r1, p1 in zip(r11, phi1):
            for r2, p2 in zip(r12, phi2):
                dist = np.sqrt((r1 - r2)**2 + (p1 - p2)**2)
                if dist < min_r:
                    min_r = dist
                    tr12 = r2
                    tphi2 = p2
        return min_r, r12[r12==tr12], phi2[phi2==tphi2]
    
    def check_circ(self, p_list, rdot, case='before'):
        ax, figout = (self.ax0, self.figout0) if case=='before' else (self.ax1, self.figout1)
        
        tx, ty = [], []
        for i, p in enumerate(p_list):
            kx, ky = self.kxy(*p, r10=self.r10, phi0=self.phi0, r11=self.r11, phi1=self.phi1)
            tx.append(kx)
            ty.append(ky)
            s = ax.scatter(kx, ky, label=f'R2={p[0]+self.r2_min}, R1={p[1]}, phi={p[2]}')
            s.set_zorder(10)
        tr = SO_Fitter.cal_r(tx, ty)
        
        dot = []
        for i in range(len(p_list)):
            dot.append(SO_Fitter.cal_dot(tx[i], ty[i], tx[(i+1)%len(p_list)], ty[(i+1)%len(p_list)]))
            
        circle = plt.Circle((0, 0), radius=(kx**2 + ky**2)**0.5, color='gray', linestyle='--', fill=False, label='circle')
        s = ax.scatter(0, 0, color='red', label='origin')
        s.set_zorder(11)
        ax.add_patch(circle)
        
        ### Check
        f=False
        kx, ky = [], []
        for r in p_list[1:]:
            kx_, ky_ = self.kxy(*r, r10=self.r10, phi0=self.phi0, r11=self.r11, phi1=self.phi1)
            kx.append(kx_)
            ky.append(ky_)
        try:
            kx.append(kx[1]-kx[0])
            ky.append(ky[1]-ky[0])
        except IndexError:
            pass
        kx0, ky0 = [kx[0]], [ky[0]]
        tx, ty = self.kxy(*p_list[0], r10=self.r10, phi0=self.phi0, r11=self.r11, phi1=self.phi1)
        kx0.append(tx)
        ky0.append(ty)
        kx0.append(0)
        ky0.append(0)
        
        f = True
        rf = True
        for i in range(len(dot)):
            if abs(dot[i]-rdot[i]) > self.tolerance:
                f = False
                break
            if abs(tr[i]-tr[(i+1)%len(tr)]) > self.tolerance:
                f = False
                rf = False
                break
            
        if case=='before':
            print('\033[36m\nBefore fitting:\033[0m')
            print('%18s'%'Angle (deg): ', [abs(i-j) for i,j in zip([p[0] for p in p_list], np.roll([p[0] for p in p_list], -1))])
            print('%18s'%'Radius Diff: ', [tr[i]-tr[(i+1)%len(tr)] for i in range(len(tr))])
            print('%18s'%'Dot Product Diff: ', [i-j for i,j in zip(dot, rdot)])
        else:
            print('\033[36m\nAfter fitting:\033[0m')
            
            ang = np.arccos(dot)*180/np.pi
            str_ang = '['
            for i in ang:
                str_ang += f'{i:.3f}, '
            str_ang = str_ang[:-2] + ']'
            
            if rf:
                r_str = [tr[i]-tr[(i+1)%len(tr)] for i in range(len(tr))]
                dot_str = '['
                for i, j in zip(dot, rdot):
                    dot_str += '\033[32m%s\033[0m, '%(i-j) if abs(i-j)<=self.tolerance else '\033[31m%s\033[0m, '%(i-j)
                dot_str = dot_str[:-2] + ']'
                fail_str = '\033[31mDot product difference exceeded the tolerance.\033[0m'
            else:
                r_str = '['
                for i in range(len(tr)):
                    r_str += '\033[31m%s\033[0m, '%(tr[i]-tr[(i+1)%len(tr)]) if abs(tr[i]-tr[(i+1)%len(tr)])>self.tolerance else '\033[0m%s\033[0m, '%(tr[i]-tr[(i+1)%len(tr)])
                r_str = r_str[:-2] + ']'
                dot_str = [i-j for i,j in zip(dot, rdot)]
                fail_str = '\033[31mRadius differences are ridiculously large!\nCheck R2/R1/Phi values of data points.\033[0m'
                
            if f:
                try:
                    kx.remove(kx[1]-kx[0])
                    ky.remove(ky[1]-ky[0])
                except IndexError:
                    pass
                kx.insert(0, tx)
                ky.insert(0, ty)
                for i in range(len(kx)):
                    ax.plot([0, kx[i]], [0, ky[i]], 'g-')
                    if i<len(kx)-1:
                        ax.plot([kx[i], kx[i+1]], [ky[i], ky[i+1]], 'g-')
                
                print('%18s'%'Angle (deg): ', str_ang)
                print('%18s'%'Radius Diff: ', r_str)
                print('%18s'%'Dot Product Diff: ', dot_str)
                print('%18s'%'Status: ', f'\033[32mPass \033[0m(Tolerance=\033[32m{self.tolerance}\033[0m)')
            else:
                print('%18s'%'Angle (deg): ', str_ang)
                print('%18s'%'Radius Diff: ', r_str)
                print('%18s'%'Dot Product Diff: ', dot_str)
                print('%18s'%'Status: ', f'\033[31mFail \033[0m(Tolerance=\033[31m{self.tolerance}\033[0m)')
                print('%18s'%' ', fail_str.split('\n')[0])
                print('%18s'%' ', fail_str.split('\n')[1] if len(fail_str.split('\n'))>1 else '')
                
        l = ax.legend(fontsize=self.size(14))
        l.set_zorder(12)
        l.set_draggable(True)
        figout.draw()
        return dot, tr
    
    @property
    def rdot(self):
        d_ang = [30, 60, 90, 120, 180]  # 12, 6, 4, 3, 2 fold
        ang = np.array([i[0] for i in self.p_list])
        d_ang_t = np.abs(np.diff(ang)).min()
        ang[0] += 360
        d_ang_r_t = np.abs(np.diff(np.roll(ang, 1))).min()
        d_ang_t = min(d_ang_t, d_ang_r_t)
        d = np.abs(d_ang-d_ang_t)
        idx = d.argmin()
        ang -= np.roll(ang, -1)
        if d_ang[idx]==30:
            rdot = np.cos(np.round(ang/30)*30/180*np.pi)
            ang = np.round(ang/30)*30
        elif d_ang[idx]==60:
            rdot = np.cos(np.round(ang/60)*60/180*np.pi)
            ang = np.round(ang/60)*60
        elif d_ang[idx]==90:
            rdot = np.cos(np.round(ang/90)*90/180*np.pi)
            ang = np.round(ang/90)*90
        elif d_ang[idx]==120:
            rdot = np.cos(np.round(ang/120)*120/180*np.pi)
            ang = np.round(ang/120)*120
        elif d_ang[idx]==180:
            rdot = np.cos(np.round(ang/180)*180/180*np.pi)
            ang = np.round(ang/180)*180
        return rdot

    def _fit(self):
        import copy
        self.r2_min = min([i[0] for i in self.p_list])
        p_list = copy.deepcopy(self.p_list)
        for i, v in enumerate(p_list):
            p_list[i][0] = v[0] - self.r2_min
        self.clf()
        self.r11, self.phi1 = 0, 0
        
        self.check_circ(p_list=p_list, rdot=self.rdot, case='before')
        
        r11 = np.linspace(-7.5, 7.5, 751)   # entry
        phi1 = np.linspace(-7.5, 7.5, 751)  # entry
        r11, phi1 = np.meshgrid(r11, phi1)
        kx, ky = [], []
        for p in p_list:
            kx_, ky_ = self.kxy(*p, r10=self.r10, phi0=self.phi0, r11=r11, phi1=phi1)
            kx.append(kx_)
            ky.append(ky_)
        r, dot = np.zeros(kx[0].shape), np.zeros(kx[0].shape)
        r_avg = []
        for i in range(len(p_list)-1):
            t_r = SO_Fitter.cal_r(kx[i], ky[i])
            r_avg.append(t_r)
            r+=np.abs(t_r-SO_Fitter.cal_r(kx[i+1], ky[i+1]))
            dot+=np.abs(SO_Fitter.cal_dot(kx[i], ky[i], kx[i+1], ky[i+1])-self.rdot[i])
        r_avg.append(SO_Fitter.cal_r(kx[-1], ky[-1]))
        r_avg = np.average(np.array(r_avg).flatten())
        value = r+dot*r_avg # weighted value d(r), rd(theta)
        mask_min = np.argmin(value)
        self.info.config(state='normal')
        self.info.insert(tk.END, f'Sample Offset:\n\tR1_Offset={r11.flatten()[mask_min]:.3f}\n\tPhi_Offset={phi1.flatten()[mask_min]:.3f}\n\n')
        self.info.see(tk.END)
        self.info.config(state='disabled')
        r11, phi1 = r11.flatten()[mask_min], phi1.flatten()[mask_min]
        self.r11, self.phi1 = r11, phi1
        
        self.check_circ(p_list=p_list, rdot=self.rdot, case='after')
        
        print('\nvalue = |d(Radius)| + |d(Dot Product)*Radius|')
        print('minimum value: ', '%s'%value.flatten()[mask_min],'')
        print('\033[33mPhi sample offset: ', '%.3f'%phi1,'\nR1 sample offset: ', '%.3f'%r11, '\n\033[0m')
        
        if self.app_pars:
            windll.user32.ShowWindow(self.app_pars.hwnd, 9)
            windll.user32.SetForegroundWindow(self.app_pars.hwnd)
    
    def clf(self):
        self.ax0.cla()
        self.ax0.set_title('Before', fontsize=self.size(20))
        self.ax0.set_xlabel(r'$k_x$ (2$\pi$/Å)')
        self.ax0.set_ylabel(r'$k_y$ (2$\pi$/Å)')
        self.ax0.set_aspect('equal')
        
        self.ax1.cla()
        self.ax1.set_title('After', fontsize=self.size(20))
        self.ax1.set_xlabel(r'$k_x$ (2$\pi$/Å)')
        self.ax1.set_ylabel(r'$k_y$ (2$\pi$/Å)')
        self.ax1.set_aspect('equal')
    
    def entry_select_all(self, event):
        event.widget.select_range(0, tk.END)
    
    def clear_points(self):
        self.clf()
        
        self.p_list = []
        self.info.config(state='normal')
        self.info.delete(1.0, tk.END)
        self.info.config(state='disabled')
        
    def add_point(self):
        self.e = self.v_e.get()
        self.k_vec = np.sqrt(2*m*self.e*self.ev)/self.hbar*1e-10
        r2 = self.v_r2.get()
        r1 = self.v_r1.get()
        phi = self.v_phi.get()
        if (r2, r1, phi) in self.p_list or (r2==0 and r1==0 and phi==0):
            self.v_r2.set(0)
            self.v_r1.set(0)
            self.v_phi.set(0)
            return
        self.p_list.append([r2, r1, phi])
        self.p_list = sorted(self.p_list, key=lambda x: x[0])  # sort by R2
        self.info.config(state='normal')
        self.info.insert(tk.END, f'Added Point: R2={r2}, R1={r1}, Phi={phi}\n')
        self.info.see(tk.END)
        self.info.config(state='disabled')
        self.v_r2.set(0)
        self.v_r1.set(0)
        self.v_phi.set(0)
        self.r2_entry.focus_set()

    def set_tolerance(self):
        self.tolerance = self.v_tol.get()
    
    def layout(self):
        frame_left = tk.Frame(self, background='white')
        frame_left.grid(row=0, column=0)
        frame_middle = tk.Frame(self, background='white')
        frame_middle.grid(row=0, column=1)
        frame_right = tk.Frame(self, background='white')
        frame_right.grid(row=0, column=2)
        
        fr_info = tk.Frame(frame_left, background='white')
        fr_info.pack()
        xscroll = tk.Scrollbar(fr_info, orient='horizontal', bg='white')
        xscroll.pack(side='bottom', fill='x')
        yscroll = tk.Scrollbar(fr_info, orient='vertical', bg='white')
        yscroll.pack(side='right', fill='y')
        self.info = tk.Text(fr_info, width=40, height=20, background='white', borderwidth=5, state='disabled', xscrollcommand=xscroll.set, yscrollcommand=yscroll.set, font=('Arial', self.size(16), 'bold'))
        self.info.pack()
        xscroll.config(command=self.info.xview)
        yscroll.config(command=self.info.yview)
        
        fr_ke = tk.Frame(frame_left, background='white')
        fr_ke.pack()
        l_e = tk.Label(fr_ke, text='Kinetic Energy (eV): ', background='white', font=('Arial', self.size(16), 'bold'))
        l_e.grid(row=0, column=0)
        self.v_e = tk.DoubleVar(value=self.e)
        self.e_entry = tk.Entry(fr_ke, width=10, textvariable=self.v_e, font=('Arial', self.size(16), 'bold'))
        self.e_entry.grid(row=0, column=1)
        
        fr_input = tk.Frame(frame_left, background='white')
        fr_input.pack()
        l_R2 = tk.Label(fr_input, text='R2 (deg): ', background='white', font=('Arial', self.size(16), 'bold'))
        l_R2.grid(row=0, column=0)
        l_R1 = tk.Label(fr_input, text='R1 (deg): ', background='white', font=('Arial', self.size(16), 'bold'))
        l_R1.grid(row=0, column=1)
        l_Phi = tk.Label(fr_input, text='Phi (deg): ', background='white', font=('Arial', self.size(16), 'bold'))
        l_Phi.grid(row=0, column=2)
        self.v_r2, self.v_r1, self.v_phi = tk.DoubleVar(), tk.DoubleVar(), tk.DoubleVar()
        self.r2_entry = tk.Entry(fr_input, width=10, textvariable=self.v_r2, font=('Arial', self.size(16), 'bold'))
        self.r2_entry.grid(row=1, column=0)
        self.r1_entry = tk.Entry(fr_input, width=10, textvariable=self.v_r1, font=('Arial', self.size(16), 'bold'))
        self.r1_entry.grid(row=1, column=1)
        self.phi_entry = tk.Entry(fr_input, width=10, textvariable=self.v_phi, font=('Arial', self.size(16), 'bold'))
        self.phi_entry.grid(row=1, column=2)
        btn_add = tk.Button(fr_input, text='Add Point', command=self.add_point, font=('Arial', self.size(16), 'bold'))
        btn_add.grid(row=1, column=3)
        
        fr_tol = tk.Frame(frame_left, background='white')
        fr_tol.pack()
        l_tol = tk.Label(fr_tol, text='Dot Product Tolerance: ', background='white', font=('Arial', self.size(16), 'bold'))
        l_tol.grid(row=0, column=0)
        self.v_tol = tk.DoubleVar(value=self.tolerance)
        self.tol_entry = tk.Entry(fr_tol, width=7, textvariable=self.v_tol, font=('Arial', self.size(16), 'bold'))
        self.tol_entry.grid(row=0, column=1)
        btn_set_tol = tk.Button(fr_tol, text='Set Tolerance', command=self.set_tolerance, font=('Arial', self.size(16), 'bold'))
        btn_set_tol.grid(row=0, column=2)
        
        fr_fit = tk.Frame(frame_left, background='white')
        fr_fit.pack()
        btn_fit = tk.Button(fr_fit, text='Clear Points', command=self.clear_points, font=('Arial', self.size(16), 'bold'))
        btn_fit.grid(row=0, column=0)
        btn_fit = tk.Button(fr_fit, text='Fit Sample Offset', command=self._fit, font=('Arial', self.size(16), 'bold'))
        btn_fit.grid(row=0, column=1)
    

        fig0 = plt.Figure(figsize=(7*self.app_pars.scale, 7*self.app_pars.scale), layout='constrained')
        self.figout0 = FigureCanvasTkAgg(fig0, master=frame_middle)
        toolbar0 = NavigationToolbar2Tk(self.figout0, frame_middle)
        toolbar0.update()
        self.figout0.get_tk_widget().pack()
        self.ax0 = fig0.add_subplot(111)
        self.ax0.set_title('Before', fontsize=self.size(20))
        self.ax0.set_xlabel(r'$k_x$ (2$\pi$/Å)')
        self.ax0.set_ylabel(r'$k_y$ (2$\pi$/Å)')
        self.ax0.set_aspect('equal')
        fig1 = plt.Figure(figsize=(7*self.app_pars.scale, 7*self.app_pars.scale), layout='constrained')
        self.figout1 = FigureCanvasTkAgg(fig1, master=frame_right)
        toolbar1 = NavigationToolbar2Tk(self.figout1, frame_right)
        toolbar1.update()
        self.figout1.get_tk_widget().pack()
        self.ax1 = fig1.add_subplot(111)
        self.ax1.set_title('After', fontsize=self.size(20))
        self.ax1.set_xlabel(r'$k_x$ (2$\pi$/Å)')
        self.ax1.set_ylabel(r'$k_y$ (2$\pi$/Å)')
        self.ax1.set_aspect('equal')
        
        self.bind('<Return>', lambda event: self.add_point())
        for entry in [self.e_entry, self.r2_entry, self.r1_entry, self.phi_entry]:
            entry.bind('<FocusIn>', self.entry_select_all)
        
        self.e_entry.focus_set()

class g_cut_plot(tk.Toplevel):
    def __init__(self, master, data_cut, cx, cy, cdx, cdy, cdensity, ty, z, x, angle, phi_offset, r1_offset, phi1_offset, r11_offset, stop_event, pool, path, e_photon, slim, sym, xmin, xmax, ymin, ymax, cube, app_pars):
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

    def save_cube(self, event=None):
        try:
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

    def save_cut(self, event=None):
        try:
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
            print('Terminated')
            self.pool.join()
            print('Joined')
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

def set_center(parent, child, w_extend=None, h_extend=None):
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

class wait(tk.Toplevel):
    def __init__(self, master, app_pars: app_param):
        self.g = master
        self.app_pars = app_pars
        super().__init__(master, background='white')
        set_center(self.g, self)
        self.title('Info')
        tk.Label(self, bg='white', text='Please wait...', font=('Arial', self.size(16), "bold")).pack(side=tk.TOP, pady=20)
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

def det_chunk(cdensity, dtype=np.float32):
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
            except:
                pass
    except:
        pass


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
                
                self.fit_so_button = tk.Button(frame2, text="Fit Sample Offsets", command=fit_so_app, font=('Arial', self.size(14), "bold"), bg='white')    #fix
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
        tk.Button(self.sym_g, text='2-fold symmetry', command=self.symmetry_2, bg='white', font=('Arial', self.size(16), "bold")).pack(side=tk.TOP, pady=5)
        tk.Button(self.sym_g, text='3-fold symmetry', command=self.symmetry_3, bg='white', font=('Arial', self.size(16), "bold")).pack(side=tk.TOP, pady=5)
        tk.Button(self.sym_g, text='4-fold symmetry', command=self.symmetry_4, bg='white', font=('Arial', self.size(16), "bold")).pack(side=tk.TOP, pady=5)
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
            
    def symmetry_2(self):
        self.gen_sym(2)
    
    def symmetry_3(self):
        self.gen_sym(3)
        
    def symmetry_4(self):
        self.gen_sym(4)
        
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
            if int(self.cdensity/180*(xlim[1]-xlim[0])) ==0: # at least 1 pixel
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
        def filter(ind, ii, r2=None):
            r1 = np.linspace(min(self.y[ind]), max(self.y[ind]), len(self.y[ind]))[:, None]
            phi = np.linspace(min(self_x), max(self_x), len(self_x))[None, :]
            r1, phi = np.broadcast_arrays(r1, phi)
            for i in range(self.sym):
                r1_offset, phi_offset = self.cal_r1_phi_offset()
                if r2 is None:
                    r1, phi = self.rot(r1, phi, r1_offset, phi_offset, angle-360//self.sym*i)
                else:
                    r1, phi = self.rot(r1, phi, r1_offset, phi_offset, angle-(r2-self.z[0])-360//self.sym*i)
                if i == 0:
                    x = np.sqrt(2*self.m*self.e*self.ev[ii])/self.hbar*10**-10*np.sin(r1/180*np.pi) * np.cos(phi/180*np.pi)  # x: r1, y: phi, at r2=0
                    y = np.sqrt(2*self.m*self.e*self.ev[ii])/self.hbar*10**-10*np.sin(phi/180*np.pi)
                else:
                    x = np.append(x, np.sqrt(2*self.m*self.e*self.ev[ii])/self.hbar*10**-10*np.sin(r1/180*np.pi) * np.cos(phi/180*np.pi), axis=0)
                    y = np.append(y, np.sqrt(2*self.m*self.e*self.ev[ii])/self.hbar*10**-10*np.sin(phi/180*np.pi), axis=0)
            ti=[]
            for i in range(r1.shape[1]):
                if any(xlim[0]<x[:,i]) and any(zlim[0]<y[:,i]) and any(x[:,i]<xlim[1]) and any(y[:,i]<zlim[1]):
                    ti.append(i)
            if len(ti) != 0:
                if min(ti)>0:
                    ti.insert(0, min(ti)-1)
                if max(ti)<len(self.y[ind])-1:
                    ti.append(max(ti)+1)
            return ind[ti]
        self.cut_show = False
        if i ==100:
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
            if self.pool:
                self.pool.terminate()
                self.pool.join()
                self.pool = None
    
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
                for i, result in enumerate(tqdm.tqdm(self.pool.imap(cut_job_y, args), total=len(self.ev), desc="Processing", file=sys.stdout, colour='blue')):
                    if self.stop_event.is_set():
                        break
                    pass
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
                for i, result in enumerate(tqdm.tqdm(self.pool.imap(cut_job_x, args), total=len(self.ev), desc="Processing", file=sys.stdout, colour='blue')):
                    if self.stop_event.is_set():
                        break
                    pass
                if not self.stop_event.is_set():
                    print("\n\033[32mPress \033[31m'Enter' \033[32mto coninue...\033[0m")
                args = None
        except Exception as e:
            args = None
            print('t_cut_job_x')
            print(f"An error occurred: {e}")
    
    def t_cut_job_x_thread(self):
        angle = self.angle
        x = [self.cx-self.cdx/2, self.cx+self.cdx/2]
        z = [self.cy-self.cdy/2, self.cy+self.cdy/2]
        phi_offset = self.phi_offset
        r1_offset = self.r1_offset
        phi1_offset, r11_offset = self.phi1_offset, self.r11_offset
        self_x = self.ox[self.slim[0]:self.slim[1]+1]
        self_volume = self.ovolume[:, self.slim[0]:self.slim[1]+1, :]
        
        self.set_xy_lim()
        try:
            os.chdir(os.path.join(tempdir))
            if os.path.exists('cut_temp_save'):
                shutil.rmtree('cut_temp_save')
            os.mkdir('cut_temp_save')
            if os.path.exists('cube_temp_save'):
                shutil.rmtree('cube_temp_save')
            os.mkdir('cube_temp_save')

            max_threads = self.pool_size*3
            with ThreadPoolExecutor(max_workers=max_threads) as self.pool:
                args = [(i, angle, phi_offset, r1_offset, phi1_offset, r11_offset, self_x, self_volume, self.cdensity, self.xmax, self.xmin, self.ymax, self.ymin, z, x, self.z, self.y, self.ev, self.e_photon, self.sym) for i in range(len(self.ev))]
                
                # results = []
                for i, result in enumerate(tqdm.tqdm(self.pool.map(cut_job_x, args), total=len(self.ev), desc="Processing", file=sys.stdout, colour='blue')):
                    if self.stop_event.is_set():
                        break
                    # results.append(result)
                    
                if not self.stop_event.is_set():
                    print("\n\033[32mPress \033[31m'Enter' \033[32mto continue...\033[0m")
        except Exception as e:
            print('t_cut_job_x')
            print(f"An error occurred: {e}")
    
    def t_cut_job_y_thread(self):
        angle = self.angle
        x = [self.cx-self.cdx/2, self.cx+self.cdx/2]
        z = [self.cy-self.cdy/2, self.cy+self.cdy/2]
        phi_offset = self.phi_offset
        r1_offset = self.r1_offset
        phi1_offset, r11_offset = self.phi1_offset, self.r11_offset
        self_x = self.ox[self.slim[0]:self.slim[1]+1]
        self_volume = self.ovolume[:, self.slim[0]:self.slim[1]+1, :]
        
        self.set_xy_lim()
        try:
            os.chdir(os.path.join(tempdir))
            if os.path.exists('cut_temp_save'):
                shutil.rmtree('cut_temp_save')
            os.mkdir('cut_temp_save')
            if os.path.exists('cube_temp_save'):
                shutil.rmtree('cube_temp_save')
            os.mkdir('cube_temp_save')
            
            max_threads = self.pool_size*3
            with ThreadPoolExecutor(max_workers=max_threads) as self.pool:
                args = [(i, angle, phi_offset, r1_offset, phi1_offset, r11_offset, self_x, self_volume, self.cdensity, self.xmax, self.xmin, self.ymax, self.ymin, z, x, self.z, self.y, self.ev, self.e_photon, self.sym) for i in range(len(self.ev))]
                
                # results = []
                for i, result in enumerate(tqdm.tqdm(self.pool.map(cut_job_y, args), total=len(self.ev), desc="Processing", file=sys.stdout, colour='blue')):
                    if self.stop_event.is_set():
                        break
                    # results.append(result)
                    
                if not self.stop_event.is_set():
                    print("\n\033[32mPress \033[31m'Enter' \033[32mto continue...\033[0m")
        except Exception as e:
            print('t_cut_job_y')
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
        print('\033[32mIf you want to stop the process, wait for 20 seconds and try.\nBut sometimes it may not work.\033[0m')
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
        time.sleep(5)
        print("\n\033[32m-----Press \033[31m'Enter' \033[32mto terminate the pool-----\033[0m\n")
        t1.join()
        self.t.join()
        print('Proccess finished\nWait a moment...')
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
                self.g.update()
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
        def filter(ind, i, r2=None):    #test the filtering process in slice_data function
            r1 = np.linspace(min(self.y[ind]), max(self.y[ind]), len(self.y[ind]))[:, None]
            phi = np.linspace(min(self.x), max(self.x), len(self.x))[None, :]
            r1, phi = np.broadcast_arrays(r1, phi)
            if r2 is None:
                r1_offset, phi_offset = self.cal_r1_phi_offset()
                r1, phi = self.rot(r1, phi, r1_offset, phi_offset, self.angle)
            else:
                r1_offset, phi_offset = self.cal_r1_phi_offset(r2)
                r1, phi = self.rot(r1, phi, r1_offset, phi_offset, self.angle-(r2-self.z[0]))
            x = np.sqrt(2*self.m*self.e*self.ev[i])/self.hbar*10**-10*np.sin(r1/180*np.pi) * np.cos(phi/180*np.pi)  # x: r1, y: phi, at r2=0
            y = np.sqrt(2*self.m*self.e*self.ev[i])/self.hbar*10**-10*np.sin(phi/180*np.pi)
            ti=[]
            for i in range(r1.shape[1]):
                if any(-0.1<x[:,i]) and any(-0.2<y[:,i]) and any(x[:,i]<0.1) and any(y[:,i]<0.2):
                    ti.append(i)
            if len(ti) != 0:
                if min(ti)>0:
                    ti.insert(0, min(ti)-1)
                if max(ti)<len(self.y[ind])-1:
                    ti.append(max(ti)+1)
            return ind[ti]
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
            if int(self.density/180*(xlim[1]-xlim[0])) ==0: # at least 1 pixel
                d = 1/self.density*180/2
                c = (xlim[0]+xlim[1])/2
                xlim = [float(c-d), float(c+d)]
                print(f'Warning: R1-axis density is too low (R2=%.2f)'%r2)
                tk.messagebox.showwarning("Warning",f'Warning: R1-axis density is too low (R2=%.2f)'%r2)
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

class CEC(loadfiles, CEC_Object):
    def __init__(self, g: tk.Tk, files: list[str] | tuple[str, ...] | str, mode: Literal['normal', 'load']= 'normal', cmap: str='viridis', app_pars: app_param=None):
        super().__init__(files, mode='eager')
        if self.sort == 'r1r2':
            r2 = []
            for i in self.r2:
                for j in i:
                    r2.append(j)
            self.r2 = r2
            r2 = None
        
        self.cmap = cmap
        self.mode = mode
        self.app_pars = app_pars
        
        self.g = g
        self.tlg = tk.Toplevel(g, bg='white')
        self.tlg.title('Constant Energy Cut')
        self.frame1 = tk.Frame(self.tlg, bg='white')
        self.frame1.grid(row=0, column=1)
        self.__check_re()
    
    def on_closing(self):
        try:
            plt.close('all')
        except:
            pass
        self.tlg.destroy()
        clear(self.view)
        clear(self)
        gc.collect()
        return
    
    def load(self, angle, cx, cy, cdx, cdy, phi_offset, r1_offset, phi1_offset, r11_offset, slim, sym, name, path):
        vs = self.view
        set_entry_value(vs.entry_min, str(slim[0]))
        set_entry_value(vs.entry_max, str(slim[1]))
        set_entry_value(vs.entry_phi_offset, str(phi_offset))
        set_entry_value(vs.entry_r1_offset, str(r1_offset))
        set_entry_value(vs.entry_phi1_offset, str(phi1_offset))
        set_entry_value(vs.entry_r11_offset, str(r11_offset))
        vs.change_mode(mode=self.mode) # change from 'real' to 'reciprocal' in 'load' mode
        set_entry_value(vs.cut_xy_x_entry, str(cx))
        set_entry_value(vs.cut_xy_y_entry, str(cy))
        set_entry_value(vs.cut_xy_dx_entry, str(cdx))
        set_entry_value(vs.cut_xy_dy_entry, str(cdy))
        vs.angle = angle
        vs.text_a.set(f'%.3f'%angle)
        vs.angle_slider.set_val(angle)
        vs.set_slim()
        vs.gen_sym(sym)
        vs.cut_xy()
        vs.mode = 'normal'
        self.info_name = name
        self.info_path = path
        self.info_angle = angle
        self.info_sym = sym
        self.info_r1_offset = r1_offset
        self.info_phi_offset = phi_offset
        self.info_r11_offset = r11_offset
        self.info_phi1_offset = phi1_offset
        self.info_slim = slim
        self.info_cut_xy_x = cx
        self.info_cut_xy_y = cy
        self.info_cut_xy_dx = cdx
        self.info_cut_xy_dy = cdy
        
    def info(self):
        print('\nData Cube Info:')
        print('\033[32mPath:\033[36m',f'{self.path[0]}')
        for i, v in enumerate(self.path):
            if i > 0:
                print(f'\033[36m      {v}')
        text = self.l1.get("1.0", tk.END).split('\n')[0:-1]
        if ':' in text[0]:
            for i in text:
                print('\033[0m%s=%7s:%s'%(i.split(":")[0].split('=')[0],i.split(":")[0].split('=')[1],i.split(":")[1]))
        else:
            s=''
            for i, v in enumerate(text):
                if i != len(text)-1:
                    s += v.split('=')[1]+', '
                else:
                    s += v.split('=')[1]
            print('\033[0m%s=%s'%(v.split('=')[0], s))
        print('Slices: %d'%len(self.r1))
        print('Data Cube Size: %f MB'%(np.float64(self.size)/1024/1024))
        if self.mode == 'load':
            print(f'\033[32mFile Name: \033[36m{self.info_name}\033[0m')
            print(f'\033[32mFile Path: \033[36m{self.info_path}\033[0m')
            print(f'\033[32mAngle: \033[36m{self.info_angle} degree\033[0m')
            if self.info_sym != 1:
                print(f'\033[32mSymmetry: \033[36m{self.info_sym}-fold\033[0m')
            else:
                print(f'\033[32mSymmetry: \033[36mN/A\033[0m')
            print(f'\033[32mR1 Manipulator Offset: \033[36m{self.info_r1_offset} degree\033[0m')
            print(f'\033[32mPhi Manipulator Offset: \033[36m{self.info_phi_offset} degree\033[0m')
            print(f'\033[32mR11 Manipulator Offset: \033[36m{self.info_r11_offset} degree\033[0m')
            print(f'\033[32mPhi1 Manipulator Offset: \033[36m{self.info_phi1_offset} degree\033[0m')
            print(f'\033[32mSlit Limit: \033[36m{self.info_slim[0]} ~ {self.info_slim[1]}\033[0m')
            print(f'\033[32mkx: \033[36m{self.info_cut_xy_x}\033[0m')
            print(f'\033[32mky: \033[36m{self.info_cut_xy_y}\033[0m')
            print(f'\033[32mkx bin: \033[36m{self.info_cut_xy_dx}\033[0m')
            print(f'\033[32mky bin: \033[36m{self.info_cut_xy_dy}\033[0m')
        if self.app_pars:
            windll.user32.ShowWindow(self.app_pars.hwnd, 9)
            windll.user32.SetForegroundWindow(self.app_pars.hwnd)
        return
    
    def __set_data(self, odata=[], density=800, *args):
        if len(odata) > 0:
            # self.tlg.geometry(f'800x600+0+{sc_y}')
            set_center(self.g, self.tlg, 0, 0)
            self.size = 0
            for i in self.path:
                self.size += os.path.getsize(i)
            if self.sort == 'r1':
                odataframe = np.stack([i.transpose() for i in odata], axis=0, dtype=np.float32)
                # odataframe = np.stack([zarr.open(self.zpath, mode='r')[i, :, :].transpose() for i in range(len(self.name))], axis=0, dtype=np.float32)
                print('Input Data Shape: '+str(odataframe.shape))   # shape: (r1, phi, ev)
                
                r1 = self.r1
                ev, phi = odata[0].indexes.values()
                e_photon = float(odata[0].attrs['ExcitationEnergy'].removesuffix(' eV'))
                self.view = VolumeSlicer(parent=self.frame1, path=self.path, volume=odataframe, cmap=self.cmap, x=phi, y=r1, ev=ev, e_photon=e_photon, density=density, g=self.tlg, app_pars=self.app_pars)
            elif self.sort == 'r1r2':
                r1 = self.r1
                r2 = self.r2
                odataframe = np.stack([i.transpose() for i in odata], axis=0, dtype=np.float32)
                # odataframe = np.stack([zarr.open(self.zpath, mode='r')[i, :, :].transpose() for i in range(len(self.name))], axis=0, dtype=np.float32)
                print('Input Data Shape: '+str(odataframe.shape))
                
                ev, phi = odata[0].indexes.values()
                e_photon = float(odata[0].attrs['ExcitationEnergy'].removesuffix(' eV'))
                self.view = VolumeSlicer(parent=self.frame1, path=self.path, volume=odataframe, cmap=self.cmap, x=phi, y=r1, z=r2, ev=ev, e_photon=e_photon, density=density, g=self.tlg, app_pars=self.app_pars)
            self.tlg.bind('<Return>', self.view.set_slim)
            if self.mode == 'normal':
                self.view.set_slim()
            self.view.show()
            self.tlg.update()
            w = self.tlg.winfo_reqwidth()
            h = self.tlg.winfo_reqheight()
            ScaleFactor = windll.shcore.GetScaleFactorForDevice(0)
            t_sc_w, t_sc_h = windll.user32.GetSystemMetrics(0), windll.user32.GetSystemMetrics(1)   # Screen width and height
            t_sc_h-=int(40*ScaleFactor/100)
            if self.app_pars.bar_pos == 'top':    #taskbar on top
                sc_y = int(40*ScaleFactor/100)
            else:
                sc_y = 0
            tx = int(t_sc_w*windll.shcore.GetScaleFactorForDevice(0)/100) if self.tlg.winfo_x()+self.tlg.winfo_width()/2 > t_sc_w else 0
            self.tlg.geometry(f'{w}x{h}+{tx}+{sc_y}')
            self.tlg.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.tlg.update()
            odataframe, odata, r1, r2, ev, phi, e_photon = None, None, None, None, None, None, None

    def __rlist(self):
        self.frame0 = tk.Frame(self.tlg, bg='white')
        self.frame0.grid(row=0, column=0)
        tk.Button(self.frame0, text='Info', width=6, height=2, font=('Arial', self.size(18), 'bold'), bg='white', bd=5, command=self.info).pack(side=tk.TOP, padx=2)
        self.l1 = tk.Text(self.frame0, wrap='none', width=30, height=9, font=('Arial', self.size(12), 'bold'), bg='white', bd=5)
        self.l1.pack(side=tk.TOP)
        
        if self.sort == 'r1r2':
            r2 = sorted(set(self.r2))
            t = 0
            s=''
            ls=0
            tt = False
            for i,v in enumerate(self.r2):
                if i<len(self.r2)-1:
                    if self.r2[i+1] == r2[t]:   # current r2 is same as next r2
                        if not tt:
                            s+=self.r2s[2]+'='+str(r2[t])+': '
                            tt = True
                        s+=str(self.r1[i])+', '
                        if i+1 == len(self.r2)-1:   # next r2 is the last r2
                            s+=str(self.r1[i+1])
                            if len(s)>ls:
                                ls=len(s)
                            self.l1.insert(tk.END, s)
                            s=''
                            tt = False
                    else:   # current r2 is different from next r2
                        if i+1 == len(self.r2)-1:   # next r2 is the last r2
                            s.removesuffix(', ')
                            s+=self.r2s[2]+'='+str(r2[t])+': '+str(self.r1[i])+'\n'
                            s+=self.r2s[2]+'='+str(r2[t+1])+': '+str(self.r1[i+1])
                        else:   # not last r2
                            if not tt:
                                s+=self.r2s[2]+'='+str(r2[t])+': '
                            s+=str(self.r1[i])+'\n'
                        if len(s)>ls:
                            ls=len(s)
                        self.l1.insert(tk.END, s)
                        s=''
                        tt = False
                        t+=1
        else:
            ls=0
            for i,v in enumerate(self.r1):
                if len(self.r1s[2]+'='+str(v))>ls:
                    ls=len(self.r1s[2]+'='+str(v))
                if i != len(self.r1)-1:
                    self.l1.insert(tk.END, self.r1s[2]+'='+str(v)+'\n')
                else:
                    self.l1.insert(tk.END, self.r1s[2]+'='+str(v))
            self.l1.config(width=ls)
    
    def __check(self, *args, f=False):
        name = self.lb.name
        t = self.preserve
        for i, v in enumerate(self.name):
            if v in name:
                t[i] = True
        self.preserve = t
        if self.sort == 'r1r2':
            r1 = []
            r2 = []
            for i, v in enumerate(self.r2):
                if self.preserve[i]:
                    r1.append(self.r1[i])
                    r2.append(v)
            self.r1 = r1
            self.r2 = r2
        elif self.sort == 'r1':
            r1 = []
            for i, v in enumerate(self.r1):
                if self.preserve[i]:
                    r1.append(v)
            self.r1 = r1
        
        if f==False:
            path = []
            name = []
            data = []
            for i, v in enumerate(self.preserve):
                if v:
                    path.append(self.path[i])
                    name.append(self.name[i])
                    data.append(self.data[i])
                    self.data[i] = None
            self.path = path
            self.name = name
            self.data = data
        if f==False:
            self.gg.destroy()
        self.__prework()
        self.tlg.focus_set()
        if isinstance(lfs, FileSequence):   # Actually always True
            for i in lfs.__dir__():
                if i not in ['path', 'r1']:
                    try:
                        setattr(self, i, None)
                    except:
                        pass
                    
    def __check_file(self):
        self.gg = RestrictedToplevel(self.g, bg='white')
        self.gg.protocol("WM_DELETE_WINDOW", self.__check)
        self.gg.title('File Check')
        text = 'Same File Name Exists\nSelect the file you want to preserve'
        tk.Label(self.gg, text=text, width=len(text), height=2, font=('Arial', self.size(14), "bold"), bg='white', bd=5).grid(row=0, column=0)
        frame1 = tk.Frame(self.gg, bg='white')
        frame1.grid(row=1, column=0)
        name = [i.split('#id#')[0] for i in self.name]
        tname = [i for i in name]
        self.preserve = []
        for i in self.name:
            if '#id#' in i:
                self.preserve.append(False)
            else:
                self.preserve.append(True)
        t = 0
        self.lb = add_lb(frame1, self.sort, self.app_pars.scale)
        n = 0
        r1 = []
        r2 = []
        ss = []
        while t < len(name):
            if self.sort == 'r1r2':
                s = ''
                fj = False
                tt = False
                ti=1
                tj=t
                for jj in range(t+1, len(name)):
                    if name[t] == name[jj]:
                        ti+=1
                        if not tt:
                            tname[t] = tname[t]+'#id#'+str(t)
                            s+=tname[t]+'\n'
                            tt = True
                        tname[jj] = tname[jj]+'#id#'+str(jj)
                        s+=tname[jj]+'\n'
                        fj = True
                        tj = jj
                if fj:
                    t = tj
                t+=1
                s.removesuffix('\n')
                if s != '':
                    n+=1
                    r2.append(self.r2[t-1])
                    ss.append(s)
                    # self.lb.add(s, self.r2[t-1])
            elif self.sort == 'r1':
                s = ''
                fj = False
                tt = False
                ti=1
                tj=t
                for jj in range(t+1, len(name)):
                    if name[t] == name[jj]:
                        ti+=1
                        if not tt:
                            tname[t] = tname[t]+'#id#'+str(t)
                            s+=tname[t]+'\n'
                            tt = True
                        tname[jj] = tname[jj]+'#id#'+str(jj)
                        s+=tname[jj]+'\n'
                        fj = True
                        tj = jj
                if fj:
                    t = tj
                t+=1
                s.removesuffix('\n')
                if s != '':
                    n+=1
                    r1.append(self.r1[t-1])
                    ss.append(s)
                    # self.lb.add(s, self.r1[t-1])
        self.lb.name = ['name' for i in range(n)]
        for i in range(n):
            if self.sort == 'r1r2':
                self.lb.add(ss[i], r2[i], self.r1s[2], self.r2s[2])
            elif self.sort == 'r1':
                self.lb.add(ss[i], r1[i], self.r1s[2], self.r2s[2])
                
        tk.Button(self.gg, text='OK', command=self.__check, width=15, height=1, font=('Arial', self.size(14), "bold"), bg='white', bd=5).grid(row=2, column=0)
        self.f1 = False
        self.gg.bind('<Return>', self.__check)
        set_center(self.g, self.gg, 0, 0)
        self.gg.focus_set()
        self.gg.limit_bind()
        return
        
    def __select_file(self):
        if self.f2:
            self.gg = RestrictedToplevel(self.g, bg='white')
            self.gg.protocol("WM_DELETE_WINDOW", self.__check)
            self.gg.title('File Check')
            if self.sort == 'r1r2':
                text = f'Same {self.r1s[2]} and {self.r2s[2]} Exists\nSelect the file you want to preserve'
            elif self.sort == 'r1':
                text = f'Same {self.r1s[2]} Exists\nSelect the file you want to preserve'
            tk.Label(self.gg, text=text, width=len(text), height=2, font=('Arial', self.size(14), "bold"), bg='white', bd=5).grid(row=0, column=0)
            frame1 = tk.Frame(self.gg, bg='white')
            frame1.grid(row=1, column=0)
            self.lb = add_lb(frame1, self.sort, self.app_pars.scale)
            self.preserve = [True for i in range(len(self.name))]        
            ti=[]
            ss=[]
            tt=[]
            t=0
            if self.sort == 'r1r2':
                for i in range(len(self.r2)):
                    for j in range(len(self.r2)):
                        if self.r2[i] == self.r2[j] and self.r1[i] == self.r1[j] and i != j:
                            ti.append(i)
                            self.preserve[i] = False
                ti = sorted(set(ti))
                while t < len(ti):
                    s=''
                    for i in range(len(np.argwhere(self.r1 == self.r1[ti[t]]))):
                        s+=self.name[ti[t]]+'\n'
                        t+=1
                    if s != '':
                        tt.append(t-1)
                        ss.append(s.removesuffix('\n'))
                self.lb.name = ['name' for i in range(len(ss))]
                for i, v in enumerate(ss):
                    self.lb.add(v, self.r2[tt[i]], self.r1s[2], self.r2s[2])
            elif self.sort == 'r1':
                for i in range(len(self.r1)):
                    for j in range(len(self.r1)):
                        if self.r1[i] == self.r1[j] and i != j:
                            ti.append(i)
                            self.preserve[i] = False
                            
                ti = sorted(set(ti))
                while t < len(ti):
                    s=''
                    for i in range(len(np.argwhere(self.r1 == self.r1[ti[t]]))):
                        s+=self.name[ti[t]]+'\n'
                        t+=1
                    if s != '':
                        tt.append(t-1)
                        ss.append(s.removesuffix('\n'))
                self.lb.name = ['name' for i in range(len(ss))]
                for i, v in enumerate(ss):
                    self.lb.add(v, self.r1[tt[i]], self.r1s[2], self.r2s[2])
                    
            tk.Button(self.gg, text='OK', command=self.__check, width=15, height=1, font=('Arial', self.size(14), "bold"), bg='white', bd=5).grid(row=2, column=0)
            self.f2 = False
            self.gg.bind('<Return>', self.__check)
            set_center(self.g, self.gg ,0 ,0)
            self.gg.focus_set()
            self.gg.limit_bind()
        else:
            self.__check(f=True)
        return
        
    def __prework(self):
        if self.f1:
            self.__check_file()
        elif self.f2:
            self.f2 = False
            if self.sort == 'r1r2':
                for i in range(len(self.r2)):
                    for j in range(len(self.r2)):
                        if self.r2[i] == self.r2[j] and self.r1[i] == self.r1[j] and i != j:
                            self.f2 = True
            elif self.sort == 'r1':
                for i in range(len(self.r1)):
                    for j in range(len(self.r1)):
                        if self.r1[i] == self.r1[j] and i != j:
                            self.f2 = True
            self.__select_file()
        else:
            self.__rlist()
            # if type(self.data[0]) != xr.DataArray:
            #     self.data = [self.get(i) for i in self.data]
            for i in self.data:
                i.data = i.data.astype(np.float32)
                # i.data = np.empty(i.data.shape, dtype=np.uint8)
            self.__set_data(odata=self.data)
        return
    
    def __gen_f1_f2(self):
        self.f1 = False
        self.f2 = False
        for i in self.name:
            if '#id#' in i:
                self.f1 = True
        if self.sort == 'r1r2':
            for i in range(len(self.r2)):
                for j in range(len(self.r2)):
                    if self.r2[i] == self.r2[j] and self.r1[i] == self.r1[j] and i != j:
                        self.f2 = True
        elif self.sort == 'r1':
            for i in range(len(self.r1)):
                for j in range(len(self.r1)):
                    if self.r1[i] == self.r1[j] and i != j:
                        self.f2 = True
        return
    
    def __check_re(self):
        self.__gen_f1_f2()
        self.__prework()
        return
    
    def size(self, s:int) -> int:
        return int(s * self.app_pars.scale)

class add_lb():
    def __init__(self, fr, sort, scale):
        self.fr = fr
        self.lb = []
        self.s = []
        self.l = []
        self.r = []
        self.name = []
        self.sort = sort
        self.scale = scale
        return
    
    def size(self, s:int) -> int:
        return int(s * self.scale)
    
    def add(self, s, r, r1s, r2s):    
        self.s.append(s.removesuffix('\n'))
        if self.sort == 'r1r2':
            ltex = r2s+':'+str(r)+' '
        elif self.sort == 'r1':
            ltex = r1s+':'+str(r)+' '
        else:
            ltex = ''
        l = tk.Label(self.fr, text=ltex, width=len(ltex), height=1, font=('Arial', self.size(14), "bold"), bg='white', bd=5)
        l.pack()
        self.l.append(l)
        self.r.append(r)
        ls=0
        for i in self.s[-1].split('\n'):
            if len(i)>ls:
                ls=len(i)+1
        listbox = tk.Listbox(self.fr, selectmode='single', font=('Arial', self.size(14), "bold"), bg='white', bd=5, width=ls, height=len(self.s[-1].split('\n')))
        listbox.pack()
        self.lb.append(listbox)
        
        self.l[-1].config(text=ltex+self.s[-1].split('\n')[0], width=len(ltex+self.s[-1].split('\n')[0]))
        
        for i in self.s[-1].split('\n'):
            listbox.insert(tk.END, i)
        
        # Set focus to the Listbox
        listbox.focus_set()
        
        # Bind events
        listbox.bind('<<ListboxSelect>>', lambda event, lb=listbox, l=l, ltex=ltex: self.__on_select(event, lb, l, ltex))
        listbox.bind('<Up>', lambda event, lb=listbox, l=l: self.__on_up(event, lb, l))
        listbox.bind('<Down>', lambda event, lb=listbox, l=l: self.__on_down(event, lb, l))
        
        # Pre-select the first item
        if listbox.size() > 0:
            listbox.select_set(0)
            listbox.event_generate('<<ListboxSelect>>')
        return
        
    def __on_up(self, event, lb, l):
        selected_index = lb.curselection()
        if selected_index and selected_index[0] > 0:
            lb.select_clear(selected_index[0])
            lb.select_set(selected_index[0] - 1)
            lb.event_generate('<<ListboxSelect>>')
        return
    
    def __on_down(self, event, lb, l):
        selected_index = lb.curselection()
        if selected_index and selected_index[0] < lb.size() - 1:
            lb.select_clear(selected_index[0])
            lb.select_set(selected_index[0] + 1)
            lb.event_generate('<<ListboxSelect>>')
        return
    
    def __on_select(self, event, lb, l, ltex):
        selected_index = lb.curselection()
        if selected_index:
            selected_item = lb.get(selected_index)
            l.config(text=ltex+selected_item, width=len(ltex+selected_item))
            for i,v in enumerate(self.r):
                if str(v)+' ' in ltex:
                    self.name[i] = selected_item.replace('\n', '')
        return