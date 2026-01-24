from MDC_cut_utility import *
from .loader import loadfiles
import os, io
import tkinter as tk
from tkinter import filedialog as fd
import sys
from ctypes import windll
import gc
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from PIL import Image
import win32clipboard
from scipy import special
from scipy.optimize import curve_fit

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
                    self.lfs = loadfiles(path, init=True, spectrogram=True, name='internal')
                elif kwargs['name'] == 'external':
                    self.lfs = loadfiles(path, spectrogram=True)
            else:
                self.lfs = loadfiles(path, spectrogram=True)
            self.data = self.lfs.get(0)
            if self.lfs.f_npz[0]:self.npzf = True
        else:
            self.data = data
        if 'app_pars' in kwargs:
            self.app_pars = kwargs['app_pars']
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
        
    def ups(self):
        self.root = tk.Toplevel(self.g,bg='white')
        self.root.title('UPS spectrum')
        
        # OptionMenu 設定
        fit_options = ["Raw Data", "Smooth Data", "Fermi-Dirac Fitting", "Linear Fits", "ERFC Fit", "First Derivative", "Second Derivative", "Smooth Data with First Derivative", "Segmented Tangents"]
        self.fit_options = fit_options
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
            self.cf_up()
        elif event.delta<0:
            self.cf_down()
    
    def cf_up(self, *args):
        now = self.namevar.get()
        for i, j in enumerate(self.lfs.name):
            if now == j:
                if i == 0:
                    self.namevar.set(self.lfs.name[-1])
                else:
                    self.namevar.set(self.lfs.name[i-1])
        self.__change_file()

    def cf_down(self, *args):
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
            - ups
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
            self.b_ups = tk.Button(self.fr_info, text='Fermi Level Fitting', command=self.ups, width=30, height=1, font=('Arial', self.size(12), "bold"), bg='white', bd=5)
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
            self.tpg.bind('<Up>', self.cf_up)
            self.tpg.bind('<Down>', self.cf_down)
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


class lfs_exp_casa(loadfiles):
    def __init__(self, lfs: FileSequence):
        super().__init__(lfs.path, init=True, name='internal', spectrogram=True)
    
    def export_casa(self, path=None):
        if path is None:
            path = fd.asksaveasfilename(title="Save as", filetypes=(("VMS files", "*.vms"),), initialdir=self.path[0], initialfile=self.name[0], defaultextension='.vms')
        if path.split('.')[-1] != 'vms':
            path += '.vms'
        if path != '.vms':
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
    Number of Regions={len(self.name)}
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
    {len(self.name)}
    '''     
            body = ''
            for i in self.data:
                s=spectrogram(self.get(i), name='internal')
                body+=s.gen_casa_body()
            f.write(head+body+'end of experiment\n')
            f.close()
