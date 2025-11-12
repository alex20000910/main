from MDC_cut_utility import RestrictedToplevel, FileSequence, CEC_Object, app_param, clear, set_entry_value, set_center
from .loader import loadfiles, get_cec_params
from .VolumeSlicer import VolumeSlicer
import os
import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
from ctypes import windll
import gc
from typing import Literal

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
        for i in loadfiles.__dict__:
            if i not in ['path', 'r1']:
                try:
                    setattr(self, i, None)
                except: # property does not have setter
                    try:
                        self.i = None
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


def call_cec(g, lfs: FileSequence):
    app_pars = lfs.app_pars
    path_to_file, name, lf_path, tlfpath, cmap = lfs.cec_pars.path_to_file, lfs.cec_pars.name, lfs.cec_pars.lf_path, lfs.cec_pars.tlfpath, lfs.cec_pars.cmap
    lfs.cec = None
    try:
        args = get_cec_params(path_to_file)
        try:
            lfs.cec = CEC(g, lf_path, mode='load', cmap=cmap, app_pars=app_pars)
            lfs.cec.load(*args, name, path_to_file)
        except:
            lfs.cec = CEC(g, tlfpath, mode='load', cmap=cmap, app_pars=app_pars)
            lfs.cec.load(*args, name, path_to_file)
    except Exception as ecp:
        if app_pars:
            windll.user32.ShowWindow(app_pars.hwnd, 9)
            windll.user32.SetForegroundWindow(app_pars.hwnd)
        print(f"An error occurred: {ecp}")
        print('\033[31mPath not found:\033[34m')
        print(lf_path)
        print('\033[31mPlace all the raw data files listed above in the same folder as the HDF5/NPZ file\nif you want to view the slicing geometry or just ignore this message if you do not need the slicing geometry.\033[0m')
        message = f"Path not found:\n{lf_path}\nPlace all the raw data files listed above in the same folder as the HDF5/NPZ file if you want to view the slicing geometry\nor just ignore this message if you do not need the slicing geometry."
        messagebox.showwarning("Warning", message)
    return lfs