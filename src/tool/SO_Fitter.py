from MDC_cut_utility import clear
from .util import app_param
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import os
if os.name == 'nt':
    from ctypes import windll
import gc

class SO_Fitter(tk.Toplevel):
    def __init__(self, master, app_pars: app_param=None):
        super().__init__(master, background='white')
        self.title('Sample Offset Fitter')
        self.ev=1.602176634e-19  # eV=1.602176634e-19 J
        h=6.62607015e-34  # J·s
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
        m=9.10938356e-31  # kg
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
            if os.name != 'nt':
                return
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
        m=9.10938356e-31  # kg
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
