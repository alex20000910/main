from MDC_cut_utility import RestrictedToplevel, set_center, clear, on_configure, cal_ver
from .util import IconManager
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import h5py, json
import os, io
import tkinter as tk
from tkinter import filedialog as fd
from threading import Thread
from abc import ABC, abstractmethod
from tkinter import colorchooser, messagebox
from ctypes import windll
from typing import override
from base64 import b64decode
from PIL import Image, ImageTk

class AboutWindow:
    def __init__(self, master: tk.Misc | None = None, scale: float = 1.0, version: str = "x.x.x", release_date: str = "YYYY-MM-DD"):
        self.g = master
        self.scale = scale
        self.version = version
        self.release_date = release_date
        icon = IconManager().icon
        self.icon = ImageTk.PhotoImage(Image.open(io.BytesIO(b64decode(icon))).resize([150, 150]))
        self.show()
    
    def size(self, size: int) -> int:
        return(int(self.scale*size))
    
    def select_all(self, event):
        event.widget.tag_add(tk.SEL, "1.0", tk.END)
        event.widget.mark_set(tk.INSERT, "1.0")
        event.widget.see(tk.INSERT)
        return 'break'
    
    def select_none(self, event):
        event.widget.tag_remove(tk.SEL, "1.0", tk.END)
        return 'break'
    
    def show(self):
        self.about_win = tk.Toplevel(self.g, bg='white')
        self.about_win.overrideredirect(True)
        self.about_win.title('About MDC_cut')
        
        bg = ["#d0d0d0", "#b0b0b0", "#909090", "#707070"]
        gap=3
        bd1_fr = tk.Frame(self.about_win, bg=bg[0], padx=gap, pady=gap)
        bd1_fr.pack()
        bd2_fr = tk.Frame(bd1_fr, bg=bg[1], padx=gap, pady=gap)
        bd2_fr.pack()
        bd3_fr = tk.Frame(bd2_fr, bg=bg[2], padx=gap, pady=gap)
        bd3_fr.pack()
        bd_fr = tk.Frame(bd3_fr, bg=bg[3], padx=gap, pady=gap)
        bd_fr.pack()
        fr = tk.Frame(bd_fr, bg='white', padx=gap/2, pady=gap/2)
        fr.pack()
        
        fr_title = tk.Frame(fr, bg='white')
        fr_title.pack()
        l_icon = tk.Label(fr_title, bg='white', image=self.icon)
        l_icon.pack(side=tk.LEFT, padx=2, pady=10)
        l1 = tk.Label(fr_title, text='MDC_cut', font=('Arial', self.size(30), "bold"), bg='white')
        l1.pack(side=tk.LEFT, pady=10)
        
        l2 = tk.Label(fr, text='Version: '+self.version, font=('Arial', self.size(16)), bg='white')
        l2.pack(pady=5)
        l3 = tk.Label(fr, text='Release Date: '+self.release_date, font=('Arial', self.size(16)), bg='white')
        l3.pack(pady=5)
        l4 = tk.Label(fr, text='Developed by Chih-Keng Hung', font=('Arial', self.size(16)), bg='white')
        l4.pack(pady=5)
        
        fr1 = tk.Frame(fr, bg='white')
        fr1.pack(pady=5)
        l_e = tk.Label(fr, text='Email: ', font=('Arial', self.size(16)), bg='white')
        l_e.pack(side=tk.LEFT, in_=fr1)
        str_email = 'alex1010512@gmail.com'
        t_email = tk.Text(fr1, width=20, height=1, font=('Arial', self.size(16)), bg='white', bd=0, wrap='none')
        t_email.tag_configure("blue", background="white", foreground="blue", selectbackground="blue", selectforeground="white")
        t_email.insert(tk.END, str_email, "blue")
        t_email.config(state=tk.DISABLED)
        t_email.pack(side=tk.LEFT)
        t_email.bind('<FocusIn>', self.select_all)
        t_email.bind('<FocusOut>', self.select_none)
        
        fr2 = tk.Frame(fr, bg='white')
        fr2.pack(pady=5)
        l_github = tk.Label(fr, text='GitHub: ', font=('Arial', self.size(16)), bg='white')
        l_github.pack(side=tk.LEFT, in_=fr2)
        str_github = 'https://github.com/alex20000910/main'
        t_github = tk.Text(fr2, width=31, height=1, font=('Arial', self.size(16)), bg='white', bd=0, wrap='none')
        t_github.tag_configure("blue", background="white", foreground="blue", selectbackground="blue", selectforeground="white")
        t_github.insert(tk.END, str_github, "blue")
        t_github.config(state=tk.DISABLED)
        t_github.pack(side=tk.LEFT)
        t_github.bind('<FocusIn>', self.select_all)
        t_github.bind('<FocusOut>', self.select_none)
        
        text = tk.Text(fr, width=60, height=10, wrap='word', font=('Arial', self.size(14)), bg='white')
        text.pack(padx=10, pady=5)
        license_text = """MIT License

Copyright (c) 2024-2026 Chih-Keng Hung

Permission is hereby granted, free of charge, to any person obtaining a copy \
of this software and associated documentation files (the "Software"), to deal \
in the Software without restriction, including without limitation the rights \
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell \
copies of the Software, and to permit persons to whom the Software is \
furnished to do so, subject to the following conditions:

1. The above copyright notice and this permission notice shall be included in \
all copies or substantial portions of the Software.

2. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR \
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, \
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE \
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER \
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, \
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE \
SOFTWARE."""
        text.insert(tk.END, license_text)
        text.config(state=tk.DISABLED)
        b1 = tk.Button(fr, text='OK', font=('Arial', self.size(16)), width=10, command=self.destroy, bd=2)
        b1.pack(pady=10)
        
        self.about_win.bind('<Return>', self.destroy)
        set_center(self.g, self.about_win, 0, 0)
        self.bind_funcid = self.g.bind('<Configure>', lambda event: set_center(self.g, self.about_win, 0, 0))
        self.about_win.resizable(False, False)
        self.about_win.grab_set()
        self.about_win.focus_force()
    
    def destroy(self, *e):
        self.g.unbind('<Configure>', self.bind_funcid)
        self.g.bind('<Configure>', lambda event: on_configure(self.g, event))
        self.about_win.grab_release()
        self.about_win.destroy()
        clear(self.about_win)

class EmodeWindow(RestrictedToplevel, ABC):
    def __init__(self, parent: tk.Misc | None = None, vfe: float=21.2, scale: float=1.0):
        super().__init__(parent, bg='white')
        self.vfe = vfe
        self.scale = scale
        v_fe = tk.StringVar(value=str(vfe))
        self.title('Fermi Level')
        fr=tk.Frame(self, bg='white')
        fr.grid(row=0,column=0)
        l_in = tk.Label(fr, text='Fermi Level (eV) : ', font=('Arial', self.size(16), "bold"), bg='white')
        l_in.grid(row=0,column=0)
        self.fe_in = tk.Entry(fr, font=("Arial", self.size(16), "bold"), width=10, textvariable=v_fe, bd=5)
        self.fe_in.grid(row=0,column=1)
        fr1 = tk.Frame(self, bg='white')
        fr1.grid(row=1,column=0)
        b1=tk.Button(fr1,text='Confirm',command=self.save_fe, width=15, height=1, font=('Arial', self.size(14), "bold"), bg='white', bd=5)
        b1.grid(row=1,column=0)
        self.bind('<Return>', self.save_fe)
        set_center(parent, self, 0, 0)
        self.focus_set()
        self.fe_in.focus_set()
        self.fe_in.icursor(tk.END)
        self.update()
        self.limit_bind()
    
    @abstractmethod
    def save_fe(self):
        # plot the data in main thread
        pass
        
    def size(self, size: int) -> int:
        return(int(self.scale*size))

class ColormapEditorWindow(tk.Toplevel, ABC):
    def __init__(self, master: tk.Misc | None, scale: float, optionList3: list[str], value3: tk.StringVar, setcmap: tk.OptionMenu, cmlf: tk.Frame, cdir: str):
        super().__init__(master)
        set_center(master, self, 0, 0)
        self.master = master
        self.scale = scale
        self.optionList3 = optionList3
        self.value3 = value3
        self.setcmap = setcmap
        self.cmlf = cmlf
        self.cdir = cdir
        self.title("Colormap Editor")
        self.colors = ['#0000ff', '#00ff00', '#ff0000']  # default three colors
        self.scales = [0, 0.5, 1]
        self.entries = []
        self.scale_entries = []
        self.vmin = tk.DoubleVar(value=0.0)
        self.vmax = tk.DoubleVar(value=1.0)
        self.colormap_name = tk.StringVar(value="custom_cmap")
        self.bind('<Configure>', self.on_configure)
        self._draw_ui()
        set_center(master, self, 0, 0)
        self.update()
        
    def on_configure(self, event):
        if self.winfo_width() != self.winfo_reqwidth() or self.winfo_height() != self.winfo_reqheight():
            self.geometry(f"{self.winfo_reqwidth()}x{self.winfo_reqheight()}")
            set_center(self.master, self, 0, 0)

    def size(self, size: int) -> int:
        return(int(self.scale*size))
    
    def _draw_ui(self):
        for widget in self.winfo_children():
            widget.destroy()
        self.entries.clear()
        self.scale_entries.clear()
        n = len(self.colors)

        # Frame for color buttons and - + buttons
        colorbar = tk.Frame(self)
        colorbar.grid(row=0, column=0, columnspan=10, pady=5)

        # - button (left)
        if n > 2:
            btn_minus = tk.Button(colorbar, font=('Arial', self.size(15)), text=" - ", command=self.remove_node)
            btn_minus.pack(side=tk.LEFT, padx=4)
        else:
            btn_minus = None

        # Color buttons and Entry vertically stacked
        for i, (color, scale) in enumerate(zip(self.colors, self.scales)):
            btn_frame = tk.Frame(colorbar)
            btn_frame.pack(side=tk.LEFT, padx=4)
            btn = tk.Button(btn_frame, bg=color, width=10, font=("Arial", self.size(15)), command=lambda i=i: self.pick_color(i))
            btn.pack(side=tk.TOP)
            self.entries.append(btn)
            scale_entry = tk.Entry(btn_frame, font=("Arial", self.size(15)), width=5, justify='center')
            scale_entry.insert(0, str(scale))
            # 讓第0個和最後一個Entry為readonly
            if i == 0 or i == n - 1:
                scale_entry.config(state='readonly')
            scale_entry.pack(side=tk.TOP, pady=(2, 0))
            self.scale_entries.append(scale_entry)

        # + button (right)
        btn_plus = tk.Button(colorbar, font=("Arial", self.size(15)), text=" + ", command=self.add_node)
        btn_plus.pack(side=tk.LEFT, padx=4)

        # Other widgets
        tk.Label(self, font=("Arial", self.size(15)), text="vmin:").grid(row=3, column=0, sticky='e')
        tk.Entry(self, font=("Arial", self.size(15)), textvariable=self.vmin, width=7).grid(row=3, column=1, sticky='w')
        tk.Label(self, font=("Arial", self.size(15)), text="vmax:").grid(row=3, column=2, sticky='e')
        tk.Entry(self, font=("Arial", self.size(15)), textvariable=self.vmax, width=7).grid(row=3, column=3, sticky='w')
        tk.Label(self, font=("Arial", self.size(15)), text="Colormap Name:").grid(row=4, column=0, sticky='e')
        tk.Entry(self, font=("Arial", self.size(15)), textvariable=self.colormap_name, width=15).grid(row=4, column=1, columnspan=2, sticky='w')
        tk.Button(self, font=("Arial", self.size(15)), text="Show Colormap", command=self.show_colormap_toplevel).grid(row=5, column=0, columnspan=max(3, len(self.colors)), pady=5)
        tk.Button(self, font=("Arial", self.size(15)), text="Register & Save", command=self.register_and_save).grid(row=6, column=0, columnspan=2, pady=5)
        tk.Button(self, font=("Arial", self.size(15)), text="Load Colormap", command=self.load_colormap).grid(row=6, column=2, columnspan=2, pady=5)
    
    def pick_color(self, idx):
        color = colorchooser.askcolor(title="Pick a color")[1]
        if color:
            self.colors[idx] = color
            self.entries[idx].config(bg=color)
        self.focus_set()

    def add_node(self):
        if len(self.colors) >= 10:
            return
        mid = len(self.colors) // 2
        self.colors.insert(mid, '#ffffff')
        # 重新等分 scale
        n = len(self.colors)
        self.scales = [round(i/(n-1), 4) for i in range(n)]
        self._draw_ui()

    def remove_node(self):
        if len(self.colors) > 2:
            self.colors.pop(-2)
            # 重新等分 scale
            n = len(self.colors)
            self.scales = [round(i/(n-1), 4) for i in range(n)]
            self._draw_ui()

    def get_colormap(self):
        try:
            self.scales = [float(e.get()) for e in self.scale_entries]
        except ValueError:
            messagebox.showerror("Error", "Please enter valid scale values.")
            return None
        if not (all(0 <= s <= 1 for s in self.scales) and all(self.scales[i] < self.scales[i+1] for i in range(len(self.scales)-1))):
            messagebox.showerror("Error", "Scales must be increasing and between 0 and 1.")
            return None
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list(self.colormap_name.get(), list(zip(self.scales, self.colors)))
        return cmap

    def show_colormap_toplevel(self):
        cmap = self.get_colormap()
        if cmap is None:
            return
        arr = np.linspace(self.vmin.get(), self.vmax.get(), 100).reshape(1, -1)
        top = tk.Toplevel(self)
        top.title(f"Colormap Preview: {self.colormap_name.get()}")
        fig, ax = plt.subplots(figsize=(5*self.scale, 3*self.scale))
        im = ax.imshow(arr, aspect='auto', cmap=cmap, vmin=self.vmin.get(), vmax=self.vmax.get())
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, orientation='horizontal')
        canvas = FigureCanvasTkAgg(fig, master=top)
        fig.tight_layout()
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        # Release resources when closing
        def on_close():
            plt.close(fig)
            top.destroy()
        top.protocol("WM_DELETE_WINDOW", on_close)

    def register_and_save(self):
        cmap = self.get_colormap()
        if cmap is None:
            return
        name = self.colormap_name.get()
        # Register to matplotlib colormap
        matplotlib.colormaps.register(cmap, name=name, force=True)
        messagebox.showinfo("Colormap", f"Colormap '{name}' has been registered to matplotlib.")
        self.optionList3 = [name, 'prevac_cmap', 'terrain', 'custom_cmap1', 'custom_cmap2', 'custom_cmap3', 'custom_cmap4', 'viridis', 'turbo', 'inferno', 'plasma', 'copper', 'grey', 'bwr']
        self.setcmap.grid_forget()
        self.value3.set(name)
        self.setcmap = tk.OptionMenu(self.cmlf, self.value3, *self.optionList3)
        self.setcmap.grid(row=0, column=1)
        self.master.update()
        # Save file
        data = {
            "colors": np.array(self.colors),
            "scales": np.array(self.scales),
            "vmin": self.vmin.get(),
            "vmax": self.vmax.get(),
            "name": name
        }
        save_path = fd.asksaveasfilename(
            title="Save custom colormap",
            defaultextension=".npz",
            filetypes=[("NumPy zip", "*.npz")],
            initialdir=self.cdir,
            initialfile=f"{name}.npz"
        )
        np.savez(save_path, **data)
        np.savez(os.path.join(self.cdir,".MDC_cut","colormaps.npz"), **data)
        if save_path:
            messagebox.showinfo("Colormap", f"Colormap has been saved to:\n{save_path}")
    
    def load_colormap(self):
        # Load npz file
        load_dir = self.cdir
        file_path = fd.askopenfilename(
            title="Select custom colormap file",
            filetypes=[("NumPy zip", "*.npz")],
            initialdir=load_dir if os.path.exists(load_dir) else "."
        )
        if not file_path:
            self.focus_set()
            return
        try:
            data = np.load(file_path, allow_pickle=True)
            self.colors = list(data["colors"])
            self.scales = list(data["scales"])
            self.vmin.set(float(data["vmin"]))
            self.vmax.set(float(data["vmax"]))
            self.colormap_name.set(str(data["name"]))
            self._draw_ui()
            messagebox.showinfo("Colormap", f"Colormap loaded: {self.colormap_name.get()}")
            cmap = self.get_colormap()
            if cmap is None:
                return
            name = self.colormap_name.get()
            # Register to matplotlib colormap
            matplotlib.colormaps.register(cmap, name=name, force=True)
            self.optionList3.append(name)
            self.setcmap.grid_forget()
            self.value3.set(name)
            self.setcmap = tk.OptionMenu(self.cmlf, self.value3, *self.optionList3)
            self.setcmap.grid(row=0, column=1)
            self.master.update()
            np.savez(os.path.join(self.cdir,".MDC_cut","colormaps.npz"), **data)
            self.focus_set()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load: {e}")

class c_attr_window(RestrictedToplevel, ABC):
    def __init__(self, parent: tk.Misc | None = None, dpath: str='', attr: float|str='', scale: float=1.0):
        super().__init__(parent, bg='white')
        self.scale = scale
        self.dpath = dpath
        self.title('Attributes Editor')
        fr=tk.Frame(self,bg='white')
        fr.grid(row=0,column=0)
        self.t_in = tk.Text(fr)
        self.t_in.grid(row=0,column=0)
        self.t_in.insert(tk.END, str(attr))
        fr1 = tk.Frame(self,bg='white')
        fr1.grid(row=1,column=0)
        b1=tk.Button(fr1,text='Confirm',command=self.attr_save_str, width=15, height=1, font=('Arial', self.size(14), "bold"), bg='white', bd=5)
        b1.grid(row=1,column=0)
        b2=tk.Button(fr1,text='Cancel',command=self.destroy, width=15, height=1, font=('Arial', self.size(14), "bold"), bg='white', bd=5)
        b2.grid(row=1,column=1)
    
    def show(self):
        set_center(self.master, self, 0, 0)
        self.update()
        self.limit_bind()
        self.grab_set()
        self.t_in.focus_set()
    
    def size(self, s: int=16)->int:
        return int(self.scale*s)
    
    
    def attr_save_str(self, *e):
        self.data = None
        s=self.string
        if s:
            tbasename = os.path.basename(self.dpath)
            if '.h5' in tbasename:
                self.attr_h5(s)
            elif '.json' in tbasename:
                self.attr_json(s)
            elif '.npz' in tbasename:
                self.attr_npz(s)
            self.pars()
        self.destroy()
    
    @abstractmethod
    def pars(self):
        pass
    
    @property
    @abstractmethod
    def string(self) -> str:
        pass
    
    @abstractmethod
    def attr_h5(self, s:str):
        pass
    @abstractmethod
    def attr_json(self, s:str):
        pass
    @abstractmethod
    def attr_npz(self, s:str):
        pass
    
    @abstractmethod
    def load_h5(self, dpath: str):
        # Implement loading h5 file and return data
        pass
    @abstractmethod
    def load_json(self, dpath: str):
        # Implement loading json file and return data
        pass
    @abstractmethod
    def load_npz(self, dpath: str):
        # Implement loading npz file and return data
        pass

class c_excitation_window(c_attr_window):
    def __init__(self, parent: tk.Misc | None = None, dpath: str='', e_photon: float=1000.0, scale: float=1.0):
        super().__init__(parent, dpath=dpath, attr=e_photon, scale=scale)
        self.t_in.config(height=1, width=60, font=('Arial', self.size(16)))
        self.bind('<Return>', self.attr_save_str)
        self.t_in.bind('<Return>', self.attr_save_str)
        self.show()
    
    @abstractmethod
    def check_string(self, s:str) -> str:
        pass

    @property
    @override
    def string(self) -> str:
        s=self.t_in.get('1.0',tk.END)
        for i in ['\n\n\n\n\n', '\n\n\n\n', '\n\n\n', '\n\n', '\n']:
            s = s.replace(i, '')
        return self.check_string(s)
    
    @override
    def attr_h5(self, s):
        with h5py.File(self.dpath, 'r+') as hf:
            # Read the dataset
            data = hf['Region']['ExcitationEnergy']['Value'][:]
            print("\nOriginal:", data)
            
            # Prepare the new data
            new_data = np.array([float(s)], dtype=float)  # Use vlen=str for variable-length strings
            
            # Delete the old dataset
            del hf['Region']['ExcitationEnergy']['Value']
            
            # Create a new dataset with the same name but with the new data
            hf.create_dataset('Region/ExcitationEnergy/Value', data=new_data, dtype=float)
            
            # Verify changes
            modified_data = hf['Region']['ExcitationEnergy']['Value'][:]
            print("Modified:", modified_data)
    
    @override
    def attr_json(self, s):
        with open(self.dpath, 'r') as f:
            data = json.load(f)
            print("\nOriginal:", data['Region']['ExcitationEnergy']['Value'])
        data['Region']['ExcitationEnergy']['Value'] = float(s)
        with open(self.dpath, 'w') as f:
            json.dump(data, f, indent=2)
            print("Modified:", data['Region']['ExcitationEnergy']['Value'])
        
    @override
    def attr_npz(self, s):
        with np.load(self.dpath, allow_pickle=True) as data:
            data_dict = {key: data[key] for key in data}
        data_dict['e_photon'] = float(s)
        np.savez(self.dpath, **data_dict)
        print(f"Modified .npz file saved to {self.dpath}")
        
class c_name_window(c_attr_window):
    def __init__(self, parent: tk.Misc | None = None, dpath: str='', name: str='', scale: float=1.0):
        super().__init__(parent, dpath=dpath, attr=name, scale=scale)
        self.t_in.config(height=1, width=60, bd=5, padx=10, pady=10, font=('Arial', self.size(20)))
        self.bind('<Return>', self.attr_save_str)
        self.t_in.bind('<Return>', self.attr_save_str)
        self.show()
    
    @property
    @override
    def string(self) -> str:
        s=self.t_in.get('1.0',tk.END)
        for i in ['\n\n\n\n\n', '\n\n\n\n', '\n\n\n', '\n\n', '\n']:
            s = s.replace(i, '')
        return s
    
    @override
    def attr_h5(self, s:str):
        with h5py.File(self.dpath, 'r+') as hf:
            # Read the dataset
            data = hf['Region']['Name'][:]
            print("\nOriginal:", data)
            
            # Prepare the new data
            new_data = np.array([bytes(s, 'utf-8')], dtype=h5py.special_dtype(vlen=str))  # Use vlen=str for variable-length strings
            
            # Delete the old dataset
            del hf['Region']['Name']
            
            # Create a new dataset with the same name but with the new data
            hf.create_dataset('Region/Name', data=new_data, dtype=h5py.special_dtype(vlen=str))
            
            # Verify changes
            modified_data = hf['Region']['Name'][:]
            print("Modified:", modified_data)

    @override
    def attr_json(self, s:str):
        with open(self.dpath, 'r') as f:
            data = json.load(f)
            print("\nOriginal:", data['Region']['Name'])
        data['Region']['Name'] = s
        with open(self.dpath, 'w') as f:
            json.dump(data, f, indent=2)
            print("Modified:", data['Region']['Name'])
    
    @override
    def attr_npz(self, s:str):
        os.chdir(os.path.dirname(self.dpath))
        old_name = os.path.basename(self.dpath)
        new_name = s+'.npz'
        try:
            os.rename(old_name, new_name)
            print(f"File renamed from {old_name} to {new_name}")
            self.dpath = os.path.normpath(os.path.dirname(self.dpath)+'/'+s+'.npz')
        except FileNotFoundError:
            print(f"File {old_name} not found.")
        except PermissionError:
            print(f"Permission denied to rename {old_name}.")
        except Exception as e:
            print(f"An error occurred: {e}")

class c_description_window(c_attr_window):
    def __init__(self, parent: tk.Misc | None = None, dpath: str='', description: str='', scale: float=1.0):
        super().__init__(parent, dpath=dpath, attr=description, scale=scale)
        self.t_in.config(height=10, width=50, bd=5, padx=10, pady=10, font=('Arial', self.size(16)))
        self.show()
    
    @property
    @override
    def string(self) -> str:
        s=self.t_in.get('1.0',tk.END)
        for i in ['\n\n\n\n\n', '\n\n\n\n', '\n\n\n', '\n\n']:
            s = s.replace(i, '\n')
        return s
    
    @override
    def attr_h5(self, s:str):
        with h5py.File(self.dpath, 'r+') as hf:
            # Read the dataset
            data = hf['Region']['Description'][:]
            print("\nOriginal:", data)
            
            # Prepare the new data
            # s1 = b'BUF : 1.68E-6 mbar'
            # s2 = b'0.50kV 100mA'
            # new_data = np.array([s1, b'\n', s2], dtype=h5py.special_dtype(vlen=str))  # Use vlen=str for variable-length strings
            
            # s='BUF : 1.68E-6 mbar\n0.50kV 100mA'
            new_data = np.array([bytes(s, 'utf-8')], dtype=h5py.special_dtype(vlen=str))  # Use vlen=str for variable-length strings
            
            # Delete the old dataset
            del hf['Region']['Description']
            
            # Create a new dataset with the same name but with the new data
            hf.create_dataset('Region/Description', data=new_data, dtype=h5py.special_dtype(vlen=str))
            
            # Verify changes
            modified_data = hf['Region']['Description'][:]
            print("Modified:", modified_data)

    @override
    def attr_json(self, s:str):
        with open(self.dpath, 'r') as f:
            data = json.load(f)
            print("\nOriginal:", data['Region']['Description'])
        data['Region']['Description'] = s
        with open(self.dpath, 'w') as f:
            json.dump(data, f, indent=2)
            print("Modified:", data['Region']['Description'])
            
    @override
    def attr_npz(self, s:str):
        with np.load(self.dpath, allow_pickle=True) as data:
            data_dict = {key: data[key] for key in data}
        data_dict['desc'] = [s]
        np.savez(self.dpath, **data_dict)
        print(f"Modified .npz file saved to {self.dpath}")

# compare the version with remote repository (only execute when using GUI)
class VersionCheckWindow(tk.Toplevel, ABC):
    def __init__(self, master: tk.Misc | None = None, scale: float = 1.0, cdir: str='', app_name: str='', __version__: str='', hwnd: int=0):
        f = self.check_github_connection()
        if not f:
            return
        icon = IconManager().icon_upgrade
        self.icon = ImageTk.PhotoImage(Image.open(io.BytesIO(b64decode(icon))).resize([150, 150]))
        self.scale = scale
        self.get_src(ver=True)
        path = os.path.join(cdir, '.MDC_cut', 'MDC_cut.py')
        with open(path, mode='r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('__version__ =') or line.startswith("__version__="):
                    remote_ver = line.split('=')[1].strip().strip('"').strip("'")
                    if cal_ver(remote_ver) > cal_ver(__version__):
                        super().__init__(master, bg='white')
                        self.title("Version Check")
                        self.resizable(False, False)
                        fr_label = tk.Frame(self, bg='white')
                        fr_label.pack(pady=10)
                        lb1 = tk.Label(fr_label, image=self.icon, width='150', height='150', bg='white')
                        lb1.pack(side=tk.LEFT)
                        lb2 = tk.Label(fr_label, text=f"A new version {remote_ver} is available.\nUpdate now?", bg='white', font=("Arial", self.size(20), 'bold'))
                        lb2.pack(side=tk.LEFT)
                        def update_now():
                            self.destroy()
                            if os.name == 'nt' and hwnd:
                                windll.user32.ShowWindow(hwnd, 9)
                                windll.user32.SetForegroundWindow(hwnd)
                            print('\033[36m\nUpdating to the latest version...\nPlease wait...\033[0m')
                            self.get_src()
                            v_check_path = os.path.join(cdir, '.MDC_cut', 'version.check')
                            if os.path.exists(v_check_path):
                                with open(v_check_path, mode='w') as f:
                                    f.write(remote_ver)
                            src = path
                            dst = os.path.join(cdir, f'{app_name}.py')
                            if os.name == 'nt':
                                os.system(f'copy "{src}" "{dst}" > nul')
                                os.system(rf'start "" cmd /C "chcp 65001 > nul && python -W ignore::SyntaxWarning -W ignore::UserWarning "{app_name}.py""')
                            elif os.name == 'posix':
                                try:
                                    os.system(f'cp "{src}" "{dst}"')
                                    os.system(rf'start "" cmd /C "chcp 65001 > nul && python3 -W ignore::SyntaxWarning -W ignore::UserWarning "{app_name}.py""')
                                except:
                                    os.system(f'cp "{src}" "{dst}"')
                                    os.system(rf'start "" cmd /C "chcp 65001 > nul && python -W ignore::SyntaxWarning -W ignore::UserWarning "{app_name}.py""')
                            os.remove(src)
                            quit()
                        yn_frame = tk.Frame(self, bg='white')
                        yn_frame.pack(pady=5)
                        btn_update = tk.Button(yn_frame, text="Update", command=update_now, font=("Arial", self.size(16), 'bold'))
                        btn_update.pack(side=tk.LEFT, padx=5)
                        def later():
                            self.destroy()
                        btn_later = tk.Button(yn_frame, text="Later", command=later, font=("Arial", self.size(16), 'bold'))
                        btn_later.pack(side=tk.LEFT, padx=5)
                        set_center(master, self, w_extend=15)
                        self.bind('<Return>', lambda e: update_now())
                        self.grab_set()
                        self.focus_set()
                    break
        os.system(f'del {path}')
        
    def size(self, s: int) -> int:
        return(int(self.scale*s))

    def check_github_connection(self) -> bool:
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

    @abstractmethod
    def get_src(self, ver: bool = False):
        pass

class CalculatorWindow(tk.Toplevel):
    def __init__(self, master: tk.Misc|None=None, scale: float=1.0):
        super().__init__(master, bg='white')
        self.scale = scale
        self.resizable(False, False)
        self.title('E-k Angle Converter')
        fr = tk.Frame(self, bg='white')
        fr.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        calkl = tk.Label(fr, text='delta k (to 0)', font=(
            "Arial", self.size(18), "bold"), bg="white", fg="black")
        calkl.grid(row=1, column=0)
        calel = tk.Label(fr, text='Kinetic Energy', font=(
            "Arial", self.size(18), "bold"), bg="white", fg="black")
        calel.grid(row=2, column=0)

        self.calk = tk.StringVar()
        self.calk.set('0')
        self.calk.trace_add('write', self.cal)
        self.cale = tk.StringVar()
        self.cale.set('21.2')
        self.cale.trace_add('write', self.cal)
        self.calken = tk.Entry(fr, font=("Arial", self.size(18), "bold"),
                        width=15, textvariable=self.calk, bd=9)
        self.calken.grid(row=1, column=1)
        self.caleen = tk.Entry(fr, font=("Arial", self.size(18), "bold"),
                        width=15, textvariable=self.cale, bd=9)
        self.caleen.grid(row=2, column=1)
        
        self.caldeg = tk.Label(self, text='Deg = 0', font=(
            "Arial", self.size(18), "bold"), bg="white", fg="black")
        self.caldeg.pack(side=tk.TOP, fill=tk.X)
        
        set_center(master, self, 0, 0)
        self.calken.focus_set()
        self.calken.select_range(0, tk.END)
    
    def size(self, s: int) -> int:
        return(int(self.scale*s))
    
    def cal_job(self):
        h = 6.62607015*10**-34  # J·s
        m = 9.1093837015*10**-31  # kg
        if '' == self.calk.get():
            self.calk.set('0')
            self.calken.select_range(0, 1)
        if '' == self.cale.get():
            self.cale.set('0')
            self.caleen.select_range(0, 1)
        ans = np.arcsin(np.float64(self.calk.get())/np.sqrt(2*m*np.float64(self.cale.get())
                        * 1.602176634*10**-19)/10**-10*(h/2/np.pi))*180/np.pi
        self.caldeg.config(text='Deg = '+'%.5f' % ans)

    def cal(self, *e):
        Thread(target=self.cal_job, daemon=True).start()

class Plot1Window(RestrictedToplevel, IconManager, ABC):
    def __init__(self, master: tk.Misc | None, scale: float):
        super().__init__(master, bg='white', padx=10, pady=10)
        IconManager.__init__(self)
        self.master = master
        self.scale = scale
        self.iconphoto(False, tk.PhotoImage(data=b64decode(self.gicon)))

    def show(self):
        set_center(self.master, self, 0, 0)
        self.focus_set()
        self.limit_bind()
        self.grab_set()
    
    def size(self, s: int=16)->int:
        return int(self.scale*s)
    
    def select_all(self, event):
        event.widget.select_range(0, tk.END)
        return 'break'
    
    def on_enter(self, event):
        self.chf()
    
    @abstractmethod
    def chf(self):
        pass

class Plot1Window_MDC_curves(Plot1Window):
    def __init__(self, master: tk.Misc | None, scale: float, d: int, l: int, p: int):
        super().__init__(master, scale)
        
        def ini():
            self.v_d.set(str(d))
            self.v_l.set(str(l))
            self.v_p.set(str(p))
            cl.focus()

        self.title('Plotting Parameter')
        fd = tk.Frame(self, bg="white")
        fd.grid(row=0, column=0, padx=10, pady=5)
        ld = tk.Label(fd, text='Energy Axis Density (1/n), n :', font=(
            "Arial", self.size(18), "bold"), bg="white", height='1')
        ld.grid(row=0, column=0, padx=10, pady=10)
        self.v_d = tk.StringVar()
        cd = tk.Entry(fd, font=(
            "Arial", self.size(16), "bold"), textvariable=self.v_d, width=10, bg="white")
        cd.grid(row=0, column=1, padx=10, pady=5)

        fl = tk.Frame(self, bg="white")
        fl.grid(row=1, column=0, padx=10, pady=5)
        ll = tk.Label(fl, text='Savgol Filter Window Length :', font=(
            "Arial", self.size(18), "bold"), bg="white", height='1')
        ll.grid(row=0, column=0, padx=10, pady=10)
        self.v_l = tk.StringVar()
        cl = tk.Entry(fl, font=(
            "Arial", self.size(16), "bold"), textvariable=self.v_l, width=10, bg="white")
        cl.grid(row=0, column=1, padx=10, pady=5)
        
        fp = tk.Frame(self, bg="white")
        fp.grid(row=2, column=0, padx=10, pady=5)
        lp = tk.Label(fp, text='Savgol Filter Polynomial Degree :', font=(
            "Arial", self.size(18), "bold"), bg="white", height='1')
        lp.grid(row=0, column=0, padx=10, pady=10)
        self.v_p = tk.StringVar()
        cp = tk.Entry(fp, font=(
            "Arial", self.size(16), "bold"), textvariable=self.v_p, width=10, bg="white")
        cp.grid(row=0, column=1, padx=10, pady=5)

        l_smooth = tk.Label(self, text='Note:\n\tPolynomial Degree 0 or 1: Moving Average\n\tPolyorder must be less than window_length', font=(
            "Arial", self.size(14), "bold"), bg="white", height='3',justify='left')
        l_smooth.grid(row=3, column=0, padx=10, pady=10)

        bflag = tk.Button(self, text="OK", font=("Arial", self.size(16), "bold"),
                          height=2, width=10, bg="white", command=self.chf)
        bflag.grid(row=4, column=0, padx=10, pady=5)
        
        cd.bind('<FocusIn>', self.select_all)
        cl.bind('<FocusIn>', self.select_all)
        cp.bind('<FocusIn>', self.select_all)
        self.bind('<Return>', self.on_enter)
        self.show()
        ini()

class Plot1Window_Second_Derivative(Plot1Window):
    def __init__(self, master: tk.Misc | None, scale: float, im_kernel: int):
        super().__init__(master, scale)
        
        def ini():
            self.v_k.set(str(im_kernel))
            ck.focus()

        self.title('Gaussian Smoothing Kernel Size')
        fd = tk.Frame(self, bg="white")
        fd.grid(row=0, column=0, padx=10, pady=5)
        ld = tk.Label(fd, text='Kernel Size :', font=(
            "Arial", self.size(18), "bold"), bg="white", height='1')
        ld.grid(row=0, column=0, padx=10, pady=10)
        self.v_k = tk.StringVar()
        ck = tk.Entry(fd, font=(
            "Arial", self.size(16), "bold"), textvariable=self.v_k, width=10, bg="white")
        ck.grid(row=0, column=1, padx=10, pady=5)
        
        l_smooth = tk.Label(self, text='Note:\n\tKernel size must be an odd number', font=(
            "Arial", self.size(14), "bold"), bg="white", height='3',justify='left')
        l_smooth.grid(row=3, column=0, padx=10, pady=10)
        
        bflag = tk.Button(self, text="OK", font=("Arial", self.size(16), "bold"),
                          height=2, width=10, bg="white", command=self.chf)
        bflag.grid(row=4, column=0, padx=10, pady=5)
        
        ck.bind('<FocusIn>', self.select_all)
        self.bind('<Return>', self.on_enter)
        self.show()
        ini()

class Plot3Window(RestrictedToplevel, IconManager, ABC):
    def __init__(self, master: tk.Misc | None, scale: float, fev: list, fk: list):
        super().__init__(master, bg='white', padx=10, pady=10)
        IconManager.__init__(self)
        self.scale = scale
        self.fev = fev
        self.fk = fk

        def on_enter(event):
            self.chf()
        
        self.title('Data Point List')
        self.iconphoto(False, tk.PhotoImage(data=b64decode(self.gicon)))
        lpos = tk.Label(self, text='Position', font=(
            "Arial", self.size(18), "bold"), bg="white", height='1')
        lpos.grid(row=0, column=0, padx=10, pady=10)

        pos = tk.Frame(self, bg="white")
        pos.grid(row=1, column=0, padx=10, pady=5)
        self.v_mpos = tk.IntVar()
        self.mpos = tk.Checkbutton(pos, text="MDC", font=(
            "Arial", self.size(16), "bold"), variable=self.v_mpos, onvalue=1, offvalue=0, height=2, width=10, bg="white")
        self.mpos.grid(row=0, column=0, padx=10, pady=5)
        self.mpos.intvar = self.v_mpos
        self.mpos.select()

        self.v_epos = tk.IntVar()
        self.epos = tk.Checkbutton(pos, text="EDC", font=(
            "Arial", self.size(16), "bold"), variable=self.v_epos, onvalue=1, offvalue=0, height=2, width=10, bg="white")
        self.epos.grid(row=0, column=1, padx=10, pady=5)
        self.epos.intvar = self.v_epos
        self.epos.select()

        lfwhm = tk.Label(self, text='FWHM', font=(
            "Arial", self.size(18), "bold"), bg="white", height='1')
        lfwhm.grid(row=2, column=0, padx=10, pady=10)

        fwhm = tk.Frame(self, bg="white")
        fwhm.grid(row=3, column=0, padx=10, pady=5)
        self.v_mfwhm = tk.IntVar()
        self.mfwhm = tk.Checkbutton(fwhm, text="MDC", font=(
            "Arial", self.size(16), "bold"), variable=self.v_mfwhm, onvalue=1, offvalue=0, height=2, width=10, bg="white")
        self.mfwhm.grid(row=0, column=0, padx=10, pady=5)
        self.mfwhm.intvar = self.v_mfwhm
        self.mfwhm.select()

        self.v_efwhm = tk.IntVar()
        self.efwhm = tk.Checkbutton(fwhm, text="EDC", font=(
            "Arial", self.size(16), "bold"), variable=self.v_efwhm, onvalue=1, offvalue=0, height=2, width=10, bg="white")
        self.efwhm.grid(row=0, column=1, padx=10, pady=5)
        self.efwhm.intvar = self.v_efwhm
        self.efwhm.select()

        bflag = tk.Button(self, text="OK", font=("Arial", self.size(16), "bold"),
                          height=2, width=10, bg="white", command=self.chf)
        bflag.grid(row=4, column=0, padx=10, pady=5)
        set_center(master, self, 0, 0)
        self.bind('<Return>', on_enter)
        self.focus_set()
        self.limit_bind()
        self.grab_set()
        self.ini()
        
    def size(self, s: int=16)->int:
        return int(self.scale*s)
    
    @abstractmethod
    def ini(self):
        pass
    
    @abstractmethod
    def chf(self):
        pass
