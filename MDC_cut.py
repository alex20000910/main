# MDC cut GUI
from lmfit.printfuncs import *
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import SpanSelector
from matplotlib.widgets import RectangleSelector
import matplotlib as mpl
from matplotlib.widgets import Cursor
from PIL import Image, ImageTk
from matplotlib.figure import Figure
import matplotlib.backends.backend_tkagg as tkagg
import matplotlib.pyplot as plt
import os
import json
import tkinter as tk
from tkinter import filedialog as fd
import io
from base64 import b64decode
import queue
import threading
import warnings
from ctypes import windll
import re


def install(s: str):
    print('\n\n"'+s+'" Module Not Found')
    a = input('pip install '+s+' ???\nProceed (Y/n)? ')
    if a == 'Y' or a == 'y':
        os.system('pip install '+s)
    else:
        quit()


try:
    import numpy as np
except ModuleNotFoundError:
    install('numpy')
    import numpy as np
try:
    import xarray as xr
except ModuleNotFoundError:
    install('xarray')
    import xarray as xr
try:
    import h5py
except ModuleNotFoundError:
    install('h5py')
    import h5py
try:
    import matplotlib
except ModuleNotFoundError:
    install('matplotlib')
    import matplotlib
matplotlib.use('TkAgg')
try:
    from scipy.optimize import curve_fit
except ModuleNotFoundError:
    install('scipy')
    from scipy.optimize import curve_fit
try:
    from lmfit import Parameters, Minimizer, report_fit
except ModuleNotFoundError:
    install('lmfit')
    from lmfit import Parameters, Minimizer, report_fit
try:
    import tqdm
except ModuleNotFoundError:
    install('tqdm')
    import tqdm

h = 6.626*10**-34
m = 9.11*10**-31
mp, ep, mf, ef = 1, 1, 1, 1
fk = []
fev = []
cdir = os.getcwd()


def load_txt(path_to_file: str) -> xr.DataArray:
    """
    Load data from a text file and convert it into an xarray DataArray.

    Parameters:
        path_to_file (str): The path to the text file.

    Returns:
        xr.DataArray: The data loaded from the text file as an xarray DataArray.
    """
    print(path_to_file)
    with open(path_to_file, 'r') as file:
        i = 0
        a = []
        e = []
        for line in file:
            if i == 1:
                a = line.split('\t')[2::]
            if i >= 2:
                e.append(line.split('\t')[1])
            i += 1
    d = np.arange(float(len(e)*len(a))).reshape(len(e), len(a))
    with open(path_to_file, 'r') as file:
        i = 0
        for line in file:
            if i >= 2:
                d[i-2][:] = line.split('\t')[2::]
                if i == 2:
                    t = line.split('\t')[2::]
            i += 1
    e_low = np.min(np.float64(e))
    e_high = np.max(np.float64(e))
    e_num = len(e)
    e_photon = 21.2
    a_low = np.min(np.float64(a))
    a_high = np.max(np.float64(a))
    a_num = len(a)
    #   attrs
    e_mode = 'Kinetic'
    LensMode = 'Unknown'
    PassEnergy = 5
    Dwell = 'Unknown'
    CenterEnergy = np.average(np.float64(e))
    Iterations = 'Unknown'
    Step = 'Unknown'
    Slit = 'Unknown'
    aq = 'Unknown'
    name = path_to_file.split('/')
    name = name[-1].removesuffix('.txt')
    if e_mode == 'Kinetic':
        e = np.linspace(e_low, e_high, e_num)
        CenterEnergy = str(CenterEnergy)+' eV (K.E.)'
        e_low = str(e_low)+' eV (K.E.)'
        e_high = str(e_high)+' eV (K.E.)'
    else:
        e = np.linspace(e_photon-e_high, e_photon-e_low, e_num)
        CenterEnergy = str(CenterEnergy)+' eV (B.E.)'
        e_low = str(e_low)+' eV (B.E.)'
        e_high = str(e_high)+' eV (B.E.)'

    a = np.linspace(a_low, a_high, a_num)
    # data=np.arange(float(len(e)*len(a))).reshape(len(e),len(a),1)
    # data[0:,0:,0]=d
    data = np.arange(float(len(e)*len(a))).reshape(len(e), len(a))
    data[0:, 0:] = d
    data = xr.DataArray(data, coords={'eV': e, 'phi': a}, attrs={'Name': name,
                                                                 'Acquisition': aq,
                                                                 'EnergyMode': e_mode,
                                                                 'ExcitationEnergy': str(e_photon)+' eV',
                                                                 'CenterEnergy': CenterEnergy,
                                                                 'HighEnergy': e_high,
                                                                 'LowEnergy': e_low,
                                                                 'Step': str(Step)+' eV',
                                                                 'LensMode': LensMode,
                                                                 'PassEnergy': str(PassEnergy)+' meV',
                                                                 'Slit': Slit,
                                                                 'Dwell': str(Dwell)+' s',
                                                                 'Iterations': Iterations
                                                                 })
    return data


def load_json(path_to_file: str) -> xr.DataArray:
    """
    Load data from a JSON file and convert it into an xarray DataArray.

    Parameters:
        path_to_file (str): The path to the JSON file.

    Returns:
        xr.DataArray: The data loaded from the JSON file as an xarray DataArray.
    """
    f = json.load(open(path_to_file, 'r'))
    e_low = np.array(f['Region']['LowEnergy']['Value'])
    e_high = np.array(f['Region']['HighEnergy']['Value'])
    e_num = np.array(f['Data']['XSize']['Value'])
    e_photon = np.array(f['Region']['ExcitationEnergy']['Value'])
    a_low = np.array(f['Region']['YScaleMin']['Value'])
    a_high = np.array(f['Region']['YScaleMax']['Value'])
    a_num = np.array(f['Data']['YSize']['Value'])
    #   attrs
    e_mode = f['Region']['EnergyMode']
    LensMode = f['Region']['LensMode']
    PassEnergy = f['Region']['PassEnergy']['Value']
    Dwell = f['Region']['Dwell']['Value']
    CenterEnergy = f['Region']['CenterEnergy']['Value']
    Iterations = f['Region']['Iterations']['Value']
    Step = f['Region']['Step']['Value']
    Slit = f['Region']['Slit']
    aq = f['Region']['Acquisition']
    name = f['Region']['Name']
    if e_mode == 'Kinetic':
        e = np.linspace(e_low, e_high, e_num)
        CenterEnergy = str(CenterEnergy)+' eV (K.E.)'
        e_low = str(e_low)+' eV (K.E.)'
        e_high = str(e_high)+' eV (K.E.)'
    else:
        e = np.linspace(e_photon-e_high, e_photon-e_low, e_num)
        CenterEnergy = str(CenterEnergy)+' eV (B.E.)'
        e_low = str(e_low)+' eV (B.E.)'
        e_high = str(e_high)+' eV (B.E.)'

    a = np.linspace(a_low, a_high, a_num)
    d = np.array(f['Spectrum']).transpose()
    # data=np.arange(float(len(e)*len(a))).reshape(len(e),len(a),1)
    # data[0:,0:,0]=d
    data = np.arange(float(len(e)*len(a))).reshape(len(e), len(a))
    data[0:, 0:] = d
    data = xr.DataArray(data, coords={'eV': e, 'phi': a}, attrs={'Name': name,
                                                                 'Acquisition': aq,
                                                                 'EnergyMode': e_mode,
                                                                 'ExcitationEnergy': str(e_photon)+' eV',
                                                                 'CenterEnergy': CenterEnergy,
                                                                 'HighEnergy': e_high,
                                                                 'LowEnergy': e_low,
                                                                 'Step': str(Step)+' eV',
                                                                 'LensMode': LensMode,
                                                                 'PassEnergy': str(PassEnergy)+' eV',
                                                                 'Slit': Slit,
                                                                 'Dwell': str(Dwell)+' s',
                                                                 'Iterations': Iterations
                                                                 })
    return data


def load_h5(path_to_file: str) -> xr.DataArray:
    """
    Load data from an HDF5 file and return it as a DataArray.

    Parameters:
        path_to_file (str): The path to the HDF5 file.

    Returns:
        xr.DataArray: The loaded data as a DataArray.

    """
    f = h5py.File(path_to_file, 'r')
    e_low = np.array(f.get('Region').get('LowEnergy').get('Value'))[0]
    e_high = np.array(f.get('Region').get('HighEnergy').get('Value'))[0]
    e_num = np.array(f.get('Data').get('XSize').get('Value'))[0]
    e_photon = np.array(f.get('Region').get(
        'ExcitationEnergy').get('Value'))[0]
    a_low = np.array(f.get('Region').get('YScaleMin').get('Value'))[0]
    a_high = np.array(f.get('Region').get('YScaleMax').get('Value'))[0]
    a_num = np.array(f.get('Data').get('YSize').get('Value'))[0]
    #   attrs
    t_e_mode = np.array(f.get('Region').get('EnergyMode'), dtype=str)
    t_LensMode = np.array(f.get('Region').get('LensMode'), dtype=str)
    PassEnergy = np.array(f.get('Region').get(
        'PassEnergy').get('Value'), dtype=str)[0]
    Dwell = np.array(f.get('Region').get('Dwell').get('Value'), dtype=str)[0]
    CenterEnergy = np.array(f.get('Region').get(
        'CenterEnergy').get('Value'), dtype=str)[0]
    Iterations = np.array(f.get('Region').get(
        'Iterations').get('Value'), dtype=str)[0]
    Step = np.array(f.get('Region').get('Step').get('Value'), dtype=str)[0]
    t_Slit = np.array(f.get('Region').get('Slit'), dtype=str)
    t_aq = np.array(f.get('Region').get('Acquisition'), dtype=str)
    t_name = np.array(f.get('Region').get('Name'), dtype=str)
    e_mode = ''
    LensMode = ''
    Slit = ''
    aq = ''
    name = ''
    for i in range(60):  # proper length long enough
        e_mode += t_e_mode[i]
        LensMode += t_LensMode[i]
        Slit += t_Slit[i]
        aq += t_aq[i]
        name += t_name[i]
    if e_mode == 'Kinetic':
        e = np.linspace(e_low, e_high, e_num)
        CenterEnergy = str(CenterEnergy)+' eV (K.E.)'
        e_low = str(e_low)+' eV (K.E.)'
        e_high = str(e_high)+' eV (K.E.)'
    else:
        e = np.linspace(e_photon-e_high, e_photon-e_low, e_num)
        CenterEnergy = str(CenterEnergy)+' eV (B.E.)'
        e_low = str(e_low)+' eV (B.E.)'
        e_high = str(e_high)+' eV (B.E.)'

    a = np.linspace(a_low, a_high, a_num)
    d = np.array(f.get('Spectrum')).transpose()
    # data=np.arange(float(len(e)*len(a))).reshape(len(e),len(a),1)
    # data[0:,0:,0]=d
    data = np.arange(float(len(e)*len(a))).reshape(len(e), len(a))
    data[0:, 0:] = d
    data = xr.DataArray(data, coords={'eV': e, 'phi': a}, attrs={'Name': name,
                                                                 'Acquisition': aq,
                                                                 'EnergyMode': e_mode,
                                                                 'ExcitationEnergy': str(e_photon)+' eV',
                                                                 'CenterEnergy': CenterEnergy,
                                                                 'HighEnergy': e_high,
                                                                 'LowEnergy': e_low,
                                                                 'Step': str(Step)+' eV',
                                                                 'LensMode': LensMode,
                                                                 'PassEnergy': str(PassEnergy)+' eV',
                                                                 'Slit': Slit,
                                                                 'Dwell': str(Dwell)+' s',
                                                                 'Iterations': Iterations
                                                                 })
    return data


def rplot(f, canvas):
    """
    Plot the raw data on a given canvas.

    Parameters:
    - f: Figure object
        The figure object on which the plot will be created.
    - canvas: Canvas object
        The canvas object on which the plot will be drawn.

    Returns:
    None
    """
    global data, ev, phi, value3, h0, ao, xl, yl, rcx, rcy
    ao = f.add_axes([0.13, 0.1, 0.68, 0.6])
    rcx = f.add_axes([0.13, 0.77, 0.545, 0.16])
    rcy = f.add_axes([0.82, 0.1, 0.12, 0.6])
    rcx.set_xticks([])
    rcy.set_yticks([])
    tx, ty = np.meshgrid(phi, ev)
    tz = data.to_numpy()
    # h1 = a.scatter(tx,ty,c=tz,marker='o',s=0.9,cmap=value3.get());
    h0 = ao.pcolormesh(tx, ty, tz, cmap=value3.get())
    f.colorbar(h0)
    # a.set_title('Raw Data',font='Arial',fontsize=16)
    ao.set_title('Raw Data', font='Arial', fontsize=16)
    ao.set_xlabel(r'$\phi$ (deg)', font='Arial', fontsize=12)
    ao.set_ylabel('Kinetic Energy (eV)', font='Arial', fontsize=12)
    xl = ao.get_xlim()
    yl = ao.get_ylim()
    # a.set_xticklabels(labels=a.get_xticklabels(),font='Arial',fontsize=10);
    # a.set_yticklabels(labels=a.get_yticklabels(),font='Arial',fontsize=10);
    canvas.draw()


def o_cal(*e):
    """
    Calculate the angle in degrees based on the given values of calk and cale.

    Parameters:
    *e : tuple
        Variable-length argument list.

    Returns:
    float
        The calculated angle in degrees.

    Raises:
    None

    """
    global calk, cale
    if '' == calk.get():
        calk.set('0')
        calken.select_range(0, 1)
    if '' == cale.get():
        cale.set('0')
        caleen.select_range(0, 1)
    ans = np.arcsin(np.float64(calk.get())/(2*m*np.float64(cale.get())
                    * 1.6*10**-19)**0.5/10**-10*(h/2/np.pi))*180/np.pi
    caldeg.config(text='Deg = '+'%.5f' % ans)


import threading

def cal(*e):
    """
    Perform a calculation using the given arguments.

    Args:
        *e: Variable number of arguments.

    Returns:
        None
    """
    t = threading.Thread(target=o_cal)
    t.daemon = True
    t.start()


def pr_load(data, path):
    global name
    key = str(data.attrs.keys()).removeprefix(
        'dict_keys([').removesuffix('])').split(',')
    lst = np.arange(len(key))
    value = str(data.attrs.values()).removeprefix(
        'dict_values([').removesuffix('])').split(',')
    name = str('%s' % value[0].removeprefix("'").removesuffix("'"))
    print('%s :' % key[0].removeprefix("'").removesuffix("'"),
          '%s' % value[0].removeprefix("'").removesuffix("'"))
    st = '%s : ' % key[0].removeprefix("'").removesuffix(
        "'")+'%s' % value[0].removeprefix("'").removesuffix("'")+'\n'
    lst[0] = len(st.split('\n')[0])
    for i in range(1, len(key)):
        st += '%s : ' % key[i].removeprefix(" '").removesuffix(
            "'")+'%s' % value[i].removeprefix(" '").removesuffix("'")+'\n'
        lst[i] = len(st.split('\n')[i])
        print('%s :' % key[i].removeprefix(" '").removesuffix(
            "'"), '%s' % value[i].removeprefix(" '").removesuffix("'"))
    print()
    # info.config(text=st,justify='left')
    info.config(state='normal')
    info.insert(tk.END, st+'\n')
    info.see(tk.END)
    info.config(height=len(key)+2, width=max(lst), state='disabled')
    global ev, phi
    ev, phi = data.indexes.values()
    ev = np.float64(ev)
    phi = np.float64(phi)
    os.chdir(cdir)
    np.savez('rd', path=path, name=name, ev=ev,
             phi=phi, st=st, key=key, lst=lst)


fpr = 0


def o_load():
    global data, h, m, limg, img, rdd, path, st, fpr
    tpath = fd.askopenfilename(title="Select Raw Data", filetypes=(
        ("HDF5 files", "*.h5"), ("JSON files", "*.json"), ("TXT files", "*.txt")))
    st.put('Loading...')
    if len(tpath) > 2:
        rdd = tpath
        fpr = 0
    else:
        rdd = path
    limg.config(image=img[np.random.randint(len(img))])
    if '.h5' in tpath:
        data = load_h5(tpath)  # data save as xarray.DataArray format
        pr_load(data, tpath)
        st.put('Done')
    elif '.json' in tpath:
        data = load_json(tpath)
        pr_load(data, tpath)
        st.put('Done')
    elif '.txt' in tpath:
        data = load_txt(tpath)
        pr_load(data, tpath)
        st.put('Done')
    else:
        st.put('')
        pass
    #   print Attributes


def o_ecut():
    global data, ev, phi, mfpath, limg, img, name, rdd, st
    limg.config(image=img[np.random.randint(len(img))])
    mfpath = ''
    os.chdir(rdd.removesuffix(rdd.split('/')[-1]))
    try:
        ndir = os.getcwd()
        if ndir.split('\\')[-1] == name+'_MDC_'+lowlim.get():
            os.chdir('../')
        os.chdir(ndir)
        os.makedirs(name+'_MDC_'+lowlim.get())
    except:
        pass
    os.chdir(name+'_MDC_'+lowlim.get())
    pbar = tqdm.tqdm(total=len(ev), desc='MDC', colour='green')
    for n in range(len(ev)):
        ecut = data.sel(eV=ev[n], method='nearest')
        x = (2*m*ev[n]*1.6*10**-19)**0.5 * \
            np.sin(phi/180*np.pi)*10**-10/(h/2/np.pi)
        y = ecut.to_numpy().reshape(len(x))
        y = np.where(y > int(lowlim.get()), y, int(lowlim.get()))
        path = 'ecut_%.3f.txt' % ev[n]
        mfpath += path
        pbar.update(1)
        # print(n+1,'/',len(ev))
        if (n+1) % (len(ev)//100) == 0:
            # print(str(round((n+1)/len(ev)*100))+'%'+' ('+str(len(ev))+')')
            st.put(str(round((n+1)/len(ev)*100))+'%'+' ('+str(len(ev))+')')
        f = open(path, 'w', encoding='utf-8')  # tab 必須使用 '\t' 不可"\t"
        f.write('#Wave Vector'+'\t'+'#Intensity'+'\n')
        for i in range(len(x)-1, -1, -1):
            f.write('%-6e' % x[i]+'\t'+'%-6e' % y[i]+'\n')
        f.close()
    os.chdir(cdir)
    np.savez('mfpath', mfpath=mfpath)
    os.chdir(rdd.removesuffix(rdd.split('/')[-1]))
    pbar.close()
    print('Done')
    st.put('Done')


def o_angcut():
    global data, ev, phi, efpath, limg, img, name, rdd, st
    limg.config(image=img[np.random.randint(len(img))])
    efpath = ''
    os.chdir(rdd.removesuffix(rdd.split('/')[-1]))
    try:
        ndir = os.getcwd()
        if ndir.split('\\')[-1] == name+'_EDC'+lowlim.get():
            os.chdir('../')
        os.chdir(ndir)
        os.makedirs(name+'_EDC'+lowlim.get())
    except:
        pass
    os.chdir(name+'_EDC'+lowlim.get())
    pbar = tqdm.tqdm(total=len(phi), desc='EDC', colour='blue')
    for n in range(len(phi)):
        angcut = data.sel(phi=phi[n], method='nearest')
        x = ev
        y = angcut.to_numpy().reshape(len(x))
        y = np.where(y > int(lowlim.get()), y, int(lowlim.get()))
        path = 'angcut_%.5d.txt' % (phi[n]*1000)
        efpath += path
        pbar.update(1)
        # print(n+1,'/',len(phi))
        if (n+1) % (len(phi)//100) == 0:
            # print(str(round((n+1)/len(phi)*100))+'%'+' ('+str(len(phi))+')')
            st.put(str(round((n+1)/len(phi)*100))+'%'+' ('+str(len(phi))+')')
        f = open(path, 'w', encoding='utf-8')  # tab 必須使用 '\t' 不可"\t"
        f.write('#Wave Vector'+'\t'+'#Intensity'+'\n')
        for i in range(len(x)-1, -1, -1):
            f.write('%-6e' % x[i]+'\t'+'%-6e' % y[i]+'\n')
        f.close()
    os.chdir(cdir)
    np.savez('efpath', efpath=efpath)
    os.chdir(rdd.removesuffix(rdd.split('/')[-1]))
    pbar.close()
    print('Done')
    st.put('Done')


def re(a, b):
    det = [1 for i in range(len(a)-1)]
    while sum(det) != 0:
        for i in range(len(a)-1):
            if a[i+1] < a[i]:
                det[i] = 1
                a[i+1], a[i] = a[i], a[i+1]
                b[i+1], b[i] = b[i], b[i+1]
            else:
                det[i] = 0
    return b


def loadmfit_re():
    file = fd.askopenfilename(
        title="Select MDC Fitted file", filetypes=(("VMS files", "*.vms"),))
    global st
    global data, rdd, lmgg
    mfpath = ''
    yy = []
    for n in range(len(ev)):
        ecut = data.sel(eV=ev[n], method='nearest')
        y = ecut.to_numpy().reshape(len(phi))
        y = np.where(y > int(lowlim.get()), y, int(lowlim.get()))
        yy.append(y)
        path = 'ecut_%.3f.txt' % ev[n]
        mfpath += path
    if len(file) > 2:
        rdd = file
        print('Loading...')
        st.put('Loading...')
    else:
        rdd = path
        lmgg.destroy()
    if ".vms" in file:
        n = -1
        fev = np.array([], dtype=float)
        t_fwhm = []
        t_pos = []
        t_kmax = []
        t_kmin = []
        os.chdir(rdd.removesuffix(rdd.split('/')[-1]))
        fc = open('rev_'+file.split('/')[-1], 'w', encoding='utf-8')
        ff = open(name+'_mdc_fitted_raw_data.txt', 'w',
                  encoding='utf-8')  # tab 必須使用 '\t' 不可"\t"
        ff.write('K.E. (eV)'+'\t'+'FWHM (k)'+'\t'+'Position (k)'+'\n')
        try:
            with open(file) as f:
                f1 = 0
                f2 = 0
                indf = 0
                for i, line in enumerate(f):
                    if line[0:11] in mfpath:
                        fi = int(mfpath.find(line[0:11])/15)
                        n = -1
                        f1 = 0
                        f2 = 0
                        indf = 0
                    if line[0:22] == 'CASA region (*Survey*)':
                        tkmax = line.split(' ')[4]
                        tkmin = line.split(' ')[5]
                        ts = line.split(' ')
                        if float(tkmax) > 1000:
                            ts[4], ts[5] = str(
                                round(-(float(ts[5])-1486.6)+1486.6, 6)), str(round(-(float(ts[4])-1486.6)+1486.6, 6))
                        else:
                            ts[4], ts[5] = str(
                                round(-float(ts[5]), 6)), str(round(-float(ts[4]), 6))
                        fc.write(' '.join(ts))
                    elif line[0:12] == 'CASA comp (*':  # 若無篩選條件   indent於此if以下
                        tpos = line.split(' ')[17]
                        tfwhm = line.split(' ')[11]
                        area = line.split(' ')[5]
                        # tkmax=line.split(' ')[18]
                        # tkmin=line.split(' ')[19]
                        #####################################
                        s = line.split(' ')
                        if float(tpos) > 1000:
                            s[17] = str(
                                round(-(float(s[17])-1486.6)+1486.6, 6))
                            s[18], s[19] = str(
                                round(-(float(s[19])-1486.6)+1486.6, 6)), str(round(-(float(s[18])-1486.6)+1486.6, 6))
                        else:
                            s[17] = str(round(-float(s[17]), 6))
                            s[18], s[19] = str(
                                round(-float(s[19]), 6)), str(round(-float(s[18]), 6))
                        fc.write(' '.join(s))
                        '''
                        s=line.split(' ')
                        s[2]='(*Survey_*)'
                        s[8]=str(0)
                        s[9]=str(0.3)     #Area B=A*2
                        s[14]=str(0)
                        s[15]=str(1)    #FWHM B=A*1
                        
                        s[17]=str(round(float(s[17]),6))
                        s[18]=str(round(float(s[18]),6))
                        s[17]=str(round(float(s[17])-0.05,6))
                        s[18]=ts[4]
                        s[19]=ts[5]
                        fc.write(' '.join(s))
                        '''
                        #####################################

                        # 以下if判斷式區段---------可自訂篩選條件------可多層if-----注意indent----------條件篩選值可至 xxxx_fitted_raw_data.txt---檢查需求
                        ##################################################################################################
                        ##################################################################################################
                        # area tfwhm,tpos(1486.6+...)
                        if (ev[fi] > 20.58 and np.float64(tpos) < 1486.6+0.023) or (ev[fi] < 20.58 and np.float64(tpos) > 1486.6+0.023) or 1 == 1:
                            fev = np.append(fev, ev[fi])  # 內容勿動 indent小最內圈if一階
                            t_fwhm.append(tfwhm)  # 內容勿動 indent小最內圈if一階
                            t_pos.append(tpos)  # 內容勿動 indent小最內圈if一階
                            t_kmax.append(tkmax)
                            t_kmin.append(tkmin)
                            if fi not in smfi:
                                smfi.append(fi)
                                skmin.append(tkmin)
                                skmax.append(tkmax)
                            elif fi in smfi:
                                smfp[fi] += 1
                            if float(tpos) > 1000:
                                # 內容勿動 indent小最內圈if一階
                                ff.write(
                                    str(ev[fi])+'\t'+tfwhm+'\t'+str(np.float64(tpos)-1486.6)+'\n')
                            else:
                                ff.write(str(ev[fi])+'\t'+tfwhm +
                                         '\t'+str(np.float64(tpos))+'\n')
                        ##################################################################################################
                        ##################################################################################################
                    elif line[0:100] == 'XPS\n':
                        fc.write('XPS'+'\n')
                        indf = 1
                    elif line[0:100] == 'Al\n':
                        fc.write('Al'+'\n')
                        indf = 0
                    elif line[0:100] == '494\n' and f1 == 0:
                        if indf == 1:
                            fc.write('494'+'\n')
                            indf = 0
                        else:
                            f1 = 1
                            ti = i
                    elif line[0:100] == '0\n' and f1 == 1:
                        if ti == i-1:
                            f2 = 1
                            ti = i
                        else:
                            f1 = 0
                            ti = 0
                    elif line[0:100] == '1\n' and f2 == 1:
                        if ti == i-1:
                            ti = 0
                            f2 = 0
                            fc.write('494'+'\n')
                            fc.write('0'+'\n')
                            fc.write('1'+'\n')
                            for j in range(len(phi)):
                                # fc.write(str(int(yy[fi][-j-1]))+'\n')
                                fc.write(str(int(yy[fi][j]))+'\n')
                            n = len(phi)
                        else:
                            f1 = 0
                            f2 = 0
                            ti = 0
                    else:
                        if n <= 0:
                            if f2 == 1:
                                fc.write('494'+'\n')
                                fc.write('0'+'\n')
                                f2 = 0
                                f1 = 0
                            elif f1 == 1:
                                fc.write('494'+'\n')
                                f1 = 0
                            fc.write(line)
                        if n > 0:
                            f1 = 2
                            n -= 1
                            if n == 0:
                                f1 = 0
                    # pass  # process line i      #勿動
        except UnicodeDecodeError:
            with open(file, encoding='utf-8') as f:
                f1 = 0
                f2 = 0
                indf = 0
                for i, line in enumerate(f):
                    if line[0:11] in mfpath:
                        fi = int(mfpath.find(line[0:11])/15)
                        n = -1
                        f1 = 0
                        f2 = 0
                        indf = 0
                    if line[0:22] == 'CASA region (*Survey*)':
                        tkmax = line.split(' ')[4]
                        tkmin = line.split(' ')[5]
                        ts = line.split(' ')
                        if float(tkmax) > 1000:
                            ts[4], ts[5] = str(
                                round(-(float(ts[5])-1486.6)+1486.6, 6)), str(round(-(float(ts[4])-1486.6)+1486.6, 6))
                        else:
                            ts[4], ts[5] = str(
                                round(-float(ts[5]), 6)), str(round(-float(ts[4]), 6))
                        fc.write(' '.join(ts))
                    elif line[0:12] == 'CASA comp (*':  # 若無篩選條件   indent於此if以下
                        tpos = line.split(' ')[17]
                        tfwhm = line.split(' ')[11]
                        area = line.split(' ')[5]
                        # tkmax=line.split(' ')[18]
                        # tkmin=line.split(' ')[19]
                        #####################################
                        s = line.split(' ')
                        if float(tpos) > 1000:
                            s[17] = str(
                                round(-(float(s[17])-1486.6)+1486.6, 6))
                            s[18], s[19] = str(
                                round(-(float(s[19])-1486.6)+1486.6, 6)), str(round(-(float(s[18])-1486.6)+1486.6, 6))
                        else:
                            s[17] = str(round(-float(s[17]), 6))
                            s[18], s[19] = str(
                                round(-float(s[19]), 6)), str(round(-float(s[18]), 6))
                        fc.write(' '.join(s))
                        '''
                        s=line.split(' ')
                        s[2]='(*Survey_*)'
                        s[8]=str(0)
                        s[9]=str(0.3)     #Area B=A*2
                        s[14]=str(0)
                        s[15]=str(1)    #FWHM B=A*1
                        
                        s[17]=str(round(float(s[17]),6))
                        s[18]=str(round(float(s[18]),6))
                        s[17]=str(round(float(s[17])-0.05,6))
                        s[18]=ts[4]
                        s[19]=ts[5]
                        fc.write(' '.join(s))
                        '''
                        #####################################

                        # 以下if判斷式區段---------可自訂篩選條件------可多層if-----注意indent----------條件篩選值可至 xxxx_fitted_raw_data.txt---檢查需求
                        ##################################################################################################
                        ##################################################################################################
                        # area tfwhm,tpos(1486.6+...)
                        if (ev[fi] > 20.58 and np.float64(tpos) < 1486.6+0.023) or (ev[fi] < 20.58 and np.float64(tpos) > 1486.6+0.023) or 1 == 1:
                            fev = np.append(fev, ev[fi])  # 內容勿動 indent小最內圈if一階
                            t_fwhm.append(tfwhm)  # 內容勿動 indent小最內圈if一階
                            t_pos.append(tpos)  # 內容勿動 indent小最內圈if一階
                            t_kmax.append(tkmax)
                            t_kmin.append(tkmin)
                            if fi not in smfi:
                                smfi.append(fi)
                                skmin.append(tkmin)
                                skmax.append(tkmax)
                            elif fi in smfi:
                                smfp[fi] += 1
                            if float(tpos) > 1000:
                                # 內容勿動 indent小最內圈if一階
                                ff.write(
                                    str(ev[fi])+'\t'+tfwhm+'\t'+str(np.float64(tpos)-1486.6)+'\n')
                            else:
                                ff.write(str(ev[fi])+'\t'+tfwhm +
                                         '\t'+str(-np.float64(tpos))+'\n')
                        ##################################################################################################
                        ##################################################################################################
                    elif line[0:100] == 'XPS\n':
                        fc.write('XPS'+'\n')
                        indf = 1
                    elif line[0:100] == 'Al\n':
                        fc.write('Al'+'\n')
                        indf = 0
                    elif line[0:100] == '494\n' and f1 == 0:
                        if indf == 1:
                            fc.write('494'+'\n')
                            indf = 0
                        else:
                            f1 = 1
                            ti = i
                    elif line[0:100] == '0\n' and f1 == 1:
                        if ti == i-1:
                            f2 = 1
                            ti = i
                        else:
                            f1 = 0
                            ti = 0
                    elif line[0:100] == '1\n' and f2 == 1:
                        if ti == i-1:
                            ti = 0
                            f2 = 0
                            fc.write('494'+'\n')
                            fc.write('0'+'\n')
                            fc.write('1'+'\n')
                            for j in range(len(phi)):
                                # fc.write(str(int(yy[fi][-j-1]))+'\n')
                                fc.write(str(int(yy[fi][j]))+'\n')
                            n = len(phi)
                        else:
                            f1 = 0
                            f2 = 0
                            ti = 0
                    else:
                        if n <= 0:
                            if f2 == 1:
                                fc.write('494'+'\n')
                                fc.write('0'+'\n')
                                f2 = 0
                                f1 = 0
                            elif f1 == 1:
                                fc.write('494'+'\n')
                                f1 = 0
                            fc.write(line)
                        if n > 0:
                            f1 = 2
                            n -= 1
                            if n == 0:
                                f1 = 0
                    # pass  # process line i      #勿動
        ff.close()
        fc.close()
        os.chdir(cdir)
        print('Done')
        st.put('Done')
        lmgg.destroy()


def loadmfit_():
    file = fd.askopenfilename(title="Select MDC Fitted file", filetypes=(
        ("VMS files", "*.vms"), ("NPZ files", "*.npz"),))
    global h, m, fwhm, fev, pos, limg, img, name, ophi, rpos, st, kmax, kmin, lmgg
    global data, rdd, skmin, skmax, smaa1, smaa2, smfp, smfi, fpr, mfi_x, smresult
    mfpath = ''
    yy = []
    for n in range(len(ev)):
        ecut = data.sel(eV=ev[n], method='nearest')
        y = ecut.to_numpy().reshape(len(phi))
        y = np.where(y > int(lowlim.get()), y, int(lowlim.get()))
        yy.append(y)
        path = 'ecut_%.3f.txt' % ev[n]
        mfpath += path
    if len(file) > 2:
        fpr = 0
        rdd = file
        print('Loading...')
        st.put('Loading...')
    else:
        rdd = path
        lmgg.destroy()
    if ".vms" in file:
        n = -1
        fev = np.array([], dtype=float)
        mfi_x = np.arange(len(ev))
        t_fwhm = []
        t_pos = []
        t_kmax = []
        t_kmin = []
        smfi = []
        skmin = []
        skmax = []
        smfp = [1 for i in range(len(ev))]
        os.chdir(rdd.removesuffix(rdd.split('/')[-1]))
        fc = open('copy2p_'+file.split('/')[-1], 'w', encoding='utf-8')
        ff = open(name+'_mdc_fitted_raw_data.txt', 'w',
                  encoding='utf-8')  # tab 必須使用 '\t' 不可"\t"
        ff.write('K.E. (eV)'+'\t'+'FWHM (k)'+'\t'+'Position (k)'+'\n')
        try:
            with open(file) as f:
                for i, line in enumerate(f):
                    if line[0:11] in mfpath:
                        fi = int(mfpath.find(line[0:11])/15)
                    if line[0:22] == 'CASA region (*Survey*)':
                        tkmax = line.split(' ')[4]
                        tkmin = line.split(' ')[5]
                    # 若無篩選條件   indent於此if以下
                    elif line[0:20] == 'CASA comp (*Survey*)':
                        tpos = line.split(' ')[17]
                        tfwhm = line.split(' ')[11]
                        area = line.split(' ')[5]
                        # tkmax=line.split(' ')[18]
                        # tkmin=line.split(' ')[19]
                        # 以下if判斷式區段---------可自訂篩選條件------可多層if-----注意indent----------條件篩選值可至 xxxx_fitted_raw_data.txt---檢查需求
                        ##################################################################################################
                        ##################################################################################################
                        # area tfwhm,tpos(1486.6+...)
                        if (ev[fi] > 20.58 and np.float64(tpos) < 1486.6+0.023) or (ev[fi] < 20.58 and np.float64(tpos) > 1486.6+0.023) or 1 == 1:
                            tkk = (2*m*ev[fi]*1.6*10**-19)**0.5 * \
                                np.sin(phi/180*np.pi)*10**-10/(h/2/np.pi)
                            d = tkk[1]-tkk[0]
                            tr = float(tpos)+float(tfwhm)/2
                            tl = float(tpos)-float(tfwhm)/2
                            ri = int((tr-tkk[0])/d)
                            li = int((tl-tkk[0])/d)
                            if ri > 492:
                                ri = 492
                            tr = tkk[ri]+(float(tr)-(tkk[0]+ri*d)
                                          )/d*(tkk[ri+1]-tkk[ri])
                            tl = tkk[li]+(float(tl)-(tkk[0]+li*d)
                                          )/d*(tkk[li+1]-tkk[li])
                            tfwhm = tr-tl
                            tpi = int((float(tpos)-tkk[0])/d)
                            if tpi > 492:
                                tpi = 492
                            tpos = tkk[tpi]+(float(tpos)-(tkk[0]+tpi*d)
                                             )/d*(tkk[tpi+1]-tkk[tpi])
                            tpi = int((float(tkmax)-tkk[0])/d)
                            if tpi > 492:
                                tpi = 492
                            tkmax = tkk[tpi]+(float(tkmax) -
                                              (tkk[0]+tpi*d))/d*(tkk[tpi+1]-tkk[tpi])
                            tpi = int((float(tkmin)-tkk[0])/d)
                            if tpi > 492:
                                tpi = 492
                            tkmin = tkk[tpi]+(float(tkmin) -
                                              (tkk[0]+tpi*d))/d*(tkk[tpi+1]-tkk[tpi])

                            fev = np.append(fev, ev[fi])  # 內容勿動 indent小最內圈if一階
                            t_fwhm.append(tfwhm)  # 內容勿動 indent小最內圈if一階
                            t_pos.append(tpos)  # 內容勿動 indent小最內圈if一階
                            t_kmax.append(tkmax)
                            t_kmin.append(tkmin)
                            if fi not in smfi:
                                smfi.append(fi)
                                skmin.append(tkmin)
                                skmax.append(tkmax)
                            elif fi in smfi:
                                smfp[fi] += 1
                            if tpos > 1000:
                                # 內容勿動 indent小最內圈if一階
                                ff.write(
                                    str(ev[fi])+'\t'+str(tfwhm)+'\t'+str(np.float64(tpos)-1486.6)+'\n')
                            else:
                                ff.write(
                                    str(ev[fi])+'\t'+str(tfwhm)+'\t'+str(np.float64(tpos))+'\n')

                    # pass  # process line i      #勿動
        except UnicodeDecodeError:
            with open(file, encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if line[0:11] in mfpath:
                        fi = int(mfpath.find(line[0:11])/15)
                    if line[0:22] == 'CASA region (*Survey*)':
                        tkmax = line.split(' ')[4]
                        tkmin = line.split(' ')[5]
                    # 若無篩選條件   indent於此if以下
                    elif line[0:20] == 'CASA comp (*Survey*)':
                        tpos = line.split(' ')[17]
                        tfwhm = line.split(' ')[11]
                        area = line.split(' ')[5]
                        # tkmax=line.split(' ')[18]
                        # tkmin=line.split(' ')[19]
                        # 以下if判斷式區段---------可自訂篩選條件------可多層if-----注意indent----------條件篩選值可至 xxxx_fitted_raw_data.txt---檢查需求
                        ##################################################################################################
                        ##################################################################################################
                        # area tfwhm,tpos(1486.6+...)
                        if (ev[fi] > 20.58 and np.float64(tpos) < 1486.6+0.023) or (ev[fi] < 20.58 and np.float64(tpos) > 1486.6+0.023) or 1 == 1:
                            tkk = (2*m*ev[fi]*1.6*10**-19)**0.5 * \
                                np.sin(phi/180*np.pi)*10**-10/(h/2/np.pi)
                            d = tkk[1]-tkk[0]
                            tr = float(tpos)+float(tfwhm)/2
                            tl = float(tpos)-float(tfwhm)/2
                            ri = int((tr-tkk[0])/d)
                            li = int((tl-tkk[0])/d)
                            if ri > 492:
                                ri = 492
                            tr = tkk[ri]+(float(tr)-(tkk[0]+ri*d)
                                          )/d*(tkk[ri+1]-tkk[ri])
                            tl = tkk[li]+(float(tl)-(tkk[0]+li*d)
                                          )/d*(tkk[li+1]-tkk[li])
                            tfwhm = tr-tl
                            tpi = int((float(tpos)-tkk[0])/d)
                            if tpi > 492:
                                tpi = 492
                            tpos = tkk[tpi]+(float(tpos)-(tkk[0]+tpi*d)
                                             )/d*(tkk[tpi+1]-tkk[tpi])
                            tpi = int((float(tkmax)-tkk[0])/d)
                            if tpi > 492:
                                tpi = 492
                            tkmax = tkk[tpi]+(float(tkmax) -
                                              (tkk[0]+tpi*d))/d*(tkk[tpi+1]-tkk[tpi])
                            tpi = int((float(tkmin)-tkk[0])/d)
                            if tpi > 492:
                                tpi = 492
                            tkmin = tkk[tpi]+(float(tkmin) -
                                              (tkk[0]+tpi*d))/d*(tkk[tpi+1]-tkk[tpi])

                            fev = np.append(fev, ev[fi])  # 內容勿動 indent小最內圈if一階
                            t_fwhm.append(tfwhm)  # 內容勿動 indent小最內圈if一階
                            t_pos.append(tpos)  # 內容勿動 indent小最內圈if一階
                            t_kmax.append(tkmax)
                            t_kmin.append(tkmin)
                            if fi not in smfi:
                                smfi.append(fi)
                                skmin.append(tkmin)
                                skmax.append(tkmax)
                            elif fi in smfi:
                                smfp[fi] += 1
                            if tpos > 1000:
                                # 內容勿動 indent小最內圈if一階
                                ff.write(
                                    str(ev[fi])+'\t'+str(tfwhm)+'\t'+str(np.float64(tpos)-1486.6)+'\n')
                            else:
                                ff.write(
                                    str(ev[fi])+'\t'+str(tfwhm)+'\t'+str(np.float64(tpos))+'\n')

                    # pass  # process line i      #勿動
        ff.close()
        fc.close()
        fwhm = np.float64(t_fwhm)     # FWHM
        if np.max(np.float64(t_pos)) > 50:
            rpos = np.float64(t_pos)-1486.6    # Pos
            kmax = np.float64(t_kmax)-1486.6
            kmin = np.float64(t_kmin)-1486.6
            skmax = np.float64(skmax)-1486.6
            skmin = np.float64(skmin)-1486.6
        else:
            rpos = np.float64(t_pos)    # Pos
            kmax = np.float64(t_kmax)
            kmin = np.float64(t_kmin)
            skmax = np.float64(skmax)
            skmin = np.float64(skmin)

        ophi = np.arcsin(rpos/(2*m*fev*1.6*10**-19)**0.5 /
                         10**-10*(h/2/np.pi))*180/np.pi
        pos = (2*m*fev*1.6*10**-19)**0.5 * \
            np.sin((np.float64(k_offset.get())+ophi) /
                   180*np.pi)*10**-10/(h/2/np.pi)
        okmphi = np.arcsin(kmin/(2*m*fev*1.6*10**-19) **
                           0.5/10**-10*(h/2/np.pi))*180/np.pi
        kmin = (2*m*fev*1.6*10**-19)**0.5 * \
            np.sin((np.float64(k_offset.get())+okmphi) /
                   180*np.pi)*10**-10/(h/2/np.pi)
        okMphi = np.arcsin(kmax/(2*m*fev*1.6*10**-19) **
                           0.5/10**-10*(h/2/np.pi))*180/np.pi
        kmax = (2*m*fev*1.6*10**-19)**0.5 * \
            np.sin((np.float64(k_offset.get())+okMphi) /
                   180*np.pi)*10**-10/(h/2/np.pi)

        rpos = re(fev, rpos)
        ophi = re(fev, ophi)
        fwhm = re(fev, fwhm)
        pos = re(fev, pos)
        kmin = re(fev, kmin)
        kmax = re(fev, kmax)
        fev = re(fev, fev)

        smfi = re(smfi, smfi)
        tkmin = re(smfi, skmin)
        tkmax = re(smfi, skmax)
        skmin, skmax = [], []
        smaa1 = np.float64(np.arange(4*len(ev)).reshape(len(ev), 4))
        smaa2 = np.float64(np.arange(8*len(ev)).reshape(len(ev), 8))
        ti = 0
        ti2 = 0
        for i, v in enumerate(ev):
            if i in smfi:
                skmin.append(tkmin[ti2])
                skmax.append(tkmax[ti2])
                ti2 += 1
                if smfp[i] == 2:  # 2peak以上要改
                    ti += 1
            else:
                skmin.append((2*m*v*1.6*10**-19)**0.5 *
                             np.sin(-0.5/180*np.pi)*10**-10/(h/2/np.pi))
                skmax.append((2*m*v*1.6*10**-19)**0.5 *
                             np.sin(0.5/180*np.pi)*10**-10/(h/2/np.pi))
            a1 = [(skmin[i]+skmax[i])/2, 10, 0.5, int(base.get())]
            a2 = [(skmin[i]+skmax[i])/2, 10, 0.5, int(base.get()),
                  (skmin[i]+skmax[i])/2, 10, 0.5, int(base.get())]

            if i in smfi:
                if smfp[i] == 1:
                    a1 = [rpos[ti], 10, fwhm[ti], int(base.get())]
                elif smfp[i] == 2:
                    a2 = [rpos[ti-1], 10, fwhm[ti-1],
                          int(base.get()), rpos[ti], 10, fwhm[ti], int(base.get())]
                ti += 1
            smaa1[i, :] = a1
            smaa2[i, :] = a2

        skmin, skmax = np.float64(skmin), np.float64(skmax)
        fpr = 1
        try:
            smresult=[]
        except:
            pass
        os.chdir(cdir)
    elif ".npz" in file:
        try:
            with np.load(file, 'rb') as f:
                rdd = str(f['path'])
                fev = f['fev']
                fwhm = f['fwhm']
                pos = f['pos']
                skmin = f['skmin']
                skmax = f['skmax']
                smaa1 = f['smaa1']
                smaa2 = f['smaa2']
                smfp = f['smfp']
                smfi = f['smfi']
                smresult = f['smresult']
                print(smresult[166][1])
            rpos = np.copy(pos)
            ophi = np.arcsin(rpos/(2*m*fev*1.6*10**-19) **
                             0.5/10**-10*(h/2/np.pi))*180/np.pi
            fpr = 1
            if '.h5' in rdd:
                data = load_h5(rdd)
                pr_load(data, rdd)
            elif '.json' in rdd:
                data = load_json(rdd)
                pr_load(data, rdd)
            elif '.txt' in rdd:
                data = load_txt(rdd)
                pr_load(data, rdd)
        except:
            pass
    if ".vms" in file:
        np.savez('mfit', ko=k_offset.get(), fev=fev, rpos=rpos, ophi=ophi, fwhm=fwhm, pos=pos, kmin=kmin,
                 kmax=kmax, skmin=skmin, skmax=skmax, smaa1=smaa1, smaa2=smaa2, smfp=smfp, smfi=smfi)
    elif ".npz" in file:
        np.savez('mfit', ko=k_offset.get(), fev=fev, rpos=rpos, ophi=ophi, fwhm=fwhm, pos=pos, kmin=kmin,
                 kmax=kmax, skmin=skmin, skmax=skmax, smaa1=smaa1, smaa2=smaa2, smfp=smfp, smfi=smfi, smresult=smresult)
    limg.config(image=img[np.random.randint(len(img))])
    print('Done')
    st.put('Done')
    lmgg.destroy()


def loadmfit_2p():
    file = fd.askopenfilename(
        title="Select MDC Fitted file", filetypes=(("VMS files", "*.vms"),))
    global st
    global data, rdd, lmgg
    mfpath = ''
    yy = []
    for n in range(len(ev)):
        ecut = data.sel(eV=ev[n], method='nearest')
        y = ecut.to_numpy().reshape(len(phi))
        y = np.where(y > int(lowlim.get()), y, int(lowlim.get()))
        yy.append(y)
        path = 'ecut_%.3f.txt' % ev[n]
        mfpath += path
    if len(file) > 2:
        rdd = file
        print('Loading...')
        st.put('Loading...')
    else:
        rdd = path
        lmgg.destroy()
    if ".vms" in file:
        n = -1
        os.chdir(rdd.removesuffix(rdd.split('/')[-1]))
        fc = open('copy2p_'+file.split('/')[-1], 'w', encoding='utf-8')
        try:
            with open(file) as f:
                f1 = 0
                f2 = 0
                indf = 0
                for i, line in enumerate(f):
                    if line[0:11] in mfpath:
                        fi = int(mfpath.find(line[0:11])/15)
                        n = -1
                        f1 = 0
                        f2 = 0
                        indf = 0
                    if line[0:22] == 'CASA region (*Survey*)':
                        ts = line.split(' ')
                        ts[4] = str(round(float(ts[4]), 6))
                        ts[5] = str(round(float(ts[5]), 6))
                        fc.write(' '.join(ts))
                        fc.write('2'+'\n')
                        n = 1
                    # 若無篩選條件   indent於此if以下
                    elif line[0:20] == 'CASA comp (*Survey*)':
                        s = line.split(' ')
                        s[17] = str(round(float(s[17]), 6))
                        s[18] = str(round(float(s[17])-0.0001, 6))
                        s[19] = str(round(float(s[17])+0.0001, 6))
                        fc.write(' '.join(s))

                        s = line.split(' ')
                        s[2] = '(*Survey_*)'
                        s[8] = str(0)
                        s[9] = str(0.3)  # Area B=A*2
                        s[14] = str(0)
                        s[15] = str(1)  # FWHM B=A*1

                        s[17] = str(round(float(s[17]), 6))
                        s[18] = str(round(float(s[18]), 6))
                        s[17] = str(round(float(s[17])-0.05, 6))
                        s[18] = ts[4]
                        s[19] = ts[5]
                        fc.write(' '.join(s))
                    elif line[0:100] == 'XPS\n':
                        fc.write('XPS'+'\n')
                        indf = 1
                    elif line[0:100] == 'Al\n':
                        fc.write('Al'+'\n')
                        indf = 0
                    elif line[0:100] == '494\n' and f1 == 0:
                        if indf == 1:
                            fc.write('494'+'\n')
                            indf = 0
                        else:
                            f1 = 1
                            ti = i
                    elif line[0:100] == '0\n' and f1 == 1:
                        if ti == i-1:
                            f2 = 1
                            ti = i
                        else:
                            f1 = 0
                            ti = 0
                    elif line[0:100] == '1\n' and f2 == 1:
                        if ti == i-1:
                            ti = 0
                            f2 = 0
                            fc.write('494'+'\n')
                            fc.write('0'+'\n')
                            fc.write('1'+'\n')
                            for j in range(len(phi)):
                                fc.write(str(int(yy[fi][-j-1]))+'\n')
                            n = len(phi)
                        else:
                            f1 = 0
                            f2 = 0
                            ti = 0
                    elif line[0:100] == '8\n':
                        if indf == 1:
                            fc.write('8'+'\n')
                            indf = 0
                        else:
                            fc.write('9'+'\n')
                    elif line[0:100] == '9\n':
                        if indf == 1:
                            fc.write('9'+'\n')
                            indf = 0
                        else:
                            fc.write('10'+'\n')
                    elif line[0:100] == '10\n':
                        if indf == 1:
                            fc.write('10'+'\n')
                            indf = 0
                        else:
                            fc.write('11'+'\n')
                    elif line[0:100] == '11\n':
                        if indf == 1:
                            fc.write('11'+'\n')
                            indf = 0
                        else:
                            fc.write('12'+'\n')
                    else:
                        if n <= 0:
                            if f2 == 1:
                                fc.write('494'+'\n')
                                fc.write('0'+'\n')
                                f2 = 0
                                f1 = 0
                            elif f1 == 1:
                                fc.write('494'+'\n')
                                f1 = 0
                            fc.write(line)
                        if n > 0:
                            f1 = 2
                            n -= 1
                            if n == 0:
                                f1 = 0
                    # pass  # process line i      #勿動
        except UnicodeDecodeError:
            with open(file, encoding='utf-8') as f:
                f1 = 0
                f2 = 0
                indf = 0
                for i, line in enumerate(f):
                    if line[0:11] in mfpath:
                        fi = int(mfpath.find(line[0:11])/15)
                        n = -1
                        f1 = 0
                        f2 = 0
                        indf = 0
                    if line[0:22] == 'CASA region (*Survey*)':
                        ts = line.split(' ')
                        ts[4] = str(round(float(ts[4]), 6))
                        ts[5] = str(round(float(ts[5]), 6))
                        fc.write(' '.join(ts))
                        fc.write('2'+'\n')
                        n = 1
                    # 若無篩選條件   indent於此if以下
                    elif line[0:20] == 'CASA comp (*Survey*)':
                        s = line.split(' ')
                        s[17] = str(round(float(s[17]), 6))
                        s[18] = str(round(float(s[17])-0.0001, 6))
                        s[19] = str(round(float(s[17])+0.0001, 6))
                        fc.write(' '.join(s))

                        s = line.split(' ')
                        s[2] = '(*Survey_*)'
                        s[8] = str(0)
                        s[9] = str(0.3)  # Area B=A*2
                        s[14] = str(0)
                        s[15] = str(1)  # FWHM B=A*1

                        s[17] = str(round(float(s[17]), 6))
                        s[18] = str(round(float(s[18]), 6))
                        s[17] = str(round(float(s[17])-0.05, 6))
                        s[18] = ts[4]
                        s[19] = ts[5]
                        fc.write(' '.join(s))
                    elif line[0:100] == 'XPS\n':
                        fc.write('XPS'+'\n')
                        indf = 1
                    elif line[0:100] == 'Al\n':
                        fc.write('Al'+'\n')
                        indf = 0
                    elif line[0:100] == '494\n' and f1 == 0:
                        if indf == 1:
                            fc.write('494'+'\n')
                            indf = 0
                        else:
                            f1 = 1
                            ti = i
                    elif line[0:100] == '0\n' and f1 == 1:
                        if ti == i-1:
                            f2 = 1
                            ti = i
                        else:
                            f1 = 0
                            ti = 0
                    elif line[0:100] == '1\n' and f2 == 1:
                        if ti == i-1:
                            ti = 0
                            f2 = 0
                            fc.write('494'+'\n')
                            fc.write('0'+'\n')
                            fc.write('1'+'\n')
                            for j in range(len(phi)):
                                fc.write(str(int(yy[fi][-j-1]))+'\n')
                            n = len(phi)
                        else:
                            f1 = 0
                            f2 = 0
                            ti = 0
                    elif line[0:100] == '8\n':
                        if indf == 1:
                            fc.write('8'+'\n')
                            indf = 0
                        else:
                            fc.write('9'+'\n')
                    elif line[0:100] == '9\n':
                        if indf == 1:
                            fc.write('9'+'\n')
                            indf = 0
                        else:
                            fc.write('10'+'\n')
                    elif line[0:100] == '10\n':
                        if indf == 1:
                            fc.write('10'+'\n')
                            indf = 0
                        else:
                            fc.write('11'+'\n')
                    elif line[0:100] == '11\n':
                        if indf == 1:
                            fc.write('11'+'\n')
                            indf = 0
                        else:
                            fc.write('12'+'\n')
                    else:
                        if n <= 0:
                            if f2 == 1:
                                fc.write('494'+'\n')
                                fc.write('0'+'\n')
                                f2 = 0
                                f1 = 0
                            elif f1 == 1:
                                fc.write('494'+'\n')
                                f1 = 0
                            fc.write(line)
                        if n > 0:
                            f1 = 2
                            n -= 1
                            if n == 0:
                                f1 = 0
                    # pass  # process line i      #勿動
        fc.close()
        os.chdir(cdir)
    print('Done')
    st.put('Done')
    lmgg.destroy()


def o_loadefit():
    file = fd.askopenfilename(title="Select EDC Fitted file", filetypes=(
        ("NPZ files", "*.npz"), ("VMS files", "*.vms"),))
    global h, m, efwhm, ffphi, fphi, epos, fk, limg, img, name, st, emin, emax
    global data, rdd, semin, semax, seaa1, seaa2, sefp, sefi, fpr, efi_x
    if len(file) > 2:
        fpr = 0
        rdd = file
        print('Loading...')
        st.put('Loading...')
    else:
        rdd = path
    if ".vms" in file:
        fphi = np.array([], dtype=float)
        efi_x = np.arange(len(phi))
        t_fwhm = []
        t_pos = []
        t_emax = []
        t_emin = []
        sefi = []
        semin = []
        semax = []
        sefp = [1 for i in range(len(phi))]
        tphi = []
        os.chdir(rdd.removesuffix(rdd.split('/')[-1]))
        ff = open(name+'_edc_fitted_raw_data.txt', 'w',
                  encoding='utf-8')  # tab 必須使用 '\t' 不可"\t"
        ff.write('Angle (deg)'+'\t'+'FWHM (eV)'+'\t'+'Position (eV)'+'\n')
        with open(file) as f:
            for i, line in enumerate(f):
                if line[0:16] in efpath:
                    if '-' in line[0:16]:
                        fi = int(efpath.find(line[0:16])/17)
                    else:
                        fi = int(
                            len(phi)//2+(efpath.find(line[0:16])-17*len(phi)//2)/16)
                if line[0:22] == 'CASA region (*Survey*)':
                    temax = line.split(' ')[4]
                    temin = line.split(' ')[5]
                # 若無篩選條件   indent於此if以下
                if line[0:20] == 'CASA comp (*Survey*)':
                    tpos = line.split(' ')[17]
                    tfwhm = line.split(' ')[11]
                    area = line.split(' ')[5]
                    # temax=line.split(' ')[18]
                    # temin=line.split(' ')[19]

                    # 以下if判斷式區段---------可自訂篩選條件------可多層if-----注意indent----------條件篩選值可至 xxxx_fitted_raw_data.txt---檢查需求
                    ##################################################################################################
                    ##################################################################################################
                    # area tfwhm,tpos(1486.6+...)
                    if np.float64(area) > 0 and np.float64(tfwhm) < 3:
                        if (phi[fi] > 20.58 and np.float64(tpos) < 1486.6+0.023) or (phi[fi] < 20.58 and np.float64(tpos) > 1486.6+0.023) or 1 == 1:

                            # 內容勿動 indent小最內圈if一階
                            fphi = np.append(fphi, phi[fi])
                            t_fwhm.append(tfwhm)  # 內容勿動 indent小最內圈if一階
                            t_pos.append(tpos)  # 內容勿動 indent小最內圈if一階
                            t_emax.append(temax)
                            t_emin.append(temin)
                            if fi not in sefi:
                                tphi.append(phi[fi])
                                sefi.append(fi)
                                semin.append(temin)
                                semax.append(temax)
                            elif fi in sefi:
                                sefp[fi] += 1
                            if float(tpos) > 1000:
                                # 內容勿動 indent小最內圈if一階
                                ff.write(
                                    str(phi[fi])+'\t'+tfwhm+'\t'+str(np.float64(tpos)-1486.6)+'\n')
                            else:
                                ff.write(str(phi[fi])+'\t'+tfwhm +
                                         '\t'+str(-np.float64(tpos))+'\n')
                    ##################################################################################################
                    ##################################################################################################
                pass  # process line i      #勿動
        ff.close()
        efwhm = np.float64(t_fwhm)     # FWHM
        if np.max(np.float64(t_pos)) > 50:
            epos = np.float64(t_pos)-1486.6    # Pos
            emax = np.float64(t_emax)-1486.6
            emin = np.float64(t_emin)-1486.6
            semax = np.float64(semax)-1486.6
            semin = np.float64(semin)-1486.6
        else:
            epos = np.float64(t_pos)    # Pos
            emax = np.float64(t_emax)
            emin = np.float64(t_emin)
            semax = np.float64(semax)
            semin = np.float64(semin)
        ffphi = np.float64(k_offset.get())+fphi
        fk = (2*m*epos*1.6*10**-19)**0.5 * \
            np.sin(ffphi/180*np.pi)*10**-10/(h/2/np.pi)

        epos = re(fphi, epos)
        ffphi = re(fphi, ffphi)
        efwhm = re(fphi, efwhm)
        fk = re(fphi, fk)
        emin = re(fphi, emin)
        emax = re(fphi, emax)
        fphi = re(fphi, fphi)

        sefi = re(tphi, sefi)
        temin = re(tphi, semin)
        temax = re(tphi, semax)
        semin, semax = [], []
        seaa1 = np.float64(np.arange(4*len(phi)).reshape(len(phi), 4))
        seaa2 = np.float64(np.arange(8*len(phi)).reshape(len(phi), 8))
        ti = 0
        ti2 = 0
        for i in range(len(phi)):
            if i in sefi:
                semin.append(temin[ti2])
                semax.append(temax[ti2])
                ti2 += 1
                if sefp[i] == 2:  # 2peak以上要改
                    ti += 1
            else:
                semin.append(np.min(ev))
                semax.append(np.max(ev))
            a1 = [(semin[i]+semax[i])/2, 10, 5, int(base.get())]
            a2 = [(semin[i]+semax[i])/2, 10, 5, int(base.get()),
                  (semin[i]+semax[i])/2, 10, 0.5, int(base.get())]
            if i in sefi:
                if sefp[i] == 1:
                    a1 = [epos[ti], 10, efwhm[ti], int(base.get())]
                elif sefp[i] == 2:
                    a2 = [epos[ti-1], 10, efwhm[ti-1],
                          int(base.get()), epos[ti], 10, efwhm[ti], int(base.get())]
                ti += 1
            seaa1[i, :] = a1
            seaa2[i, :] = a2
        semin, semax = np.float64(semin), np.float64(semax)
        fpr = 1
        os.chdir(cdir)
    elif ".npz" in file:
        try:
            with np.load(file, 'rb') as f:
                rdd = str(f['path'])
                fphi = f['fphi']
                efwhm = f['efwhm']
                epos = f['epos']
                semin = f['semin']
                semax = f['semax']
                seaa1 = f['seaa1']
                seaa2 = f['seaa2']
                sefp = f['sefp']
                sefi = f['sefi']
            ffphi = np.float64(k_offset.get())+fphi
            fk = (2*m*epos*1.6*10**-19)**0.5 * \
                np.sin(ffphi/180*np.pi)*10**-10/(h/2/np.pi)
            fpr = 1
            if '.h5' in rdd:
                data = load_h5(rdd)
                pr_load(data, rdd)
            elif '.json' in rdd:
                data = load_json(rdd)
                pr_load(data, rdd)
            elif '.txt' in rdd:
                data = load_txt(rdd)
                pr_load(data, rdd)
        except:
            pass
    if ".vms" in file or ".npz" in file:
        np.savez('efit', ko=k_offset.get(), fphi=fphi, epos=epos, ffphi=ffphi, efwhm=efwhm, fk=fk,
                 emin=emin, emax=emax, semin=semin, semax=semax, seaa1=seaa1, seaa2=seaa2, sefp=sefp, sefi=sefi)
    limg.config(image=img[np.random.randint(len(img))])
    print('Done')
    st.put('Done')


##################################################################
##################################################################
##################################################################
##################################################################
##################################################################
'''
def gl(x,x0,a,w,y0):
    v=a/(1+(x-x0)**2/(1/2*w)**2)+y0
    return v

def o_efitplot(*e):
    global ef,st
    if len(evv)==0:
        ef=1
        egg.destroy()
    else:
        ef=0
        i=efiti.get()
        efitfig.clear()
        fitax=efitfig.subplots()
        fitax.set_title('Pos:'+str(round(eaa[i,0],3))+' (eV)'+', FWHM:'+str(round(eaa[i,2],3))+' (eV)'+', Deg:'+str(round(evv[i],3))+r' $^{\circ}$')
        fitax.scatter(fex[i,:],fey[i,:],c='k',s=4)
        fitax.plot(fexx[i,np.argwhere(fexx[i,:]>=-20)],gl(fexx[i,np.argwhere(fexx[i,:]>=-20)],eaa[i,0],eaa[i,1],eaa[i,2],eaa[i,3]),'b-',lw=2)
        fitax.scatter(fexx[i,np.argwhere(fexx[i,:]>=-20)],feyy[i,np.argwhere(feyy[i,:]>=-20)],c='g',s=4)
        fitax.axvline(min(fexx[i,np.argwhere(fexx[i,:]>=-20)]),c='r')
        fitax.axvline(max(fexx[i,np.argwhere(fexx[i,:]>=-20)]),c='r')
        fitax.set_xlabel('Kinetic Energy (eV)',fontsize=14)
        fitax.set_ylabel('Intensity (Counts)',fontsize=14)
        efitout.draw()

def o_mfitplot(*e):
    global mf,st
    if len(mvv)==0:
        mf=1
        mgg.destroy()
    else:
        mf=0
        i=mfiti.get()
        mfitfig.clear()
        fitax=mfitfig.subplots()
        fitax.set_title('Pos:'+str(round(maa[i,0],3))+r' $(\frac{2\pi}{\AA})$'+', FWHM:'+str(round(maa[i,2],3))+r' $(\frac{2\pi}{\AA})$'+', Kinetic Energy:'+str(round(mvv[i],3))+' eV')
        fitax.scatter(fmx[i,:],fmy[i,:],c='k',s=4)
        fitax.plot(fmxx[i,np.argwhere(fmxx[i,:]>=-20)],gl(fmxx[i,np.argwhere(fmxx[i,:]>=-20)],maa[i,0],maa[i,1],maa[i,2],maa[i,3]),'b-',lw=2)
        fitax.scatter(fmxx[i,np.argwhere(fmxx[i,:]>=-20)],y,c='g',s=4)
        fitax.axvline(min(fmxx[i,np.argwhere(fmxx[i,:]>=-20)]),c='r')
        fitax.axvline(max(fmxx[i,np.argwhere(fmxx[i,:]>=-20)]),c='r')
        fitax.set_xlabel(r'k ($\frac{2\pi}{\AA}$)',fontsize=14)
        fitax.set_ylabel('Intensity (Counts)',fontsize=14)
        mfitout.draw()
    
def o_ejob():
    global g,efiti,efitfig,efitout,egg
    egg=tk.Toplevel(g)
    egg.title('EDC Lorentz Fit')
    fr=tk.Frame(master=egg,bd=5)
    fr.grid(row=0,column=0)
    efitfig = Figure(figsize=(8,6),layout='constrained')
    efitout = tkagg.FigureCanvasTkAgg(efitfig, master=fr)
    efitout.get_tk_widget().grid(row=0,column=0)
    # bstop=tk.Button(gg,command=stop,text='Stop',font=('Arial',20),bd=10)
    # bstop.grid(row=1,column=0)
    efiti=tk.IntVar()
    efiti.set(0)
    efiti.trace_add('write',o_efitplot)
    sc=tk.Frame(master=egg,bd=5)
    sc.grid(row=1,column=0)
    chi=tk.Scale(sc,from_=0,to=np.size(eaa,0)-1,orient='horizontal',variable=efiti,state='active',bg='white',fg='black',length=400,width=50,resolution=1)
    chi.pack()
    o_efitplot()
    egg.update()

def o_mjob():
    global g,mfiti,mfitfig,mfitout,mgg
    mgg=tk.Toplevel(g)
    mgg.title('MDC Lorentz Fit')
    fr=tk.Frame(master=mgg,bd=5)
    fr.grid(row=0,column=0)
    mfitfig = Figure(figsize=(8,6),layout='constrained')
    mfitout = tkagg.FigureCanvasTkAgg(mfitfig, master=fr)
    mfitout.get_tk_widget().grid(row=0,column=0)
    # bstop=tk.Button(gg,command=stop,text='Stop',font=('Arial',20),bd=10)
    # bstop.grid(row=1,column=0)
    mfiti=tk.IntVar()
    mfiti.set(0)
    mfiti.trace_add('write',o_mfitplot)
    sc=tk.Frame(master=mgg,bd=5)
    sc.grid(row=1,column=0)
    chi=tk.Scale(sc,from_=0,to=np.size(maa,0)-1,orient='horizontal',variable=mfiti,state='active',bg='white',fg='black',length=400,width=50,resolution=1)
    chi.pack()
    o_mfitplot()
    mgg.update()

def o_fitm():
    global mf,pos,fwhm,base,k_offset,st,mvv,maa,fmxx,fmyy,fmx,fmy
    try:
        fmxx=np.float64(np.arange(len(phi)*len(fev)).reshape(len(fev),len(phi)))
        fmyy=np.float64(np.arange(len(phi)*len(fev)).reshape(len(fev),len(phi)))
        fmxx=fmxx/fmxx*-50
        fmyy=fmyy/fmyy*-50
        fmx=np.float64(np.arange(len(phi)*len(fev)).reshape(len(fev),len(phi)))
        fmy=np.float64(np.arange(len(phi)*len(fev)).reshape(len(fev),len(phi)))
        mvv=np.float64(np.arange(len(fev)))
        maa=np.float64(np.arange(4*len(fev)).reshape(len(fev),4))
        for i,v in enumerate(fev):
            ecut=data.sel(eV=v,method='nearest')
            x = (2*m*v*1.6*10**-19)**0.5*np.sin((phi+np.float64(k_offset.get()))/180*np.pi)*10**-10/(h/2/np.pi)
            y=ecut.to_numpy().reshape(len(x))
            tx=x[np.argwhere(x>=kmin[i])].flatten()
            xx=tx[np.argwhere(tx<=kmax[i])].flatten()
            ty=y[np.argwhere(x>=kmin[i])].flatten()
            yy=ty[np.argwhere(tx<=kmax[i])].flatten()
            yy=np.where(yy>int(base.get()),yy,int(base.get()))
            a,b=curve_fit(gl,xx,yy,bounds=([pos[i]-0.01,(np.max(y)-int(base.get()))/10,0,int(base.get())-1],[pos[i]+0.01,np.max(y)-int(base.get()),0.3,int(base.get())+1]))
            pos[i]=a[0]
            fwhm[i]=a[2]
            print('MDC '+str(round((i+1)/len(fev)*100))+'%'+' ('+str(len(fev))+')')
            st.put('MDC '+str(round((i+1)/len(fev)*100))+'%'+' ('+str(len(fev))+')')
            
            fmxx[i,0:len(xx)]=xx
            fmyy[i,0:len(yy)]=yy
            fmx[i,:]=x
            fmy[i,:]=y
            mvv[i]=v
            maa[i,:]=a
            
        tt1=threading.Thread(target=o_mjob)
        tt1.daemon=True
        tt1.start()
    except:
        mf=1
        pass
def o_fite():
    global ef,pos,fwhm,base,k_offset,st,evv,eaa,fexx,feyy,fex,fey
    try:
        fexx=np.float64(np.arange(len(ev)*len(fphi)).reshape(len(fphi),len(ev)))
        feyy=np.float64(np.arange(len(ev)*len(fphi)).reshape(len(fphi),len(ev)))
        fexx=fexx/fexx*-50
        feyy=feyy/feyy*-50
        fex=np.float64(np.arange(len(ev)*len(fphi)).reshape(len(fphi),len(ev)))
        fey=np.float64(np.arange(len(ev)*len(fphi)).reshape(len(fphi),len(ev)))
        evv=np.float64(np.arange(len(fphi)))
        eaa=np.float64(np.arange(4*len(fphi)).reshape(len(fphi),4))
        for i,v in enumerate(fphi):
            angcut=data.sel(phi=v,method='nearest')
            x = ev
            y=angcut.to_numpy().reshape(len(x))
            tx=x[np.argwhere(x>=emin[i])].flatten()
            xx=tx[np.argwhere(tx<=emax[i])].flatten()
            ty=y[np.argwhere(x>=emin[i])].flatten()
            yy=ty[np.argwhere(tx<=emax[i])].flatten()
            yy=np.where(yy>int(base.get()),yy,int(base.get()))
            
            a,b=curve_fit(gl,xx,yy,bounds=([epos[i]-0.1,(np.max(y)-int(base.get()))/10,0,int(base.get())-1],[epos[i]+0.1,np.max(y)-int(base.get()),3,int(base.get())+1]))
            epos[i]=a[0]
            efwhm[i]=a[2]
            print('EDC '+str(round((i+1)/len(fphi)*100))+'%'+' ('+str(len(fphi))+')')
            st.put('EDC '+str(round((i+1)/len(fphi)*100))+'%'+' ('+str(len(fphi))+')')
            
            fexx[i,0:len(xx)]=xx
            feyy[i,0:len(yy)]=yy
            fex[i,:]=x
            fey[i,:]=y
            evv[i]=v
            eaa[i,:]=a
            
        tt2=threading.Thread(target=o_ejob)
        tt2.daemon=True
        tt2.start()
    except:
        ef=1
'''
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################


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

wr1 = 0
wr2 = 0
fa1 = 0
fa2 = 0

#######################################################
#######################################################
#######################################################
#######################################################
# User function to calculate residuals
def pr_fitgl2(params, x, data):
    fitter0 = Minimizer(fgl2, params, fcn_args=(x, data))
    result = fitter0.minimize()
    pars = Parameters()
    pars.add('y1', value=0, vary=False)
    pars.add('y2', value=0, vary=False)
    pars.add('x1', value=result.params['x1'].value, vary=False)
    pars.add('x2', value=result.params['x2'].value, vary=False)
    pars.add('h1', value=result.params['h1'].value)
    pars.add('h2', value=result.params['h2'].value)
    pars.add('w1', value=result.params['w1'].value,
             min=result.params['w1'].min, max=result.params['w1'].max)
    pars.add('w2', value=result.params['w2'].value,
             min=result.params['w2'].min, max=result.params['w2'].max)
    return pars['h1'], pars['h2'], pars['w1'], pars['w2'], pars['x1'], pars['x2'], pars['y1'], pars['y2']


def fgl2_1(params, x, data):
    par = params
    h1, h2, w1, w2, x1, x2, y1, y2 = pr_fitgl2(par, x, data)
    area1 = np.sum(gl1(x, x1, h1, w1, y1))
    area2 = np.sum(gl1(x, x2, h2, w2, y2))
    return area2/fa2 - area1/fa1


def fgl2_a(params, x, data):
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
    area1 = np.sum(gl1(x, x1, h1, w1, y1))
    area2 = np.sum(gl1(x, x2, h2, w2, y2))
    return model - data + area1/fa1 - area2/fa2


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


def toa1():
    a1 = []
    a1.append(result.params['x'].value)
    a1.append(result.params['h'].value)
    a1.append(result.params['w'].value)
    a1.append(result.params['y'].value)
    return a1


def toa2():
    a2 = []
    a2.append(result.params['x1'].value)
    a2.append(result.params['h1'].value)
    a2.append(result.params['w1'].value)
    a2.append(result.params['y1'].value)
    a2.append(result.params['x2'].value)
    a2.append(result.params['h2'].value)
    a2.append(result.params['w2'].value)
    a2.append(result.params['y2'].value)
    return a2


fit_warn = 0


def checkfit():
    global fit_warn
    fit_warn = 0
    t = 0
    for i in result.params:
        par = result.params[i]
        if par.value != 0:
            try:
                if par.stderr/par.value > 0.2:  # uncertainty 20%
                    t += 1
            except TypeError:
                fit_warn = 1
    if t >= 2:
        fit_warn = 1


def putfitpar(inpars, modelpars=None, show_correl=True, min_correl=0.1,
              sort_pars=False, correl_mode='list'):
    from lmfit.parameter import Parameters
    if isinstance(inpars, Parameters):
        result, params = None, inpars
    if hasattr(inpars, 'params'):
        result = inpars
        params = inpars.params

    if sort_pars:
        if callable(sort_pars):
            key = sort_pars
        else:
            key = alphanumeric_sort
        parnames = sorted(params, key=key)
    else:
        # dict.keys() returns a KeysView in py3, and they're indexed
        # further down
        parnames = list(params.keys())

    buff = []
    add = buff.append
    namelen = max(len(n) for n in parnames)
    for name in parnames:
        if name != 'y1' and name != 'y2':
            par = params[name]
            space = ' '*(namelen-len(name))
            nout = f"{name}:{space}"
            inval = '(init = ?)'
            if par.init_value is not None:
                inval = f'(init = {par.init_value:.5g})'
            if modelpars is not None and name in modelpars:
                inval = f'{inval}, model_value = {modelpars[name].value:.5g}'
            try:
                sval = gformat(par.value)
            except (TypeError, ValueError):
                sval = ' Non Numeric Value?'
            if par.stderr is not None:
                serr = gformat(par.stderr)
                try:
                    spercent = f'({abs(par.stderr/par.value):.2%})'
                except ZeroDivisionError:
                    spercent = ''
                sval = f'{sval} +/-{serr} {spercent}'

            if par.vary:
                add(f"    {nout} {sval} {inval}")
            elif par.expr is not None:
                add(f"    {nout} {sval} == '{par.expr}'")
            else:
                add(f"    {nout} {par.value: .5g} (fixed)")
    return buff


def fitpar1(result, lm1, lm2, lm3, lm4, lm5, lm6):
    s = putfitpar(result)
    x = s[0]
    h = s[1]
    w = s[2]
    for l, v in zip([lm1, lm2, lm3, lm4, lm5, lm6], [x, h, w, '', '', '']):
        l.config(text=v)
        l.config(anchor='w')


def fitpar2(result, lm1, lm2, lm3, lm4, lm5, lm6):
    s = putfitpar(result)
    for i in s:
        '''preprocess the string to put values in the labels'''
        if 'x1*xr1+xr2' in i:
            if xr2>=0:
                i = i.replace(' == \'x1*xr1+xr2\'', '='+str(xr1)+'*x1+'+str(xr2))
            else:
                i = i.replace(' == \'x1*xr1+xr2\'', '='+str(xr1)+'*x1-'+str(-xr2))
        if "(x2-xr2) / xr1" in i:
            if xr2>=0:
                i = i.replace(' == \'(x2-xr2) / xr1\'','=(x2-'+str(xr2) + ')/'+str(xr1))
            else:
                i = i.replace(' == \'(x2-xr2) / xr1\'','=(x2+'+str(-xr2) + ')/'+str(xr1))
        if 'w1/wr1*wr2' in i:
            i = i.replace(' == \'w1/wr1*wr2\'', '=w1/'+str(wr1)+'*'+str(wr2))
            
        '''assign the values to the labels'''
        if 'x1:' in i:
            x1 = i
        if 'x2:' in i:
            x2 = i
        if 'h1:' in i:
            h1 = i
        if 'h2:' in i:
            h2 = i
        if 'w1:' in i:
            w1 = i
        if 'w2:' in i:
            w2 = i
    for l, v in zip([lm1, lm2, lm3, lm4, lm5, lm6], [x1, x2, h1, h2, w1, w2]):
        l.config(text=v)
        l.config(anchor='w')
# Define the parameters
# pars = Parameters()
# wr1,wr2=int(wf1.get()),int(wf2.get())
# fa1,fa2=int(af1.get()),int(af2.get())
# pars.add('wr1',value=wr1)
# pars.add('wr2',value=wr2)
# pars.add('x1',value=0,min=-5,max=5)
# pars.add('x2',value=0,min=-5,max=5)
# pars.add('h1',value=0.3,min=0.2,max=1)
# pars.add('h2',value=0.3,min=0.2,max=1)
# pars.add('w1', value=0.5, min=0, max=1)
# if wr1!=0 and wr2!=0:
#     pars.add('w2', expr='w1/wr1*wr2')
# else:
#     pars.add('w2', value=0.5, min=0, max=1)
# pars.add('y1',value=0,min=-5,max=5)
# pars.add('y2',value=0,min=-5,max=5)

# Create Minimizer object and fit the data
# fitter = Minimizer(fgl2, pars, fcn_args=(x, data))
# result = fitter.minimize()

# print(result.params['h1'].value)
# Print fitting results
# report_fit(result)

#######################################################
#######################################################
#######################################################
#######################################################


def lnr_bg(x: np.ndarray, n_samples=5) -> np.ndarray:
    while len(x) < 2*n_samples:
        if len(x) < 2:
            o = np.array([])
        n_samples -= 1
    left, right = np.mean(x[:n_samples]), np.mean(x[-n_samples:])
    o = np.ones(len(x))*np.mean([left, right])
    return o


def shirley_bg(
        xps: np.ndarray, eps=1e-7, max_iters=50, n_samples=5) -> np.ndarray:
    global bg_warn
    """Core routine for calculating a Shirley background on np.ndarray data."""
    background = np.copy(xps)
    cumulative_xps = np.cumsum(xps, axis=0)
    total_xps = np.sum(xps, axis=0)

    rel_error = np.inf

    i_left = np.mean(xps[:n_samples], axis=0)
    i_right = np.mean(xps[-n_samples:], axis=0)

    iter_count = 0

    k = i_left - i_right
    for iter_count in range(max_iters):
        cumulative_background = np.cumsum(background, axis=0)
        total_background = np.sum(background, axis=0)

        new_bkg = np.copy(background)

        for i in range(len(new_bkg)):
            new_bkg[i] = i_right + k * (
                (total_xps - cumulative_xps[i] -
                 (total_background - cumulative_background[i]))
                / (total_xps - total_background + 1e-5)
            )

        rel_error = np.abs(np.sum(new_bkg, axis=0) -
                           total_background) / (total_background)

        background = new_bkg

        if np.any(rel_error < eps):
            break

    if (iter_count + 1) == max_iters:
        bg_warn = 1
        warnings.warn(
            "Shirley background calculation did not converge "
            + "after {} steps with relative error {}!".format(max_iters, rel_error)
        )
    else:
        bg_warn = 0

    return background


def fecgl2():
    global ebcgl2, emin, emax, flecgl2, eicgl2, efp
    i = efiti.get()
    flecgl2 *= -1
    if flecgl2 == 1:
        eicgl2 = i
        ebcgl2.config(text='End Add 2 Peaks', bg='red')
    else:
        ti = sorted([i, eicgl2])
        for i in np.linspace(ti[0], ti[1], ti[1]-ti[0]+1, dtype=int):
            efp[i] = 2
            if i not in efi_x:
                efi_x.append(i)
            if i in efi:
                efi.remove(i)
            if i in efi_err:
                efi_err.remove(i)
        ebcgl2.config(text='Start Add 2 Peaks', bg='white')
        efitplot()


def efitjob():
    global fexx, feyy, fex, fey, evv, eaa1, eaa2, emin, emax, efi, efi_err, efi_x, st, est, result, fa1, fa2, fit_warn, bg_warn
    if len(efi) < 1:
        efi, efi_err, efi_x = [], [], []
    else:
        efi, efi_err, efi_x = list(efi), list(efi_err), list(efi_x)
    pbar = tqdm.tqdm(total=len(phi), desc='Fitting EDC', colour='blue')
    for i in range(len(phi)):
        ebase[i] = int(base.get())  # 待調整
        fexx[i, :] = fexx[i, :]/fexx[i, :]*-50
        feyy[i, :] = feyy[i, :]/feyy[i, :]*-50
        angcut = data.sel(phi=phi[i], method='nearest')
        x = ev
        y = angcut.to_numpy().reshape(len(x))
        tx = x[np.argwhere(x >= emin[i])].flatten()
        xx = tx[np.argwhere(tx <= emax[i])].flatten()
        ty = y[np.argwhere(x >= emin[i])].flatten()
        yy = ty[np.argwhere(tx <= emax[i])].flatten()
        yy = np.where(yy > ebase[i], yy, ebase[i])
        try:
            # if (emin[i],emax[i])==(np.min(ev),np.max(ev)) and i not in efi:
            if i not in efi:
                if i not in efi_x:
                    efi_x.append(i)
                if i in efi:
                    efi.remove(i)
                if i in efi_err:
                    efi_err.remove(i)
                a1 = [(emin[i]+emax[i])/2, (np.max(y)-ebase[i]), 5, ebase[i]]
                a2 = [(emin[i]+emax[i])/2, (np.max(y)-ebase[i]), 5, ebase[i],
                      (emin[i]+emax[i])/2, (np.max(y)-ebase[i]), 5, ebase[i]]
            # elif (emin[i],emax[i])!=(np.min(ev),np.max(ev)):
            else:
                if efp[i] == 1:
                    if i in efi and i not in efi_err:
                        a1 = eaa1[i, :]
                    else:
                        a1, b = curve_fit(gl1, xx, yy-shirley_bg(yy), bounds=(
                            [emin[i], (np.max(y)-ebase[i])/10, 0, 0], [emax[i], np.max(y)-ebase[i]+1, 3, 0.01]))
                        fit_warn = 0
                elif efp[i] == 2:
                    if i in efi and i not in efi_err:
                        a2 = eaa1[i, :]
                    else:
                        pars = Parameters()
                        wr1, wr2 = int(ewf1.get()), int(ewf2.get())
                        fa1, fa2 = int(eaf1.get()), int(eaf2.get())
                        pars.add(
                            'x1', value=emin[i]+(emax[i]-emin[i])*0.3, min=emin[i], max=emax[i])
                        pars.add(
                            'x2', value=emax[i]-(emax[i]-emin[i])*0.3, min=emin[i], max=emax[i])
                        pars.add('h1', value=(
                            np.max(y)-ebase[i])+1, min=(np.max(y)-ebase[i])/10, max=np.max(y)-ebase[i]+1)
                        pars.add('h2', value=(
                            np.max(y)-ebase[i])+1, min=(np.max(y)-ebase[i])/10, max=np.max(y)-ebase[i]+1)
                        pars.add('w1', value=1, min=0, max=3)
                        if wr1 != 0 and wr2 != 0:
                            pars.add('wr1', value=wr1, vary=False)
                            pars.add('wr2', value=wr2, vary=False)
                            pars.add('w2', expr='w1/wr1*wr2')
                        else:
                            pars.add('w2', value=1, min=0, max=3)
                        pars.add('y1', value=0, vary=False)
                        pars.add('y2', value=0, vary=False)
                        fitter = Minimizer(
                            fgl2, pars, fcn_args=(xx, yy-shirley_bg(yy)))
                        result = fitter.minimize()
                        a2 = toa2()
                        checkfit()
                        if fit_warn == 1:
                            t = 5
                            while t > 0 and fit_warn == 1:
                                result = fitter.minimize()
                                a2 = toa2()
                                checkfit()
                                t -= 1
                        # p0=[emin[i]+(emax[i]-emin[i])*0.3,(np.max(y)-ebase[i])+1,1,0,emax[i]-(emax[i]-emin[i])*0.3,(np.max(y)-ebase[i])+1,1,0]
                        # a2,b=curve_fit(gl2,xx,yy-shirley_bg(yy),p0=p0,bounds=([emin[i],(np.max(y)-ebase[i])/10,0,0,emin[i],(np.max(y)-ebase[i])/10,0,0],[emax[i],np.max(y)-ebase[i]+1,3,0.01,emax[i],np.max(y)-ebase[i]+1,3,0.01]))

                if bg_warn == 0 and fit_warn == 0:  # shirley base line warn
                    if i not in efi:
                        efi.append(i)
                    if i in efi_x:
                        efi_x.remove(i)
                    if i in efi_err:
                        efi_err.remove(i)
                else:
                    if i not in efi_err:
                        efi_err.append(i)
                    if i in efi_x:
                        efi_x.remove(i)
                    if i in efi:
                        efi.remove(i)
        except RuntimeError:
            if i not in efi_err:
                efi_err.append(i)
            if i in efi_x:
                efi_x.remove(i)
            if i in efi:
                efi.remove(i)
            a1 = [(emin[i]+emax[i])/2, (np.max(y)-ebase[i]), 5, ebase[i]]
            a2 = [(emin[i]+emax[i])/2, (np.max(y)-ebase[i]), 5, ebase[i],
                  (emin[i]+emax[i])/2, (np.max(y)-ebase[i]), 5, ebase[i]]
        # epos[i]=a[0]
        # efwhm[i]=a[2]

        fexx[i, :len(xx)] = xx
        feyy[i, :len(yy)] = yy
        fex[i, :] = x
        fey[i, :] = y
        evv[i] = phi[i]
        if efp[i] == 1:
            eaa1[i, :] = a1
        elif efp[i] == 2:
            eaa2[i, :] = a2
        pbar.update(1)
        # print('Fitting EDC '+str(round((i+1)/len(phi)*100))+'%'+' ('+str(len(phi))+')')
        st.put('Fitting EDC '+str(round((i+1)/len(phi)*100)) +
               '%'+' ('+str(len(phi))+')')
        est.put('Fitting EDC '+str(round((i+1)/len(phi)*100)) +
                '%'+' ('+str(len(phi))+')')
    pbar.close()
    efitplot()


def efit():
    global fexx, feyy, fex, fey, evv, eaa1, eaa2, emin, emax, efi, efi_err, efi_x, result, fa1, fa2, fit_warn, bg_warn
    efi, efi_err, efi_x = list(efi), list(efi_err), list(efi_x)
    i = efiti.get()
    ebase[i] = int(base.get())  # 待調整
    fexx[i, :] = fexx[i, :]/fexx[i, :]*-50
    feyy[i, :] = feyy[i, :]/feyy[i, :]*-50
    angcut = data.sel(phi=phi[i], method='nearest')
    x = ev
    y = angcut.to_numpy().reshape(len(x))
    tx = x[np.argwhere(x >= emin[i])].flatten()
    xx = tx[np.argwhere(tx <= emax[i])].flatten()
    ty = y[np.argwhere(x >= emin[i])].flatten()
    yy = ty[np.argwhere(tx <= emax[i])].flatten()
    yy = np.where(yy > ebase[i], yy, ebase[i])
    try:
        if efp[i] == 1:
            a1, b = curve_fit(gl1, xx, yy-shirley_bg(yy), bounds=(
                [emin[i], (np.max(y)-ebase[i])/10, 0, 0], [emax[i], np.max(y)-ebase[i]+1, 3, 0.01]))
            fit_warn = 0
        elif efp[i] == 2:
            pars = Parameters()
            wr1, wr2 = int(ewf1.get()), int(ewf2.get())
            fa1, fa2 = int(eaf1.get()), int(eaf2.get())
            pars.add('x1', value=emin[i]+(emax[i]-emin[i])
                     * 0.4, min=emin[i], max=emax[i])
            pars.add('x2', value=emax[i]-(emax[i]-emin[i])
                     * 0.4, min=emin[i], max=emax[i])
            pars.add('h1', value=(
                np.max(y)-ebase[i])+1, min=(np.max(y)-ebase[i])/10, max=np.max(y)-ebase[i]+1)
            pars.add('h2', value=(
                np.max(y)-ebase[i])+1, min=(np.max(y)-ebase[i])/10, max=np.max(y)-ebase[i]+1)
            pars.add('w1', value=1, min=0, max=3)
            if wr1 != 0 and wr2 != 0:
                pars.add('wr1', value=wr1, vary=False)
                pars.add('wr2', value=wr2, vary=False)
                pars.add('w2', expr='w1/wr1*wr2')
            else:
                pars.add('w2', value=1, min=0, max=3)
            pars.add('y1', value=0, vary=False)
            pars.add('y2', value=0, vary=False)
            fitter = Minimizer(fgl2, pars, fcn_args=(xx, yy-shirley_bg(yy)))
            result = fitter.minimize()
            a2 = toa2()
            checkfit()
            if fit_warn == 1:
                t = 5
                while t > 0 and fit_warn == 1:
                    result = fitter.minimize()
                    a2 = toa2()
                    checkfit()
                    t -= 1
            report_fit(result)
            # p0=[emin[i]+(emax[i]-emin[i])*0.3,(np.max(y)-ebase[i])+1,1,0,emax[i]-(emax[i]-emin[i])*0.3,(np.max(y)-ebase[i])+1,1,0]
            # a2,b=curve_fit(gl2,xx,yy-shirley_bg(yy),p0=p0,bounds=([emin[i],(np.max(y)-ebase[i])/10,0,0,emin[i],(np.max(y)-ebase[i])/10,0,0],[emax[i],np.max(y)-ebase[i]+1,3,0.01,emax[i],np.max(y)-ebase[i]+1,3,0.01]))

        if (emin[i], emax[i]) == (np.min(ev), np.max(ev)):
            if i not in efi_x:
                efi_x.append(i)
            if i in efi:
                efi.remove(i)
            if i in efi_err:
                efi_err.remove(i)
        elif (emin[i], emax[i]) != (np.min(ev), np.max(ev)):
            if bg_warn == 0 and fit_warn == 0:  # shirley base line warn
                if i not in efi:
                    efi.append(i)
                if i in efi_x:
                    efi_x.remove(i)
                if i in efi_err:
                    efi_err.remove(i)
            else:
                if i not in efi_err:
                    efi_err.append(i)
                if i in efi_x:
                    efi_x.remove(i)
                if i in efi:
                    efi.remove(i)
    except RuntimeError:
        if i not in efi_err:
            efi_err.append(i)
        if i in efi_x:
            efi_x.remove(i)
        if i in efi:
            efi.remove(i)
        a1 = [(emin[i]+emax[i])/2, (np.max(y)-ebase[i]), 5, ebase[i]]
        a2 = [(emin[i]+emax[i])/2, (np.max(y)-ebase[i]), 5, ebase[i],
              (emin[i]+emax[i])/2, (np.max(y)-ebase[i]), 5, ebase[i]]

    # epos[i]=a[0]
    # efwhm[i]=a[2]

    fexx[i, :len(xx)] = xx
    feyy[i, :len(yy)] = yy
    fex[i, :] = x
    fey[i, :] = y
    evv[i] = phi[i]
    if efp[i] == 1:
        eaa1[i, :] = a1
    elif efp[i] == 2:
        eaa2[i, :] = a2


def fermv():
    global ebrmv, flermv, eirmv, emin, emax, efi, efi_err, efi_x, cei, efp
    i = efiti.get()
    flermv *= -1
    if flermv == 1:
        eirmv = i
        ebrmv.config(text='End Remove', bg='red')
    else:
        ti = sorted([i, eirmv])
        for i in np.linspace(ti[0], ti[1], ti[1]-ti[0]+1, dtype=int):
            efp[i] = 1
            emin[i], emax[i] = np.min(ev), np.max(ev)
            if i not in efi_x:
                efi_x.append(i)
            if i in efi:
                efi.remove(i)
            if i in efi_err:
                efi_err.remove(i)
            if i in cei:
                cei.remove(i)
        eplfi()
        ebrmv.config(text='Start Remove', bg='white')
        efitplot()


def feedmove(event):
    global eedxdata, eedydata, eedfitout
    if event.xdata != None:
        eedfitout.get_tk_widget().config(cursor="crosshair")
        eedxdata.config(text='xdata:'+str('%.3f' % event.xdata))
        eedydata.config(text='ydata:'+str('%.3f' % event.ydata))
    else:
        eedfitout.get_tk_widget().config(cursor="")
        try:
            eedxdata.config(text='xdata:')
            eedydata.config(text='ydata:')
        except NameError:
            pass


def saveefit():
    path = fd.asksaveasfilename(title="Save EDC Fitted Data", initialdir=rdd,
                                initialfile=name+"_efit.npz", filetype=[("NPZ files", ".npz"),])
    if len(path) > 2:
        np.savez(path, path=rdd, fphi=fphi, efwhm=efwhm, epos=epos, semin=semin,
                 semax=semax, seaa1=seaa1, seaa2=seaa2, sefp=sefp, sefi=sefi)


scei = []


def feend():
    global epos, efwhm, fphi, eedxdata, eedydata, eedfitout, semin, semax, seaa1, seaa2, sefp, sefi, fk, fpr, scei
    fphi, epos, efwhm = [], [], []
    semin, semax, seaa1, seaa2 = emin, emax, eaa1, eaa2
    sefp = efp
    sefi = efi
    for i, v in enumerate(efi):
        if efp[v] == 1:
            fphi.append(phi[v])
            epos.append(eaa1[v, 0])
            efwhm.append(eaa1[v, 2])
        elif efp[v] == 2:
            fphi.append(phi[v])
            fphi.append(phi[v])
            epos.append(eaa2[v, 0])
            epos.append(eaa2[v, 4])
            efwhm.append(eaa2[v, 2])
            efwhm.append(eaa2[v, 6])
    fphi, epos, efwhm = np.float64(fphi), np.float64(epos), np.float64(efwhm)
    ffphi = np.float64(k_offset.get())+fphi
    fk = (2*m*epos*1.6*10**-19)**0.5 * \
        np.sin(ffphi/180*np.pi)*10**-10/(h/2/np.pi)
    scei = cei
    fpr = 1
    eendg = tk.Toplevel(g)
    eendg.title('EDC Lorentz Fit Result')
    fr = tk.Frame(master=eendg, bd=5)
    fr.grid(row=0, column=0)
    efitfig = Figure(figsize=(8, 6), layout='constrained')
    eedfitout = tkagg.FigureCanvasTkAgg(efitfig, master=fr)
    eedfitout.get_tk_widget().grid(row=0, column=0)
    eedfitout.mpl_connect('motion_notify_event', feedmove)

    a = efitfig.subplots()
    a.scatter(fphi, epos+efwhm/2, c='r', s=10)
    a.scatter(fphi, epos-efwhm/2, c='r', s=10)
    a.scatter(fphi, epos, c='k', s=10)
    a.set_xlabel(r'$\phi$'+' (deg)')
    a.set_ylabel('Kinetic Energy (eV)', fontsize=14)
    eedfitout.draw()

    xydata = tk.Frame(master=fr, bd=5)
    xydata.grid(row=1, column=0)

    eedxdata = tk.Label(xydata, text='xdata:', font=(
        "Arial", 12, "bold"), width='15', height='1', bd=10, bg='white')
    eedxdata.grid(row=0, column=0)
    eedydata = tk.Label(xydata, text='ydata:', font=(
        "Arial", 12, "bold"), width='15', height='1', bd=10, bg='white')
    eedydata.grid(row=0, column=1)

    bsave = tk.Button(master=eendg, text='Save Fitted Data', command=saveefit,
                      width=30, height=1, font={'Arial', 18, "bold"}, bg='white', bd=10)
    bsave.grid(row=1, column=0)

    eendg.update()


def fefall():
    t = threading.Thread(target=efitjob)
    t.daemon = True
    t.start()


def func_cei():
    global cei, emin, emax
    if efiti.get() not in cei:
        cei.append(efiti.get())
    if len(cei) >= 2:
        cei.sort()
        for i in range(len(cei)-1):
            emin[cei[i]:cei[i+1] +
                 1] = np.linspace(emin[cei[i]], emin[cei[i+1]], cei[i+1]-cei[i]+1)
            emax[cei[i]:cei[i+1] +
                 1] = np.linspace(emax[cei[i]], emax[cei[i+1]], cei[i+1]-cei[i]+1)


def fchei(*e):
    global efitout, edxdata, edydata
    try:
        efitout.get_tk_widget().delete('rec')
        edxdata.config(text='dx:')
        edydata.config(text='dy:')
    except:
        pass
    efitplot()


def eplfi():
    global eiout, eifig, elind, erind
    i = efiti.get()
    eifig.clear()
    eiax = eifig.add_axes([0, 0, 1, 1])
    eiax.scatter(efi_x, [0 for i in range(len(efi_x))], marker='|', c='k')
    eiax.scatter(efi, [0 for i in range(len(efi))], marker='|', c='b')
    eiax.scatter(efi_err, [0 for i in range(len(efi_err))], marker='|', c='r')
    if i in efi_x:
        elind.config(bg='white')
        erind.config(bg='white')
    if i in efi:
        elind.config(bg='blue')
        erind.config(bg='blue')
    if i in efi_err:
        elind.config(bg='red')
        erind.config(bg='red')
    try:
        eiax.set_xlim([np.min([efi, efi_x, efi_err]),
                      np.max([efi, efi_x, efi_err])])
    except ValueError:
        pass
    eiax.set_yticks([])
    eiout.draw()


def efitplot():  # efiti Scale
    global efitax, exl, eyl, elmin, elmax, texl, emin, emax
    i = efiti.get()
    efitfig.clear()
    efitax = efitfig.subplots()
    # 'Pos:'+str(round(eaa1[i,0],3))+' (eV)'+', FWHM:'+str(round(eaa1[i,2],3))+' (eV)'
    efitax.set_title(
        'Deg:'+str(round(evv[i], 3))+r' $^{\circ}$'+', '+str(efp[i])+' Peak')
    efitax.scatter(fex[i, :], fey[i, :], c='k', s=4)
    sbg = shirley_bg(feyy[i, np.argwhere(feyy[i, :] >= -20)])
    if efp[i] == 1:
        if eaa1[i, 0] == (emin[i]+emax[i])/2 and eaa1[i, 2] == 5:
            fl, = efitax.plot(fexx[i, np.argwhere(fexx[i, :] >= -20)], gl1(
                fexx[i, np.argwhere(fexx[i, :] >= -20)], *eaa1[i, :])+sbg, 'r-', lw=2)
        else:
            gl1_1 = np.float64(np.concatenate(
                gl1(fexx[i, np.argwhere(fexx[i, :] >= -20)], *eaa1[i, :]))+sbg.transpose())
            fl, = efitax.plot(fexx[i, np.argwhere(fexx[i, :] >= -20)], gl1(
                fexx[i, np.argwhere(fexx[i, :] >= -20)], *eaa1[i, :])+sbg, 'b-', lw=2)
            efitax.fill_between(np.concatenate((fexx[i, np.argwhere(fexx[i, :] >= -20)])), np.float64(
                np.concatenate(sbg.transpose())), np.concatenate(gl1_1), facecolor='blue', alpha=0.5)

    elif efp[i] == 2:
        if eaa2[i, 0] == (emin[i]+emax[i])/2 and eaa2[i, 2] == 5:
            fl, = efitax.plot(fexx[i, np.argwhere(fexx[i, :] >= -20)], gl2(
                fexx[i, np.argwhere(fexx[i, :] >= -20)], *eaa2[i, :])+sbg, 'r-', lw=2)
        else:
            gl2_1 = np.float64(np.concatenate(
                gl1(fexx[i, np.argwhere(fexx[i, :] >= -20)], *eaa2[i, :4]))+sbg.transpose())
            gl2_2 = np.float64(np.concatenate(
                gl1(fexx[i, np.argwhere(fexx[i, :] >= -20)], *eaa2[i, -4:]))+sbg.transpose())
            fl, = efitax.plot(fexx[i, np.argwhere(fexx[i, :] >= -20)], gl2(
                fexx[i, np.argwhere(fexx[i, :] >= -20)], *eaa2[i, :])+sbg, 'b-', lw=2)
            efitax.fill_between(np.concatenate(fexx[i, np.argwhere(fexx[i, :] >= -20)]), np.float64(
                np.concatenate(sbg.transpose())), np.concatenate(gl2_1), facecolor='green', alpha=0.5)
            efitax.fill_between(np.concatenate(fexx[i, np.argwhere(fexx[i, :] >= -20)]), np.float64(
                np.concatenate(sbg.transpose())), np.concatenate(gl2_2), facecolor='yellow', alpha=0.5)

    if bg_warn == 1:  # shirley base line warn
        efitax.plot(fexx[i, np.argwhere(fexx[i, :] >= -20)], sbg, 'r--')
    else:
        efitax.plot(fexx[i, np.argwhere(fexx[i, :] >= -20)], sbg, 'g--')

    efitax.scatter(fexx[i, np.argwhere(fexx[i, :] >= -20)],
                   feyy[i, np.argwhere(feyy[i, :] >= -20)], c='g', s=4)
    if (emin[i], emax[i]) != (np.min(ev), np.max(ev)):
        elmin = efitax.axvline(emin[i], c='r')
        elmax = efitax.axvline(emax[i], c='r')
    else:
        elmin = efitax.axvline(emin[i], c='grey')
        elmax = efitax.axvline(emax[i], c='grey')
        fl.set_alpha(0.3)

    efitax.set_xlabel('Kinetic Energy (eV)', fontsize=14)
    efitax.set_ylabel('Intensity (Counts)', fontsize=14)
    exl = efitax.get_xlim()
    eyl = efitax.get_ylim()
    texl = np.copy(exl)
    efitout.draw()
    eplfi()


def emove(event):
    global exdata, eydata, edxdata, edydata, x2, y2, efitax, efitout, elmin, elmax, emin, emax, tpx1, tpx2, tpy1, tpy2, tx2, ty2
    if event.xdata != None:
        # efitout.get_tk_widget().config(cursor="crosshair")
        # try:
        #     # efitout.get_tk_widget().delete('rec')
        #     # efitout.get_tk_widget().delete('x1')
        #     # efitout.get_tk_widget().delete('y1')
        #     # efitout.get_tk_widget().delete('x2')
        #     # efitout.get_tk_widget().delete('y2')
        # except:
        #     pass
        if emof == -1:
            x2, y2 = event.xdata, event.ydata
            px2, py2 = event.x, event.y

            if felmin == 1 and temin+(x2-x1) >= exl[0] and temin+(x2-x1) <= exl[1]:
                elmin.remove()
                elmin = efitax.axvline(x2, c='r')
                emin[efiti.get()] = x2
                elmax.set_color('r')
                efitout.draw()
            elif felmax == 1 and temax+(x2-x1) >= exl[0] and temax+(x2-x1) <= exl[1]:
                elmax.remove()
                elmax = efitax.axvline(x2, c='r')
                emax[efiti.get()] = x2
                elmin.set_color('r')
                efitout.draw()
            elif feregion == 1 and temin+(x2-x1) >= exl[0] and temax+(x2-x1) <= exl[1]:
                elmin.remove()
                elmin = efitax.axvline(temin+(x2-x1), c='r')
                emin[efiti.get()] = temin+(x2-x1)
                elmax.remove()
                elmax = efitax.axvline(temax+(x2-x1), c='r')
                emax[efiti.get()] = temax+(x2-x1)
                efitout.draw()
            elif felmin == 0 and felmax == 0 and feregion == 0:
                efitout.get_tk_widget().delete('rec')
                tpx1, tpy1, tpx2, tpy2 = px1, py1, px2, py2
                efitout.get_tk_widget().create_rectangle(
                    (px1, 600-py1), (px2, 600-py2), outline='grey', width=2, tag='rec')
                [tpx1, tpx2] = sorted([tpx1, tpx2])
                [tpy1, tpy2] = sorted([tpy1, tpy2])
                tx2, ty2 = x2, y2
                edxdata.config(text='dx:'+str('%.3f' % abs(x2-x1)))
                edydata.config(text='dy:'+str('%.3f' % abs(y2-y1)))
        exdata.config(text='xdata:'+str('%.3f' % event.xdata))
        eydata.config(text='ydata:'+str('%.3f' % event.ydata))
    else:
        efitout.get_tk_widget().config(cursor="")
        try:
            exdata.config(text='xdata:')
            eydata.config(text='ydata:')
        except NameError:
            pass

    # print("event.xdata", event.xdata)
    # print("event.ydata", event.ydata)
    # print("event.inaxes", event.inaxes)
    # print("x", event.x)
    # print("y", event.y)
emof = 1


def epress(event):
    # event.button 1:left 3:right 2:mid
    # event.dblclick : bool
    # print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
    #       ('double' if event.dblclick else 'single', event.button,
    #        event.x, event.y, event.xdata, event.ydata))
    global x1, y1, emof, px1, py1, efitax, efitout, elmin, elmax, felmin, felmax, cei, ebase, feregion, temin, temax, tx1, ty1
    if event.button == 1 and event.inaxes:
        x1, y1 = event.xdata, event.ydata
        px1, py1 = event.x, event.y
        felmin, felmax, feregion = 0, 0, 0
        temin, temax = emin[efiti.get()], emax[efiti.get()]
        if efitout.get_tk_widget().find_withtag('rec') != () and px1 > tpx1 and px1 < tpx2 and py1 > tpy1 and py1 < tpy2:
            pass
        elif abs(x1-emin[efiti.get()]) < (texl[1]-texl[0])/80:
            felmin = 1

        elif abs(x1-emax[efiti.get()]) < (texl[1]-texl[0])/80:
            felmax = 1

        elif x1 > emin[efiti.get()] and x1 < emax[efiti.get()]:
            feregion = 1

        elif efitout.get_tk_widget().find_withtag('rec') == ():
            tx1, ty1 = x1, y1
        emof = -1
    elif event.button == 3:
        try:
            efitout.get_tk_widget().delete('rec')
            edxdata.config(text='dx:')
            edydata.config(text='dy:')
        except:
            pass
        efitax.set_xlim(exl)
        efitax.set_ylim(eyl)
        efitout.draw()
        emof = 1


def erelease(event):
    global x1, y1, x2, y2, emof, efitout, efitax, felmax, felmin, elmin, elmax, emin, emax, feregion, texl
    if event.button == 1 and emof == -1 and event.inaxes:
        x2, y2 = event.xdata, event.ydata
        if emin[efiti.get()] > emax[efiti.get()]:
            emin[efiti.get()], emax[efiti.get()
                                    ] = emax[efiti.get()], emin[efiti.get()]
            elmin, elmax = elmax, elmin
        else:
            emin[efiti.get()], emax[efiti.get()
                                    ] = emin[efiti.get()], emax[efiti.get()]
            elmin, elmax = elmin, elmax
        if felmin == 0 and felmax == 0 and feregion == 0 and (x2, y2) == (x1, y1) and px1 > tpx1 and px1 < tpx2 and py1 > tpy1 and py1 < tpy2:
            try:
                efitout.get_tk_widget().delete('rec')
            except:
                pass
            efitax.set_xlim(sorted([tx1, tx2]))
            efitax.set_ylim(sorted([ty1, ty2]))
            texl = sorted([x1, x2])
            efitout.draw()
        elif felmin == 1 or felmax == 1 or feregion == 1:
            func_cei()
            x1, x2, y1, y2 = [], [], [], []
            efit()
            efitplot()
        emof = 1


def testate():
    try:
        while True:
            estate.config(text=str(est.get()))
    except KeyboardInterrupt:
        pass


def eflind():
    global efiti
    ti = efiti.get()
    if ti in efi:
        for i in range(ti+1):
            if ti-i not in efi:
                efiti.set(ti-i)
                break
    elif ti in efi_err:
        for i in range(ti+1):
            if ti-i not in efi_err:
                efiti.set(ti-i)
                break
    elif ti in efi_x:
        for i in range(ti+1):
            if ti-i in efi or ti-i in efi_err:
                efiti.set(ti-i)
                break
        if i == ti and ti != 0:
            efiti.set(ti-1)


def efrind():
    global efiti
    ti = efiti.get()
    if ti in efi:
        for i in range(len(phi)-ti):
            if ti+i not in efi:
                efiti.set(ti+i)
                break
    elif ti in efi_err:
        for i in range(len(phi)-ti):
            if ti+i not in efi_err:
                efiti.set(ti+i)
                break
    elif ti in efi_x:
        for i in range(len(phi)-ti):
            if ti+i in efi or ti+i in efi_err:
                efiti.set(ti+i)
                break
        if i == len(phi)-ti-1 and ti != len(phi)-1:
            efiti.set(ti+1)


def o_fewf1(*e):
    global ewf1
    if '' == ewf1.get():
        ewf1.set('0')
        ein_w1.select_range(0, 1)


def fewf1(*e):
    t = threading.Thread(target=o_fewf1)
    t.daemon = True
    t.start()


def o_fewf2(*e):
    global ewf2
    if '' == ewf2.get():
        ewf2.set('0')
        ein_w2.select_range(0, 1)


def fewf2(*e):
    t = threading.Thread(target=o_fewf2)
    t.daemon = True
    t.start()


def o_feaf1(*e):
    global eaf1
    if '' == eaf1.get():
        eaf1.set('0')
        ein_a1.select_range(0, 1)


def feaf1(*e):
    t = threading.Thread(target=o_feaf1)
    t.daemon = True
    t.start()


def o_feaf2(*e):
    global eaf2
    if '' == eaf2.get():
        eaf2.set('0')
        ein_a2.select_range(0, 1)


def feaf2(*e):
    t = threading.Thread(target=o_feaf2)
    t.daemon = True
    t.start()


def ejob():     # MDC Fitting GUI
    global g, efiti, efitfig, efitout, egg, exdata, eydata, edxdata, edydata, eiout, eifig, efi, efi_err, efi_x, ebrmv, flermv, ebcgl2, efp, flecgl2, fpr, est, estate, ewf1, ewf2, eaf1, eaf2, elind, erind, ein_w1, ein_w2, ein_a1, ein_a2
    egg = tk.Toplevel(g, bg='white')
    egg.title('EDC Lorentz Fit')
    est = queue.Queue(maxsize=0)
    estate = tk.Label(egg, text='', font=(
        "Arial", 20, "bold"), bg="white", fg="black")
    estate.grid(row=0, column=0)

    fr = tk.Frame(master=egg, bg='white')
    fr.grid(row=1, column=0)
    frind = tk.Frame(master=fr, bg='white')
    frind.grid(row=0, column=0)
    elind = tk.Button(frind, text='<<', command=eflind, width=10,
                      height=5, font={'Arial', 50, "bold"}, bg='white')
    elind.grid(row=0, column=0)
    erind = tk.Button(frind, text='>>', command=efrind, width=10,
                      height=5, font={'Arial', 50, "bold"}, bg='white')
    erind.grid(row=0, column=2)

    efiti = tk.IntVar()
    efiti.set(0)
    efiti.trace_add('write', fchei)
    chi = tk.Scale(frind, label='Index', from_=0, to=len(phi)-1, orient='horizontal',
                   variable=efiti, state='active', bg='white', fg='black', length=580, width=50, resolution=1)
    chi.grid(row=0, column=1)

    efi, efi_err, efi_x = [], [], [i for i in range(len(phi))]
    eifig = Figure(figsize=(6, 0.2), layout='tight')
    eiout = tkagg.FigureCanvasTkAgg(eifig, master=frind)
    eiout.get_tk_widget().grid(row=1, column=1)

    efitfig = Figure(figsize=(8, 6), layout='constrained')
    efitout = tkagg.FigureCanvasTkAgg(efitfig, master=fr)
    efitout.get_tk_widget().grid(row=1, column=0)
    efitout.mpl_connect('motion_notify_event', emove)
    efitout.mpl_connect('button_press_event', epress)
    efitout.mpl_connect('button_release_event', erelease)

    xydata = tk.Frame(master=fr, bd=5, bg='white')
    xydata.grid(row=2, column=0)

    exdata = tk.Label(xydata, text='xdata:', font=(
        "Arial", 12, "bold"), width='15', height='1', bd=10, bg='white')
    exdata.grid(row=0, column=0)
    eydata = tk.Label(xydata, text='ydata:', font=(
        "Arial", 12, "bold"), width='15', height='1', bd=10, bg='white')
    eydata.grid(row=0, column=1)
    edxdata = tk.Label(xydata, text='dx:', font=(
        "Arial", 12, "bold"), width='15', height='1', bd=10, bg='white')
    edxdata.grid(row=0, column=2)
    edydata = tk.Label(xydata, text='dy:', font=(
        "Arial", 12, "bold"), width='15', height='1', bd=10, bg='white')
    edydata.grid(row=0, column=3)

    # bstop=tk.Button(gg,command=stop,text='Stop',font=('Arial',20),bd=10)
    # bstop.grid(row=1,column=0)

    frpara = tk.Frame(master=egg, bd=5, bg='white')
    frpara.grid(row=1, column=1)
    try:
        if fpr == 1:
            efp = list(sefp)
            efi = list(sefi)
        else:
            efp = [1 for i in range(len(phi))]
    except:
        efp = [1 for i in range(len(phi))]
    flecgl2 = -1
    frpara00 = tk.Frame(master=frpara, bd=5, bg='white')
    frpara00.grid(row=0, column=0)
    l1 = tk.Label(frpara00, text='Index Operation', font=(
        "Arial", 12, "bold"), width='15', height='1', bd=5, bg='white')
    l1.grid(row=0, column=0)
    froperind = tk.Frame(master=frpara00, bd=5, bg='white')
    froperind.grid(row=1, column=0)
    ebcgl2 = tk.Button(froperind, text='Start Add 2 Peaks', command=fecgl2,
                       width=30, height=1, font={'Arial', 18, "bold"}, bg='white')
    ebcgl2.grid(row=0, column=0)
    ebrmv = tk.Button(froperind, text='Start Remove', command=fermv,
                      width=30, height=1, font={'Arial', 18, "bold"}, bg='white')
    ebrmv.grid(row=0, column=1)

    frwr = tk.Frame(master=froperind, bd=5, bg='white')
    frwr.grid(row=1, column=0)
    l2 = tk.Label(frwr, text='FWHM Ratio', font=(
        "Arial", 12, "bold"), width='15', height='1', bd=5, bg='white')
    l2.grid(row=0, column=1)
    l3 = tk.Label(frwr, text=':', font=("Arial", 12, "bold"),
                  width='15', height='1', bd=5, bg='white')
    l3.grid(row=1, column=1)
    ewf1 = tk.StringVar()
    ewf1.set('0')
    ewf1.trace_add('write', fewf1)
    ein_w1 = tk.Entry(frwr, font=("Arial", 10, "bold"),
                      width=7, textvariable=ewf1, bd=5)
    ein_w1.grid(row=1, column=0)
    ewf2 = tk.StringVar()
    ewf2.set('0')
    ewf2.trace_add('write', fewf2)
    ein_w2 = tk.Entry(frwr, font=("Arial", 10, "bold"),
                      width=7, textvariable=ewf2, bd=5)
    ein_w2.grid(row=1, column=2)

    frar = tk.Frame(master=froperind, bd=5, bg='white')
    frar.grid(row=2, column=0)
    l2 = tk.Label(frar, text='Area Ratio', font=(
        "Arial", 12, "bold"), width='15', height='1', bd=5, bg='white')
    l2.grid(row=0, column=1)
    l3 = tk.Label(frar, text=':', font=("Arial", 12, "bold"),
                  width='15', height='1', bd=5, bg='white')
    l3.grid(row=1, column=1)
    eaf1 = tk.StringVar()
    eaf1.set('0')
    eaf1.trace_add('write', feaf1)
    ein_a1 = tk.Entry(frar, font=("Arial", 10, "bold"),
                      width=7, textvariable=eaf1, bd=5)
    ein_a1.grid(row=1, column=0)
    eaf2 = tk.StringVar()
    eaf2.set('0')
    eaf2.trace_add('write', feaf2)
    ein_a2 = tk.Entry(frar, font=("Arial", 10, "bold"),
                      width=7, textvariable=eaf2, bd=5)
    ein_a2.grid(row=1, column=2)

    frout = tk.Frame(master=egg, bd=5, bg='white')
    frout.grid(row=2, column=0)
    bfall = tk.Button(frout, text='Fit All', command=fefall,
                      width=30, height=1, font={'Arial', 18, "bold"}, bg='white')
    bfall.grid(row=0, column=0)
    flermv = -1
    bend = tk.Button(frout, text='Finish', command=feend, width=30,
                     height=1, font={'Arial', 18, "bold"}, bg='white')
    bend.grid(row=1, column=0)

    if eprfit == 1:
        fefall()
    else:
        efitplot()
    tt = threading.Thread(target=testate)
    tt.daemon = True
    tt.start()
    egg.update()
################################# efit above ####################################################

####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
#################################### original mfit below ######################################################
# finish
#################################### original mfit above ######################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################


def fmcgl2():
    global mbcgl2, kmin, kmax, flmcgl2, micgl2, mfp, mbcomp1, mbcomp2, flmcomp1, flmcomp2
    mbcomp1.config(state='active')
    mbcomp2.config(state='active')
    flmcomp1, flmcomp2 = -1, -1
    i = mfiti.get()
    flmcgl2 *= -1
    if flmcgl2 == 1:
        micgl2 = i
        mbcgl2.config(text='End Add 2 Peaks', bg='red')
    else:
        ti = sorted([i, micgl2])
        for i in np.linspace(ti[0], ti[1], ti[1]-ti[0]+1, dtype=int):
            mfp[i] = 2
            if i not in mfi_x:
                mfi_x.append(i)
            if i in mfi:
                mfi.remove(i)
            if i in mfi_err:
                mfi_err.remove(i)
        mbcgl2.config(text='Start Add 2 Peaks', bg='white')
        mfitplot()

def pack_fitpar(mresult):
    if len(smresult) > 0:
        o=smresult
        for ii,result in enumerate(mresult):
            try:
                s = putfitpar(result)
                for i in range(len(o[ii])):
                    o[ii][i]=""
                for i in s:
                    '''preprocess the string to put values in the labels'''
                    if 'x1*xr1+xr2' in i:
                        if xr2>=0:
                            i = i.replace(' == \'x1*xr1+xr2\'', '='+str(xr1)+'*x1+'+str(xr2))
                        else:
                            i = i.replace(' == \'x1*xr1+xr2\'', '='+str(xr1)+'*x1-'+str(-xr2))
                    if "(x2-xr2) / xr1" in i:
                        if xr2>=0:
                            i = i.replace(' == \'(x2-xr2) / xr1\'','=(x2-'+str(xr2) + ')/'+str(xr1))
                        else:
                            i = i.replace(' == \'(x2-xr2) / xr1\'','=(x2+'+str(-xr2) + ')/'+str(xr1))
                    if 'w1/wr1*wr2' in i:
                        i = i.replace(' == \'w1/wr1*wr2\'', '=w1/'+str(wr1)+'*'+str(wr2))
                    
                    '''assign the values to the labels'''
                    if 'x:' in i:
                        o[ii][0]=i
                    if 'h:' in i:
                        o[ii][1]=i
                    if 'w:' in i:
                        o[ii][2]=i
                    if 'x1:' in i:
                        o[ii][0]=i
                    if 'x2:' in i:
                        print('x2')
                        print(i)
                        print(str(i))
                        o[ii][1]=i
                        print(o[ii][1])
                    if 'h1:' in i:
                        o[ii][2]=i
                    if 'h2:' in i:
                        o[ii][3]=i
                    if 'w1:' in i:
                        o[ii][4]=i
                    if 'w2:' in i:
                        o[ii][5]=i
            except:
                pass
    else:
        o=[[]for i in range(len(mresult))]
        for ii,result in enumerate(mresult):
            try:
                s = putfitpar(result)
            except:
                s=[]
                if mfp[ii]==2:
                    for i in ['x1: nofit','x2: nofit','h1: nofit','h2: nofit','w1: nofit','w2: nofit']:
                        s.append(i)
                elif mfp[ii]==1:
                    for i in ['x: nofit','h: nofit','w: nofit']:
                        s.append(i)
            for i in s:
                if 'x:' in i:
                    o[ii].append(i)
                if 'h:' in i:
                    o[ii].append(i)
                if 'w:' in i:
                    o[ii].append(i)
                if 'x1:' in i:
                    o[ii].append(i)
                if 'x2:' in i:
                    o[ii].append(i)
                if 'h1:' in i:
                    o[ii].append(i)
                if 'h2:' in i:
                    o[ii].append(i)
                if 'w1:' in i:
                    o[ii].append(i)
                if 'w2:' in i:
                    o[ii].append(i)
    return o
def mfitjob():
    global fmxx, fmyy, fmx, fmy, mvv, maa1, maa2, kmin, kmax, mfi, mfi_err, mfi_x, st, mst, result, fa1, fa2, fit_warn, wr1, wr2, mresult, xr1, xr2
    if len(mfi) < 1:
        mfi, mfi_err, mfi_x = [], [], []
    else:
        mfi, mfi_err, mfi_x = list(mfi), list(mfi_err), list(mfi_x)
    pbar = tqdm.tqdm(total=len(ev), desc='Fitting MDC', colour='green')
    for i in range(len(ev)):
        mbase[i] = int(base.get())  # 待調整
        fmxx[i, :] = fmxx[i, :]/fmxx[i, :]*-50
        fmyy[i, :] = fmyy[i, :]/fmyy[i, :]*-50
        ecut = data.sel(eV=ev[i], method='nearest')
        x = (2*m*ev[i]*1.6*10**-19)**0.5 * \
            np.sin(phi/180*np.pi)*10**-10/(h/2/np.pi)
        y = ecut.to_numpy().reshape(len(x))
        tx = x[np.argwhere(x >= kmin[i])].flatten()
        xx = tx[np.argwhere(tx <= kmax[i])].flatten()
        ty = y[np.argwhere(x >= kmin[i])].flatten()
        yy = ty[np.argwhere(tx <= kmax[i])].flatten()
        yy = np.where(yy > mbase[i], yy, mbase[i])
        try:
            # if (kmin[i],kmax[i])==((2*m*ev[i]*1.6*10**-19)**0.5*np.sin(-0.5/180*np.pi)*10**-10/(h/2/np.pi),(2*m*ev[i]*1.6*10**-19)**0.5*np.sin(0.5/180*np.pi)*10**-10/(h/2/np.pi)) and i not in mfi:
            # if i not in mfi:
            #     if i not in mfi_x:
            #         mfi_x.append(i)
            #     # if i in mfi:
            #     #     mfi.remove(i)
            #     if i in mfi_err:
            #         mfi_err.remove(i)
            #     a1=[(kmin[i]+kmax[i])/2,(np.max(y)-mbase[i]),5,mbase[i]]
            #     a2=[(kmin[i]+kmax[i])/2,(np.max(y)-mbase[i]),5,mbase[i],(kmin[i]+kmax[i])/2,(np.max(y)-mbase[i]),5,mbase[i]]
            # elif (kmin[i],kmax[i])!=((2*m*ev[i]*1.6*10**-19)**0.5*np.sin(-0.5/180*np.pi)*10**-10/(h/2/np.pi),(2*m*ev[i]*1.6*10**-19)**0.5*np.sin(0.5/180*np.pi)*10**-10/(h/2/np.pi)):
            if mfp[i] == 1:
                if i in mfi_err and (kmin[i], kmax[i]) != ((2*m*ev[i]*1.6*10**-19)**0.5*np.sin(-0.5/180*np.pi)*10**-10/(h/2/np.pi), (2*m*ev[i]*1.6*10**-19)**0.5*np.sin(0.5/180*np.pi)*10**-10/(h/2/np.pi)):
                    # a1,b=curve_fit(gl1,xx,yy-lnr_bg(yy),bounds=([kmin[i],(np.max(y)-mbase[i])/10,0,0],[kmax[i],np.max(y)-mbase[i]+1,0.3,0.01]))
                    # fit_warn=0
                    pars = Parameters()
                    pars.add(
                        'x', value=kmin[i]+(kmax[i]-kmin[i])*0.3, min=kmin[i], max=kmax[i])
                    pars.add('h', value=(
                        np.max(y)-mbase[i])+1, min=(np.max(y)-mbase[i])/10, max=np.max(y)-mbase[i]+1)
                    pars.add('w', value=0.1, min=0.01, max=0.2)
                    pars.add('y', value=0, vary=False)
                    fitter = Minimizer(
                        fgl1, pars, fcn_args=(xx, yy-lnr_bg(yy)))
                    result = fitter.minimize()
                    a1 = toa1()
                    checkfit()
                    if fit_warn == 1:
                        t = 5
                        while t > 0 and fit_warn == 1:
                            result = fitter.minimize()
                            a1 = toa1()
                            checkfit()
                            t -= 1
                else:
                    if i in mfi:
                        result = mresult[i]
                    a1 = maa1[i, :]
                    if (kmin[i], kmax[i]) == ((2*m*ev[i]*1.6*10**-19)**0.5*np.sin(-0.5/180*np.pi)*10**-10/(h/2/np.pi), (2*m*ev[i]*1.6*10**-19)**0.5*np.sin(0.5/180*np.pi)*10**-10/(h/2/np.pi)):
                        fit_warn = 2
                    elif i not in mfi:
                        # a1,b=curve_fit(gl1,xx,yy-lnr_bg(yy),bounds=([kmin[i],(np.max(y)-mbase[i])/10,0,0],[kmax[i],np.max(y)-mbase[i]+1,0.3,0.01]))
                        # fit_warn=0
                        pars = Parameters()
                        pars.add(
                            'x', value=kmin[i]+(kmax[i]-kmin[i])*0.3, min=kmin[i], max=kmax[i])
                        pars.add('h', value=(
                            np.max(y)-mbase[i])+1, min=(np.max(y)-mbase[i])/10, max=np.max(y)-mbase[i]+1)
                        pars.add('w', value=0.1, min=0.01, max=0.2)
                        pars.add('y', value=0, vary=False)
                        fitter = Minimizer(
                            fgl1, pars, fcn_args=(xx, yy-lnr_bg(yy)))
                        result = fitter.minimize()
                        a1 = toa1()
                        checkfit()
                        if fit_warn == 1:
                            t = 5
                            while t > 0 and fit_warn == 1:
                                result = fitter.minimize()
                                a2 = toa2()
                                checkfit()
                                t -= 1
                    else:
                        fit_warn = 0
            elif mfp[i] == 2:
                if i in mfi_err and (kmin[i], kmax[i]) != ((2*m*ev[i]*1.6*10**-19)**0.5*np.sin(-0.5/180*np.pi)*10**-10/(h/2/np.pi), (2*m*ev[i]*1.6*10**-19)**0.5*np.sin(0.5/180*np.pi)*10**-10/(h/2/np.pi)):
                    pars = Parameters()
                    xr1, xr2 = float(mxf1.get()), float(mxf2.get())
                    wr1, wr2 = float(mwf1.get()), float(mwf2.get())
                    fa1, fa2 = float(maf1.get()), float(maf2.get())
                    pars.add(
                        'x1', value=kmin[i]+(kmax[i]-kmin[i])*0.3, min=kmin[i], max=kmax[i])
                    if flmposcst == 1:
                        pars.add('xr1', value=xr1, vary=False)
                        pars.add('xr2', value=xr2, vary=False)
                        pars.add('x2', expr='x1*xr1+xr2')
                    else:
                        pars.add(
                            'x2', value=kmax[i]-(kmax[i]-kmin[i])*0.3, min=kmin[i], max=kmax[i])
                    pars.add('h1', value=(
                        np.max(y)-mbase[i])+1, min=(np.max(y)-mbase[i])/10, max=np.max(y)-mbase[i]+1)
                    pars.add('h2', value=(
                        np.max(y)-mbase[i])+1, min=(np.max(y)-mbase[i])/10, max=np.max(y)-mbase[i]+1)
                    pars.add('w1', value=0.02, min=0, max=0.2)
                    if wr1 != 0 and wr2 != 0:
                        pars.add('wr1', value=wr1, vary=False)
                        pars.add('wr2', value=wr2, vary=False)
                        pars.add('w2', expr='w1/wr1*wr2')
                    else:
                        pars.add('w2', value=0.02, min=0, max=0.2)
                    pars.add('y1', value=0, vary=False)
                    pars.add('y2', value=0, vary=False)
                    if fa1 != 0 and fa2 != 0:
                        fitter = Minimizer(
                            fgl2_a, pars, fcn_args=(xx, yy-lnr_bg(yy)))
                        result = fitter.minimize()
                    else:
                        fitter = Minimizer(
                            fgl2, pars, fcn_args=(xx, yy-lnr_bg(yy)))
                        result = fitter.minimize()
                    a2 = toa2()
                    checkfit()
                    if fit_warn == 1:
                        t = 5
                        while t > 0 and fit_warn == 1:
                            result = fitter.minimize()
                            a2 = toa2()
                            checkfit()
                            t -= 1
                    # p0=[kmin[i]+(kmax[i]-kmin[i])*0.3,(np.max(y)-mbase[i])+1,0.1,0,kmax[i]-(kmax[i]-kmin[i])*0.3,(np.max(y)-mbase[i])+1,0.1,0]
                    # a2,b=curve_fit(gl2,xx,yy-lnr_bg(yy),p0=p0,bounds=([kmin[i],(np.max(y)-mbase[i])/10,0,0,kmin[i],(np.max(y)-mbase[i])/10,0,0],[kmax[i],np.max(y)-mbase[i]+1,0.3,0.01,kmax[i],np.max(y)-mbase[i]+1,0.3,0.01]))
                else:
                    if i in mfi:
                        result = mresult[i]
                    a2 = maa2[i, :]
                    if (kmin[i], kmax[i]) == ((2*m*ev[i]*1.6*10**-19)**0.5*np.sin(-0.5/180*np.pi)*10**-10/(h/2/np.pi), (2*m*ev[i]*1.6*10**-19)**0.5*np.sin(0.5/180*np.pi)*10**-10/(h/2/np.pi)):
                        fit_warn = 2
                    elif i not in mfi:
                        pars = Parameters()
                        xr1, xr2 = float(mxf1.get()), float(mxf2.get())
                        wr1, wr2 = float(mwf1.get()), float(mwf2.get())
                        fa1, fa2 = float(maf1.get()), float(maf2.get())
                        pars.add(
                            'x1', value=kmin[i]+(kmax[i]-kmin[i])*0.3, min=kmin[i], max=kmax[i])
                        if flmposcst == 1:
                            pars.add('xr1', value=xr1, vary=False)
                            pars.add('xr2', value=xr2, vary=False)
                            pars.add('x2', expr='x1*xr1+xr2')
                        else:
                            pars.add(
                                'x2', value=kmax[i]-(kmax[i]-kmin[i])*0.3, min=kmin[i], max=kmax[i])
                        pars.add('h1', value=(
                            np.max(y)-mbase[i])+1, min=(np.max(y)-mbase[i])/10, max=np.max(y)-mbase[i]+1)
                        pars.add('h2', value=(
                            np.max(y)-mbase[i])+1, min=(np.max(y)-mbase[i])/10, max=np.max(y)-mbase[i]+1)
                        pars.add('w1', value=0.02, min=0.01, max=0.2)
                        if wr1 != 0 and wr2 != 0:
                            pars.add('wr1', value=wr1, vary=False)
                            pars.add('wr2', value=wr2, vary=False)
                            pars.add('w2', expr='w1/wr1*wr2')
                        else:
                            pars.add('w2', value=0.02, min=0.01, max=0.2)
                        pars.add('y1', value=0, vary=False)
                        pars.add('y2', value=0, vary=False)
                        if fa1 != 0 and fa2 != 0:
                            fitter = Minimizer(
                                fgl2_a, pars, fcn_args=(xx, yy-lnr_bg(yy)))
                            result = fitter.minimize()
                        else:
                            fitter = Minimizer(
                                fgl2, pars, fcn_args=(xx, yy-lnr_bg(yy)))
                            result = fitter.minimize()
                        a2 = toa2()
                        checkfit()
                        if fit_warn == 1:
                            t = 5
                            while t > 0 and fit_warn == 1:
                                result = fitter.minimize()
                                a2 = toa2()
                                checkfit()
                                t -= 1
                    else:
                        fit_warn = 0
            try:
                '''using lmfit'''
                mresult[i] = result
                result = []
            except:
                '''Casa Result'''
                pass
            if fit_warn == 0:
                if i not in mfi:
                    mfi.append(i)
                if i in mfi_x:
                    mfi_x.remove(i)
                if i in mfi_err:
                    mfi_err.remove(i)
            elif fit_warn == 2:
                if i not in mfi_x:
                    mfi_x.append(i)
                if i in mfi:
                    mfi.remove(i)
                if i in mfi_err:
                    mfi_err.remove(i)
            else:
                if i not in mfi_err:
                    mfi_err.append(i)
                if i in mfi_x:
                    mfi_x.remove(i)
                if i in mfi:
                    mfi.remove(i)
            # if bg_warn==0:  #shirley base line warn
            #     if i not in mfi:
            #         mfi.append(i)
            #     if i in mfi_x:
            #         mfi_x.remove(i)
            #     if i in mfi_err:
            #         mfi_err.remove(i)
            # else:
            #     if i not in mfi_err:
            #         mfi_err.append(i)
            #     if i in mfi_x:
            #         mfi_x.remove(i)
            #     if i in mfi:
            #         mfi.remove(i)
        except RuntimeError:
            print('runtime error')
            if i not in mfi_err:
                mfi_err.append(i)
            if i in mfi_x:
                mfi_x.remove(i)
            if i in mfi:
                mfi.remove(i)
            a1 = [(kmin[i]+kmax[i])/2, (np.max(y)-mbase[i]), 5, mbase[i]]
            a2 = [(kmin[i]+kmax[i])/2, (np.max(y)-mbase[i]), 5, mbase[i],
                  (kmin[i]+kmax[i])/2, (np.max(y)-mbase[i]), 5, mbase[i]]
        # pos[i]=a[0]
        # fwhm[i]=a[2]
        fmxx[i, :len(xx)] = xx
        fmyy[i, :len(yy)] = yy
        fmx[i, :] = x
        fmy[i, :] = y
        mvv[i] = ev[i]
        if mfp[i] == 1:
            maa1[i, :] = a1
        elif mfp[i] == 2:
            maa2[i, :] = a2
        pbar.update(1)
        # print('Fitting MDC '+str(round((i+1)/len(ev)*100))+'%'+' ('+str(len(ev))+')')
        st.put('Fitting MDC '+str(round((i+1)/len(ev)*100)) +
               '%'+' ('+str(len(ev))+')')
        mst.put('Fitting MDC '+str(round((i+1)/len(ev)*100)) +
                '%'+' ('+str(len(ev))+')')
    pbar.close()
    mfitplot()


def mfit():
    global fmxx, fmyy, fmx, fmy, mvv, maa1, maa2, kmin, kmax, mfi, mfi_err, mfi_x, result, fa1, fa2, fit_warn, wr1, wr2, flmcomp1, flmcomp2, mbcomp1, mbcomp2, mresult, xr1, xr2
    mbcomp1.config(bg='white')
    mbcomp2.config(bg='white')
    mfi, mfi_err, mfi_x = list(mfi), list(mfi_err), list(mfi_x)
    i = mfiti.get()
    mbase[i] = int(base.get())  # 待調整
    fmxx[i, :] = fmxx[i, :]/fmxx[i, :]*-50
    fmyy[i, :] = fmyy[i, :]/fmyy[i, :]*-50
    ecut = data.sel(eV=ev[i], method='nearest')
    x = (2*m*ev[i]*1.6*10**-19)**0.5*np.sin(phi/180*np.pi)*10**-10/(h/2/np.pi)
    y = ecut.to_numpy().reshape(len(x))
    tx = x[np.argwhere(x >= kmin[i])].flatten()
    xx = tx[np.argwhere(tx <= kmax[i])].flatten()
    ty = y[np.argwhere(x >= kmin[i])].flatten()
    yy = ty[np.argwhere(tx <= kmax[i])].flatten()
    yy = np.where(yy > mbase[i], yy, mbase[i])
    try:
        if mfp[i] == 1:
            # a1,b=curve_fit(gl1,xx,yy-lnr_bg(yy),bounds=([kmin[i],(np.max(y)-mbase[i])/10,0,0],[kmax[i],np.max(y)-mbase[i]+1,0.3,0.01]))
            # fit_warn=0
            pars = Parameters()
            pars.add('x', value=kmin[i]+(kmax[i]-kmin[i])
                     * 0.2, min=kmin[i], max=kmax[i])
            pars.add('h', value=(
                np.max(y)-mbase[i])+1, min=(np.max(y)-mbase[i])/10, max=np.max(y)-mbase[i]+1)
            pars.add('w', value=0.1, min=0.01, max=0.2)
            pars.add('y', value=0, vary=False)
            fitter = Minimizer(fgl1, pars, fcn_args=(xx, yy-lnr_bg(yy)))
            result = fitter.minimize()
            a1 = toa1()
            checkfit()
            if fit_warn == 1:
                t = 5
                while t > 0 and fit_warn == 1:
                    result = fitter.minimize()
                    a1 = toa1()
                    checkfit()
                    t -= 1
        elif mfp[i] == 2:
            pars = Parameters()
            xr1, xr2 = float(mxf1.get()), float(mxf2.get())
            wr1, wr2 = float(mwf1.get()), float(mwf2.get())
            fa1, fa2 = float(maf1.get()), float(maf2.get())
            if flmcomp == 1:
                if flmcomp1 == 1:
                    flmcomp1 = -1
                    pars.add('x1', value=maa2[i, 0], min=kmin[i], max=kmax[i])
                    if flmposcst == 1:
                        pars.add('xr1', value=xr1, vary=False)
                        pars.add('xr2', value=xr2, vary=False)
                        pars.add('x2', expr='x1*xr1+xr2')
                    else:
                        pars.add('x2', value=maa2[i, 4], min=kmin[i], max=kmax[i])
                elif flmcomp2 == 1:
                    flmcomp2 = -1
                    pars.add('x2', value=maa2[i, 4], min=kmin[i], max=kmax[i])
                    if flmposcst == 1:
                        pars.add('xr1', value=xr1, vary=False)
                        pars.add('xr2', value=xr2, vary=False)
                        pars.add('x1', expr="(x2-xr2) / xr1")
                    else:
                        pars.add('x1', value=maa2[i, 0], min=kmin[i], max=kmax[i])
                        
                
                pars.add('h1', value=maa2[i, 1], min=(
                    np.max(y)-mbase[i])/10, max=np.max(y)-mbase[i]+1)
                pars.add('h2', value=maa2[i, 5], min=(
                    np.max(y)-mbase[i])/10, max=np.max(y)-mbase[i]+1)
                pars.add('w1', value=maa2[i, 2], min=0.01, max=0.2)
                if wr1 != 0 and wr2 != 0:
                    pars.add('wr1', value=wr1, vary=False)
                    pars.add('wr2', value=wr2, vary=False)
                    pars.add('w2', expr='w1/wr1*wr2')
                else:
                    pars.add('w2', value=maa2[i, 6], min=0.01, max=0.2)
            else:
                pars.add('x1', value=kmin[i]+(kmax[i] -
                         kmin[i])*0.3, min=kmin[i], max=kmax[i])
                if flmposcst == 1:
                    pars.add('xr1', value=xr1, vary=False)
                    pars.add('xr2', value=xr2, vary=False)
                    pars.add('x2', expr='x1*xr1+xr2')
                else:
                    pars.add(
                        'x2', value=kmax[i]-(kmax[i]-kmin[i])*0.3, min=kmin[i], max=kmax[i])
                pars.add('h1', value=(
                    np.max(y)-mbase[i])+1, min=(np.max(y)-mbase[i])/10, max=np.max(y)-mbase[i]+1)
                pars.add('h2', value=(
                    np.max(y)-mbase[i])+1, min=(np.max(y)-mbase[i])/10, max=np.max(y)-mbase[i]+1)
                pars.add('w1', value=0.02, min=0.01, max=0.2)
                if wr1 != 0 and wr2 != 0:
                    pars.add('wr1', value=wr1, vary=False)
                    pars.add('wr2', value=wr2, vary=False)
                    pars.add('w2', expr='w1/wr1*wr2')
                else:
                    pars.add('w2', value=0.02, min=0.01, max=0.2)

            pars.add('y1', value=0, vary=False)
            pars.add('y2', value=0, vary=False)
            if fa1 != 0 and fa2 != 0:
                fitter = Minimizer(fgl2_a, pars, fcn_args=(xx, yy-lnr_bg(yy)))
                result = fitter.minimize()
            else:
                fitter = Minimizer(fgl2, pars, fcn_args=(xx, yy-lnr_bg(yy)))
                result = fitter.minimize()
            a2 = toa2()
            checkfit()
            if fit_warn == 1:
                t = 5
                while t > 0 and fit_warn == 1:
                    result = fitter.minimize()
                    a2 = toa2()
                    checkfit()
                    t -= 1
        report_fit(result)
        mresult[i] = result
        # p0=[kmin[i]+(kmax[i]-kmin[i])*0.3,(np.max(y)-mbase[i])+1,0.1,0,kmax[i]-(kmax[i]-kmin[i])*0.3,(np.max(y)-mbase[i])+1,0.1,0]
        # a2,b=curve_fit(gl2,xx,yy-lnr_bg(yy),p0=p0,bounds=([kmin[i],(np.max(y)-mbase[i])/10,0,0,kmin[i],(np.max(y)-mbase[i])/10,0,0],[kmax[i],np.max(y)-mbase[i]+1,0.3,0.01,kmax[i],np.max(y)-mbase[i]+1,0.3,0.01]))

        if (kmin[i], kmax[i]) == ((2*m*ev[i]*1.6*10**-19)**0.5*np.sin(-0.5/180*np.pi)*10**-10/(h/2/np.pi), (2*m*ev[i]*1.6*10**-19)**0.5*np.sin(0.5/180*np.pi)*10**-10/(h/2/np.pi)):
            if i not in mfi_x:
                mfi_x.append(i)
            if i in mfi:
                mfi.remove(i)
            if i in mfi_err:
                mfi_err.remove(i)
        elif (kmin[i], kmax[i]) != ((2*m*ev[i]*1.6*10**-19)**0.5*np.sin(-0.5/180*np.pi)*10**-10/(h/2/np.pi), (2*m*ev[i]*1.6*10**-19)**0.5*np.sin(0.5/180*np.pi)*10**-10/(h/2/np.pi)):
            if fit_warn == 0:
                if i not in mfi:
                    mfi.append(i)
                if i in mfi_x:
                    mfi_x.remove(i)
                if i in mfi_err:
                    mfi_err.remove(i)
            else:
                if i not in mfi_err:
                    mfi_err.append(i)
                if i in mfi_x:
                    mfi_x.remove(i)
                if i in mfi:
                    mfi.remove(i)
            # if bg_warn==0:  #shirley base line warn
            #     if i not in mfi:
            #         mfi.append(i)
            #     if i in mfi_x:
            #         mfi_x.remove(i)
            #     if i in mfi_err:
            #         mfi_err.remove(i)
            # else:
            #     if i not in mfi_err:
            #         mfi_err.append(i)
            #     if i in mfi_x:
            #         mfi_x.remove(i)
            #     if i in mfi:
            #         mfi.remove(i)
    except RuntimeError:
        if i not in mfi_err:
            mfi_err.append(i)
        if i in mfi_x:
            mfi_x.remove(i)
        if i in mfi:
            mfi.remove(i)
        a1 = [(kmin[i]+kmax[i])/2, (np.max(y)-mbase[i]), 0.5, mbase[i]]
        a2 = [(kmin[i]+kmax[i])/2, (np.max(y)-mbase[i]), 0.5, mbase[i],
              (kmin[i]+kmax[i])/2, (np.max(y)-mbase[i]), 0.5, mbase[i]]

    # pos[i]=a[0]
    # fwhm[i]=a[2]

    fmxx[i, :len(xx)] = xx
    fmyy[i, :len(yy)] = yy
    fmx[i, :] = x
    fmy[i, :] = y
    mvv[i] = ev[i]
    if mfp[i] == 1:
        maa1[i, :] = a1
    elif mfp[i] == 2:
        maa2[i, :] = a2


def fmrmv():
    global mbrmv, flmrmv, mirmv, kmin, kmax, mfi, mfi_err, mfi_x, cki, mfp, mresult, smresult
    i = mfiti.get()
    flmrmv *= -1
    if flmrmv == 1:
        mirmv = i
        mbrmv.config(text='End Remove', bg='red')
    else:
        ti = sorted([i, mirmv])
        for i in np.linspace(ti[0], ti[1], ti[1]-ti[0]+1, dtype=int):
            mfp[i] = 1
            kmin[i], kmax[i] = (2*m*ev[i]*1.6*10**-19)**0.5*np.sin(-0.5/180*np.pi)*10**-10/(
                h/2/np.pi), (2*m*ev[i]*1.6*10**-19)**0.5*np.sin(0.5/180*np.pi)*10**-10/(h/2/np.pi)
            if i not in mfi_x:
                mfi_x.append(i)
            if i in mfi:
                mfi.remove(i)
            if i in mfi_err:
                mfi_err.remove(i)
            if i in cki:
                cki.remove(i)
            for ii in range(len(smresult[i])):
                    smresult[i][ii]=''
            mresult[i]=[]
        mplfi()
        mbrmv.config(text='Start Remove', bg='white')
        mfitplot()


def fmedmove(event):
    global medxdata, medydata, medfitout
    if event.xdata != None:
        medfitout.get_tk_widget().config(cursor="crosshair")
        medxdata.config(text='xdata:'+str('%.3f' % event.xdata))
        medydata.config(text='ydata:'+str('%.3f' % event.ydata))
    else:
        medfitout.get_tk_widget().config(cursor="")
        try:
            medxdata.config(text='xdata:')
            medydata.config(text='ydata:')
        except NameError:
            pass


def savemfit():
    path = fd.asksaveasfilename(title="Save MDC Fitted Data", initialdir=rdd,
                                initialfile=name+"_mfit.npz", filetype=[("NPZ files", ".npz"),])
    if len(path) > 2:
        np.savez(path, path=rdd, fev=fev, fwhm=fwhm, pos=pos, skmin=skmin,
                 skmax=skmax, smaa1=smaa1, smaa2=smaa2, smfp=smfp, smfi=smfi, smresult=pack_fitpar(mresult))


scki = []


def fmend():
    global rpos, pos, fwhm, fev, medxdata, medydata, medfitout, skmin, skmax, smaa1, smaa2, smfp, smfi, fpr, scki
    fev, pos, fwhm = [], [], []
    skmin, skmax, smaa1, smaa2 = kmin, kmax, maa1, maa2
    smfp = mfp
    smfi = mfi
    for i, v in enumerate(mfi):
        if mfp[v] == 1:
            fev.append(ev[v])
            pos.append(maa1[v, 0])
            fwhm.append(maa1[v, 2])
        elif mfp[v] == 2:
            fev.append(ev[v])
            fev.append(ev[v])
            pos.append(maa2[v, 0])
            pos.append(maa2[v, 4])
            fwhm.append(maa2[v, 2])
            fwhm.append(maa2[v, 6])
    rpos, fev, pos, fwhm = np.float64(pos), np.float64(
        fev), np.float64(pos), np.float64(fwhm)
    scki = cki
    fpr = 1
    mendg = tk.Toplevel(g)
    mendg.title('MDC Lorentz Fit Result')
    fr = tk.Frame(master=mendg, bd=5)
    fr.grid(row=0, column=0)
    mfitfig = Figure(figsize=(8, 6), layout='constrained')
    medfitout = tkagg.FigureCanvasTkAgg(mfitfig, master=fr)
    medfitout.get_tk_widget().grid(row=0, column=0)
    medfitout.mpl_connect('motion_notify_event', fmedmove)

    a = mfitfig.subplots()
    a.scatter(pos+fwhm/2, fev, c='r', s=10)
    a.scatter(pos-fwhm/2, fev, c='r', s=10)
    a.scatter(pos, fev, c='k', s=10)
    a.set_xlabel(r'k ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=14)
    a.set_ylabel('Kinetic Energy (eV)', font='Arial', fontsize=14)
    medfitout.draw()
    xydata = tk.Frame(master=fr, bd=5)
    xydata.grid(row=1, column=0)

    medxdata = tk.Label(xydata, text='xdata:', font=(
        "Arial", 12, "bold"), width='15', height='1', bd=10, bg='white')
    medxdata.grid(row=0, column=0)
    medydata = tk.Label(xydata, text='ydata:', font=(
        "Arial", 12, "bold"), width='15', height='1', bd=10, bg='white')
    medydata.grid(row=0, column=1)

    bsave = tk.Button(master=mendg, text='Save Fitted Data', command=savemfit,
                      width=30, height=2, font={'Arial', 18, "bold"}, bg='white', bd=10)
    bsave.grid(row=1, column=0)

    mendg.update()


def fmfall():
    t = threading.Thread(target=mfitjob)
    t.daemon = True
    t.start()


def func_cki():
    global cki, kmin, kmax
    if mfiti.get() not in cki:
        cki.append(mfiti.get())
    if len(cki) >= 2:
        cki.sort()
        for i in range(len(cki)-1):
            kmin[cki[i]:cki[i+1] +
                 1] = np.linspace(kmin[cki[i]], kmin[cki[i+1]], cki[i+1]-cki[i]+1)
            kmax[cki[i]:cki[i+1] +
                 1] = np.linspace(kmax[cki[i]], kmax[cki[i+1]], cki[i+1]-cki[i]+1)


def fchki(*e):
    global mfitout, mdxdata, mdydata, mbcomp1, mbcomp2
    i = mfiti.get()
    try:
        mfitout.get_tk_widget().delete('rec')
        mdxdata.config(text='dx:')
        mdydata.config(text='dy:')
        if mfp[i] == 2:
            mbcomp1.config(state='active')
            mbcomp2.config(state='active')
        else:
            mbcomp1.config(state='disabled')
            mbcomp2.config(state='disabled')
    except:
        pass
    mfitplot()


def mplfi():
    global miout, mifig, mlind, mrind
    i = mfiti.get()
    mifig.clear()
    miax = mifig.add_axes([0, 0, 1, 1])
    miax.scatter(mfi_x, [0 for i in range(len(mfi_x))], marker='|', c='k')
    miax.scatter(mfi, [0 for i in range(len(mfi))], marker='|', c='b')
    miax.scatter(mfi_err, [0 for i in range(len(mfi_err))], marker='|', c='r')
    if i in mfi_x:
        mlind.config(bg='white')
        mrind.config(bg='white')
    if i in mfi:
        mlind.config(bg='blue')
        mrind.config(bg='blue')
    if i in mfi_err:
        mlind.config(bg='red')
        mrind.config(bg='red')
    try:
        miax.set_xlim([np.min([mfi, mfi_x, mfi_err]),
                      np.max([mfi, mfi_x, mfi_err])])
    except ValueError:
        pass
    miax.set_yticks([])
    miout.draw()


def mfitplot():  # mfiti Scale
    global mfitax, mxl, myl, klmin, klmax, tmxl, kmin, kmax, maa2, flmcomp, lm1, lm2, lm3, lm4, lm5, lm6
    i = mfiti.get()
    mfitfig.clear()
    mfitax = mfitfig.subplots()
    # 'Pos:'+str(round(maa1[i,0],3))+r' $(\frac{2\pi}{\AA})$'+', FWHM:'+str(round(maa1[i,2],3))+r' $(\frac{2\pi}{\AA})$'
    mfitax.set_title('Kinetic Energy:' +
                     str(round(mvv[i], 3))+' eV, '+str(mfp[i])+' Peak')
    mfitax.scatter(fmx[i, :], fmy[i, :], c='k', s=4)
    tyl = mfitax.get_ylim()
    txl = mfitax.get_xlim()
    dy = (tyl[1]-tyl[0])/20
    dx = (txl[1]-txl[0])/50
    tymin = tyl[0]
    tymax = tyl[1]
    txmin = txl[0]
    txmax = txl[1]
    mfitax.axhline(tymax+dy, c='grey')
    x = fmxx[i, np.argwhere(fmxx[i, :] >= -20)].flatten()
    y = fmyy[i, np.argwhere(fmxx[i, :] >= -20)].flatten()
    lbg = lnr_bg(y)
    if i in mfi_x:
        for l, v in zip([lm1, lm2, lm3, lm4, lm5, lm6], ['', '', '', '', '', '']):
            l.config(text=v)
    if mfp[i] == 1:
        if maa1[i, 0] == (kmin[i]+kmax[i])/2 and maa1[i, 2] == 0.5:
            fl, = mfitax.plot(x, gl1(x, *maa1[i, :])+lbg, 'r-', lw=2)
        else:
            gl1_1 = gl1(x, *maa1[i, :])+lbg
            fl, = mfitax.plot(x, gl1(x, *maa1[i, :])+lbg, 'b-', lw=2)
            mfitax.fill_between(x, lbg, gl1_1, facecolor='blue', alpha=0.5)
        if i in mfi_err or i in mfi:
            if i in mfi:
                mfitax.plot(x, gl1(x, *maa1[i, :]) +
                            lbg-y+tymax+dy, color='gray', lw=1)
            else:
                mfitax.plot(x, gl1(x, *maa1[i, :]) +
                            lbg-y+tymax+dy, color='red', lw=1)

            # s=(np.sum((gl1(x,*maa1[i,:])+lbg-y)**2)/(max(x)-min(x)))**0.5
            s = np.std(gl1(x, *maa1[i, :])+lbg-y)  # STD
            mfitax.text(txmin+dx, tymax-dy, 'Residual STD: '+str(round(s, 2)))
            s = np.sqrt(np.mean((gl1(x, *maa1[i, :])+lbg-y)**2))  # RMS
            mfitax.text(txmin+dx, tymax-2*dy,
                        'Residual RMS: '+str(round(s, 2)))
            ty = gl1(x, *maa1[i, :])
            s = np.sum(np.array([((ty[i]+ty[i+1])/2)for i in range(len(x)-1)])
                       # Area
                       * np.array(([(x[i+1]-x[i])for i in range(len(x)-1)])))
            mfitax.text(txmin+dx, tymax-3*dy, 'Area: '+str(round(s, 2)))
            vv = []
            for ii in range(6):
                if ii > 2:
                    vv.append(f"")
                else:
                    vv.append(f"{gformat(maa1[i, ii])}")
                    
            for l, n, v in zip([lm1, lm2, lm3, lm4, lm5, lm6], [f"x: ", f"h: ", f"w: ", f"", f"", f""], vv):
                l.config(text=n+v)
                l.config(anchor='center')
            try:
                vv=smresult[i]
                for l, v in zip([lm1, lm2, lm3, lm4, lm5, lm6], vv):
                    if 'nofit' not in v:
                        l.config(text=v)
                        l.config(anchor='w')
            except:
                pass
            try:
                fitpar1(mresult[i], lm1, lm2, lm3, lm4, lm5, lm6)
            except:
                pass
    elif mfp[i] == 2:
        flmcomp = 0
        if maa2[i, 0] == (kmin[i]+kmax[i])/2 and maa2[i, 2] == 0.5:
            fl, = mfitax.plot(x, gl2(x, *maa2[i, :])+lbg, 'r-', lw=2)
        else:
            if flmcomp1 == 1:
                maa2[i, :4] = [
                    mcpx1, mcpy1-lbg[np.argwhere(abs(x-mcpx1) < 0.01)].flatten()[0], 0.02, 0]
                flmcomp = 1
            elif flmcomp2 == 1:
                maa2[i, -4:] = [mcpx2, mcpy2 -
                                lbg[np.argwhere(abs(x-mcpx2) < 0.01)].flatten()[0], 0.02, 0]
                flmcomp = 1
            gl2_1 = gl1(x, *maa2[i, :4])+lbg
            gl2_2 = gl1(x, *maa2[i, -4:])+lbg
            fl, = mfitax.plot(x, gl2(x, *maa2[i, :])+lbg, 'b-', lw=2)
            mfitax.fill_between(x, lbg, gl2_1, facecolor='green', alpha=0.5)
            mfitax.fill_between(x, lbg, gl2_2, facecolor='yellow', alpha=0.5)
        if i in mfi_err or i in mfi:
            if i in mfi:
                mfitax.plot(x, gl2(x, *maa2[i, :]) +
                            lbg-y+tymax+dy, color='gray', lw=1)
            else:
                mfitax.plot(x, gl2(x, *maa2[i, :]) +
                            lbg-y+tymax+dy, color='red', lw=1)
            # s=(np.sum((gl2(x,*maa2[i,:])+lbg-y)**2)/(max(x)-min(x)))**0.5
            s = np.std(gl2(x, *maa2[i, :])+lbg-y)  # STD
            mfitax.text(txmin+dx, tymax-dy, 'Residual STD: '+str(round(s, 2)))
            s = np.sqrt(np.mean((gl2(x, *maa2[i, :])+lbg-y)**2))  # RMS
            mfitax.text(txmin+dx, tymax-2*dy,
                        'Residual RMS: '+str(round(s, 2)))
            ty = gl1(x, *maa2[i, :4])
            s = np.sum(np.array([((ty[i]+ty[i+1])/2)for i in range(len(x)-1)])
                       # Area 1
                       * np.array(([(x[i+1]-x[i])for i in range(len(x)-1)])))
            mfitax.text(txmin+dx, tymax-3*dy, 'Area 1: '+str(round(s, 2)))
            ty = gl1(x, *maa2[i, -4:])
            s = np.sum(np.array([((ty[i]+ty[i+1])/2)for i in range(len(x)-1)])
                       # Area 2
                       * np.array(([(x[i+1]-x[i])for i in range(len(x)-1)])))
            mfitax.text(txmin+dx, tymax-4*dy, 'Area 2: '+str(round(s, 2)))
            vv = []
            for ii in range(6):
                if ii < 3:
                    vv.append(f"{gformat(maa2[i, ii])}")
                else:
                    vv.append(f"{gformat(maa2[i, ii+1])}")

            for l, n, v in zip([lm1, lm3, lm5, lm2, lm4, lm6], [f"x1: ", f"h1: ", f"w1: ", f"x2: ", f"h2: ", f"w2: "], vv):
                l.config(text=n+v)
                l.config(anchor='center')
            try:
                vv=smresult[i]
                for l, v in zip([lm1, lm2, lm3, lm4, lm5, lm6], vv):
                    if 'nofit' not in v:
                        l.config(text=v)
                        l.config(anchor='w')
            except:
                pass
            try:
                fitpar2(mresult[i], lm1, lm2, lm3, lm4, lm5, lm6)
            except:
                pass
    mfitax.plot(fmxx[i, np.argwhere(fmxx[i, :] >= -20)], lbg, 'g--')
    # if bg_warn==1:  #shirley base line warn
    #     mfitax.plot(fmxx[i,np.argwhere(fmxx[i,:]>=-20)],lbg,'r--')
    # else:
    #     mfitax.plot(fmxx[i,np.argwhere(fmxx[i,:]>=-20)],lbg,'g--')

    mfitax.scatter(fmxx[i, np.argwhere(fmxx[i, :] >= -20)], y, c='g', s=4)
    if (kmin[i], kmax[i]) != ((2*m*ev[i]*1.6*10**-19)**0.5*np.sin(-0.5/180*np.pi)*10**-10/(h/2/np.pi), (2*m*ev[i]*1.6*10**-19)**0.5*np.sin(0.5/180*np.pi)*10**-10/(h/2/np.pi)):
        klmin = mfitax.axvline(kmin[i], c='r')
        klmax = mfitax.axvline(kmax[i], c='r')
    else:
        klmin = mfitax.axvline(kmin[i], c='grey')
        klmax = mfitax.axvline(kmax[i], c='grey')
        fl.set_alpha(0.3)
    mfitax.set_xlabel(r'k ($\frac{2\pi}{\AA}$)', fontsize=14)
    mfitax.set_ylabel('Intensity (Counts)', fontsize=14)
    mxl = mfitax.get_xlim()
    myl = mfitax.get_ylim()
    tmxl = np.copy(mxl)
    mfitout.draw()
    mplfi()


def mmove(event):
    global mxdata, mydata, mdxdata, mdydata, x2, y2, mfitax, mfitout, klmin, klmax, kmin, kmax, tpx1, tpx2, tpy1, tpy2, tx2, ty2, mcpx1, mcpy1, mcpx2, mcpy2
    if event.xdata != None:
        # mfitout.get_tk_widget().config(cursor="crosshair")
        # try:
        #     # mfitout.get_tk_widget().delete('rec')
        #     # mfitout.get_tk_widget().delete('x1')
        #     # mfitout.get_tk_widget().delete('y1')
        #     # mfitout.get_tk_widget().delete('x2')
        #     # mfitout.get_tk_widget().delete('y2')
        # except:
        #     pass
        if mmof == -1:
            x2, y2 = event.xdata, event.ydata
            px2, py2 = event.x, event.y
            if flmcomp1 == 1:
                mcpx1 = x2
                mcpy1 = y2
                mfitplot()
            elif flmcomp2 == 1:
                mcpx2 = x2
                mcpy2 = y2
                mfitplot()
            elif fklmin == 1 and tkmin+(x2-x1) >= mxl[0] and tkmin+(x2-x1) <= mxl[1]:
                klmin.remove()
                klmin = mfitax.axvline(x2, c='r')
                kmin[mfiti.get()] = x2
                klmax.set_color('r')
                mfitout.draw()
            elif fklmax == 1 and tkmax+(x2-x1) >= mxl[0] and tkmax+(x2-x1) <= mxl[1]:
                klmax.remove()
                klmax = mfitax.axvline(x2, c='r')
                kmax[mfiti.get()] = x2
                klmin.set_color('r')
                mfitout.draw()
            elif fkregion == 1 and tkmin+(x2-x1) >= mxl[0] and tkmax+(x2-x1) <= mxl[1]:
                klmin.remove()
                klmin = mfitax.axvline(tkmin+(x2-x1), c='r')
                kmin[mfiti.get()] = tkmin+(x2-x1)
                klmax.remove()
                klmax = mfitax.axvline(tkmax+(x2-x1), c='r')
                kmax[mfiti.get()] = tkmax+(x2-x1)
                mfitout.draw()
            elif fklmin == 0 and fklmax == 0 and fkregion == 0:
                mfitout.get_tk_widget().delete('rec')
                tpx1, tpy1, tpx2, tpy2 = px1, py1, px2, py2
                mfitout.get_tk_widget().create_rectangle(
                    (px1, 600-py1), (px2, 600-py2), outline='grey', width=2, tag='rec')
                [tpx1, tpx2] = sorted([tpx1, tpx2])
                [tpy1, tpy2] = sorted([tpy1, tpy2])
                tx2, ty2 = x2, y2
                mdxdata.config(text='dx:'+str('%.3f' % abs(x2-x1)))
                mdydata.config(text='dy:'+str('%.3f' % abs(y2-y1)))
        mxdata.config(text='xdata:'+str('%.3f' % event.xdata))
        mydata.config(text='ydata:'+str('%.3f' % event.ydata))
    else:
        mfitout.get_tk_widget().config(cursor="")
        try:
            mxdata.config(text='xdata:')
            mydata.config(text='ydata:')
        except NameError:
            pass

    # print("event.xdata", event.xdata)
    # print("event.ydata", event.ydata)
    # print("event.inaxes", event.inaxes)
    # print("x", event.x)
    # print("y", event.y)
mmof = 1


def mpress(event):
    # event.button 1:left 3:right 2:mid
    # event.dblclick : bool
    # print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
    #       ('double' if event.dblclick else 'single', event.button,
    #        event.x, event.y, event.xdata, event.ydata))
    global x1, y1, mmof, px1, py1, mfitax, mfitout, klmin, klmax, fklmin, fklmax, cki, mbase, fkregion, tkmin, tkmax, tx1, ty1
    if event.button == 1 and event.inaxes:
        x1, y1 = event.xdata, event.ydata
        px1, py1 = event.x, event.y
        fklmin, fklmax, fkregion = 0, 0, 0
        tkmin, tkmax = kmin[mfiti.get()], kmax[mfiti.get()]
        if flmcomp1 == 1:
            pass
        elif flmcomp2 == 1:
            pass
        elif mfitout.get_tk_widget().find_withtag('rec') != () and px1 > tpx1 and px1 < tpx2 and py1 > tpy1 and py1 < tpy2:
            pass
        elif abs(x1-kmin[mfiti.get()]) < (tmxl[1]-tmxl[0])/80:
            fklmin = 1

        elif abs(x1-kmax[mfiti.get()]) < (tmxl[1]-tmxl[0])/80:
            fklmax = 1

        elif x1 > kmin[mfiti.get()] and x1 < kmax[mfiti.get()]:
            fkregion = 1

        elif mfitout.get_tk_widget().find_withtag('rec') == ():
            tx1, ty1 = x1, y1
        mmof = -1
    elif event.button == 3:
        try:
            mfitout.get_tk_widget().delete('rec')
            mdxdata.config(text='dx:')
            mdydata.config(text='dy:')
        except:
            pass
        mfitax.set_xlim(mxl)
        mfitax.set_ylim(myl)
        mfitout.draw()
        mmof = 1


def mrelease(event):
    global x1, y1, x2, y2, mmof, mfitout, mfitax, fklmax, fklmin, klmin, klmax, kmin, kmax, fkregion, tmxl
    if event.button == 1 and mmof == -1 and event.inaxes:
        x2, y2 = event.xdata, event.ydata
        if kmin[mfiti.get()] > kmax[mfiti.get()]:
            kmin[mfiti.get()], kmax[mfiti.get()
                                    ] = kmax[mfiti.get()], kmin[mfiti.get()]
            klmin, klmax = klmax, klmin
        else:
            kmin[mfiti.get()], kmax[mfiti.get()
                                    ] = kmin[mfiti.get()], kmax[mfiti.get()]
            klmin, klmax = klmin, klmax
        if fklmin == 0 and fklmax == 0 and fkregion == 0 and (x2, y2) == (x1, y1) and px1 > tpx1 and px1 < tpx2 and py1 > tpy1 and py1 < tpy2:
            try:
                mfitout.get_tk_widget().delete('rec')
            except:
                pass
            mfitax.set_xlim(sorted([tx1, tx2]))
            mfitax.set_ylim(sorted([ty1, ty2]))
            tmxl = sorted([x1, x2])
            mfitout.draw()
        elif fklmin == 1 or fklmax == 1 or fkregion == 1:
            func_cki()
            x1, x2, y1, y2 = [], [], [], []
            mfit()
            mfitplot()
        mmof = 1


def tmstate():
    try:
        while True:
            mstate.config(text=str(mst.get()))
    except KeyboardInterrupt:
        pass


def mflind():
    global mfiti
    ti = mfiti.get()
    if ti in mfi:
        for i in range(ti+1):
            if ti-i not in mfi:
                mfiti.set(ti-i)
                break
    elif ti in mfi_err:
        for i in range(ti+1):
            if ti-i not in mfi_err:
                mfiti.set(ti-i)
                break
    elif ti in mfi_x:
        for i in range(ti+1):
            if ti-i in mfi or ti-i in mfi_err:
                mfiti.set(ti-i)
                break
        if i == ti and ti != 0:
            mfiti.set(ti-1)


def mfrind():
    global mfiti
    ti = mfiti.get()
    if ti in mfi:
        for i in range(len(ev)-ti):
            if ti+i not in mfi:
                mfiti.set(ti+i)
                break
    elif ti in mfi_err:
        for i in range(len(ev)-ti):
            if ti+i not in mfi_err:
                mfiti.set(ti+i)
                break
    elif ti in mfi_x:
        for i in range(len(ev)-ti):
            if ti+i in mfi or ti+i in mfi_err:
                mfiti.set(ti+i)
                break
        if i == len(ev)-ti-1 and ti != len(ev)-1:
            mfiti.set(ti+1)


def mfcomp1():
    global mbcomp1, flmcomp1, mbcomp2, flmcomp2
    if flmcomp2 == -1:
        flmcomp1 *= -1
        if flmcomp1 == 1:
            mbcomp1.config(text='Comp 1', bg='green')
        else:
            mbcomp1.config(text='Comp 1', bg='white')
    else:
        flmcomp2 *= -1
        flmcomp1 *= -1
        mbcomp1.config(text='Comp 1', bg='green')
        mbcomp2.config(text='Comp 2', bg='white')


def mfcomp2():
    global mbcomp1, flmcomp1, mbcomp2, flmcomp2
    if flmcomp1 == -1:
        flmcomp2 *= -1
        if flmcomp2 == 1:
            mbcomp2.config(text='Comp 2', bg='yellow')
        else:
            mbcomp2.config(text='Comp 2', bg='white')
    else:
        flmcomp1 *= -1
        flmcomp2 *= -1
        mbcomp2.config(text='Comp 2', bg='yellow')
        mbcomp1.config(text='Comp 1', bg='white')


def ffitcp():
    mfit()
    mfitplot()


def fmaccept():
    global mfi, mfi_x, mfi_err
    i = mfiti.get()
    if i not in mfi:
        mfi.append(i)
    if i in mfi_x:
        mfi_x.remove(i)
    if i in mfi_err:
        mfi_err.remove(i)
    mplfi()


def fmreject():
    global mfi, mfi_x, mfi_err
    i = mfiti.get()
    if i not in mfi_x:
        mfi_x.append(i)
    if i in mfi:
        mfi.remove(i)
    if i in mfi_err:
        mfi_err.remove(i)
    mplfi()


def o_fmwf1(*e):
    global mwf1
    if '' == mwf1.get():
        mwf1.set('0')
        min_w1.select_range(0, 1)


def fmwf1(*e):
    t = threading.Thread(target=o_fmwf1)
    t.daemon = True
    t.start()


def o_fmwf2(*e):
    global mwf2
    if '' == mwf2.get():
        mwf2.set('0')
        min_w2.select_range(0, 1)


def fmwf2(*e):
    t = threading.Thread(target=o_fmwf2)
    t.daemon = True
    t.start()


def o_fmaf1(*e):
    global maf1
    if '' == maf1.get():
        maf1.set('0')
        min_a1.select_range(0, 1)


def fmaf1(*e):
    t = threading.Thread(target=o_fmaf1)
    t.daemon = True
    t.start()


def o_fmaf2(*e):
    global maf2
    if '' == maf2.get():
        maf2.set('0')
        min_a2.select_range(0, 1)


def fmaf2(*e):
    t = threading.Thread(target=o_fmaf2)
    t.daemon = True
    t.start()


def o_fmxf1(*e):
    global mxf1
    if '' == mxf1.get():
        mxf1.set('1')
        min_x1.select_range(0, 1)


def fmxf1(*e):
    t = threading.Thread(target=o_fmxf1)
    t.daemon = True
    t.start()


def o_fmxf2(*e):
    global mxf2
    if '' == mxf2.get():
        mxf2.set('0')
        min_x2.select_range(0, 1)


def fmxf2(*e):
    t = threading.Thread(target=o_fmxf2)
    t.daemon = True
    t.start()


def fmposcst():
    global mbposcst, flmposcst, min_x1, min_x2
    flmposcst *= -1
    if flmposcst == 1:
        min_x1.config(state='normal')
        min_x2.config(state='normal')
        mbposcst.config(bg='yellow')
    else:
        min_x1.config(state='disabled')
        min_x2.config(state='disabled')
        mbposcst.config(bg='white')


def mjob():     # MDC Fitting GUI
    global g, mfiti, mfitfig, mfitout, mgg, mxdata, mydata, mdxdata, mdydata, miout, mifig, mfi, mfi_err, mfi_x, mbrmv, flmrmv, mbcgl2, mfp, flmcgl2, fpr, mst, mstate, mwf1, mwf2, maf1, maf2, mxf1, mxf2, mlind, mrind, mbcomp1, flmcomp1, mbcomp2, flmcomp2, min_w1, min_w2, min_a1, min_a2, min_x1, min_x2, lm1, lm2, lm3, lm4, lm5, lm6, mresult, mbposcst, flmposcst
    mgg = tk.Toplevel(g, bg='white')
    mgg.title('MDC Lorentz Fit')
    mst = queue.Queue(maxsize=0)
    mstate = tk.Label(mgg, text='', font=(
        "Arial", 20, "bold"), bg="white", fg="black")
    mstate.grid(row=0, column=0)
    fr = tk.Frame(master=mgg, bg='white')
    fr.grid(row=1, column=0)
    frind = tk.Frame(master=fr, bg='white')
    frind.grid(row=0, column=0)
    mlind = tk.Button(frind, text='<<', command=mflind, width=10,
                      height=5, font={'Arial', 50, "bold"}, bg='white')
    mlind.grid(row=0, column=0)
    mrind = tk.Button(frind, text='>>', command=mfrind, width=10,
                      height=5, font={'Arial', 50, "bold"}, bg='white')
    mrind.grid(row=0, column=2)

    mfiti = tk.IntVar()
    mfiti.set(0)
    mfiti.trace_add('write', fchki)
    chi = tk.Scale(frind, label='Index', from_=0, to=len(ev)-1, orient='horizontal',
                   variable=mfiti, state='active', bg='white', fg='black', length=580, width=50, resolution=1)
    chi.grid(row=0, column=1)

    mfi, mfi_err, mfi_x = [], [], [i for i in range(len(ev))]
    mifig = Figure(figsize=(6, 0.2), layout='tight')
    miout = tkagg.FigureCanvasTkAgg(mifig, master=frind)
    miout.get_tk_widget().grid(row=1, column=1)

    mfitfig = Figure(figsize=(8, 6), layout='constrained')
    mfitout = tkagg.FigureCanvasTkAgg(mfitfig, master=fr)
    mfitout.get_tk_widget().grid(row=1, column=0)
    mfitout.mpl_connect('motion_notify_event', mmove)
    mfitout.mpl_connect('button_press_event', mpress)
    mfitout.mpl_connect('button_release_event', mrelease)

    xydata = tk.Frame(master=fr, bd=5, bg='white')
    xydata.grid(row=2, column=0)

    mxdata = tk.Label(xydata, text='xdata:', font=(
        "Arial", 12, "bold"), width='15', height='1', bd=10, bg='white')
    mxdata.grid(row=0, column=0)
    mydata = tk.Label(xydata, text='ydata:', font=(
        "Arial", 12, "bold"), width='15', height='1', bd=10, bg='white')
    mydata.grid(row=0, column=1)
    mdxdata = tk.Label(xydata, text='dx:', font=(
        "Arial", 12, "bold"), width='15', height='1', bd=10, bg='white')
    mdxdata.grid(row=0, column=2)
    mdydata = tk.Label(xydata, text='dy:', font=(
        "Arial", 12, "bold"), width='15', height='1', bd=10, bg='white')
    mdydata.grid(row=0, column=3)

    # bstop=tk.Button(gg,command=stop,text='Stop',font=('Arial',20),bd=10)
    # bstop.grid(row=1,column=0)

    frpara = tk.Frame(master=mgg, bd=5, bg='white')
    frpara.grid(row=1, column=1)
    try:
        if fpr == 1:
            mfp = list(smfp)
            mfi = list(smfi)
        else:
            mfp = [1 for i in range(len(ev))]
    except:
        mfp = [1 for i in range(len(ev))]
    flmcgl2 = -1
    frpara00 = tk.Frame(master=frpara, bd=5, bg='white')
    frpara00.grid(row=0, column=0)

    frfitpar = tk.Frame(master=frpara00, bd=5, bg='white')
    frfitpar.grid(row=0, column=0)
    lm1 = tk.Label(frfitpar, anchor='w', text='', font=(
        "Arial", 16, "bold"), width='55', height='1', bd=5, bg='white')
    lm1.grid(row=0, column=0)
    lm2 = tk.Label(frfitpar, anchor='w', text='', font=(
        "Arial", 16, "bold"), width='55', height='1', bd=5, bg='white')
    lm2.grid(row=1, column=0)
    lm3 = tk.Label(frfitpar, anchor='w', text='', font=(
        "Arial", 16, "bold"), width='55', height='1', bd=5, bg='white')
    lm3.grid(row=2, column=0)
    lm4 = tk.Label(frfitpar, anchor='w', text='', font=(
        "Arial", 16, "bold"), width='55', height='1', bd=5, bg='white')
    lm4.grid(row=3, column=0)
    lm5 = tk.Label(frfitpar, anchor='w', text='', font=(
        "Arial", 16, "bold"), width='55', height='1', bd=5, bg='white')
    lm5.grid(row=4, column=0)
    lm6 = tk.Label(frfitpar, anchor='w', text='', font=(
        "Arial", 16, "bold"), width='55', height='1', bd=5, bg='white')
    lm6.grid(row=5, column=0)

    frYN = tk.Frame(master=frfitpar, bd=5, bg='white')
    frYN.grid(row=6, column=0)
    mbaccept = tk.Button(frYN, text='Accept', command=fmaccept,
                         width=30, height=1, font={'Arial', 24, "bold"}, bg='white')
    mbaccept.grid(row=0, column=0)
    mbreject = tk.Button(frYN, text='Reject', command=fmreject,
                         width=30, height=1, font={'Arial', 24, "bold"}, bg='white')
    mbreject.grid(row=0, column=1)

    l1 = tk.Label(frpara00, text='Index Operation', font=(
        "Arial", 12, "bold"), width='15', height='1', bd=5, bg='white')
    l1.grid(row=1, column=0)
    froperind = tk.Frame(master=frpara00, bd=5, bg='white')
    froperind.grid(row=2, column=0)
    mbcgl2 = tk.Button(froperind, text='Start Add 2 Peaks', command=fmcgl2,
                       width=30, height=1, font={'Arial', 18, "bold"}, bg='white')
    mbcgl2.grid(row=0, column=0)
    mbrmv = tk.Button(froperind, text='Start Remove', command=fmrmv,
                      width=30, height=1, font={'Arial', 18, "bold"}, bg='white')
    mbrmv.grid(row=0, column=1)
    mbcomp1 = tk.Button(froperind, text='Comp 1', command=mfcomp1,
                        width=20, height=1, font={'Arial', 18, "bold"}, bg='white')
    mbcomp1.grid(row=1, column=0)
    mbcomp2 = tk.Button(froperind, text='Comp 2', command=mfcomp2,
                        width=20, height=1, font={'Arial', 18, "bold"}, bg='white')
    mbcomp2.grid(row=1, column=1)

    mbfitcp = tk.Button(master=frpara00, text='Fit Components', command=ffitcp,
                        width=40, height=1, font={'Arial', 18, "bold"}, bg='white')
    mbfitcp.grid(row=3, column=0)

    frwr = tk.Frame(master=frpara00, bd=5, bg='white')
    frwr.grid(row=4, column=0)
    l2 = tk.Label(frwr, text='FWHM Ratio', font=(
        "Arial", 12, "bold"), width='15', height='1', bd=5, bg='white')
    l2.grid(row=0, column=1)
    l3 = tk.Label(frwr, text=':', font=("Arial", 12, "bold"),
                  width='15', height='1', bd=5, bg='white')
    l3.grid(row=1, column=1)
    mwf1 = tk.StringVar()
    mwf1.set('0')
    mwf1.trace_add('write', fmwf1)
    min_w1 = tk.Entry(frwr, font=("Arial", 10, "bold"),
                      width=7, textvariable=mwf1, bd=5)
    min_w1.grid(row=1, column=0)
    mwf2 = tk.StringVar()
    mwf2.set('0')
    mwf2.trace_add('write', fmwf2)
    min_w2 = tk.Entry(frwr, font=("Arial", 10, "bold"),
                      width=7, textvariable=mwf2, bd=5)
    min_w2.grid(row=1, column=2)

    frar = tk.Frame(master=frpara00, bd=5, bg='white')
    frar.grid(row=5, column=0)
    l2 = tk.Label(frar, text='Area Ratio', font=(
        "Arial", 12, "bold"), width='15', height='1', bd=5, bg='white')
    l2.grid(row=0, column=1)
    l3 = tk.Label(frar, text=':', font=("Arial", 12, "bold"),
                  width='15', height='1', bd=5, bg='white')
    l3.grid(row=1, column=1)
    maf1 = tk.StringVar()
    maf1.set('0')
    maf1.trace_add('write', fmaf1)
    min_a1 = tk.Entry(frar, font=("Arial", 10, "bold"),
                      width=7, textvariable=maf1, bd=5)
    min_a1.grid(row=1, column=0)
    maf2 = tk.StringVar()
    maf2.set('0')
    maf2.trace_add('write', fmaf2)
    min_a2 = tk.Entry(frar, font=("Arial", 10, "bold"),
                      width=7, textvariable=maf2, bd=5)
    min_a2.grid(row=1, column=2)

    mbposcst = tk.Button(frpara00, text='Position constraint', command=fmposcst,
                         width=30, height=1, font={'Arial', 18, "bold"}, bg='white')
    mbposcst.grid(row=6, column=0)

    frxr = tk.Frame(master=frpara00, bd=5, bg='white', padx=30)
    frxr.grid(row=7, column=0)
    l3 = tk.Label(frxr, text='x2 =', font=("Arial", 12, "bold"),
                  width='5', height='1', bd=5, bg='white')
    l3.grid(row=0, column=0)
    mxf1 = tk.StringVar()
    mxf1.set('1')
    mxf1.trace_add('write', fmxf1)
    min_x1 = tk.Entry(frxr, font=("Arial", 10, "bold"), width=7,
                      textvariable=mxf1, bd=5, state='disabled')
    min_x1.grid(row=0, column=1)
    l3 = tk.Label(frxr, text='* x1 +', font=("Arial", 12, "bold"),
                  width='5', height='1', bd=5, bg='white')
    l3.grid(row=0, column=2)
    mxf2 = tk.StringVar()
    mxf2.set('0')
    mxf2.trace_add('write', fmxf2)
    min_x2 = tk.Entry(frxr, font=("Arial", 10, "bold"), width=7,
                      textvariable=mxf2, bd=5, state='disabled')
    min_x2.grid(row=0, column=3)

    frout = tk.Frame(master=mgg, bd=5, bg='white')
    frout.grid(row=2, column=0)
    bfall = tk.Button(frout, text='Fit All', command=fmfall,
                      width=30, height=1, font={'Arial', 18, "bold"}, bg='white')
    bfall.grid(row=0, column=0)

    flmposcst = -1
    flmrmv = -1
    flmcomp1 = -1
    flmcomp2 = -1

    bend = tk.Button(frout, text='Finish', command=fmend, width=30,
                     height=1, font={'Arial', 18, "bold"}, bg='white')
    bend.grid(row=1, column=0)

    mresult = [[]for i in range(len(ev))]
    if mprfit == 1:
        fmfall()
    else:
        mfitplot()
    tt = threading.Thread(target=tmstate)
    tt.daemon = True
    tt.start()
    mgg.update()


#################################### prefit ######################################################
mprfit = 0


def fitm():
    global ev, phi, data, mvv, maa1, maa2, fmxx, fmyy, fmx, fmy, kmin, kmax, cki, mbase, mprfit
    mprfit = 0
    cki = []
    mbase = [0 for i in range(len(ev))]

    if fpr == 1:
        try:
            kmin, kmax = skmin, skmax
        except NameError:
            kmin, kmax = np.float64((2*m*ev*1.6*10**-19)**0.5*np.sin(-0.5/180*np.pi)*10**-10/(
                h/2/np.pi)), np.float64((2*m*ev*1.6*10**-19)**0.5*np.sin(0.5/180*np.pi)*10**-10/(h/2/np.pi))
        if len(scki) >= 2:
            cki = scki
    else:
        kmin, kmax = np.float64((2*m*ev*1.6*10**-19)**0.5*np.sin(-0.5/180*np.pi)*10**-10/(
            h/2/np.pi)), np.float64((2*m*ev*1.6*10**-19)**0.5*np.sin(0.5/180*np.pi)*10**-10/(h/2/np.pi))
    # fmxx=np.float64((np.arange(len(phi)*len(ev))+1).reshape(len(ev),len(phi)))
    # fmyy=np.float64((np.arange(len(phi)*len(ev))+1).reshape(len(ev),len(phi)))
    # fmxx=fmxx/fmxx*-50
    # fmyy=fmyy/fmyy*-50
    fmxx = np.float64((np.ones(len(phi)*len(ev))).reshape(len(ev), len(phi)))
    fmyy = np.float64((np.ones(len(phi)*len(ev))).reshape(len(ev), len(phi)))
    fmxx *= -50
    fmyy *= -50
    fmx = np.float64(np.arange(len(phi)*len(ev)).reshape(len(ev), len(phi)))
    fmy = np.float64(np.arange(len(phi)*len(ev)).reshape(len(ev), len(phi)))
    mvv = np.float64(np.arange(len(ev)))
    maa1 = np.float64(np.arange(4*len(ev)).reshape(len(ev), 4))
    maa2 = np.float64(np.arange(8*len(ev)).reshape(len(ev), 8))
    pbar = tqdm.tqdm(total=len(ev), desc='MDC', colour='green')
    for i, v in enumerate(ev):
        ecut = data.sel(eV=v, method='nearest')
        x = np.float64((2*m*v*1.6*10**-19)**0.5 *
                       np.sin(phi/180*np.pi)*10**-10/(h/2/np.pi))
        y = ecut.to_numpy().reshape(len(x))
        tx = x[np.argwhere(x >= kmin[i])].flatten()
        xx = tx[np.argwhere(tx <= kmax[i])].flatten()
        ty = y[np.argwhere(x >= kmin[i])].flatten()
        yy = ty[np.argwhere(tx <= kmax[i])].flatten()
        yy = np.where(yy > int(base.get()), yy, int(base.get()))
        try:
            if i in smfi and fpr == 1:
                a1 = smaa1[i, :]
                a2 = smaa2[i, :]
                if smaa1[i, 1] == 10 or smaa2[i, 1] == 10:
                    mprfit = 1
            else:
                a1 = [(kmin[i]+kmax[i])/2, (np.max(y) -
                                            int(base.get())), 0.5, int(base.get())]
                a2 = [(kmin[i]+kmax[i])/2, (np.max(y)-int(base.get())), 0.5, int(base.get()),
                      (kmin[i]+kmax[i])/2, (np.max(y)-int(base.get())), 5, int(base.get())]
        except:
            a1 = [(kmin[i]+kmax[i])/2, (np.max(y) -
                                        int(base.get())), 0.5, int(base.get())]
            a2 = [(kmin[i]+kmax[i])/2, (np.max(y)-int(base.get())), 0.5, int(base.get()),
                  (kmin[i]+kmax[i])/2, (np.max(y)-int(base.get())), 0.5, int(base.get())]

        # try:
        #     a,b=curve_fit(gl1,xx,yy,bounds=([kmin[i],(np.max(y)-int(base.get()))/10,0,0],[kmax[i],np.max(y)-int(base.get())+1,0.3,0.01]))
        #     a,b=curve_fit(gl2,xx,yy,bounds=([kmin[i],(np.max(y)-int(base.get()))/10,0,0,kmin[i],(np.max(y)-int(base.get()))/10,0,0],[kmax[i],np.max(y)-int(base.get())+1,0.3,0.01,kmax[i],np.max(y)-int(base.get())+1,0.3,0.01]))
        # except RuntimeError:
        #     # a=[(kmin[i]+kmax[i])/2,(np.max(y)-int(base.get())),0.5,int(base.get())]
        #     a=[(kmin[i]+kmax[i])/2,(np.max(y)-int(base.get())),0.5,int(base.get()),(kmin[i]+kmax[i])/2,(np.max(y)-int(base.get())),0.5,int(base.get())]

        # pos[i]=a[0]
        # fwhm[i]=a[2]
        fmxx[i, :len(xx)] = xx
        fmyy[i, :len(yy)] = yy
        fmx[i, :] = x
        fmy[i, :] = y
        mvv[i] = v
        maa1[i, :] = a1
        maa2[i, :] = a2
        pbar.update(1)
        # print('MDC '+str(round((i+1)/len(ev)*100))+'%'+' ('+str(len(ev))+')')
        # st.put('MDC '+str(round((i+1)/len(fev)*100))+'%'+' ('+str(len(fev))+')')
    pbar.close()
    tt1 = threading.Thread(target=mjob)
    tt1.daemon = True
    tt1.start()


eprfit = 0


def fite():
    global ev, phi, data, evv, eaa1, eaa2, fexx, feyy, fex, fey, emin, emax, cei, ebase, eprfit
    cei = []
    ebase = [0 for i in range(len(phi))]
    if fpr == 1:
        try:
            emin, emax = semin, semax
        except NameError:
            emin = np.float64([np.min(ev) for i in range(len(phi))])
            emax = np.float64([np.max(ev) for i in range(len(phi))])
        if len(scei) >= 2:
            cei = scei
    else:
        emin = np.float64([np.min(ev) for i in range(len(phi))])
        emax = np.float64([np.max(ev) for i in range(len(phi))])
    # fexx=np.float64((np.arange(len(ev)*len(phi))+1).reshape(len(phi),len(ev)))
    # feyy=np.float64((np.arange(len(ev)*len(phi))+1).reshape(len(phi),len(ev)))
    # fexx=fexx/fexx*-50
    # feyy=feyy/feyy*-50
    fexx = np.float64((np.ones(len(ev)*len(phi))).reshape(len(phi), len(ev)))
    feyy = np.float64((np.ones(len(ev)*len(phi))).reshape(len(phi), len(ev)))
    fexx *= -50
    feyy *= -50
    fex = np.float64(np.arange(len(ev)*len(phi)).reshape(len(phi), len(ev)))
    fey = np.float64(np.arange(len(ev)*len(phi)).reshape(len(phi), len(ev)))
    evv = np.float64(np.arange(len(phi)))
    eaa1 = np.float64(np.arange(4*len(phi)).reshape(len(phi), 4))
    eaa2 = np.float64(np.arange(8*len(phi)).reshape(len(phi), 8))
    pbar = tqdm.tqdm(total=len(phi), desc='EDC', colour='blue')
    for i, v in enumerate(phi):
        angcut = data.sel(phi=v, method='nearest')
        x = np.float64(ev)
        y = angcut.to_numpy().reshape(len(x))
        tx = x[np.argwhere(x >= emin[i])].flatten()
        xx = tx[np.argwhere(tx <= emax[i])].flatten()
        ty = y[np.argwhere(x >= emin[i])].flatten()
        yy = ty[np.argwhere(tx <= emax[i])].flatten()
        yy = np.where(yy > int(base.get()), yy, int(base.get()))
        try:
            if i in sefi and fpr == 1:
                a1 = seaa1[i, :]
                a2 = seaa2[i, :]
                if seaa1[i, 1] == 10 or seaa2[i, 1] == 10:
                    eprfit = 1
            else:
                a1 = [(emin[i]+emax[i])/2, (np.max(y) -
                                            int(base.get())), 5, int(base.get())]
                a2 = [(emin[i]+emax[i])/2, (np.max(y)-int(base.get())), 5, int(base.get()),
                      (emin[i]+emax[i])/2, (np.max(y)-int(base.get())), 5, int(base.get())]
        except:
            a1 = [(emin[i]+emax[i])/2, (np.max(y) -
                                        int(base.get())), 5, int(base.get())]
            a2 = [(emin[i]+emax[i])/2, (np.max(y)-int(base.get())), 5, int(base.get()),
                  (emin[i]+emax[i])/2, (np.max(y)-int(base.get())), 5, int(base.get())]

        # try:
        #     a,b=curve_fit(gl1,xx,yy,bounds=([emin[i],(np.max(y)-int(base.get()))/10,0,0],[emax[i],np.max(y)-int(base.get())+1,3,0.01]))
        #     a,b=curve_fit(gl2,xx,yy,bounds=([emin[i],(np.max(y)-int(base.get()))/10,0,0,emin[i],(np.max(y)-int(base.get()))/10,0,0],[emax[i],np.max(y)-int(base.get())+1,3,0.01,emax[i],np.max(y)-int(base.get())+1,3,0.01]))

        # except RuntimeError:
        #     # a=[(emin[i]+emax[i])/2,(np.max(y)-int(base.get())),5,int(base.get())]
        #     a=[(emin[i]+emax[i])/2,(np.max(y)-int(base.get())),5,int(base.get()),(emin[i]+emax[i])/2,(np.max(y)-int(base.get())),5,int(base.get())]

        # epos[i]=a[0]
        # efwhm[i]=a[2]

        fexx[i, :len(xx)] = xx
        feyy[i, :len(yy)] = yy
        fex[i, :] = x
        fey[i, :] = y
        evv[i] = v
        eaa1[i, :] = a1
        eaa2[i, :] = a2
        pbar.update(1)
        # print('EDC '+str(round((i+1)/len(phi)*100))+'%'+' ('+str(len(phi))+')')
        # st.put('EDC '+str(round((i+1)/len(fphi)*100))+'%'+' ('+str(len(fphi))+')')
    pbar.close()
    tt2 = threading.Thread(target=ejob)
    tt2.daemon = True
    tt2.start()


def cmfit():
    t1 = threading.Thread(target=fitm)
    t1.start()


def cefit():
    t1 = threading.Thread(target=fite)
    t1.start()

############################################################
############################################################
############################################################
############################################################


def o_fitgl():
    try:
        # global pos,fwhm,epos,efwhm,base,k_offset,st,evv,eaa,fexx,feyy,fex,fey,mvv,maa,fmxx,fmyy,fmx,fmy
        global st
        print('fitting')
        st.put('fitting')
        t1 = threading.Thread(target=fitm)
        t2 = threading.Thread(target=fite)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        print('Done')
        st.put('Done')
    except:
        pass


def clmfit():
    global rpos, pos, fwhm, fev, ophi
    rpos = []
    pos = []
    fwhm = []
    fev = []
    ophi = []


def clefit():
    global fphi, epos, ffphi, efwhm, fk
    fphi = []
    epos = []
    ffphi = []
    efwhm = []
    fk = []


def cminrange(*e):
    if vcmax.get()-vcmin.get() < 1:
        try:
            vcmax.set(vcmin.get())
        except:
            pass
    try:
        h0.set_clim([vcmin.get(), vcmax.get()])
        out.draw()
    except:
        pass


def cmaxrange(*e):
    if vcmax.get()-vcmin.get() < 1:
        try:
            vcmin.set(vcmax.get())
        except:
            pass
    try:
        h0.set_clim([vcmin.get(), vcmax.get()])
        out.draw()
    except:
        pass


def o_fbb_offset(*e):
    global bb_offset
    if '' == bb_offset.get():
        bb_offset.set('0')
        bboffset.select_range(0, 1)


def fbb_offset(*e):
    t = threading.Thread(target=o_fbb_offset)
    t.daemon = True
    t.start()


def o_fbbk_offset(*e):
    global bbk_offset
    if '' == bbk_offset.get():
        bbk_offset.set('1')
        bbkoffset.select_range(0, 1)


def fbbk_offset(*e):
    t = threading.Thread(target=o_fbbk_offset)
    t.daemon = True
    t.start()


def o_fbase(*e):
    global base
    if '' == base.get():
        base.set('0')
        in_fit.select_range(0, 1)


def fbase(*e):
    t = threading.Thread(target=o_fbase)
    t.daemon = True
    t.start()


def o_flowlim(*e):
    global lowlim
    if '' == lowlim.get():
        lowlim.set('0')
        in_lowlim.select_range(0, 1)


def flowlim(*e):
    t = threading.Thread(target=o_flowlim)
    t.daemon = True
    t.start()


def o_reload(*e):
    global k_offset, fev, ophi, rpos, pos, ffphi, fwhm, fk, st, kmin, kmax
    if '' == k_offset.get():
        k_offset.set('0')
        koffset.select_range(0, 1)
    ophi = np.arcsin(rpos/(2*m*fev*1.6*10**-19)**0.5 /
                     10**-10*(h/2/np.pi))*180/np.pi
    pos = (2*m*fev*1.6*10**-19)**0.5 * \
        np.sin((np.float64(k_offset.get())+ophi)/180*np.pi)*10**-10/(h/2/np.pi)
    okmphi = np.arcsin(kmin/(2*m*fev*1.6*10**-19)**0.5 /
                       10**-10*(h/2/np.pi))*180/np.pi
    kmin = (2*m*fev*1.6*10**-19)**0.5 * \
        np.sin((np.float64(k_offset.get())+okmphi) /
               180*np.pi)*10**-10/(h/2/np.pi)
    okMphi = np.arcsin(kmax/(2*m*fev*1.6*10**-19)**0.5 /
                       10**-10*(h/2/np.pi))*180/np.pi
    kmax = (2*m*fev*1.6*10**-19)**0.5 * \
        np.sin((np.float64(k_offset.get())+okMphi) /
               180*np.pi)*10**-10/(h/2/np.pi)
    ffphi = np.float64(k_offset.get())+fphi
    fk = (2*m*epos*1.6*10**-19)**0.5 * \
        np.sin(ffphi/180*np.pi)*10**-10/(h/2/np.pi)
    os.chdir(cdir)
    try:
        np.savez('mfit', ko=k_offset.get(), fev=fev, rpos=rpos,
                 ophi=ophi, fwhm=fwhm, pos=pos, kmin=kmin, kmax=kmax)
    except:
        try:
            np.savez('efit', ko=k_offset.get(), fphi=fphi, epos=epos,
                     ffphi=ffphi, efwhm=efwhm, fk=fk, emin=emin, emax=emax)
        except:
            pass
        pass

    print('k_offset changed')
    st.put('k_offset changed')


def climon():
    cm.set(h0.get_clim()[0])
    cM.set(h0.get_clim()[1])
    lcmax.config(fg='black')
    lcmin.config(fg='black')
    Cmax.config(from_=cm.get(), to=cM.get(), state='active', fg='black')
    Cmin.config(from_=cm.get(), to=cM.get(), state='active', fg='black')
    vcmin.set(cm.get())
    vcmax.set(cM.get())


def climoff():
    cm.set(-10000)
    cM.set(10000)
    lcmax.config(fg='white')
    lcmin.config(fg='white')
    Cmax.config(from_=cm.get(), to=cM.get(), state='disabled', fg='white')
    Cmin.config(from_=cm.get(), to=cM.get(), state='disabled', fg='white')
    vcmin.set(cm.get())
    vcmax.set(cM.get())


def chcmp(*e):
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    a = lcmpd.subplots()
    h = lcmpd.colorbar(mpl.cm.ScalarMappable(
        norm=norm, cmap=value3.get()), cax=a, orientation='vertical', label='')
    h.set_ticklabels(h.get_ticks(), font='Arial')
    cmpg.draw()


def Chcmp(*e):
    global st, f, out, h0, h1, h2, f0
    limg.config(image=img[np.random.randint(len(img))])
    try:
        if value.get() == 'MDC Normalized':
            plot1()
            print('Colormap changed')
            st.put('Colormap changed')
        else:
            h0.set_cmap(value3.get())
            h0.set_clim([vcmin.get(), vcmax.get()])
            try:
                h1.set_cmap(value3.get())
                h1.set_clim([vcmin.get(), vcmax.get()])
                h2.set_cmap(value3.get())
                h2.set_clim([vcmin.get(), vcmax.get()])
                f.canvas.draw_idle()
                f0.canvas.draw_idle()
            except:
                pass
            out.draw()
            print('Colormap changed')
            st.put('Colormap changed')
    except:
        print('Fail to execute')
        st.put('Fail to execute')


def o_exptm():
    global name, pos, fwhm, fev, st
    st.put('Processing...')
    os.chdir(rdd.removesuffix(rdd.split('/')[-1]))
    ff = open(name+'_mdc_fitted_data.txt', 'w',
              encoding='utf-8')  # tab 必須使用 '\t' 不可"\t"
    ff.write('K.E. (eV)'+'\t'+'FWHM (k)'+'\t'+'Position (k)'+'\n')
    for i in range(len(fev)):
        ff.write(str(fev[i])+'\t'+str(fwhm[i])+'\t'+str(pos[i])+'\n')
    ff.close()
    print('Done')
    st.put('Done')


def o_expte():
    global name, epos, efwhm, ffphi, st
    st.put('Processing...')
    os.chdir(rdd.removesuffix(rdd.split('/')[-1]))
    ff = open(name+'_edc_fitted_data.txt', 'w',
              encoding='utf-8')  # tab 必須使用 '\t' 不可"\t"
    ff.write('Angle (deg)'+'\t'+'FWHM (eV)'+'\t'+'Position (eV)'+'\n')
    for i in range(len(ffphi)):
        ff.write(str(ffphi[i])+'\t'+str(efwhm[i])+'\t'+str(epos[i])+'\n')
    ff.close()
    print('Done')
    st.put('Done')


def interp(x: float, xp: float, fp: float) -> np.ndarray:
    if xp[1] >= xp[0]:
        y = np.interp(x, xp, fp)
    else:
        y = np.interp(x, xp[::-1], fp[::-1])
    return y


def o_bareband():
    file = fd.askopenfilename(title="Select TXT file",
                              filetypes=(("TXT files", "*.txt"),))

    # global be,k,rx,ry,ix,iy,limg,img
    global be, k, limg, img, st
    if len(file) > 0:
        print('Loading...')
        st.put('Loading...')
    t_k = []
    t_be = []
    with open(file) as f:
        for i, line in enumerate(f):
            if i != 0:  # ignore 1st row data (index = 0)
                t_k.append(line.split('\t')[0])
                t_be.append(line.split('\t')[1].replace('\n', ''))
    # [::-1] inverse the order for np.interp (xp values should be increasing)
    be = np.float64(t_be)*1000
    # [::-1] inverse the order for np.interp (xp values should be increasing)
    k = np.float64(t_k)
    os.chdir(cdir)
    np.savez('bb', path=file, be=be, k=k)
    limg.config(image=img[np.random.randint(len(img))])
    print('Done')
    st.put('Done')


def o_plot1(*e):
    global value, value1, value2, data, ev, phi, mfpath, fig, out, pflag, k_offset, value3, limg, img, optionList, h0, ao, xl, yl, st
    if value.get() in optionList:
        limg.config(image=img[np.random.randint(len(img))])
        print('Plotting...')
        st.put('Plotting...')
        pflag = 1
        value1.set('---Plot2---')
        value2.set('---Plot3---')
        fig.clear()
        try:
            ev
        except:
            print('Please load Raw Data')
            st.put('Please load Raw Data')
        if value.get() == 'Raw Data':
            rplot(fig, out)
        else:
            if value.get() == 'First Derivative':
                ao = fig.subplots()
                pz = np.diff(data.to_numpy())/np.diff(phi)
                px, py = np.meshgrid(phi[0:-1], ev)
                px = (2*m*np.full_like(np.zeros([len(phi[0:-1]), len(ev)], dtype=float), ev)*1.6*10**-19).transpose(
                )**0.5*np.sin((np.float64(k_offset.get())+px+np.diff(phi)/2)/180*np.pi)*10**-10/(h/2/np.pi)
                h0 = ao.pcolormesh(px, py, pz, cmap=value3.get())
                cb = fig.colorbar(h0)
                cb.set_ticklabels(cb.get_ticks(), font='Arial')
            elif value.get() == 'Second Derivative':
                ao = fig.subplots()
                pz = np.diff(data.to_numpy())/np.diff(phi)
                pz = np.diff(pz)/np.diff(phi[0:-1])
                px, py = np.meshgrid(phi[0:-2], ev)
                px = (2*m*np.full_like(np.zeros([len(phi[0:-2]), len(ev)], dtype=float), ev)*1.6*10**-19).transpose(
                )**0.5*np.sin((np.float64(k_offset.get())+px+np.diff(phi[0:-1])/2*2)/180*np.pi)*10**-10/(h/2/np.pi)
                h0 = ao.pcolormesh(px, py, pz, cmap=value3.get())
                cb = fig.colorbar(h0)
                cb.set_ticklabels(cb.get_ticks(), font='Arial')
            else:
                ao = fig.subplots()
                if value.get() == 'E-K Diagram':
                    # h1=a.scatter(mx,my,c=mz,marker='o',s=0.9,cmap=value3.get());
                    px, py = np.meshgrid(phi, ev)
                    px = (2*m*np.full_like(np.zeros([len(phi), len(ev)], dtype=float), ev)*1.6*10**-19).transpose(
                    )**0.5*np.sin((np.float64(k_offset.get())+px)/180*np.pi)*10**-10/(h/2/np.pi)
                    pz = data.to_numpy()
                    h0 = ao.pcolormesh(px, py, pz, cmap=value3.get())
                    cb = fig.colorbar(h0)
                    cb.set_ticklabels(cb.get_ticks(), font='Arial')
                if value.get() == 'MDC Normalized':
                    pbar = tqdm.tqdm(
                        total=len(ev)-1, desc='MDC Normalized', colour='red')
                    for n in range(len(ev)-1):
                        ecut = data.sel(eV=ev[n], method='nearest')
                        x = (2*m*ev[n]*1.6*10**-19)**0.5*np.sin(
                            (np.float64(k_offset.get())+phi)/180*np.pi)*10**-10/(h/2/np.pi)
                        y = ecut.to_numpy().reshape(len(ecut))
                        # mz[len(phi)*n:len(phi)*(n+1)]=np.array(y,dtype=float)
                        # mx[len(phi)*n:len(phi)*(n+1)]=x
                        # ty=np.arange(len(x), dtype=float)
                        # my[len(phi)*n:len(phi)*(n+1)]=np.full_like(ty, ev[n])
                        # a.scatter(x,np.full_like(ty, ev[n]),c=np.array(y,dtype=int),marker='o',s=0.9,cmap=value3.get());
                        px, py = np.meshgrid(x, ev[n:(n+2)])
                        ao.pcolormesh(px, py, np.full_like(
                            np.zeros([2, len(phi)], dtype=float), y), cmap=value3.get())
                        pbar.update(1)
                        # print(str(round((n+1)/(len(ev)-1)*100))+'%'+' ('+str(len(ev)-1)+')')
                        st.put(str(round((n+1)/(len(ev)-1)*100)) +
                               '%'+' ('+str(len(ev)-1)+')')
                    pbar.close()
            ao.set_title(value.get(), font='Arial', fontsize=16)
            ao.set_xlabel(r'k ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=14)
            ao.set_ylabel('Kinetic Energy (eV)', font='Arial', fontsize=14)
            xl = ao.get_xlim()
            yl = ao.get_ylim()
        try:
            if value.get() != 'MDC Normalized':
                climon()
                out.draw()
            else:
                climoff()
                out.draw()
        except:
            pass
        print('Done')
        st.put('Done')


def o_plot2(*e):
    global fig, out, fwhm, fev, pos, value, value1, value2, k, be, rx, ry, ix, iy, pflag, limg, img, bb_offset, bbk_offset, optionList1, st
    if value1.get() in optionList1:
        limg.config(image=img[np.random.randint(len(img))])
        print('Plotting...')
        st.put('Plotting...')
        pflag = 2
        value.set('---Plot1---')
        value2.set('---Plot3---')
        fig.clear()
        climoff()
        if value1.get() == 'MDC fitted Data':
            try:
                x = (fev-21.2)*1000
                # y = (fwhm*6.626*10**-34/2/3.1415926/(10**-10))**2/2/(9.11*10**-31)/(1.6*10**-19)*1000
            except:
                print(r'Please Load MDC fitted file')
                st.put(r'Please Load MDC fitted file')
            try:
                a = fig.subplots(2, 1)
                a[0].set_title('MDC Fitting Result', font='Arial', fontsize=18)
                a[0].set_xlabel('Binding Energy (meV)',
                                font='Arial', fontsize=14)
                a[0].set_ylabel(
                    r'Position ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=14)
                a[0].tick_params(direction='in')
                a[0].scatter(x, pos, c='black', s=5)

                a[1].set_xlabel('Binding Energy (meV)',
                                font='Arial', fontsize=14)
                a[1].set_ylabel(
                    r'FWHM ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=14)
                a[1].tick_params(direction='in')
                a[1].scatter(x, fwhm, c='black', s=5)
            except:
                print('Please load MDC fitted file')
                st.put('Please load MDC fitted file')
        elif value1.get() == 'EDC fitted Data':
            try:
                x = fk
            except:
                print(r'Please Load EDC fitted file')
                st.put(r'Please Load EDC fitted file')
            try:
                a = fig.subplots(2, 1)
                a[0].set_title('EDC Fitting Result', font='Arial', fontsize=18)
                a[0].set_xlabel(
                    r'Position ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=14)
                a[0].set_ylabel('Binding Energy (meV)',
                                font='Arial', fontsize=14)
                a[0].tick_params(direction='in')
                a[0].scatter(x, (epos-21.2)*1000, c='black', s=5)

                a[1].set_xlabel(
                    r'Position ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=14)
                a[1].set_ylabel('FWHM (meV)', font='Arial', fontsize=14)
                a[1].tick_params(direction='in')
                a[1].scatter(x, efwhm*1000, c='black', s=5)
            except:
                print('Please load EDC fitted file')
                st.put('Please load EDC fitted file')
        elif value1.get() == 'Real Part':
            try:
                x = (fev-21.2)*1000
                y = pos
            except:
                print('Please load MDC fitted file')
                st.put('Please load MDC fitted file')
            try:
                yy = interp(y, k*np.float64(bbk_offset.get()), be +
                            # interp x into be,k set
                            np.float64(bb_offset.get()))
            except:
                print('Please load Bare Band file')
                st.put('Please load Bare Band file')
            a = fig.subplots(2, 1)
            a[0].set_title('Real Part', font='Arial', fontsize=18)
            a[0].plot(x, x-yy, c='black', linestyle='-', marker='.')

            rx = x
            ry = x-yy
            a[0].tick_params(direction='in')
            a[0].set_xlabel('Binding Energy (meV)', font='Arial', fontsize=14)
            a[0].set_ylabel(r'Re $\Sigma$ (meV)', font='Arial', fontsize=14)

            h1 = a[1].scatter(y, x, c='black', s=5)
            h2 = a[1].scatter(k*np.float64(bbk_offset.get()),
                              be+np.float64(bb_offset.get()), c='red', s=5)

            a[1].legend([h1, h2], ['fitted data', 'bare band'])
            a[1].tick_params(direction='in')
            a[1].set_ylabel('Binding Energy (meV)', font='Arial', fontsize=14)
            a[1].set_xlabel(
                r'Pos ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=14)

            # a[0].set_xlim([-1000,50])
            # a[0].set_ylim([-100,500])
            # a[1].set_ylim([-600,200])
            # a[1].set_xlim([-0.05,0.05])
        elif value1.get() == 'Imaginary Part':
            try:
                tbe = (fev-21.2)*1000
            except:
                print(r'Please Load MDC fitted file')
                st.put(r'Please Load MDC fitted file')
            try:
                x = interp(tbe, be+np.float64(bb_offset.get()),
                           k*np.float64(bbk_offset.get()))
                y = interp(x, k*np.float64(bbk_offset.get()),
                           be+np.float64(bb_offset.get()))
            except:
                print('Please load Bare Band file')
                st.put('Please load Bare Band file')
            xx = np.diff(x)
            yy = np.diff(y)

            # eliminate vf in gap
            for i in range(len(yy)):
                if yy[i]/xx[i] > 20000:
                    yy[i] = 0
            v = yy/xx
            v = np.append(v, v[-1])  # fermi velocity
            try:
                yy = v*fwhm/2
            except:
                print('Please load MDC fitted file')
                st.put('Please load MDC fitted file')
            xx = tbe
            ax = fig.subplots(2, 1)
            a = ax[0]
            b = ax[1]
            a.set_title('Imaginary Part', font='Arial', fontsize=18)
            a.plot(xx, yy, c='black', linestyle='-', marker='.')

            ix = xx
            iy = yy
            a.tick_params(direction='in')
            a.set_xlabel('Binding Energy (meV)', font='Arial', fontsize=14)
            a.set_ylabel(r'Im $\Sigma$ (meV)', font='Arial', fontsize=14)

            x = (fev-21.2)*1000
            y = fwhm
            b.plot(x, y, c='black', linestyle='-', marker='.')
            b.tick_params(direction='in')
            b.set_xlabel('Binding Energy (meV)', font='Arial', fontsize=14)
            b.set_ylabel(r'FWHM ($\frac{2\pi}{\AA}$)',
                         font='Arial', fontsize=14)
        out.draw()
        print('Done')
        st.put('Done')


def o_plot3(*e):
    global fig, out, rx, ry, ix, iy, fwhm, pos, value, value1, value2, pflag, k, be, k_offset, value3, limg, img, bb_offset, bbk_offset, optionList2, h0, bo, xl, yl, posmin, posmax, eposmin, eposmax, tb0, tb0_, tb1, tb1_, tb2, st
    if value2.get() in optionList2:
        limg.config(image=img[np.random.randint(len(img))])
        print('Plotting...')
        st.put('Plotting...')
        pflag = 3
        value.set('---Plot1---')
        value1.set('---Plot2---')
        fig.clear()
        try:
            x = (fev-21.2)*1000
            y = pos
        except:
            print('Please load MDC fitted file')
            st.put('Please load MDC fitted file')
        if value2.get() != 'Data Plot with Pos':
            try:
                yy = interp(y, k*np.float64(bbk_offset.get()), be +
                            # interp x into be,k set
                            np.float64(bb_offset.get()))
                rx = x
                ry = x-yy
                tbe = (fev-21.2)*1000
                x = interp(tbe, be+np.float64(bb_offset.get()),
                           k*np.float64(bbk_offset.get()))
                y = interp(x, k*np.float64(bbk_offset.get()),
                           be+np.float64(bb_offset.get()))
                xx = np.diff(x)
                yy = np.diff(y)

                # eliminate vf in gap
                for i in range(len(yy)):
                    if yy[i]/xx[i] > 20000:
                        yy[i] = 0
                v = yy/xx
                v = np.append(v, v[-1])  # fermi velocity
                yy = v*fwhm/2
                xx = tbe
                ix = xx
                iy = yy
            except:
                print('Please load Bare Band file')
                st.put('Please load Bare Band file')
        if value2.get() == 'Real & Imaginary':
            a = fig.subplots(2, 1)
            a[0].set_title(r'Self Energy $\Sigma$', font='Arial', fontsize=18)
            a[0].plot(rx, ry, c='black', linestyle='-', marker='.')
            a[0].tick_params(direction='in')
            a[0].set_xlabel('Binding Energy (meV)', font='Arial', fontsize=14)
            a[0].set_ylabel(r'Re $\Sigma$ (meV)', font='Arial', fontsize=14)
            a[1].plot(ix, iy, c='black', linestyle='-', marker='.')
            a[1].tick_params(direction='in')
            a[1].set_xlabel('Binding Energy (meV)', font='Arial', fontsize=14)
            a[1].set_ylabel(r'Im $\Sigma$ (meV)', font='Arial', fontsize=14)
        elif value2.get() == 'KK Transform':
            ax = fig.subplots(2, 1)
            a = ax[0]
            b = ax[1]
            a.set_title('Self Energy', font='Arial', fontsize=18)
            a.plot(rx, ry, c='black', linestyle='-',
                   marker='.', label=r'Re $\Sigma$')
            a.tick_params(direction='in')
            a.set_xlabel('Binding Energy (meV)', font='Arial', fontsize=14)
            a.set_ylabel(r'Re $\Sigma$ (meV)', font='Arial', fontsize=14)

            kx = ix
            ky = np.linspace(0, 1, len(kx))
            # de=np.linspace(0,1,len(kx))
            # de[0:-1]=np.diff(kx)
            # de[-1]=de[-2]
            de = np.diff(kx)
            de = np.append(de, de[-1])

            for i in range(len(kx)):
                # ky[i]=np.trapz(y=iy/(iy-kx[i]),x=iy,dx=de)
                intg = 0
                for j in range(len(kx)):
                    if i != j:
                        tval = iy[j]/(kx[j]-kx[i])*de[j]
                        if str(iy[j]) == 'nan':
                            tval = 0
                        intg += tval
                ky[i] = -1/np.pi*intg
            a.plot(kx, ky, c='red', linestyle='-', marker='.',
                   label=r'Re $\Sigma_{KK}$=KK(Im $\Sigma$)')
            handles, labels = a.get_legend_handles_labels()
            a.legend(handles, labels)
            # a.legend([h1,h2],['measured data','KK transform'])

            #   KK Re
            b.plot(ix, iy, c='black', linestyle='-',
                   marker='.', label=r'Im $\Sigma$')
            b.tick_params(direction='in')
            b.set_xlabel('Binding Energy (meV)', font='Arial', fontsize=14)
            b.set_ylabel(r'Im $\Sigma$ (meV)', font='Arial', fontsize=14)

            kx = rx
            ky = np.linspace(0, 1, len(kx))
            # de=np.linspace(0,1,len(kx))
            # de[0:-1]=np.diff(kx)
            # de[-1]=de[-2]
            de = np.diff(kx)
            de = np.append(de, de[-1])

            for i in range(len(kx)):
                # ky[i]=np.trapz(y=iy/(iy-kx[i]),x=iy,dx=de)
                intg = 0
                for j in range(len(kx)):
                    if i != j:
                        tval = ry[j]/(kx[j]-kx[i])*de[j]
                        if str(ry[j]) == 'nan':
                            tval = 0
                        intg += tval
                ky[i] = 1/np.pi*intg
            b.plot(kx, ky, c='red', linestyle='-', marker='.',
                   label=r'Im $\Sigma_{KK}$=KK(Re $\Sigma$)')
            handles, labels = b.get_legend_handles_labels()
            b.legend(handles, labels)
        elif value2.get() == 'Data Plot with Pos' or value2.get() == 'Data Plot with Pos and Bare Band':
            bo = fig.subplots()
            px, py = np.meshgrid(phi, ev)

            px = (2*m*np.full_like(np.zeros([len(phi), len(ev)], dtype=float), ev)*1.6*10**-19).transpose(
            )**0.5*np.sin((np.float64(k_offset.get())+px)/180*np.pi)*10**-10/(h/2/np.pi)
            print(np.float64(k_offset.get()))
            pz = data.to_numpy()
            h0 = bo.pcolormesh(px, py, pz, cmap=value3.get())
            cb = fig.colorbar(h0)
            cb.set_ticklabels(cb.get_ticks(), font='Arial',
                              fontsize=14, minor=False)
            #   MDC Norm
            # for i in range(len(ev)):
            #     b.scatter(mx[len(phi)*i:len(phi)*(i+1)],my[len(phi)*i:len(phi)*(i+1)],c=mz[len(phi)*i:len(phi)*(i+1)],marker='o',s=0.9,cmap='viridis',alpha=0.3)
            # a.set_title('MDC Normalized')
            bo.set_title(value2.get(), font='Arial', fontsize=18)
            # a.set_xlabel(r'k ($\frac{2\pi}{\AA}$)',fontsize=14)
            # a.set_ylabel('Kinetic Energy (eV)',fontsize=14)
            bo.set_xlabel(r'k ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=16)
            bo.set_ylabel('Kinetic Energy (eV)', font='Arial', fontsize=16)
            # b.set_xticklabels(labels=b.get_xticklabels(),fontsize=14)
            # b.set_yticklabels(labels=b.get_yticklabels(),fontsize=14)
            try:
                if mp == 1:
                    tb0 = bo.scatter(pos, fev, marker='.', s=0.3, c='black')
                if mf == 1:
                    ophimin = np.arcsin(
                        (rpos-fwhm/2)/(2*m*fev*1.6*10**-19)**0.5/10**-10*(h/2/np.pi))*180/np.pi
                    ophimax = np.arcsin(
                        (rpos+fwhm/2)/(2*m*fev*1.6*10**-19)**0.5/10**-10*(h/2/np.pi))*180/np.pi
                    posmin = (2*m*fev*1.6*10**-19)**0.5*np.sin(
                        (np.float64(k_offset.get())+ophimin)/180*np.pi)*10**-10/(h/2/np.pi)
                    posmax = (2*m*fev*1.6*10**-19)**0.5*np.sin(
                        (np.float64(k_offset.get())+ophimax)/180*np.pi)*10**-10/(h/2/np.pi)
                    tb0_ = bo.scatter([posmin, posmax], [
                                      fev, fev], marker='|', c='grey', s=10, alpha=0.8)
            except:
                pass
            try:
                if ep == 1:
                    tb1 = bo.scatter(fk, epos, marker='.', s=0.3, c='black')
                if ef == 1:
                    eposmin = epos-efwhm/2
                    eposmax = epos+efwhm/2
                    tb1_ = bo.scatter(
                        [fk, fk], [eposmin, eposmax], marker='_', c='grey', s=10, alpha=0.8)

            except:
                pass
            try:
                if value2.get() == 'Data Plot with Pos and Bare Band':
                    tb2, = bo.plot(k*np.float64(bbk_offset.get()), (be +
                                   np.float64(bb_offset.get()))/1000+21.2, linewidth=0.3, c='red')
            except:
                bo.set_title('Data Plot with Pos w/o Bare Band',
                             font='Arial', fontsize=18)
                print('Please load Bare Band file')
                st.put('Please load Bare Band file')
        try:
            if value2.get() != 'Real & Imaginary' and value2.get() != 'KK Transform':
                xl = bo.get_xlim()
                yl = bo.get_ylim()
                climon()
                out.draw()
            else:
                climoff()
                out.draw()
        except:
            pass
        print('Done')
        st.put('Done')


props = dict(facecolor='green', alpha=0.3)


def select_callback(eclick, erelease):
    global ta0, ta0_, ta1, ta1_, ta2, a, f
    """
    Callback for line selection.

    *eclick* and *erelease* are the press and release events.
    """
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    if eclick.button == 1:
        a.set_xlim(sorted([x1, x2]))
        a.set_ylim(sorted([y1, y2]))
        f.show()
        if abs(x1-x2) < (xl[1]-xl[0])/3*2 or abs(y1-y2) < (yl[1]-yl[0])/3*2:
            try:
                if mp == 1:
                    ta0.remove()
                if mf == 1:
                    ta0_.remove()
            except:
                pass
            try:
                if ep == 1:
                    ta1.remove()
                if ef == 1:
                    ta1_.remove()
            except:
                pass
            try:
                ta2.remove()
            except:
                pass
            if value2.get() == 'Data Plot with Pos and Bare Band' or value2.get() == 'Data Plot with Pos':
                try:
                    if mp == 1:
                        ta0 = a.scatter(pos, fev, marker='.', s=30, c='black')
                    if mf == 1:
                        ta0_ = a.scatter([posmin, posmax], [
                                         fev, fev], marker='|', c='grey', s=50, alpha=0.8)
                except:
                    pass
                try:
                    if ep == 1:
                        ta1 = a.scatter(fk, epos, marker='.', s=30, c='black')
                    if ef == 1:
                        ta1_ = a.scatter(
                            [fk, fk], [eposmin, eposmax], marker='_', c='grey', s=50, alpha=0.8)
                except:
                    pass

                if value2.get() == 'Data Plot with Pos and Bare Band':
                    ta2, = a.plot(k*np.float64(bbk_offset.get()), (be +
                                  np.float64(bb_offset.get()))/1000+21.2, linewidth=5, c='red')
            f.show()
        else:
            try:
                if mp == 1:
                    ta0.remove()
                    ta0 = a.scatter(pos, fev, marker='.', s=0.3, c='black')
                if mf == 1:
                    ta0_.remove()
                    ta0_ = a.scatter([posmin, posmax], [fev, fev],
                                     marker='|', c='grey', s=10, alpha=0.8)
            except:
                pass
            try:
                if ep == 1:
                    ta1.remove()
                    ta1 = a.scatter(fk, epos, marker='.', s=0.3, c='black')
                if ef == 1:
                    ta1_.remove()
                    ta1_ = a.scatter([fk, fk], [eposmin, eposmax],
                                     marker='_', c='grey', s=10, alpha=0.8)
            except:
                pass
            try:
                if value2.get() == 'Data Plot with Pos and Bare Band':
                    ta2.remove()
                    ta2, = a.plot(k*np.float64(bbk_offset.get()), (be +
                                  np.float64(bb_offset.get()))/1000+21.2, linewidth=0.3, c='red')
            except:
                pass
            f.show()
    else:
        a.set_xlim(xl)
        a.set_ylim(yl)
        try:
            if mp == 1:
                ta0.remove()
                ta0 = a.scatter(pos, fev, marker='.', s=0.3, c='black')
            if mf == 1:
                ta0_.remove()
                ta0_ = a.scatter([posmin, posmax], [fev, fev],
                                 marker='|', c='grey', s=10, alpha=0.8)
        except:
            pass
        try:
            if ep == 1:
                ta1.remove()
                ta1 = a.scatter(fk, epos, marker='.', s=0.3, c='black')
            if ef == 1:
                ta1_.remove()
                ta1_ = a.scatter([fk, fk], [eposmin, eposmax],
                                 marker='_', c='grey', s=10, alpha=0.8)
        except:
            pass
        try:
            if value2.get() == 'Data Plot with Pos and Bare Band':
                ta2.remove()
                ta2, = a.plot(k*np.float64(bbk_offset.get()), (be +
                              np.float64(bb_offset.get()))/1000+21.2, linewidth=0.3, c='red')
        except:
            pass
        f.show()
    # print(f"({x1:3.2f}, {y1:3.2f}) --> ({x2:3.2f}, {y2:3.2f})")
    # print(f"The buttons you used were: {eclick.button} {erelease.button}")
# def toggle_selector(event):
#     print('Key pressed.')
#     if event.key == 't':
#         for selector in selectors:
#             name = type(selector).__name__
#             if selector.active:
#                 print(f'{name} deactivated.')
#                 selector.set_active(False)
#             else:
#                 print(f'{name} activated.')
#                 selector.set_active(True)


def onselect(xmin, xmax):
    global f, f0, h1, h2
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    # vcmin.set(xmin)
    # vcmax.set(xmax)
    h2.set_clim(xmin, xmax)
    # f0.canvas.draw_idle()
    f0.show()
    h1.set_clim(xmin, xmax)
    # f.canvas.draw_idle()
    f.show()


def onmove_callback(xmin, xmax):
    global f, f0, h1, h2
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    # vcmin.set(xmin)
    # vcmax.set(xmax)
    h2.set_clim(xmin, xmax)
    # f0.canvas.draw_idle()
    f0.show()
    h1.set_clim(xmin, xmax)
    # f.canvas.draw_idle()
    f.show()


cf = True


def cut_move(event):
    global cxdata, cydata, acx, acy, a, f
    # ,x,y
    f.canvas.get_tk_widget().config(cursor="")
    if event.inaxes:
        cxdata = event.xdata
        cydata = event.ydata
        xf = (cxdata > a.get_xlim()[0] and cxdata < a.get_xlim()[1])
        yf = (cydata > a.get_ylim()[0] and cydata < a.get_ylim()[1])
        if xf and yf and cf:
            f.canvas.get_tk_widget().config(cursor="crosshair")
            dx = data.sel(
                eV=cydata, method='nearest').to_numpy().reshape(len(phi))
            dy = data.sel(
                phi=cxdata, method='nearest').to_numpy().reshape(len(ev))
            # try:
            #     x.remove()
            #     y.remove()
            # except:
            #     pass
            # x=a.axvline(cxdata,color='r')
            # y=a.axhline(cydata,color='r')
            acx.clear()
            acy.clear()
            acx.plot(phi, dx, c='black')
            acy.plot(dy, ev, c='black')
            acx.set_xticks([])
            acy.set_yticks([])
            acx.set_xlim(a.get_xlim())
            acy.set_ylim(a.get_ylim())
            # f.canvas.draw_idle()
            f.show()


def cut_select(event):
    global cf, a, f, x, y, acx, acy
    if event.button == 1 and cf:
        cf = False
        x = a.axvline(event.xdata, color='red')
        y = a.axhline(event.ydata, color='red')
    elif event.button == 1 and not cf:
        x.remove()
        y.remove()
        x = a.axvline(event.xdata, color='red')
        y = a.axhline(event.ydata, color='red')
        dx = data.sel(eV=event.ydata,
                      method='nearest').to_numpy().reshape(len(phi))
        dy = data.sel(phi=event.xdata,
                      method='nearest').to_numpy().reshape(len(ev))
        acx.clear()
        acy.clear()
        acx.plot(phi, dx, c='black')
        acy.plot(dy, ev, c='black')
        acx.set_xticks([])
        acy.set_yticks([])
        acx.set_xlim(a.get_xlim())
        acy.set_ylim(a.get_ylim())

    elif event.button == 3:
        cf = True
        x.remove()
        y.remove()
    # f.canvas.draw_idle()
    f.show()

# def cut_click(event):
#     if event.button is MouseButton.LEFT:
#         print('disconnecting callback')
# def o_exp():


def exp(*e):
    global value, value1, value2, value3, data, ev, phi, mx, my, mz, mfpath, fev, fwhm, pos, k, be, rx, ry, ix, iy, pflag, k_offset, limg, img, bb_offset, bbk_offset, h1, h2, a0, a, b, f0, f, selectors, acx, acy, posmin, posmax, eposmin, eposmax
    limg.config(image=img[np.random.randint(len(img))])
    selectors = []
    cursor = []
    h1 = []
    h2 = []
    f = []
    f0 = []
    if pflag == 1:
        mz = data.to_numpy()
        f0 = plt.figure(figsize=(8, 7), layout='constrained')
        a0 = plt.axes([0.13, 0.45, 0.8, 0.5])
        a1 = plt.axes([0.13, 0.08, 0.8, 0.2])
        a0.set_title('Drag to select specific region')
        selectors.append(RectangleSelector(
            a0, select_callback,
            useblit=True,
            button=[1, 3],  # disable middle button
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True,
            props=props))
        # f0.canvas.mpl_connect('key_press_event',toggle_selector)
        if value.get() != 'Raw Data':
            f, a = plt.subplots(dpi=150)
        if value.get() == 'Raw Data':
            f = plt.figure(figsize=(9, 7), layout='constrained')
            a = plt.axes([0.13, 0.08, 0.68, 0.6])
            acx = plt.axes([0.13, 0.75, 0.545, 0.15])
            acy = plt.axes([0.82, 0.08, 0.15, 0.6])
            plt.connect('motion_notify_event', cut_move)
            plt.connect('button_press_event', cut_select)
            mx, my = np.meshgrid(phi, ev)
            # h1 = a.scatter(mx,my,c=mz,marker='o',s=0.9,cmap=value3.get());
            h1 = a.pcolormesh(mx, my, mz, cmap=value3.get())
            cb = f.colorbar(h1)
            cb.set_ticklabels(cb.get_ticks(), font='Arial')
            h2 = a0.pcolormesh(mx, my, mz, cmap=value3.get())
            cb1 = f0.colorbar(h2)
            cb1.set_ticklabels(cb1.get_ticks(), font='Arial')

            acx.set_xticks([])
            acy.set_yticks([])

            n = a1.hist(mz.flatten(), bins=np.linspace(
                min(mz.flatten()), max(mz.flatten()), 50), color='green')
            a1.set_xlabel('Intensity')
            a1.set_ylabel('Counts')
            a1.set_title('Drag to Select the range of Intensity ')
            selectors.append(SpanSelector(
                a1,
                onselect,
                "horizontal",
                useblit=True,
                props=dict(alpha=0.3, facecolor="tab:blue"),
                onmove_callback=onmove_callback,
                interactive=True,
                drag_from_anywhere=True,
                snap_values=n[1]
            ))
        elif value.get() == 'First Derivative':
            pz = np.diff(data.to_numpy())/np.diff(phi)
            px, py = np.meshgrid(phi[0:-1], ev)
            px = (2*m*np.full_like(np.zeros([len(phi[0:-1]), len(ev)], dtype=float), ev)*1.6*10**-19).transpose(
            )**0.5*np.sin((np.float64(k_offset.get())+px+np.diff(phi)/2)/180*np.pi)*10**-10/(h/2/np.pi)
            h1 = a.pcolormesh(px, py, pz, cmap=value3.get())
            h2 = a0.pcolormesh(px, py, pz, cmap=value3.get())
            cb = f.colorbar(h1)
            cb.set_ticklabels(cb.get_ticks(), font='Arial')
            cb1 = f0.colorbar(h2)
            cb1.set_ticklabels(cb1.get_ticks(), font='Arial')

            n = a1.hist(pz.flatten(), bins=np.linspace(
                min(pz.flatten()), max(pz.flatten()), 50), color='green')
            a1.set_xlabel('Intensity')
            a1.set_ylabel('Counts')
            a1.set_title('Drag to Select the range of Intensity ')
            selectors.append(SpanSelector(
                a1,
                onselect,
                "horizontal",
                useblit=True,
                props=dict(alpha=0.3, facecolor="tab:blue"),
                onmove_callback=onmove_callback,
                interactive=True,
                drag_from_anywhere=True,
                snap_values=n[1]
            ))
        elif value.get() == 'Second Derivative':
            pz = np.diff(data.to_numpy())/np.diff(phi)
            pz = np.diff(pz)/np.diff(phi[0:-1])
            px, py = np.meshgrid(phi[0:-2], ev)
            px = (2*m*np.full_like(np.zeros([len(phi[0:-2]), len(ev)], dtype=float), ev)*1.6*10**-19).transpose(
            )**0.5*np.sin((np.float64(k_offset.get())+px+np.diff(phi[0:-1])/2*2)/180*np.pi)*10**-10/(h/2/np.pi)
            h1 = a.pcolormesh(px, py, pz, cmap=value3.get())
            h2 = a0.pcolormesh(px, py, pz, cmap=value3.get())
            cb = f.colorbar(h1)
            cb.set_ticklabels(cb.get_ticks(), font='Arial')
            cb1 = f0.colorbar(h2)
            cb1.set_ticklabels(cb1.get_ticks(), font='Arial')

            n = a1.hist(pz.flatten(), bins=np.linspace(
                min(pz.flatten()), max(pz.flatten()), 50), color='green')
            a1.set_xlabel('Intensity')
            a1.set_ylabel('Counts')
            a1.set_title('Drag to Select the range of Intensity ')
            selectors.append(SpanSelector(
                a1,
                onselect,
                "horizontal",
                useblit=True,
                props=dict(alpha=0.3, facecolor="tab:blue"),
                onmove_callback=onmove_callback,
                interactive=True,
                drag_from_anywhere=True,
                snap_values=n[1]
            ))
        else:
            if value.get() == 'E-K Diagram':
                # h1=a.scatter(mx,my,c=mz,marker='o',s=0.9,cmap=value3.get());
                px, py = np.meshgrid(phi, ev)
                px = (2*m*np.full_like(np.zeros([len(phi), len(ev)], dtype=float), ev)*1.6*10**-19).transpose(
                )**0.5*np.sin((np.float64(k_offset.get())+px)/180*np.pi)*10**-10/(h/2/np.pi)
                pz = data.to_numpy()
                h1 = a.pcolormesh(px, py, pz, cmap=value3.get())
                h2 = a0.pcolormesh(px, py, pz, cmap=value3.get())
                cb = f.colorbar(h1)
                cb.set_ticklabels(cb.get_ticks(), font='Arial')
                cb1 = f0.colorbar(h2)
                cb1.set_ticklabels(cb1.get_ticks(), font='Arial')

                n = a1.hist(pz.flatten(), bins=np.linspace(
                    min(pz.flatten()), max(pz.flatten()), 50), color='green')
                a1.set_xlabel('Intensity')
                a1.set_ylabel('Counts')
                a1.set_title('Drag to Select the range of Intensity ')
                selectors.append(SpanSelector(
                    a1,
                    onselect,
                    "horizontal",
                    useblit=True,
                    props=dict(alpha=0.3, facecolor="tab:blue"),
                    onmove_callback=onmove_callback,
                    interactive=True,
                    drag_from_anywhere=True,
                    snap_values=n[1]
                ))
            if value.get() == 'MDC Normalized':
                for n in range(len(ev)-1):
                    ecut = data.sel(eV=ev[n], method='nearest')
                    x = (2*m*ev[n]*1.6*10**-19)**0.5*np.sin(
                        (np.float64(k_offset.get())+phi)/180*np.pi)*10**-10/(h/2/np.pi)
                    y = ecut.to_numpy().reshape(len(ecut))
                    # mz[len(phi)*n:len(phi)*(n+1)]=np.array(y,dtype=float)
                    # mx[len(phi)*n:len(phi)*(n+1)]=x
                    # ty=np.arange(len(x), dtype=float)
                    # my[len(phi)*n:len(phi)*(n+1)]=np.full_like(ty, ev[n])
                    # a.scatter(x,np.full_like(ty, ev[n]),c=np.array(y,dtype=int),marker='o',s=0.9,cmap=value3.get());
                    px, py = np.meshgrid(x, ev[n:n+2])
                    a.pcolormesh(px, py, np.full_like(
                        np.zeros([2, len(phi)], dtype=float), y), cmap=value3.get())
                    a0.pcolormesh(px, py, np.full_like(
                        np.zeros([2, len(phi)], dtype=float), y), cmap=value3.get())

        a.set_title(value.get(), font='Arial', fontsize=18)
        a.set_xlabel(r'k ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=16)
        a.set_ylabel('Kinetic Energy (eV)', font='Arial', fontsize=16)
        a0.set_xlabel(r'k ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=16)
        a0.set_ylabel('Kinetic Energy (eV)', font='Arial', fontsize=16)
        if value.get() == 'Raw Data':
            a.set_xlabel(r'$\phi$ (deg)', font='Arial', fontsize=16)
            a0.set_xlabel(r'$\phi$ (deg)', font='Arial', fontsize=16)
        # a.set_xticklabels(labels=a.get_xticklabels(),fontsize=10)
        # a.set_yticklabels(labels=a.get_yticklabels(),fontsize=10)
        cursor = Cursor(a, useblit=True, color='red', linewidth=1)
    if pflag == 2:
        f, a = plt.subplots(2, 1, dpi=150)
        if value1.get() == 'MDC fitted Data':
            x = (fev-21.2)*1000

            a[0].set_title('MDC Fitting Result', font='Arial', fontsize=18)
            a[0].set_xlabel('Binding Energy (meV)', font='Arial', fontsize=14)
            a[0].set_ylabel(
                r'Position ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=14)
            a[0].tick_params(direction='in')
            a[0].scatter(x, pos, c='black', s=5)

            a[1].set_xlabel('Binding Energy (meV)', font='Arial', fontsize=14)
            a[1].set_ylabel(
                r'FWHM ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=14)
            a[1].tick_params(direction='in')
            a[1].scatter(x, fwhm, c='black', s=5)
        elif value1.get() == 'EDC fitted Data':
            x = fk

            a[0].set_title('EDC Fitting Result', font='Arial', fontsize=18)
            a[0].set_xlabel(
                r'Position ($\frac{2\pi}{\AA}$', font='Arial', fontsize=14)
            a[0].set_ylabel('Binding Energy (meV))', font='Arial', fontsize=14)
            a[0].tick_params(direction='in')
            a[0].scatter(x, (epos-21.2)*1000, c='black', s=5)

            a[1].set_xlabel(
                r'Position ($\frac{2\pi}{\AA}$', font='Arial', fontsize=14)
            a[1].set_ylabel('FWHM (meV)', font='Arial', fontsize=14)
            a[1].tick_params(direction='in')
            a[1].scatter(x, efwhm*1000, c='black', s=5)
        elif value1.get() == 'Real Part':
            x = (fev-21.2)*1000
            y = pos
            a[0].set_title('Real Part', font='Arial', fontsize=18)
            a[0].plot(rx, ry, c='black', linestyle='-', marker='.')

            a[0].tick_params(direction='in')
            a[0].set_xlabel('Binding Energy (meV)', font='Arial', fontsize=14)
            a[0].set_ylabel(r'Re $\Sigma$ (meV)', font='Arial', fontsize=14)

            h1 = a[1].scatter(y, x, c='black', s=5)
            h2 = a[1].scatter(k*np.float64(bbk_offset.get()),
                              be+np.float64(bb_offset.get()), c='red', s=5)

            a[1].legend([h1, h2], ['fitted data', 'bare band'])
            a[1].tick_params(direction='in')
            a[1].set_ylabel('Binding Energy (meV)', font='Arial', fontsize=14)
            a[1].set_xlabel(
                r'Pos ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=14)

            # a[0].set_xlim([-1000,50])
            # a[0].set_ylim([-100,500])
            # a[1].set_ylim([-600,200])
            # a[1].set_xlim([-0.05,0.05])
        elif value1.get() == 'Imaginary Part':

            tbe = (fev-21.2)*1000

            x = interp(tbe, be+np.float64(bb_offset.get()),
                       k*np.float64(bbk_offset.get()))
            y = interp(x, k*np.float64(bbk_offset.get()),
                       be+np.float64(bb_offset.get()))
            xx = np.diff(x)
            yy = np.diff(y)

            # eliminate vf in gap
            for i in range(len(yy)):
                if yy[i]/xx[i] > 20000:
                    yy[i] = 0
            v = yy/xx
            v = np.append(v, v[-1])  # fermi velocity
            yy = v*fwhm/2
            xx = tbe
            ax = a
            a = ax[0]
            b = ax[1]
            a.set_title('Imaginary Part', font='Arial', fontsize=18)
            a.plot(xx, yy, c='black', linestyle='-', marker='.')

            ix = xx
            iy = yy
            a.tick_params(direction='in')
            a.set_xlabel('Binding Energy (meV)', font='Arial', fontsize=14)
            a.set_ylabel(r'Im $\Sigma$ (meV)', font='Arial', fontsize=14)

            x = (fev-21.2)*1000
            y = fwhm
            b.plot(x, y, c='black', linestyle='-', marker='.')
            b.tick_params(direction='in')
            b.set_xlabel('Binding Energy (meV)', font='Arial', fontsize=14)
            b.set_ylabel(r'FWHM ($\frac{2\pi}{\AA}$)',
                         font='Arial', fontsize=14)

            x = (fev-21.2)*1000
            y = pos
            yy = interp(y, k*np.float64(bbk_offset.get()), be +
                        np.float64(bb_offset.get()))  # interp x into be,k set
    if pflag == 3:
        if value2.get() == 'Real & Imaginary':
            f, a = plt.subplots(2, 1, dpi=150)
            a[0].set_title(r'Self Energy $\Sigma$', font='Arial', fontsize=18)
            a[0].plot(rx, ry, c='black', linestyle='-', marker='.')
            a[0].tick_params(direction='in')
            a[0].set_xlabel('Binding Energy (meV)', font='Arial', fontsize=14)
            a[0].set_ylabel(r'Re $\Sigma$ (meV)', font='Arial', fontsize=14)
            a[1].plot(ix, iy, c='black', linestyle='-', marker='.')
            a[1].tick_params(direction='in')
            a[1].set_xlabel('Binding Energy (meV)', font='Arial', fontsize=14)
            a[1].set_ylabel(r'Im $\Sigma$ (meV)', font='Arial', fontsize=14)
        elif value2.get() == 'KK Transform':
            f, ax = plt.subplots(2, 1, dpi=150)
            a = ax[0]
            b = ax[1]
            a.set_title('Self Energy', font='Arial', fontsize=18)
            a.plot(rx, ry, c='black', linestyle='-',
                   marker='.', label=r'Re $\Sigma$')
            a.tick_params(direction='in')
            a.set_xlabel('Binding Energy (meV)', font='Arial', fontsize=14)
            a.set_ylabel(r'Re $\Sigma$ (meV)', font='Arial', fontsize=14)

            kx = ix
            ky = np.linspace(0, 1, len(kx))
            # de=np.linspace(0,1,len(kx))
            # de[0:-1]=np.diff(kx)
            # de[-1]=de[-2]
            de = np.diff(kx)
            de = np.append(de, de[-1])

            for i in range(len(kx)):
                # ky[i]=np.trapz(y=iy/(iy-kx[i]),x=iy,dx=de)
                intg = 0
                for j in range(len(kx)):
                    if i != j:
                        tval = iy[j]/(kx[j]-kx[i])*de[j]
                        if str(iy[j]) == 'nan':
                            tval = 0
                        intg += tval
                ky[i] = -1/np.pi*intg
            a.plot(kx, ky, c='red', linestyle='-', marker='.',
                   label=r'Re $\Sigma_{KK}$=KK(Im $\Sigma$)')
            handles, labels = a.get_legend_handles_labels()
            a.legend(handles, labels)
            # a.legend([h1,h2],['measured data','KK transform'])

            #   KK Re
            b.plot(ix, iy, c='black', linestyle='-',
                   marker='.', label=r'Im $\Sigma$')
            b.tick_params(direction='in')
            b.set_xlabel('Binding Energy (meV)', font='Arial', fontsize=14)
            b.set_ylabel(r'Im $\Sigma$ (meV)', font='Arial', fontsize=14)

            kx = rx
            ky = np.linspace(0, 1, len(kx))
            # de=np.linspace(0,1,len(kx))
            # de[0:-1]=np.diff(kx)
            # de[-1]=de[-2]
            de = np.diff(kx)
            de = np.append(de, de[-1])

            for i in range(len(kx)):
                # ky[i]=np.trapz(y=iy/(iy-kx[i]),x=iy,dx=de)
                intg = 0
                for j in range(len(kx)):
                    if i != j:
                        tval = ry[j]/(kx[j]-kx[i])*de[j]
                        if str(ry[j]) == 'nan':
                            tval = 0
                        intg += tval
                ky[i] = 1/np.pi*intg
            b.plot(kx, ky, c='red', linestyle='-', marker='.',
                   label=r'Im $\Sigma_{KK}$=KK(Re $\Sigma$)')
            handles, labels = b.get_legend_handles_labels()
            b.legend(handles, labels)
        elif value2.get() == 'Data Plot with Pos' or value2.get() == 'Data Plot with Pos and Bare Band':
            f0 = plt.figure(figsize=(8, 7), layout='constrained')
            a0 = plt.axes([0.13, 0.45, 0.8, 0.5])
            a1 = plt.axes([0.13, 0.08, 0.8, 0.2])
            a0.set_title('Drag to select specific region')
            selectors.append(RectangleSelector(
                a0, select_callback,
                useblit=True,
                button=[1, 3],  # disable middle button
                minspanx=5, minspany=5,
                spancoords='pixels',
                interactive=True,
                props=props))
            # f0.canvas.mpl_connect('key_press_event',toggle_selector)
            f, a = plt.subplots(dpi=150)
            px, py = np.meshgrid(phi, ev)
            px = (2*m*np.full_like(np.zeros([len(phi), len(ev)], dtype=float), ev)*1.6*10**-19).transpose(
            )**0.5*np.sin((np.float64(k_offset.get())+px)/180*np.pi)*10**-10/(h/2/np.pi)
            pz = data.to_numpy()
            h1 = a.pcolormesh(px, py, pz, cmap=value3.get())
            cb = f.colorbar(h1)
            cb.set_ticklabels(cb.get_ticks(), font='Arial',
                              fontsize=14, minor=False)
            a.set_title(value2.get(), font='Arial', fontsize=18)
            a.set_xlabel(r'k ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=16)
            a.set_ylabel('Kinetic Energy (eV)', font='Arial', fontsize=16)
            try:
                if mp == 1:
                    a.scatter(pos, fev, marker='.', s=0.3, c='black')
                if mf == 1:
                    ophimin = np.arcsin(
                        (rpos-fwhm/2)/(2*m*fev*1.6*10**-19)**0.5/10**-10*(h/2/np.pi))*180/np.pi
                    ophimax = np.arcsin(
                        (rpos+fwhm/2)/(2*m*fev*1.6*10**-19)**0.5/10**-10*(h/2/np.pi))*180/np.pi
                    posmin = (2*m*fev*1.6*10**-19)**0.5*np.sin(
                        (np.float64(k_offset.get())+ophimin)/180*np.pi)*10**-10/(h/2/np.pi)
                    posmax = (2*m*fev*1.6*10**-19)**0.5*np.sin(
                        (np.float64(k_offset.get())+ophimax)/180*np.pi)*10**-10/(h/2/np.pi)
                    a.scatter([posmin, posmax], [fev, fev],
                              marker='|', c='grey', s=10, alpha=0.8)
            except:
                pass
            try:
                if ep == 1:
                    a.scatter(fk, epos, marker='.', s=0.3, c='black')
                if ef == 1:
                    eposmin = epos-efwhm/2
                    eposmax = epos+efwhm/2
                    a.scatter([fk, fk], [eposmin, eposmax],
                              marker='_', c='grey', s=10, alpha=0.8)
            except:
                pass
            h2 = a0.pcolormesh(px, py, pz, cmap=value3.get())
            cb1 = f0.colorbar(h2)
            cb1.set_ticklabels(cb1.get_ticks(), font='Arial',
                               fontsize=14, minor=False)
            a0.set_xlabel(r'k ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=16)
            a0.set_ylabel('Kinetic Energy (eV)', font='Arial', fontsize=16)
            try:
                if mp == 1:
                    a0.scatter(pos, fev, marker='.', s=0.3, c='black')
                if mf == 1:
                    ophimin = np.arcsin(
                        (rpos-fwhm/2)/(2*m*fev*1.6*10**-19)**0.5/10**-10*(h/2/np.pi))*180/np.pi
                    ophimax = np.arcsin(
                        (rpos+fwhm/2)/(2*m*fev*1.6*10**-19)**0.5/10**-10*(h/2/np.pi))*180/np.pi
                    posmin = (2*m*fev*1.6*10**-19)**0.5*np.sin(
                        (np.float64(k_offset.get())+ophimin)/180*np.pi)*10**-10/(h/2/np.pi)
                    posmax = (2*m*fev*1.6*10**-19)**0.5*np.sin(
                        (np.float64(k_offset.get())+ophimax)/180*np.pi)*10**-10/(h/2/np.pi)
                    a0.scatter([posmin, posmax], [fev, fev],
                               marker='|', c='grey', s=10, alpha=0.8)
            except:
                pass
            try:
                if ep == 1:
                    a0.scatter(fk, epos, marker='.', s=0.3, c='black')
                if ef == 1:
                    eposmin = epos-efwhm/2
                    eposmax = epos+efwhm/2
                    a0.scatter([fk, fk], [eposmin, eposmax],
                               marker='_', c='grey', s=10, alpha=0.8)
            except:
                pass
            # b.set_xticklabels(labels=b.get_xticklabels(),font='Arial',fontsize=14)
            # b.set_yticklabels(labels=b.get_yticklabels(),font='Arial',fontsize=14)

            n = a1.hist(pz.flatten(), bins=np.linspace(
                min(pz.flatten()), max(pz.flatten()), 50), color='green')
            a1.set_xlabel('Intensity')
            a1.set_ylabel('Counts')
            a1.set_title('Drag to Select the range of Intensity ')
            selectors.append(SpanSelector(
                a1,
                onselect,
                "horizontal",
                useblit=True,
                props=dict(alpha=0.3, facecolor="tab:blue"),
                onmove_callback=onmove_callback,
                interactive=True,
                drag_from_anywhere=True,
                snap_values=n[1]
            ))
            try:
                if value2.get() == 'Data Plot with Pos and Bare Band':
                    a.plot(k*np.float64(bbk_offset.get()), (be +
                           np.float64(bb_offset.get()))/1000+21.2, linewidth=0.3, c='red')
                    a0.plot(k*np.float64(bbk_offset.get()), (be +
                            np.float64(bb_offset.get()))/1000+21.2, linewidth=0.3, c='red')
            except:
                pass
            cursor = Cursor(a, useblit=True, color='red', linewidth=1)
    try:
        if value1.get() == '---Plot2---' and value2.get() != 'Real & Imaginary' and value2.get() != 'KK Transform':
            try:
                h1.set_clim([vcmin.get(), vcmax.get()])
                h2.set_clim([vcmin.get(), vcmax.get()])
            except:
                pass
            a0.set_xlim(xl)
            a0.set_ylim(yl)
            if value.get() != 'Raw Data':
                plt.tight_layout()
            # if value.get()=='Raw Data':
            #     plt.connect('motion_notify_event', cut_move)
            plt.show()
            try:
                h1.set_clim([cm.get(), cM.get()])
                h2.set_clim([cm.get(), cM.get()])
            except:
                pass
        else:
            plt.tight_layout()
            plt.show()
        # f.ion()
        # f0.ion()
    except:
        print('fail to export graph')
        pass

    # fp=fd.asksaveasfilename(filetypes=(("PNG files", "*.png"),))
    # f.savefig(fname=fp)


def move(event):
    global xdata, ydata, x1, y1, x2, y2
    if event.xdata != None:
        out.get_tk_widget().config(cursor="crosshair")
        try:
            out.get_tk_widget().delete('rec')
            # out.get_tk_widget().delete('x1')
            # out.get_tk_widget().delete('y1')
            # out.get_tk_widget().delete('x2')
            # out.get_tk_widget().delete('y2')
        except:
            pass
        if mof == -1 and value1.get() == '---Plot2---' and value2.get() != 'Real & Imaginary' and value2.get() != 'KK Transform':
            x2, y2 = event.xdata, event.ydata
            px2, py2 = event.x, event.y
            out.get_tk_widget().create_rectangle((px1, 600-py1), (px2, 600-py2),
                                                 outline='black', width=2, tag='rec')
        if value.get() == 'Raw Data':
            if event.inaxes:
                cxdata = event.xdata
                cydata = event.ydata
                xf = (cxdata > ao.get_xlim()[0] and cxdata < ao.get_xlim()[1])
                yf = (cydata > ao.get_ylim()[0] and cydata < ao.get_ylim()[1])
                if xf and yf:
                    dx = data.sel(
                        eV=cydata, method='nearest').to_numpy().reshape(len(phi))
                    dy = data.sel(
                        phi=cxdata, method='nearest').to_numpy().reshape(len(ev))
                    # try:
                    #     x.remove()
                    #     y.remove()
                    # except:
                    #     pass
                    # x=a.axvline(cxdata,color='r')
                    # y=a.axhline(cydata,color='r')
                    rcx.clear()
                    rcy.clear()
                    rcx.plot(phi, dx, c='black')
                    rcy.plot(dy, ev, c='black')
                    rcx.set_xticks([])
                    rcy.set_yticks([])
                    rcx.set_xlim(ao.get_xlim())
                    rcy.set_ylim(ao.get_ylim())
                    out.draw_idle()
        xdata.config(text='xdata:'+str('%.3f' % event.xdata))
        ydata.config(text='ydata:'+str('%.3f' % event.ydata))
    else:
        out.get_tk_widget().config(cursor="")
        xdata.config(text='xdata:')
        ydata.config(text='ydata:')

    # print("event.xdata", event.xdata)
    # print("event.ydata", event.ydata)
    # print("event.inaxes", event.inaxes)
    # print("x", event.x)
    # print("y", event.y)
mof = 1


def press(event):
    # event.button 1:left 3:right 2:mid
    # event.dblclick : bool
    # print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
    #       ('double' if event.dblclick else 'single', event.button,
    #        event.x, event.y, event.xdata, event.ydata))
    global x1, y1, mof, px1, py1, ao, bo, out, tb0, tb0_, tb1, tb1_, tb2
    if event.button == 1:
        x1, y1 = event.xdata, event.ydata
        px1, py1 = event.x, event.y
        mof = -1
    elif event.button == 3 and value1.get() == '---Plot2---' and value2.get() != 'Real & Imaginary' and value2.get() != 'KK Transform':
        if value2.get() == '---Plot3---':
            ao.set_xlim(xl)
            ao.set_ylim(yl)
            out.draw()
        else:
            bo.set_xlim(xl)
            bo.set_ylim(yl)
            try:
                if mp == 1:
                    tb0.remove()
                    tb0 = bo.scatter(pos, fev, marker='.', s=0.3, c='black')
                if mf == 1:
                    tb0_.remove()
                    ophimin = np.arcsin(
                        (rpos-fwhm/2)/(2*m*fev*1.6*10**-19)**0.5/10**-10*(h/2/np.pi))*180/np.pi
                    ophimax = np.arcsin(
                        (rpos+fwhm/2)/(2*m*fev*1.6*10**-19)**0.5/10**-10*(h/2/np.pi))*180/np.pi
                    posmin = (2*m*fev*1.6*10**-19)**0.5*np.sin(
                        (np.float64(k_offset.get())+ophimin)/180*np.pi)*10**-10/(h/2/np.pi)
                    posmax = (2*m*fev*1.6*10**-19)**0.5*np.sin(
                        (np.float64(k_offset.get())+ophimax)/180*np.pi)*10**-10/(h/2/np.pi)
                    tb0_ = bo.scatter([posmin, posmax], [
                                      fev, fev], marker='|', c='grey', s=10, alpha=0.8)
            except:
                pass
            try:
                if ep == 1:
                    tb1.remove()
                    tb1 = bo.scatter(fk, epos, marker='.', s=0.3, c='black')
                if ef == 1:
                    tb1_.remove()
                    eposmin = epos-efwhm/2
                    eposmax = epos+efwhm/2
                    tb1_ = bo.scatter(
                        [fk, fk], [eposmin, eposmax], marker='_', c='grey', s=10, alpha=0.8)
            except:
                pass
            try:
                if value2.get() == 'Data Plot with Pos and Bare Band':
                    tb2.remove()
                    tb2, = bo.plot(k*np.float64(bbk_offset.get()), (be +
                                   np.float64(bb_offset.get()))/1000+21.2, linewidth=0.3, c='red')
            except:
                pass
            out.draw()
        mof = 1


def release(event):
    global x2, y2, mof, tb0, tb0_, tb1, tb1_, tb2, out, ao, bo
    try:
        out.get_tk_widget().delete('rec')
    except:
        pass
    if event.button == 1 and mof == -1 and value1.get() == '---Plot2---' and value2.get() != 'Real & Imaginary' and value2.get() != 'KK Transform':
        x2, y2 = event.xdata, event.ydata
        if value2.get() == '---Plot3---':
            ao.set_xlim(sorted([x1, x2]))
            ao.set_ylim(sorted([y1, y2]))
            out.draw()
        else:
            bo.set_xlim(sorted([x1, x2]))
            bo.set_ylim(sorted([y1, y2]))
            if abs(x1-x2) < (xl[1]-xl[0])/3*2 or abs(y1-y2) < (yl[1]-yl[0])/3*2:
                try:
                    if mp == 1:
                        tb0.remove()
                    if mf == 1:
                        tb0_.remove()
                except:
                    pass
                try:
                    if ep == 1:
                        tb1.remove()
                    if ef == 1:
                        tb1_.remove()
                except:
                    pass
                try:
                    tb2.remove()
                except:
                    pass
                if value2.get() == 'Data Plot with Pos' or value2.get() == 'Data Plot with Pos and Bare Band':
                    try:
                        if mp == 1:
                            tb0 = bo.scatter(
                                pos, fev, marker='.', s=30, c='black')
                        if mf == 1:
                            ophimin = np.arcsin(
                                (rpos-fwhm/2)/(2*m*fev*1.6*10**-19)**0.5/10**-10*(h/2/np.pi))*180/np.pi
                            ophimax = np.arcsin(
                                (rpos+fwhm/2)/(2*m*fev*1.6*10**-19)**0.5/10**-10*(h/2/np.pi))*180/np.pi
                            posmin = (2*m*fev*1.6*10**-19)**0.5*np.sin(
                                (np.float64(k_offset.get())+ophimin)/180*np.pi)*10**-10/(h/2/np.pi)
                            posmax = (2*m*fev*1.6*10**-19)**0.5*np.sin(
                                (np.float64(k_offset.get())+ophimax)/180*np.pi)*10**-10/(h/2/np.pi)
                            tb0_ = bo.scatter([posmin, posmax], [
                                              fev, fev], marker='|', c='grey', s=50, alpha=0.8)

                    except:
                        pass
                    try:
                        if ep == 1:
                            tb1 = bo.scatter(
                                fk, epos, marker='.', s=30, c='black')
                        if ef == 1:
                            eposmin = epos-efwhm/2
                            eposmax = epos+efwhm/2
                            tb1_ = bo.scatter(
                                [fk, fk], [eposmin, eposmax], marker='_', c='grey', s=50, alpha=0.8)
                    except:
                        pass
                    if value2.get() == 'Data Plot with Pos and Bare Band':
                        tb2, = bo.plot(k*np.float64(bbk_offset.get()), (be +
                                       np.float64(bb_offset.get()))/1000+21.2, linewidth=5, c='red')
            else:
                try:
                    if mp == 1:
                        tb0.remove()
                        tb0 = bo.scatter(pos, fev, marker='.',
                                         s=0.3, c='black')
                    if mf == 1:
                        tb0_.remove()
                        ophimin = np.arcsin(
                            (rpos-fwhm/2)/(2*m*fev*1.6*10**-19)**0.5/10**-10*(h/2/np.pi))*180/np.pi
                        ophimax = np.arcsin(
                            (rpos+fwhm/2)/(2*m*fev*1.6*10**-19)**0.5/10**-10*(h/2/np.pi))*180/np.pi
                        posmin = (2*m*fev*1.6*10**-19)**0.5*np.sin(
                            (np.float64(k_offset.get())+ophimin)/180*np.pi)*10**-10/(h/2/np.pi)
                        posmax = (2*m*fev*1.6*10**-19)**0.5*np.sin(
                            (np.float64(k_offset.get())+ophimax)/180*np.pi)*10**-10/(h/2/np.pi)
                        tb0_ = bo.scatter([posmin, posmax], [
                                          fev, fev], marker='|', c='grey', s=10, alpha=0.8)
                except:
                    pass
                try:
                    if ep == 1:
                        tb1.remove()
                        tb1 = bo.scatter(fk, epos, marker='.',
                                         s=0.3, c='black')
                    if ef == 1:
                        tb1_.remove()
                        eposmin = epos-efwhm/2
                        eposmax = epos+efwhm/2
                        tb1_ = bo.scatter(
                            [fk, fk], [eposmin, eposmax], marker='_', c='grey', s=10, alpha=0.8)
                except:
                    pass
                try:
                    if value2.get() == 'Data Plot with Pos and Bare Band':
                        tb2.remove()
                        tb2, = bo.plot(k*np.float64(bbk_offset.get()), (be+np.float64(
                            bb_offset.get()))/1000+21.2, linewidth=0.3, c='red')
                except:
                    pass
            out.draw()
        mof = 1


def angcut():
    t0 = threading.Thread(target=o_angcut)
    t0.daemon = True
    t0.start()


def ecut():
    t1 = threading.Thread(target=o_ecut)
    t1.daemon = True
    t1.start()


def loadmfit():
    t2 = threading.Thread(target=o_loadmfit)
    t2.daemon = True
    t2.start()


def loadefit():
    t3 = threading.Thread(target=o_loadefit)
    t3.daemon = True
    t3.start()


def reload(*e):
    t4 = threading.Thread(target=o_reload)
    t4.daemon = True
    t4.start()


def expte():
    t5 = threading.Thread(target=o_expte)
    t5.daemon = True
    t5.start()


def exptm():
    t6 = threading.Thread(target=o_exptm)
    t6.daemon = True
    t6.start()


def bareband():
    t7 = threading.Thread(target=o_bareband)
    t7.daemon = True
    t7.start()


def plot1(*e):
    t8 = threading.Thread(target=o_plot1)
    t8.daemon = True
    t8.start()


def plot2(*e):
    t9 = threading.Thread(target=o_plot2)
    t9.daemon = True
    t9.start()


def plot3(*e):
    if value2.get() == 'Data Plot with Pos' or value2.get() == 'Data Plot with Pos and Bare Band':
        def ini():
            global mp, ep, mf, ef
            if len(fev) <= 0:
                mp = 0
                mpos.deselect()
                mpos.config(state='disabled')
                mf = 0
                mfwhm.deselect()
                mfwhm.config(state='disabled')
            if len(fk) <= 0:
                ep = 0
                epos.deselect()
                epos.config(state='disabled')
                ef = 0
                efwhm.deselect()
                efwhm.config(state='disabled')

        def chf():
            global mp, ep, mf, ef
            mp = v_mpos.get()
            ep = v_epos.get()
            mf = v_mfwhm.get()
            ef = v_efwhm.get()
            t10 = threading.Thread(target=o_plot3)
            t10.daemon = True
            t10.start()
            gg.destroy()

        gg = tk.Toplevel(g, bg="white", padx=10, pady=10)
        gg.title('Data Point List')
        gg.iconphoto(False, tk.PhotoImage(data=b64decode(gicon)))
        lpos = tk.Label(gg, text='Position', font=(
            "Arial", 18, "bold"), bg="white", height='1')
        lpos.grid(row=0, column=0, padx=10, pady=10)

        pos = tk.Frame(gg, bg="white")
        pos.grid(row=1, column=0, padx=10, pady=5)
        v_mpos = tk.IntVar()
        mpos = tk.Checkbutton(pos, text="MDC", font=(
            "Arial", 16, "bold"), variable=v_mpos, onvalue=1, offvalue=0, height=2, width=10, bg="white")
        mpos.grid(row=0, column=0, padx=10, pady=5)
        mpos.intvar = v_mpos
        mpos.select()

        v_epos = tk.IntVar()
        epos = tk.Checkbutton(pos, text="EDC", font=(
            "Arial", 16, "bold"), variable=v_epos, onvalue=1, offvalue=0, height=2, width=10, bg="white")
        epos.grid(row=0, column=1, padx=10, pady=5)
        epos.intvar = v_epos
        epos.select()

        lfwhm = tk.Label(gg, text='FWHM', font=(
            "Arial", 18, "bold"), bg="white", height='1')
        lfwhm.grid(row=2, column=0, padx=10, pady=10)

        fwhm = tk.Frame(gg, bg="white")
        fwhm.grid(row=3, column=0, padx=10, pady=5)
        v_mfwhm = tk.IntVar()
        mfwhm = tk.Checkbutton(fwhm, text="MDC", font=(
            "Arial", 16, "bold"), variable=v_mfwhm, onvalue=1, offvalue=0, height=2, width=10, bg="white")
        mfwhm.grid(row=0, column=0, padx=10, pady=5)
        mfwhm.intvar = v_mfwhm
        mfwhm.select()

        v_efwhm = tk.IntVar()
        efwhm = tk.Checkbutton(fwhm, text="EDC", font=(
            "Arial", 16, "bold"), variable=v_efwhm, onvalue=1, offvalue=0, height=2, width=10, bg="white")
        efwhm.grid(row=0, column=1, padx=10, pady=5)
        efwhm.intvar = v_efwhm
        efwhm.select()

        bflag = tk.Button(gg, text="OK", font=("Arial", 16, "bold"),
                          height=2, width=10, bg="white", command=chf)
        bflag.grid(row=4, column=0, padx=10, pady=5)

        ini()
    else:
        t10 = threading.Thread(target=o_plot3)
        t10.daemon = True
        t10.start()


def load():
    t11 = threading.Thread(target=o_load)
    t11.daemon = True
    t11.start()


def fitgl():
    t12 = threading.Thread(target=o_fitgl)
    t12.daemon = True
    t12.start()


def tstate():
    try:
        while True:
            state.config(text=str(st.get()))
    except KeyboardInterrupt:
        pass


try:
    with np.load('rd.npz', 'rb') as f:
        path = str(f['path'])
        name = str(f['name'])
        ev = f['ev']
        phi = f['phi']
        st = str(f['st'])
        key = f['key']
        lst = f['lst']
        print('Raw Data preloaded:\n\n')
        print(st+'\n')
        if '.h5' in path:
            data = load_h5(path)
        elif '.json' in path:
            data = load_json(path)
        else:
            data = load_txt(path)
        rdd = path
except:
    print('No Raw Data preloaded')

try:
    with np.load('bb.npz', 'rb') as f:
        path = str(f['path'])
        be = f['be']
        k = f['k']
        print('Bare Band file preloaded:')
        print(path+'\n')
except:
    print('No Bare Band file preloaded')

try:
    with np.load('efpath.npz', 'rb') as f:
        efpath = str(f['efpath'])
        print('EDC Fitted path preloaded')
except:
    print('No EDC Fitted path preloaded')

try:
    with np.load('mfpath.npz', 'rb') as f:
        mfpath = str(f['mfpath'])
        print('MDC Fitted path preloaded')
except:
    print('No MDC Fitted path preloaded')

try:
    with np.load('efit.npz', 'rb') as f:
        ko = str(f['ko'])
        fphi = f['fphi']
        epos = f['epos']
        ffphi = f['ffphi']
        efwhm = f['efwhm']
        fk = f['fk']
        emin = f['emin']
        emax = f['emax']
        semin = f['semin']
        semax = f['semax']
        seaa1 = f['seaa1']
        seaa2 = f['seaa2']
        sefp = f['sefp']
        sefi = f['sefi']
        print('EDC Fitted Data preloaded (Casa)')
    fpr = 1
except:
    print('No EDC fitted data preloaded (Casa)')

try:
    with np.load('mfit.npz', 'rb') as f:
        ko = str(f['ko'])
        fev = f['fev']
        rpos = f['rpos']
        ophi = f['ophi']
        fwhm = f['fwhm']
        pos = f['pos']
        kmin = f['kmin']
        kmax = f['kmax']
        skmin = f['skmin']
        skmax = f['skmax']
        smaa1 = f['smaa1']
        smaa2 = f['smaa2']
        smfp = f['smfp']
        smfi = f['smfi']
        print('MDC Fitted Data preloaded (Casa)')
        try:
            smresult = f['smresult']
            print('MDC Fitted Data preloaded (lmfit)')
        except:
            pass
    fpr = 1
except:
    print('No MDC fitted data preloaded (Casa)')

'''
try:
    with np.load('efpara.npz','rb') as f:
        rdd=f['path']
        fphi=f['fphi']
        efwhm=f['efwhm']
        epos=f['epos']
        semin=f['semin']
        semax=f['semax']
        seaa1=f['seaa1']
        seaa2=f['seaa2']
        sefp=f['sefp']
        sefi=f['sefi']
        print('EDC Fitted Data preloaded')
except:
    print('No EDC fitted data preloaded')
    
try:
    with np.load('mfpara.npz','rb') as f:
        rdd=f['path']
        fev=f['fev']
        fwhm=f['fwhm']
        pos=f['pos']
        skmin=f['skmin']
        skmax=f['skmax']
        smaa1=f['smaa1']
        smaa2=f['smaa2']
        smfp=f['smfp']
        smfi=f['smfi']
        print('MDC Fitted Data preloaded')
except:
    print('No MDC fitted data preloaded')
'''


def lm2p():
    t = threading.Thread(target=loadmfit_2p)
    t.daemon = True
    t.start()


def lmre():
    t = threading.Thread(target=loadmfit_re)
    t.daemon = True
    t.start()


def lm():
    t = threading.Thread(target=loadmfit_)
    t.daemon = True
    t.start()


def o_loadmfit():
    global g, st, lmgg
    lmgg = tk.Toplevel(g)
    lmgg.title('Load MDC fitted File')
    lmgg.geometry('400x200')  # format:'1400x800'
    b1 = tk.Button(lmgg, command=lm2p, text='vms 1 peak to 2 peaks', font=(
        "Arial", 12, "bold"), fg='red', width=30, height='1', bd=10)
    b1.pack()
    b2 = tk.Button(lmgg, command=lmre, text='reverse vms axis', font=(
        "Arial", 12, "bold"), fg='red', width=30, height='1', bd=10)
    b2.pack()
    b3 = tk.Button(lmgg, command=lm, text='load MDC fitted File', font=(
        "Arial", 12, "bold"), fg='red', width=30, height='1', bd=10)
    b3.pack()
    lmgg.update()


g = tk.Tk()
windll.shcore.SetProcessDpiAwareness(1)
ScaleFactor = windll.shcore.GetScaleFactorForDevice(0)
g.tk.call('tk', 'scaling', ScaleFactor/75)
g.title('MDC cut')
g.config(bg='white')
g.geometry('1920x980')  # format:'1400x800'
g.resizable(True, True)
icon = "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"
icon1 = "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"
icon2 = "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"
icon3 = "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"
icon4 = "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iigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACkpaSgCvfXH2e1kl6BRXxD+0NrT6l46u3U5j8sCvsvxhMYfDl4VPzbOK+BfHV5cXuvXTMjNyRyPevls+quNFQXc+q4ep3ruo+3+RzMZBxnrXIeL5z80ee9dZIrqu4Rtx7V5/4kmllvD+6cj6V8bhY807n3WIfumT0Sul8HwFpN/Sua2S7TmJ/yruPCNq8dikhjbr6V6GIlamclFXkdDtznJoVd7qo5p8NpJcSYVG57Yr1j4Z/CG58Q30LSxFYcbjkc15lDD1MRJRgjpr4inhouVR2sa3wD+HE+saobyWP91Gw7V9g2FrFZ20cUY2qowAKxPBPhG18KaakUCbWI+biukx7V+mZfhFg6Kgtz8xzDGPGVnNbC0lLRXpHljOn1rmvGnjKDwppsksxG7af5Vs6xqsej2T3EpAC8mvjv43/FO51vWruG3J+yE4Ra8vMMZHB0ubqerl+BljatuhwXj7xlJ4o1SaSRmKeY2BntmuTdtzDaOKRmaQ5ZDzz0oy68BDt+lfmFSbqy5p7s/U6VONKPJFWSA9aSjDf3Gow/9xqgsKKMP/cajD/3GoAKKMP/AHGow/8AcagApyNtNN2t/cb8qCGPRGqozcHzR6ClFTXK+pdRwwpc1SUuv8LCp42fP3WzX0WHzKLSVS9zwK2XyTbgycL0OKeIzMDtP1q9pOjXWpXMcaRMd3tXq3gz4FXuoyJJIhEZPNezR/f6w2PFrv2PxvU800Hwvd6y4W3jLnOOma+hfhn8B1Rbe4vgDnDFa9N8E/CfS/DcMcgizKOSGFd6sQjXCIqjtgV7VLCxjqzwa2MlP3YlTTtLg0uFYLZNkWOMVfpMbcAdKWu/0PMCiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKT1paKAK15ZxXls8Uq7kYYNecX3wT0e9upJTZRNu7kV6eaF6VjUo06vxq5tTrVKXwOx5K3wB0VlINlDz7Vi3f7MWh3Mu42UP5V7pQKxjhaMNom7xld7yPBj+y3ofP+hQn8K1LD9nfR7OBI1s4gAa9l/hoIzRLCUZLWILGV1tI810z4I6BZXCu+nwtj2rutN0Ox0kKLS1jhAGMqK0MdqQKF5Fawo06XwRsYTrVKnxybF5ooWlrfYxCiiigDF8TaGdcszCG4IxXmU37Pun3UjPNBHIx7tXsvRqBzXLWwtKu06ivY6qWKq4dWpu1zxb/hnbTP+faOj/hnbTf+faOvatoo2isP7Pw/8p0f2jif5jxX/hnbTf8An2jo/wCGdtN/59o69q2ijaKP7Pw/8of2jif5jxX/AIZ203/n2jo/4Z203/n2jr2raKNoo/s/D/yh/aOJ/mPFf+GdtN/59o6P+GdtN/59o69q2ijaKP7Pw/8AKH9o4n+Y8V/4Z203/n2jo/4Z203/AJ9o69q2ijaKP7Pw/wDKH9o4n+Y8V/4Z30zIzbR1o2vwB0SLBazhY/SvWGUYpMYprA4eLvykyx+JkrORyuk/DPw/pca7NNhEg/ixzXSWtlBZrtgiWNfYVODzTq7IxjH4VY4pTlL4ncT60tJ3pa0ICiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAP//Z"
icon5 = "/9j/4AAQSkZJRgABAQAA2ADYAAD/4QBYRXhpZgAATU0AKgAAAAgAAgESAAMAAAABAAEAAIdpAAQAAAABAAAAJgAAAAAAA6ABAAMAAAAB//8AAKACAAQAAAABAAABbaADAAQAAAABAAABbQAAAAD/7QA4UGhvdG9zaG9wIDMuMAA4QklNBAQAAAAAAAA4QklNBCUAAAAAABDUHYzZjwCyBOmACZjs+EJ+/+ICKElDQ19QUk9GSUxFAAEBAAACGGFwcGwEAAAAbW50clJHQiBYWVogB+YAAQABAAAAAAAAYWNzcEFQUEwAAAAAQVBQTAAAAAAAAAAAAAAAAAAAAAAAAPbWAAEAAAAA0y1hcHBs7P2jjjiFR8NttL1PetoYLwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKZGVzYwAAAPwAAAAwY3BydAAAASwAAABQd3RwdAAAAXwAAAAUclhZWgAAAZAAAAAUZ1hZWgAAAaQAAAAUYlhZWgAAAbgAAAAUclRSQwAAAcwAAAAgY2hhZAAAAewAAAAsYlRSQwAAAcwAAAAgZ1RSQwAAAcwAAAAgbWx1YwAAAAAAAAABAAAADGVuVVMAAAAUAAAAHABEAGkAcwBwAGwAYQB5ACAAUAAzbWx1YwAAAAAAAAABAAAADGVuVVMAAAA0AAAAHABDAG8AcAB5AHIAaQBnAGgAdAAgAEEAcABwAGwAZQAgAEkAbgBjAC4ALAAgADIAMAAyADJYWVogAAAAAAAA9tUAAQAAAADTLFhZWiAAAAAAAACD3wAAPb////+7WFlaIAAAAAAAAEq/AACxNwAACrlYWVogAAAAAAAAKDgAABELAADIuXBhcmEAAAAAAAMAAAACZmYAAPKnAAANWQAAE9AAAApbc2YzMgAAAAAAAQxCAAAF3v//8yYAAAeTAAD9kP//+6L///2jAAAD3AAAwG7/wAARCAFtAW0DAREAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9sAQwACAgICAgIDAgIDBQMDAwUGBQUFBQYIBgYGBgYICggICAgICAoKCgoKCgoKDAwMDAwMDg4ODg4PDw8PDw8PDw8P/9sAQwECAwMEBAQHBAQHEAsJCxAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQ/90ABAAu/9oADAMBAAIRAxEAPwD9/KACgAoAKACgAoA5rxZ4hi8M6NLqbjcV+VAehY8/0rxc3zBZfgquMkr8qvY7sHhniK8KK6s8o8HfF6617XYdK1CBIlnLBWX2Ukfyr8m4Z44r5jj44PEwS5r2a8k3r9x9tm/DsMJhnXpyvbf77HvlfuZ+dhQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQB/9D9/KACgBCQASTgCgBkckcq7o2Dr6jkUASUAFAHlHxk/wCRQP8A12X/ANBavhuMf+RJifT9UfQZH/yMKXqfNnw//wCRx0z/AH2/9Aav5r4G/wCR9h/+3v8A0iR+u8Sf8i2r8vzR91V/Zp+BBQAUARLNEzmJXBdeozyKAJaACgAoAKACgAoAKACgAoAKACgAoAKACgAoAKACgAoAKACgAoAKACgAoAKACgAoAKACgAoAKACgAoAKAP/R/fygAoA8q+LN5rdnoKnSdyxsW810JBXpjp681+dcb4jF0MpnPB3vdXa3S1u/L1PpMip0KmNjGvt+p4H4O+Imr+Hb1VuZWuLckhkdj3OTjOcV+A8OcYYvLq/+0yc6b3u7teav+R+nZtkFHE0+ahFRkux9e6Rq9jrdkl9YSiSN/Q8g+h96/rfC4ujiqUa1CSlF9UfidWlOlNwqKzRqV2GJ5T8Y1LeEDgZ/fL/6C1fDcY/8iTE+n6o+gyT/AJGFL1Pmv4fAnxjpmBn52/8AQGr+a+Bv+R9h/wDt7/0iR+u8Sf8AItq/L80fdVf2afgQUAeQ/Eb4jQeH7d9M0xxJeyDBYHiPn+dfmHFfFdLKqTo0Xes1ou3m/wAT6vJsmqY2alLSC3Z4P4T8QeJrrxNbSW88szyypvUsxBG7oa/B+Hs4zWtm9FxqSlzSXMrtqzeumy0+4/SM1y/BUcDJOKTS073tofacZYxqXGGIGR71/Y5+GD6ACgAoAKACgAoAKACgAoAKACgAoAKACgAoAKACgAoAKACgAoAKACgAoAKACgAoAKACgAoAKACgAoA//9L9/KACgCC5toLyB7a5QSRSAqynoQeKzqU41IuE1dMqMnF3W58ofEf4cTaFM2p6Yu+zkboOqkjofyr+WeMeD5YGUsbg1ek91/L/AMD59T9fyHPlVSw+IfvdH3Oc8DeOb/wrfKGYvasf3iHpjHX8K8DhXiitlVZUp60pPVdm9Lr8D1M8yani6bqw0mj7M06+g1OwttStTmG6jSVD6q4yP0Nf2JSqRqQjUhs1c/DZRcW4vdFPWrPTNUsZdN1Nl8qUYIY4P1FcuNo0K9GVDEW5ZaNM1oTqU5qpS3R574W8CeENC1Mana3YuJos7d7ABSRgnr6GvhMl4XyrLcR9Yoz5p9Lvb01PoMdm2MxVP2VRaHrKsrruQgg9xX6UfLnlvxM8dHwrax2Nnxe3Sllbj5UHBI984r874t4kWUYZezV6ktv82fTZNlTx1Wzfurc+ULa31LxFqSxJummmbr1r+TKVPF5pi1FXnUmz9snPD4DD32ij668A+AbTwvZrcXCB76QAs393joK/rjhjhihlFC7V6rtd/ovI/EM1zWpjal72itkek19+fOBQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFAH/9P9/KACgAoAhuIIbqF7e4QPG4wynoRUThGcXGSumUm07o8NvfglZTav9rguhHZdWiYZYj69K/Gq3hzgp4v21ObjC9+X/LTY+2hxNiVQ9lJXfcj8Y/Ea08K2SeGfDYG+1jEAY8hFQbQB05HrRxLxnSytfUcAlKaVvKNvzKynIamNft6+kX+J88ajr+q6rN599cPNIOjMeRX8643OsdjJOWIqt36dD9Ww2V4XDpKEDMFxMvRjz715EK1SD5oSaZ6M6FOatKJ6F4W+JWu6BcxCSYz2qlQY25AUddvviv0bJeNswwM1GrL2kOqe/wAmfIZhw3hq8XKkuWR7frGh6N8WNMt9VsJxDdwqUyRkKpJyCMZ61+4Y7L8BxVgqdelOzX4b6PQ/OsPicTk+IlCUTo/BXw/0/wAJReYSJ7s9ZMYA+gr2+HuGMLk8H7PWb3b/ACXkcGZZrWx0k56JdD0KvuDwQoAKACgAoAKACgAoAKACgAoAKACgAoAKACgAoAKACgAoAKACgAoAKACgAoAKACgAoAKACgAoAKACgAoA/9T9/KAGllBwWFAw8xP7w/OgADK3Qg0COE+I3iNvDvhuaaBwtzP8kWe/I3f+O5r4/ifNP7Oy2riI/Fay+bS/C9z2sqwf1rFQpdOp8TXE8t1K88rF2ckkk5JzX8SVKkqk3Obu27t+Z/RVOnGnFQitEQ1kaBQAUAemfDHxRNoOvQxSSbbSc7JAT8oUnOcetfqPAucywWYxoSfuVNPJPSzPh+JcvjXwzrJe9E+z8jr2r+vz8PE8xP7w/OgYeYn94fnQAoIPIOaBC0AFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFAH/9X94/EF9daZot3f2MPnzwIWRB/EfSsqsnGDkty4pNpM+Mbzxv4pkuXeW9kZieT0/QCv4rrcW5zOpKbrtX6aafgfvNDIcB7NXhcq/wDCZ+Jf+fyT86w/1qzn/oJl+H+R0f2FgP8An2vxNnQfHXimDUomjuXkycFeDuHpXt5TxXnH1ymnVc03a1lr+B5mPyTARw8ny2fc9Z+OsrnTLBCOAxYexPBr9Q8TZyWDw8Vs5O/3I+R4Rinipt9EfMVfzIfsoUAFABQBasv+PqLv81deGnKFaEovVM5cTFSpSTPrf4qa9q+i+H7eTTGMbTNtdhjgDFf1rxtmuMwGXxng3Zydm+2357H4dkODoYnEuNfZdO580/8ACZ+Jf+fyT86/m/8A1qzn/oJl+H+R+r/2FgP+fa/H/MP+Ez8S/wDP5J+dH+tWc/8AQTL8P8g/sLAf8+1+J7H8I/FfiHUdWbT713uLZ1JJIGIyBnOevOMV+0cAZ7mGOq1aGLm5xSum/usfn3EuW4bCqE6Cs2fR1fvB+eBQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQB//9b9/KAORuPA3he5k82SyQMfQV8xW4dyutN1KuHi293Y9SGZYqEVGFRpEH/CvfCv/PmtY/6r5P8A9A0fu/4Jr/auM/5+MuWXgvw3p9wl1bWSCWM5ViOVPqK7cLkeXYWp7XD0VGXdHPVx+Jqx5ak20cX8ZNHl1Lw0t3Fk/YX3kDuHIXp7ZzXyPHeWyxeVSnBXlDVfer/ge3w9i1QxseZ6S0PkEggkHqK/j0/fBKACgAoA6fwho0uua9bafHuAkYAsv8I9favr+Gctlj8zpUUrpO79D53O8WsPhJSvq9Efc99ptlqdqbLUIVnhbGVbpxyP1r+2a1CnWpulVjeL6H8+wqShLng7M5g/D7wqST9iXmvmv9V8n/6Bo/cep/auN/5+sP8AhXvhX/nzWj/VfJ/+gaP3f8Ef9q4z/n4zf0vQ9K0ZSunW6w7hglRyfrXuYPL8Ng4OGGgoryPPrYirWfNVldmtXonMFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFAH//X/fygAoAKACgDlr7xL4Ze6fRLu8iMkmUZCw4J7H3rwMRmuXRqfVK1aPM9LNq+p6FPCYlx9tCDsutj5m+IXw6vNBu5NS09DLp8pLKyjO0HnBx+h71/NfFnB9bBVZYnCRcqUrvRfD/wD9YyTP6dWCo4h2ktPU8nZWU4YYNfkR9+mnqhtAy/YaZe6ncpa2cTSSSEABRknNehg8FiMXVVHDQcpPscWJxlHDwc6skj6n8F+GtL+Hmmf2pr8yRXk64O4gbQTnaM9TwDX9UcO5PhOHcJ9Yxs1Gct29LeS/rofimZ4+vmdbkoxbS2S/M9V03VdP1e3+1adOs8ecEqc4Poa/TMNi6GJpqrh5qUe6dz5arRnSlyVFZ+Zo12GAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAf/Q/fygAoAKAPGviR8SYvD8T6VpL7r5vvMMERgjP59K/IuLuL4ZdB4XCu9V/wDkv/BPtckySWLl7WorQX4nzPpVlq3iLVUhtS73MxPzc5Bx1/Cv5ty7C4vMsbCFK8pt3v8Ajc/VsbVw+DwslKyVj7j0fTXtdBsNL1Ei5kgt4o5GYZ3MqgE8+pr+5KNHloRpVNbJJ+eh/PVSd5ucdNTi9Z+FfhTUsyRx/YmySzIeufqcV8dj+D8oxcnUnSSk+qbX4Xse1hs6xtBKMJ6HJaZ8IvCc87eVqcl1sOCvy9Rz25r57D8C5HKfu+9bpd/oz0qnEGYKO9j1jRfCeg6Ap/s20SN2xuY8kle/OcfhX6JgMqwWBhyYWkor+ur1PmsRi69eXNVk2eSfGrQdWvvsurWbM1tbxtHIg/vE5DfgOPxr8s8RcsxWJw0MRR1jDdffqfX8MYyjRrOnV3ezPG/BfjbUfCWoq6MWt2OJIz0I/wAa/FOG+JMRlFdNO9N7r9Ufoeb5PTxtNyjpJbM+y9D1uw1/T49R0990cgBI7qT1B9xX9iZfmFDHUI4jDu8WfhWIw9ShUdOorNGvXpnKFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAf/R/fygAoA82+Jnii+8NaMDp6Ey3Idd+MhMDr9eeK+C4wzatluWSr0PibtftfqfQ5Lg4YrFxp1Ntz5I0/TdV8TamIYEaaeYkn8+STX8k4LBYrNMV7Kiuact3+rP3HEYnD4DD3eiR9ieCPBFl4SsgAN93J9+T+g9q/sHh7h7D5Rh+SnrN7vv/wAA/CcyzKrjanNPbojvK+yPDPNfivf3Nh4Sla2baZXCMf8AZIJ/pXxvFlapRybEVKTs0t/mj28npxqY6lCaumz5v+HGqXsHjDTwkhxIzAg9CCjV/NnAdepDO6MIy0lzX8/db/M/WuJKFP8As+c7aq1vvR9tV/Yp+FEcsaTRtFIoZW4IPek0noxny98Svhm+ms+saLGWtjy6Dqpz/Kv5n4y4NeHcsfl8fc+1Ht5ryP1fIM/vbDYl69Gcb8O/FGqeHtcht4MvBcyIkiYJyCcfmM8V8bwfneIwOYU6FN3hUlFNertf8T3s/wAuo4jDSrbSim/1PtSNvMjV8Y3AHHpmv7LPwkfQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFAH/9L9/KACgDM1bSLHWrN7DUY/Mhk6juPoa4sXhKOKpOhiIqUX0ZvSrTpTU6bs0Y/h3wXoHhcN/ZMG0vnLOd7cnOMnnFebluTYLL0/qtNJvr1+86cTja+I1qyudXXvnnhQB5R8ZP8AkUD/ANdl/wDQWr4bjH/kSYn0/VH0GR/8jCl6nzZ8P/8AkcdM/wB9v/QGr+a+Bv8AkfYf/t7/ANIkfrvEn/Itq/L80fdVf2afgQUARyxRzxtFKodHGCDyCKmUVJNNaMabWqOKsPh14W07Vm1m1tiLgncMtlAc54XoK+Xw/DmWUMR9apUUp77LT0PVq5liqlP2M5ux3NfVHkhQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFAH/9P9/KACgAoAKACgAoA8/wDiXo11rfheW2sxmSNhJj1ABGB+dfJcT4Wrispr0KKvJrRfNHsZVWhRxlOpN6Jnzl8NfDmp3Xiy1nERRLVmLlgQB8p/xr+d+B8qxizmlWlTajC92/OLX5s/UuIswoSwEqcZXcrW+9H2bX9an4qFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUANZlTliBn1oAUkDk0AVY7+ylmMEVxG8g/hDAn8qhSTdrmrpzUeZrQt1ZkFABQAUAFABQAUAFABQAUAf//U/fygAoAKACgAoAKADrQAgAHQYpWGLTEFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAx3SNS7sFVQSSeAAKAPzT/aS+NWuzeNG0LwrqbQWWm7RuhbAdygJJ9cEkV+X55mtVV/ZUJWUd7H9L8FcL4aWB+tY6kpOe1+ivb9DzRf2l/igNFl0d74P5owZiCZQDj+LP9K8v+38Z7PkuvXqfUf6h5R7f23K/S6t+R554S+IniHwx4utvFaXTyzxyq8m453qGywP1rycPjq1Guq97u+vmfVZhkuFxeClgnBJWsvJ20P0q0T9qb4WX1latf37213Ko3p5MhCt35AxX6vRzvB1IpuVmfyzjOC82oVJRVK67po+hNM1Ox1ixi1LTZRPbzjcjr0Ir6CM4zSlF3TPgqtKdKbp1FZroX6sxCgAoAKACgAoAKACgD//1f38oAKACgAoAKACgAoAKACgAoAKACgAoAKACgAoAKACgAoAKACgAoAKACgAoAKACgAoAKACgAoAKAPym+PHxX+ILeOtY0IahNZWlrcTQxpE7IGiVyqk4IzkCvyXOcxxSxEqKlZI/qvg/h/LZ5fSxUqanJ73119D5gmmknlaaZi7uckk5JP1r45tt3Z+vRjGKUYqyI6RYZFADgxU8HFFh3P2B/ZtuYLj4TaR5cwldFIcA5KNx8pr9uyZr6jTs76H8V8YRks6xDlG12e817x8MFABQAUAFABQAUAFAH//1v38oAKACgAoAKACgAoAKACgAoAKACgAoAKACgAoAKACgAoAKACgAoAKACgAoAKACgAoAKACgAoAz9U1O00bT7jVL+QRW9shd2PQAVMpKMXJ7I1pU5VJqnBXb0Pxq+M/jCw8b+PtR1rTYwls0jKhH8QDHDH69a/D81xUcTipVYbbfcf2rwtllXLssp4es/e3flfp8jzKxsp9SvYNPtRmW4bYufWvGjFyaij6ytVjSpyqz2Wp7L4y+COp+AW0u61y8iksb2VY5Xj3ZTJ5zkD1FerXwEqPK5vRnwWV8W0szVWGHg1OKbV+pW+MPhDwV4Um0tfB979r+0xlpRnOMBcH8cmpxtGlTcfZO5rwvmeYY2NX69Dls9PxPGa8s++PoX9n34ieLfC/jCz0jR2e5tL6QJJb9VII5Ptjr+FfU5JjK9PERpQ1i918j8x40yjBYjAVMVWSU4pWfz2/E/Xev2M/kEKACgAoAKACgAoAKAP/1/38oAKACgAoAKACgAoAKACgAoAKACgAoAKACgAoAKACgAoAKACgAoAKACgAoAKACgAoAKACgAoA434geFI/G/hDUfDEkrQi9RQGU4IKMHH6jmuXE0fbUpUn1PUy7GPB4qniUr8rufjr8Rvh7rvw61+bR9ai2/MxjcHKumeCCPbFfiGNwNTCVPZ1Pk+5/amS53h82w/t6D16rszg4ZpbeZLiBikkZyrDqDXmptO6PopRjOLhNXTPVv8Ahamq+JNa0pviA/2/SrFyXiUHDcHBPXJzj8q9D63Kc4+31SPinw7RwmHrf2WuWpLZnNfELVPDOseKbm/8I2ptNOfG1CCOR1OD0zWGJnTnUcqSsj18kw+MoYKNPHS5p9ziq5T6A/QH9jPw74fni1HxBKok1W3comf4EKqMj3O4iv0rhejTcJ1X8V7fKyP5y8S8ZXVelhU/3bin87v/ACR991+hH4AFABQAUAFABQAUAFAH/9D9/KACgAoAKACgAoAqXd9Z2CeZeTLCpOMsQKBpN7HlHjj4jx6WUtdCuEklxlnX5gMnGPrxW0IX1Z9NlmVvENzq6RMTTPjC0djtv4RNOo+9nbuPuMcU3TfQ7avD9Tn/AHcvdOj8HfEf/hIdTewuolhDj92d2eeOOlTKFkefmGVSwtNVE7o9arI+bCgAoAKACgAoAKACgAoAKACgAoAKACgAoAKACgAoAKACgAoAKAPz8/bQsb+e70OaK1Z4VWb94q5/558E1+dcUxk/ZSS01/Q/oXwxq04/WoSlZvlsv/Aj4JeKWP8A1ilfqMV+cH9D3XQZQAUAb3hfw7qHi3W4tA0kB7yYEomeSAccfnXfhcHVxMuSkj5nOOIMvypR+uVLOWy6v0R+nf7OPwa1/wCF8WoXmvSqZb5QqxocgAEHJ96/U8lyypg4ydR6s/mnjHiShm9SCw8bRj1e73PqOvqj8vCgCle6jY6bGJb+dIEY4Bc4BNZVKsKa5qkrIuMXLSKuSW93bXkYmtZVlQ9CpyKqM4yV4u4mmtzA1jxjoGiP5V5dL52QpjUgsMjPSvDxud4DBy5MTWjF9mzuoYHEVlzUoNr0OmVg6hlOQa9888dQAUAf/9H9/KACgAoAKACgAoA+Y/i3f3Q8QeQkx8tUHAPA4FdNNH33D9KLpzlJdTyKSR5W3Odx9a3PtUraJDKBlyxv7nTrhbq1cpInIIpNX3MatKFWDhUV0fQHw78b6xrF9Hp98zXIfcC5AG3C5BOMdcYrmnFJaH55m2XU8Mozpvc9wrE+UCgAoAKACgAoAKACgAoAKACgAoAKACgAoAKACgAoAKACgAoAzdT0fTNZg+zapbR3Mfo6g4+lRKEZL3lc2p1Z03enJr0PAPi78AvDHirwpcR+HdOjtdUhPmRMmRuIBG0jpg5z07V4GZZVSxNJqCSl0PveHeKMTl2KjKrNypvdb/mfBPgn9n7x74x1Y2K2ps4YmIlllBCrgZ7AmvzrDZHiq0+WUeVd2f0HmXGuWYWj7SnPnk9kj0zxT+yB420mFbjRZ4dSz1jjLbwfxAFeriOGq0daMrnyuB8ScJO6xdNr0t/mfFniCHxV8PfG8i3Qk07U7Bl2gHDL8oYfnnNb4bDzwdovSR/HfiHxRUzjO5V6d4whZRT3Wiu/m9T9k/2f/jHY/FzwfHeswTVbQ+XcxnGcgA7gB2O7Fff4auq0ObqdeVZgsZR5n8S0Z73XYe6FAHlHxe024vfDP2qA82bbiBnncQvavzjjjCVcRlFT2N7xs9PVX/A+lyGtTp42PtNn/kfNOgeM9b8OtOLSZv3qFCGOcHBAIB9M1/NOT8T4/LW1Sm3Fp6Pv3+R+vY/JcNi7Sas/IzrI6jrusxruM08z8Bj1J968nD08RmmOjBu85s66/scFhG0rJI+9rWEW9vHCDkIAK/vGKskj+cnuWKoQUAf/0v38oAKACgAoAKAMbxBLewaPdS2AzMsbEY6jg8imtzejGMqkYz2vqfF+rXd3eXsj3rFpM85613LyP2TD0qdOmo09hdGNiNQi/tL/AFGTn8jj9a5sT7T2T9luRivaeyfsdzoho+naxrr21g3lW6rk9sn2ry/rNWjh1KpqzyvrVWhhlOqryMDXNLGkX7WavvAAIP4V6OFr+2p89j08JiPb0vaWsdt8MvENvoessLnAjuVEZPpzx+tb1I3Wh4ud4SVakpx+yfVwOea5T8zFoAKACgAoAKACgAoAKACgAoAKACgAoAKACgAoAKAPl/8Aab8UeMvBei6T4i8L3Jt4IJJFuAOjFtnl59uDXl4+tUpU+emfI8QYvE4WjGth31sz5q0D9sjxrZsia3Z291EvXahVz+JY14tPN5L+JH7j46hxViY/xYqX4H0p4Q/as+HniJhBqbPpczHChwWB/wCBAYFexRzGjUdr2Z9ZheJcHVtGp7r8/wDM+j9M1jS9ZtxdaVdR3UTAHdGwYc9Oleqmmro+up1YVFzQd15GlTNRAAOgxQAtAHxV+158EY/GvhxvG2hW+dX00AyhcZkhAI6d2B249q8rHYb2sLx3R8bn2Xe3p+3pr3l+KPz1+AvxPv8A4VeP7PUg5FnM3lXMZ6FGBXJ/3Sc/hXz2DrulUV9mfAZbjXhcQqnTr6H7o6ZqVnrGn22qafIJba7jSWNx0ZHAZT+INfap3V0ft0JqUVKOzL1Ms+aPin8Qb9L+TQtJkMMUPyyMvBZs8jPpjFfztxxxXXpVnluDdkl7z6+n9dz9O4eyWFSKxVf5L9T56MwldyW3NnLfU1/PTvuz9Xi47R6EscslvidG2ds5x+FdFCVanJVaLaa6owrKlNezq7H1Z8JvF+q61C+nampk8oZWXrn2Nf1RwNnuLx1KWHxkXzR2lZ6r177n4nxBgKGHqqeHej6Htdfrx8aFAH//0/38oAKACgAoAKAEIB4NAHxX4tuLa81q5uLNdkLOwAA7qdrfqDXbD4UfreVX+qQ5v61OcicRypIw3BSDg96cleLR68leLibuqa79ruIbixj+yNEuPk4z09MV59DC8kXGo+a552HwXs4OFV81zDmmmuJDLO5dz1JOa9CMVFcsVZHoxjGC5YqyPXfhn4LttcLateN+6t5NuwdSygMPpjINZzlbQ+PzrMJU/wDZ4dVufTIGOB0Fcp+ei0AFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFAHnPxY8Ix+NfAWraGyeZK0LyQgDJMyKSn61z16aqU5Q7nl5jhVicNOj1advXofilqVlNp1/PY3ClJIHKMD2INfmsouMnF9D+eZJpuL6FMEqcg4NSSeg+Dfif4x8D3cdzomoSxpGykxFyY2wehXOK7aGLq0X7rPRwuPxGGkpUZteXT7j9CfhJ+1F4f8YCLSfFOzTdQIwHZsRuc46nGK+uwuYwq6S0Z+o5ZxHSrtU8R7svwZ9ZqyuoZCGB6EdK9k+5uOoGRyxRzRtDKodHGCCMg0AfiH+018Kz8MPiHNFYxlNNv1Fxb4GAoZmG3PqNtfGY+h7OpdbM/Fs3wX1XEOMfheq/yPun9i74nHxV4Jm8I6jNuvND8tY9zZZomBCgA9lCV72Arc9Ple6Pt+HsY6tB0Zbxt93T8j7XPTmvVPsj8oP2u/i5az+L18NeDnWBrAf6TPC2C0pLAr8voMHOa/Ps5y/AV6nv0YuXV2V/vPhs24kxlKosNhKrSjvZv7jnfgPp3i3UvD+qeItSjmnsmkiVJ5NzAsd+QCfwr8X4wyKVOjTxeGp2grp2Xprofqnhrm9Wq68MXNvmas27977/I/QH4PeE7WXRL261qxjuY7qVWh86MMAFXadu4HuO1ffcAZTD+zHWxNNPnd1dX2uup9DxJjW8Xy0pNW3sz3O10/T9OQrZW8dsvfYoQfoK/ZqdKnTjy04pLy0PhpTlJ3k7nIaz8S/BmhapaaJfanD9vvZkgjhVwZC8jBRx9TSlWpxaUnqzy6uPw9KapzmuZ6WO7ByAR3rY9E//U/b34k+Obf4d+GJfE11C1xFCwBRcZOQTxkj0rnrVVSg6jWiPMx+MWEoOu43SMTwF8aPA/xBiT+yrsQ3LZ/cSkK4wM/T9aijiaVVe4zmwWb4XF/wAOWvZnrIIIBHINdZ7gtABQAUAeda/8N9F1fMsI+yyEljt6MT1z/wDWrRTaPZw2Z16CUYy07Hynqlt/Zc8kd0REE67zjH1rrT6n6jSxEKlJVm7Jmba3drexmWzlWZAcEoc4/KhNPY0pVqdVc1OSa8j0Dw74D1jxFA09mFRVOMvkD9BUSmk7Hi4zN6OGn7Nq7PorwJ4Um8KadNa3EiySTSmT5c4GVUd/pXNKV3c+AzDGfWqvtLW0sdzUHlBQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFAH5X/ALVHwzbwn4ufxHYRhdP1UqwCg4R9uGB9yVJr4vNcPyT9qtmfi3EeAdDE+2ivdl+f9K58n18+fGGNqWrCxdY1Xcx5+laRjfU3hT5lctaZrCTkPC5ilX8DQ4uOop02j74/Z5/aMnsJ7fwb4znaS2c7IJ2IJQ44Vj1xke/WvpsBmN2qdV/M+9yTPpU5LD4p3j0fb18j9FI5EljWWNgyOAQR0INfVH6ymmrofQB8i/tjfD6PxZ8MpdegjBvdDZZFwPmZHdUI47AEmvNx1H2lJ90fKZ/hfbYZ1FvHX/M/PH9mLxzN4H+LOkszFYNRmjtJR2xMwTJ+ma+ey+ryVUujPgcmxLo4uD6PT7z9Yfjn8Tbf4Y/Du78Rxvm5nxFa4xzI4LDr2wpr6qvVVKm5s/UczxiwuHdTrsvU/FLwloGs/ErxvaaTCWuLvUphvY8nAyzE/QAmvi6UJVqiXc/GqFGeIqqnHVs/djwX4A0Lwd4MsvB1tbI1tBCscvHEjhQrOe+Sea+zlh6cqXsZq8bWsfuuDpLC04wpaWL+u+IvDXgHRfteqTJY2cPCj1JycAfnTSp0KaS0ihYvGU6MXWrysj89fit+1drviFpdK8G7tOsW4Mhx5rDPfrj8K+ZxOauXu0dPM/Jsx4krVrww3ux/E8R+Fbar4j+K3hqa5ma4kXU7WVi5J4WZWavMwalVxEeZ31ueBlUJVsfSvq7p/c7n7SAYGPSv0I/oE//V/Z/47aMdb+Fev2kab5Vg3xj/AGgR/jXJioc9GUe6PFzei62CqU472Pxqt7y80y6E9pK0UsRyGU4INfnEZOLvHc/n5Nppo+x/g/8AtU6toktvofjhvtdhlUE+CZEHQk8nIA9q+lwmaNWhW+8+9yziSpSap4rWPfqv8z9EtB1/SfEunRaro1wtzbTDIZT/ADHavq4yUldM/V6NenWgqlJ3TNmqNwoAKAPyj/bl8QGPxzZaFpz+VH9jSWYJxulMkgO78AK+dzOtKLUIs/OOJMbVVSOHUny2vb5s7v8AYm8CWWs+Fte1LVBJsnlgWMg4wU8zdjjvkZrqyu6puV9z2OEa9ahTqVKcrJu33f8ADn6J6fp9rpdnHY2abIohgD68k/ia9k+znNzk5SepepGYUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAcD8SfAenfEXwtceHdQABf54n/uSAEBvyJFYVqSqwcJHmZhgoYyg6M/8Ahj8ZPF3hTVvBut3Gh6xCYZ4DjB5yPUHoa/Oa1GVKbhI/n/EYepQqOlVVmjgtT0xL5QwO2RQcH1qIysTTqcpxk0NxZS4cFGHQ10ppo7k00dTpOubnWOdtsinhq55QtqjlqUuqP1F/ZX+M7a7ZjwJ4huAbm2RBaO3V0AIK59sDFfX5bi/aR9lLdH6Zw3mrqR+qVnqtvT/gH23X0B+imP4g0yPWdC1DSpVDLdwSR4PqykD9alq6sZ1IKcHB9Ufz++KdMufBHjzUdNUlZdJvpUU+8MhAP6V8HUTpVnboz8GrQdGvKC+y3+DPc/2j/jEvxEi8PaLp8mbTTbODzMdGm8sZP4ZIr0cfiPaKMYntZrmCxPs4xeiSv6n0V+wx8M1S2vfiPqEXz72t7bPUEKpZh9Q5Fd+W0bRdR9T6DhvB6SxMvRf5/ofbXxI+I2h/DbQJdX1eUeYVbyYv4pHUdAPrjP1r1a9eFGDnM+tzHMKWCourU+S7s/JD4l/FPxJ8S9Zlv9XnPkBv3UK5CIo4GASfx96+DxWLnXld7H4dj8wrYyp7Sq/RdEeY1wHlH2P+x54QbVfGdz4klTdBpcYHPQPKG2n8NtfSZRTvOVR9D7zhbDc+JlX/AJV+dz9Oa+wP2A//1v30u7aK8tpLWdA8cqlSpGQQfagTV1Y/ED4j+Frnwb4v1HQbkHdbSFc46+4r83xVL2VaUD+c8bhnhq86L6P8Dhq4zgPdPg58atf+GerRgStcaZIcSwOx24zkkZzg+9etg8bKg7PWJ9DlWbVMDU7we6/yP1v8N+I9K8V6RBrejTLPbXAyCDnHsfevu4TU0pR2P3OhXp16aq0ndM3as6AoA/Dn9qTXDrnxn1wqweO2l8lCDkbVJP8AMmvjMwles/I/Fs6q+0xs/LQ/Sz9kjRF0X4MaZlNr3btOSRgkSAEfhzX0mDjy0Yn6LkdL2eCj56/efTVd59IFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAc54o8U6P4R0qXVtZnWGKMZAJwWPTAHeonNQV5GkISm+WKPlEftb2ba4tsNLAsDJt8zzDnYTjdjb1xzjNeY8cua1tD11l8uW99T680XWLHX9Lt9X02US29yu5WU5HoR+B4r1ItSV0eNKLi7M8J+PPwQsvibpLajpyrDrVon7tgo/eqDnaT+JxXn4zCRrw8z5TOcnhjafPHSa2f6M/KfxD4c1jwvqU2la1bPbTwsVKupXocd6+Dq0p05cs1qfidajUo1HTqqzRy93Zw3keyUfQ9xWabT0M4ycXocNfWM1jLhgSvZq6oyTPQhJSR3PgLxte+GNesdSgkKTWksckb5xyjA4P5UQlKnNVIdAjKdGpGtT3i7n7o/DfxlaePPB2neI7RgftEeHGckOhKtn6kE1+hUqiqQU11P3nA4qOJw8K0ev59Tuq2PQPxQ/a78OjQPjHqTIAEvgLngfxSksf518fmULVr9z8bz2lyY2dutmfNmnWU+pX0NnApkklYKAOSc15cIuUlFHzsU5NRW5++3hDR9J+FHw3tdPnZIbfSbctI/C7mHOT7ngV99FKnC3RH7pRpwwmGUW9Io/Kn40/FLUPiR4pudRnmI0+3eRbZCflWPPBx0BIAzXwmNxTr1NNlsfiGZ4+eNxDnf3ei8v+CeLwXttcMUhcMR2BrzWmtzx3BrVmla20t5cR2sCl5JWCgDqSaEm3ZE2vsfsP+z/APD1fAHgCzinj232oIk85Iww3LkKf93JFfomDoexoqPXqfvGSYH6rhIxfxPV/Pp8j3Gu8+jP/9f9/KAPhH9r74YveW8Hj3SocmBRFdbR23HDH6lsV85muG5o+1juj804py9tLGQ6aP79PzPzrdhGrM3AUEn8K+OPy9K5kQ65aSzeTyvOAT0rRwdjd0mlc+2/2Wfi7N4a8QJ4P1ic/wBmaix2FjxHLt4Pr820CvoMrxXLL2MtnsfZcN5k6Fb6tUfuy/B/0j9Pgc819gfsRFM4SJ3Y4CgmgD+ejxZqEniTxve37533lxk59zXwdZ89Z+bPwLEVHVrSn3Z+7Xwn05NK+GnhexQY8vTbTPu3krk19vSjywUfI/b8FT9nhqcOyX5HoVancFABQAUARzSpBE88p2pGpZj6ADJoA+Wda/ao8LaV4gk0tLOaa2hco0q7cHacEj5uh7V5ssbCL5bHqwwFSUVK5674L+LXg7xwoXS7oR3BBYwyEBwAcfSuunWhNe6ziqUJ03aSPTa3OcKACgAoAKACgAoAKACgAoAKAM3V9WsdD06fVdSlENvbruZj0ApNpK7KSbdkflj8X/irqnxB1yXZMyadEdsUWcDAOcnHfmvmcRXdR+R9ZhsOqUfM8XBxzXEdx+gv7KHjKTUNJu/Ct05d7UGWIHsm75h+bV7+CqXjyvofN5hTtNTXU+w69U8c8h+J3wa8J/EyxZNRgWC/VW8q4QYYFh3xwRkDrXHiMNTrRtNHhZjlNDGw99Wl3PzV+JXwA8a/D65kke3N5YBgEniBKnIz6DGO9fHYnL6tJ3WqPyDMMmxODd5K8e6PBLuzWVGhuE/PtXk6o8GLtqjz6/sZLGbYfunlTXVGVz0Yz5kfol+xF8V4Ybq7+H2s3Gzz1822LnguGA2D3O4n8K+myuqrOk/kfe8N4tQnLDSej1R+mVfRn6YflH+3jpyQeNtJ1IDDXcDKT6+WEx/Ovmc1WsX6/ofmXE8LVacu9/wseNfsr+E18V/F7So5o/Ngsc3MqnoURlXn8WFcWX0+atfseJkuH9tjI36a/cfdn7YHxBfR9BtvBNjKUmv1WacD/nkG+Xnr95K9TNcRyU+Rbs+m4pxrp0o4aO8t/Q/KXWtTMztaxH5RncfU18lCNtT8zpQtqN0G0le4+0jhFGPrRUlpYK0lax+gX7L/AMFJfEGpx+NvEEH/ABLrNyIkfI8yQLwfoNwPXqK9/LMHd+2mvQ+v4dyl1prFVl7q283/AJH6XABQFUYA7V9afr4tAH//0P38oAztW0uy1rTrjStRjEtvcoUdT3B/wqWk1ZmdSnGpBwnsz8jPjv8ABXU/h1rVxJDE0mjXjyGCQc7UJ4VvcAj618JjsHKjPmj8J+FZtlk8DW0XuPZ/ofH1/ZSWVwUP3TyDXJGSaOCElJHYeF9dlguYXD7Li3YMjfTmsneLU0ctSDi+eJ+4Xwa8Yr43+HulawX3zpEkMx9ZY1AY/ia/Q8NV9rSjM/esrxaxWFhV6219bam/8RNUGieCNZ1UtsFtAz59ORXRJ2TZ6FefJSlPsfgt4Qs5NW8caXar8zT3cQP03DP6V8FRXNVj6n4NQjzVIrzR/Qjp9qlhYW1jGMLbxpGB7KMV9+f0AlZWLlAwoAKACgDl/Gs0lv4U1WWI4YW7j8xg1E/hZpD4kfjReMWu5if77fzr497n26JLHULzTp1ubOUxSIcgjsaak1qhOKkrM+1PhF+0rL5kWh+N33Kxwlxg7hxxu656frXs0MZ9moeFiMD9qmfbljf2mpWsd7YyrNDKAyspyCCM17Kd1c8Nprct0CCgAoAKACgAoAKACgAoA/P79pf4s/2penwXosv+jWzAzOv8T7emfQbsH3FeJjK/2EfQYGhZe0kfHFeMe4FAHv8A+zhrsukfEmxt0fYt/wDuHz/dJDY/Na9DBytU9TzcdHmpeh+pNfSHyoUARyxRzIYpVDq3BBoA8A8a/s2fDjxeXuEtW065I4a3IVSc9WBBzXnVsFRq7o+WxmQYPEXly8r7o+TPG/7FnicI6+HLmPUEbJUFhGy/UtgGvGnlLTvTkfHVuF8RTfNRkmu3U+GdT0rxP8MPFjWV8j2GqabIDweQcZBBGRyCDXlSU6NTXdHzc4VKFXllpJH7X/Ab4r2nxZ8DwayWUahbkxXUa8YcAHIB7YYV9rQrKrBSR+y5Zjo4ugp9Voz4v/b7ktjq3hqNT++VLgt9D5WK8bNbWh8/0PkOKH71L5/oN/YI0ZW1zXdfIH7q2MGe/wA7o3/stPKo6SkTwxTTqVKnZW+//hjxT9qHx0/iD4kaoIZfMSBvIiPpGpJ/mTXkY+XtMQ/LQ+Yzir9Yx85PaOi+R806Zp8+p3SwRKWyRux71wN2PHnLlR9/fAn9mO/8StBrXiuJ7LSUGVTIEkvPYckZ55xXs4LLnJ+0rbdj6fKMhqYpqviVaH4v/gH6bafp9lpVnFp+nQrBbQLtRF4Cj0FfXJJI/YIxjBKMdi5TKCgD/9H9/KACgDnvE/hfRvF2kT6Lrdus9vOrKdwBK5GMgnoR61nOEZxcZLQ5cThqeIpulVV0z8tfjl+zJr3hKWe+0SJ9Q0k/MsiJlo8jowGeh4zXx+Ky+dKXPS1R+PZlktbBSdSmuaH5ev8AmfETx3Gn3Wx1McsZ78V5m54d1JH6mfsOeMBqekazoE0nzQmCSNM9C3mbyPyFfTZVL3JU+x+hcK1Go1aD2TTXzv8A5H0R+0lqaaZ8GPEjv/y2g8sc46nP9K9fEy5aUmfVZvNxwVRrt+p+RPwB0z+2vi54fsTkl52fj/pmjP8A0r5LAq9eJ+V5TDnxtOP9bM/eqvtz9vCgAoAKACgDD8S263Xh7UoGG7dby4HuFJH61MtmXHSSPxi1OF4NQuIpBtdXbcPQ55FfHyVm0fbxd0mUKkoUEqcjgigD6L+Dfxz1TwLfR6drEjXWkylVZWYnyx0yuc4wD0716OHxLh7stjzMThI1PejufpFoHiDSvEumxarpE6zwSjIKkHHqD+NfQxkpK6PmZRcXZm1VEBQAUAFABQAUAFAHiXxy+I8HgLwpMkD41C/R44QDgrkYLfUbgRXJiKqpx8ztw1F1Jrsflbe3c99dS3dyxeSVizEnPJr5hu+rPrkklZGDqus2Wjw+bdNyeijqaxnNQV2b06cpuyLlndxX1rHdQnKSKCPxq4u6ujOUeVtM7LwXqbaT4lsr9X8sxODuHat6LtNM5q8b05I/Z1DlFPqBX1x8UOoAKACgAoA+Cf20vg+mu6CvxG0eEm9sCq3QRfvRbT87Ec/LhRzXi5hh+eHPHdHwnEWA54fWqa1W/ofIn7K/xUn+HfxEtbO8lK6XqpMM6k8AlSEOP97bXk5fX9nU5Xsz5bJcb9WxKUn7stGd1+3JqbXXxOt7HOYre0idD2PmqCcflXRmj96KPQ4klfERj2R1f7J3i/T/AIc/DTxl421YgRW0scUaltvmuy52j34P5Vtl81ToynI3yPEQw2HrV5+R8QyDUvGniYLbo091qEoVVGWOWNeC71ami1Z8V71Se2rZ+tHwO/ZX8NeB9OtNY8TxjUNWkSORo5EGyJ8ZK4Od2Ccc46V9Vh8DCnaUtWfp2XZBSo2q1/el26I+v440iRYolCIgwFAwAB2Ar1j7NKw+gYUAFAH/0v38oAKACgCKaGG4jMU6LIjdVYAg/gaBSSasz8+P2q/2b9F/4R6bx34KsxbXNnhrmFNxDoWA3AZPI3ZPTgV4eNwcZRc4LU/Ps6yenGDxGHja26Pjr9nT4oSfC74iWd9ckjT7t0guR2CM20t/wEEmvGwVf2VTXZnymVY14XERk9no/wCvI/RX9r7XoD8EEurRw0OqXEKIRyCrxO4/lX0uNf7iTP0PPqieAbWzsfB/7HllFd/HDRXlHES3BH1+zyV8/lsb1b9j4bIYt46D7X/Jn7WV9gfsYUAFABQAUAQ3EK3FvJA3SRWU/QjFAH49fE7Szo/jrWrHbtWO7nC+6iQ4NfKV42qSPs8PLmpRZ494h146HHC4i8wSNg+3WvOq1ORJnpUaXtL6lzSdcstYj3WzYcdVPUVcKkZ7EVKUoPU2K1MT2v4SfF7V/h5qih3afTpOJIs56kHIz0Nd2HxDpuz2ODE4ZVVdbn6geHPEel+KdKh1jSJRLBMMgjqPY19JGSkro+VlFxdmbtUQFABQAUAFAFLUb+30uxn1G7bbDboXY+wpN2VxpXdkfk78YfiDc+PvFlze7ybOF3SBT2jzhePXAGa+XxFX2kr9D6/DUfZwS6nj9zcRWkD3M7bUQZJrjbSV2d0U27I8E1fVJ9Yu3uZT8vRV7AV4lSfO7s+hp01BWR6p4KmMuiIp52MRXp4d3geRilaodxZEC6iyMjcM11J21OM/ZzwlqP8Aa3hvT9SJz9oiDV9ktj4V7nRUyQoAKACgDO1bS7HW9NuNJ1KIT210hR0boQfWk0noyJxUouMloz8CviX4Svvhp8QL/QWJWTT5sxuO4B4Ir4XEU3RqtI/CMXh3h68qT6Mv/Fjx6PiDeaNqbkm4tdOtraYnqZIY1Vj+JBq8VWVTlfkXjMS8RNTe9kvuRycni3UP+ETTwhAxjs/PFxIB/HIoZVJ+gYisnWfslTRzutL2Xsul7n29+xT8HY9Vv5PiVrcO6Gxd4rVWHBkKDJ9MAPx7ivby3D6e1l8j7Ph3Ac83iprRaL17/ifqLX0R+mBQAUAFABQB/9P9/KACgAoAKAK93aw3trNZ3K7op0ZHHqrDBH5UClFNNM/Dv9o/4WT/AAw+IV5BBGV06/kkntW7BGbdtH+6GAr4vHUPZ1LrZn4rm2CeFxDil7r1X+XyOn134vL4r/ZytPBGoyBtR0jUY9mc5aExzHOf9ksBXRLEc+F5XujWrjvaZfHDyesZfetTo/2IbEXPxXF11NrDI303Rsv9arK/4kju4bV8W35H7BV9UfrAUAFABQAUAFAH5o/tR6EmmePmvYE2xXaB8+rnlv1NfP42Np37n02XyvTcex8b+NLM3OjmVRloGDfhg14NeN4H0WGlaZ5Ba3M9nMtzbNsde4ry02ndHtSipKzPZfDniOHWIRHKQlygG4evuK9elVU15nh1qLg9NjqK3OU+gPgf8YLvwBrCWF+5k0q6bEinJ2kjAYfpnjpXoYbEOm7PY87FYZVI8y3P0603UbPVrGDUbCQS29wiujDuGGRX0aaauj5Vpp2ZepiCgAoAKAPjT9qD4nLYWY8EaVJ++nAa4I7Lu4X6grmvKxlay5EezgaF37SWyPgMnJyeprwD6M8y8dascppMJ6fM/wCXT9a8/Ez+wj1cJT+2zzevPPTPXfAR/wCJQ/8AvmvVw3wM8XF/Gju4TiVSPWus4T9h/hfMJ/h/ocy4w9uDx9TX19N+4j4mqrTaO9rUxCgAoAKACgD8uv28fBv2XXtI8ZwR7Y7qEWzkd5EZ3JP4MK+czSntM/NOJqFqkKy6q33f8OfnpXzR8EdH4S8OX/izxDZaBpkZluLuQIqj1relTc5qKN6NKVWapw3eh+//AIF8Jaf4G8Laf4Z0xcQ2UYXPdj3Jr76EFCKij93w1CNClGlDZHXVZ1BQAUAFABQB/9T9/KACgAoAKACgD5V/a2+GqeOfhrcataw79Q0JZLhCBk+UF3SD8kFefjaPtKT8j5fPcH7fDOa3jd/Lqfi62+MtGcjBwR0r4jXY/ID7s/YMt9/j7Wbgn7loMD/gRr6HKl70mfccML/aJ+n6n6wV9Mfp4UAFABQAUAFAHx7+1v4fa80LS9djj4s2lSRgP+emzaD+Rry8dC8FI9jL52m49z887mBLm3kt36SDBr59q6sfTJ2dz53uIHtriS2k+9GcGvAas7H0kXdXRc02DUTJ9s05WZ4SCdvX9PpVwUr3iRUcEuWR7Roeq/2nahpUMc6cOp6g+tevTnzI8OrT5H5G2Dg5HUVqYH1f8Gv2gl8E6JcaJ4gV7mKPabfkkjrkd+Oleth8UoR5Znj4rBupLmgeiah+15aRf8g7Rln/AN+Up/7Ka6Hjo9Ecqy6fVmLcftf3bkGDRkj9R5pP/stZ/X/7pr/Zv94qR/td6oCd+lIef756f980fX3/ACh/Zy/mNyD9r6N1cTaMqYX5T5pyW9xtq1jl1Rm8ufRnxh4i16/8S6vcaxqMhkmuG3Ekk14s5ucnJnuwgoRUYnPyyeXG8gG4qCQPXFRsarc8ovtCm8q51zWGKlydkfcnt+gry5U3ZzmevGqrqEDi5YZrfaJ0KFhkAjHFclmtzvTT2PXfA8ezRs/32Nerhl7h4uKd6h20X+sX611nEfrz8HP+SYeHP+vVf5mvraPwI+Lr/wARnpdbHOFABQAUAFAHyb+2V4cGtfB671IKWfSZI5VAH/PSVEP6GvNx0Oaiz5biCkpYKU+qt+aPxhr4k/ID7y/Yb8Af2x4wvfGl5Fut9LiKREjpOzIQc/7oavosrpXbqM+24bw3PWlWa0j+Z+r1fTH6kFABQAUAFABQB//V/fygAoAKACgAoAqX1lb6jZT6fdoHguY2jdT0KuMEflQTJJppn4FfGTwfL4G+I+teHpV2iGbeo7bZVEi/owr4XF0/Z1ZI/Csfh/YYmdLs/wA9T64/YFjQ+JvELkfMLZMH/gVetlX2j6rhj+NU9P1P1Ir6Q/TQoAKACgAoAKAPLvjLoH/CRfDvV7TvBC9wB6mJGYD8awrR5qbR00JctSLPyNmjaKZ42GCpIIr5LZn2Z4t43svsuq+eows65/HJrysRG07ntYWV4W7G34X0UtFDqtlM0TZAdGHBx1FbUae0kzHEVdXBo9FWNFO9VAY9cDFd55l2SUCCgAoAKACgAoAKAA80AQzW8FwFEyBwpyAemaTSe5SbWxymueEk1e4N0JvLbG0DHHFc1Shzu9zrpYhwVrG9o+njS7CKyzuKDkjue9bwjyxSOepPnk5GvF/rF+tWZH6+/CKJofhr4eibqtqv8zX11L4EfFVv4jPR62MAoAKACgAoA8e+PumnVvhF4ksh/FArf9+5Ff8ApWNZXpyPKzOClhKifY/BN1zKVHqRX56fhh+1P7IfhOLw58HtPviALjVWkml/4BK6r/46BX3ODp8lFI/Ysio+zwcXbV6/i7fgfUld59MFABQAUAFABQB//9b9/KACgAoAKACgAoA/Kb9u3wtHYeMtN8SwL/yEof3p/wBtBsH/AI6tfM5pDWMz8w4lo8taNVdV+JqfsCEf8JH4iGefsyf+hCryr7RXDH8ap6fqfqLX0Z+mhQAUAFABQAUAVry1hvrSayuBuinRkYeqsMGgZ+OXxA0h9D8X6np7rtCTNtB9DyP0NfJVo8s2j7SjPnpqR59eaZZX7xvdx7/KOVzXLKKludcZyjflLqRpGoRBtUcACrM/MdQAUAFABQAUAFABQAUAFABQAUAFAE1vjz0DcDNAH6w/BrxVY694Zh02y+caWvktIOjMMHj86+tpTjJe70Pi61OUJe/1PYa3OcKACgAoAKAOE+J8Xn/D7xBHnGbObn6LUT+FnHi/4FT0f5H8/Vlam41KGzzy8ipke5xX59GN5KJ+DRi5NJdT+hLwFpkOj+C9E063GEitIfzZQT+pr9BgrRSP3vDQ5KMIdkjrqs6goAKACgAoAKAP/9f9/KACgAoATIoAMj1oAMj1oA/P79veEf8ACJaHdAAn7WqZ9ikp6/hXiZov3SfmfD8TRX1eEv736M8E/Yf1k6f8Um0zdgalA6Yz/wA80Z/6Vw5W/fkjweG52xTj3R+vmR619SfqwZHrQAZHrQAZHrQAZHrQAZHrQAhZVBJOAOaAPzS/aj0Q2Hj59UVcRagiMhA4OxFU8/UV89jY2nfufTYCV6dux8zV5h6wUAFABQAUAFABQAUAFABQAUAFABQAUAFAH6LfsmQOng29uW+69yy/iEQ19Dgf4b9T5nMP4i9D6xyPWvTPJDI9aADI9aADI9aADI9aAPOvi1cLB8NvEMjNgfZJBn/eGP61nUdoM4cbLlw9R+T/ACPwm8AwC48d6DaSfMJtQtoz/wAClUV8Jh9asPVH4jhI82IprvJfmj+hDS0WHTLSJcYSGNfyUCvvz96Sski9ketBQZHrQAuc0AFABQAUAf/Q/fygAoA8v+LXxR0X4UeFJ/EerZkkwywRDG6STHA5xxnGawrVY0oOcjzMdjYYSk6s/kflV4t/a++Leu6hLNpOonS7VmykUaIQABjqyk+/WvmKmZVW/d0PzCtn2MqSvGXKvI5P/hp742f9DJL/AN8R/wDxNY/2hW7nN/bOO/5+P8A/4ad+Nv8A0Mkv/fEX/wATR/aFbuH9s47/AJ+P8Di/Gnxe+IPxBsYtN8Waq9/bwuJEVlQYYAjPygdmNYVcVUqR5Zs5MRj8RiIqFad0c14T8X+IPBGrx674au2s72IMFkUAkbgVPDAjkE1jSqypy5oHLRr1KM1UpOzR6z/w078bf+hkl/74i/8Aia7f7Qrdz1f7Zx3/AD8f4B/w078bf+hkl/74i/8AiaP7Qrdw/tnHf8/H+Af8NO/G3/oZJf8AviL/AOJo/tCt3D+2cd/z8f4B/wANO/G3/oZJf++Iv/iaP7Qrdw/tnHf8/H+Af8NPfGwdfEko+qRf/E0fX6/cf9sY7/n4/wAP8ijc/tZfFqzBa58WtHj1EX/xNWsbiHsbQzPMpP3Zv7v+Acte/tt+Ol/0O48YyOJv3ZCpHzu4xnZWyxOJe7Paw6zurJct7eiPWNQ8Y+IvFlnYy6zqD3kMcQMSuF+Xf8x5ABOSe5qJ1JT+Jn7lhcLKm+dy3S07OyMesT1AoAKACgAoAKACgAoAKACgAoAKACgAoAKAItS+NXjb4U6QRpPiBtLsbmckIFQguQoJ+ZSegHet1XqwjaDPzPiOhjoXxNCb10suiMy2/az+LN4N1r4taTPoIv8A4ms3jcQt2flc8zzKDtKbXy/4Bf8A+GnvjWeR4klP0SI/+y1H1+v3Mf7Yx3/Px/h/kH/DTvxt/wChkl/74i/+Jo/tCt3F/bOO/wCfj/AP+Gnfjb/0Mkv/AHxF/wDE0f2hW7h/bOO/5+P8A/4ad+Nv/QyS/wDfEX/xNH9oVu4f2zjv+fj/AAMzWP2hvi5r+mz6Rquvyz2tyu2RCkYDDOey5qZY6s1ZszqZrjKkXCdRtP0PH7C/utMv7fU7NzHc2siyxuOquh3A/gRXnxk4yUl0PJhOUJKUd0e4p+038ao0VF8SShVGB8kfQf8AAa9L+0K3c9r+2cd/z8f4Dv8Ahp342/8AQyS/98Rf/E0f2hW7h/bOO/5+P8A/4ad+Nv8A0Mkv/fEX/wATR/aFbuH9s47/AJ+P8DpfDH7Xfxg0S+jn1HUzqUCnLRSogDD6qoNawzGqn72pvSz3Gwldy5vU/UP4K/GfQfjF4eOpaeDBe2523EDYypwDkYJ4Oa+loV41o80T9Ly7MYYynzx0a3R7TXUeyFAH/9H9/KACgD8w/wBvq/vzqvhywYkWqJcMvoWPlZr53NW7RXqfnPFDd6S9f0Pzrr5k/PQoAKACgAoAKAAkAFmOAO9Azidc+IPhjQQVuLoSyj+CMFv1FdEaM5HvYPJsZin+6hp32PJtW+N905ZNGs1RecNJz+PauuOGXU+7wvBr3xM/kjzfVPiD4r1fK3F6Y0P8MeVH9a6I04rZH2OG4dwFDVQu/M5CW4nnO6eRnJ9TWp9FTo06ekI2IMmPDr1Xn8qDoP1h+HWqx614H0fUY23b4Ap+qEp/SuV7nbHY7SpKAn1oAiE9uzbFlVm9ARmmBLSAKACgAoAKACgAoAKAImnt0ba8qq3oWGaYEtIAoAKAPjb9q/WV26P4fRuT+/Yex3AfqtbwRhUfQ+OYp5oGDQSNGR3BrU4KlGnUVqkbnU6b478U6WQLa+dlH8LnIrN04vdHgYjIMBW3p29D03SfjhdoQmsWauvdo+D/AFrmlho9D4vFcG9cPP5M9Z0P4h+F9dCpBdCKVsfJJ8vJ9ziuWVCSPhMZkuMwv8SGnlqduCGAZTkHoR3rmPAatuLQIKACgAoAKACgD7G/Yo1bUrP4t22nWpP2W+jmWcdsJE7L/wCPAV72Vt88kfX8OSaxbS6o/Yqvqj9ZCgD/0v38oAKAPnH9pH4Lj4veEVSwwuraYJHtiR97cAWQ/XaBXDisOq0OXqfPZvl31yj7vxR2/wAj8b/E3gDxb4S1GXTNc0ye2mibGHjYZ9xx0NfH1MPUg7SR+Q1sPVovlqxaOTkt7iH/AFsbJ9Riudpo5iGpAKACgCjqeo2uk2E2o3r7IYFLE/QE4H1qoxbdkdFCjOtUjSpq7Z8o+L/idrOvzPb2DtZ2YOFCEhmHqSMV61OkorzP3DKuGcPhoqpXXNP8EeYMSzb3JZj3PJroPu0klZKyEoGFABQBbsHtotQtZLxd9usqeYvqm75v0oA/VH4d6Dpvh7wvaWujzNNZSqJowxzt35ZgDz/ETXK3qd0VZaHcVJRynjua6t/Betz2JZZ47OdkKnDAiNiCCKpbky2Py5i8YeLbW5NzDq90sqsTzM/XP1rpsjjuz9LfhT4mvPF3gbTtbv8A/j4cbZCO7DnP61zyVmdcXdHolQWFAHDeKfFDacws7Ahpzyx67eelcdaty6R3PQoUObWRb8Ja1davby/ax88RHzDvnP8AhV0KjmnczxFKMGuU66uk4woA8/8Ail4luvCfgTVNaseLmOMpGfRmU4P4GqirsmTsrn5m3njPxZezveT6vdGSQ7iRM4x9Bmuqxx3Z+ongKe8ufBukT35Zp3gBYtyScnk5rle52R2OuqSgoA+Df2kfC+qw6r/wmGq3KhLhvIt7cHJEandk/ix7dq6IM5qie58u1oYBQAUAKpKsGQkMOhHBoE0mrNXR6V4S+JuteHZVhu3a8s+hVySw57E5rCdKMkfEZpw1h8SnOiuWf4H1lpepWmr2MWo2DiSGUZBH9a8mUXF2Z+F16E6FR0qis0X6g5woAUAk4UZNAFtNPvpBuS3dh6hTVcr7Aa2k+E/EeuXiWGladPczv0RI2Y/pWsaM5OyRrCnOo7QTb8j9Zv2VfgBd/DHTpvE3iZAusXw2rERzFHgdT1DE5yPSvrcHhfYxu92fqmSZXLDRdWr8T/BH2RXqH14UAf/T/fygAoAKAMfUPD+h6sD/AGlYQXGe7xqx/MjNJpPcynThP44pnmfjH4D/AA08ZabNY3ujQxSyKQssYKMh7EbSBWM6NOatJHm4jLcNXjyygvlofjR8ZPhtd/CzxzfeFrht8cbF4W/vRMTsJ98YzXxmKoeyqOJ+QY7CvC15UX0/LoeVVwnmhQBwfxJ0q81fwnd29jkyR/vNo/iCgkiumhJKep9HkeJp4fHU6lTa58YsjRsUcEMvBB9RXrn9IRkpJSi9GNoGFAHU+GPBniPxhc/ZdCtGn9Wx8o+prlrYinRV6jsdNHD1KrtBHvumfsu69PEJNU1OGBz/AAru4/8AHa8KedU07Rie7DJ6j+KRk+JP2afFWmWslzpVzFqCoCSi7t+AO3AFa0s4pSdpKxlVymrFXi7n0Z8APElzqPhI+HNWVo9S0VmjkR+G2liwP4AivYbUlzR2PLimvde57xUlkc0MVxDJbzqHjlUoynoVYYI/KgD5p1v9mDwpqWoS3theSWaSsWMY5Az1xnPetVMxdNHuXg3wtZeDPD9v4e09y8Nv/E3Un1/SobuaJWR1FSUcn4k8SRaREYLc7rl+g/u+9c1WqoaLc7KFBzd3seTWlreaxeiFMySynknt6k15kU5ysexKShG57lpGmQaTZpaxDkAbj6t3NexCChHlR4FSbnLmZp1oZBQBzXi/wxZ+MfD114dv2KQ3QwWXqDggEfnVJ2JavoeDaR+y94SsL1Li/vJbyJDnyzwD7HGK052Zqmj6dgt1ihSG3j2xxjCgDgCsW11ZvYcQQcMMGgQnTrUuSjq2VGLlokfHvxx8J+O/iJ4lit9FsmOm6cgVWY8NJlskY9iK5v7Qw8N5G7wOIntE+dtb+Enj3QYGub7TXMSAksnOMVtTx9Co7Rkc9TA14K8onnDKyMUcFWBwQRgivQPOG0wCgBcEkKoyT2FAm0tWfZXwx0i90fwnDBfZWSVzIFP8IIAx+leRXknPQ/nPP8VTxOOnUpbHoNcx8wFAH6Pfsqfs16B4h0SPx943txdxzOwt7diwUrtGGOCO5P5V9PgcHHlVSaPv8lyenVh9Yrq6ey/U++7X4beBLOFbe20S1SNeAPLB/U173JHsffLC0FooL7kbNh4X8O6Wd2n6bbwN6rGoP54pqKWyNY0acfhil8jeqjYKACgD/9T9/KACgAoAKAILm6t7OB7m6kWKKMbmZjgACgTaSuz8UP2rfHmleO/ipeXWjOJLWyUWwcdGaL5SQe4OOK+OzCrGdSy6H4znWKjXxUnDZaHzLXkHzwUAFAHlvi74WaR4id7yzP2S8bkkfcb6jFdlOu46M+yyviLE4O0H70ezPnzXPh74o0Jj9otTLGP44yGGPoM13xqRlsz9XwXEWCxKtzcr7MufDfwFe+OfFEOkFGjt4yr3DHjagPI57kA4rkxmJVCk5deh9vgaKxNRKDuvI/SXQNA0rw1pkWk6RCIIIhjjqx7k/U1+c1asqknOb1P0enSjTioQWhsViah0oA5S/wBHg0/WV8W6fHtugvl3AH/LWLjr7/Kte5l+MdKXJP4WeNjsIqi54r3j1218P3F5ax3lvKrpKNykEcj619I8ZTTsfOrDSauZ13p93ZHE8ZA9RXXCrCfws5505R3RS47VqZCMwUFmOAO5oGcH4g8YRWoa104h5uQWxwtcVWulpHc76OH5vekebQW19q92ViBlmkOSf/r1wJSk9D1HKMI6ns+haFb6Lb4X5pn+81etTpqC8zw6tV1H5G/WxzhQAqqzHaoyfQUbasPQ2bTQb66wxXy09TXJUxMIeZ1QoSkdTaeG7KDDTZlb36V5lTGTltod0MNFbm5HBDENsaBR7VwOcnuzqUUtjkfFsdvFFFMFCuSenfpXdQxHs4SlLZHNUo+0lGMd2eeu5c5NfNYjEzrSvJn0tDDwpK0UNzXGdQnYjsetAHyx8d/hJZ3mnS+L/D0Hl3cHzXEa9HTGMgevA719RlmOakqVR6dD5nMsDFxdWmtep8TorSkLEpcnsOTX2h8VKUYq8nY7XQ/h/wCJtedfs1qY4jjLvhQAe+CaylUjHdnzGM4hwWGT97mfZHv/AIR+Fek6AyXuoH7Zdj1+4p9hj+tcFSu5aRPyrNOI8RjL04+7Dser1xnxQUAKDg5oA/Z39kf4i6J4m+GOn+G4p1S/0YGAxHhioAbd7/eP5V9vgqqnSS6o/XchxcKmFjSv70T6yr0T6sKACgAoAKAP/9X9p/i38WNA+Enhp9e1hhJK7BIYAcNI5BP5cHmuetWjSjzSPKx+Ohg6XtJ/Jdz8yPEP7aXxT1O+eXSnisbbOVjChsf8CwM183PM6jfuqx+bVeIMZOV4vlXY5/8A4a/+Mv8A0EI/++BUf2lWMv7dxv8AP+CD/hr/AOMv/QQj/wC+BR/aVYP7dxv8/wCCOK8W/tD/ABU8ZWr2Oq6xIlvIMMkTGMEeh2kZrCpjqs1a5xV80xdZcs56fceIsxZizHJPJJrzjxxKQBQAUAFACH5gVbkHselMd7HdeCNGsLKG51G3t44pbhgpZFCkhc9SB7187mNVykoX2P6o8OcNOGXzxFRt8zsvlf8AzO7rxD9jCgChJqmnRStDLcIjr1BOK6FQqSXMoux4VbOsvo1XQq14xkujaJVvLKUfLMjA+4NS6U1ujsp5hg6i9ytF/M9b8KXNo2jxW8BVBD8gUEfX+tepSk5RuzgrOnGdotHSvHHMhSQB1PrzXQm07oysmcZrehW9rbTahC2xYhuZfavVpYx/DM4KmFT+E+Y9Z8V32qkxxEwwHsDyfxqqlaUtEdNKhGHqePeK/Huj+F4mRnFxdkHESnnI9euOaKVCVT0JrYmFNeZ6j+z5r134m8N32r3ygSmfauB0XnjP4V6ypRhojxnVlU1ke/AE9Bn8KYrF230y+uiBFC2D3I4rGVWnHdmsaU5bI6K08KyNhrqTHsK4J41L4EdcML/MyHXdd8O+CrXzJwsk5+7HwXPb6158qtSpuzvhRitkS+D/ABtY+L0lFvGYZoMbkJ7HOCPyrCUWjVqx2201mIaxVVLMQAPWmD01Z5T4k1IX97sQ5jiyBg/rWGOvCMafzOvA2m5VF6HPV4p7JHNKkETTSttRBkmrjFyfKjnr16dCnKtVdordnLXXi+yiJW2jaU/kK9Wnl838TsflGO8RMuo3WHi5v7kcvqfiO91G3ktcLHFKNrDGcj05r0qeBpwd3qfmeO8RMwrpxoxUF9551png3wzpGPsdhHuH8TKGP5kV7Eqs31PzOvmWKrv95Ub+Z0wAVQijao6AcAfhWR5bb3CkIKACgAoA6Dw54o17wnqCan4fvZbK4T+KJyme3OCM1rTqzg7xZtTqzpy56bsz3u0/a2+MtpbrbjU1kCcZZck/jXpLMax7qzzGr7f4Is/8Nf8Axl/6CEf/AHwKr+0qxX9u43+f8EKP2v8A4yjrqCf98Cj+0qwv7dx38/4I9/8Ag1+2lqN7rEGhfEaOMxXTpGt0p2CMscfMMYxyMnNd+HzHmajUR7mA4hnzqnidn1P0mtriG7t4rq3YPFModGHIKsMgivoD9HTTV0f/1vsP9vz7f/a/hwAt9k8h8j+HfuP64zXzuaXtHsfmvE/P7Sn2t+J+dFfMnwIUAFABQAUAFABQAUAFABQB614eQJpEOO+T/KvksY71ZH9p8H01DJqFuqv+CNquA+5Kd/eJY2ktzJ/AOB6k8Ct6NN1JqCPHzXMKeAwdTFVHpFfjsvxPHJ5nuZpJ5OWkYsfxr7OMVGKiuh/DmNxlTFYmpiaj1k2/vdyMEjocVRyKrNbSZPFd3cH+pmdPoxo9Df63X/nf3s0ofEmvW4/dX0i/jmk0md1PN8dT+Gq/vLzeNPE7RPA9+7RyDDAgcj8qnkh2PThxNmkPhrv8P8jjvs/o5FVyx7HdDjDN4/8AL78F/kef6v8ACvw1rN7JqF0ZBLKSWweMnn1rqjXcVZIl8WZi3zOR6T8P8fDrSpNI0YCSGR958wc55/xqZ1XLU6KfGOYQ00+49Ij+JmsxfcghB/3a5mr7s61xvmK6R+4tL8WvEq8KsQ+i1k6MDT/XrM/7v3Feb4reLJEZFkRC3cKOP0p+xh2M3x1ml919x5tezXuozm5vrp5pW6lsVpyQ7GUuN82enP8AgjT0PW9V8OvLJpVwYnmADHA5Az6/Whwg90ck+MM2l/y+t8l/kbUvjvxZN9/UX/AKP6UuSC6HBPibNJ/FXf4f5GXN4k164BWa+kYHtmqSSeiPMqZvjamk6rfzO48Ls76UHkYszO3JOfSvmcwd6x/U3AnM8njKTu3JnRV5R+lmP4glWHSLgn+IAD8SK7cJFutE+N4rxEaGTYiUuqS+9pfqeR19efxKFABQAUAFABQAUAFABQAUAFABQAZf/lnnd2x1zQNbn9Bvwn+1/wDCtvDv27PnfY485647fpiv0OlfkVz92wHN9Wp8+9kf/9f9qPi38KNC+LXhh9A1jMciMJIZl6o4BA7HjDGuetRjVjyyPLx+Bp4yl7Ofqj8w/EP7F3xY0u7dNMt4r62B+WRZUUt+BORXzc8tqJ+7qfmtbh/GQlaC5l6r9Tnf+GRvjP8A9Alf+/0f/wAVWf8AZ1bsc/8AYWP/AOff4r/MT/hkb40f9Alf+/0f/wAVR/Z1bsH9hY//AJ9/iv8AM8/8a/A34keAYDd+INJeOBeTIhEigE45Kk45rnq4OrTV2jhxGX4nDq9aFl9/5HkRGOK888sKACgAoAKACgD1nw7IJNIhI7ZH5Yr5LGK1Vn9pcHVFPJaFuit+CNonA3E4Arh3PuW0ldvQ8z8S6ut/OLe3bMEXf1bua+nweHdOPNLdn8q8c8SRx9dYTDSvTh17v+rHMV6h+PBQAUAFABQAUAFABQAUAFABQAUAFABQAUAeoeFGDaQo9Hb+lfL4/wDin9gcBTUslh5Sf6HSV5Z+mHn3i3URLKlhEcrHy/8Aven6V9HgKNouo+p/N/iNnSqVIZdRekdZevb5WOMr2T8DCgAoAKACgAoAcqM7BEBZjwAKYHuPhX9nT4reMLJdQ0vR3EDjKtIyx5HsGINelTwNaavY9mhlWMrR56cNPkvzOq/4ZG+NH/QJX/v9H/8AFVr/AGdW7HT/AGFj/wDn3+K/zF/4ZG+M/wD0CV/7/R//ABVH9nVuwf2Fj/8An3+K/wAw/wCGRvjP/wBAlf8Av9H/APFUf2dW7B/YWP8A+ff4r/Mcv7Inxndgv9lIM/8ATaP/AOKo/s6t2D+wsf8A8+/xX+Z9DfBz9izU7LWYNc+I7rHFaOki2sbq/mFTn5iuRjjke9ejh8u5WpVGe9gOHZqaqYl2S6H6SW8EVrBHawLsjhUIoHZVGAK98/R0klZH/9D9/KACgAoAKAMzWNJsNd0y40nU4EuLa5Qo6OoYEHvg8cdRSaTVmZ1KcZxcJK6Z+BPxb8O2XhT4j+INC04j7NaXtxHGB2RZGAH4CvhcVTUKrij8KxtFUcRUpx2TZ5xXEcAUAFABQAUAbFjrl/p8PkW7DZknBGeTXJVw1Oo+aS1Pv8o4vzDLaH1ai1y72aG3et6lerslmIX0Xj+VFPDUoaxRjmXF2aY6Lp1Klo9lp+Rk11nwwUAFABQAUAFABQAUAFABQAUAFABQAUAFABQB1ega9baXbSW9wrNubI215WKwkq0lKLP23hHi7C5Xg54bExb1urFm+8XySIY7KLyyf4icn8qypZek71Hc9TNPEiU4OngKdn3f+RxrMzsXclmPJJ6mvaSSVkfglWrOrN1Kju3u2NoMQoAKACgAoAKAPpb9lPwRpPjj4sWNprAWS3sh9p8tgCHaI7gpB6g4wRXrZfTUqt30PoMloQrYyMam2/3H7awQQ20KW9ugjjjAVVUYAA6AAV9kfs6VtES0DCgAoAKACgAoA//R/fygAoAKAPNfFPxe+HXgy4+zeItbgtJem0nJ6Z7ZrOVSMd2cFXG0KT5ak0meC/E/9r74e+HNEnj8KXJ1bUZ48RGMfIhY4y2cdOvFcNbG0oRdndng4zPsNTg1RfNI/IvxBrd74k1q913UX33N/NJPIfV5GLN+pr46pNzm5PqflNWpKpOVSW7d/vMesjIKACgAoAKACgAoAKACgAoAKACgAoAKACgAoAKACgAoAKACgAoAKACgAoAKACgAoAKACgAoAKAO/wDhn4+1P4a+MLDxXpnLWsimRP78eRuX8RxXVh6zpTU0dmExM8PVjWhuj9cvCH7Wfwm8R6Ul7qOo/wBkzhR5kU45Dd8bc8elfYU8XSmrpn6vh89wlWK5pWfZnUf8NL/BT/oZoPyb/Ct/bU+53f2phP8An4ia3/aP+DN1MkEPiWBncgAcjk/hR7aHcazPCPaaPYdM1XT9ZtEvtMnS4gkGVZCCP0rZO56cJxmuaLujQplhQAUAf//S/fygAoA+e/2l/iNf/Dj4ZXmpaS2y+uybeJsfd3ggsPcZGK48VV9nSckfP5zi5YfCuUN3p95+Imoarf6pdy319M0s0zFmZjkkmvhpScnzM/GZScnzS1Znkk8nmpJEpAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAFABQAUAKCV5BxTAf5kn940gDzJP7xoA+x/2Qfir4g0D4g2Xg6SZptM1dvKMbchHJBDj34I/Gvdy+vJT9m9mfV5DjZ0sQqN/dlp8z9ha+rP10KACgD//T/fygAoA8d+Ofw0X4p/D+98ORMEuwDLAxGf3ig7R7AnFc2Ipe1puB5GZ4P61h5UlvuvU/FTxL8K/H/hXU5tM1TQrtGiYqG8l9rdwQcc5FfFzw1WDs4n43VwlalLlqRaZzn/CJeKf+gRdf9+m/wrP2U/5TD2cuwf8ACJeKf+gRdf8Afpv8KPZT/lD2cuwf8Il4p/6BF1/36b/Cj2U/5Q9nLsH/AAiXin/oEXX/AH6b/Cj2U/5Q9nLsH/CJeKf+gRdf9+m/wo9lP+UPZy7B/wAIl4p/6BF1/wB+m/wo9lP+UPZy7B/wiXin/oEXX/fpv8KPZT/lD2cuwf8ACJeKf+gRdf8Afpv8KPZT/lD2cuwf8Il4p/6BF1/36b/Cj2U/5Q9nLsH/AAiXin/oEXX/AH6b/Cj2U/5Q9nLsH/CJeKf+gRdf9+m/wo9lP+UPZy7B/wAIl4p/6BF1/wB+m/wo9lP+UPZy7B/wiXin/oEXX/fpv8KPZT/lD2cuwf8ACJeKf+gRdf8Afpv8KPZT/lD2cuwf8Il4p/6BF1/36b/Cj2U/5Q9nLsH/AAiXin/oEXX/AH6b/Cj2U/5Q9nLsH/CJeKf+gRdf9+m/wo9lP+UPZy7B/wAIl4p/6BF1/wB+m/wo9lP+UPZy7B/wiXin/oEXX/fpv8KPZT/lD2cuwf8ACJeKf+gRdf8Afpv8KPZT/lD2cuwf8Il4p/6BF1/36b/Cj2U/5Q9nLsH/AAiXin/oEXX/AH6b/Cj2U/5Q9nLsH/CJeKf+gRdf9+m/wo9lP+UPZy7B/wAIl4p/6BF1/wB+m/wo9lP+UPZy7B/wiXin/oEXX/fpv8KPZT/lD2cuwf8ACJeKf+gRdf8Afpv8KPZT/lD2cuwf8Il4p/6BF1/36b/Cj2U/5Q9nLsH/AAiXin/oEXX/AH6b/Cj2U/5Q9nLsH/CJeKf+gRdf9+m/wo9lP+UPZy7B/wAIl4p/6BF1/wB+m/wo9lP+UPZy7B/wiXin/oEXX/fpv8KPZT/lD2cuwf8ACJeKf+gRdf8Afpv8KPZT/lD2cuwf8Il4p/6BF1/36b/Cj2U/5Q9nLsH/AAiXin/oEXX/AH6b/Cj2U/5Q9nLsH/CJeKf+gRdf9+m/wo9lP+UPZy7B/wAIl4p/6BF1/wB+m/wo9lP+UPZy7B/wiXin/oEXX/fpv8KPZT/lD2cuw5PB/iyRhGmj3bMxwAImySfwo9jU/lD2UuzPuv8AZM/Z68TQ+KLfx/4sspdNt9P+e3jmQo8kmRjg4+XGefpXvYDCyjL2k1Y+2yLK6vtViKyslt6n6h19GfpgUAFAH//U/fygAoAKAKc+n2F0266topj6uisf1FKyIlCMviVyD+xdG/58Lf8A79J/hRZEexp/yr7g/sXRv+fC3/79J/hRZB7Gn/KvuD+xdG/58Lf/AL9J/hRZB7Gn/KvuD+xdG/58Lf8A79J/hRZB7Gn/ACr7g/sXRv8Anwt/+/Sf4UWQexp/yr7g/sXRv+fC3/79J/hRZB7Gn/KvuD+xdG/58Lf/AL9J/hRZB7Gn/KvuD+xdG/58Lf8A79J/hRZB7Gn/ACr7g/sXRv8Anwt/+/Sf4UWQexp/yr7g/sXRv+fC3/79J/hRZB7Gn/KvuD+xdG/58Lf/AL9J/hRZB7Gn/KvuD+xdG/58Lf8A79J/hRZB7Gn/ACr7g/sXRv8Anwt/+/Sf4UWQexp/yr7g/sXRv+fC3/79J/hRZB7Gn/KvuD+xdG/58Lf/AL9J/hRZB7Gn/KvuD+xdG/58Lf8A79J/hRZB7Gn/ACr7g/sXRv8Anwt/+/Sf4UWQexp/yr7g/sXRv+fC3/79J/hRZB7Gn/KvuD+xdG/58Lf/AL9J/hRZB7Gn/KvuD+xdG/58Lf8A79J/hRZB7Gn/ACr7g/sXRv8Anwt/+/Sf4UWQexp/yr7g/sXRv+fC3/79J/hRZB7Gn/KvuD+xdG/58Lf/AL9J/hRZB7Gn/KvuD+xdG/58Lf8A79J/hRZB7Gn/ACr7g/sXRv8Anwt/+/Sf4UWQexp/yr7g/sXRv+fC3/79J/hRZB7Gn/KvuD+xdG/58Lf/AL9J/hRZB7Gn/KvuD+xdG/58Lf8A79J/hRZB7Gn/ACr7g/sXRv8Anwt/+/Sf4UWQexp/yr7g/sXRv+fC3/79J/hRZB7Gn/KvuD+xdG/58Lf/AL9J/hRZB7Gn/KvuD+xdG/58Lf8A79J/hRZB7Gn/ACr7g/sXRv8Anwt/+/Sf4UWQexp/yr7g/sXRv+fC3/79J/hRZB7Gn/KvuD+xdG/58Lf/AL9J/hRZB7Gn/KvuD+xdG/58Lf8A79J/hRZB7Gn/ACr7g/sXRv8Anwt/+/Sf4UWQexp/yr7hRo2jqcixgBH/AEyX/CiyBUqf8q+40FVY1CoAqjoAMCmbDqACgAoA/9k="
icon6 = "/9j/4AAQSkZJRgABAQEBSgFKAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wAARCAMLAwwDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD9UqKKWgBKKWigBKKWigBKRqdTWoASiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACkZlX73SlpkihsbjgCgA8lW5zgVVuJ7a3yZJlTHWuG+JnxYsPA9i+9XkcD+A18h+Pv2jtQ1kXQ055IDghd9AH3WuvWIYBblSfqKuJN9qXMb7h7GvzR8OfFzxVBMs1zfCRByVUnNfTnwb/AGgoNUaO1ulkWTgFm6UAfTaAhQD1p1Vra+jurWOZGDhxuGKlSYO2OlAElFKeKSgAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigB1FLRQAlFLRQAlFLRQAlFLRQAlLSUtABRRRQAUUUUAFNanU1qAEooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKa0qKcFgDQA6ik3DGc8Uo56UAFYPjLW00LR3ndtvBArd3D1rx79pi+ubPwfataqzkykPs7DFAHyb8afiRc65rktqshK7iOvvXN6H8Nb3xBGrRxlt9YetYvNb8wnc275vavsP9mvSbDU9LWSTYzRruxQB8keJPBd94RYrNEyqPaq3h/xFLpsu+3Yo49DX3H8dPhfZatoc1zDAC23ORXwZf6fJo+uSwbCAGxj8aAPuj4HfEr+3tPs7WWTc6xhTk969uuWKMpHQ18Rfs3XF2ut4KsE3ccV9wQKJrWMtwcUATqxZFJ9KWkOFwOnFIZFHU0AOoqPz4843jNPVg3Q5oAWiiigAopNw9aWgAooooAKKKKACiiigAooooAKKKKACiiigB9FFFABRRRQAUUUUAFFFFACUtJS0AFFFFABRRRQAU1qdTWoASiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAqtNEsj564qzUcbKu7JoAjDdFApJEmE3ynEePWpy8aKWry34jfGGHwndSWu4Bgm7rQB6Rd6haWMe6eRQcZOTXjPx58baXdeFzbwzKz7jkA+1fP/wARP2h73UJJI7eUjPHBrx/UvG2ragxa6lYxMcDNACagscd9PKvds16t8C/i9H4PkKztti/i/OvGftHmH5j1qveKLWNgrFd3HBoA+8de+P8AomseGZYo5EJZf6V8aeKNUt9R8SSyxnILViabdSLaeWJiRjH3jS6foslzdbwSTmgD1H4W/Fi18K35DRrlDjJzXtkn7XVjDtTavH1r5Lh8M332iQpExBPBxS3HhHUC24xvQB9laf8AtZ6PcKvnBVb6Gus0n9obw/qQAMiDPtX57ahpM9rO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UUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFACr1qpc/8eN1j0b+VW161UuP+PG5+jUAfn1+0Rn/hLJvTea9H/ZQaJZhgjOa83/aK/wCRrm/3jVb4BePl8L6gfNfA3H+dAH6MbQ6juPenrgDA6Vw3gf4kWvia3QI4JxXcR/d9e9ADqKKKACiiigAooooAKKKKACiiigAooooAKKKKACqVx8tyv0NXao33yyK3+yaAPgP9qqbd4+IB7rXjun2bXevQxjnL/wBa9T/akmz8RMe61yXw904X3imDIyN9AH3j8BdFbT/CsBYbcrXqLJ826ud8CW62Phe1UDHyj+VdGrBlH0oAKKKKACiiigAooooAKKKKAH0UUUAFFFFABRRRQAUUUUAJS0lLQAUUUUAFFFFABTWp1NagBKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAQ5I4ODTWg3QvGTncOaWTIU461Xj3+Z96gD57+MXwMg8Q3MlzFa5lOTuxXx54o8NXfha6nEULReW5H5Gv1PkgSaPDqGyK8S+JvwTstWt55Y4V3NlulAHx38NPjNq3h26RFvGjAr7C+F3xwXXdJiF5dCSctjJPNfIvjv4USeGryR1QqM1zPh/xbe+HdVjijkYIpB60AfqNp+ppqNuHjOcjPFTw+Z5jbjkV4D8EvikuoWsMU8vzYA619AW1wtzCJE5BoAlooooAKKKKACiiigAooooAKKKKACiiigArO1IO1vOw+8oO324rRqOaMSROp/iBoA/Nv8AaHjuJvG8ss5LYbg4964/wPqlzaeLbMRSFQzZPvXsn7VGhCx1x5duMmvF/BWz/hKtPLHnP9aAP0s+HFw934ZtzKd5C9/oK6qE5j+lcr8MQp8L2+Ofk/oK6uEYU/WgB1FFFABRRRQAUUUUAFFFFAD6KKKACiiigAooooAKKKKAEpaSloAKKKKACiiigAprU6mtQAlFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFAB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"
icon7 = "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wAARCAJqAmkDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD9U6KKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACoZz8pAPuR61NXPeNvF2k+BPD99r2u6jFpuj2ERnuZpeioOpoA1kuTtAZSjHhU65qa3k8yPO7cc4PFcP8ADb40eC/i94ZfXvCPiC01PS1kaH7Qrbdrr1GGwa7G31K1kVFW6gd26BZF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"
icon8 = "/9j/4AAQSkZJRgABAQAASABIAAD/4QCwRXhpZgAATU0AKgAAAAgABAEaAAUAAAABAAAAPgEbAAUAAAABAAAARgEoAAMAAAABAAIAAIdpAAQAAAABAAAATgAAAAAAAABIAAAAAQAAAEgAAAABAAeQAAAHAAAABDAyMjGRAQAHAAAABAECAwCgAAAHAAAABDAxMDCgAQADAAAAAQABAACgAgAEAAAAAQAAAligAwAEAAAAAQAAAlikBgADAAAAAQAAAAAAAAAA/8AAEQgCWAJYAwEiAAIRAQMRAf/EAB8AAAEFAQEBAQEBAAAAAAAAAAABAgMEBQYHCAkKC//EALUQAAIBAwMCBAMFBQQEAAABfQECAwAEEQUSITFBBhNRYQcicRQygZGhCCNCscEVUtHwJDNicoIJChYXGBkaJSYnKCkqNDU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6g4SFhoeIiYqSk5SVlpeYmZqio6Slpqeoqaqys7S1tre4ubrCw8TFxsfIycrS09TV1tfY2drh4uPk5ebn6Onq8fLz9PX29/j5+v/EAB8BAAMBAQEBAQEBAQEAAAAAAAABAgMEBQYHCAkKC//EALURAAIBAgQEAwQHBQQEAAECdwABAgMRBAUhMQYSQVEHYXETIjKBCBRCkaGxwQkjM1LwFWJy0QoWJDThJfEXGBkaJicoKSo1Njc4OTpDREVGR0hJSlNUVVZXWFlaY2RlZmdoaWpzdHV2d3h5eoKDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uLj5OXm5+jp6vLz9PX29/j5+v/bAEMAAgICAgICAwICAwUDAwMFBgUFBQUGCAYGBgYGCAoICAgICAgKCgoKCgoKCgwMDAwMDA4ODg4ODw8PDw8PDw8PD//bAEMBAgMDBAQEBwQEBxALCQsQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEP/dAAQAJv/aAAwDAQACEQMRAD8A/fyiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooA//Q/fyiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooA//R/fyiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooA//S/fyiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooA//T/fyiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooA//U/fyiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooA//V/fyiiigAooooAKKKKACiiigAooooAKKKKACio6KAJKjqSo6AJKKjooAkoqOigCSo6KKAJKKKKACi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EVj/AGbrFxaf3Gr9Q/8AgnL8Xf8AhFvGVvpN1JtWeQKPqW4r5D/am+HUngP4gXdv5TRpJKa4z4C+I5PD/wAQtHvPN8tEuI6AP7c9H1G31WwhvLf7kiVsV4R+z94jt/Efw9064jff+5jr3OgCSiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKjoAkr8+P27vibb+DvhpqNv5ux5E/vV9+XdwlvbSzv0jXdX83//AAU0+Mv9raw/hqzl+T95/FQB+OvjDVZNZ8Q3d/I+/wAxq/Rv/gnl8L5PE/ja11C4i3pA++vzHggmvbgRR/O71/Tx/wAE2fhPHofhhNYvLfY8kNAH6u6Hp1vpumwW8SbPLSt2o9g24qSgAooooAKKKKACiiigCOipKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAP/9P9/KKKKACqs6JJC8f99amooA/Az/gpT+z15ltdeLNPt/8AUeY/yLX4LWk9xoeqpJ9x4Hr+2745/DXT/iN4L1HR7iJZHuItlfyG/tKfC27+HXjq8haDy7dpH2fTdx+lAH78/wDBOn4z23iLwvbaLdz/ADxx9M9gtfrOjfLX8d/7Gvxs1D4c+ObK3+0MkEk0aV/Wr8PfFtn4t8PWuqW77/MSgDu6koooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiimfc96APKvjB4sh8J+Db++ZtreW+PrtOP1r+PT9qDxxJ41+I17cebvSN5K/fj/gop8cLfwr4MutDs7j9/JX8xWq3U+s6vJcH53negD339mb4WXvxH8f6XZpF5kDy4ev69vgh4Es/BHgmx06CPY6JHX46/wDBMv4H+W6eJNQtP9X5bpvr97oIUgiWOPpQBZooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAP//U/fyiiigAooooAjdA64evx1/4KDfsvW/irRLrxRo9v+/gTf8AItfsdXMeKPDmn+J9Kn0vUEV0nTZzQB/CzNBqngjxDiRGgntHr+hL/gnz+1fZ6rpVl4X8SXex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CiiigAooooAKKKjoA+Bv28dRjtPhdqEcn8cMlfyTeJpPO8Q3cn/TWv6tv+Cgt/aQfDW4jmbY7hsf981/KTqMcd14kkjj/wCWktAH9EH/AAS0/wCQE2fX/wBlFftlX5J/8E1PCEumeDVvj3C/ov8A9ev1soAkooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigD/9H9/KKKKACiiigAoqOpKAPwc/4Ki+Do59NfUP8Abr8FvCWovpXiO0uf7ksdf1J/8FEfA/8AbngC6vI08x4031/KnOkljqn7z/lm9AH9g37FXjuPxX8OrL597+VX21P/AKpvpX4xf8EwPFsmpeG7XT5Hr9n3+5QB/OV/wU+/5Clfkn8GP+R+0r/rvF/6GK/cb/gqF4D/ANG/tSNP4N9fhz8GFZviBpbKOBNGST0A3CgD+yj9nn/knWl/9cY698rwP9niMx/DnTN/XyY698oAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACo6kooA/KP/AIKXyPH4CXHXLf8AoJr+ZDQ7b7d4vtbb/npdY/Wv6av+Cl//ACIS465b/wBBNfzgfDDTf7S+I9hb/wDT1F/6GKAP6t/2GvDP9jfC2Jz1cKfzWvuWvnf9mTS/7K+Gdjb/AOxH/KvoigCSiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAP/9L9/KKKKACiiigAooooA8h+MPw+tviJ4UvdIuE3+ZDIlfz++OP+CcHiSfxU8mnxN5Mj/wB2v6XKha1t2beY1z/u0AfBn7Hf7MsfwW0SD7RFsnjr75oCIn3eKKAPlr9pb4E2fxm8NTWFxErvs2V+bPwh/wCCa0fhnxUmsahb/JA+9Plr9y6Nka9FoA5fwl4ft/DWjwaXAmxIE2V1lR1JQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFR1JRQB+Uf/BSuF5PAK+Uvdv/AEE1/OD8NtP1pviHYtYxkMt1EST0A3iv7Gvjv8GdL+L/AIdfR9QTfXyN8J/2APA/gfXv7YktPnjff89AH1T+zH/aH/CtLL+0N2/ZHX0hWPoej22h6emn2i7I0rYoAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooA//T/fyiiigAooooAKKKKACiiigCOpKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAP/1P38ooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAP/2Q=="
icon9 = "iVBORw0KGgoAAAANSUhEUgAAAP0AAAD9CAYAAAB3NXH8AAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAACxIAAAsSAdLdfvwAAMJ8SURBVHhe7Z0FnJXV1saH7u6QRkXBRMHuVuxur3r1mtfW67WujS1Id0oo2IrY3Yp0w8DMEMPA9AzwfOu/h8V9Pd8AE4ByOevH5pzzzhs71rNqr73fBMUpTnHaqSgO+jjFaSejOOjjFKedjOKgj1OcdjKKgz5OcdrJKA76OMVpJ6M46OMUp52M4qCPU5x2MoqDPk5x2skoDvo4xWknozjo4xSnnYzioI9TnHYyioM+TnHaySgO+jjFaSejOOjjFKedjOKgj1OcdjKKgz5OcfoL0vr160OB1q5dq/z8/PAZJf97cSkO+jjF6U8mB3gsiKO/+b5u3bo/nBv9e3EoDvo4xWkHotKA3SkO+jjF6S9EaHPMeNfqfI+CPA76OMVpB6dNAdgB78RvFwguFEpKcdDHKU5/QXKQxxLHSgN4KA76OMXpTyTX5oA8OztbqampofAdWrNmTfi+KYugJBQHfZzi9BcgwP3rr79q/PjxGjBggEaNGqV3331XM2fO1JIlS5SZmbnhzP9aASUVBHHQxylOpSCAFy3FJTfV58yZo0ceeURdu3ZVx44dw+dhhx2mCy+8UM8884y+/PJLJScnBytg5cqVysnJ2fg8PmNNfv9bXl5e+HSLgs846OMUp1JQLND5HS3ugwO26Lmxmnru3Lm6/fbb1aFDB1WrVk21a9dWvXr11KxZMx1yyCH6+9//rqefflq9e/dW//79NXLkyGAVTJw4cePnRx99pG+//VY//vhjsBqmTZum33//XbNmzdLixYs3CoA46OMUp61IUaBHQR39DvF3J74vXbpU999/v9q2basqVaqoYcOGqlSpkhISElSmTBnVr19fu+22m/bbbz/tv//+4RNr4MADD9QBBxygI444Qmeffbauuuoq3XDDDbrzzjv18MMP65577tG9996rF154IQiEZcuWxUEfpzhtTQLchYE9Os3GsSjooYULF+qJJ54Imn2fffbRqaeeqiOPPFI1a9YMwKeULVtWVatW/YMw4LNcuXLBOsAyaNKkiVq0aBEExN577x2EA58nnXSShg0bFiyKOOjjFKetQLFg3xxFzyWnnu9paWm66667AkgB+913360rrrgigLhixYoB3Gj/Ll266Pjjj9exxx4bPo866qjwHU3fuXNn7bnnnsFacEuhcuXKwVXAGiBAuGDBgjjo4xSn0hKgjTXn+Q2g+cSX9r/xO1bLQykpKcEMb9q0qY455pjg359//vlBswP4ChUqBJBjpr/55puaPHmyPv74Y7333nt6++23NWLEiPC3//znP7rppptCHAA3oXr16kHTY/J/9tlnYZYgDvo4xamU5KB34jtTbETaZ8+evTGwNm/evFASExODZk9PTw8+NtF4Am3//ve/g4ZGKxO133XXXQPgXcsjFKZPn66MjIzgLuTm5gbBkpWVFe4zf/78MAuAQLjxxhvVvHnzcN25556rCRMmhGdwfhz0cYpTKSkW9Gh2AA+4Aepll10Wys0336xbbrklBNeIxL/44ov6+uuvQ4T9t99+Cxq6cePG2n333YOJj5bGX8e851ifPn2CoADwTNkhOHgW4Af4ABoiPsD0X7t27cJsAG7DokWLgiDi/Djo4xSnUhKgd+CjhSE0LyY3kXaCcfjWderUCVqbgBxmfOvWrUPQ7qyzzlKPHj2CD492Bqh77LFHCM5xfvny5YOpjz/O/ZmaI3Fn7NixIYnnnXfe0euvvx6OA26IoCC+fPv27TVw4MBgTSAcqGMc9HGKUykIEPkcPAWNC+GjDx06NIDYTfRddtklTL3VqFEjCADAT6AOkF966aUhGMdxpuLOOOOMEMlHWODXX3zxxZoxY0awCv71r3/pvPPOC0E8zj3xxBN18sknB0vhjTfe0KpVq/T4448HqwEXYdCgQcHy8NhCHPRxilMpCMA76CGABa1YsSKAvlWrVsFEZwoNUD7wwAO67bbbgrm/7777qm7dumGqjfn2vfbaKwAVHxxNDZjR1ggMsvNIwmHa7bjjjgvCpEGDBiHA51N4mPME8piWe/DBB/8AegSBm/9x0McpTqUgN+udEAAQ/jbmPWY6wD3ttNP0/vvvB7MfX59Eme7duwcTH23PVBumOHPsCIdPPvkkTLEBdgCNi3DNNdeEKDxTcwgMrAE0PpF6rAeEx6233hriAwgWrkEQcB80fRz0cYrTViLX8lFiamz48OEB1CTRYIoTVXfCx8YPx6dHa5Nvj1YG9HfccUeIwrPYhuAdwgDgIxgw5zmP6bunnnoqCAeCgkT8cR2uvfZaff/997ruuuvClB3P79evX7A88OmhOOjjFKdSkPv0sYSmx6wGdAD2oIMOCgE3zqfwdzQ/IG7UqFEAfadOnUJiDdF2ptewCn755Zdg5mPG4wpgsuP3H3rooXrppZfCCjyEC/EAfH+m+licc+WVV4ZAIO4FgmP58uUbF+nEQR+nOJWCoua9a3w+ATVmNb48U25HH330HzQ9guLnn3/W4YcfHoCMcOBcgnpM8wFm7kEiDuY792AKDyDjw5O5x5Qfc/PEDpji4xziASThEPgrTNPHQR+nOG1Fco0PsDDv0fRoWsx7fHPX9O5bY8JznHOI1KPlATPz+ETqmW8nIOczAAAeIKP1OQ8NDuiZtjvllFNC7ACwf/755wH8zA5wT1bl4U5s9+i9NzYqDamES8ko+TlOXMfuIe6TcA2dyvpi5iWJVkaXEbqU9N1HnGLvGx2kWBMtei7fY6+FCqv7X4283rHtia07v3eE9mxvih13fnvx3xB95zzEJyADbJjcBNTw6dHM33zzTQi0TZo0Kfwdsx4Nj9/uGv/ZZ5/VmDFj9Morr6hly5ZBaBCowxVg3h/wYxEwE8C9OJcFNUzvkbOPdUDgEM1PIJFluD6rAG1z0BfGSN5B/jc6ju/RjgSwq1evDmYJ0gxTCAn2wQcf6LXXXgumE7nGvXr1Cp1JUOPJJ5/Uc889F8yZ0aNHh9TDTz/9NJg7XEv2E+mQ3I9oJs8gwylaR+oVHcjCiL/7OTsKRft7S0TbONeZOE7FJ/zncePGheAaYGSOHm18/fXXh+j8JZdcEoJxHAfwgJhIO5l3f/vb3/Tyyy8HX55rATzf4e9//vOfG89Fq7O+nnl7BAJTeMwI4EYAeiwCBAo+f3TTjT/FvC+MqdDmZBuR1IC2ZkqDDCOyjmgYa43pDKYpkJqYNzSI4AedQMdRkGxMZxA4wXQiaYHOphO4lkAH0x7MZ/bs2VMffvhhkJYIAadYUPtvr3dRwfNXoFiw8z22LXx3in6P03/J+8v7LkpuwUb7k5RXQA+o0bho61q1aoVceAQB/jkmO9NsaHc0PryLtgfArKZDy3MO/IwpT94+Wps5fQQB/M65XEeQj/NxKb744ouwNJfnYt4PHjw41Mfxtl1AT6cgaSjuV0B0EH/DFGIhARKK5APMGxYfMHVBIgIdgGSjYQQxSHagE73QMdES/RspjEg8fCHPhMJ/IhGCiCcSl80GMJGwBnATsC42p+UKG/i/MjkzRgUA9Xd3KU6lo8J4ASuV3WyIsjPVxjQbnxS0Ob8BO0A955xzdOaZZ4bjpOdScAvwydu0aRPy9Zm+g8i4I7CHy4CvD3/D8wAcRYgVTPQeywCeR+j07ds31Ge7Tdk5k0WJY2j1pKSkMFdJpTBRSEVkbTDaG4nnDY8FNsAHvCQ1eMfRODoBQNNpXI9UpTMQEnQOphLXcj/uy2/uw3VITAQA9cDXYmUU5j9CySnalh0F+NSxMAHGMQQwxYUBJdqmzQm+nYXog9hxpp84Hts/fh5/J6b0ww8/BEuVJB13NynwPAVLgP3v8MHJwIPv4V/4lgJ/slrurbfeCgoTQjG6e0AiDkoR5QXIEQ7M0ePGMlsA73Mf8vpRZKQIU8ftoum9M/gE7GQkfffdd6EDML/R5ICWRkbBDTiRVphASDwSGZiaQDLiGxHldF+eTiPf+NFHH9VDDz2k++67L8x3/uMf/wiLFQh0nHDCCaGDkKhofX+GfyIQECRMo7CembgBVogPbuzg72hEOxh4UjIZAwKhMOZPP/0UCnPCuFZbsnR2VmL8vTjF/ob4Tf+hNNzCjSoMJ84h6EykHsWHm+o+PiY7vE7En620IBQQQWvMd8aL2BTxKl9TT/wKNxWFBQ+j2LAIsJxRsGCPemxz0HsHQDAcATl8dFYUoV0BHxLJgY5G9mSFgw8+OEg15i0JUKCBkZxEPhEaU6dODZ1GoWMoMDOrkTCH+DvPc2sC1wHpyvORkuwphkmEVYA14HVA0NDxBFqQ0twD88h9tx2JqC/1hgHpJ5iFPkQ4Mh1EaieFeAmbLxI78TYTX4lTARUGbLcCmUlCs0etQud5PimALWoh8N0j6vQzwWgAz4YX8CVK7NVXXw087ffiGkz0aCQe4jiChaA0RLAa5ebWLUoRXFBP7lV60NMXW8CBN5B5SQB3+umnB5MccP0X8GVDoGOfffYLUclHHnk0nItU+3XKb4FhM7Iyw71iH8cAhONhYJCofKdj8415fwygh5ERFAge6oJQQHhgETCniWTl+VHw02EcJ7ZAuiOa0RMceN6OQAwyDEGGF3ELgphM+xBMQrChDdyFIvbBnDFMh9mJJbaz0x95q6A/0aZsjoGVRBwIcNJXxIMceE5c54X7RPmGYwgKLCtiWATosHpJq0VrM0/vfrg/34n7MKbR4y4MsNjYfQf3mDF97LHHwtQ2iqv0oOd5AWf2X6ROfjg3zySUfVmyJEnTfp+u5555PkTUmWcsU64AWLVq1Fbrlm3U5YCDddEFF+ulF18x8+Rnpa5KD9eH+9j9c/PzlLM23+BsktL+D5/rCyTr2rUFnQfI7VsoOXmZSlm+VFdfe6UOO+JQ7bXP3rrkskvVt38/ff3tN0pNW6X8dQXzqVOmTAlTfwgjZgTorHIVyiuhjGn9alVVr0F9dT34ID3y6H+CAEpZvsyeXSB83O/3gY0dnD+LosyFSUj0F6sGy8pXbnkQiAJzUAA/gpioL8Elch+caFv0vrFUWB/EMnn0b5v67sQxGNk1nZPfJ/aawo5Hf3tdor/dz4UYSwrk56JpARdBOSxN3EmsIha6EH8iqEbeO4IUFxLBSpTdARiti5Mf80/SbQEmihC/HmWHICgOcS8EDu4ZEXxmq4h3IdCxlhlH78dSg35drjWOulvJz1+n/AKch7LW+m3+vMUmDb/UP2+5TYcefIiaN222geHKhs+Dux6ka668Si+/2MP8knetctO0YuUqA7pdb/cA1tHCMUDvxZ9VUBAGecpfn61Vacs0d95MnXDiMWrYsH54Fm4DZs9DjzysyR9/pFlzZoeOYOCxQgioIHE5Z5eWLVS3fj0TTqYJy5dToyaNdchhh+q2O27XhDcmata8uSZYcjeCPpahoO0dHefZUYaiXkxHEptA8jOP65YMn1hantaJMGAayWMcHMfUxC3CwoGcafw53H9L5PUpjLgevxQ3jOATlhe+KQU/ld8cpw2uqQp7ptcnSoXVL1p/p6hAAag+jgCIupELQvyIwJhH3bGG6C9iTViHBIMRpMSdAP7zzz8ftD7mdJQKqyMEwIlHAVLvc+IuXpfNUfQc8usROhdccEGoI+PLWOK2YfJ7f2w9894Kbcgz4AP23Nx8q/hqffLRpwHwBx9wkCqWN01ilaAi9evX1Z4dd1PvXi/p4w/f1aKFc62T0pSVn61cgy6xSgpbEiB7gQ+Fav9Xn5sFYL+RqQW/sQwKzsjOWqPkpMW69ZabtM9ee6tGNWNocyH4JCB4//33GcgnhsGl8xl8Ah1MHTIXStSUfAC0IdIyoayZv5Uqqv1uu+q007tp8NAh+uSzTzduQRRlsMIEwLai6DP47r+pAxoEwBMQQuA52CnMcBDsIciJhcPuLQQ6YVzOoxD5ZR91LCHux739MwoWyJ8dW5y4jr4ioAQDAmpcLoKx9PVFF10U8inQnhSi0fxGo7J4hKWi5Fbg+wIK5p7JOSf2ADgZO6fY51L8e5Rij/t1jCWxD54D2LH+3A0NvLChfyjwhwtK3EEEK7NBbFcF8J1i+8N/U4gHYEEwLY3FwE43HnQrDuG3cx/iYQhw6oz1Rv/S5259bJVAHjfz9gB4ytKlyfp48if625VXq8t+B6p65SqqUKasylnntGq5i844o5teePEZG7DpppWTlbcuywCcayBfpwyD7mq7V5qVlVZIm+E3cpM3eiEMXABkmpltBpoJBrMy7BNtb8Oo/Owspa1Yrk8+nKwH73tAB3Xuqib1GwfgV69aTe3btTGGOidER/Hv0SJODALJQTAVswt0YuWqVYLWx+Rv2bqVzjz7LD38n0dCZyJhGSAfVD5jQbGtyJ8JRZkE64UoLqYogUpnUhgUbYVvzzQSPikalRVfMBtCAA3GuUR+YXiYHz+WNvnzeBaFY1g0CE9cJawC4i8wIGYx5ib1AOSYnYABQYJAZc4aIcM0FdoStwqrwy0Qtz7QpsQgEFRoMOrPmCC8WUJK8JE6MpY8C18YLRvtm8LIxwiQu9DmGqw+tDWCBxAHd88Epbs+9Av19bwR71e3ojgfgYrlyDgURj5WPA/eI9iGpqdNuBEIyKKS3wu+RWERo8JioK+ohy/M8fZuFdDTtV6oANHI8WPH6fq/X6c2LVqqcd36qpBQRuWt7Na2nc49+xxjuJGa8vvPysnPMLDmGIjzlW5gXWEl2Xx4JikwKhOtLLQbzzOUL8hep6U2NqvsGAIAmGIFICTScrOUsmqlFixaGAIZH30wSa+PeU39evbRDVffoEO7HKYalaurUtmKQfBUr1xJzZo2DgyIdGQulI7xwcdsx8SCkWAqZhKQ5C7V0YL4XzAxgHFf0DvWP7cHOXNHmRwAkqhBEhLSHoZEKwEwAnUEJvFXuYZAH8yJlUNE37U91/n67gLBXnB/mJT7A2gEBn3Hog/iBkwP+bQpueGYlggSQE5/cT8Aw715xkZLyr5T+M6xwgp/878DNsDFvWgTQgCtzKwQACKJBW0NEGhfYX3k5IILQngR+CV702Mf9B3gJ/BJsI1ZHQQClhGgQjDCG94m6kgAmFkRAn5RXvDnR0FPX8JHCGcEGvPqUSW0JfJ70QbiN/AjMSpmAMjfJ4DtlhDPKzXoCbDRDAJbBMaoLIxww/X/UNfOB6hq+YoG9gRVNA3bof2uuvziS/TOW28HgKWbNs4xkCPTVtldkuz7ArvZjNz1+s0w9JtZI7/aWPxoJ3ybmq3vV2ZpWsZaLTBcJlpZlJmrRGvMjMRF+vjbbzRy3Bh1f+Y5/eMfN6rbqafrqCOODQHCtq13Va3q9VSlEia+mWNWF9yMqlUqBWlIgAs/CG3hQIAQAGgrMpzYyZTUXrRPlEE5RqCEpIjo4Ea/b2uKZWSYAOsD0x7tEdVCmPmsw6ZdsYRvSRAJbeNtBFT4qfjVXIPJClOxvoE8cGZaECIkiQAImB2hAbiZgqJ/AQyg8HtSqBP9578BMiBGq1M81sA5XqLXF1a4J9ciXHALyLUA/IwN44rgYlzoLweKk/8mCEeGJkD2+2KuIwQQYvQpkXosFxJlsAYRbOxmg4Xi/YwVgCWD4EWQOPlY+Sf1YayYLeGZ8CLCE0FQFKLesXzH85j7p91YLbHPLzXoqTpmNUGtpJRkY4h3zYf4h/bqtKca168XNHyNSlXUoW17/f1vV4dg3eLEpUrLyAz++tLcdZqXt04zc/I1zQAO0L/MWK9JK/L0zvJcvb0sR2+nZOudpRmalLRGny/P1FfL0/XxgiS9O2W6Xhk9Ro8895wuNyFzLKb4PvurUePmql6jjmrVbqD6jZqrcrXaqlazgRo3a6V27TuoXv2GgcFca2NCwigwcVQyB0awBq5YtlyffvyJ/nXvfSHwWL9uPZW1dpUzd4XBJTeaTD78SzQmwiIWiNuSYp9F/bFSMBPdtHctBCiJRFNH2ucFgjkw+YlIO/MCRoQATIkGYu0CGhXfE60EoNGIUYb3Qv8Cdgcsf6evETwIBkCBpiTISLSZe6OpL7/88iCEiTPwN3x8PjmXSDnXMbtAtBs/mIQqxtOficDgGfwdYXT11VeH+gNU4hMAyjW79wNEP+Lu8GyEHfejv/iNJQOIyAHBdaB/IYQJliXjT3IN/cV17qZgJRLH8DGKHSsEEVoY4URGKjvfoDSjQC0qeTsgvtNGbyfkfy8V6Kk+Jc+0fHZujr79/hszBW8zJtlLDerUDhq+ZuWqatG4qS4+7yJr2FuaMcc0fH5BoA4DZo7VY4pp7W9Nm3+StlbvGdhfW5qpsUuz9NrydXo1MUtjEzM1flGaxs9N0dipCzXwix/15Jg3ddeLvXVotzO0+/4HqFqDRkqoYP5VGabayqtKzbpq1tZ8P9P0hxx/so44uZtOv+gynXPJ5TrkyGPUrGUr1ahZOzAJg+Ogj86NBrIGrl+7TosXmjUx+SPd+I8b1GG33VWtSlWVL1tgMuNrYv7jB8IUHjDZXsAvDPT4hL4xIyAAfNSVCDRM5W5MlBBYJC+RL+4CEaCiPUmkAuiAFhC7LxtbOJ/+5BpiAwgFQAlIuR4QooXRjgTn2OwRhmfhE5F6AEScBN+cZajkaXz11Vfhk3ojlOhnAoDEJbAyMLcRvIwDz/a684kw4DiCi2AhWhxNjcmLewK4MP8h+gTTHuGCkKSd1JcIPvXy86IAptDX1A/BhCVIH9Df1IVnYj1EtbGTAxMirsQCMCwJrODi8k4U8H9QWhuIY86XxQZ99EZUi8AZU2dTpv6mhx550EyU9qpWvZKqVaqoKhXMh2zUWHfccru++vI7paxYrXSrD94Fs5D461+Yc/5OqjQ+OUejDOAjDOAjluaEMnJJtsYm5+nVxas1dNpi9frqF905YJTOuuPf2v2EU7XLgYcooWY9laluvleVmqrRqImat9tNnbocFAB+ze13qcfIMRr69vt67fOv9P6PP+n9b77XSwMGqfPhR6pmvYYqU9bMyurVtEfHPXX5lVdo2ozpBS7Lhk7fqBHz1yotdZU+++RT3f7P29Rxjz0D8GEuQMUAY9KiUZDsDirvr1hG8e9bm/y51GHIkCFB21FH6kcQiuQPmBCAR+sGU2DlACo0a9SNcSZ2LeaF+wFotDYaEQ3MJ+DG54XhiREQdcfXx/IAcIAY87MwF2NTRB1pGyAlZoRwBbRobuIT+LDkngM86oDQoY5uZQBiT+dGQBPsIshJEgz3QHtjAZBoQ9CQ84kTEBgjqMszidtEYxuQ9zdtQZDRzzzPwY/lwvXRa2LHnb5HoCA8KHwvTEhsLdoi6KlgFOixBOgXLl2k1yaM1/kXnqf6DczUq4zfXBAsu+SiizV8+EgtWJKiNdY/GEVJ1uZZ2evMV1+nd0zdjzHQD0/O1+DEbA0xDT88KUejrbyalKVeP8/TU5O/090jJ+j8B59S+xPOUL19DlSFFm2VUK+xKjZqobot26vtXp11TLezdM3Nt+vR51/UqHfe14e//Kr3f52iL+bN17dLluqn5BR9v3iJ3vz6G1175z3a/7AjlGCMXN7qWb9hAx1+9FEaMmyoVqSuDAItgN3bHg6sV5LdZ9yYsbr4wovUqkXBBgcMLgXQsy4APw4m4FpnCh/oaH9url+LSn4Pv7//hkkBMBrWNTOFGAQCAfK64MtzDAbHFIbZHeCuNbkHbUTT4zIgTLAaMMUxbdFQaElcBzQ3/i7MjuWECQygMGPRbDC0ly0RdfS2FUaAkGAd/Y0gQaiQa4HQIYKN6xG1SmgH4Ecw0BfEK9z3x9KgLbQR4DNliMWBz+0UO2a0iXZwDqY8AV6egyKgv7gHFkq0rbHt4Z60A7BT+B77nK1JRQK9M4dTdCCycjI1afIHuvbvV2n3Du1UqWJZlS2ToLp1amm/ffbVmDHjNGPughBlJ+JORH6GWTRfrcjUW8npBvJM9U3KVj8rQ1IM7MvzNSrJjs9OUv9fZgewX/TI09rztPNUsV1H0+wNlFDRtFCFaipTv6l2P+RonXjhlbrniec05t3J+mH6bM1YmqL5q1drQWaWpq1J1wzTalMz7HtmtqbZ588pyzTorXd10/0PqoaBvUwl83fLl9MubVvrsquu1Lc//mDWi7XTCgHKghSgAqLdaJlePV9Rt1MLdidxfxltiI+JpsS0ZvoKjerXeZ9tTYKZovf17wCZJBcY3+uIqYs/jrb089D6mKasbyAaDeA5D4alTVwHCDDTEWqY+WgvIvS0kUwvAAfzextj6+RU2LHikN8/Sv6bZ1LgU7QlACZyjbVBmxFShbkkHMNaoe3sOoMQwJ/nfFayOQj9/lC0Hmh/nomlgNtB4NTvi4Ch/7EeuEeU/F5O3I/7UGLbuLWpyOZ9LOidZs2Zqf889pB2272tqteorArljVGsdN5/35AUMG3WbK1Zu34j4KeZW/Tl6jy9nbRGoxNXqf+SDA1YlqfhK9dq7Grz4VPS9fIPU3Xb0HG6xIDcods5qt5pPyU0aGYmvJnxFc2nbNJSrQ84VAefeaEe6jNY/d6apE+nztG8VVlaYQIl1foz1eoYXAgD7czcHP1gpvlPNjC/pmfq59TVmjxrnoa++4GOP+ssVa1n9y1nWqByRbXZfVc9+vSTSiLV1q6npd5abzcDNuXX3zRowMAQtWXe1jUiAEMTksKK34gWgrg2OtBba2Bj7+O/eS6MjykL81E/TE/SoGFQgMFcOuCFudF6BCVpA+fzSbvwh3EJmNokJ5y5ZyL4aG9vG1RYe5xn+Fu0lJS41oHhIIzypf8N4m/MJBErIAjHFCIBOfqDWQXGydtK32AREBcgUo+WJkjJjM2miOdE20NgD9Aj9LkvQpP7ElMh/wHhAfn5bgH+GVQq0CO9vv76S1162YWqV9cY3xpLadmiqa677lq9P+kDJa1aHabkmID4PXO9PkvN1tvLszUycbWZ86s1bLn57Gbij0tbpxGLU/WC+e1XvdhLe517iWru3VkJdUyzV64WfPaE2o1UY4/9dPgl1+rWF/rpuQkfaPxvs/Xp4hWaY2DHdSBegG5dY9V14C80s/yntNX6weryY3qWWRmr9FXKSn1qFsh9Tz+l+i0RKKatDfjVG9TVcd1O0efffi0yA4FpQdKP9cGGTwpmKowPEAAMA80gU2AkAkcs5gFYTjAi5My7LcjHBgsDLU5EHCZ35mZOG3OTJZlMQRHAREjxdwcC5+EWYLHgi+MmYObSXg88Uf9oe7xEKfZ3lPhbUfpgc/dwoh5RTRqtC59oY9J4CQwiANDoaH/a6L4/Be2MW4PFhvmPoEPAuXBzoEfb7Z9YPNwXa8j5AOAT52C7Ku83v8ZB7/eMJY5tKx4pknkPeQX8Nw1PTFyku++5Xfvs21EVKxQAvnq1ijr8sK7q16+Pps+epdX5a4OWn2/99MWKjGDSj0427b4kPWj50SvXaUxKnvr8vkj3T5ik8x7prvYnn6GEpq2UUMnATqlVTzV231t7nHKOLnv0eT3+xsca9NM8vbF4jd5IXKnJy9L0S2a+gbvAhUCmUgA/wMcjm52br19My/+Umacvlqfpq5Vr9M2yFRo16X0dePThqlLftL1ZKAlVK6pO00b6zzNPae7ihcpdtzYUWk0B+GGxzYZ+IG+c5JXovC4FrUliDFt+xS6eKGyQS0vcM3pfxgsznnl35n7dGkGLkTQC0zOvDtgpzvQICDS7R9Yx3QtLFNkcQ8a2j9+cD89QSsrM3sZocQBGieOQ/43f/kzAR8CS5cXkuyOw8d8BqLs0FI75ijeSjxAaaHN/LuTfEa7MreMa4ApxPfeir0lxZl9H6uLXQV436kWJ/o3vfnxbULFB7xVE+n366cdmxu+thg1qC8Djyzdv1lBXXnGxaYbvlZaVsVHL/5aep/cMoOOWrtHwZVnqa/57v+SCCH2vH+bphr5j1PWqW1XvgCOU0HAX07wmgWvUVfPOXdX2iGN0xm3/0u0DXtUzn/ykfjOWaazd9HVD+Nhl6ZqYvFKfp2ZoWnqulllfIlPzrdp0a2pOnh1bryX2e2b+Ov1uAvaLVZnh/C+WrdSXs2fq5n/fo+a7m1lWvbK5D2WDALjqhus0d+li0/Jo+oJ7UeiNjcC376SnojV5N5lHygEXAMJkxEQkmaUk865FIR8fPv075N+ZW8a3ZeoJRgbgmLAeZaauHENDkTWHtgIQTCExdRSlKONG+YHjsUwNbSumjaUogCD/DXEsth5oWQQZ6cLEJXy5MX3gkXsHLtF++uXOO+8MZjpJStG0azLgmJUgYIcrxHUU4iLkChAYRDFAsfXg+i2VbUFFNu+9AlScTsVsHTFiWEjAKW9mcTnTkmXKJqhd+1a69967wwo2tC06YrFd+uXKTI2dv0yjl6Zr+Io8DVyer6HLctX/5wV6cOS7Ovyq21S23T5myptfWcm0bpNWarhvV5116z268aV+6vHZzxo8I0mjU/I1ZNk69Ular5cWZGiwCZFXl5ipnpYVMvnI0wf0ppxluFSWIXZFdn4w85dYmWvlOxMOn6dmmuWRqnem/KqnBvTTbl0OVBUWpZQvq2oN6uvY00/RTzOnbsjpJxfBmMWEB71AYVkvxj/9QYQahkBzYhoGjWECkNV55OmzJJfMKJiNOf+wFNkolgm2FjFWzviYvcQWMNMJwsGQPh0H4Kkv0Wy0FFl2AIGFOtFZBx/7KBOWtO7R+5WGCquX12lTv6FovfkOgBFw5KcTh8Enj5r8FAQkAp3+YwqSmQmEPX3LbAVJNQh4t5hwk7D0uAZLK3aVolO0Lt4eL9uSigx6iMpgHhEEIlnhb1deFZJvyGXHHyYKfnK30zT61fHmU7IIxoBmyP/N7OwJC9M0aNYyDVq4WmNWrlff+avU8+e5uu7pHup07KlKaNTatHsdJTRooQp77K9WJ56j8x97SQN/nqdBM5M0bFGGhtjN+lrpuTRfLyevU6+U9eqdmKExZjl8lrFO04xPTR6E5wayvss1tGZaXyMMyOefZ99/t/r8sDpX36zK0EQb8BFff6UzrvuH6rfd3YSOmWflyqtVxz10w9236/Off9Qvs2Zq1txFWrwoWWmp6SFSzXp98ySVnVuwHnve3Nl6/LH/mBZtamAvE/ojoWI5VaxeVbt37GTM8rBSkpaFOoViIoP1/6UlZ6Qoo/gxPmE4QE/EnSk2zHwHPJ9kkeGekGoMI7vvCUVB49/59GdFmTbK0LHMXRRCyETvB/lzom1z4lhhx0tC1JcpRXx+wE/CDxoegUg/IcQBNKY7gTmsJgKgvCMeC4GYCJod6877lpgA2XVMnRI49T6hznzfXJvoh2hBwbIGhJWF5BT4eSWlYoEeosKY9kRxjz36GFUrW0kVEsqpUrWqarfH7rrl9rv05Tc/ylxoYdCSJ//pinV6LSlHA+emhvn3QfNWqftX03RDv1HqdOLpqtaijYHNmLHFrqp9wOHa/ZzLdfnLQ/To5J/s3DQxfz/U8DLY1HVfc9B7pUg97XtvM/EHWxmzcp0+XLNWvxm/miwIvnwg06go1mzrHywONn9aaH0/NT1f36Ss0UcLkzTUANH/40919j9uVYN2expQrR7lKqpm40Y66LijdNsD9+tfjz2uhx/trqeffFlDBo7Qhx9M1vRZU5SetSpMWdInudk5+uyjybrowvPVqo2Z+Wb1JFQ2jV+poho0a2Z+9N/1w9ffG3dbhYKmh8H/yOQlIR/8KGD4jhYigEXiClloLK9l2g2mJFgFoxLEw2clsEfcIZaZovf045v6e2HnFoUcDBDfYwHB98J+R49tDeJ+gBOtz5w9i7BYU0Egjn4D+IAaLc5vZkHYZhptju/uQhThwDGWAzNVhxvBOHh9+aSN0f5y8nbxNy+4hSQQEYdhnQO5FARU/fySULE1PVFhkiCYAmnSqHHIraeQlHD2+edp2PDRIfMOfYF//ZP53e+lZGvs0gy9nloA/h7fztQFT/VW29MuVEKdJkqoakBr3EpNjzpRp9z3iO4e/65e/GGWhpOok5yvYQb0IQbuQVb6GvgBfW/77IsQSMq1czL11sos/ZCZr0TrB4QNsdywCMg+061/F2fka+qKDH21MFlv/DpN/T/4TM+Om6jbX+qpm5/orkNOOkvV6+2iMuVrqGKl6qpYtVoI6DVr304tO+yp5m33VMs2e6pLlyMM2JeZpH9EY8aN0i+//apFiYutc6TlScl6dfRIA9jhqlEX4WHANyYoV6myjjjsSA3sM0AZqWlWMQbcS+nJGciZgN9oGNJZ0VpMU2FuuibCp+c4L1QIC58i8QZnNsiZ0MmPR8nPCa7LhnNjPzdH0edBfr/CyP9WlPsWlXh27P185oPkIjQ/YCN/Ac3vmhxwM5/vlhOF30zhIiwQHFgP0X6B+O5tjn0uv6N/pyCEWPOA+8CMEDECxpb7Rq2y4tAWQR+VxBAMgvagI2pWrxHy6wF9+7btdM899+n7734OQTQCeHPNtP8sNVdvWRm1JE0TzZcfOi1Rdw6eoF1Pu0QJzc2cZu69eVu1OPY0Xfz0S3rmsx/Uc8o8DViQpldNYAwyYA8w0Pc3kPc216hXkvSKCTo++yav14ClZgUk2r2TVgfQ4z0VTNuZ5M7JUWpWjmYtSdKkH6Zo4NuT9Oigkbr+8Wd0mmn2g8+/VI33PVDN9uqsWs3bG0gZQAJcBHLMP2MKy0z9hBp2vGpdA3A1JVSoqSq1G6rNrrvp0COPMMvmDj374kuaMW2mcjIytSRxkW655Sa1am/uCto++Pdl1bzpLvrnjbdq9tTpxmkMNoz+/0FUEoqOEcyA+0FwkSWmTNGhiWBStBTLQwk6YpoSZ2D+2JmP+2AhwGxR8r/HHuc358MTuAZoNCdn4KKQ15+6E1MgWMbiJZSLJzdBsfeLrU9piHtH70dd0PxE+pnrZ/oV4ekLcbx45iJKjwQf+pxkp2gWn5P3Cc/x71GK/t0L9yFjEPeBZzHNSvoxFIvNolKRNT0PoCBlmJ5A6hCxZ418xbLldPCBXfTSCy9r8cKlQbviQ0/NWa93UzM1Zrn58cvSNWjGYnV/9wudcN3dqtHhAAN8fdPwbdTkqJN0wRMv6onJ36r/nGSNTluvoaukHua/Y8L3MuCj3V8xRAP8fva9v30OSlqnYWYNvL48Sx+l5WqKqXcSTJOM+eaa1p02f55ef/8D9Rg6Qjc99KhOuuJa7W3uROP9D1a5Fgby2o0MzDaICJ5qjVS2qplxFeqYJAf4NpjlK6gC2Vn77q9Ox56s3Q4/Xk06HagKDVvYdTVUrkYt7XvQITrulFM0YMAgLUtO0arlyzRi+FB12nsvVa5uYKto2rWMmYYVKuvUE07ROxPe0PocqyhRxvUlG7RYijIr8RaixUxHse1XNIeeaUXSbFltRrAuyoCQM10sOXPFnsc9mI5i7psoOAEu0nmj9SkOATB4ixkPti4nrZXZh8KmDKHC6loa4n7RutNujgF+ND8mP6v8sJRc61MAJEE+0n9ZV+CJONF+4778jvZ3LHHcz/VCfzJmaHqmU1kWTb/zDH9OcalIoKcibk4QVKBxVADJw0qzOqYJzz/7HL352htavSojBM6WWt/9YtbH+GWrNHTJcvPNk/TEpK90zj2Pq2nno5RQdxcDTgPV73qsOl91ix5881P1n75EI5fn6TWTGq+k5Bb47TbevUyCAHxM+4H2OcTKYNPyQ00ovLEyP2T3vbNwmSbPW6JPZszRuEkf6ZnevXXvfx7VKeddoM7HHK+Ge+6jhIbNTWuboKmyYYbASqVWu6nGrp3U+oAj1bbz4WrYooMBtZbKlAWs5dWswx46/eq/655e/XXvwJG66onnddRVN2jPY0/RbgcfqX2POFqnnXu++vUfbKA36W6m++9TftUVV12pRs2aBtO+XPmKJiDLau8OndT9kce03CwPD+aVlpyB+ATwPoXEyjP34ZmCIjCFnwkwmW1whiyMuBd/Z8yh2HNhRjQwGhBwsiyW4BarFMlLQPM7026JHAhcwwo//GSsE6a7yIpjhRsWSZSKct+iUvRetNv7E4r+DeuJDEc0LTkO9KdP7RHdx6xHIxMf4VwIvHA/7kM7va3RZ0TJn885Xkj6YVYBIYOiJdHHhSB/LwkVWdPzAJITiNqzNNKztypUKKfWrVronjvu1Mzps5RvSmy18cjMrLX6KitPryavDKB/+ssfddmzvdT4gGPMjzdNWauRKpqffMhVt+vOEe9rwNSlGr44Q0OW5WuwafkB5mb2MfOeAuh7L19nJr4Bffl6DUnO06BFZjnMWqZ+38/UoC9/0YtvTNK/evTTpbfeZf75qWq/T2e16bSvKjVsZs8zjV7dwF7TPpu2V51OXdXm8JPU8eRzdMx1t+qCBx/XHX2G6har32FnXqIytRqH3P6EclVUv+2uuvju+9Tvi+80etZC9flpmh596yPdN3ScHh40Si+PHqsJkybrl9+nKy3NKmvjuXTR4rDqa7cOu4d99QgClTPQN6hVT1decKm+/fxLrc8zQBU+9sUiZyAAikAme44VZACHMaIwDUV2GQzpfiZUmKZwpuMcTHc/5uR/R9sQIOTebnqSzILA8aCgP2dz5M/CjCWfgIQm16BYKQgVzH3I68FntE7bmqgjgEXjk6HH6kWm6YK1a/49xX1u3qdA4I1gt4Oewj0o/ttLlPyYn0shxwO3gv4gGEsA3YVqrDAuKhXLp2f6h00GSFbAR0TCVaxYXnvv3UnPPN3dNE12QUKMCaAfU9P10SoD5oJEK0v12KQvdeR1d6hyu/0MUBvm4bsepQdf+0QvfDtXo5fkaSTz74nZ6mPAB/T9DUO9AXxKnvqaGT9waaaGLcnU0AWrNMCsgn4/zNKjY97WDU+9rEPPv0It9jtItVvvrrJ1GtozqoY8/YS6BuBGLVS+TceQ+NPxjEt06q0P6Ppn++neIePV65uf1ffn6RoxfYF6fPK9zrnjQVVotmuBJVDWzPwadXS6+eJ9vvlJry9bo1dT1mj4opV6fclqfbEqR3NzTRja2GVam8MQWgfkZWXrjQkTQxqrC8eKZcxVSCinEw4/Rq+NHhf8/60BeifGie2uWGfuQGSMYEbqQSQZE9UZbVOAdnLm29QxrAWEP8ksHrkGsAAXwQIVhylhZFbqOegRlCS7IKx8G+6SMvnmKNo+Jwecf48S/UbcgQQmVlQSyPMAKQXgE3gjk3FT5P0Y2x6O+9+8sK8g6cLcm/0asdQKi7sUh4ps3kO+qIBKBNPeCnvCH3/iCfryq2+0Ii1d6fnrQw78TxnZ+mh1pkYkLlf376boby8PUUIbA3ydNirXco+QanvZsy+r75SlGjQ/RyPMTx9ipW/yWvVKzg3A70fyTnKW+e6ZGpmcoVELV+qZT37U3aPe1On/elx7n3uF6u93qGp2sPvWN9Md37yCAZ1VeE1aqHqHvbXrSafroMuv03mPPKs7hr6u7pO+Vf8f52j0zGSNXZCq8SuzzKVYY9ZIqgbPWqp+X/+uPU48uyCoV7aaytVtol2POUk39B+m0UlpmmDKcbDVaWjiGr27MldTzKIhN4BwUx7jYF213gTB/LnzwvpuT/SokFBeNavUUJN6jfTM408rccHCcO6WyBkhSrHHYAJ8P/xhdr3BFAQ0CBw2hMDnRjM7oxSHYfw5fg2/YVZMbqwZ5vlhfNw9otaYwLgZxSVMVvIJmBViZRqCiy2q2OfO4w+x7S4N4xeVHJjebojnkpZLwI6ELE+9RcgiqJi7J9EJita3sDYURv4cLCb2IgBvjCcLhtzqiRUYxaFigR5NT4DIFxWQcda8xS666tprNGX6DKXn5CvdTiVu+W1apt5PzQjZcq/8NldXvjBU5XY/RAkt9lGro7vpnEef1KMffxnm4Aca2AcEwK8PoO9ngAfsfRelabCVATMT9dwXv5hV8F5YU7/PuZep9j6HqFzrPQum/DDf8dUb7hKA3uKI43Xw5deo293/1tPvf6rnP/9JvX6ZrYEzkzRkfqqGmWswamm2Ri0zNyElQ4NWZpslka4hi1er709zdMzVt6sMLkjleiZAaqmpCZaLzJroO2NxWBzUx6yNPnaPccuyw4zBYut/2DzoTroqb71SkpL1wnPPBzObAcOnx8Rv0bi5brj2H1pURNBHqTAGYvDxrzEpWdUI+Hge/iYMSPCJhI6oqR0F8JZoU8DCbyXIhgVB1hlmPVYgWt7vW5T7O3Eu2h6tjhZlPpoZCHjOp6Y4J3rP4ty/pBR9RhRouEbMl7NxBtqdPvcZEhJ4CELGavui1tefQ9tJEfbYjIOeMdmmoPeO5kFMFfBSSIIsQbIZ6Fu0aqnHnnpS883kybA2mUUeUl6/XZOlyWtyDPSrNGzeCt0x/D0dePmdOuBi86H/87KeMh95xPJ0DceMNxO+Z9JavbQ4S70MjIOW52pYipnyC1foGfPXH544SWfd96h2PfEslW21hwHczHfSdSvXMQ1vPnurXVVlny5q3+0cnXj7fbqh71C99MWPGr1wmQbOSdIgsxCGmqUwZHmOBhrQByxbq4HL12ngyvXqtzJffex5g8wnGZCYpVd+mq9ruw9Qlab2nIoG+jI1Vav13jr1tgf1wpdTTVDkh6zAXotMaKTk6nNr4xzT7LQ7eMiMq+EkMz0jbK+F5sUNqlCuIJjX2DT98Ucdp8mTPjJm3rLP64zi4xAlP4bZToAJDeM+JsIG3x7Ax06lFQf0sRS9Ho1OwdcFsO4yFJdi6wGguC+MHRU6nBc9tyT1Lw0hOGPr8/HHH4cAJgIW0IMLND9WyrBhwzbWMbauhdU9ei4Fa4L7+DoJpsm3hqtT5EAeDcb8YouhjSaraZN99ttXE99+S6tskDBxYX7myr9aZZo+zYCxJE2vJmfr5e/mG3i/0YPjv9Bzn081Db9Ko+0CfHei9C8l5uqVpBwNW7VOo1Pz1M+0+7Of/xim8g658h+quNs+SqhpWh2gVzQzvn5LlWmxq/a7+Eodc8e9uvzF3nrwrQ/V+6cZZqYvCRtxjEk1FwHtnZypASvMejC/o++GKcBXzBwJZcU6DbY69E5Zr8EpazV0TpoeHvOh9j3hXJWt1jSAvlqTXXXE5TfriXe+0piUfA2z6/ubkEIATF6VZSZ+Xtijn/aHYTO+YHutpYlLQmCHqTIAX76s+fUG/sYNm+iJJ55SYmJBZtXmqDCAOlNAmMUE7zDjfT0840Lwh+md6Dw396JE71VUgsn8uVtiuGj9ikKcW1i9YgEWe07079uDaDeCzfvCj7FTksdvHPi4KGTRIRALo8LqHm0f33GhmHZlPHGlya/wF2jE9lVxqMigR1swv8tSQZc8+I6nnna6ps6cFebm0SdMVgB6VrK9tTJDw5PTzWfO1/AlVhbkGagyNDIx38z5XPVLXRtAR3Zd/+UGuNT1Gmkmc68p83T76AlhmW2ZXfdWQtN25q8b2KthwrcO0ffj/naTLnv0Wf1rwnt6/LPv9NKvszRo0XKNMm0+dJkJD7vvELtfn+Rc9TLt3tPu3yOU9VYM7ICfICFJPwb4gfa97xID8pI89fpqhk665naVq9M8BB0r1GuhvU+5UPcMfUOjFiBMZC7Jeg1LztF7KzP13eos2W1C2wMcGI9165WXkxvypTvu1cnMvwqqVIkdYyuoSRPTwuddoCm///cNKJuiKBM4eVQYYoqOLC1SP2E4zEzcL/Lp3f+LpeIwjD83yuiFUWF/K85znGKfw/M55oCI/q0k9y8uxT6D314XJ7Qv8QzMcA9qovnZr89z5WNpU8f8OM8gGw/rDcADfPZGYPbMycemuFRk0OPDuUahAjSMTK/LLr9Si81/BfToFBgf/fVNeq7eWpWtIcvSC1bUGcgGmzQYukSmyc0vNpT0SEK756nP0hz1X5JloF2j3r/N023DX9f+l11jgN+rYKqtqpnZtZuYv76v9jvn0gD2Z9/7VL2/m6J+sxM1eKn5/ma+D11hZvqKteq9NM/uawIl2Ux2tLsVAA7QsSqChgfsVlitx9w/OQD9kqSRyeuDGX/lf54vmOJD2FSpq8b7HaHLH++pvr8u0WgTHGj6/onpmrhstb5MXaNFG9ruoPe8mx9++lGnn3mG9VklYwj8PjK4yqpDp730Ys9XiuzWRwfYGQPwkyaKlnHrixVzmIT42r4XXizFMu3mCMA5Ra+Jfud+UbDyWdT7xxLXba5+pbl3SWhLz6PdBLixqgAoY4DgZfYECw8Tf0uBzcLuTx+QkcjCHYQ5woTlu8RusDa4pihTooVRkUFPw3gZBBsyuAlD1Pbmm29VyrLUwOz49DD+IqvL1+brTjCffeDyTPVaaea6Aa5H4joDiwHfvvczoPU0zdrXNCur53pPT9Gdr0/Wifc8qhYnnaVKROSbtAw+e5V2e2rv087VJQ8/pSfe+kA9vv1Vw+YY+JLXGPDSNGhlbthya4Bpdxbk9EwqsB54ZtDkGwq5+v2tAHJSewkeDrD69E5cr+Gr7bidMyAxR32nLtYtfYeYkOlkwDeTuUK1MONwwi3/0rOf/a6RKQVxgX4G+rFLV+ij5amaZ23GkPNhcNDzdtw77rlbjU27h9TeDaVpi5a68dZ/Fhn0sYzBgKMJsLzYl81NQJJECKxhATgQHUR+j+j3LVH0PL8uFuSxtDnQxhL3Kez8oty7qM/YGsSzYp9HXZzIgeBNPihCn6ZluhQLgIAc7XTa3H2cOIcYGmnTCBHuxyIfNv7EZeDvhV1XFCoS6HkAUVkWaLDqyINFNOruO+9RSsryAHrX9PON87/LXKs3UrM02LRvr5Vr9fKy9Xp+sYESMJrGJ3jXx8BJVl3/6Ul64M3Pddq/u6vF8WcpoW1HA/suIYGn8UFH69Arr9cNrwzSU5O+0MgFyzVs6SoNBOwpGQbeHA0xh5qgXF/TwGj3l82VIG23D8A2LQ/4+9l3nh3AvgHwaPb+5PEvNMFj5yEMesxNVe8Zibp3zEQ1OfRIq8eGRJ0GLXToFTfqiUnfarQJGNKEB5obMWrxSk1alqaZJCVZ24kz/ze1vmCThae6P60mTZurStXqISUX0Ddo0lQnndZti6B3BvFPwO5TdOw+y3JZnzLCvCTYQ966Z4VFye8BRb9viTg39nz/XZz7FIf8voDFA2gci4KnpExfHOKZ0RKl6DH8b6ZofTdcCtqeuXUCrT4DAXm9/dpom5z4G2nJgN7zAJgeZFaDdQ6xdSkOFQn0dDrMyzwwyzGpBP58uzZt9cZrr2venLnKzM1TWl5+yLmfmbNOP+RI761Zp+HmuwMmQN7LwPaKmc+Y3r1T1mqwacynzZS+59V3ddQt96t658PNZ28VfPg6Bx6hTuderttHTNCD73+pF35boAFLMzV4ZUHUHa1OcC4E6JZlh+Phu/ntfa30Ss4PQmAg4LQC+EOxemBlIBBwMXovzlG/pbl2jPX52SYMsjVmRY4efedjdb30GhM8DZRQ3kBfuZZ2OeQ43TroVY1blqlXzWLjXDYFeTNxjb4yAbDA2hw18QF9hml6cvHRwAxceV7IYaCvVLm69t2v8/9LMYV8QGM/GQeYB0nPTreYkz6dgxAm9RZNQDbY1gQE94Ixo/VxEPrv2E/+7rMGrORjjz00FwwbzaWPZXi/B7Q127AtyfuCgB6BNwc91pe/poq/e1sLa1fsMcYa3oBvfP0EAoWpwMJ4pji0RdC7hMJEwWwkYwoTBmZji+uvPvs8LDTJzjffxipqeNJ0A/23BoC3V+WbVs7WIAMXZjVR856m8V9JyQ9ZdoNSsvTwpK90/F0PqdahRysBH77NHmFN/ZHX3RbW2z//3Qy9NH1p8Pv7YR0gPOw+wVw3UDOnz9ZbTLkNNIsCgTDIzulr9+dvTAH2TsoJAT2OkQfgpV9ynvon54SgItf3tfoMNAEyLjVP3T/+TifedJdp+uYKa+wr1lazA4/Sjb2HaczS1RptfNt3aUaY75+4JF2fLs/VbOsqovgbZTrBPNO4r48do8MOOTQMXNlyBSv4ylfgrTp769tvv/8DWKKfkDMUBCMwNUYElx1cyIZzLcD6eObpmWEBbFsLMP58L7EUrV/0XAhwkzaKvwuzkmhCxh7JQrSB9kCcX5i221HI28u6g27dum0cE8x8koxYeYcg9vOifeQUO170DcKSGJpbD+RgsG0XGYGl6a8i+/RUGu1CWiTTQfiSDCQvf1i7dr1yrBHLTdsn2vcpGbn6wTj/g/T1IW12QNC6BWAliEYkvedyA2VKpu5+9xPtfe0NSti/ixL26azGx52scx95Ui98+o36/jJLY5IzNWrFWg0zNKGlg3ZeaoAlAGf3Grphnr8//rtZD4CZHP1+9slzydUfZv73UBM0lGFJZpqbZh/KPvtLs8KWXYOS7ftKZhTYe99AvHqd+v04Q1c/2UNlmrRVQhW0fV3V6XSI/vbsAA03F2AEgmdxlkbYfcYvWaPJ5mawvbdVI8zXhyE10K/Py9WH772rk088KUh+gnkB9OUqq0OHPcO7/XzA/TOWISCYgOk3EkJIXmEZJ0yFhme5J4teCLT6ks7CGKsk5EEjaEuM6sRxAM+CHN5Hh1lKVhkAYJtptuYim4956Nh78BuG3tS9/8qEIOPV2Z6azPgwjYowJigXJW+f92lse+kD1lKQ6syqPkCPO8125VhM0dyL4tIWQR+tDPnWLABgGgoJzlTFmjW8KNrMWgP7StP2i/LW6ufV6fohe70+NhN4FCZ5Uq76p6wLJnUAvmljps96Gegf/OJ7nduzl/a89h/a7/obdUXPPur+2dchUDfWfHZefjHczsdFYGqtl/nrvZnySy5YbcexAnehIJsPq2K4IW/gEgPzokyNXbE+vBprxKIMDZ69XH1+T1SPn+bo+W+n65mvp6rHlEV67rf56jE9ST1nLlXfWUv1mpnso39fqAeHjFXZxu03gr5K2/10wcMva+DUFPPrTdAk5oWlvWOTVuuDFZnhTbt2OGTnocPCgK5fq6+/+iJMt2AhAXhKxfJVtPuuHcKOs7EMEP2MFgCIn8eLKZiig7FgBsx89sDDBfNrtxZFQQ9RVxjSi1OUT/juacFYID7bQ6HOBIBZ8Ydm3Fx9t3ZbtjUR9+L10B58o9B2ZlMQgLH9CPmxaP9B/EaAo1hZ0ce96DvWvZCpWNJltVCRNT2VoxLM/QJ8opXffPOd5i9I1NKUVM1LXqYlmZmauyZT3y9bqR+z8vRtnjRxufm95u8OMvMac7uPgZDps57mQ79iWnVQyho99dMUPfjZN3rqu1/U8/fZGrwAUGVpZKppbtOk+OcE6EI8YKlp4SQDdYo02IRIn6UFU36Dlq3T8OXrwnTaqwZydszt8eM8Pf7+d3rkzc90z+h3dGPfUbrimT469+FndNq9j+oEcyvOe7qXzny6py7pMUTX9B2tmweN1xNvfqLn3vhId788UOUatVFCdTOvytdWhZYdddbdT6nfT4vNr5dGWhmSlK5RSWl6Z2WGfspeFzbfxK/HxA9vx1mXb332S0inDLvRJhSk5FarWFW7td89+IFQLGigqPkLEZyDeVja6eu5YQbe4kIKrLtiWxMs1CV6Pwf7pp7B+fwdXmGxCNoJZqUwu+Dg57VTBKWcYtsf/f1XJu8HPmk3bcIS9gg+hW22UZRR7ezt4zovUeI3sRty+Fli7OPNNCDP2FTST1GoSKCHmXyBAX49mxxgdpx++pl6uvvz6tFngAaPGafvZszU7+bff7c0Rb9n54uEwU/W5OuN1IItrQjEBR/cNHEv096smcfEH5GabWZ4hgYaeIbY78HmV+MvFyTWrA0uAUHA3qbZicKH6Ltp+sFE/00oDDUXYkRiesirH2T+f0/T4g+89mEAedcrbtT+F16rDqddqBZHnqy6nQ9T5T32Vzn2w2uzh8rtZW7FngeqRpfj1PSoM9TyuLPU5cJrdOLVt+rwcy5X2XrNlFDLQG9+PavvTrnxX+r52VSNTcwJO/MONf9+RHKq3l6xWt9l5YaXcjIcwI+3+a5dm2e+2dyQJkvSTLky5QPwq1eqpvZtd9Wro0bbmf8NaDHYzhBoWSf+jibHP/SUT7QIGp/8ejY3gaKAiWWk0lJhzLkp4jwW3/gGk55gQr35jdmPlbO5+23t+m8L8jp6nyN8aZvPqDhQGbfo+w/8Oj431U60OQlx5PbTZ24lsa8hef3OM8WlLYIeP5IlmwRg2FGVaCS+GbnddevWD+9773LwEbr0muv05kefaEZysqavWBneOU+M8Sfj/o9X52mcgYMFLQCfOXUi7KykI+pOMG2QmfF8D0LBtHXIx8cVsE8vLLNl+i1sn2U+/SDzqVl51//Xmeo+6Uv9c/BoXfbUSzrln/dp7zMvVrNDj1NC83ZKaNK6ICDHe/Cq1bVSy7R3bbGv/sZNNWo1VUJddsRppISGbVSrbUdVbmbXVatp59r5FasooV4THXbRVXpsvLk4vy7QkNkpGrEk1VyY5Xp3Wap+SM/RIhs/1/SAfr2Z9ytWLAurE5leIxUXTV+1QhW13KWVXunRk27+f5IfYlDRDmh88iTYpoy97QrchIKXU8Bg7JTq10cZwY+VhqIMGb23xxeI9eC/I6D4u5/PJ4EoXnsd3WKKOiMI2H4KvvJzvdBe7lmSlXp/JnlfkzvBIhySpLzNvCuPAJy/RyDaXv/tFP1OfxILYB29J1/Rf6R2sztSSU38LYKezRV5bzYphYCdh2K6IHlq1qyt+vUaq2273XVKt7M1/u13NW/ZMi1OTw9RfMOmZltffLc6R28tWaFRzLEvXhWCZYCcBJe+Sflhjpw98F4m/x4T3i7saaV3mtTDbkTabI+UdSFQ19+0K67CIDPphy9arZc+/Vp3Dxqik265Tbsceawq79ZRZVuaH97AQM52WJUMtJWtw1h2W90GwoCbYGAu33b3cG7lPfdVQsvdVa51RyU03c3AbUKgkpU6jU0A2LX4o5hW7JVXpbr2O+l03dNnsHp8/I1e+W6qRi9errFLUvRe0kr9tDpTS6y97tNj3hs7GAOnh8QmZj7Kb9hFhxz8Zk2ah4g2FGWE6MADLgAFw2AiMgaerIE24eUULD+FHHhbk9zFgPju9QPwbBkNb7AJJPEePxcAYB3i17PrC5aJAwDmJcmE7bWcos8gbXXMmDFBY0b74a9O0X4B4ADdk9gQemwG4kKO/uFcFxTRdsa2mfuRcelTs6TAs9qObblYclwSSnAmiUppCCkCo7EpgwcmGDCkDtsCscSWQM2jjz5upsvTGjBwqD798iulmEYiSQfvhdnYX5Yk6+3fpunJ8W/qhl6D1fP76Xo1KUNv2kmjDNRDTHv3NVO9D8kymP2A3Y4xNcfe9oPIlDNXgFz619KlNw1RvAfvsU9+0k1Dx+jQy69U2yMPV7mWrUxrG0j9VVjVWJTTTDU77KOEXdqrYZcjTftfpCOuvjFYAuf++zHdbOC9qfeQkPhz4wsDdcG9T+qka+/QPqdeoFod91Ojffcz66CGgd6AX84GsErVsEFH3Q5764grrtftA0ep34+/a+CvU/XFynRNXWPaL2dtyEykJ0N/2j/8L5j84EMPES/BwEzj3fZsLIqv72ByimpojvN33pSLBoGZogBilRf+cxTs/j16n9IQdfD6OaABJS4egCbGwLQcjOivqGbjTdYEEL12X55PfpOwgmlP3ZmCRKvDwAhG9oPDnGV+moAVf+N50f7x+kSP/dnkQMaEp20oSV8AxXgj6LDIiktE8OEdn6tH4WLpkX4NlaQfNmp6LowyCSYWc74EXAhAkV2EpkKzEEzCAmCKImNNuia9/4H69Oob8r1/nzFdi1OSlJmfq0UrlqvnsGE6/4Ybte9pZ6rz+Zfq5r7DAvBHLlylVw3IRNuZhmMePyTw2CeFBJreS7LDfPgwExKshe/500w9OflL3ThguI698351Ou8SVTWTOaGBdS5zo5XNBK9ZVxVat1ezrodrj1PO0uUPd9ffn+2tB0a/pafe/UIvffaLBv48V/2s9LXSf+qCsNa+7/ez1Puraepv5Zm3PtE13XvowvsfUOtDDla5hqb5y5t2Ne2cUM60fvX6qr7bPtr1lHN02n0PhZTdkd/+rB/NzDe5tCFyb/+Rmmf/WGZL3xx59FEB9BQkdvWq1YIG8Ow57//oJwXGp78BmZvKMBJLnBkDDxD54Pv1xWWGzZHfCwFEgg3amM0g0WbUAzOeaVz4hU/2usMVdFfEC3zkr4ZmmTa75VDgK7I9iXtgRRIA5PhfHfSFKUvcMAQbY+VWGdhhmy3+HjvOmyNwyLv0mAXhPtyP/sGvZ9xL0hcJsRd4Rcj6odP91b0MBkv7MOU8chgY0hj6xx++08UXXmTSeT917/6Upkz9TXPnz9E7k97V/gd1UQ0b5PING6t8q3bqcsnVevb9L/TaojS9vrIgKw/fnnl2lr72WbE2ZNuR4tpv/gr1/HWOnvn0W/1r1ARd9dQLOubaf6jV0ccqoVWbAn+7ijEVb7WtXV+Vdu2gNsecrKOuvVl/695T942cqCG/zteQqYs1bPYKDZuXpqEL0zXSXIpXzaIYbYJlROq6YHGMsee/au4G03sjF6Sq75R56v3tT7r0oUfUcC9et2WStlxV63h7VnmzInADajVV5b26qP3JZ+nWZ1/WVzPnhzl6sF6g6gsKL8JgivPkU09RuQqm9Qz0bDNWpUqlsFTZF8Z44M41Nb8BGVNBaFA0anSFI9rQt6aC/LriMMCWKJYxuTfxBVKAPREF4LvLx28+qSeCycGOlvffrvEJSuHfE53mk79xDhYM++2RXQhj0w/RNjnPbs12lpSi9aCvKExlM5+OsvQoPoLxkUceCQFxP9/He3PE+JPNiECknwE9/UdgGL7x5/s9i0L/D/QwDpIVE42AAYBHYjENQe49iQFeWa7LSF+t18aPVaeOe5jfUU9nn32mvvzycxMaszR27KsFb3vBLzZflv3m2h55op576+OQzDLZFNSIlaTOFgT2+q5gYU6+Bq7IVe95y/T0V7+q20NP6fh/3h2shCZdzTxubverVr0A6LyCqlV71divqzqccb5OvPUeXf1CHz08cbIG/DpP45MyNW5ZroE8V8OW5mq0CZVhuA7meDPtx1babLHdz6yJYG2krAsJPYOTs8OS4GHzluoRc0uadTlEZarWU7mKtVS5aiM12GUPVWvRSQn1WpjWN1+rXlN1Pu0sjXz7Q61ck6PcHAMKWKFY1/L+OkzZc847VxUrm0VioC9XjvfIVQz7rHniRmGgp5ANCQBIcoGJAAY+3vXXX/+HHHu/zqk4jLApigU9RJ2oM1NyaGwyxXwunrpR3KTnOFobIPsxivu7AJ/zMV9haH6z+Qc78RChhhdj3R/n2a3Rvq1J3v++zRXxFxfStI11EQhvFw60qyhEvAT3gLGnD+k7sIlwKUlfbDTvvcJUhmgs+5kThOEhSCm+Y9bjg3Hufxu4TO9/8LYxZGeVr1hOB5k5PGnyB8YU00y7TQzHq1QzDcmS0pp11O6wo/XCxA/09ep1+tpuMd6MhoErTdsb0Ci9k7L00pwU3fHOZzrz6ZeV0GFfJbTZzYBl5o1r9Tr1VGuvfdXmhNN0xA136KxHntPNQ18P++a/8sucsHHm2JXrNMH8/6EmTEISzxITJEvXhRdlvLx4bYghsI9+T+IIpulfCktx129I7zULJDVP/ecn6YHX31J1ts8uX9PaV09167fVAYefpoPOuEJNDz7O2tTINH591d69o+576jnNXbS0YIVdBPQUgl7soIp5G5igrJm6pukBDQtkIO9T/4QpGA+m48jCIwkHTQpIyHBjO2TMX4hBjwXo1gRFlLkoaGD8TQJ5zChgfsLY1M2ZHI0P2NF4rATkO0Igaq24CYxy4TcZe7SLe9MP3gfelmgd/mpEfakr1hfxLiyY6NQdJj79hXtUVKKdWFZkMdI/LjjJ/EMxl6Q/NgbyogyDZmEdMEkFPAAf7PLLLw9R4j9Kp3WmadZo8scfqNsZp6l8lQrac++9NGrsGCUvSzFm/lGndzs15A6XweeuVkNN9zpAjw8fp69X5uhz0/RjU0mnLYjkkzvPevgec1foxnHv6ZA7HjSwNwsBuYQGTZXQuJkq7tYxbHZ5+j0PhG2xHnn3Sz337XQNmLNcI9hA06wGTHbuE3LtDcC4DYCZQCFxgwBsKz1N2/dIMaDb30MugAkCwM86/35Wv0FLVur20a8pofXuZtpXV5kKdVS7YTuddP7VuuHJV3Tuff9R6+NPVdODDlWHo47Toy+/ojkLl9hIha4x1Np/fLfy848/hRRm93HLm6avUb1qWCSDvx4dNGdyHxsCdSNGjAi7sQAYJD2vWeI6NKGT36O4TLAl2tT9qB/CjKAu5qczpWtu/HzWg7NnnKfhokCwUggAErADCN4mBBrXkKnncYoo4CGvS2H1+TOosLowJuS0kDTlKbm0D4HPUlvM9aIS92YGhzgJ/YOAhH8I6hJz8+fH1mFzFEAfBTyEr0A0FnOSCiOlyR8maORzgzyEbLO8teYK/PSNLr7yUtWsX1cdjDEHDh+q7NycIK0ffuiBMKfP22ISKlZWDQPQnS/11XtzkjXJzHkSXMimG7BsfZi+65OYqxdnLNONr32kI+97UrucdqE6nn+FDvvbjTrxlrt01oOP6breQ/X4h1+p3/RFGrggLexgM3yF3Wd5wbp6XAQSf55ZtFr9zfrti/m+woSLafThZlmEef4Nabw8kxx+Aoq8QmuQnTMwbb0GrMzSoKUr9K8Jb6mCaXGm/MpWqaeaDVvrjKtu1IvvfKyn356kO/oP1PXdn9XdL/TQ+998r+SVq61vrIMc9BuCeb/+/Evw34na82agSuXKqnatGmGenaWXPmjO5AX9WzAuntIKaAAWZh4aEeFc2GBznQuMrUGx/OHEs/FR0czEFwAvTIk2Z8YHYcDsBPsE4qOzSIjZIAJ8gPu5554LzOzakOvZKML3gXN3x8n7xctfhbwu/km9SaQitwXsEL9wsPp25FhoRRkj7sn9cAPpHxeqBICxEL0v/NlFoWDeu/b2T3wS9u4mAolmQoqT7820ERLYGRPKz8/V79Om6ILLLlHzNq20j5nzA4YM1pr0TC1bsTwMLJq+UvDDq4aEl1uf6aUJMxI1MTFTQxKzQ6INIOxr5veA5HVhr7pnf1msG8d+qMt6j9StI97Uf979Qs989pOe/3aqek5ZoCFL1micAXqEae7hpqnZ6LKXmfCk7PYz4Pax8qJZDyT4hCQfu28vpgAN4KTvhl18DOh9FmZr4FKzMJLY4GNtED7EGPqkZGhocpoen/yZau5/kMrUbaqKtZuoSqNWOunqm9Tv61808JfpGv7r73pr1mx9OGO2Zi5fFbYAp2fW2WfBl4JPtDLTURtBX6Gi6tauEywo3+K5oD8L/FfvYz7x6ehHLC4Az+Bj6hPZhWIHnGv8PluDoveJ3pdP3LvXXx+vgw/uqjr1aodAJS/4QIuH2RwDMMC+6IILw7oBtBx7x8G0/CY4TBAPZqZtuJG+i6y7Lk48L1r+KuS4caJuWGdMOYIdBCCAB/h8JzaGIEeDb4m4F/dngw6sJ/qI+2AxkBJfkv5IiEpxlzwkBLBwANOMyiKteL0Q/vwfiOfY5fPmzNedd9+lWvVrq0271nqpx8t2j2RlZGZryNDhYd14WGhCVlv5ajrhmlv05sI0jeKd8+ZHhx1sosWAN9CAT2Gr6X4mHAaar09EvyCpx/xtUniT8/5wXX8rmO0E5cLafQM7nxSOhU00DOwDrQwmf98+SQqikMvvW3HzfJ4xjA09Tch0POviME1HKm6lFrvq7DsfDIHCVxenatScBZqUtExTM7JlMkNr1pq/m7+24MW0oXvMTF+/Th988KFOOO5EA3o9Az1R2ArB1OUtMYUxN+TjgTYlow+/GTMRSY+pH3vd9qd1ysxI008/f6fzLzxHtRvUKnhpZ4Uy6rDn7nrggfvDVt9XX/U33fD36zTt96n6/PMvw25L//rXv/XWW++EGQlMYBiZT6Ylo+mqOyKBKQKsAJ+9Cj3PhXEDT+S6YP7HCotYcj4gBsCiHdwD7sMn/ICvXxL6A+j9O0wGM+KHUUnMSjR/lMlChaiT8WXivEXh/Xb1G9ZTixbNQ57xvHkLtHpNlkaOelX7d+kaQF82JM7U1JGXXRdeNjHafGfeSefA+wOA0bhW0Ny80opcfUrBslwTBmauh2m+DddGAe+gjxYHPef99xkFx1j2y++NoA/fCza+7PndDHU857IN6bq1VbltR51z32MaMDUxJAm9umCJJi9bqelZ+eHd+FHQF8hEk9Tr1gYGP+aoY1WDaUYy8ipUUvNdWgY3yrWnD3L0k4KFxSo6X1cN82Deu9/755FZFGuzNWPmFP3tmisKQM/ruSsmqFGThqbhTwt7/x+w73466IAD9dh/HtVdd9ytrl0PVrduZ+iZZ54LL4QkYo8wwwwmch+dhtxRyMcM4jtjAyjJouSFID5ugBbtz9x7UYU2oMcNxOqmn3CHwGN0M5LiUJiyc/Lv+CP4Xr6wg0qTLfX/gkacbqBPSUwOiz6at+DFgw3Ctb/99rtWpWVo/GsTdPDh7CayAfQGnP26XahXvpyqsQTvWBJroKMA8gItW1D43YdXXbEyDxPdznnZfoeUXOMLdsJhuW643sAbdsPZUDYCHm0fWzYIEArnusD5f6BPytMr389Wp3OvLIjSV6sb3qZzySPdNXRWkl7l9duLl+kL8+Nn56yVyaZg3ufmrTUwGGDpJyuAfty413T4oUeEt9eGpbUVK6t1m3bBH3ZyoRsVxPQ5Pj+gRxPCPIwJPnSsz7v9iXrmK3HpQt153x2q26hOAeitVKhUPiiAvTvtpSaNGodtv1u1aB1Sj5s0aqq2rdtpr457B9MekxVAMLWH7+oaLMqbf3WirtH6Isgx30krJm2WWQrX9LhpZNSR8VqUNmL54PZ47APhyJRmYVuiFYWCTx/7YHKE2e+LihKUwZQkmygaeAjXcJmN++oVaWFThF13b6969eqE+WOW3aatztRbb7+rY044MTB6QvlKQdPvctCxun/M+6bpcwzcprUBqgE7LLSJAT5r5Mm5L1hauy7svANwicIH0Nt1oXCPTZT/gnxtKK+kGphDKXh2MPsjgI+Cvud3s0zTX2Ga3rRsrQaqv28Xka336vzlYRHRBymr9cPqbC00/PFOO1P4BtT8jaYbXcT3kcNH6ZCDDg0LblhlRw5+m7btQ2ab97/3L6D3Y2gMND3ulWt6tAVB1i2Zh9uD1q7PV0qqCf3ujwftXgbQk3Vo9axUqcLG2QpcGtpd0XgAwcfbfvhep1bt8HcEmc/R/z83cgcgxis6bhDjCbBxlZnBcNDj1oEvcl5irymMcLeJrxHL4XoS5tgwpaTuXaGgZwUPUVckC9FitApbHDlxfriGywz02WbGM8W39757mRSqHYJTn332hdIzcvTxJ5/p7PMvMGYw0Jcpb1qgmqq021tXPtu34EWUBvqgcTcB+kFodIC5oWDS+z53nO+g53tBKdDcXgquK/gb2X59VuSpV2pOQSE/gL9vBvQvfjVNu59xiWl507J1GoX03pt79teExFWamLRaX6ZlB9PeTg+r67JN0+fkFCTVeL8i8QcNGKyDOncNL7IEAIC+/a67h9cz+4yIa3iYxa/lb2yPhFB1y4ugF2PyZ5v31DDfTL2Va1L1cs+XwiwNQUqKA9l3CiKGwacvNqqQUDEsM67AQiY7lzYBerSi98eORIwX4+dj6ESwlVx8ZiwAPQVFShSfKTfGenPA528IB2Y7EPb0FdOc7GlRlEBgYbRF0BM0IMcajeS08Xw+rI25mVkmeUbqgC6dzTqoFeYnP/74U+XkrtOPv/6m62+6WdVqmGliTJBQvqrY/PLUux4J+fQEzGJB78Dnkz3s+yTlh6k1tsDqb0AkwDfEzPzBG853UBdc+0fQFxQ/pyDrD+D3WZlT8Gmgp8SeD+iH27Oe+ugntTn5PIU369RtrNaHHaU7+w7We8lr9M7SNP2SmR+0PHF0WDU3z8z77Dzl5xa8WZSCj/rKiz21z557bwR91aoFG2OyGYlLbO9X9/H5RHgQEMI3RFsADgYf6+vP9n2prbGsVmetCe5f65atTIObJWJlI/ABe9lyKluxksps2PO/AOym7VECG87DxAf0pK/GAuevTFHs+HhHifFjIRLJNS7c+MRSIx+BMd4c8Ln+hx9+CHkZXIe2Z30DOTMlFfobM/IgHkohxRLzg0EA+KTgopGcNlauYNRD/v2oUSN0YNf9A+gvOO98a9An4hXs8xcn6j9PPKkmzZoa4K3BpM7Wbapj/3GnBk5bIt4t9/+1tQN4vfqZ/z7YeJv0Wbbd6jdvlfrNSdXIxZkaZ8eZ3mMHXD8f0LqPv7EEbe9lw3kIAPvuoA8a359tf0OwjFiao0fe/VrNjz0juCUJ9Rpp1yOO0f0Dhuqj5as1aelKzcpdLzs1rCzE2M6z3zlZueHtNrzaikFjyu2pR59Um11aq/yGnXMaNm4U3vbL+mv89uiAcw2MgMbjk4ANlhZRYGcaNmbgvn8mbRh+ZebmaPhQs/Q67KnK1rZKpt1pY3DpylhB2FP4bsewcohpBM2/QfthUcJvWIlOhYHgr0aMj9eTz8LqzBQkiVkINrQ8FhBTrmS9MtYO/Nhr+Y02J4WbtS+MO64BU78E26MxtuLQH0Dv0gZG5KV8+PTMDZJc4aCPbdjavHVak7ZaI0YMM02/n2rXrqnzzjlXkyd/LON5rbC/vdynl9rt2l7lKplJV6OWgb6Jjrr2Fg2eulhDksmaKwCig47iQoBNNYasNOAnZ+v5XxfqX+98pdtHv6d/v/axnvjge40yYHIPCkDFXSi4VwHI/wjq/2pytyTCOUHoIAQKLAGsD+7HTrcPTfxETY88pQD0dRtpzyOP0yMDh+oL8+U/Tlyx0Zen++kV+iMv87+gB7izZszUg/c9oEZ1GwZfFt+2VZvWOvfCC8LiJfqTQXdiMGEGLADGhIANabhhE44NoGcalXH6M4n2slFI3tp8vT52nA7d/0BVN0um+oaXdRZoeQM6i4zss0wFO75B2xcIhII3vVLQYICevfPCva1PYrXmX5EYp2g9o9iA+A1wSW4jAAfoyaxjOhxN72PN+Mdqe74T32CZMmBHOBL4ZHUibkNJYzob03CduBHRYm5MJTHvyaYikOcV4nNjQ9etD8z51jusIjtB9evX1XHHHKvxY19Ttvm6qWvSNe6NCep68EHmz9vAVzFNX7OB9jr9Ao2clqghi7LUbym75xQA0vekx+R/ZYWZ9QbCQeZ7vzA1UVcNHq8GJ52vqgccp1bHnqMDL75BNw+aoHsmfqaXfpmvMeTbm4093Gxttr5+YXF6eEEm93opOVc9Dcjcf7BZDWTfhS2wNwC974ocDTCTf3Bavoam5mvEsly9mpihh1//SDU6dVVCjXoG+oba08z7l0eP11eLUzQ7Z33YMwAjK8s0/IY9M7TOTPu1ZuYw2IB+6pTfddmFl6pGpWqqVaVG0PZ77bO3xk+cQA+G/mQc3Jzn06U/RO49OxGzpzr56ww+mp7NNqFYRnGKft8WxN1zDPCr09foo/c+0IXdzlIlE0iVEsqoBslYZcsY4M0PrWhAJ7jHb/usWr1awe8NhfbAaygZFpJAsQDYUYl2AFxeR+WBWKwaZsQmTpz4h2m3qPCAXBAQ6Sd4h3DkOly9kgIe+sM8PcRvAkcPPfRQeBASmLlUVlT5uQzGxgGxj+zsTL37/js6tdtJG0H/+vgJwdQF9K+9MdFA36Vg0CubT1+ttvY4/kwN+GaGhi/M0sBEMvEKAmiAkug8O+b2XJmvFxPNX0zO0GPfTtOpj7+khD1MeDRoq7ItOqrWngep05mX69Br79Blz/TTI+98rt6/z9fABSs0cOkaDVqeEwRGn+RM9UxMV5+lvHHHzHYD/bDlZhmwUy975qdkqb9p7gHL0jRkeYaGL+Mlm6s1dv5KdX/zYzXufIQSqtRSuTr1ddAJJ2vYG29r2vK0MC9PnBlfPiTjUEy7A3oGBcBTvv36G51tgKhSrpKqVagSfN4uB3XVR598bBcU9OfmQM88LYEb8vQZD0BCUI9gEJqCc7kH47O9gWJPDHX4/suvde3Fl6t6QvkAejIPC4BuoK5YoM1Zb4Cvz7LisD8BiTzwhB0D8CgYLExv9/Zuy7YieID9Ahkzxg5rDY2NyxZNsIli0XmC6TpyYHz7LRQwVrf3UUnoD+a9E0zGvDDmJKYIkUd25YS5nKIDkpeXow8+fF/dzjhFDQz0xx99jIH+Da2zeqVlZOr1N9/QQYeYtoQJzK9JqFRVrQ45Vs+8w2uqDIiL12lQ0gbQo+UjoOfNtv0MlE/+MFMX9xyiukeepoQmuxfsZVfBOqJq47BhZbsjTla3m+/VP/sM0xPvfqpXfpimQbOXqM/Mxeo5Y6FemblIAxcu09jlmXrDtPmbq/I0YUW2xi/P0hhAnrJKI1JWaGTSSo1IXK5Rc5M0ZtpC3dtnpJrte6jVuYaq1KuvY0/tpiFjx+uXeQu1cNUarcjKUZgtpzusrDcNvzbfTHvzdnPycpWemaE33nhLR5uFUNEAUdHMWkBPMOa333/lyo0DXBjo+RvmPYEblmb67rKYwiRNMVaFgST6fVuSP2X21Om664ZbVM/AXsHaV7kyG44UgL5yFdqdoCpWsATKWylXwcz78gkqC/jtN7EjpvfYdLWw9uzIRDt426xP21FQqAMHDgzjF1WmTnyHB1jvQjwA64B+YnMS3067pP1TKOgxR1gCyNQAUolXWbF4AInlFH1g/ro8ffjpJJ1x1mmqV7e2jj78CI0b87oxv7Q6M0cT3nrzv6Bn4Y1pvCb7dNUDI3n1c8ZG0JMW+1/QkzO/TsPMWR5sTnOfeSv11Be/6eLufXTgJderzl4GxBpNjanM1y5bXWVrN1Ozvbuqy1kX6/Rb79XlTzyvf7wySLcPH6vbR7yme8e+pSc/+FR9f5iiUTPma+ycxXp11kIN/m22+v0yQz1/+FkvfvudnvniSz314Sd6/I1JenLMWzrpyptUr/UeBvoqqlSjhvbev7NuufNO9RkyTK+/+4F+mTpDWQb00B0hz35tAL3pXWXkZmrFylXWd7219x57GRjKBYavYcC48srLtXgJ77ot6MtNgR5CkzJXy+4yYfGS3QOtCPBZdBEdl+1NBfsASksXLNJ/7r1PjWrWCm1kjr4SYK9UVhVt3GvasbpWmlepptqm3auzBZkdL2tCASsAXxdthu/7v0jMiJHhSlujoGcGJlbIOaAZdyL/AJ1ruJZFW99//304r6RU6JQdmoUpGFI93Y/Ar4gmA2ysIDptfb4++nyyzjy7m+rUrqnDDjpYI4e/Gnz61Vm5mvDOW2beH7AB9CbZy1VQwz321d0Dxpimz9DAJfmFaPqChTL9WW9vFtBQM8cHL0ozYM7Q/eM+0Bl3P65djzlT1XfZTWWYQy9nPmQF83fr7aJqrTuqvgmAZl2O1p6nnquOp52n/c+5VMdcfYPOu/sBXf3YM/pH9xesvKSL7n9M59z7sE67824dd8vNOuL669T1iqvU+dxL1PnU81TDrAh2yy1n1gmmdb0GdbVbh111gJnnx554koaPflXLV6Qa8PLCZhkYvLg7YSorO10LlyzVvff+Sy2a7BK0PBqvdbNd9MgjD2lNRoE/R19uDvR8ohGYq/fUaA8IYYGx2g7y+zjFjuu2IMaeNq8wofTso4+paf36wZKpXLWCGjSsrYM676O2BuZWJuj3rlVXhzRvrf1NcNWvUkmV0PTGE87QZBzedtttod7bo+7bkxz0tJUpV8x7Emww76NjBnn7mZJDq7OBBmOOlccuwmj/0vTP/wM9pgYPI3CH/0AlmSp65plnNkaaIf/EjKV8+tUnOue8M1W3Ti0d0qWrhg4eodVpmVqTnafX335zI+jLV0TTV1CjPffTPQNfNZ8+XQMM9Kyuc9CHQJ4BHtAHAZCSbz53fvDDR7D4ZnpyeInFTb1GqOs5l6vNQUepcvP24i23CdWtVDEhUIl986z472rmCtRvquqtdlODTvuqMa/QssJS38ot26t8i5ZKaG6WQxM738z4hKp1TEDVMnfE7lGmaoi4449WqVxejZvUtz5pG/zyl17pqUWJi4OgZKkxPQKYc9bnBdD/Nm26Lrn0ctWuWjOAvmqZ8jrioEP06qujNgDmv2AtDPTezwR8CAbxhhO0PEzAFBCbKZBBuXE8In5h9Pu2Isae/9NXpar3yy+oZbPGoZ9q1K6ijnvuqle6P6XzDj9cRzVtqYv3OkCPnH+Zbjq5mzq3aqk6Zg1UQBHY+Qgw/HrWfJQ06eSvSD4urIhjxgU8IbBRpARnwVRh48QxeIrt6Tyxh+AtK/SiW26VhP4f6PkOszFFxI6eDAhSiUU0SCWv4EYms0EH9l9+94UuvPi84NMf2uUQDRk0XKtS05Wek6/X3nrDANJZZcyUqxh2lq2g5nsfoAeHva5hi1aHt8YCaqbRNkbvrbzCHLspQza6YMEM8/VD2Ct/cYYGzVul/r8v1n1mgl/1bA8dd9OdZvZfo33OuFjtjj5Njfc+TLXa76eyDVsroaaBuZKBuHJtEwwNVa4Be/Y1Vfl6jVWlbjMrTVSpUSNV3aWZarZprbrtdlWDNnuoccsOpn1aqUrFWmGqDQ3WtlUznXTiMbr8sot011136IMPJ2lF6spgYgN6LJ/0nDXKWJejtNwMTf78cx19zHEG+PIG+IqqX62mLr/wYv344/fhXMj7vDDQe39j6jGrwvbKHgWGeRACmID8PUrcMzqu24pog3mfyjNXZtSwwdpt1zZBuDduVl8nnnC0Hvrnzbr6+BN1ccfOGnjtLXrp0mv0z2NPUoca1dWyNjMZBaDHVSEnnQVI/0ughxhD3DBeTAJ4Eda4zmydhYL1cYqOF9cgEFibwQ48XMfsDbj04J/zRnEpgD56sT+YVUC8FJEBQUKxIQL+h58bmIrPAPt1+u6Hb3XppRerccNGOvygwwLoU1ca8+fkhuh9ly4HqGw5zL4qpj0rqeV+XfXomDfDCzD6hnfdFYA+bHAB2K2EHW5MCJBjP9jMfObUX0kseLU022ANXpKhd4w/CNg9+tG3uvf1SfrnkNfDq6tOufUhHX75zWp37JlqcfDxZu4frHp7HqBGex2opvt2UdP97NM0/r5dDw8v6zj42ON09Bln6LRLL9V511yny677p6687jbT6Huby9I4pI7WMUY9/ZQT9UrPF/XpJx+axF2gVavTAtsj/hz0uabls9bmaE1OVpiW22+/zkFo1KxcXc0aNNLtt9wari0K6KNEkgfmPHO27gfDEGytHI0CQyVliOISL/MIaUnm27/x2lh12quDyprZ3qZdS11+8QW6rFs3nbzH3rrzmFPV+4obNPL2+3Vkg+ZqXd76s1zZAHriRpiuBI5ZSOL98b9CtIVArCfYAHoyKtk114V1wNMG7EGMH3PxbJ7CHoRcx74EJPR4DCdW0BeVNq6yi4IZ6UNesL9NBV+LnGg2BfBzwqcVGJ5PtNAjDz2sti3aqFWzlnrqiaetUgVz1byvbf999wuDW6Eq+8hX0e5HHq3n3p0cXnfFq6UHGqB5MWWfxQZm8+8HG+AJ7GHyFyTUFIA+JOwgCOw7u+YiMHjV9JBlueFd+FgB/WevUI9fF+vFH+bo+qFv6roB4/U3cwWufHGQrnmxf9jr/l/Dxqr7mNc16o239OprE/Xae+/r3S+/0gc//KKPfp6mST/8rrc/+04XX32DmrRoZ51ewfqhYZjKZPOH6ADxPdqPRO0zs7OUlJKso445OgSoypUpG7a8PrDzARo2ZGhgBL+Oa6KAjwVsFAAwCpF/mICC1ifAx6YMUHQctznxiFyrG7GM/PWaPuU3XX/93zdMxSXoBGv73i3bqNs+XfX0FTeqz0336bojT1HnBruoofn4RPJbNC0ITFLYWIOcBO+X7dKGbUSx9WfBFLMtvuUVayfQ9Iw55OdGx544DmMLbrgGa5sVsJj98ElJ6Q/z9DyY31SEwAOr5ZhKQTKx2QHbHvM3zoERAXx4SaN9JiYuVf++A3TaiafqyMOPCkttWShAiu74seMC6MuSjVXJNL1pqF3M7394/AQNWJgasu54Ky0JOv2WrC8AvZXw3cDObjcsvAmLb0xADEizYoqNUrCRpQkIux7zH+HRd0mOei1IV4+5qeH11IPYgGPh6rCP3uA5SRq9aKXeWr5GX6zKDK+WXpOfFxbLMOe+zLpjcfZ6zVyZoR/nLdWVN9+l5u33CDMOjZq10BNPPKWFCwt2dnGiP3yQvQ8xzVgdx7qF6DQb7xtHCEav4XtRQU/klrUN+PXck4xJIrqMF8T9thvxKHgvlPWaM2NmcD9q1jZ3yJi0y/6dtVfLdjpuz8667oSz9M/TLtRZ+x2mXWs1UmUDeQVP4rHvmPbdu3cPCgcqrB92FPJx9cJcO5tpELOgXxg3slwR4A56Jx8/tDhp1iRgYdUxZccMGoIg9pri0v/bRIPCg0nQYXcO5hYZFCKPbOvLUkG/hup5QatPnTJNo0aMDuCnQWg6Iv4s1iHBh7TLsBW2gb5ep710x+BhemVWUnhRJVlyrKBjW2ov7F4b/U0evm+SAdgp7GqLFYDmD3vchcJmm+y7tza8QYd35LHjDkk4Q5PXaOSydL22Olcfr8lSolU+1UAF4Fk0s9T6c056vn5LWa2vZy3U1bffp13YI698ZbVqt5v5nD02RssdjDCoDzB9w3EGjJcPEnzxwBtaGRONgJwPrl+zKdD73/183kfgU3cwD0zEjro8i2uj5NdsM+L2VBVtb5/JiUsCcEkxDn69uXrtm7ZQxyYtdVCL3XRoi921T4MWalaxRtDyVSpUDKvs6B/2xiMBKdzW6l1axv4zifpHC0E8dgRirNyqYQEOAjwq0DnXiWlzBDkLc3xmwxe9OT+UtI/+ME/vlYR4KP4DgSI0PSbGcccdF7KIWPXlZI8PY89/5J2nrlilZctWKCMrMxxng0yCD2FnXZZR2kAD+lp7dtQ/BwxW71nJwUTnfXZo8QBku30oG8AcCkE+Az3r69nrrmfSWiv8NmGAVZBkf2evPfau3xAbQPuzQq83u+LyUo3kbDsnw9yG1Rq2IlMTlq/WjOxcJZlUtVOFgTwnc62mpmXr5+RV+mLGfJ177U2q37KtMXEF7d6xk/r2GxCkLYQ0ZtDofB88/83mCayjZudX+g6AIjhZvciUixP9vTnQQ9FxQWswdccedM5A5FGQV1HSnVRKRVRrQ8nKyAzZYoceflgAPe1uWLuu6leurhbV6qh11boB8PXKVwkzGb4Sj0AxCgXLCKIPYgXYjkQ+Xj5m4IjsSdfytJlEK3cT/bzoJ/hDWLg/j/KICkWKW0XFpf8XyIO4IcyLn8576xgUHoymuvbaa4MW9wgr1QT4G74UFCN+AgNWYL06dkzYD5/llQlVzLyvXEkN99tf9418NbxempVy4Z125q+zFVZ4l50Jgb6mocmNR2OzYy6F99phEfQ1tRw20gDcBnw21ySHv9/SgiW4CAgsgLBNliEagdLfrh3ArrkrTciszNN4A/5sqyh6226jhVb331dn69fULH2/dKU+mT5XB59ylsrWrmc+aoWw7dfwEaM2gMt895z/Rpl9wGBWBgMpzuIRfDgGGsGJqY9v55YCxHVbAn2UCOIw1YMAZkwoCBZWRTponCm2B218in1hvQF7sV92xeWqwlJqAz7tr1i2nGpXqKL6FaqqtllMNStUDhreNR/pxbwByANUtN/Ljkje/15IwmFVHYFX2otlw9oW5wNvp58P9rCQ6UvWJHANypdUXv7u5IqmuBRAH3uxV4IHU2GkFFKbh5Ow89hjj4W54cCgPux8mHBG21MvfP307CytXJ2mwcOHaf8DOhfsfY+pW62qWh52mJ548x2NXpxufjeANnCixdlJZ9X6AHhefNE/OSv45EOWFpRBpq3ZFLP3cgoANvPdQD0Qq8A0PJo+FO5nSPYZgLDnvZXwyiy7bvCKPI1blqkpZiEtsPripc/MWaefVmWF8kPKKn00fY5a7NvFLBPeWltRJ5xyqia++UYIpBC1zs3N3qjtXTMxKIAeweizH5hnLJRhc0PMcPrVB5hCfxcV9JzPjrL49a41YCaythA0fm1JGaI4xJDzFFbahfavM7dwcWLYuXefzvurcs0Ni24M/Ex3Eqn3NOTA/NWrhVwQtDzxHyja9s31w1+ZomPLTse4Y75ghnYzS8EsTOxuxn4NPIU1OXLkyBAPIq6GAmFV5Zb4oyj0B03vD4/+JmeYt49iXsBkRPORWmyJzRwiJgoR1/wNfp3LAJghIy9HcxYtCEtr23fY3fxiExxo+iqVtasJkpcmfazRC9doaGJeyMoLG2UYKNHKfcwMf3G2mf5zlmngnBQNm7dCwxamatiS1Rq6LEuDU3M0aLWZ9yYc+hILsGsoYbPMDQWg4yY46In6FyyrXRum/MYty9ZvxqvzrK6sTJ+et14/r8nRL+m5+i4lVR9MmaGarXezeldS1br1dfnfrtYnn326QcOjif5oggI0LCCm1gi6MC3DIKPl2ckWKwmfPHaQ6e/NgT52fLDASFcF7M5IRIPJ3vJzSzqdUxyiNvTAxl6wA2j7X376WU8/071gW/T2bVS9bu3wOq+qlQte2omma9CoofYzwUCmJ5tEeLbn9hBW25p8XCnseovw99x5Cvvdffnll8Gy8TGNXoPSYIxZh0AWKAt1eHcdrh3jW5AT8kceKQ6FKTt/sHe4/4aQVGyFxR5dYerJTDK0PuuBCUbwUj5epfvjNz8ocf5irVi20szfdK1KX6Pk1BX6ZdrvevK5Z9SsZQuT+sagaPuKFbTH8SfolY8/14h5qzRoofnaYYltbkECTup69WKF24+zdUXvEbqh/xjdPepNPfrOp3ru61/Ua9pcDVxswmB5uvn1meqxrEDr9+W9eOYWUII5b8VjBGEzjWSzIEywsPsO6+XHpphWN2zwDv251tbfcvL13apM/ZiWoc8XJen1b35UlWYtrb5V1ahla916x+3GoN8ZMAEUnV7Q8QwABbMfCc0mJEzt4WszyAwciRlEcKN+t/c9124K9Pzdx8WPMz2H9uC+zky+w6oTjBMdx21B3N3YVjlm9VDvsIGCHWQfgTnz5mriu2/r/sce0Wnnna2Dzc8//MgjgpnK5pcP/+cRvT5xQpi/drMe8jZG+2BHIx9XCoqRjDrGyLGDK+Z7+0cxR5v5RACibNkqm+vIkyEj1gU5fy+NcPxDIK8w4kFEonmDLfP2gJ1IogeniO6zlQ9TdXffdpcGDxwS9nj/5IvP9dEXn+nFnj106hmnq37DBiGAV6ZaddVu00Zn3nCjRv48ReMMtEyrsYMO8+2shR9qIL5p3CS1u+QGJex1mMrv0UV1Ox+pXY4+Jbyk8q4RY9Xzu1/09Le/mJmeG7a+6md++sA01sbn6KXENL28aLV6Ls0Oefto+N4pJkiW5IbpvEHJuRqxcq0m2nW/GK+6lv80MVnfp67REvv96bxE3fX8ywb6XUIGYRszyV546UUDHH9lcOj0gkg9vhmvd2J9NOY7L3nAT2WKjn5CWBKd9pcXxg40vykAh89Yhuc8iOMU3As2YICJXNOTpEMiRwDfBoq9z9YmagUbRjV9kIMF1Q3W3qzFCzRt/hz9NmOafvt9SohIM9e8NDkpZDL+rxBj5OMHYbpjjjM1R1IOY4VlRgAWiwzh7yD2cUXo+/jjHhL85jqsa6whhCPnQM47JaEtgp6G8CAagMZHW2GuUHly8jHVatWoqV0aN9d+nfbVicefpPPOu0DnX3yRzrnoAh1xzNFq2bpVeOsJb7nBTN6j60G667kX9OovU/X68iwNWwHoc9RnSbr6GFif/22RLnh5mOode7YSGu6qhJrNlVC3pcq12E2NDzxcB1xwmc69/2Fd8mwPPfnNb3r6t3nqu8DM/5RMDUlKV/9FqRq4NEOjVq0PAB+YlKP+BvYBiVkalJihoUsyw3vvXktao+/XZGu+Mek867/vVqTq57TMIAQ+mDZLtz/ZXeUMsMw6tGrXVo8++qh++vF7zTQG5vXc3333TZDIzGiQWILlw5tIeGsLgtDNbwQjC0n8LaPOGEUBvQMe8u9o8c8//zxoegJlgB5LgilWmC16zbYkngLg/wD6AlkYiK+wdYFg2JDTYW0L7bbf26eW25aiY+Xf+cSCYTxQkvjljBF+/dVXXx0sQQS3m/c+Xoy9/2Zal23qAD0BdCL5Ub4pDW0R9P4AGJI0XCpM9JjsIF6ix2aIDes3UI3K1VWlfGVVq1JdtWrVUY06tVW1dk3Vb9woJGsgrZo0ba7OBx2q2x94SB/8+Ku+XZFm2jZTw1Mx67PVPzlT/RPXqOf0JfrH8De0/1W3KaGNmUa1zTWoUNeKdR6lfnNV320vNexypM7899O6/IVBuufV9/Tk+1/r8Q++1qPvfaXnPp+iflMSNWBakgbOTNHg2Ss1dPYKDZ+zXMNmJ2v4rKUaPX2+3ps5T98nLdOPKSs0afosvf3rVE36fYZ6jXtd5133D1WoUye4JfigF114vp54/FH9+/77dO3V1+iqK64MyROsUQDYtBGtTjSdpBkGmu8kNiEwo1Od9CvMQSku6DmPoB1xAp6JYOE5+H3RPIptTdQGwAdW5AclBvQuFP7bigIC9L4s93+FHJTEbTDHyU1BKWIRowAw15lr970NEd5c4+MaHTeEuq+uQ1hgIfnfSxuv2SLooVjGI9uMhjGFcNcddwbmP+nYE9V1/y7af9/OOvDArurctYv2O/CAkKiBVXD0kUfprDPO1uNPdNc3v0zR/NR0/Z6epXftc9SKdNPSqzckzmRrvGn+AQb8f0/8WIdee5c6dbtUdTuaqdPEtD4r51hKy+60leqoVseD1eKIbupywd91wnV368Tr79Hx/7hHZ9/zpK7u3l/3Dn9L941+Vw+On6xHX/9Yj038SP8Z/74eGvuWHh75mh7pN0jPDx+lZ4eO0J3PPq/rH3g4lDP/do1226+zqgF6os9lE9SmxS7ab69OatG0iapUqhyyyRzgPjXnPjYSmk80PltAIfkZrGhfMoiUooA+dgywGoh8o+09wBp9kykUvc+2IJ4Cm/8B9DxyQ1X9b5RQJ/Yb2PBuP2jDxw5N3tduduPqkXGJO0cGHjyAUCYQzgs8each5Ocz5oWNF9jydGtAT96H/51rSkNFAn2UqKAXHj596jR9/eVXevO1NzRs4FAN7D9IQ4cO17ARwzVo2FC90ruXXh01WhPGjdf40ePMPP5NGbnrlG7tTLQ2/GLlw/QcTUhZo1GLV4YI/fjlOZq4Yq1GLkjVU5O/172j3jFzvrs6n3eV6ux9sCq16WTmvml/3z2nciOVa2iuRqtOoVRptZdq79FFzbser44nna+9ul2s/c66Ugee9zd1Pe9KdTn7Mu1/xgXa/9Qzw553+xx+lDoecriaddpb9dq0V7M9Oqlh67Z270qqVLnghZNMM9WpXkVNTJPXrl4tvKShbes2IXmCKRikOrnV/hYaH2xMfgKdvl21953TlkDvFL0GYk6e2QBAz7MQMjAae6x7YIx7bUuiRgHQ/iNS+OBvmPV89+OhGPBpz/+CpnccQMzcsHMtO91g+fk0N9mTxMOwkKPbljM+UX6IjjHmPOnVKBMUC+tePDfG+SN6fnGoSKDnIYU9IDCVHSZam5uRo/TUNUpLXR2i96lpq8JbazFl2C03K22N8uwc9pbiThgoSfY9xT5nWPk+a50mLc/U6Llmfs9boVFLMjQiMV2D56eq37REPfPFr7r/tfd1ZfdeOub6O7TbCeeocecjzec3cNZoYsBnDb1J1gq1rNQscANYR896eP5WraGVBgUvomStfBU7j+242bGVQnow2zJXra6q9RuqYk27PmzZzE43ldWxXRuddNQRuviss3XVJZfo7tvv0PPPPhfmUglyMoXJ0kdyyD1ow2CRloppFgVgdNCKCvpYYvqGPQ8w8ZkSpJ5E8Mmh8CSdotyntMRYBs7wL+FHwcdGgQDxhepQIuf8L5CDnrgXs1lodSwvxoQgHsIY7c/0dmFUGLZIg2fLOpJziAmwGaYvqoI2hcmi0BZBH2UcZ9I/PNDNtciAcgmgn79wQYgBpCxZqrzVGYELctfkas7shZoyfYF+nDFPc5anaWpKmqZnrNWUbNP67EK7ME3DlmRqcHJWyJwr8PXT1G9ekvpNXaBnPvle/x77nu7qP0pn3navDr3wUrU94hg12PcA1dxjL1Vut7vKt9xV5Vq0N6HQUgkNzCqo38ysg8ZKqNdYZeo1UsUGjVW5YSO1bLebmrVqq/am3Q88/EgdfvyJOuTIY9Rm1w5hC2e0/D577KG7br5R44YP1Wfvf6BvTOpOn/K7liYuCVMvRO6ZQmNqxjUv0ywE9ZDY+G5QLAiLCvrY3xDaHCYjwuvr60niYDUkaZ8EirYHMfSBEzZ+KfgA8BtBHw7Yf7y+mxIObvzYoclxwHZmEyZMCIlsLH1GyzN7gyIAsMRaGF/GzfkB4nqOb8TTBuIcZoIQ6tyHDD5fVFVaKrJ57wzqxO9QUerKh43whlks8/kzNG3GdH340eSwPnri6xM0c8pUJS9aqt9//F3jx0xU3wFD9dqb72vMxHf19kef66f5SVpghsCv6dKklCy9sSJH41Lz1HeRaXrz9YevztXw1GxNMF5mc8sxS1dr+KxEDZ8y23z1t3Xp48/ouJvv0CFX/V2dL75Ce517kTqeeb46nX6B9jz1bHU48TR1OO4kdTz2RO1//MnqevKpOurUbrrmhht05VVX6+bbbtcT3Z/RMy+8rDvuuVdHHX1s6Oy6NWvp7NNO08SxY5WduqpAouXnFbzBxiwckigIzhBpJesODY+PT4YcU2iFRdPpRz9WFNBzrDDifASLvwoZ4JPqG91Ca1vTH1tWQBxz0BOwCwdo0gb+CIes/f9L5j2bZLCTr7/enQL/sFLVsw0h5tjdn4f4DsB9zBlr5w38eHgKAdKtW7cwTcv5nBO9R3Gp2D59LMUyNMRiAfxYopUwJHOVLz7/QkjRfOetd3W8adO69Rpojz331p4d99HxJ3XTQ088q6+mzJTBKuTCf52aqfFzEjVu8bKwO+2QxBUatGSlhixL17CVZgGkWlmeoVEp2RqdlKURi9dowOxkvfLbXD3/7e96+ouf1f3T7/X0h1/qyXc/1hOvv6unxr2hFye8rcHvf6TxX3ynSd/9rLmLlgQpjVUye+4czZg5O6TaokExmytVqKh77ro77F2/YtlyZawxqWRNRmJzHdsXsXLOc6SR8mRcYfZjom2J6L8AABvIaPHjWyK0CwKG+AHPpx7k5bP/GuT3jhLHYgXL1iZq/ofa+4H/94cdm+hLFlDh3sEzHtMhoEv0HWHArI0L+uiY+jHI/4YAgQj6ks/Atlrcj9iRJ19FrykJlRr0Tl4BJBBgIP/apR5RTOaxWUDAwgrfhgsJxhtMmzZpoVNPO1M9+gzQlHmLwhJXwh3kLH2zOluTVqzWhKRUjVxifqyBf8DSleqbtEoDkswCSMnVqGX5BcW+M/8+cukajTZLgLfKvpaYqomLUvT2giRNWpCsr01w/L4yQwsz12u5CcvMvALmB8QIK2Yl0NCsMUB7MgXHarnodBu71DBlBrDQ8ACdc4my4r/xjgBiGUWZWvHBLinoqS/7ovuiKArfSZ2OruaLUlHuG6eik/M07hX9jwAmsEu0nhR118qxY1oY6F1Acw2xGV+dR6wIi9L/DkXvVRzaaqB3zQGjY/KShkpF6QRMXuazmYZg6goBgET0qHiFhPJq3nQXnXnWeXqpVz99/ds0JWblKdXuxybRs+zWv1m/fZ21XpNX5Whi8iqNMfCOSVyl18zvf920PeW1pZkax0sq7DhvlX17ySpNSkrTJymr9M3y1fptVabmpOdqSc56pdo98Xpz1/13XTKgB8wkHwF2ND3WCqmUHjkFzGTesfoQjcoA+6Cg4QnY4Oe7xN4SRQc7Wvz4lohzyPQjks/sAf3NFB65ASy3pU1+f7+ff25rbb8zEKY5c/K+NTm8wDjggzM9Fx1D+jva59HvnOfjBPE3lCdLsQkKojh79+79/xbplIS2uqan0pi1aEeyxAAOCQqXXnppMFcAD+vrkWDMc1diiSWMaqVFi1Y65dQzdO8Dj2jImNf1/jc/anZqhuZm5WueYWiGKc6fV+cF0/9b8/F/zJE+WZGtj1fm6bPU/PD56coc+56tr1bn6If0vLCA5vf0HM0yIZJo56PdV1u/Zlh/Z9tnbl7BIhkK+QcjRowI0VcGkOg7wouACvPifJKURO44bXOAEbRjfTTCwCOs9MP20PQQ00AE7xBERPC97kwXsrDDg3rcF3Jm899xKjnhoxNDIReFIC57JpAvgf8dHT/63McV8jF38vGOCgJmaBAoAJ57YzmgUEtLpQZ9tJJOgIeOIJrtoCfg5FNJfLKc8pijjtYuTRqrasWCKHnF8gXTXLvu1kEHH3G0up13oV4eOFTDJryjN7/8Vu//8Jsm/TJVX89epFkGeuA110A82/puvlVjDsV+zzEBMd+OLbL+XWyfSVaW29/WWMmygkXPtm6UvA2ghzDxyXlGYxOLwJ9id1YkLNlumO4cZwol+PuVKgUBwSDzymkPnjGY9Av32xL5YHNNtPjxohLMh/vELAJ5+NSPgCJp08wgUB+/X2FjFqeSE4IVQLIqDreWF076NtX0tY9pdAz45JhT9LgTOPJpWSxKrDly8Es7ftsE9Gg48tF91x1Az1ryqLmDqd+79ys6+qgj1KolWz8V+KNkv4UMuEoVVdNcgOat22mfrgfrpLPP1UVXX6ub7rlfLw0Yos9/+l2JpsHNlQ8bYBAA9IJbwCfr2dgGC0ijczG4w9P5j2rzuaGP3RzH/WCji7POOisEyNDgrDDEZCMGQcGUZykrbgD7tJMnjVSGuA+lqFNm9IczQLT48aKQjwFuCnVhFReaBxeKjEAYx3f7cSrqveO0ZXKXln0OUGi4VFBhfRwd19jv0U8IZUSg1resO/fcc0OSTlRYlIRKDfrCKoufQ5DLd3dBI6I9YUhnUHyT2bNnqmevHrr86su0657tVa12VZWrXF684yzsqMobT61UMJ+mhvnY9QxsrdrvrmNPOlUP/OdxjXvrXU2ZO18zFi/RwmWrlLQmQ8szs5Wak6c00+Cr89eGl0visYd0Ans0S+DXZudrXUauctOztSw5JQwUAom1zwRL8McIzLA0lqgp8+BMnWDGA3S0JxYA5zL/SrCGdjEYtJ1SVKLfKFHAU/x4USgaP6BfaQdJQWy8QConG6FEEzug4tw/TpsmxsrJv3uMqDCK9jk8478LGwssReI1HsFn2g5LtKjxok3RVg/kUXkkHcEv/HYqi6YnDRGpBSC8c2jm4uREDRszTOdfdr467NNBjXZpYhrezOfqlVUWAbBh5xVec13wSqyyqlm3jg467HD97e/X6T9PPKnnX+6hYaNG670PJ+u7n37WzLnzlGhgXpa6SukZWcowQZCZnqXMtAytWZGmFUuXKXl+ohbPXRjeKMtUCK8LolOZZiFyz8YFmFQEUQA9x9i4Ar+dhCPiFgDM20L7aRvS2QezsIGMJT+P+0RLca538rrwSR3ZvZhxIHkIU9H/VpT7xqloFB0rB7tjAXDyPdrnseNV2HH/jgXBi0yw2hz0LMT500EfrSxEhWB8zHu2eybQRRCC5ab4llHKX7dWiSlLdNs9d+jUM0/TxVdcotPPOUPnXXy+zr3oPLXr0F71mzRQ7QZ1VLkGW2cXCIAy5RJUpVpV1Tbw42OT+84bQ8gHOOucs3XPffdqyLChmvzxR8Hk+vLzL/T+ux/o9TGvaZi5Bi8984IevPff+ucNt+jM088IQMeE8mWqfDL9hqYnGIa5TICP+hM0YzB8MGkvhe8O9pIQ10XvCQNROF7Se0L4+tyX4r95BlokViNt6jleB79PtEQZkN9O/j3aL9F7+LP8eCxt6vjWptjnRNtQnOfDFyThRN2o6L1KSky7Yl3Cl1htgJ560e/FsSijtNU0vRODTGUAPWBCW6LpAb2vMPLOoOKTJn+gk045Ua3atFSXgw7UMccdrTvvvkPDRgzViy+/oOtvuE5XXX2lgflMHX7kYeqw5+5q3JTlujVMC1cJ786rWaOaqlSuGArJKZhDmOdsO/z3a67VFZddrvPOOV+nnnCKjjr0SB2wT2d1aLe7WjdvFWYQMN2Za0eAMO+O7wTQ8dd5lxjSFhcAbQlQqL+D00uUuUtCXBe9byzoS3pfiPtSPA+BuAUCDCYl9uD33tQzOO738LpEC+T1hjjGuUWh6D2itKnj25OK+nwsPoJ5LJ8moMtU29Yi3EesZHDExiyY9/Q1pSizQ4XRVgO9dxCDjRbBpwf0zFuiOYnee+TRz0U4jBszVgfut3/BKrYaNdVht9310AMPas6s2UpemhQ2qWC5KJ1Jpz7++KPhLSrnnHOWSb7jdewxR6lrlwO0a/u2atignqpVraxKFctvjK4z3UG0vUaNWqpauVpIBuIVVRXLVwrfmS3ASkBAAHIy6Yg9OCgI0AGMKBPz3Zk8FvAlZVS/NnpfFzClua/Xm/uRu80iDmYbiASzsQdRZp4Zpdhn8dvrEUuFXevPjFJh1xZ27M8grzPFx7I4xEKae+65J2hk+hRgRpO5SkPw3yWXXBICyCgyfHzG0utbEtrqoIdgVoJcJLYAeqQUqapozGhF2Sf9ow8m6ZjDjih4p5mVutVr6p833ax5BvrM1eaHclsrbLiYZj76vHlz9P3331rjJ+ntt9/Uu++8pbFjRqt3r5565OEHdcM/rtPZZ50RNDZz1kTd+ezc+UAz/w/TUUcdYyb7KSY9z9XFF18a3sTDggiiogDAfXIGn3YgwOhk2ufMEQUm351R/JySUmH353dp7s21EGYny27JFGMKiMg+U3sIAtoZJZ4fJa9XYc/3+zvF1tOv9T5y4nj0OX6dlx2FcJdIyMKVJdDLPD3Kg/TbrdEOrLMrrrgiJLiRcEU+Rux4FZe2CegZTDqC6L2DniBEYaB//823dfxhR4ZXOG8E/Q03afb0GVq9MjWAfeNKPitc72As8GmMofJzlb4mTSnJSzV9+lSTtB8Fjd2jR4+Q3MDOtFgJ1Ontd97T5I8+0ZdffaMffvw5SFI6NsqAPAPTyefvaRt/jwKdczjOp38v7SD7PbifP4/P6POK+wyupb8wE4no+xoBYi1kkZF0FGsmck2UvE7RZ/sxJ75Hf0N+jf/Nf0fJjxVW+Nu2psKe6f1dFMKUZzktFiV9yifJWyg9xq+0RMo3rjGaHtBj8Zb2vtsE9HwnH5l5bkBPhfFLSB7xzuQcQN+7R0/tu2enAPiq5SuqVdPm+s+DDyl12XKtX7tO2ZlZG4NOFBg0l1Vu6woin+xMu37De+G9sB89mo0IO4PiJjrX5681ANtpLkeiRN3C/a1wLoNPB/Pb6xDLEHynHtH2l4a4D/ek+PP9mSV9DsKL9GGmItHygJ40Y8YHYUC7ohRtHxT7XL5TL/rE68gzEJ70M5s+ogELI2+b0+baU5K2bk0qyvNpJ/EfZnvgc/qW708//fTG3I3SELEkhAiKEyVKph99Xhra6oE8J1JWqSygpzBPz9SRdyQDD3M89eTj2q1tm+DT44u3ad3STO4njIEK3iJDA1mCSclZmx/emJORkx0+sw38aWmp4dysLECdo3Vr82KEwB+Jp1MMQkFwRIFM8e9OMDTnUDhO/f1ciN9ethb5MxxQscAvDnEdzMcsBtYWi0LQSAQ7sYDQJNx/cxTbPuoB4AE48/9YC7wumxkOXCXyAhD6xEQIcmGR+fV8enucOMZv72e+R5+3rYnnwYuuHIpDXIsf76vrAD7WFFukbY3lzYybb5vl0Xsn58Hi0lYHvQ8WZggbOiChqDBZbSSNONFZBDtefukFdeq4h0jDLV8Ok7OJmUsPKXlZkjKzMwLYASh3hU1yDaiAn8J3g4cdBYRo5GxlZaYr2wQAJn/B8f+a6hReIw3YA+DtHlHmcoaEob1A1BVmdABGmZZrtjaDcj/qTYk+j9/FfRZtpp8ZD+IcTJ8SvMQHxfJyYRZL0efEtpH6cF+AwiwNOf+82AOBwv1ZR05eA2vJeRMSFhfnO3nbnKgDMyNMe1F8lmR7EG1JTEwMATJcQra0YpqMthWlrzmHHA8sKF9lB+gxyZkpKS0RVPb978EQLjLEc6l7SWirgN4B4ERlSEtk0KksU3b4I3SodyRmEaY36a4EP5CQRNxZFkoaLFKX+3iBSVzjUwAsBWC6Oc551IVPfjtDb674eX6938NB9mcT/UXxOnnx40UhtLm/k9BNUNYSYPLTf7TZiX6AeIb3hZM/j0+0OG+n4a2qvsUz4+f3Zz052g8XgrXmBEoxVVl0xXhBWAFoQ1Ky2X+BoCrnsiyYLchQEkzzIrQKayvj48ejbSjuuLFZBSY66yjY3uryyy8PlmosRdvvhGXA2gxyRWg7Qo/2EzymTbg8Xjeui14brXNhxLXskc/iLpQnqezMDMTep7i0VUHvFeGTwQX0mJJ0Ah2Kn8P0BoONWYgWQIr5ggI6jOwjzEPOoVO8OEi9+PEo6B0QzrCxxworhQE++pw/m3yAY+vtx7dEaFimHxGu+PGMBVOZAIz0Ye4FRe+FwEUgYFoyZRpNAqGPGEMYHSZEqzHGbtayyMetOwrPJP8B14IFKY8//niIQKOxmNa95pprQgIUey/gCxMIoyCg2O2XhSyAh2f6ePDpwim2D6L94m3bHHEv+oH+IBmLKDmuz/33379J/zn2mf4WG4Qebac/iOSzqCxqtUTrBm3p3oCeLFYSxbgnljP5AFxXGt7cKqD3CkQrjKYncISWd0YD4EQ6iaRjFrIQBIYA7DQKBmElG6uUYFYa52DkGdHix2O1vP+d70Upfp/Y62MH6M8ir0dsvYtaP7Q8CUZoHtfCpBUDWv7m5PfiGKCE6ZnPZ/bDzVT6CFOYTUIwNXETuB9mLUxP3IapWTIjyXBE2xPP4RzGF6HAeHMeS63ZPw6GdrPYz6P4b+5DUgoAQgBRh1iiP5yK0idR8rFGs8N7/ly2HWP7KywR7+8o8duPIbxos7/rAOCT8EXuB26CWzYQz3PyZ0cp+ncEBhaP77nH2noEOPwefX5xaZsE8qgMzAFjoblhDgYSSYgZxGATlMD0o3P4G41CumNm0TA6mg5wQPI9Wvx4LOC9M/w8jhe1RK//q5DXJ7auRa0n2gIh6wClrxkDGBXTNPYe5OyjjRESmLqMFdrGz8MUJkjF37gfaxN4oSnBLIJ47JWApYaZjgDgejSna3/GGbOflYooBI5TOM69qCdMHhUEKAy2osIlIcHLifGCvD+wUOA7BBe//e9FIcDNDkS+8Qt8ySpRfOrNBfd4DkKQVZdu2tPHvt5kc6DnWu9XJ9rixAwUc/5uQbHTMRu0+Dmx1xaVtgroow/nO5XCPCdazyYUmCVIeDoEyU/n0Ah8QQaUQWfAMeeIKHvUk3ttCfSxgHfiOyX2uk0VPz96j78CeZ1oY7QUta70D+YxjEP/I3hJ0CHa7v3mhADF5ULDo4EBIuOFm0bQD5+cCD0LqRg37kfyE78BPT4+1hvv8uvVq1dYjcgnFp8vTUb7+7VeWMlIkBFFQFYbmh3BxHkAiDrwHe2LQCLxxYnAHy9ZZWdgwMerx3g+x4pC3rcQc+vUAxPf2wY/Ru8V7S+Ia7GGcF/dYqUgvIijIHTdvIec15xix9Drwnm4wFhE9BmCEGHK/Yoy7pujUoOeCkQ7wn/DbAAfJiITjKwi9ykpMCHaAokO4H3eGDMrKlkd2NwzWmKPF9YRRekczvHyVySvG8wQLdF6b6kQlYZp6HfAgykO6Ok3J86jPxkvUkoJrvpYkaaMBcauQYwjJrdrZ1wF/FcsAwrnki6K+Y/Jy/vc0FAIGsBEARTcl2uxBgjcYY2QPAWv8B1fHvOYeA+CivPhE2YdqAcBLuJG77//frAoiRcgFNg/gBTjkmTEEWim7t5ugIub47n03C/aZxBjgXAjHgE46ROuRQiwEzTxEM6BfDyiFPvbz0VQ4CIjhLgf48EKT/97aWirgD5a8dhGQCSAMPBIdBqA6YPZx7QGA09hcJFq+G2YQ964LYE+9vn/a+Ttoz+ixY9vqdBXmNy4ToANpiQTD63h+eGc44R5DHORYwHjAlIEM0BnGg6LzUHIJ/dDgMP07tOimQEon6y/YAETgToHBAWtCEPjZiDoeSZ7FBDsZWNS9vfDnOd58Idfh3LAnYB37rzzzhAERKBQVwKA1BXeIspdHGI2CauEqUeeg1DDDcFNIdDn/ekU/Q7f0l6vI+WEE04IC5t8rKDYezhFj/lYICyYOnQLGUEdbRPnFXavotA28ekhGgtRMSQl0VEYgwbAdJiQaAEEARqfY0RyfUrH/aCigH5TVNJO+SsRbaBEAe+MVJRCX7GuPhrIA8gExvwtuk58p085jvZE0zI2+LeAAJADKr4jQLAecAMALy4c0XxAw1hyjQOAKSeP3aANKYw7m6UyLUfMAEsCwcQcPwyOBYCpj7AB6O4ScA8Ahu8N0BE01Is20T6ASj3Y4NTzLDZH3n5MaQJ3bhFxL56Be4pv7X3uRL9CPAPB433r9cPawNKNEtfH9nf0E/LvCCEUoMdOEJ64WBDnME7UqSS0zUDvROUAPWYSgw3DMLjXXXdd8L0I8jCgDBrTSphrDmY61pkccyda/DjnRTvtf41cyHl7Kfz2NkePF1YIbuHv3nTTTcEMh4GIpRB44l1pBIZi/UT6F41P0AxTGjBcdNFFwf2C+QjOYsajYQErlgSZYoydJ+sQn2G80fb+PHI10MaAlPl9AEXd0JSAjXOpHzwCT/DJMSxDStRSgF8QEHwCMoQIFgDCB7Oa+EMsX/DbwQrxm4LPjlvBAiTu6c9A4HCce/n59E2UmBkhyBitM8AnjhGd6twSMVZ+fz4RJsQYaBttJMMRHBUWfC0ubXPQ0xikKJFRD3QAeoJDNArmYUDpMExKMse4BioO6EvbEX9Vol2bAj0lerywgrZhzh2TGQELEBkDtBg+N3PMRKjJmnNAcF+izjAZLgDAZH4YMxwQEBgkYEesANcNRuRaruE+LIUG+FgBjCvmOT4+0Ww0F8eIjJOwQ/INGh0ecJDDJ9QP64BzPe4D8zsgEQBcw7me/ceLR9gFicShKOC8D/l0om/QphyjjWwoSn1dUHFfhBsvpERw+jUUJ77jtnr8w6+lboCeKbeiEPfxuvl3MIN1TLtpJzkD3A+F6BRtT3Fom4OezkaSItWpvIOeve/HjRsXNADSlb9hTpIO6czHp3f0zgx6SrQvKM7E0WOFFTQGhcAWJifay4FDv7OIA7eK+WCW2UZNYu9Tf5YT5wBwIueQj5fXB0HDVJMHtwAuWhjQA1AsDiwEBAQCByuC8adOgA1XBOuC87EY8NHJ8YBXiOrDP2hlXAmy59h5mVwC+GJLRB29UG/qwFQjwMVVgD8RPrgk7AFBMM3v6+2E+I7QIOkIYFJ32sC1CCmUXKx5vykqrM+ZtqZt3BvrKLpYzcnPLS5tc9BD+ERMpXjnMOgkGuBrEnHF/GOwHfTeGBrpzBsFPMWP+wCWtAN2BKJt0b7wUtix2EJfcT3MTVQcE9sB5oUoNeDHlMdMR8twnT87Wpz47udAaHs0JlqW5B78asbUNR9AZX6fPHUCcJjFZNnxPM513gB4+PNYFuxvACBxA3EhsC4I8BELQqsDdoJlHpCEvI580v7od/8b3xFatBNNjtviz6cAMoQMwU4AyXXuckL0OzEnzH4EFEDnOoQbn8Q02AoboVAUKswNINbBTAVjRZwiOk1JPbwuJaHtAno6GIby6C4MgI+Jn4I/hMbBz0OyMxUT7dwo80aLH/cOKE0n7AjkfQHzUbz9WypO9BnaA+F76KGHBlMWrcZ48EmwDB+dqDgmK2Y/03poUAQGgtuzJP1+mJv8HfeBnAyAzDiz0wv3BwyY5NyfqTTcOfLs8fsRQAgHnsH5PN99diLWAJ3pQ8AV5Qcfa757n0Aco37UK0rRPiB2gavCc12AMEXnCTnUF8Bj1jP3TiISxD0Av9+L53IvMkeJJVBvBz4Fq4Z8huizN0dRTe+E8qMeWEqA3rea87Z7n5SEtgvo6SA6EY3u5j3mHQMO6DGH0AikZ8ZBXzh5Xzjoiwp+zvG+QaOQTw/oMOkBIhrOwQbjMv2GCY2/T5IMQoIttkhSwQWAmXHLACW+PWnVmLJE4gnywaDO/NyP+zO+TMWhvQEKmp1z+WSaDH+avHeuQUhQH0x6tDwmsmvbzRH9Q1uj/cR3CIGFwKPeuB0oF9pO8BCFA08CLiweAIsrCsj8ep7N9+hvAp0Ep2kvCssFKNYNU5S+MGZLxDmxgopjWCC4NtwbgUgkf2vRdgE9Zg7mDgxF5+A/YeoRGAL0MAbHYQICed5ZfDrzxkFfMtBDfLqWJr5CYAorC6A64yN0o2AFeBwDsIwb5+GjI7AJCFLQyORXwPiMIfdx0KI9GWf8blw5THmeS8DNp24xXakDpvpVV131h3gDz+T+CBT+jpBBYAAmrAq0MP42hVkGlAW8g4aEr7A6mPtHmyO4mC1CiHFPLE7XzA5W2kBAkdexMd1HX22Kp7BcqQdTle7CcB8Kriv5BbS3KOSCJEo8F7cFJUh/oixj95eESsrz2wX0MBzLZZFYdDbMQKCGKR46zv0pfLmopufTmTcO+pKBnn4h4EY+OsLXzVT8YLLYADVaDqA6ABwM/psxi37nfJjd58c5DuMDdDQobgIMizVBNJ0AIc+HD3Dr4AOu43r83759+4YMPObnAV8UlNSPdFh8bGZ3EA5YiQgDotvECK6++uqgvQmqkRJMsJBAH9cxl4+QiY1j8HznOz6xbphxQMN7Rij967wVBSeChtkQ+sDvR/u5D9qZWEB0MdPmKMq30e+4TQhL+ohZDARd1MWAChMYRaHtAno6kf3ZMJ9gHvw9ZwgYhEbROFY5RTW9dz6/aSyDAOCjDE/DvXBetOP+14i20U7vC++PqBD04sfdJ/b+8XNhKkxZQEUAykENcPH5Mb8BC5l0gAfNjlZ3JgdE7hZgmjMF56vy8Jnxx9F2PN/rzHfGHaD7fZjOA7RMSVGImPNM6uXChu+AygUTmo97UOAprAIEBXVCIMFLLjQofOd6/03hHlgWKCCmE9HwzCRgCVFXJ/oMYeV+N3/D7cDf9nvyyTOwgrAoiBvg0tLuohD39LHkGgpBRqZEaRt4IVZCPfz80vD5dtP0gB4JTyfBWORwY3559B7JS+oi2t8b5NFPfkc7xhmeY95JFM4rTWf81Ym20U7vC+8PB3K0+HGY1c+lADyOE3lGO3rEmYKmwvIi956MPMYMXxzzGD+X2RVPiXXAAzqSqtDWmKBE70lbJciHhUEEnoJJTN2Zv0aYOCgBK0DBXOc8ovRoXDQ2pjiaH9ADZOoKr+BGEHCjLhyPWil8urDgN3zF9ZwLyIm2I9BoC64G5j/PpO4Anj5ycl5ywQVRR9YKEMDjGW7pwNtE29HyBDah6L02R867/gnRfwhP+oe24g4TUIW8LiWlbQ56GgHjwUBUng5iAIjYAno6CmmGtGQ6Ce3vDfepDH47o8OwzsAcowO8cJ5f+79I3j7aGu0PioPdix+HYf2Y9xkFBgfQgIoxoTAOWFtktJF0A2jJ2IOJAbMvIeVcGJ4CkHATWJIb7X+ejRlMsI+pNQJT3M/n711owBOY7KT+Qow51+G/Ay7cBdfmWBRE23EDSPJBKGCBYMJT+M45mPdYMdQVPx5Xg2ewMAaXgDrhRgJ0AnK4OiimwsDkfQ0BRGIF1MHB7kIFs56pRLQ8fV5UivZZ9DvPZJrO4yS4vvj5W4O2C+hhAI/e01EEhUg8QKszP8ygIp2JGJPx5Z3sncc9OBYH/R9BT3Fw++/CjjtxnRNMzlwwWg9AMS6YxzAvmh3zP5YIcGEKO+jRsmgixhLQ8jyIZwJctD+CnHti1gN8ND2WnoMGnuBvztA+5gCMjD14AsFCsgxpwwgQCuYuMwpujVCYZWDRDjMCnIPwIq8fjclUIRF8hBn+NvwD8RntF+9f5yNvE4RgI5YA/1J3t1Z8sRj3d5eU64rLi5wfHS8EiAc9EXQIqigV9/5O2w30DIhrenwx/D+YjogpTAATIZmZEvKB94HxzuA39+KTwjEGyAvnlbQjdhTyNnqb6Q8KfREtftz7BeK7E/1HNB0tjYZkXCiMBX61x1a8QCyGInvOzwW4mNwIbgDNcyEsO6aYADOaCq2Oxib4xTEY2f1gAEMwzoWM34O6ExOAJ7AMuYYXiDLliOWBe4LfiyvB/gt855M2Mc2H0ACA3MfrD/l3Pp2/Nkd+Pvcjmcwj6rTfA4HEBVjnj/nt50f7vajENbTfxwnBCdh5Bu4EAsyJexf3/k7bxaenc4neuylJAxhAUgvxK90/wjRjLtgDFt54GkdncB8GkU+Kd5CX0nTEjkTeTtrs4KYvosWPR/uL4n3px/HdCdy51iIQhmlM8oqT9y2gwg/2qDXXoKmJA2COO3EuoCRA5wE/xhhznTls116AB0CzAQd5/tTHr6fg56Ph4Q0Axmo2FAXxgShxbpQHolTYsSj53zd3Di4HvIp7QEyA+tAmzHrawiIfgpDEoLgf53v/FoU29WziB+CDvkJhMm3pxBhurs6bo+0CeirI22bQIjQATQ/oMbnoMMw3GIhpE0wzpGqUaBz3oBOjTMwxH+yiDN7/Cnk7KQ5s+iJa/LgX9+3pJ/rOf2NCE9TCTIeZGR+AyNJbF74Qz0KDEuTDHfPINaBGG5FoBTmj+/ZTuACch8Yn4Ie5z/05BujR9EzfEu12om48D5cB0PMs6sYsAnP2JNtAtGVzxD2c/DufXjZF/nfvK9a140ZgddAO6k59aBurBQkGIuS4xqfVtlS3KMXWxX9jvbCen+cgYEiq8vs6/5eEthvo8b+YeqDDAD8aA4a7+OKLw1QHoGdulUGFuaIdwXfuQUNhYD690XSCF86L7cD/VfK2Oqjpi2iJPU5/+XXRv2OSMu+NJnZzlcLiHMzoKPOiyZhP51w3cWFIptgwfaPEtfjUbhXgBgAQkrHc4mPMEfgIB4AcHTuexfw+yS5YEzyHwB3WAud5vbwfCiM/Hvv3wq6PPceJehATID3Z+8eFI/XHXSEgiID0fvZ7beqesRR7ntcN1wWBgqBBKLPhCM+AGDsf0+LSdvHpaQTmH3ugIbUxVYhG4teTdOFMgFRHOMQGkbiHd6g3lsIx/lbUzv1fJPqWfvD+8T7y4sc2VQAnWppZFMDF+KCJSYMl0h21uhDGmLn8zaf6YHyEOTEbJ+qCuU52HOYvgAUo3JcMPcbfgcNU15VXXhniBZCPJZ8E3QjkeeCMfA4W6+C3OzBKQ7F8E3tPzHT6Br7FoqHNuD/0EUBESRFco560OUrci/4tLnkdqBuCkDUK3n6sL1/ZWBqe3y6aHuZDWtJ5VB5/CKkNA2DeYbo4sxEFJUU06rfRQGfqWNDTSaXpgB2daL+XwoDvvzdViGQTlfbNKx2MgJUAHRF7zvE+xoxFW7s7QEEbM10VZXzGj4g8U2sAhvO4J7kYWApusgMexpwpQcifwyeCgHiDJwUhlBz0RR1z54/Y4kQfRInzIaYqcT9xR7BGeD4CjnrzHVeUuXMUVFQwRp/h9yoORa9ByBLY9vaT688UY2lpu4Ae04dcZSKf3mnOXF7oUCQp/j6mJQzjHUAHbg70lOhAOhV27H+NogxGiQK/KAVCo8BcZNX59B3jA8gwa9F2gADC3yYY5xtccC7Ra9ZWMM7e55xH9B2BgFB34YB1x8YpaH/GHM1JXAe/GfLr+eR6phQ9FuSgjy6l3RJF+SNaCiMXWtyffBFcT88n8E8KQot0YrIO4Ucn73+//6aeszmKXkOcA5PeQU/b6ZOS3DdK2wX0mEnkDjOfi9T3zoMRvDPdbIJByMbC33eikZsCPcUHNUr8Lm3n7Ejk7XXG8+J9tanCtBbnobEIGmF9ub/uQGValcgx59D/zJcDRB9LtDnz5p56Sj0QAOSxcz2+MAAngYbMPqL9CAyOAX4Cib6MNTpmaH/X9JyLoCGC7lukl5S8r/xZ1NkLwTNfCej707lwg0c5xvp7ovVYHE7cy3nRKdqWLZGfG72G/iR4R/sZC9qOoGHcSkPbxaenM2AA5oQxC+lANAoBIBpEZ1I8WMN5NM7J70FjY0Hvn85sTnynRAe0sPK/RN7maLu9rzZX6D+0PczOjkZktBGAA2gwG2Bj6oggK/Pk/sJGxtHHkmNRfxOXgLl+LDdAg0AnH4AkLTLa3M/nb1gO0fXiTrgduBK4fdQFhYBJ7VZHUSjaJ16cb3wGg2MQ7gQJRdTPpxUp7opg5pPZx7ShCx6u9b72+zjF/i4K+TV8Yt4TDGW2BOVIEDO6nqGktF1AD9GhLDnERKMDSdYg8omf7z4TAgE/ks0WosEa7uEDVRTQ80nhGH/bXPFzN1V2RIrW3/tqU8WJviA1l4xIzEiA78lUAA7gYcaj0Vn9xXf/G8D1rDwfM8aPnHbuAcMCHDQ9goMgrrt5DnpfLx7tc6L35Ax44A/FgFCCl4o6NtTHSyz5MXgKEJM3j2URBTzCiU+EHKv4iE3RT1Bh9ywuxbbDfzMeRO/Zr4C4B/3E5ib49DsM6GkAqZKAHkZBgpOwQ3CCgaVjieQS1GFjRRrmxD3oBJg0FvT+m07wZ/FJ4RjnlKbs6OR9tanizEN/McdM1B2tS4488+8AwF0wwEdGHgtk8Ms55kE6THbiAowzBOgxTbEYcBcYc6ZmSY9lsRXXUvgbVkQ0r9zHkXqQsIWG51wEBQk/WIHFZfro+d5WAnC4FbzkAi3K4i9mImiP1w3XhHojqDiPKc7Ye3l9If4W/b0lim2HXwv/k5yDK4XgoR+ZIcGaYtxKQ9vFp6dhdDARXk9uYM6W7DvSF2EuGAtNj/aP7gcG0REAkMZuCvR89wGg8EyKg3dTxe+1qbKjU2Ftihb6CP+baDu/6TvmptnlBqGMQGZKFROfsSOwh/Yh7oLmdWsAXxcBDojoV4JhBKEYV9eWnMNiF4Jg/KYgDAjw4jY4USfqwTFcgmgcCIHhzygKcR4xJdqHC0O90Ja4DggaeJB7EsvweXja5YIKd5OkIvIQACH8Rt34hACnH4N4HvWH/HNzFD2He/h9EEpYNMQPEEL0Aa8Nox1+jZ9bXNrmoHeN7au68E9oBKAnMEE2FsEeOpr5SKZIeL2RkzeMTzrX78dvBpBAFEwLw0J0Ouc4I9N5/jeIv3MMxo5dr+/3L2wQ+c0n9+LvfEb/5sT36O8dgbzO3j4HCVqFvfIYIyLsaHjGhqAswCWKDTDQwAgG5vpZOYm2RyOygw2g98AgbhwLYpidYay5FqAxF088ged5XSgIf+IFaDl4BhMbpYDZz/0BL1OI5Brg5/ObAjiZOwc0CA7WBeBqAHDyQOBDlu/ikqDFua8nEVEnTGniTfj2pCMTtOMZ9IvXbVuR3xs+ZEEaQo+2oxARtuTjw39QSeuxzUEPMCCkLJrDA0B0Kll5TOFgQtIwGADNgjYAmDTcCwzJMYIb3miPPPN3BoTgEYPOYBPwYJUSggEG9muiRKdx3yjgXSC4ZuA7AsTrw3neJigW9DsiUX8vtIc20masM0AF+Nhggk+EN30KGLDQ3DcH2CSrEJ3HReBcsslcywN+XABmAXAd4AP+xnXEBwjwoX15PkT/4y54EIt7IGTI0wcMJP5wfxQHMQjyDNhrj09+A2qEFVqauBGCBd5i8wt4DzMeVxPXgfvTDrdcaAfBZPbUg5/gq+iYb8vx9nvTfgQVfQM2sLSIpyDcXImVtB7bzacHoERG0eoMINl3ZGIhuYnwcgwpS4cjjTk/CnyIewFCZwxAie/IwCD9iRYT7WRpKDniMAhTf2SRYRIi/dFCMLMLAQbT7+dEpzrwo8R5ruG9XXxy3EtJB+LPpmi9+U4b6Xc/7owG8R1/m/UTAMZBQ0yG/HSyyNj1hWkvxhSmBUwE7Bgn/ob/zPlciwBgrzzGyYUv2hqLAMHA9fAHlgL5AATcUBYEBrmW6UMK3ylYEQQeKQDZBZPzmNfJj/EM6kJyEtOTgIu4EsIrdjy39fhyfwqWDOtQUIjUF+GHlRQb4C4JbRefnsoBYAI7HgAC6IAeyczgMQgUBhaLANB74/wTEGJmwRBoc7QJGodYAQAnukrgB43A/SkEfsgtR7vQaZxPhJqgDJFYosbMP2NauiDYFNEOGN4F0abO94Hzev/VyetLiRK/oxrOieOMA6Y+WtMTegAXfY5mJTjHuDqwYF4Ai/WFL03AFoC6wMCMRXMzT45mJR7gy0qxBnAf8G/Rvgh2hAsmuV8PMPxZFHgpCu7obwpAR8sDJsDOcmL4A1OeRCF4LdovlCjxu7C+KS35c8hRgGcRltSbeAibmKCM/JzYOhWVtlsgD4CQXouZRacjmYlMEpxgkwX3+2AapibQ4nQqZibmJOY6fh/nE2lFCxBJ5lqCMARcYCICHlEmgDFgSoQJ5/F8pgWR6Jh+mHHkNONSYBXQ2ZiwCB2AjTvB96im8wGnTQgz/hYdAL7T5h0J9Jurrx+PthFQ+GId+tWDbYwt/ieF74wFGhctj4tAwgnaCrMcgcE1ABfmJkkHgUzQjPMZT65n/LAM8MnRwAhr33yF5+KDcx73ojD28BPH+c3f4ANAjgvpYAdIuJisoGPunfYAKgfz9hy/aN9CxCKoG/WlH+krhGxU0MReU1TabpoegBBQIfeagcAEI9EByc1uLB45ZYAxzfFdABtzv2xECHMRWAHcCAz8Mc5lQOkULz7w0cJxGMElPdfAEDAM98EyYINOdlvFYiDQwzpzzEhyzzFJqQ8CIGryQrQrCgoXBgiC7ck0pSGvN/WNts2/8+klVsDxhhoi+rhrjA0AJ8oPsOhfgEWkH//czWUEOsAmxTY6PtwDK8EX/yAMUAJcjzXHajb6lik7XEWm0ciBR1Mzhm7i46/jRlIfPrH+WCTE+WzTBpjIPmTmCJcPsEfH1T+9P/j07162NsXek4U8KCbcE/oI5Yb76lSaOmwX0EMwC1Ia7UojYA5ATyTY0zo5joYgAEOUEjMP7Y5GZyBdcju4XZtT3FyD4Xx6CUHiwgHzEG3CMZjJr/OCIODZMBmMRMCHwA/JIGglrA92fkETsY4AtwBTFVcDzUUg0ZmDtu5ooI8ythPfY5nLz4UQFLhZ9Ad+MH1ERBzBzl57CAOi/GgoxpM+gbgeIYqGBZSMK5qZcWG8GD/GhHEj/51krWgmGnUgjoMQ4B6Al+W5Hshj6hDrjVWC7tKxfgBeQ2AAciyOaH2iFG1jYbSlv5eEvJ+5L/1Km/0NN/QFgo9pVKeoxi8ubVfQE/wBwIAWiQzY8afR7ACNwQesaFt8RjQroEfDO2MAzqh2xxfEvCSTypkN3w+NwXV8999Ie/xENAA+JiYe94oKDy9+f/5Gx3MudWQpMNqCCDFMjn/J4gz8QOq7tZlhexFM54wXS96maNvQuE60m/HCDQNUCHcsACwk/HeOQzBq9DkEVxkTsv8QuIDeAY+VgEmLkMXl4hoHKd/9+RzDJOcZCAIK7hnBWoQEgdtofChKXg8HULRN/jeIa6PX8zd+R88pKcXeg7qgQHBlsF4c9PA1/cr5FO+LklCpQU8FqCi+d2zH+mAwCHQ+fgrBO4AEiDCxAD07jBLpZeAxB5HYTN9wLYLC87XZqoj5fYCH5MMtQMJjhtMhpFLCgJiPBOb45Lk+bwxDEqjhfKaW8OdZWYZGZ9YA85J6eYAIIePgjwoBjruPSJ2JF2DBEGiB0WG22EGJHVx+F8aIOxsBaLL0MPXpSyw+xoNUX+b5i0r0JXxIv/NJ2VH7F6uGKUcsHZQdvIiyhH+dStO2raLpXfpASClMPtYiowHxjfGRMcEAM+BFmjNny98A5FdffRUGGZOfQafBMAPSmkAZZjTBGyLuBIMw0/C1MbG5Pupr++B7ffjt9eM79+O+CBQixZimxBUwA1k0glBCqsKEpArjLyKIGACkLsB3a8AtBAYGv5HpRw9WQTAgz/Q6eV3i9F+ij/CrSephqhaBjHZn+g7BXVTyPv5fAD28j4uLq4PbijVKchQKxak0fFRq0DtTUwk0O34WUy/4wqRXMkUHaDCnWWTjZjjZRYCWa9g6mDlJAI/GZQ6WoI8PIGYXwgRrAnDzm8LfHMzR4h2yqY7huBeI5xA/wM8kcAhwsT5IJME3xP0A0EwJYmmg2RFcaHwK4CcwieWBO4KP73X3z03VZWcn+oX+gQ8AOYWZAcabvisqMe7R/qZwbEci5xGUHKsJcXXR8vAXsxm4Q06l4adSg947GA0KeAmekEqLz45mpNJoQ6QV0y4AiCAF4IKoPJoabYuQQGNuat91zo0ObnRQ/W9e+F1YiSVnjug13h58QVwGLBcsCywXfHgEAbMJBCKxBvBLifzTdkww7sH9ovePkv899nic/ts3xSWucb7gs7B+/6sTbYeioHdrEvfXVyKWlraKpofQwszDA1wPxlDIbSZQgxbEj8avj810Q2sjMJiWIVKOeYfZDtERzgjeKVHGiGrSaCmMNve3LRHXUU9cAywCpC5uBpodF4GYAsE84ggQ9ducpuLv1DtOBRQ7vsUdJ+/PHRn0Tpj3gB4cuTvJ9B3Y2Rq01UCP2e3plVQSCYUZzAomAnEEZTD9GUy/BgDxnYIgAPj490SCHRD+9yhA+L05QEFFOQcqjLk4xrW0ycnr7Z/OYO5yRIm/0zaCh8QNKJj8mLBR4rydnaL9X5r+iI4Jn5QdtX8d9Gh6DyCjNLE2twZtNdCjBVmRReSVShKEIMmCiDbmMYPg4HAwYkJHid+AiHtB3LswUMYS52xqoPlbYffgmJ+7qXOgotQh9rm4K8QFSCAhiMk6aOb4sYRwa4gfxNZzZ6Vo39In3i980q9FJT//fwH0mPe4j8xmoDzBE1POWJNbg0oNeic6G01PZhv+O5F45sdh/ih5RDY6uBCDFAsu/hY9Fvs99u9bIs71EkvRY9x3U/eOvYczpgsqPpkZIG+a/AHcG1ZKkR/AtAs7xzB4sS7Ozkr0o/MAFPu7qMQ1OzronaeYriapiUQyzyMh+M2MxtagUoPeAYB2Y6GM73TKnDrZWYDcO9/PdeI7gwNQHDz+d7cGdjSi3mSfsUaAfALm8j2+QcIJ/YIFgMm/ozFlnLYtOc+T0EQCGPxCIJxPYl3u07uCiWKpOLTVND1zrf72ExictFmCeszX46NHTXmY3Rk+FtxRAbGjgIJ6Ul+vM/P0DBxrBghismyTPmE+n2xA8hCYkoxTnCAHr/M76bbMBsEv8A2zYFiOPk9fWlxsNdDD5CQUYNa7T8+cPBKK99ixgADfPhrMiq08jcfv907YUSgW9BCWC9YP03uY9j71wnw+y04JWu5o7YzTtqEoDvjOOgHMeUx7eAbfnqxFUoyh0vLNVgnkkUiBdCIhB9BTUWdwph3wbfFnmX9n8QVz2cyBR8nN+x2NHOxO0QGEmLFAaruZT2YfA0o6cBz0cYKifACW2HfCl6CjLLAUmf2KYiaW74pDW0XTY6ITfMAEwZwlS82BT8X5xFTB3yfNlew7NsBAqwMSAM89YhsRC6AdjWgT202xsMj7hHgHmYksqIid6otTnAjwEvNhFswxxApRgr9bCw9bRdMDWqahWCNNDj3mLGm3nqYKo3sDCEowlcdGGEzn4e9zrd+LsiNpQG9/rKXCcUBPUI8NP9D0BGUQgiwnZbERUj1OcXKC70nuwh3GOnbMsAgMd9DPKS1tFdA7EZxi9xkkFYtXSFPFlGXRSnSxCsxPoI88dpanssiCmAAA8cUqOyL5oJGZR5IRu7GwkIe0ZISdDyLBTmY2PHsvTjsvRUHs/MO0roMev56Udn/XnytFShR7xaGtFsiLTiMQrMNvJ3JP6i2bHmLS4t97QMs/mcfG7GdBC9NcTEtwfWxU/69K1BWTjLl5XBz8MbaKZrqO3U4QeB7nQOgBfhYWof2Ls4osTv+bFLUQwQ5WL5qeDTTgGaxkrGZfVss5paWtBvpNEZKL9EGW1rJogFV0mLe+goiGkYTASjzcAjQgO50g2TxwgUSLlWpbajx/d0G0OeI87o2Qwcf2+AKdjz/OJym0WDEsAsIiIdee7Dq2KMZKIT+BaThW4flGkb4kkvb54LErD8LgiSeeCIsnSNWNU5yc4EV4niXkJ510UuAblCPKgzUeHgMrLW1z0AMiGsIcI2vi0eZsaUSjWFFHo3w1Hn7vbrvtFub32Tl1UxFujsUed/BS/Hth1gLH6TzSfXElOJfvbPRBrgESFSFFJ5NWzJQbG3UA6quvvjpE4tkTANcECwX3halJkm7Q6FGg0y7ahE+G5GbHVdwZ9gr0OEacdm6K5WV4kyxW+MzdYXaHIo0byxBFFntNcWmbg94rB/gBHFqT3GKW0rIQB+3O3nXeQEDDHD8gYdFBlGIbym/uuTkiWIZGBdTkCfiLGwD1hAkTglXB5g3s8cbeamzVRSwCrU3AEWAjiPzlCMxCIKiorxcHuRfaANC5jqW37KpD4hKBTlwAz1UozcDF6X+DXElFifUZbN7pOzhhIcKX7MfoVBTe3xRtc9A7xZramP0E7tg9B20JQDDxaSimMNINvxeNGL2WxroAicYR+O7gpnMQLEQ8ES68fAE/iSlFtsjChUCSAmx2yGG9ADEHsuUIsrl5DoABNSAHyMxEAHoKWpzisxOu0bkHJjyCAwuB+VViG9SHusUj9nGKpVjQs3Sb2S3m5+E9CsFgcj62Bm030Dt54MIlFX4zZj+N5I0pvASBxASW5GJeY3pzjW9qgYlDZhJaG7+Y2QKy/VxrE/nEhMaFIFno+OOPDxoby4FYAh0JWOlIQO3ARaJGTXPADpj5RBCRaASgiaoyh8rLFvgkaEe9sUrYDoyNNHhRA2vtiQEQ4PPApA+utz12sOO0c5NrbtZlsLMUPOc8icUJn8cqz5LQNgd9UUwQNDRaH21Iw3jxBP48vwE8wAfkZP0RPGOzDaYEifijtXlJAuDGzyaQhsWAy+AaGfDSebGmOL9dk7sQoCAE6HCsD/bLY6sitvBCMLHlMu4A05LsooPQIQaAScY6aAJ+br7HKU7FIccKSo1ZL6L2rnTgQ6aB/RwURkmVxnbR9K7ZYgUAv/1vTu6De+QeDQ+Y/L3maFj8a3auxRRHczMTEOtrR8HNcYDv++BjhjNthq/k3xEUTB9i5rPTLj4VFgORepY0EuBjUwzcDeoGsCnUL2q9+EDw6cc3RX5unHZeivKAfydhDVeU6V6UFVqemBOB5q1B20XTxzJ3LPghAOLHo9/RnOyUSu4+O/GghQGyT/fFFv7OXDiCgIJ/TlQdIYEV4FtosykBWxCRQ4C1gCZn4wKi9f5yhSjRBuoU2x6+b+43FG2v3yNOcYol5xtcXhQOKe24pKSuo+VRMJwTy4PFpe3u0xeX8G8w9wErmhkgA2wkINob8FMwgaK+N/kAWAWsS8bnBsxMexDYw3VAezPnToyAoCLWBW6Ez9WXtmPjFKfikIMZggdxbVltB+8TkI4NAHN+SfnzLw965tIx74nys0yX3XgIoOFvA2wCawgEfHqSfwisISVZ1ENwj2Aa1gJZc0hKAmqubfkO0DlOR/vf4hSnP4OiICZgx9JsZnzcTXS+LS395UEPAUg0Mj40mXrsR0+SD9lwmP5MixEt529Eyznfo/7eYU4uUSmxf4tTnLY3RTU2n/AkgHdwRwVBFPB8j/6tOPSXB320oZCb31gABNQ21fDCAM25mzrf/8bzoiVOcdqWVBhPwrvOv3w6X0bPKw1v7hCavjAA+u/YzoD4HZ0XL4z4mxe/f2ElTnH6M2lb8OBfHvRR4MZKQHxxp+jfohQFd2G0pb/HKU7biwA4POy8yG9Mff8dy6clFQg7lHkfC0xAH9twOo2O4jP6N757p8XeJ05x+rOoMEW1KYKHY/m9JLRDaHqKN5bv+PWx5KCOkv8urLP8b7HXQJzLYOAixClOfybBn4XxbmF8W1TaIXz6OMUpTluP4qCPU5x2MoqDPk5x2skoDvo4xWknozjo4xSnnYzioI9TnHYyioM+TnHaySgO+jjFaSejOOjjFKedjOKgj1OcdjKKgz5OcdrJKA76OMVpJ6M46OMUp52M4qCPU5x2MoqDPk5x2skoDvo4xWknozjo4xSnnYzioI9TnHYyioM+TnHaySgO+jjFaSejOOjjFKedjOKgj1OcdjKKgz5OcdrJKA76OMVpJ6M46OMUp52M4qCPU5x2MoqDPk5x2skoDvo4xWknozjo4xSnnYzioI9TnHYyioM+TnHaySgO+jjFaSejOOjjFKedjOKgj1OcdjKKgz5OcdrJKA76OMVpJ6M46OMUp52KpP8DE4/OHp8QbOkAAAAASUVORK5CYII="
icon10 = "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"
icon11 = "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"
icon12 = "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"
icon13 = "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"
icon14 = "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"
gicon = "iVBORw0KGgoAAAANSUhEUgAABHEAAARxCAMAAACRNutzAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAKIUExURQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAH/WAsUAAADXdFJOUwACAwQGBwgJCgwODxAREhMUFRgZGhscHh8gISIjJCUmKCkqLC0uLzAxMjM0NTY3ODk6Ozw9Pj9AQkNERUZISUpLTE1QUVJTVFVWV1hZWltcXV5fYGFiY2RlZmdoamtsbW5vcHJzdHZ4eXp7fH2AgYKDhIaHiImKi46PkJKTlpeYmZucnZ+goqOkpaanqqusrq+wsrO0tba3uLm6u72+v8DBwsPExcbHyMnLzc7P0NHS09TV1tfY2tvc3d7f4OHi4+Xm5+jp6+zt7u/w8fLz9Pb3+Pr7/P3+G4cLtwAAAAlwSFlzAAAywAAAMsABKGRa2wAAExxJREFUeF7t3f+3VWWBx3HqBnXN1MyctAwaSiRLKGDMKInIIcXSwi85lKSEVKNpmZlhpFlhiXkLQw095tWgnNvAlNOdOhmTJglIMaH178wPflwr16IfTO45z/Ps1+s/+Dxrve/ad5+z95kGAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAVGZi9be/Ntvd5kf89fgRduT7/f693+jXXLZ89IVRzarMvHJnNowIvXH7t8Turi+Y5ZddtjOSXg8NkzdunxqYw47qMbH8/xAIfb3k0fOyGtMW3anA1P5GCAqbH/5nnpres+cOuBnAkwdZ7ZtOJlia67Rs7Z9EzOA5him1cekfI6anYvJwEMwsSCtNdFb7ru9zkGYDCeWn9K+uuaGau25wyAwZm4rJP/Ws25JfuBwbr1tFTYIct/lPHAoG3/cDrsitFP/y7TgcH7w1VHpcVOGL0ru4Hh2DqaGjvglNszGhiWOzvzOfkiVzgwfA8sSZGNWzKewcAw/WxFmmzaLE9tQhn2d+Afq0WucKAUO96XLps11z0cKMeDC1Nmo2b6lApKsuXktNkmVzhQlt5I4mzRZzMSKMXnU2eDzvEuYyjNvpXpszlzH8xEoBwPvzOFtsZNHCjReJu3clZnHlCWK9JoU05+OOuAsjwyP5W25CsZB5TmplTakLP2ZhtQmj+fl07bMZFpQHkmW7t5fEGGASW6NKU2Yvrm7AJKdN+r0mobzs0soEwXpdU2eGQcyrZlRmJtwYKMAkq1NLW24GvZBJRqY2ptwBt+lU1AqR5/S3qt36pMAsq1Nr3W744sAsp1T3qt3vz9WQSU6y/vTrG1W5dBQMmuSbG182UcqMHdKbZyr+5nD1Cy3Sem2bqdmTlA2c5Os3W7MmuAsn0pzdatlzVA2SbSbN1+nTVA2XYfkWhrdmLGAKVr4UfIz8gWoHTLUm3NLs4WoHRrUm3NvpAtQOluTLU1G8sWoHRbU23N7skWoHTbU23NHsoWoHQ/T7U125EtQOkeTbU18xwn1OLJVFuz3dkClO5gAz8hkylA+VJtzVzjQC2efkWyrdhvsgUo3d5UW7Od2QKUbleqrdmPswUo3S9Sbc1+mC1A6X6SamvmuSqoRS/V1uyL2QKUbkOqrdkl2QKU7opUW7PF2QKUbnmqrdlJ2QKUbm6qrdlLfAUQ6vDkkam2auNZA5Stjd+r+lzWAGX7cpqt29KsAcr2oTRbt2N3ZQ5Qsn0z02zl7sgeoGT3ptjarcseoGTXp9jaLfy/DAIK9v4UW73NGQSU64EGXnL8rMuyCCjXlem1fv/8v5kElGrfaem1Ad/MJqBU302tLfAlQCjditTagul3ZhRQpvuPTq1NuCCrgDJ9Mq224QjvV4eSbX9tWm3Ev2UXUKIW3jf6t0YeyTCgPI818+2/55zrUQco1qp02pCbMg0ozbenJ9OGzP/vjAPKsmtxKm2Kl1ZAma5Jo20Z+WnmASXZ2dxt42ed/p8ZCJRj8swU2pyL/5SJQDFWp88GXZeJQCnWp84WuZUDhdk5mjqbdOo9mQmUYHxh2mzUIr8JDOWYaPau8XPe+3CmAsM2+cF02bAF+zMWGK6DS1Nl0/71PzIXGKbJC9Nk4858MIOB4Zk4O0U271+8ERCGbXxJeuyAYycyGhiOncemxk74p+u9oAuGaP0b02JXXPJfWQ4M2i8bfpbq73nXWMYDg7Wl+e/9Hcrx//4/2Q8MzqNXvz4Nds3CW57OGQAD8q13pb8uOv/enAIwCOMXvTTxddNxq+/PSQBT7aE1r0t53XX0Jd5gAYNw3ydek+o6btldB3MkwBTZumIkwTHtlCu25liAw6/3mbenNWLOtdtc6cAU2HbtqamM5xl97w0T/urA4XNw4oZlR6YvDumkpZffuHVbv+9PD/yjDvb727ZuWLt0lls3AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAhRiZvWztzbf1epP9PX8FXrg9/X6vd/s31i2fPSNVcWizLh+bzKEBL15/7PI5qYvnO2bVbY/llIDDZ8/YpcenMuK4j258PMcDHG57N33shLTGtGlzNjyRgwGmxv6b56W3rvvArQdyJsDUeWbTipcluu4aOWfTMzkPYIptXnlEyuuo2b2cBDAIEwvSXhe96brf5xiAwXhq/Snpr2tmrNqeMwAGZ+KyTv5rNeeW7AcG69bTUmGHLP9RxgODtv3D6bArRj/9u0wHBu8PVx2VFjth9K7sBoZj62hq7IBTbs9oYFju7Mzn5Itc4cDwPbAkRTZuyXgGA8P0sxVpsmmzPLUJZdjfgX+sFrnCgVLseF+6bNZc93CgHA8uTJmNmulTKijJlpPTZptc4UBZeiOJs0WfzUigFJ9PnQ06x7uMoTT7VqbP5sx9MBOBcjz8zhTaGjdxoETjbd7KWZ15QFmuSKNNOfnhrAPK8sj8VNqSr2QcUJqbUmlDztqbbUBp/nxeOm3HRKYB5Zls7ebxBRkGlOjSlNqI6ZuzCyjRfa9Kq204N7OAMl2UVtvgkXEo25YZibUFCzIKKNXS1NqCr2UTUKqNqbUBb/hVNgGlevwt6bV+qzIJKNfa9Fq/O7IIKNc96bV68/dnEVCuv7w7xdZuXQYBJbsmxdbOl3GgBnen2Mq9up89QMl2n5hm63Zm5gBlOzvN1u3KrAHK9qU0W7de1gBlm0izdft11gBl231Eoq3ZiRkDlK6FHyE/I1uA0i1LtTW7OFuA0q1JtTX7QrYApbsx1dZsLFuA0m1NtTW7J1uA0m1PtTV7KFuA0v081dZsR7YApXs01dbMc5xQiydTbc12ZwtQuoMN/IRMpgDlS7U1c40DtXj6Fcm2Yr/JFqB0e1NtzXZmC1C6Xam2Zj/OFqB0v0i1NfthtgCl+0mqrZnnqqAWvVRbsy9mC1C6Dam2ZpdkC1C6K1JtzRZnC1C65am2ZidlC1C6uam2Zi/xFUCow5NHptqqjWcNULY2fq/qc1kDlO3LabZuS7MGKNuH0mzdjt2VOUDJ9s1Ms5W7I3uAkt2bYmu3LnuAkl2fYmu38EAGAQV7f4qt3vczCCjXAw285PhZq7MIKNeV6bV+sx7NJKBU+05Lrw34ejYBpfpuam3B6dkElGpFam3BS7+XUUCZ7j86tTbhI1kFlOmTabUNo3dnFlCi7a9Nq43w7lEoWQvvG/1bI49kGFCex5r59t9zVvwp04DirEqnDflqpgGl+fb0ZNqQt/kBcijTrsWptCmfyjqgLNek0baMeMU6lGhnc7eNnzX/pxkIlGPyzBTanPP3ZCJQjNXps0FXZyJQivWps0UjvYwEyrBzNHU26c2bMxMowfjCtNmo+a5yoBwTzd41fs5iH1hBKSY/mC4bNm9/xgLDdXBpqmzaWa5yoASTF6bJxr3HvRwYvomzU2Tz3uETKxi28SXpsQNGXeXAcO08NjV2wjFXe+ABhmj9G9NiV5zv/jEMyy8bfpbq75n/nYwHBmtL89/7O5Sj1uzIfmBwfnv169Ng17x1/VM5A2BAvnVG+uuieRM5BWAQ+svSXke98sIf5CSAqfbQmtelvO56+XljOQ1gKt33idekuo47feOBHAkwRbauGElwTJv58bE/5lyAw673mbenNeKEdXd5jwVMgW3XnprKeJ6ReVf1nsghAS/ewYkblh2ZvjikY+atvHas1+v3XfHAP+hgv79t64a1S2e5dQMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAC8QCOzl629+bZeb7K/56/AC7en3+/1bv/GuuWzZ6QqDm3W5WOTOTTgxeuPXT4ndfF8x6y67bGcEnD47Bm79PhURhz30Y2P53iAw23vpo+dkNaYNm3OhidyMMDU2H/zvPTWdR+49UDOBJg6z2xa8bJE110j52x6JucBTLHNK49IeR01u5eTAAZhYkHa66I3Xff7HAMwGE+tPyX9dc2MVdtzBsDgTFzWyX+t5tyS/cBg3XpaKuyQ5T/KeGDQtn84HXbF6Kd/l+nA4P3hqqPSYieM3pXdwHBsHU2NHXDK7RkNDMudnfmcfJErHBi+B5akyMYtGc9gYJh+tiJNNm2WpzahDPs78I/VIlc4UIod70uXzZrrHg6U48GFKbNRM31KBSXZcnLabJMrHChLbyRxtuizGQmU4vOps0HneJcxlGbfyvTZnLkPZiJQjoffmUJb4yYOlGi8zVs5qzMPKMsVabQpJz+cdUBZHpmfSlvylYwDSnNTKm3IWXuzDSjNn89Lp+2YyDSgPJOt3Ty+IMOAEl2aUhsxfXN2ASW671VptQ3nZhZQpovSahs8Mg5l2zIjsbZgQUYBpVqaWlvwtWwCSrUxtTbgDb/KJqBUj78lvdZvVSYB5VqbXut3RxYB5bonvVZv/v4sAsr1l3en2NqtyyCgZNek2Nr5Mg7U4O4UW7lX97MHKNnuE9Ns3c7MHKBsZ6fZul2ZNUDZvpRm69bLGqBsE2m2br/OGqBsu49ItDU7MWOA0rXwI+RnZAtQumWptmYXZwtQujWptmZfyBagdDem2pqNZQtQuq2ptmb3ZAtQuu2ptmYPZQtQup+n2prtyBagdI+m2pp5jhNq8WSqrdnubAFKd7CBn5DJFKB8qbZmrnGgFk+/ItlW7DfZApRub6qt2c5sAUq3K9XW7MfZApTuF6m2Zj/MFqB0P0m1NfNcFdSil2pr9sVsAUq3IdXW7JJsAUp3Raqt2eJsAUq3PNXW7KRsAUo3N9XW7CW+Agh1ePLIVFu18awBytbG71V9LmuAsn05zdZtadYAZftQmq3bsbsyByjZvplptnJ3ZA9QsntTbO3WZQ9QsutTbO0WHsggoGDvT7HV+34GAeV6oIGXHD9rdRYB5boyvdZv1qOZBJRq32nptQFfzyagVN9NrS04PZuAUq1IrS146fcyCijT/Uen1iZ8JKuAMn0yrbZh9O7MAkq0/bVptRHePQola+F9o39r5JEMA8rzWDPf/nvOij9lGlCcVem0IV/NNKA0356eTBvyNj9ADmXatTiVNuVTWQeU5Zo02pYRr1iHEu1s7rbxs+b/NAOBckyemUKbc/6eTASKsTp9NujqTARKsT51tmikl5FAGXaOps4mvXlzZgIlGF+YNhs131UOlGOi2bvGz1nsAysoxeQH02XD5u3PWGC4Di5NlU07y1UOlGDywjTZuPe4lwPDN3F2imzeO3xiBcM2viQ9dsCoqxwYrp3HpsZOOOZqDzzAEK1/Y1rsivPdP4Zh+WXDz1L9PfO/k/HAYG1p/nt/h3LUmh3ZDwzOb69+fRrsmreufypnAAzIt85If100byKnAAxCf1na66hXXviDnAQw1R5a87qU110vP28spwFMpfs+8ZpU13GnbzyQIwGmyNYVIwmOaTM/PvbHnAtw2PU+8/a0Rpyw7i7vsYApsO3aU1MZzzMy76reEzkk4MU7OHHDsiPTF4d0zLyV1471ev2+Kx74Bx3s97dt3bB26Sy3bgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOisadP+H6qOB9MAN0sLAAAAAElFTkSuQmCC"

g.iconphoto(False, tk.PhotoImage(data=b64decode(icon)))
# g.wm_iconphoto(False, tk.PhotoImage(data=b64decode(icon)))

fr = tk.Frame(g)
fr.grid(row=0, column=0, padx=10, pady=10)
# info = tk.Label(g,text='                                   \n\n\n\n\n\n\n\n\n\n\n\n\n', font=("Arial", 14, "bold"), bg="white", fg="black",padx = 30,pady=30)
xscroll = tk.Scrollbar(fr, orient='horizontal')
xscroll.pack(side='bottom', fill='x')
yscroll = tk.Scrollbar(fr, orient='vertical')
yscroll.pack(side='right', fill='y')
info = tk.Text(fr, wrap='none', font=("Arial", 14, "bold"), bg="white", fg="black", state='disabled',
               height=10, width=30, padx=15, pady=15, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
info.pack()
xscroll.config(command=info.xview)
yscroll.config(command=info.yview)
try:
    info.config(state='normal')
    info.insert(tk.END, st+'\n')
    info.see(tk.END)
    info.config(height=len(key)+2, width=max(lst), state='disabled')
except:
    pass

img1 = Image.open(io.BytesIO(b64decode(icon1))).resize([250, 250])
tk_img1 = ImageTk.PhotoImage(img1)
img2 = Image.open(io.BytesIO(b64decode(icon2))).resize([250, 250])
tk_img2 = ImageTk.PhotoImage(img2)
img3 = Image.open(io.BytesIO(b64decode(icon3))).resize([250, 250])
tk_img3 = ImageTk.PhotoImage(img3)
img4 = Image.open(io.BytesIO(b64decode(icon4))).resize([250, 250])
tk_img4 = ImageTk.PhotoImage(img4)
img5 = Image.open(io.BytesIO(b64decode(icon5))).resize([250, 250])
tk_img5 = ImageTk.PhotoImage(img5)
img6 = Image.open(io.BytesIO(b64decode(icon6))).resize([250, 250])
tk_img6 = ImageTk.PhotoImage(img6)
img7 = Image.open(io.BytesIO(b64decode(icon7))).resize([250, 250])
tk_img7 = ImageTk.PhotoImage(img7)
img8 = Image.open(io.BytesIO(b64decode(icon8))).resize([250, 250])
tk_img8 = ImageTk.PhotoImage(img8)
img9 = Image.open(io.BytesIO(b64decode(icon9))).resize([250, 250])
tk_img9 = ImageTk.PhotoImage(img9)
img10 = Image.open(io.BytesIO(b64decode(icon10))).resize([250, 250])
tk_img10 = ImageTk.PhotoImage(img10)
img11 = Image.open(io.BytesIO(b64decode(icon11))).resize([250, 250])
tk_img11 = ImageTk.PhotoImage(img11)
img12 = Image.open(io.BytesIO(b64decode(icon12))).resize([250, 250])
tk_img12 = ImageTk.PhotoImage(img12)
img13 = Image.open(io.BytesIO(b64decode(icon13))).resize([250, 250])
tk_img13 = ImageTk.PhotoImage(img13)
img14 = Image.open(io.BytesIO(b64decode(icon14))).resize([250, 250])
tk_img14 = ImageTk.PhotoImage(img14)
img = [tk_img1, tk_img2, tk_img3, tk_img4, tk_img5, tk_img6, tk_img7,
       tk_img8, tk_img9, tk_img10, tk_img11, tk_img12, tk_img13, tk_img14]

mid = tk.Frame(g, bg='white')
mid.grid(row=0, column=1)

st = queue.Queue(maxsize=0)
state = tk.Label(mid, text='', font=(
    "Arial", 20, "bold"), bg="white", fg="black")
state.grid(row=0, column=0)

limg = tk.Label(mid, image=img[np.random.randint(
    len(img))], width='250', height='250', bg='white')
limg.grid(row=1, column=0)

calf = tk.Frame(mid, bg='white')
calf.grid(row=2, column=0)
caldeg = tk.Label(mid, text='Deg = 0', font=(
    "Arial", 12, "bold"), bg="white", fg="black")
caldeg.grid(row=3, column=0)

calkl = tk.Label(calf, text='delta k (to 0)', font=(
    "Arial", 12, "bold"), bg="white", fg="black")
calkl.grid(row=1, column=0)
calel = tk.Label(calf, text='Kinetic Energy', font=(
    "Arial", 12, "bold"), bg="white", fg="black")
calel.grid(row=2, column=0)


calk = tk.StringVar()
calk.set('0')
calk.trace_add('write', cal)
cale = tk.StringVar()
cale.set('0')
cale.trace_add('write', cal)
calken = tk.Entry(calf, font=("Arial", 12, "bold"),
                  width=15, textvariable=calk, bd=10)
calken.grid(row=1, column=1)
caleen = tk.Entry(calf, font=("Arial", 12, "bold"),
                  width=15, textvariable=cale, bd=10)
caleen.grid(row=2, column=1)


step = tk.Frame(g, bg='white')
step.grid(row=1, column=0, padx=30, pady=30)

l1 = tk.Label(step, text='Step 1', font=(
    "Arial", 12, "bold"), bg="white", fg='red')
l1.grid(row=0, column=0)
l2 = tk.Label(step, text='Step 2', font=(
    "Arial", 12, "bold"), bg="white", fg='blue')
l2.grid(row=1, column=0)
l3 = tk.Label(step, text='k offset (deg)', font=(
    "Arial", 12, "bold"), bg="white", fg="black")
l3.grid(row=2, column=0)
l4 = tk.Label(step, text='Step 3', font=(
    "Arial", 12, "bold"), bg="white", fg='blue')
l4.grid(row=3, column=0)
l5 = tk.Label(step, text='Step 4', font=("Arial", 12, "bold"),
              bg="white", fg="blue", height=1)
l5.grid(row=5, column=0)

fremfit = tk.Frame(master=step)
fremfit.grid(row=0, column=1)
lf = tk.Button(fremfit, text='Load Raw Data', font=(
    "Arial", 12, "bold"), fg='red', width=15, height='1', command=load, bd=10)
lf.grid(row=0, column=0)
bmfit = tk.Button(fremfit, text='MDC Fit', font=(
    "Arial", 12, "bold"), fg='red', width=8, height='1', command=cmfit, bd=10)
bmfit.grid(row=0, column=1)
befit = tk.Button(fremfit, text='EDC Fit', font=(
    "Arial", 12, "bold"), fg='red', width=8, height='1', command=cefit, bd=10)
befit.grid(row=0, column=2)


cut = tk.Frame(step, bg='white')
cut.grid(row=1, column=1)
mdccut = tk.Button(cut, text='MDC cut', font=(
    "Arial", 12, "bold"), width=8, height='1', command=ecut, bd=10, fg='blue')
mdccut.grid(row=0, column=0)
edccut = tk.Button(cut, text='EDC cut', font=(
    "Arial", 12, "bold"), width=8, height='1', command=angcut, bd=10, fg='black')
edccut.grid(row=0, column=1)
l_lowlim = tk.Label(cut, text='Lower Limit', font=(
    "Arial", 10, "bold"), bg="white", fg="black", height=1)
l_lowlim.grid(row=0, column=2)
lowlim = tk.StringVar()
lowlim.set('0')
lowlim.trace_add('write', flowlim)
in_lowlim = tk.Entry(cut, font=("Arial", 10, "bold"),
                     width=7, textvariable=lowlim, bd=5)
in_lowlim.grid(row=0, column=3)


k_offset = tk.StringVar()
try:
    k_offset.set(ko)
except:
    k_offset.set('0')
k_offset.trace_add('write', reload)
koffset = tk.Entry(step, font=("Arial", 12, "bold"),
                   width=15, textvariable=k_offset, bd=10)
koffset.grid(row=2, column=1)

lfit = tk.Frame(step, bg='white')
lfit.grid(row=3, column=1)
lmfit = tk.Button(lfit, text='Load MDC fitted File', font=(
    "Arial", 12, "bold"), width=15, height='1', command=loadmfit, bd=10, fg='blue')
lmfit.grid(row=0, column=0)
lefit = tk.Button(lfit, text='Load EDC fitted File', font=(
    "Arial", 12, "bold"), width=15, height='1', command=loadefit, bd=10, fg='black')
lefit.grid(row=0, column=1)

cfit = tk.Frame(step, bg='white')
cfit.grid(row=4, column=1)
cmfit = tk.Button(cfit, text='Clear MDC fitted File', font=(
    "Arial", 12, "bold"), width=15, height='1', command=clmfit, bd=5, fg='blue')
cmfit.grid(row=0, column=0)
cefit = tk.Button(cfit, text='Clear EDC fitted File', font=(
    "Arial", 12, "bold"), width=15, height='1', command=clefit, bd=5, fg='black')
cefit.grid(row=0, column=1)


lbb = tk.Button(step, text='Load Bare Band File', font=(
    "Arial", 12, "bold"), width=15, height='1', command=bareband, bd=10, fg='blue')
lbb.grid(row=5, column=1)

plots = tk.Frame(g, bg='white')
plots.grid(row=1, column=1)


cmf = tk.Frame(plots, bg='white')
cmf.grid(row=0, column=1)

bchcmp = tk.Button(cmf, text='Change Colormap', font=(
    "Arial", 12, "bold"), height='1', command=Chcmp, border=5)
bchcmp.grid(row=0, column=0)

cmlf = tk.Frame(cmf, bg='white')
cmlf.grid(row=1, column=0)


# Define your custom colors (as RGB tuples)
# (value,(color))
custom_colors = [(0, (1, 1, 1)),
                 (0.5, (0, 0, 1)),
                 (0.85, (0, 1, 1)),
                 (1, (1, 1, 0.26))]

# Create a custom colormap
custom_cmap = LinearSegmentedColormap.from_list(
    'custom_cmap', custom_colors, N=256)
mpl.colormaps.register(custom_cmap)
# plt.register_cmap('custom_cmap', custom_cmap)
optionList3 = ['terrain', 'custom_cmap', 'viridis', 'turbo',
               'inferno', 'plasma', 'copper', 'grey', 'bwr']   # 選項
cmp = plt.colormaps()
value3 = tk.StringVar()                                        # 取值
value3.set('terrain')
value3.trace_add('write', chcmp)
setcmap = tk.OptionMenu(cmlf, value3, *optionList3)
setcmap.grid(row=0, column=1)
cm = tk.OptionMenu(cmlf, value3, *cmp)
cm.grid(row=1, column=1)

c1 = tk.Label(cmlf, text='Commonly Used:', font=(
    "Arial", 12), bg="white", height='1')
c1.grid(row=0, column=0)
c2 = tk.Label(cmlf, text='All:', font=("Arial", 12), bg="white", height='1')
c2.grid(row=1, column=0)


optionList = ['Raw Data', 'E-K Diagram', 'MDC Normalized',
              'First Derivative', 'Second Derivative']   # 選項
value = tk.StringVar()                                        # 取值
value.set('---Plot1---')
# 第二個參數是取值，第三個開始是選項，使用星號展開
menu1 = tk.OptionMenu(plots, value, *optionList)
menu1.grid(row=1, column=1)
value.trace_add('write', plot1)

frfit = tk.Frame(plots, bg='white')
frfit.grid(row=2, column=1)
optionList1 = ['MDC fitted Data', 'EDC fitted Data',
               'Real Part', 'Imaginary Part']   # 選項
value1 = tk.StringVar()                                        # 取值
value1.set('---Plot2---')
# 第二個參數是取值，第三個開始是選項，使用星號展開
menu2 = tk.OptionMenu(frfit, value1, *optionList1)
menu2.grid(row=0, column=0)
value1.trace_add('write', plot2)
l_fit = tk.Label(frfit, text='Base counts:', font=(
    "Arial", 10, "bold"), bg="white", height='1', bd=5)
l_fit.grid(row=0, column=1)
base = tk.StringVar()
base.set('0')
base.trace_add('write', fbase)
in_fit = tk.Entry(frfit, font=("Arial", 10), width=5, textvariable=base, bd=5)
in_fit.grid(row=0, column=2)
b_fit = tk.Button(frfit, text='Fit FWHM', font=(
    "Arial", 10, "bold"), bg="white", height='1', bd=5, command=fitgl)
b_fit.grid(row=0, column=3)

optionList2 = ['Real & Imaginary', 'KK Transform',
               'Data Plot with Pos', 'Data Plot with Pos and Bare Band']   # 選項
value2 = tk.StringVar()                                        # 取值
value2.set('---Plot3---')
# 第二個參數是取值，第三個開始是選項，使用星號展開
menu3 = tk.OptionMenu(plots, value2, *optionList2)
menu3.grid(row=3, column=1)
value2.trace('w', plot3)

bb_offset = tk.StringVar()
bb_offset.set('0')
bb_offset.trace_add('write', fbb_offset)
bboffset = tk.Entry(plots, font=("Arial", 12, "bold"),
                    width=15, textvariable=bb_offset, bd=10)
bboffset.grid(row=4, column=1)
bbk_offset = tk.StringVar()
bbk_offset.set('1')
bbk_offset.trace_add('write', fbbk_offset)
bbkoffset = tk.Entry(plots, font=("Arial", 12, "bold"),
                     width=15, textvariable=bbk_offset, bd=10)
bbkoffset.grid(row=5, column=1)

lcmp = tk.Frame(plots, bg='white')
lcmp.grid(row=0, column=0)

lcmpd = Figure(figsize=(0.75, 1), layout='constrained')
cmpg = tkagg.FigureCanvasTkAgg(lcmpd, master=lcmp)
cmpg.get_tk_widget().grid(row=0, column=1)
lsetcmap = tk.Label(lcmp, text='Colormap:', font=(
    "Arial", 12, "bold"), bg="white", height='1', bd=10)
lsetcmap.grid(row=0, column=0)
chcmp()

m1 = tk.Label(plots, text='Raw', font=(
    "Arial", 12, "bold"), bg="white", fg='red')
m1.grid(row=1, column=0)
m2 = tk.Label(plots, text='Fit', font=(
    "Arial", 12, "bold"), bg="white", fg='blue')
m2.grid(row=2, column=0)
m3 = tk.Label(plots, text='Transform', font=(
    "Arial", 12, "bold"), bg="white", fg="blue")
m3.grid(row=3, column=0)
l6 = tk.Label(plots, text='Bare band E offset (meV)', font=(
    "Arial", 12, "bold"), bg="white", fg="black", height=1)
l6.grid(row=4, column=0)
l7 = tk.Label(plots, text='Bare band k ratio', font=(
    "Arial", 12, "bold"), bg="white", fg="black", height=1)
l7.grid(row=5, column=0)


figfr = tk.Frame(g, bg='white')
figfr.grid(row=0, column=2, padx=10, pady=10)

fig = Figure(figsize=(8, 6), layout='constrained')
out = tkagg.FigureCanvasTkAgg(fig, master=figfr)
out.get_tk_widget().grid(row=0, column=0)
out.mpl_connect('motion_notify_event', move)
out.mpl_connect('button_press_event', press)
out.mpl_connect('button_release_event', release)

xydata = tk.Frame(figfr, bg='white')
xydata.grid(row=1, column=0)

xdata = tk.Label(xydata, text='xdata:', font=(
    "Arial", 12, "bold"), width='15', height='1', bd=10, bg='white')
xdata.grid(row=0, column=0)
ydata = tk.Label(xydata, text='ydata:', font=(
    "Arial", 12, "bold"), width='15', height='1', bd=10, bg='white')
ydata.grid(row=0, column=1)

exf = tk.Frame(g, bg='white')
exf.grid(row=1, column=2)

clim = tk.Frame(exf, bg='white')
clim.grid(row=0, column=0)
lcmax = tk.Label(clim, text='Maximum', font=(
    'Arial', 12), bg='white', fg='white')
lcmax.grid(row=0, column=0)
lcmin = tk.Label(clim, text='Minimum', font=(
    'Arial', 12), bg='white', fg='white')
lcmin.grid(row=1, column=0)
cmax = tk.Frame(clim, bg='white', width=15, bd=10)
cmax.grid(row=0, column=1)
cmin = tk.Frame(clim, bg='white', width=15, bd=10)
cmin.grid(row=1, column=1)


cM = tk.DoubleVar()
cm = tk.DoubleVar()
cM.set(10000)
cm.set(-10000)
vcmax = tk.DoubleVar()
vcmax.set(10000)
vcmax.trace_add('write', cmaxrange)
Cmax = tk.Scale(cmax, from_=cm.get(), to=cM.get(), orient='horizontal',
                variable=vcmax, state='disabled', bg='white', fg='white')
Cmax.pack()
vcmin = tk.DoubleVar()
vcmin.set(-10000)
vcmin.trace_add('write', cminrange)
Cmin = tk.Scale(cmin, from_=cm.get(), to=cM.get(), orient='horizontal',
                variable=vcmin, state='disabled', bg='white', fg='white')
Cmin.pack()
# Cmax=tk.Scrollbar(cmax,orient='horizontal',bd=10,width=15)
# Cmax.pack(fill='x')
# Cmin=tk.Scrollbar(cmin,orient='horizontal',bd=10,width=15)
# Cmin.pack(fill='x')


ex = tk.Button(exf, text='Export Graph', font=(
    "Arial", 12, "bold"), height='1', command=exp, bd=10)
ex.grid(row=1, column=0)
extm = tk.Button(exf, text='Export MDC Fitted Data (k offset)', font=(
    "Arial", 12, "bold"), height='1', command=exptm, bd=10)
extm.grid(row=2, column=0)
exte = tk.Button(exf, text='Export EDC Fitted Data (k offset)', font=(
    "Arial", 12, "bold"), height='1', command=expte, bd=10)
exte.grid(row=3, column=0)
# fig = plt.figure(layout='constrained')
# axs = fig.subplots()
tt = threading.Thread(target=tstate)
tt.daemon = True
tt.start()
g.update()
g.mainloop()
