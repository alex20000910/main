from MDC_cut_utility import *
import os, inspect
import sys, shutil
from typing import Literal, Any, override
import numpy as np
import xarray as xr
import h5py
import json
import zarr
from PIL import ImageTk
from tkinter import filedialog as fd

def load_txt(path_to_file: str) -> xr.DataArray:    #for BiSe txt files 
#Liu, J. N., Yang, X., Xue, H., Gai, X. S., Sun, R., Li, Y., ... & Cheng, Z. H. (2023). Surface coupling in Bi2Se3 ultrathin films by screened Coulomb interaction. Nature Communications, 14(1), 4424.
    """
    Load data from a text file and convert it into an xarray DataArray.

    Parameters:
        path_to_file (str): The path to the text file.

    Returns:
        xr.DataArray: The data loaded from the text file as an xarray DataArray.
    """
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
    Step = (e_high - e_low)/(e_num - 1)
    Slit = 'Unknown'
    aq = 'Unknown'
    name = os.path.basename(path_to_file).removesuffix('.txt')
    if e_mode == 'Kinetic':
        e = np.linspace(e_low, e_high, e_num)
        CenterEnergy = str(CenterEnergy)+' eV'
        e_low = str(e_low)+' eV (K.E.)'
        e_high = str(e_high)+' eV (K.E.)'
    else:
        e = np.linspace(e_photon-e_high, e_photon-e_low, e_num)
        CenterEnergy = str(CenterEnergy)+' eV'
        e_low = str(e_low)+' eV (B.E.)'
        e_high = str(e_high)+' eV (B.E.)'

    a = np.linspace(a_low, a_high, a_num)
    # data=np.arange(float(len(e)*len(a))).reshape(len(e),len(a),1)
    # data[0:,0:,0]=d
    data = np.arange(float(len(e)*len(a))).reshape(len(e), len(a))
    data[0:, 0:] = d
    data = xr.DataArray(
        data=data,
        coords={
            'eV': e,
            'phi': a
        },
        name='Spectrum',
        attrs={
            'Name': name,
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
            'Iterations': Iterations,
            'Description': 'BiSe',
            'Path': path_to_file
        }
    )
    return data

def load_txt_sk(path_to_file: str) -> xr.DataArray:    #for sklearn txt files
    """
    Args:
        path_to_file (str): File Path

    Returns:
        xr.DataArray: shape=(len(eV), len(phi))
    """
    name = os.path.basename(path_to_file).removesuffix('.txt')
    e = np.linspace(21.2-1, 21.2+0.2, 659)    #fix BE 1~-0.2
    # e = np.linspace(21.2-2, 21.2+1, 284)     #scan BE 2~-1
    a = np.linspace(-10, 10, 494)     #A20
    description='SKNET'
    e_low = str(np.min(np.float64(e)))+ ' eV (K.E.)'
    e_high = str(np.max(np.float64(e)))+ ' eV (K.E.)'
    e_photon = 21.2
    #   attrs
    e_mode = 'Kinetic'
    LensMode = 'Unknown'
    PassEnergy = 'Unknown'
    Dwell = 'Unknown'
    CenterEnergy = str(np.average(np.float64(e)))+ ' eV (K.E.)'
    Iterations = 'Unknown'
    Step = abs(e[1]-e[0])
    Slit = 'Unknown'
    aq = 'SRNET'
    data = np.loadtxt(path_to_file).transpose()*100
    data = xr.DataArray(
        data=data,
        coords={
            'eV': e,
            'phi': a
        },
        name='Spectrum',
        attrs={
            'Name': name,
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
            'Dwell': Dwell,
            'Iterations': Iterations,
            'Description': description,
            'Path': path_to_file
        }
    )
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
    description = f['Region']['Description']
    if e_mode == 'Kinetic':
        e = np.linspace(e_low, e_high, e_num)
        CenterEnergy = str(CenterEnergy)+' eV'
        e_low = str(e_low)+' eV (K.E.)'
        e_high = str(e_high)+' eV (K.E.)'
    else:
        e = np.linspace(e_photon-e_high, e_photon-e_low, e_num)
        CenterEnergy = str(CenterEnergy)+' eV'
        e_low = str(e_low)+' eV (B.E.)'
        e_high = str(e_high)+' eV (B.E.)'

    a = np.linspace(a_low, a_high, a_num)
    d = np.asarray(f['Spectrum']).transpose()
    # data=np.arange(float(len(e)*len(a))).reshape(len(e),len(a),1)
    # data[0:,0:,0]=d
    data = np.arange(float(len(e)*len(a))).reshape(len(e), len(a))
    data[0:, 0:] = d
    data = xr.DataArray(
        data=data,
        coords={
            'eV': e,
            'phi': a
        },
        name='Spectrum',
        attrs={
            'Name': name,
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
            'Iterations': Iterations,
            'Description': description,
            'Path': path_to_file
        }
    )
    return data

def get_cec_params(path_to_file: str) -> tuple:
    '''
    Get CEC parameters from the file.
    
    Parameters:
        path_to_file (str): The path to the file.
        
    Returns:
        (angle, cx, cy, cdx, cdy, phi_offset, r1_offset, phi1_offset, r11_offset, slim, sym) (tuple) : A tuple containing the CEC parameters.
    
    '''
    if path_to_file.endswith('.h5') or path_to_file.endswith('.H5'):
        f = h5py.File(path_to_file, 'r')
        angle = np.array(f.get('VolumeSlicer').get('angle'))[0]
        cx = np.array(f.get('VolumeSlicer').get('cx'))[0]
        cy = np.array(f.get('VolumeSlicer').get('cy'))[0]
        cdx = np.array(f.get('VolumeSlicer').get('cdx'))[0]
        cdy = np.array(f.get('VolumeSlicer').get('cdy'))[0]
        phi_offset = np.array(f.get('VolumeSlicer').get('phi_offset'))[0]
        r1_offset = np.array(f.get('VolumeSlicer').get('r1_offset'))[0]
        try:
            phi1_offset = np.array(f.get('VolumeSlicer').get('phi1_offset'))[0]
            r11_offset = np.array(f.get('VolumeSlicer').get('r11_offset'))[0]
        except:
            phi1_offset = 0
            r11_offset = 0
            print('\033[31mNo sample offset info found in the h5 file, set to zero.\033[0m')
        slim = np.array(f.get('VolumeSlicer').get('slim'))[0]
        try:
            '''
            After 6.0 version, the symmetry information is added to attributes
            '''
            sym = np.array(f.get('VolumeSlicer').get('sym'))[0]
        except:
            sym = 1
    elif path_to_file.endswith('.npz') or path_to_file.endswith('.NPZ'):
        f = np.load(path_to_file)
        angle = f['angle']
        cx = f['cx']
        cy = f['cy']
        cdx = f['cdx']
        cdy = f['cdy']
        phi_offset = f['phi_offset']
        r1_offset = f['r1_offset']
        try:
            phi1_offset = f['phi1_offset']
            r11_offset = f['r11_offset']
        except:
            phi1_offset = 0
            r11_offset = 0
            print('\033[31mNo sample offset info found in the npz file, set to zero.\033[0m')
        slim = f['slim']
        try:
            '''
            After 6.0 version, the symmetry information is added to attributes
            '''
            sym = f['sym']
        except:
            sym = 1
    return angle, cx, cy, cdx, cdy, phi_offset, r1_offset, phi1_offset, r11_offset, slim, sym

def get_cec_attr(path_to_file: str, f: h5py.File | Any, name: str, cmap: str, app_pars: Any) -> tuple[str, str, str, str, str, str]:
    '''
    Call the CEC window.

    Parameters:
        path_to_file (str): The path to the HDF5 or NPZ file.
        f (Any): The file object opened using h5py.File() or numpy.load().
        name (str): The name of the dataset.
        cmap (str): The colormap to be used.
        app_pars (Any): Application parameters.
    
    Returns:
        (PassEnergy, Dwell, Iterations, Slit, lf_path, tlfpath) (tuple): A tuple containing the CEC indicators and info parameters.
        - PassEnergy (str): Attributes string.
        - Dwell (str): Attributes string.
        - Iterations (str): Attributes string.
        - Slit (str): Attributes string.
        - lf_path (str): List of raw data file paths.
        - tlfpath (str): List of raw data file paths.

    '''
    if isinstance(f, h5py.File):
        tlf_path = np.array(f.get('VolumeSlicer').get('path'), dtype='S')
        lf_path = [i.tobytes().decode('utf-8') for i in tlf_path]
    else:   # np.load
        lf_path = f['path']
    # try:
    tlfpath = []
    try:    #load path that saved in h5/npz
        tbasename = os.path.basename(lf_path[0])
        if '.h5' in tbasename:
            td=load_h5(lf_path[0])
        elif '.json' in tbasename:
            td=load_json(lf_path[0])
        elif '.txt' in tbasename:
            td=load_txt(lf_path[0])
    except: #try load file in the same folder as h5/npz
        td = None
        for i in lf_path:
            tbasename = os.path.basename(i)
            tpath = os.path.normpath(os.path.join(os.path.dirname(path_to_file), tbasename))
            tlfpath.append(tpath)
            try:
                if '.h5' in tbasename:
                    td=load_h5(tpath)
                elif '.json' in tbasename:
                    td=load_json(tpath)
                elif '.txt' in tbasename:
                    td=load_txt(tpath)
            except:
                pass
    PassEnergy = td.attrs['PassEnergy']
    Dwell = td.attrs['Dwell']
    Iterations = td.attrs['Iterations']
    Slit = td.attrs['Slit']
        # if f_npz is False:
        #     f_npz = True
        #     args = get_cec_params(f)
        #     try:
        #         cec = CEC(g, lf_path, mode='load', cmap=cmap, app_pars=app_pars)
        #         cec.load(*args, name, path_to_file)
        #     except:
        #         cec = CEC(g, tlfpath, mode='load', cmap=cmap, app_pars=app_pars)
        #         cec.load(*args, name, path_to_file)
    # except Exception as ecp:
    #     if f_npz is False:
    #         f_npz = True
    #         if app_pars:
    #             windll.user32.ShowWindow(app_pars.hwnd, 9)
    #             windll.user32.SetForegroundWindow(app_pars.hwnd)
    #         print(f"An error occurred: {ecp}")
    #         print('\033[31mPath not found:\033[34m')
    #         print(lf_path)
    #         print('\033[31mPlace all the raw data files listed above in the same folder as the HDF5/NPZ file\nif you want to view the slicing geometry or just ignore this message if you do not need the slicing geometry.\033[0m')
    #         message = f"Path not found:\n{lf_path}\nPlace all the raw data files listed above in the same folder as the HDF5/NPZ file if you want to view the slicing geometry\nor just ignore this message if you do not need the slicing geometry."
    #         messagebox.showwarning("Warning", message)
    return PassEnergy, Dwell, Iterations, Slit, lf_path, tlfpath

def load_h5(path_to_file: str, **kwargs) -> xr.DataArray:
    """
    Load data from an HDF5 file and return it as a DataArray.

    Parameters:
        path_to_file (str): The path to the HDF5 file.

    Returns:
        xr.DataArray: The loaded data as a DataArray.

    """
    if 'cec' in kwargs and 'f_npz' in kwargs and 'cmap' in kwargs:
        cec = kwargs['cec']
        f_npz = kwargs['f_npz']
        cmap = kwargs['cmap']
        app_pars = kwargs['app_pars']
    else:
        cec = None
        f_npz = True
    
    with h5py.File(path_to_file, 'r') as f:
        # f = h5py.File(path_to_file, 'r')
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
        try:
            flag = np.array(f.get('Region').get('Name'), dtype=str)[1]
            t_name = np.array(f.get('Region').get('Name'), dtype=str)
        except:        
            t_name = np.array(f.get('Region').get('Name'), dtype='S')  # Read as bytes
            t_name = t_name.tobytes().decode('utf-8')   # Convert to string
        try:
            flag = np.array(f.get('Region').get('Description'), dtype=str)[1]
            t_description = np.array(f.get('Region').get('Description'), dtype=str)
        except:
            t_description = np.array(f.get('Region').get('Description'), dtype='S')  # Read as bytes
            t_description = t_description.tobytes().decode('utf-8')   # Convert to string
        try:
            flag = np.array(f.get('Region').get('EnergyMode'), dtype=str)[1]
            t_e_mode = np.array(f.get('Region').get('EnergyMode'), dtype=str)
            t_LensMode = np.array(f.get('Region').get('LensMode'), dtype=str)
            t_Slit = np.array(f.get('Region').get('Slit'), dtype=str)
            t_aq = np.array(f.get('Region').get('Acquisition'), dtype=str)
        except:
            flag = 'pass_byte'
            t_e_mode = np.array(f.get('Region').get('EnergyMode'), dtype='S')  # Read as bytes
            t_e_mode = t_e_mode.tobytes().decode('utf-8')   # Convert to string
            t_LensMode = np.array(f.get('Region').get('LensMode'), dtype='S')  # Read as bytes
            t_LensMode = t_LensMode.tobytes().decode('utf-8')   # Convert to string
            t_Slit = np.array(f.get('Region').get('Slit'), dtype='S')  # Read as bytes
            t_Slit = t_Slit.tobytes().decode('utf-8')   # Convert to string
            t_aq = np.array(f.get('Region').get('Acquisition'), dtype='S')  # Read as bytes
            t_aq = t_aq.tobytes().decode('utf-8')   # Convert to string        
            
        e_mode = ''
        LensMode = ''
        Slit = ''
        aq = ''
        name = ''
        description = ''
        if flag != 'pass_byte':
            for i in range(60):  # proper length long enough
                e_mode += t_e_mode[i]
                LensMode += t_LensMode[i]
                Slit += t_Slit[i]
                aq += t_aq[i]
        else:
            e_mode = t_e_mode
            LensMode = t_LensMode
            Slit = t_Slit
            aq = t_aq
        for i in range(600):
            try:
                name += t_name[i]
            except:
                pass
            try:
                description += t_description[i]
            except:
                pass
        if e_mode == 'Kinetic':
            e = np.linspace(e_low, e_high, e_num)
            CenterEnergy = str(CenterEnergy)+' eV'
            e_low = str(e_low)+' eV (K.E.)'
            e_high = str(e_high)+' eV (K.E.)'
        else:
            e = np.linspace(e_photon-e_high, e_photon-e_low, e_num)
            CenterEnergy = str(CenterEnergy)+' eV'
            e_low = str(e_low)+' eV (B.E.)'
            e_high = str(e_high)+' eV (B.E.)'
        if aq == 'VolumeSlicer':
            if 'cec' in kwargs and 'f_npz' in kwargs:
                if f_npz is False:
                    f_npz = True
                    cec = 'CEC_Object'
                PassEnergy, Dwell, Iterations, Slit, lf_path, tlfpath = get_cec_attr(path_to_file, f, name, cmap, app_pars)
                cec_pars = cec_param(path_to_file, name, lf_path, tlfpath, cmap)
            
        a = np.linspace(a_low, a_high, a_num)
        d = np.asarray(f.get('Spectrum')).transpose()
        if flag != 'pass_byte':
            pass
        else:
            Dwell = Dwell.removesuffix(' s')
            PassEnergy = PassEnergy.removesuffix(' eV')
            d = d.T
        data = xr.DataArray(
            data=d,
            coords={
                'eV': e,
                'phi': a
            },
            name='Spectrum',
            attrs={
                'Name': name,
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
                'Iterations': Iterations,
                'Description': description,
                'Path': path_to_file
            }
        )
    if 'cec' in kwargs and 'f_npz' in kwargs:
        return data, cec, f_npz, cec_pars
    else:
        return data

def load_npz(path_to_file: str, **kwargs) -> xr.DataArray:
    """
    Load data from a NumPy NPZ file and convert it into an xarray DataArray.
    
    Parameters:
        path_to_file (str): The path to the NPZ file.
        
    Returns:
        xr.DataArray: The data loaded from the NPZ file as an xarray DataArray.
    """
    if 'cec' in kwargs and 'f_npz' in kwargs and 'cmap' in kwargs:
        cec = kwargs['cec']
        f_npz = kwargs['f_npz']
        cmap = kwargs['cmap']
        app_pars = kwargs['app_pars']
    else:
        cec = None
        f_npz = True
    name = os.path.basename(path_to_file).split('.npz')[0]
    f = np.load(path_to_file)
    if 'cec' in kwargs and 'f_npz' in kwargs:
        if f_npz is False:
            f_npz = True
            cec = 'CEC_Object'
        PassEnergy, Dwell, Iterations, Slit, lf_path, tlfpath = get_cec_attr(path_to_file, f, name, cmap, app_pars)
        cec_pars = cec_param(path_to_file, name, lf_path, tlfpath, cmap)
    
    data = f['data']
    k = f['x']
    ev = f['y']
    ExcitationEnergy = f['e_photon']
    desc = f['desc'][0]
    data = xr.DataArray(
        data=data,
        coords={
            'eV': ev,
            'phi': k
        },
        name='Spectrum',
        attrs={
            'Name': name,
            'Acquisition': 'VolumeSlicer',
            'EnergyMode': 'Kinetic',
            'ExcitationEnergy': str(ExcitationEnergy)+' eV',
            'CenterEnergy': '%.3f'%((ev[0]+ev[-1])/2)+' eV',
            'HighEnergy': str(ev[-1])+' eV (K.E.)',
            'LowEnergy': str(ev[0])+' eV (K.E.)',
            'Step': str(ev[1]-ev[0])+' eV',
            'LensMode': 'Angular',
            'PassEnergy': PassEnergy,
            'Slit': Slit,
            'Dwell': Dwell,
            'Iterations': Iterations,
            'Description': desc,
            'Path': path_to_file
        }
    )
    if 'cec' in kwargs and 'f_npz' in kwargs:
        return data, cec, f_npz, cec_pars
    else:
        return data


class loadfiles(FileSequence):
    """
    A class to handle loading and organizing multiple file types for analysis.
    It supports h5, json, txt, and npz files, and organizes them based on R1 and R2 values.
    
    Parameters
    ----------
        files (list, tuple or string) : File paths to be loaded. It can include h5, json, txt, and npz files.
        mode (str) : Loading mode, either 'lazy' or 'eager'. Default is 'lazy'.
        init (bool) : If True, read the existing zarr file if using lazy loading mode. Else, rewrite/create the zarr file instead. Default is False.
    
    Attributes
    ----------
        data (list): List of loaded data from files.
        path (list): List of file paths.
        name (list): List of file names without extensions.
        r1 (list): List of R1 values extracted from file names.
        r2 (list): List of R2 values extracted from file names.
        f_npz (list): Boolean list indicating if the corresponding file is a npz file
        n (list): Indices of files that are npz or VolumeSlicer h5 files.
    Returns
    ----------
        object (FileSequence): File Sequence object.
        
    """
    __slots__ = ['f_npz', 'n', 'opath', 'oname', 'r1s', 'r2s', 'sep', 'r1_splitter', 'r2_splitter', 'or2', 'or1', 'path1', 'r11', 'path', 'name', 'r1', 'r2', 'data', 'sort', 'zpath', 'xr_name', 'xr_coords', 'xr_attrs', 'load_mode', 'cec', 'f_npz_', 'app_pars', 'cec_pars']
    def __init__(self, files: list[str] | tuple[str, ...] | str, mode: Literal['lazy', 'eager'] = 'lazy', init: bool = False, **kwargs):
        if isinstance(files, str):
            files = [files] # Convert string to list ensuring compatibility
        self.f_npz = [False for i in files]
        self.n = []
        self._name = 'external'
        if 'app_pars' in kwargs:
            self.app_pars = kwargs['app_pars']
        else:
            self.app_pars = None
        tempdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        tempdir = os.path.dirname(tempdir)
        if 'name' in kwargs:
            if kwargs['name'] == 'internal':
                self._name = 'internal'
                self.zpath = os.path.join(tempdir, 'lfs_data')
            elif kwargs['name'] == 'external':
                self.zpath = os.path.join(tempdir, 'ext_lfs_data')
        else:
            self.zpath = os.path.join(tempdir, 'ext_lfs_data')
        self.load_mode = mode
        self.init = init
        self.cec, self.f_npz_ = None, False
        self.cec_pars = None
        if 'spectrogram' in kwargs:
            self.cec = False
        if 'cmap' in kwargs:
            cmap = kwargs['cmap']
        else:
            cmap = 'viridis'
        for i, v in enumerate(files):
            tf=False
            try:
                if '.npz' in os.path.basename(v):
                    if self.cec is None:
                        data, self.cec, self.f_npz_, self.cec_pars = load_npz(v, cec=self.cec, f_npz=self.f_npz_, cmap=cmap, app_pars=self.app_pars)
                    else:
                        data = load_npz(v)
                else:
                    if self.cec is None:
                        data, self.cec, self.f_npz_, self.cec_pars = load_h5(v, cec=self.cec, f_npz=self.f_npz_, cmap=cmap, app_pars=self.app_pars)
                    else:
                        data = load_h5(v)
                if data.attrs['Acquisition'] in ['VolumeSlicer', 'DataCube']:
                    tf=True
                clear(data)
            except: pass
            if '.npz' in os.path.basename(v) or tf:
                self.f_npz[i] = True
                self.n.append(i)
        self.opath = [f for f in files]
        files = None
        self.oname = [os.path.basename(f).split('#id#')[0].split('#d#')[0].split('id')[0].replace('.h5', '').replace('.json', '').replace('.txt', '').replace('.npz', '') for f in self.opath]
        self.r1s = ['R1_', 'R1 ', 'R1', 'r1_', 'r1 ', 'r1']
        self.r2s = ['R2_', 'R2 ', 'R2', 'r2_', 'r2 ', 'r2']
        self.sep = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
                    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
                    ',','!','@','#','$','%','^','&','*','(',')','_','-','+','=','[',']','{','}','|','\\',';',':','\'','"',
                    ' ','.txt','.TXT']
        self.__set_r1_r2()
        self.__set_data()
        self.__store()
        return
    
    def __str__(self)-> str:
        string = '\033[36m<MDC_cut.loadfiles File Loader>\n'
        string += f"Loaded {len(self.name)} files:\033[32m\n"
        for i, name in enumerate(self.name):
            string += f"  {i + 1}. {name}\n"
        string += f'\033[0mSize: {np.sum([self.get(i).nbytes/1024**2 for i in range(len(self.name))]):.3f} MB\n'
        return string
    
    @override
    def get(self, ind: int = 0) -> xr.DataArray:
        if self.load_mode == 'eager':
            return self.data[ind]
        elif self.load_mode == 'lazy':
            path = os.path.join(self.zpath, f'lfs_{ind}.zarr')
            tdata = zarr.open(path, mode='r')
            data = xr.DataArray(
                data=tdata,
                coords=self.xr_coords[ind],
                name=self.xr_name[ind],
                attrs=self.xr_attrs[ind]
            )
            tdata, path = None, None
        return data
    
    @property
    def __check_name(self) -> bool:
        flag = True
        try:
            tpath = os.path.join(self.zpath, f'lfs_name.check')
            with open(tpath, 'r') as f:
                string = f.read().split('__sep__')[:-1]
            for i, v in enumerate(self.name):
                if string[i] != v:
                    flag = False
        except:
            flag = False
        return flag
    
    def __store(self):
        if self.load_mode == 'lazy':
            exist_f = False
            match_f = False
            if os.path.exists(self.zpath):
                exist_f = True
                match_f = self.__check_name
                if not self.init:
                    shutil.rmtree(self.zpath)
            if not exist_f or not self.init or not match_f:
                try:
                    os.makedirs(self.zpath, exist_ok=True)
                except:
                    pass
                tpath = os.path.join(self.zpath, f'lfs_name.check')
                with open(tpath, 'w') as f:
                    for i in range(len(self.name)):
                        data = self.data[i].data
                        dpath = os.path.join(self.zpath, f'lfs_{i}.zarr')
                        zarr.save(dpath, data)
                        f.write(self.name[i]+'__sep__')
                set_hidden(self.zpath)
                data, dpath, tpath = None, None, None
        self.xr_coords, self.xr_name, self.xr_attrs = [], [], []
        if self.load_mode == 'lazy':
            for i in self.data:
                i.data = np.empty(i.data.shape, dtype=np.uint8)
                self.xr_coords.append(i.coords)
                self.xr_name.append(i.name)
                self.xr_attrs.append(i.attrs)
            self.data = None
            self.data = [i for i in range(len(self.name))]
        elif self.load_mode == 'eager':
            for i, v in enumerate(self.data):
                v.data = v.data.astype(np.float32)
    
    def __set_data(self):
        self.data = []
        for i in self.path:
            tbasename = os.path.basename(i)
            if '.h5' in tbasename:
                self.data.append(load_h5(i))
            elif '.json' in tbasename:
                self.data.append(load_json(i))
            elif '.txt' in tbasename:
                self.data.append(load_txt(i))
            elif '.npz' in tbasename:
                self.data.append(load_npz(i))
            else:
                self.data.append([])
    
    def __set_r1_r2(self):
        def __sort_r1_r2():
            self.or2 = self.gen_r2(self.oname, self.r1_splitter, self.r2_splitter)
            self.or1 = self.gen_r1(self.oname, self.r1_splitter, self.r2_splitter)
            self.path1 = []
            self.r11 = []
            tpath = []
            tr1 = []
            tr2 = []
            self.r2 = []
            r2 = sorted(set(self.or2))
            for i in r2:
                for j,k,l in zip(self.opath, self.or1, self.or2):
                    if i == l:
                        tpath.append(j)
                        tr1.append(k)
                        tr2.append(l)
                self.path1.append(tpath)
                self.r11.append(tr1)
                self.r2.append(tr2)
                tpath = []
                tr1 = []
                tr2 = []
            self.path = []
            self.r1 = []
            t = 0
            for i in self.path1:
                opath = [f for f in i]
                path = res(self.r11[t], opath)
                r1 = res(self.r11[t], self.r11[t])
                for j,v in enumerate(path):
                    self.path.append(v)
                    self.r1.append(r1[j])
                t+=1
            self.name = [os.path.basename(f).split('#id#')[0].split('#d#')[0].split('id')[0].replace('.h5', '').replace('.json', '').replace('.txt', '') for f in self.path]
            self.name = self.check_repeat(self.name)
            
        def __sort_r1():
            self.or1 = self.gen_r1(self.oname, self.r1_splitter, self.r2_splitter)
            self.path = res(self.or1, self.opath)
            self.r1 = res(self.or1, self.or1)
            self.name = [os.path.basename(f).split('#id#')[0].split('#d#')[0].split('id')[0].replace('.h5', '').replace('.json', '').replace('.txt', '') for f in self.path]
            self.name = self.check_repeat(self.name)

        self.r1_splitter , self.r2_splitter= [], []
        r1s = self.r1s
        r2s = self.r2s
        for i in self.oname:
            tj, tk = False, False
            for j,k in zip(r1s, r2s):
                if j in i and tj == False:
                    self.r1_splitter.append(j)
                    tj = True
                if k in i and tk == False:
                    self.r2_splitter.append(k)
                    tk = True
            if not tj:
                self.r1_splitter.append('No_r1')
            if not tk:
                self.r2_splitter.append('No_r2')
        r1s, r2s = None, None
        try:
            f = True
            t=0
            for i in self.oname:
                if len(i.split(self.r1_splitter[t]))<=1 or len(i.split(self.r2_splitter[t]))<=1:
                    f = False
                t+=1
            if f:   # r1 and r2 exist
                __sort_r1_r2()
                self.sort = 'r1r2'
                if self._name == 'internal':
                    print('Sort by r1 and r2\n')
            elif len(self.oname[0].split(self.r1_splitter[0]))>1:   # only r1 exist
                __sort_r1()
                self.sort = 'r1'
                if self._name == 'internal':
                    print('Sort by r1\n') 
            else:   # no r1 and r2
                if self.r1s[0] == 'X_':
                    self.r1s, self.r2s = ['R1_', 'R1 ', 'R1', 'r1_', 'r1 ', 'r1'], ['R2_', 'R2 ', 'R2', 'r2_', 'r2 ', 'r2']
                    self.path = self.opath
                    self.name = self.check_repeat(self.oname)
                    self.sort = 'no'
                    if self._name == 'internal':
                        print('No Sort\n')
                else:
                    self.r1s, self.r2s = ['X_', 'X ', 'X', 'x_', 'x ', 'x'], ['Z_', 'Z ', 'Z', 'z_', 'z ', 'z']
                    self.__set_r1_r2()
        except Exception as ecp:
            print(f"An error occurred in loadfiles().__set_r1_r2(): {ecp}")
            self.r1s, self.r2s = ['R1_', 'R1 ', 'R1', 'r1_', 'r1 ', 'r1'], ['R2_', 'R2 ', 'R2', 'r2_', 'r2 ', 'r2']
            self.path = self.opath
            self.name = self.check_repeat(self.oname)
            self.sort = 'no'
            if self._name == 'internal':
                print('No Sort (Exception)\n')
        
    def check_repeat(self, name):
        fl = False
        tname = [f for f in name]
        if len(name) != len(set(name)):
            fl = True
        if fl:
            t = 0
            while t < len(tname):
                fj = False
                tt = False
                tj = t
                for j in range(t+1, len(name)):
                    if name[t] == name[j]:
                        if not tt:
                            tname[t] = tname[t]+'#id#'+str(t)
                            tt = True
                        tname[j] = tname[j]+'#id#'+str(j)
                        fj = True
                        tj = j
                if fj:
                    t = tj
                t+=1
        return tname
        
    @override
    def gen_r1(self, name, r1_splitter, r2_splitter):
            try:
                r1 = []
                for i,v in enumerate(name):
                    tf=True
                    t=v.split(r1_splitter[i])[1].split(r2_splitter[i])[0].split(' ')[0].split('_')[0]
                    while tf:
                        try:
                            a=float(t)
                            tf=False
                        except:
                            for j in self.sep:
                                if j in t:
                                    t=t.split(j)[0]
                    r1.append(a)
                return np.float64(r1)
            except:
                print('Error in loadfiles().gen_r1')
                print(sys.exc_info())
                return name
    
    @override
    def gen_r2(self, name, r1_splitter, r2_splitter):
            try:
                r2 = []
                for i,v in enumerate(name):
                    tf=True
                    t=v.split(r2_splitter[i])[1].split(r1_splitter[i])[0].split(' ')[0].split('_')[0]
                    while tf:
                        try:
                            a=float(t)
                            tf=False
                        except:
                            for j in self.sep:
                                if j in t:
                                    t=t.split(j)[0]
                    r2.append(a)
                return np.float64(r2)
            except:
                print('Error in loadfiles().gen_r2')
                print(sys.exc_info())
                return name

class mloader:
    def __init__(self, st, data, ev, phi, rdd, cdir, lowlim=0):
        self.st = st
        self.data = data
        self.ev = ev
        self.phi = phi
        self.rdd = rdd
        self.cdir = cdir
        self.lowlim = lowlim
    
    def loadparam(self, k_offset: str, base: str, npzf: bool, fpr: int, name: str):
        self.k_offset = k_offset
        self.base = base
        self.npzf = npzf
        self.mfi_x = []
        self.fload = False
        self.fpr = fpr
        self.name = name
        
        self.fev = []
        self.rpos = []
        self.ophi = []
        self.fwhm = []
        self.pos = []
        self.kmin = []
        self.kmax = []
        self.skmin = []
        self.skmax = []
        self.smfp = []
        self.smfi = []
        self.smaa1 = []
        self.smaa2 = []
        self.smresult = []
        self.smcst = []
    
    def loadmfit_2p(self, file: str):
        # file = fd.askopenfilename(title="Select MDC Fitted file", filetypes=(("VMS files", "*.vms"),))
        mfpath = ''
        yy = []
        for n in range(len(self.ev)):
            ecut = self.data.sel(eV=self.ev[n], method='nearest')
            y = ecut.to_numpy().reshape(len(self.phi))
            y = np.where(y > int(self.lowlim), y, int(self.lowlim))
            yy.append(y)
            path = 'ecut_%.3f.txt' % self.ev[n]
            mfpath += path
        if len(file) > 2:
            self.rdd = file
            print('Loading...')
            self.st.put('Loading...')
        else:
            self.rdd = path
            # self.lmgg.destroy()
        if ".vms" in file:
            n = -1
            # os.chdir(self.rdd.removesuffix(self.rdd.split('/')[-1]))
            os.chdir(os.path.dirname(self.rdd))
            fc = open('copy2p_'+os.path.basename(file), 'w', encoding='utf-8')
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
                                for j in range(len(self.phi)):
                                    fc.write(str(int(yy[fi][-j-1]))+'\n')
                                n = len(self.phi)
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
                                for j in range(len(self.phi)):
                                    fc.write(str(int(yy[fi][-j-1]))+'\n')
                                n = len(self.phi)
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
            os.chdir(self.cdir)
        print('Done')
        self.st.put('Done')
        # self.lmgg.destroy()
    
    def loadmfit_(self, file: str):
        # file = fd.askopenfilename(title="Select MDC Fitted file", filetypes=(("NPZ files", "*.npz"), ("VMS files", "*.vms"),))
        # global h, m, fwhm, fev, pos, limg, img, name, ophi, rpos, st, kmax, kmin, lmgg
        # global data, rdd, skmin, skmax, smaa1, smaa2, smfp, smfi, fpr, mfi_x, smresult, smcst
        m=9.1093837015e-31  # electron mass kg
        h=6.62607015e-34   # Planck constant J·s
        mfpath = ''
        yy = []
        for n in range(len(self.ev)):
            ecut = self.data.sel(eV=self.ev[n], method='nearest')
            y = ecut.to_numpy().reshape(len(self.phi))
            y = np.where(y > int(self.lowlim), y, int(self.lowlim))
            yy.append(y)
            path = 'ecut_%.3f.txt' % self.ev[n]
            mfpath += path
        if len(file) > 2:
            self.fpr = 0
            self.rdd = file
            print('Loading...')
            self.st.put('Loading...')
        else:
            self.rdd = path
            # self.lmgg.destroy()
        if ".vms" in file:
            n = -1
            fev = np.array([], dtype=float)
            self.mfi_x = np.arange(len(self.ev))
            t_fwhm = []
            t_pos = []
            t_kmax = []
            t_kmin = []
            smfi = []
            skmin = []
            skmax = []
            smfp = [1 for i in range(len(self.ev))]
            # os.chdir(self.rdd.removesuffix(self.rdd.split('/')[-1]))
            os.chdir(os.path.dirname(self.rdd))
            fc = open('copy2p_'+os.path.basename(file), 'w', encoding='utf-8')
            ff = open(self.name+'_mdc_fitted_raw_data.txt', 'w',
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
                            if (self.ev[fi] > 20.58 and np.float64(tpos) < 1486.6+0.023) or (self.ev[fi] < 20.58 and np.float64(tpos) > 1486.6+0.023) or 1 == 1:
                                if self.npzf:tkk = self.phi
                                else:tkk = (2*m*self.ev[fi]*1.602176634*10**-19)**0.5*np.sin(self.phi/180*np.pi)*10**-10/(h/2/np.pi)
                                if float(tpos) > 1200:
                                    tkk+=1486.6
                                d = tkk[1]-tkk[0]
                                tr = float(tpos)+float(tfwhm)/2
                                tl = float(tpos)-float(tfwhm)/2
                                ri = int((tr-tkk[0])/d)
                                li = int((tl-tkk[0])/d)
                                tr = tkk[ri]+(float(tr)-(tkk[0]+ri*d)
                                            )/d*(tkk[ri+1]-tkk[ri])
                                tl = tkk[li]+(float(tl)-(tkk[0]+li*d)
                                            )/d*(tkk[li+1]-tkk[li])
                                tfwhm = tr-tl
                                tpi = int((float(tpos)-tkk[0])/d)
                                tpos = tkk[tpi]+(float(tpos)-(tkk[0]+tpi*d)
                                                )/d*(tkk[tpi+1]-tkk[tpi])
                                tpi = int((float(tkmax)-tkk[0])/d)
                                tkmax = tkk[tpi]+(float(tkmax) -
                                                (tkk[0]+tpi*d))/d*(tkk[tpi+1]-tkk[tpi])
                                tpi = int((float(tkmin)-tkk[0])/d)
                                if tpi > 492:
                                    tpi = 492
                                tkmin = tkk[tpi]+(float(tkmin) -
                                                (tkk[0]+tpi*d))/d*(tkk[tpi+1]-tkk[tpi])

                                fev = np.append(fev, self.ev[fi])  # 內容勿動 indent小最內圈if一階
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
                                        str(self.ev[fi])+'\t'+str(tfwhm)+'\t'+str(np.float64(tpos)-1486.6)+'\n')
                                else:
                                    ff.write(
                                        str(self.ev[fi])+'\t'+str(tfwhm)+'\t'+str(np.float64(tpos))+'\n')

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
                            if (self.ev[fi] > 20.58 and np.float64(tpos) < 1486.6+0.023) or (self.ev[fi] < 20.58 and np.float64(tpos) > 1486.6+0.023) or 1 == 1:
                                if self.npzf:tkk = self.phi
                                else:tkk = (2*m*self.ev[fi]*1.602176634*10**-19)**0.5*np.sin(self.phi/180*np.pi)*10**-10/(h/2/np.pi)
                                if float(tpos) > 1200:
                                    tkk+=1486.6
                                d = tkk[1]-tkk[0]
                                tr = float(tpos)+float(tfwhm)/2
                                tl = float(tpos)-float(tfwhm)/2
                                ri = int((tr-tkk[0])/d)
                                li = int((tl-tkk[0])/d)
                                tr = tkk[ri]+(float(tr)-(tkk[0]+ri*d)
                                            )/d*(tkk[ri+1]-tkk[ri])
                                tl = tkk[li]+(float(tl)-(tkk[0]+li*d)
                                            )/d*(tkk[li+1]-tkk[li])
                                tfwhm = tr-tl
                                tpi = int((float(tpos)-tkk[0])/d)
                                tpos = tkk[tpi]+(float(tpos)-(tkk[0]+tpi*d)
                                                )/d*(tkk[tpi+1]-tkk[tpi])
                                tpi = int((float(tkmax)-tkk[0])/d)
                                tkmax = tkk[tpi]+(float(tkmax) -
                                                (tkk[0]+tpi*d))/d*(tkk[tpi+1]-tkk[tpi])
                                tpi = int((float(tkmin)-tkk[0])/d)
                                if tpi > 492:
                                    tpi = 492
                                tkmin = tkk[tpi]+(float(tkmin) -
                                                (tkk[0]+tpi*d))/d*(tkk[tpi+1]-tkk[tpi])

                                fev = np.append(fev, self.ev[fi])  # 內容勿動 indent小最內圈if一階
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
                                        str(self.ev[fi])+'\t'+str(tfwhm)+'\t'+str(np.float64(tpos)-1486.6)+'\n')
                                else:
                                    ff.write(
                                        str(self.ev[fi])+'\t'+str(tfwhm)+'\t'+str(np.float64(tpos))+'\n')

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

            ophi = np.arcsin(rpos/(2*m*fev*1.602176634*10**-19)**0.5 /
                            10**-10*(h/2/np.pi))*180/np.pi
            pos = (2*m*fev*1.602176634*10**-19)**0.5 * \
                np.sin((np.float64(self.k_offset)+ophi) /
                    180*np.pi)*10**-10/(h/2/np.pi)
            okmphi = np.arcsin(kmin/(2*m*fev*1.602176634*10**-19) **
                            0.5/10**-10*(h/2/np.pi))*180/np.pi
            kmin = (2*m*fev*1.602176634*10**-19)**0.5 * \
                np.sin((np.float64(self.k_offset)+okmphi) /
                    180*np.pi)*10**-10/(h/2/np.pi)
            okMphi = np.arcsin(kmax/(2*m*fev*1.602176634*10**-19) **
                            0.5/10**-10*(h/2/np.pi))*180/np.pi
            kmax = (2*m*fev*1.602176634*10**-19)**0.5 * \
                np.sin((np.float64(self.k_offset)+okMphi) /
                    180*np.pi)*10**-10/(h/2/np.pi)

            rpos = res(fev, rpos)
            ophi = res(fev, ophi)
            fwhm = res(fev, fwhm)
            pos = res(fev, pos)
            kmin = res(fev, kmin)
            kmax = res(fev, kmax)
            fev = res(fev, fev)

            smfi = res(smfi, smfi)
            tkmin = res(smfi, skmin)
            tkmax = res(smfi, skmax)
            skmin, skmax = [], []
            smaa1 = np.float64(np.arange(4*len(self.ev)).reshape(len(self.ev), 4))
            smaa2 = np.float64(np.arange(8*len(self.ev)).reshape(len(self.ev), 8))
            ti = 0
            ti2 = 0
            for i, v in enumerate(self.ev):
                if i in smfi:
                    skmin.append(tkmin[ti2])
                    skmax.append(tkmax[ti2])
                    ti2 += 1
                    if smfp[i] == 2:  # 2peak以上要改
                        ti += 1
                else:
                    skmin.append((2*m*v*1.602176634*10**-19)**0.5 *
                                np.sin(-0.5/180*np.pi)*10**-10/(h/2/np.pi))
                    skmax.append((2*m*v*1.602176634*10**-19)**0.5 *
                                np.sin(0.5/180*np.pi)*10**-10/(h/2/np.pi))
                a1 = [(skmin[i]+skmax[i])/2, 10, 0.5, int(self.base)]
                a2 = [(skmin[i]+skmax[i])/2, 10, 0.5, int(self.base),
                    (skmin[i]+skmax[i])/2, 10, 0.5, int(self.base)]

                if i in smfi:
                    if smfp[i] == 1:
                        a1 = [rpos[ti], 10, fwhm[ti], int(self.base)]
                    elif smfp[i] == 2:
                        a2 = [rpos[ti-1], 10, fwhm[ti-1],
                            int(self.base), rpos[ti], 10, fwhm[ti], int(self.base)]
                    ti += 1
                smaa1[i, :] = a1
                smaa2[i, :] = a2

            skmin, skmax = np.float64(skmin), np.float64(skmax)
            self.fpr = 1
            try:
                smresult=[]
            except:
                pass
            os.chdir(self.cdir)
        elif ".npz" in file:
            try:
                with np.load(file, 'rb') as f:
                    self.rdd = str(f['path'])
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
                    smcst = f['smcst']
                rpos = np.copy(pos)
                ophi = np.arcsin(rpos/(2*m*fev*1.602176634*10**-19) **
                                0.5/10**-10*(h/2/np.pi))*180/np.pi
                self.fpr = 1
                tbasename = os.path.basename(self.rdd)
                if '.h5' in tbasename:
                    # data = load_h5(self.rdd)
                    # pr_load(data)
                    self.fload = True
                elif '.json' in tbasename:
                    # data = load_json(self.rdd)
                    # pr_load(data)
                    self.fload = True
                elif '.txt' in tbasename:
                    # data = load_txt(self.rdd)
                    # pr_load(data)
                    self.fload = True
            except:
                pass
        self.fev, self.rpos, self.ophi, self.fwhm, self.pos = fev, rpos, ophi, fwhm, pos
        self.skmin, self.skmax = skmin, skmax
        self.smaa1, self.smaa2 = smaa1, smaa2
        self.smfp, self.smfi = smfp, smfi
        if ".vms" in file:
            np.savez(os.path.join(self.cdir, '.MDC_cut', 'mfit.npz'), ko=self.k_offset, fev=fev, rpos=rpos, ophi=ophi, fwhm=fwhm, pos=pos, kmin=kmin,
                    kmax=kmax, skmin=skmin, skmax=skmax, smaa1=smaa1, smaa2=smaa2, smfp=smfp, smfi=smfi)
            self.kmin, self.kmax = kmin, kmax
        elif ".npz" in file:
            np.savez(os.path.join(self.cdir, '.MDC_cut', 'mfit.npz'), ko=self.k_offset, fev=fev, rpos=rpos, ophi=ophi, fwhm=fwhm, pos=pos, kmin=skmin,
                    kmax=skmax, skmin=skmin, skmax=skmax, smaa1=smaa1, smaa2=smaa2, smfp=smfp, smfi=smfi, smresult=smresult, smcst=smcst)
            self.kmin, self.kmax = skmin, skmax
            self.smresult, self.smcst = smresult, smcst
        # self.limg.config(image=self.img[np.random.randint(len(self.img))])
        print('Done')
        self.st.put('Done')
        # self.lmgg.destroy()