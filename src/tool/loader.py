from MDC_cut_utility import *
import os, inspect
import sys, shutil
from typing import Literal, Any, override
import numpy as np
import xarray as xr
import h5py
import json
import zarr

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
            