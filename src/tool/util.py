from MDC_cut_utility import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.collections import PathCollection, QuadMesh
from matplotlib.lines import Line2D
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from matplotlib.text import Annotation
from matplotlib.widgets import SpanSelector, RectangleSelector
from typing import Literal
from tkinter import messagebox
import queue
import gc, tqdm
from scipy.signal import hilbert

# Deprecated class
class MDC_param:
    def __init__(self, ScaleFactor, sc_y, g, scale, npzf, vfe, emf, st, dpath, name, k_offset, value3, ev, phi, data, base, fpr, skmin, skmax, smfp, smfi, smaa1, smaa2, smresult, smcst, mdet):
        self.ScaleFactor = ScaleFactor
        self.sc_y = sc_y
        self.g = g
        self.scale = scale
        self.npzf = npzf
        self.vfe = vfe
        self.emf = emf
        self.st = st
        self.dpath = dpath
        self.name = name
        self.k_offset = k_offset
        self.value3 = value3
        self.ev = ev
        self.phi = phi
        self.data = data
        self.base = base
        self.fpr = fpr
        self.skmin = skmin
        self.skmax = skmax
        self.smfp = smfp
        self.smfi = smfi
        self.smaa1 = smaa1
        self.smaa2 = smaa2
        self.smresult = smresult
        self.smcst = smcst
        shape=data.shape
        tdet=data.data[shape[0]//2, shape[1]//2]
        self.mdet = tdet if mdet==-1 else mdet

class EDC_param:
    def __init__(self, ScaleFactor, sc_y, g, scale, npzf, vfe, emf, st, dpath, name, k_offset, value3, ev, phi, data, base, fpr, semin, semax, sefp, sefi, seaa1, seaa2, edet):
        self.ScaleFactor = ScaleFactor
        self.sc_y = sc_y
        self.g = g
        self.scale = scale
        self.npzf = npzf
        self.vfe = vfe
        self.emf = emf
        self.st = st
        self.dpath = dpath
        self.name = name
        self.k_offset = k_offset
        self.value3 = value3
        self.ev = ev
        self.phi = phi
        self.data = data
        self.base = base
        self.fpr = fpr
        self.semin = semin
        self.semax = semax
        self.sefp = sefp
        self.sefi = sefi
        self.seaa1 = seaa1
        self.seaa2 = seaa2
        # self.seresult = seresult
        # self.secst = secst
        shape=data.shape
        tdet=data.data[shape[0]//2, shape[1]//2]
        self.edet = tdet if edet==-1 else edet

class origin_util:
    def __init__(self, be: np.ndarray, k: np.ndarray,
                 k_offset: tk.StringVar, bb_offset: tk.StringVar, bbk_offset: tk.StringVar,
                 limg: tk.Label, img: list[tk.PhotoImage], st: queue.Queue, im_kernel: int,
                 emf: Literal['KE', 'BE'], data: xr.DataArray, vfe: float, ev: np.ndarray, phi: np.ndarray,
                 pos: np.ndarray, fwhm: np.ndarray, rpos: np.ndarray, ophi: np.ndarray, fev: np.ndarray,
                 epos: np.ndarray, efwhm: np.ndarray, fk: np.ndarray, ffphi: np.ndarray, fphi: np.ndarray,
                 cdir: str, dpath: str, bpath: str, app_name: str, npzf: bool,
                 pos_err: list[float], fwhm_err: list[float],
                 g: tk.Tk, gori: tk.Toplevel, v1: tk.IntVar, v2: tk.IntVar, v3: tk.IntVar, v4: tk.IntVar, v5: tk.IntVar, v6: tk.IntVar,
                 v7: tk.IntVar, v8: tk.IntVar, v9: tk.IntVar, v10: tk.IntVar, v11: tk.IntVar) -> None:
        self.suffix = 'opju'
        self.be, self.k = be, k
        self.k_offset, self.bb_offset, self.bbk_offset = k_offset, bb_offset, bbk_offset
        self.limg, self.img, self.st, self.im_kernel = limg, img, st, im_kernel
        self.emf, self.data, self.vfe, self.ev, self.phi = emf, data, vfe, ev, phi
        self.pos, self.fwhm, self.rpos, self.ophi = pos, fwhm, rpos, ophi
        self.fev, self.epos, self.efwhm = fev, epos, efwhm
        self.fk, self.ffphi, self.fphi = fk, ffphi, fphi
        self.cdir, self.dpath, self.bpath, self.app_name, self.npzf = cdir, dpath, bpath, app_name, npzf
        self.pos_err, self.fwhm_err = pos_err, fwhm_err
        self.g, self.gori = g, gori
        self.v1, self.v2, self.v3 = v1, v2, v3
        self.v4, self.v5, self.v6 = v4, v5, v6
        self.v7, self.v8, self.v9 = v7, v8, v9
        self.v10, self.v11 = v10, v11
    
    def ch_suffix(self, dpath, label: tk.Label, button: tk.Button, *event) -> None:
        if self.suffix == 'opj':
            self.suffix = 'opju'
        else:
            self.suffix = 'opj'
        button.config(text=f'(.{self.suffix})')
        label.config(text=f"{dpath.removesuffix('.h5').removesuffix('.json').removesuffix('.txt')}.{self.suffix}")
        set_center(self.g, self.gori, 0, 0)
    
    @property
    def func(self) -> str:
        body = r'''
def note():
    nt=op.new_notes('Data Info')
    nt.syntax = 0   # Markdown; 0(Normal Text), 1(HTML), 2(Markdown), 3(Origin Rich Text)
    nt.view = 0    # Render Mode; 0(Text Mode), 1(Render Mode)
    nt.append(f'Region')
    nt.append(f'        File Path: {dpath}')
    for i in range(len(dkey)):
        if dkey[i] != 'Description':
            if dkey[i] == 'Path':
                pass
            else:
                nt.append(f'        {dkey[i]}: {dvalue[i]}')
        else:
            for j,k in enumerate(dvalue[i].split('\n')):
                if j == 0:
                    nt.append(f'        {dkey[i]}:')
                    nt.append(f'                {k}')
                else:
                    nt.append(f'                {k}')
    nt.append(f'\nParameters\n'+\
        f'        Energy Mode: {le_mode}\n'+\
        f'        Fermi Energy: {vfe} eV\n'+\
        f'        k offset: {ko} deg\n'+\
        f'        Gaussian Filter Kernel Size: {im_kernel}\n')
    if bpath is not None:
        nt.append(f'        Bare Band Path: {bpath}\n'+\
            f'        Bare Band Offset: {bbo} meV\n'+\
            f'        Bare Band k Ratio: {bbk}\n')
    else:
        nt.append(f'        Bare Band Path: None\n')

def plot2d(x=tx, y=ty, z=tz, x1=[], x2=[], y1=[], y2=[], title='E-Phi (Raw Data)', xlabel=r"\g(f)", ylabel=f'{le_mode}', zlabel='Intensity', xunit="deg", yunit='eV', zunit='Counts'):
    try:
        if title!='E-Phi (Raw Data)':
            if not npzf:
                x = np.sqrt(2*m*tev*1.602176634*10**-19)*np.sin((np.float64(ko)+x)/180*np.pi)*10**-10/(h/2/np.pi)
            xlabel='k'
        else:   # title == E-Phi (Raw Data)
            if npzf:
                title = 'E-k (Sliced Data)'
                xlabel='k'
        if 'Second Derivative' in title:
            z=sdz
        if 'Data plot with pos' in title:
            x1 = pos
            if emf=='KE':
                y1=fev
            else:
                y1= vfe-fev
        if title=='Data plot with pos & bare band':
            x2 = k*bbk
            if emf=='KE':
                y2 = (be - np.float64(bbo))/1000+vfe
            else:
                y2 = (-be + np.float64(bbo))/1000
        if xlabel=='k':
            xunit=r"2\g(p)Å\+(-1)"
        x,y,z = x.flatten(), y.flatten(), z.flatten()
        # create a new book
        wb = op.new_book('w',title)
        # access the first sheet
        sheet = wb[0]
        # add data to the sheet
        sheet.from_list(0, x, lname=xlabel, units=xunit, axis='X')     #col, data, lname='', units='', comments='', axis='', start=0(row offset)
        sheet.from_list(1, y, lname=ylabel, units=yunit, axis='Y')
        sheet.from_list(2, z, lname=zlabel, units=zunit, axis='Z')
        temp_path = os.path.join(cdir, ".MDC_cut", "viridis_2D.otp")
        if os.path.exists(temp_path):
            gr=op.new_graph(title, temp_path)
        else:
            gr=op.new_graph(title, 'TriContgray')
        gr[0].add_plot(sheet, 1, 0, 2)
        if ylabel=='Binding Energy':
            ylm=gr[0].ylim
            gr[0].set_ylim(ylm[1],ylm[0])
            gr[0].set_ylim(step=-1*float(ylm[2]))
        if len(x1) != 0:
            sheet.from_list(3, x1, lname='x1', units=xunit, axis='X')
            sheet.from_list(4, y1, lname='y1', units=yunit, axis='Y')
            g1=gr[0].add_plot(sheet, 4, 3,type='s')
            g1.symbol_size = 5
            g1.symbol_kind = 2
        if len(x2) != 0:
            sheet.from_list(5, x2, lname='x2', units=xunit, axis='X')
            sheet.from_list(6, y2, lname='y2', units=yunit, axis='Y')
            g2=gr[0].add_plot(sheet, 6, 5,type='l')
            g2.symbol_size = 5
            g2.symbol_kind = 2
            g2.color = 'red'
        gr[0].rescale()
        wb.show = False
    except Exception as e:
        print(f"Error in plot2d: {e}")
        try:
            print(title)
        except:
            pass

def plot1d(x=[1,2,3], y1=[1,2,3], y2=[], y1err=[], y2err=[], title='title', xlabel='x', ylabel='y', ylabel1='y1', ylabel2='y2', xunit='arb', yunit='arb'):
    try:
        # create a new book
        wb = op.new_book('w',title)
        # access the first sheet
        sheet = wb[0]
        # add data to the sheet
        if ylabel1 == 'y1':
            ylabel1 = ylabel
        sheet.from_list(0, x, lname=xlabel, units=xunit, axis='X')     #col, data, lname='', units='', comments='', axis='', start=0(row offset)
        sheet.from_list(1, y1, lname=ylabel1, units=yunit, axis='Y')
        sheet.from_list(2, y1err, lname='y1err', units=yunit, axis='E')
        gr=op.new_graph(title, 'scatter')
        if len(y1err) != 0:
            gr[0].add_plot(sheet, 1, 0, 2)
            g1=gr[0].add_plot(sheet, 1, 0)
        else:
            g1=gr[0].add_plot(sheet, 1, 0)
        g1.symbol_size = 5
        g1.symbol_kind = 2
        if len(y2) != 0:
            if len(y1err) != 0:
                sheet.from_list(3, y2, lname=ylabel2, units=yunit, axis='Y')
                sheet.from_list(4, y2err, lname='y2err', units=yunit, axis='E')
                gr[0].add_plot(sheet, 3, 0, 4)
                g2=gr[0].add_plot(sheet, 3, 0)
            else:
                sheet.from_list(2, y2, lname=ylabel2, units=yunit, axis='Y')
                g2=gr[0].add_plot(sheet, 2, 0)
            g2.symbol_size = 5
            g2.symbol_kind = 2
            g2.color = 'red'
            gr[0].label('yl').text = f'{ylabel} ({yunit})'
        if xlabel=='Binding Energy':
            xlm=gr[0].xlim
            gr[0].set_xlim(xlm[1],xlm[0])
            gr[0].set_xlim(step=-1*float(xlm[2]))
        gr[0].rescale()
        wb.show = False
    except Exception as e:
        print(f"Error in plot1d: {e}")
        try:
            print(title)
        except:
            pass
'''+rf'''
def save(format='{self.suffix}'):
    """
    Save the Origin data in .opj format.
    Can be saved in .opju format as well.
    """
    tbasename = os.path.basename(dpath)
    if '.h5' in tbasename:
        op.save(dpath.removesuffix('.h5').replace("/","\\")+'.'+format)
    elif '.json' in tbasename:
        op.save(dpath.removesuffix('.json').replace("/","\\")+'.'+format)
    elif '.txt' in tbasename:
        op.save(dpath.removesuffix('.txt').replace("/","\\")+'.'+format)
    elif '.npz' in tbasename:
        op.save(dpath.removesuffix('.npz').replace("/","\\")+'.'+format)
'''
        return body
    
    def pr_exp_origin(self) -> None:
        ex_raw,ex_ek,ex_mp,ex_mf,ex_ep,ex_ef,ex_ser,ex_sei,ex_dpp,ex_dppbb,ex_sd='','','','','','','','','','',''
        cmdlist=dict({0:f'{ex_raw}',1:f'{ex_ek}',2:f'{ex_mp}',3:f'{ex_mf}',4:f'{ex_ep}',5:f'{ex_ef}',6:f'{ex_ser}',7:f'{ex_sei}',8:f'{ex_dpp}',9:f'{ex_dppbb}',10:f'{ex_sd}'})
        no=[]
        be, k, vfe = self.be, self.k, self.vfe
        k_offset, bb_offset, bbk_offset = self.k_offset, self.bb_offset, self.bbk_offset
        pos, fwhm, rpos = self.pos, self.fwhm, self.rpos
        fev, epos, efwhm = self.fev, self.epos, self.efwhm
        fk, ffphi, fphi = self.fk, self.ffphi, self.fphi
        emf, ev, phi = self.emf, self.ev, self.phi
        cdir, dpath = self.cdir, self.dpath
        pos_err, fwhm_err = self.pos_err, self.fwhm_err
        m, h = 9.1093837015*10**-31, 6.62607015*10**-34
        try:
            cmdlist[0]=f'''plot2d()\n'''
        except:
            no.append(0)
        try:
            cmdlist[1]=f'''plot2d(title='E-k (Processed Data)')\n'''
        except:
            no.append(1)
        try:
            ophi = np.arcsin(rpos/(2*m*fev*1.602176634*10**-19)**0.5 /
                            10**-10*(h/2/np.pi))*180/np.pi
            pos = (2*m*fev*1.602176634*10**-19)**0.5 * \
                np.sin((np.float64(k_offset.get())+ophi)/180*np.pi)*10**-10/(h/2/np.pi)
            cmdlist[2]=rf'''plot1d(x={pre_process((vfe-fev)*1000)}, y1={pre_process(pos)}, y1err={pre_process(pos_err)}, title='MDC Fit Position', xlabel='Binding Energy', ylabel='k', xunit='meV', yunit=r"2\g(p)Å\+(-1)")
'''
        except:
            no.append(2)
        try:
            cmdlist[3]=rf'''plot1d(x={pre_process((vfe-fev)*1000)}, y1={pre_process(fwhm)}, y1err={pre_process(fwhm_err)}, title='MDC Fit FWHM', xlabel='Binding Energy', ylabel='k', xunit='meV', yunit=r"2\g(p)Å\+(-1)")
'''
        except:
            no.append(3)
        try:
            cmdlist[4]=rf'''plot1d(x={pre_process(fk)}, y1={pre_process((vfe-epos)*1000)}, title='EDC Fit Position', xlabel='k', ylabel='Binding Energy', xunit=r"2\g(p)Å\+(-1)", yunit='meV')
'''
        except:
            no.append(4)
        try:
            if len(efwhm)==0:
                raise ValueError("efwhm is empty")
            cmdlist[5]=rf'''plot1d(x={pre_process(fk)}, y1={pre_process(efwhm)}, title='EDC Fit FWHM', xlabel='k', ylabel='Binding Energy', xunit=r"2\g(p)Å\+(-1)", yunit='meV')
'''
        except:
            no.append(5)
        try:
            ophi = np.arcsin(rpos/(2*m*fev*1.602176634*10**-19)**0.5 /
                            10**-10*(h/2/np.pi))*180/np.pi
            pos = (2*m*fev*1.602176634*10**-19)**0.5 * \
                np.sin((np.float64(k_offset.get())+ophi)/180*np.pi)*10**-10/(h/2/np.pi)
            x = (vfe-fev)*1000
            y = pos
            yy = interp(pos, k*np.float64(bbk_offset.get()), be -
                        # interp x into be,k set
                        np.float64(bb_offset.get()))
            x = (vfe-fev)*1000
            rx = x
            ry = -(x+yy)
            tbe = (vfe-fev)*1000
            x = interp(tbe, -be+np.float64(bb_offset.get()),
                        k*np.float64(bbk_offset.get()))
            y = interp(x, k*np.float64(bbk_offset.get()),
                        -be+np.float64(bb_offset.get()))
            xx = np.diff(x)
            yy = np.diff(y)

            # eliminate vf in gap
            for i in range(len(yy)):
                if yy[i]/xx[i] > 20000:
                    yy[i] = 0
            v = yy/xx
            # v = np.append(v, v[-1])  # fermi velocity
            v=interp(pos,x[0:-1]+xx/2,v)
            y1err = pos_err*interp(pos, k*np.float64(bbk_offset.get()), be)/pos
            yy = np.abs(v*fwhm/2)
            xx = tbe

            ix = xx
            iy = yy
            ix=(tbe-tbe[-1])*-1
            cix=np.append(ix+ix[0],ix)
            tix=cix[0:len(cix)-1]*-1
            # kx=ix
            kx = np.append(cix,tix[::-1])
            ky = np.linspace(0, 1, len(kx))
            ciy=np.append(iy*0+np.mean(iy),iy)
            tiy=ciy[0:len(ciy)-1]
            ciy = np.append(ciy,tiy[::-1])

            #for imaginary part
            ix=(tbe-tbe[-1])*-1
            cix=np.append(ix+ix[0],ix)
            tix=cix[0:len(cix)-1]*-1
            kx = np.append(cix,tix[::-1])
            ky = np.linspace(0, 1, len(kx))
            cry=np.append(ry*0,ry)
            tcry=cry[0:len(cry)-1]*-1
            cry = np.append(cry,tcry[::-1])

            # Hilbert transform
            analytic_signal_r = hilbert(cry)
            analytic_signal_i = hilbert(ciy)
            # Reconstructed real and imaginary parts
            reconstructed_real = np.imag(analytic_signal_i)
            reconstructed_imag = -np.imag(analytic_signal_r)
            cmdlist[6]=rf'''plot1d(x={pre_process((vfe-fev)*1000)}, y1={pre_process(-1*((vfe-fev)*1000+interp(pos, k*np.float64(bbk_offset.get()), be - np.float64(bb_offset.get()))))}, y2={pre_process(reconstructed_real[len(ix):2*len(ix)]+(ry-np.mean(reconstructed_real[len(ix):2*len(ix)])))}, y1err={pre_process(y1err)}, title='Self Energy Real Part', xlabel='Binding Energy', ylabel=r"Re \g(S)", ylabel1=r"Re \g(S)", ylabel2=r"Re \g(S)\-(KK)=KK(Im \g(S))", xunit='meV', yunit='meV')
'''
        except:
            no.append(6)
        try:
            ophi = np.arcsin(rpos/(2*m*fev*1.602176634*10**-19)**0.5 /
                            10**-10*(h/2/np.pi))*180/np.pi
            pos = (2*m*fev*1.602176634*10**-19)**0.5 * \
                np.sin((np.float64(k_offset.get())+ophi)/180*np.pi)*10**-10/(h/2/np.pi)
            x = (vfe-fev)*1000
            y = pos
            yy = interp(pos, k*np.float64(bbk_offset.get()), be -
                        # interp x into be,k set
                        np.float64(bb_offset.get()))
            x = (vfe-fev)*1000
            rx = x
            ry = -(x+yy)
            tbe = (vfe-fev)*1000
            x = interp(tbe, -be+np.float64(bb_offset.get()),
                        k*np.float64(bbk_offset.get()))
            y = interp(x, k*np.float64(bbk_offset.get()),
                        -be+np.float64(bb_offset.get()))
            xx = np.diff(x)
            yy = np.diff(y)

            # eliminate vf in gap
            for i in range(len(yy)):
                if yy[i]/xx[i] > 20000:
                    yy[i] = 0
            v = yy/xx
            # v = np.append(v, v[-1])  # fermi velocity
            v=interp(pos,x[0:-1]+xx/2,v)
            yy = np.abs(v*fwhm/2)
            y1err = fwhm_err/2*np.abs(v)
            xx = tbe

            ix = xx
            iy = yy
            ix=(tbe-tbe[-1])*-1
            cix=np.append(ix+ix[0],ix)
            tix=cix[0:len(cix)-1]*-1
            # kx=ix
            kx = np.append(cix,tix[::-1])
            ky = np.linspace(0, 1, len(kx))
            ciy=np.append(iy*0+np.mean(iy),iy)
            tiy=ciy[0:len(ciy)-1]
            ciy = np.append(ciy,tiy[::-1])

            #for imaginary part
            ix=(tbe-tbe[-1])*-1
            cix=np.append(ix+ix[0],ix)
            tix=cix[0:len(cix)-1]*-1
            kx = np.append(cix,tix[::-1])
            ky = np.linspace(0, 1, len(kx))
            cry=np.append(ry*0,ry)
            tcry=cry[0:len(cry)-1]*-1
            cry = np.append(cry,tcry[::-1])

            # Hilbert transform
            analytic_signal_r = hilbert(cry)
            analytic_signal_i = hilbert(ciy)
            # Reconstructed real and imaginary parts
            reconstructed_real = np.imag(analytic_signal_i)
            reconstructed_imag = -np.imag(analytic_signal_r)
            cmdlist[7]=rf'''plot1d(x={pre_process((vfe-fev)*1000)}, y1={pre_process(iy)}, y2={pre_process(reconstructed_imag[len(ix):2*len(ix)]+(iy-np.mean(reconstructed_imag[len(ix):2*len(ix)])))}, y1err={pre_process(y1err)}, title='Self Energy Imaginary Part', xlabel='Binding Energy', ylabel=r"Im \g(S)", ylabel1=r"Im \g(S)", ylabel2=r"Im \g(S)\-(KK)=KK(Re \g(S))", xunit='meV', yunit='meV')
'''
        except:
            no.append(7)
        try:
            if len(pos)==0:
                raise ValueError("pos is empty")
            cmdlist[8]=f'''plot2d(title='Data plot with pos')\n'''
        except:
            no.append(8)
        try:
            x2 = k*float(bbk_offset.get())
            if emf=='KE':
                y2 = (be - float(bb_offset.get()))/1000+vfe
            else:
                y2 = (-be + float(bb_offset.get()))/1000
            cmdlist[9]=f'''plot2d(title='Data plot with pos & bare band')\n'''
        except:
            no.append(9)
        try:
            cmdlist[10]=f'''plot2d(title='Second Derivative (Processed Data)')\n'''
        except:
            no.append(10)
        self.cmdlist, self.no = cmdlist, no
            
    def exp_origin(self, *e):
        if os.name == 'posix':
            messagebox.showinfo("Info", "OriginPro project export is only supported on Windows OS.")
            return
        app_name, cdir, npzf, bpath, dpath, emf = self.app_name, self.cdir, self.npzf, self.bpath, self.dpath, self.emf
        pos, fwhm, rpos = self.pos, self.fwhm, self.rpos
        be, k, vfe = self.be, self.k, self.vfe
        epos, efwhm = self.epos, self.efwhm
        fev = self.fev
        fphi = self.fphi
        k_offset, bb_offset, bbk_offset = self.k_offset, self.bb_offset, self.bbk_offset
        im_kernel = self.im_kernel
        limg, img, st = self.limg, self.img, self.st
        gori = self.gori
        v1, v2, v3, v4, v5, v6 = self.v1, self.v2, self.v3, self.v4, self.v5, self.v6
        v7, v8, v9, v10, v11 = self.v7, self.v8, self.v9, self.v10, self.v11
        m, h = 9.1093837015*10**-31, 6.62607015*10**-34
        
        origin_temp_var = f'''from {app_name} import *
import originpro as op

cdir = r"{cdir}"
npzf = {npzf}
dpath = r"{dpath}"      # Data Path
emf = r"{emf}"             # Energy Mode: KE or BE
ko = {k_offset.get()}
bbo = {bb_offset.get()}
bbk = {bbk_offset.get()}
vfe = {vfe}
im_kernel = {im_kernel}     # Gaussian Filter Kernel Size
nan = np.nan
'''
        try:
            origin_temp_var += f'''
bpath = r"{bpath}"         # Bare Band Path
be = np.float64({pre_process(be)})
k = np.float64({pre_process(k)})
'''
        except:
            origin_temp_var += f'''
bpath = None
'''
        try:
            ophi = np.arcsin(rpos/(2*m*fev*1.602176634*10**-19)**0.5 /
                            10**-10*(h/2/np.pi))*180/np.pi
            pos = (2*m*fev*1.602176634*10**-19)**0.5 * \
                np.sin((np.float64(k_offset.get())+ophi)/180*np.pi)*10**-10/(h/2/np.pi)
            origin_temp_var += f'''
fev = np.float64({pre_process(np.asarray(fev, dtype=np.float64))})
pos = np.float64({pre_process(np.asarray(pos, dtype=np.float64))})
fwhm = np.float64({pre_process(np.asarray(fwhm, dtype=np.float64))})
'''
        except: pass
        try:
            ffphi = np.float64(k_offset.get())+fphi
            fk = (2*m*epos*1.602176634*10**-19)**0.5 * \
                np.sin(ffphi/180*np.pi)*10**-10/(h/2/np.pi)
            origin_temp_var += f'''
fk = np.float64({pre_process(np.asarray(fk, dtype=np.float64))})
epos = np.float64({pre_process(np.asarray(epos, dtype=np.float64))})
efwhm = np.float64({pre_process(np.asarray(efwhm, dtype=np.float64))})
'''
        except: pass
        if '.h5' in os.path.basename(dpath):
            tload = f'''
data = load_h5(dpath)        
'''
        elif '.json' in os.path.basename(dpath):
            tload = f'''
data = load_json(dpath)
'''
        elif '.txt' in os.path.basename(dpath):
            tload = f'''
data = load_txt(dpath)
'''
        elif '.npz' in os.path.basename(dpath):
            tload = f'''
data = load_npz(dpath)
'''
        origin_temp_var += tload
        origin_temp_var += f'''
dvalue = list(data.attrs.values())
dkey = list(data.attrs.keys())
ev, phi = data.indexes.values()
ev, phi = np.float64(ev), np.float64(phi)

if emf=='KE':
    le_mode='Kinetic Energy'
    tx, ty = np.meshgrid(phi, ev)
    tev = ty.copy()
else:
    le_mode='Binding Energy'
    tx, ty = np.meshgrid(phi, vfe-ev)
    tev = vfe-ty.copy()
tz = data.to_numpy()
sdz = laplacian_filter(data.to_numpy(), im_kernel)
'''
        origin_temp_exec = r'''
op.new()
op.set_show(True)

'''
        origin_temp_save = r'''
note()
save()
'''
        cl=[v1.get(),v2.get(),v3.get(),v4.get(),v5.get(),v6.get(),v7.get(),v8.get(),v9.get(),v10.get(),v11.get()]
        gori.destroy()
        for i in self.cmdlist.keys():
            if cl[i]==1:
                origin_temp_exec+=self.cmdlist[i]
            
        with open(cdir+os.sep+'origin_temp.py', 'w', encoding='utf-8') as f:
            f.write(origin_temp_var+self.func+origin_temp_exec+origin_temp_save)
        f.close()
        def j():
            # os.system(f'code {cdir+r"\origin_temp.py"}')
            limg.config(image=img[np.random.randint(len(img))])
            print('Exporting to Origin...')
            st.put('Exporting to Origin...')
            temp=os.sep+"origin_temp.py"
            os.system(f'python -W ignore::SyntaxWarning -W ignore::UserWarning "{cdir+temp}"')
            os.system(f'del "{cdir+temp}"')
            limg.config(image=img[np.random.randint(len(img))])
            print('Exported to Origin')
            st.put('Exported to Origin')
        threading.Thread(target=j,daemon=True).start()

    def patch_origin(self, *e):
        limg, img, st = self.limg, self.img, self.st
        limg.config(image=img[np.random.randint(len(img))])
        def j():
            print('Patching OriginPro...')
            st.put('Patching OriginPro...')
            exe=rf"\Origin.exe"
            cmd=f'start "" cmd /C "dir "{exe}" /s"'
            result = os.popen(cmd)
            context = result.read()
            for line in context.splitlines():
                if '的目錄' in line or 'Directory of' in line:
                    path = line.removeprefix('Directory of ')
                    path = line.removesuffix(' 的目錄')
                    path = path.removeprefix(" ")
                    path = rf"{path}"
                    path = rf"{path}{exe}"
                    if path.split(os.sep)[-2] != 'Crack':
                        ori_temp_path = path.removesuffix(os.sep+path.split(os.sep)[-1])
                        print('Origin Path: '+ori_temp_path)
                        os.system(f"\"{path}\"")
            result.close()
            print('Patching OriginPro...Done')
            st.put('Patching OriginPro...Done')
        threading.Thread(target=j,daemon=True).start()

class motion:
    def __init__(self, scale: float, value: tk.StringVar, value1: tk.StringVar, value2: tk.StringVar, k_offset: tk.StringVar,
                 be: np.ndarray, k: np.ndarray, bb_offset: tk.StringVar, bbk_offset: tk.StringVar,
                 ao: Axes, out: FigureCanvasTkAgg, figy: float|int,
                 rcx: Axes, rcy: Axes, xdata: tk.Label, ydata: tk.Label,
                 emf: Literal['KE', 'BE'], data: xr.DataArray, vfe: float, ev: np.ndarray, phi: np.ndarray,
                 pos: np.ndarray, fwhm: np.ndarray, rpos: np.ndarray, ophi: np.ndarray, fev: np.ndarray,
                 epos: np.ndarray, efwhm: np.ndarray, fk: np.ndarray, ffphi: np.ndarray, fphi: np.ndarray,
                 mp: int, ep: int, mf: int, ef: int, xl: tuple[float], yl: tuple[float],
                 tb0: PathCollection, tb0_: PathCollection, tb1: PathCollection, tb1_: PathCollection, tb2: Line2D
                 ) -> None:
        self.mof = 1
        self.scale = scale
        self.value, self.value1, self.value2, self.k_offset = value, value1, value2, k_offset
        self.be, self.k, self.bb_offset, self.bbk_offset = be, k, bb_offset, bbk_offset
        self.ao, self.out, self.figy = ao, out, figy
        self.rcx, self.rcy, self.xdata, self.ydata = rcx, rcy, xdata, ydata
        self.emf, self.data, self.vfe, self.ev, self.phi = emf, data, vfe, ev, phi
        self.pos, self.fwhm, self.rpos, self.ophi, self.fev = pos, fwhm, rpos, ophi, fev
        self.epos, self.efwhm, self.fk, self.ffphi, self.fphi = epos, efwhm, fk, ffphi, fphi
        self.mp, self.ep, self.mf, self.ef, self.xl, self.yl = mp, ep, mf, ef, xl, yl
        self.tb0, self.tb0_, self.tb1, self.tb1_, self.tb2 = tb0, tb0_, tb1, tb1_, tb2
    
    def size(self, s: int) -> int:
        return int(self.scale*s)
    
    def move(self, event):
        if event.xdata != None:
            self.out.get_tk_widget().config(cursor="crosshair")
            try:
                self.out.get_tk_widget().delete('rec')
            except:
                pass
            if self.mof == -1 and self.value1.get() == '---Plot2---' and self.value2.get() != 'Real & Imaginary' and 'KK Transform' not in self.value2.get() and 'MDC Curves' not in self.value.get():
                px2, py2 = event.x, event.y
                if os.name == 'nt':
                    self.out.get_tk_widget().create_rectangle((self.px1, int(self.figy*100)-self.py1), (px2, int(self.figy*100)-py2),
                                                    outline='black', width=2, tag='rec')
                elif os.name == 'posix':
                    self.out.get_tk_widget().create_rectangle((self.px1, int(self.figy*100*self.scale)-self.py1), (px2, int(self.figy*100*self.scale)-py2),
                                                    outline='black', width=2, tag='rec')
            if self.value is not None and self.ao is not None:
                if self.value.get() == 'Raw Data':
                    if event.inaxes:
                        cxdata = event.xdata
                        cydata = event.ydata
                        xf = (cxdata >= self.ao.get_xlim()[0] and cxdata <= self.ao.get_xlim()[1])
                        if self.emf=='KE':
                            yf = (cydata >= self.ao.get_ylim()[0] and cydata <= self.ao.get_ylim()[1])
                        else:
                            yf = (cydata <= self.ao.get_ylim()[0] and cydata >= self.ao.get_ylim()[1])
                        if xf and yf:
                            if self.emf=='KE':
                                dx = self.data.sel(
                                    eV=cydata, method='nearest').to_numpy().reshape(len(self.phi))
                            else:
                                dx = self.data.sel(
                                    eV=self.vfe-cydata, method='nearest').to_numpy().reshape(len(self.phi))
                            dy = self.data.sel(
                                phi=cxdata, method='nearest').to_numpy().reshape(len(self.ev))
                            if self.rcx is not None:
                                self.rcx.clear()
                                self.rcy.clear()
                                self.rcx.set_title('            Raw Data', font='Arial', fontsize=self.size(16))
                                self.rcx.plot(self.phi, dx, c='black')
                                if self.emf=='KE':
                                    self.rcy.plot(dy, self.ev, c='black')
                                else:
                                    self.rcy.plot(dy, self.vfe-self.ev, c='black')
                                self.rcx.set_xticks([])
                                self.rcy.set_yticks([])
                                self.rcx.set_xlim(self.ao.get_xlim())
                                self.rcy.set_ylim(self.ao.get_ylim())
                            self.out.draw()
            self.xdata.config(text='xdata:'+str(' %.3f' % event.xdata))
            self.ydata.config(text='ydata:'+str(' %.3f' % event.ydata))
        else:
            if self.value.get() == 'Raw Data':
                if self.rcx is not None:
                    self.rcx.clear()
                    self.rcy.clear()
                    self.rcx.set_xticks([])
                    self.rcx.set_yticks([])
                    self.rcy.set_xticks([])
                    self.rcy.set_yticks([])
                    self.rcx.set_title('            Raw Data', font='Arial', fontsize=self.size(16))
                    self.out.draw()
            self.out.get_tk_widget().config(cursor="")
            self.xdata.config(text='xdata:')
            self.ydata.config(text='ydata:')

    def press(self, event):
        m, h = 9.10938356*10**-31, 6.62607015*10**-34
        if event.button == 1:
            self.x1, self.y1 = event.xdata, event.ydata
            if self.value1.get() == '---Plot2---' and self.value2.get() != 'Real & Imaginary' and 'KK Transform' not in self.value2.get() and 'MDC Curves' not in self.value.get():
                self.px1, self.py1 = event.x, event.y
                self.mof = -1
        elif event.button == 3 and self.value1.get() == '---Plot2---' and self.value2.get() != 'Real & Imaginary' and 'KK Transform' not in self.value2.get() and 'MDC Curves' not in self.value.get():
            if self.value2.get() == '---Plot3---':
                if self.ao:
                    self.ao.set_xlim(self.xl)
                    self.ao.set_ylim(self.yl)
                    self.out.draw()
            else:
                self.ao.set_xlim(self.xl)
                self.ao.set_ylim(self.yl)
                try:
                    if self.mp == 1:
                        self.tb0.remove()
                        if self.emf=='KE':
                            self.tb0 = self.ao.scatter(self.pos, self.fev, marker='.', s=self.scale*self.scale*0.3, c='black')
                        else:
                            self.tb0 = self.ao.scatter(self.pos, self.vfe-self.fev, marker='.', s=self.scale*self.scale*0.3, c='black')
                            
                    if self.mf == 1:
                        self.tb0_.remove()
                        ophimin = np.arcsin(
                            (self.rpos-self.fwhm/2)/np.sqrt(2*m*self.fev*1.602176634*10**-19)/10**-10*(h/2/np.pi))*180/np.pi
                        ophimax = np.arcsin(
                            (self.rpos+self.fwhm/2)/np.sqrt(2*m*self.fev*1.602176634*10**-19)/10**-10*(h/2/np.pi))*180/np.pi
                        posmin = np.sqrt(2*m*self.fev*1.602176634*10**-19)*np.sin(
                            (np.float64(self.k_offset.get())+ophimin)/180*np.pi)*10**-10/(h/2/np.pi)
                        posmax = np.sqrt(2*m*self.fev*1.602176634*10**-19)*np.sin(
                            (np.float64(self.k_offset.get())+ophimax)/180*np.pi)*10**-10/(h/2/np.pi)
                        if self.emf=='KE':
                            self.tb0_ = self.ao.scatter([posmin, posmax], [
                                            self.fev, self.fev], marker='|', c='grey', s=self.scale*self.scale*10, alpha=0.8)
                        else:
                            self.tb0_ = self.ao.scatter([posmin, posmax], [self.vfe-self.fev, self.vfe-self.fev], marker='|', c='grey', s=self.scale*self.scale*10, alpha=0.8)
                except:
                    pass
                try:
                    if self.ep == 1:
                        self.tb1.remove()
                        if self.emf=='KE':
                            self.tb1 = self.ao.scatter(self.fk, self.epos, marker='.', s=self.scale*self.scale*0.3, c='black')
                        else:
                            self.tb1 = self.ao.scatter(self.fk, self.vfe-self.epos, marker='.', s=self.scale*self.scale*0.3, c='black')
                            
                    if self.ef == 1:
                        self.tb1_.remove()
                        eposmin = self.epos-self.efwhm/2
                        eposmax = self.epos+self.efwhm/2
                        if self.emf=='KE':
                            self.tb1_ = self.ao.scatter(
                                [self.fk, self.fk], [eposmin, eposmax], marker='_', c='grey', s=self.scale*self.scale*10, alpha=0.8)
                        else:
                            self.tb1_ = self.ao.scatter(
                                [self.fk, self.fk], [self.vfe-eposmin, self.vfe-eposmax], marker='_', c='grey', s=self.scale*self.scale*10, alpha=0.8)
                except:
                    pass
                try:
                    if self.value2.get() == 'Data Plot with Pos and Bare Band' and self.k is not None:
                        self.tb2.remove()
                        if self.emf=='KE':
                            self.tb2, = self.ao.plot(self.k*np.float64(self.bbk_offset.get()), (self.be -
                                        np.float64(self.bb_offset.get()))/1000+self.vfe, linewidth=self.scale*0.3, c='red', linestyle='--')
                        else:
                            print('plotting bb0')
                            self.tb2, = self.ao.plot(self.k*np.float64(self.bbk_offset.get()), (-self.be +
                                        np.float64(self.bb_offset.get()))/1000, linewidth=self.scale*0.3, c='red', linestyle='--')
                            print('plotted bb0')
                except:
                    pass
                self.out.draw()
            self.mof = 1

    def release(self, event):
        m, h = 9.10938356*10**-31, 6.62607015*10**-34
        try:
            self.out.get_tk_widget().delete('rec')
        except:
            pass
        if event.button == 1 and self.mof == -1 and self.value1.get() == '---Plot2---' and self.value2.get() != 'Real & Imaginary' and 'KK Transform' not in self.value2.get() and 'MDC Curves' not in self.value.get():
            x2, y2 = event.xdata, event.ydata
            if self.x1 is None or x2 is None:
                self.mof = 1
                return
            if self.value2.get() == '---Plot3---':
                if self.ao:
                    self.ao.set_xlim(sorted([self.x1, x2]))
                    if self.emf=='KE':    
                        self.ao.set_ylim(sorted([self.y1, y2]))
                    else:
                        self.ao.set_ylim(sorted([self.y1, y2], reverse=True))
                    self.out.draw()
            else:
                self.ao.set_xlim(sorted([self.x1, x2]))
                if self.emf=='KE':    
                    self.ao.set_ylim(sorted([self.y1, y2]))
                else:
                    self.ao.set_ylim(sorted([self.y1, y2], reverse=True))
                if abs(self.x1-x2) < (self.xl[1]-self.xl[0])/3*2 or abs(self.y1-y2) < (self.yl[1]-self.yl[0])/3*2:
                    try:
                        if self.mp == 1:
                            self.tb0.remove()
                        if self.mf == 1:
                            self.tb0_.remove()
                    except:
                        pass
                    try:
                        if self.ep == 1:
                            self.tb1.remove()
                        if self.ef == 1:
                            self.tb1_.remove()
                    except:
                        pass
                    try:
                        self.tb2.remove()
                    except:
                        pass
                    if self.value2.get() == 'Data Plot with Pos' or self.value2.get() == 'Data Plot with Pos and Bare Band':
                        try:
                            if self.mp == 1:
                                if self.emf=='KE':
                                    self.tb0 = self.ao.scatter(
                                        self.pos, self.fev, marker='.', s=self.scale*self.scale*30, c='black')
                                else:
                                    self.tb0 = self.ao.scatter(
                                        self.pos, self.vfe-self.fev, marker='.', s=self.scale*self.scale*30, c='black')
                            if self.mf == 1:
                                ophimin = np.arcsin(
                                    (self.rpos-self.fwhm/2)/np.sqrt(2*m*self.fev*1.602176634*10**-19)/10**-10*(h/2/np.pi))*180/np.pi
                                ophimax = np.arcsin(
                                    (self.rpos+self.fwhm/2)/np.sqrt(2*m*self.fev*1.602176634*10**-19)/10**-10*(h/2/np.pi))*180/np.pi
                                posmin = np.sqrt(2*m*self.fev*1.602176634*10**-19)*np.sin(
                                    (np.float64(self.k_offset.get())+ophimin)/180*np.pi)*10**-10/(h/2/np.pi)
                                posmax = np.sqrt(2*m*self.fev*1.602176634*10**-19)*np.sin(
                                    (np.float64(self.k_offset.get())+ophimax)/180*np.pi)*10**-10/(h/2/np.pi)
                                if self.emf=='KE':
                                    self.tb0_ = self.ao.scatter([posmin, posmax], [
                                                    self.fev, self.fev], marker='|', c='grey', s=self.scale*self.scale*50, alpha=0.8)
                                else:
                                    self.tb0_ = self.ao.scatter([posmin, posmax], [self.vfe-self.fev, self.vfe-self.fev], marker='|', c='grey', s=self.scale*self.scale*50, alpha=0.8)

                        except:
                            pass
                        try:
                            if self.ep == 1:
                                if self.emf=='KE':
                                    self.tb1 = self.ao.scatter(
                                        self.fk, self.epos, marker='.', s=self.scale*self.scale*30, c='black')
                                else:
                                    self.tb1 = self.ao.scatter(
                                        self.fk, self.vfe-self.epos, marker='.', s=self.scale*self.scale*30, c='black')
                            if self.ef == 1:
                                eposmin = self.epos-self.efwhm/2
                                eposmax = self.epos+self.efwhm/2
                                if self.emf=='KE':
                                    self.tb1_ = self.ao.scatter(
                                        [self.fk, self.fk], [eposmin, eposmax], marker='_', c='grey', s=self.scale*self.scale*50, alpha=0.8)
                                else:
                                    self.tb1_ = self.ao.scatter(
                                    [self.fk, self.fk], [self.vfe-eposmin, self.vfe-eposmax], marker='_', c='grey', s=self.scale*self.scale*50, alpha=0.8)
                        except:
                            pass
                        if self.value2.get() == 'Data Plot with Pos and Bare Band' and self.k is not None:
                            if self.emf=='KE':
                                self.tb2, = self.ao.plot(self.k*np.float64(self.bbk_offset.get()), (self.be -
                                            np.float64(self.bb_offset.get()))/1000+self.vfe, linewidth=self.scale*5, c='red', linestyle='--')
                            else:
                                self.tb2, = self.ao.plot(self.k*np.float64(self.bbk_offset.get()), (-self.be +
                                            np.float64(self.bb_offset.get()))/1000, linewidth=self.scale*5, c='red', linestyle='--')
                else:
                    try:
                        if self.mp == 1:
                            self.tb0.remove()
                            if self.emf=='KE':
                                self.tb0 = self.ao.scatter(self.pos, self.fev, marker='.',
                                                s=0.3, c='black')
                            else:
                                self.tb0 = self.ao.scatter(self.pos, self.vfe-self.fev, marker='.',
                                                s=0.3, c='black')
                        if self.mf == 1:
                            self.tb0_.remove()
                            ophimin = np.arcsin(
                                (self.rpos-self.fwhm/2)/np.sqrt(2*m*self.fev*1.602176634*10**-19)/10**-10*(h/2/np.pi))*180/np.pi
                            ophimax = np.arcsin(
                                (self.rpos+self.fwhm/2)/np.sqrt(2*m*self.fev*1.602176634*10**-19)/10**-10*(h/2/np.pi))*180/np.pi
                            posmin = np.sqrt(2*m*self.fev*1.602176634*10**-19)*np.sin(
                                (np.float64(self.k_offset.get())+ophimin)/180*np.pi)*10**-10/(h/2/np.pi)
                            posmax = np.sqrt(2*m*self.fev*1.602176634*10**-19)*np.sin(
                                (np.float64(self.k_offset.get())+ophimax)/180*np.pi)*10**-10/(h/2/np.pi)
                            if self.emf=='KE':
                                self.tb0_ = self.ao.scatter([posmin, posmax], [
                                                self.fev, self.fev], marker='|', c='grey', s=self.scale*self.scale*10, alpha=0.8)
                            else:
                                self.tb0_ = self.ao.scatter([posmin, posmax], [self.vfe-self.fev, self.vfe-self.fev], marker='|', c='grey', s=self.scale*self.scale*10, alpha=0.8)
                    except:
                        pass
                    try:
                        if self.ep == 1:
                            self.tb1.remove()
                            if self.emf=='KE':
                                self.tb1 = self.ao.scatter(self.fk, self.epos, marker='.',
                                                s=0.3, c='black')
                            else:
                                self.tb1 = self.ao.scatter(self.fk, self.vfe-self.epos, marker='.',
                                            s=0.3, c='black')
                        if self.ef == 1:
                            self.tb1_.remove()
                            eposmin = self.epos-self.efwhm/2
                            eposmax = self.epos+self.efwhm/2
                            if self.emf=='KE':
                                self.tb1_ = self.ao.scatter(
                                    [self.fk, self.fk], [eposmin, eposmax], marker='_', c='grey', s=self.scale*self.scale*10, alpha=0.8)
                            else:
                                self.tb1_ = self.ao.scatter(
                                    [self.fk, self.fk], [self.vfe-eposmin, self.vfe-eposmax], marker='_', c='grey', s=self.scale*self.scale*10, alpha=0.8)
                    except:
                        pass
                    try:
                        if self.value2.get() == 'Data Plot with Pos and Bare Band' and self.k is not None:
                            self.tb2.remove()
                            if self.emf=='KE':
                                self.tb2, = self.ao.plot(self.k*np.float64(self.bbk_offset.get()), (self.be+np.float64(
                                    self.bb_offset.get()))/1000+self.vfe, linewidth=self.scale*0.3, c='red', linestyle='--')
                            else:
                                self.tb2, = self.ao.plot(self.k*np.float64(self.bbk_offset.get()), (self.be+np.float64(
                                    self.bb_offset.get()))/1000, linewidth=self.scale*0.3, c='red', linestyle='--')
                    except:
                        pass
                self.out.draw()
            self.mof = 1

class exp_motion:
    def __init__(self, scale: float, value: tk.StringVar, value1: tk.StringVar, value2: tk.StringVar, k_offset: tk.StringVar,
                 be: np.ndarray, k: np.ndarray, bb_offset: tk.StringVar, bbk_offset: tk.StringVar,
                 a: Axes, a0: Axes,f: Figure, f0: Figure, h1: QuadMesh, h2: QuadMesh,
                 acx: Axes, acy: Axes, annot: Annotation,
                 emf: Literal['KE', 'BE'], data: xr.DataArray, vfe: float, ev: np.ndarray, phi: np.ndarray,
                 pos: np.ndarray, fwhm: np.ndarray, rpos: np.ndarray, ophi: np.ndarray, fev: np.ndarray,
                 epos: np.ndarray, efwhm: np.ndarray, fk: np.ndarray, ffphi: np.ndarray, fphi: np.ndarray,
                 mp: int, ep: int, mf: int, ef: int, xl: tuple[float], yl: tuple[float],
                 posmin: np.ndarray, posmax: np.ndarray, eposmin: np.ndarray, eposmax: np.ndarray
                 ) -> None:
        self.cf = True
        self.scale = scale
        self.value, self.value1, self.value2, self.k_offset = value, value1, value2, k_offset
        self.be, self.k, self.bb_offset, self.bbk_offset = be, k, bb_offset, bbk_offset
        self.a, self.a0, self.f, self.f0, self.h1, self.h2 = a, a0, f, f0, h1, h2
        self.acx, self.acy, self.annot = acx, acy, annot
        self.emf, self.data, self.vfe, self.ev, self.phi = emf, data, vfe, ev, phi
        self.pos, self.fwhm, self.rpos, self.ophi, self.fev = pos, fwhm, rpos, ophi, fev
        self.epos, self.efwhm, self.fk, self.ffphi, self.fphi = epos, efwhm, fk, ffphi, fphi
        self.mp, self.ep, self.mf, self.ef, self.xl, self.yl = mp, ep, mf, ef, xl, yl
        self.posmin, self.posmax, self.eposmin, self.eposmax = posmin, posmax, eposmin, eposmax
        self.ta0, self.ta0_, self.ta1, self.ta1_, self.ta2 = None, None, None, None, None
    
    def size(self, s: int) -> int:
        return int(self.scale*s)
        
    def select_callback(self, eclick, erelease):
        """
        Callback for line selection.

        *eclick* and *erelease* are the press and release events.
        """
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if eclick.button == 1:
            self.a.set_xlim(sorted([x1, x2]))
            if self.emf=='KE':
                self.a.set_ylim(sorted([y1, y2]))
            else:
                self.a.set_ylim(sorted([y1, y2], reverse=True))
            self.f.show()
            if abs(x1-x2) < (self.xl[1]-self.xl[0])/3*2 or abs(y1-y2) < (self.yl[1]-self.yl[0])/3*2:
                try:
                    if self.mp == 1:
                        self.ta0.remove()
                    if self.mf == 1:
                        self.ta0_.remove()
                except:
                    pass
                try:
                    if self.ep == 1:
                        self.ta1.remove()
                    if self.ef == 1:
                        self.ta1_.remove()
                except:
                    pass
                try:
                    self.ta2.remove()
                except:
                    pass
                if self.value2.get() == 'Data Plot with Pos and Bare Band' or self.value2.get() == 'Data Plot with Pos':
                    try:
                        if self.mp == 1:
                            if self.emf=='KE':
                                self.ta0 = self.a.scatter(self.pos, self.fev, marker='.', s=self.scale*self.scale*30, c='black')
                            else:
                                self.ta0 = self.a.scatter(self.pos, self.vfe-self.fev, marker='.', s=self.scale*self.scale*30, c='black')
                        if self.mf == 1:
                            if self.emf=='KE':
                                self.ta0_ = self.a.scatter([self.posmin, self.posmax], [
                                            self.fev, self.fev], marker='|', c='grey', s=self.scale*self.scale*50, alpha=0.8)
                            else:
                                self.ta0_ = self.a.scatter([self.posmin, self.posmax], [self.vfe-self.fev, self.vfe-self.fev], marker='|', c='grey', s=self.scale*self.scale*50, alpha=0.8)
                    except:
                        pass
                    try:
                        if self.ep == 1:
                            if self.emf=='KE':
                                self.ta1 = self.a.scatter(self.fk, self.epos, marker='.', s=self.scale*self.scale*30, c='black')
                            else:
                                self.ta1 = self.a.scatter(self.fk, self.vfe-self.epos, marker='.', s=self.scale*self.scale*30, c='black')
                                
                        if self.ef == 1:
                            if self.emf=='KE':
                                self.ta1_ = self.a.scatter(
                                    [self.fk, self.fk], [self.eposmin, self.eposmax], marker='_', c='grey', s=self.scale*self.scale*50, alpha=0.8)
                            else:
                                self.ta1_ = self.a.scatter(
                                    [self.fk, self.fk], [self.vfe-self.eposmin, self.vfe-self.eposmax], marker='_', c='grey', s=self.scale*self.scale*50, alpha=0.8)
                    except:
                        pass

                    if self.value2.get() == 'Data Plot with Pos and Bare Band' and self.k is not None:
                        if self.emf=='KE':
                            self.ta2, = self.a.plot(self.k*np.float64(self.bbk_offset.get()), (self.be -
                                        np.float64(self.bb_offset.get()))/1000+self.vfe, linewidth=self.scale*2, c='red', linestyle='--')
                        else:
                            self.ta2, = self.a.plot(self.k*np.float64(self.bbk_offset.get()), (-self.be +
                                        np.float64(self.bb_offset.get()))/1000, linewidth=self.scale*2, c='red', linestyle='--')
                self.f.show()
            else:
                try:
                    if self.mp == 1:
                        self.ta0.remove()
                        if self.emf=='KE':
                            self.ta0 = self.a.scatter(self.pos, self.fev, marker='.', s=self.scale*self.scale*0.3, c='black')
                        else:
                            self.ta0 = self.a.scatter(self.pos, self.vfe-self.fev, marker='.', s=self.scale*self.scale*0.3, c='black')
                            
                    if self.mf == 1:
                        self.ta0_.remove()
                        if self.emf=='KE':
                            self.ta0_ = self.a.scatter([self.posmin, self.posmax], [self.fev, self.fev],
                                            marker='|', c='grey', s=self.scale*self.scale*10, alpha=0.8)
                        else:
                            self.ta0_ = self.a.scatter([self.posmin, self.posmax], [self.vfe-self.fev, self.vfe-self.fev],
                                            marker='|', c='grey', s=self.scale*self.scale*10, alpha=0.8)
                except:
                    pass
                try:
                    if self.ep == 1:
                        self.ta1.remove()
                        if self.emf=='KE':
                            self.ta1 = self.a.scatter(self.fk, self.epos, marker='.', s=self.scale*self.scale*0.3, c='black')
                        else:
                            self.ta1 = self.a.scatter(self.fk, self.vfe-self.epos, marker='.', s=self.scale*self.scale*0.3, c='black')
                            
                    if self.ef == 1:
                        self.ta1_.remove()
                        if self.emf=='KE':
                            self.ta1_ = self.a.scatter([self.fk, self.fk], [self.eposmin, self.eposmax],
                                        marker='_', c='grey', s=self.scale*self.scale*10, alpha=0.8)
                        else:
                            self.ta1_ = self.a.scatter([self.fk, self.fk], [self.vfe-self.eposmin, self.vfe-self.eposmax],
                                        marker='_', c='grey', s=self.scale*self.scale*10, alpha=0.8)
                except:
                    pass
                try:
                    if self.value2.get() == 'Data Plot with Pos and Bare Band' and self.k is not None:
                        self.ta2.remove()
                        if self.emf =='KE':
                            self.ta2, = self.a.plot(self.k*np.float64(self.bbk_offset.get()), (self.be -
                                    np.float64(self.bb_offset.get()))/1000+self.vfe, linewidth=self.scale*0.3, c='red', linestyle='--')
                        else:
                            self.ta2, = self.a.plot(self.k*np.float64(self.bbk_offset.get()), (-self.be +
                                    np.float64(self.bb_offset.get()))/1000, linewidth=self.scale*0.3, c='red', linestyle='--')
                except:
                    pass
                self.f.show()
        else:
            self.a.set_xlim(self.xl)
            self.a.set_ylim(self.yl)
            try:
                if self.mp == 1:
                    self.ta0.remove()
                    if self.emf=='KE':
                        self.ta0 = self.a.scatter(self.pos, self.fev, marker='.', s=self.scale*self.scale*0.3, c='black')
                    else:
                        self.ta0 = self.a.scatter(self.pos, self.vfe-self.fev, marker='.', s=self.scale*self.scale*0.3, c='black')
                        
                if self.mf == 1:
                    self.ta0_.remove()
                    if self.emf=='KE':
                        self.ta0_ = self.a.scatter([self.posmin, self.posmax], [self.fev, self.fev],
                                    marker='|', c='grey', s=self.scale*self.scale*10, alpha=0.8)
                    else:
                        self.ta0_ = self.a.scatter([self.posmin, self.posmax], [self.vfe-self.fev, self.vfe-self.fev],
                                    marker='|', c='grey', s=self.scale*self.scale*10, alpha=0.8)
            except:
                pass
            try:
                if self.ep == 1:
                    self.ta1.remove()
                    if self.emf=='KE':
                        self.ta1 = self.a.scatter(self.fk, self.epos, marker='.', s=self.scale*self.scale*0.3, c='black')
                    else:
                        self.ta1 = self.a.scatter(self.fk, self.vfe-self.epos, marker='.', s=self.scale*self.scale*0.3, c='black')
                        
                if self.ef == 1:
                    self.ta1_.remove()
                    if self.emf=='KE':
                        self.ta1_ = self.a.scatter([self.fk, self.fk], [self.eposmin, self.eposmax],
                                    marker='_', c='grey', s=self.scale*self.scale*10, alpha=0.8)
                    else:
                        self.ta1_ = self.a.scatter([self.fk, self.fk], [self.vfe-self.eposmin, self.vfe-self.eposmax],
                                    marker='_', c='grey', s=self.scale*self.scale*10, alpha=0.8)
            except:
                pass
            try:
                if self.value2.get() == 'Data Plot with Pos and Bare Band' and self.k is not None:
                    self.ta2.remove()
                    if self.emf=='KE':
                        self.ta2, = self.a.plot(self.k*np.float64(self.bbk_offset.get()), (self.be -
                                    np.float64(self.bb_offset.get()))/1000+self.vfe, linewidth=self.scale*0.3, c='red', linestyle='--')
                    else:
                        self.ta2, = self.a.plot(self.k*np.float64(self.bbk_offset.get()), (-self.be +
                                np.float64(self.bb_offset.get()))/1000, linewidth=self.scale*0.3, c='red', linestyle='--')
            except:
                pass
            self.f.show()

    def cur_move(self, event):
        if event.inaxes == self.a and event.xdata is not None and event.ydata is not None:
            self.f.canvas.get_tk_widget().config(cursor="crosshair")
            try:
                self.xx.remove()
                self.yy.remove()
            except:
                pass
            self.xx=self.a.axvline(event.xdata, color='red')
            self.yy=self.a.axhline(event.ydata, color='red')
        else:
            self.f.canvas.get_tk_widget().config(cursor="")
            try:
                self.xx.remove()
                self.yy.remove()
            except:
                pass
        self.f.show()
        

    def cur_on_move(self, event):
        if event.inaxes == self.a and event.xdata is not None and event.ydata is not None:
            self.annot.xy = (event.xdata, event.ydata)
            text = f"x={event.xdata:.3f}\ny={event.ydata:.3f}"
            self.annot.set_text(text)
            # 取得座標軸範圍
            xlim = self.a.get_xlim()
            ylim = self.a.get_ylim()
            # 設定 annotation 方向
            offset_x, offset_y = 20, 20
            # 靠近右邊界
            if event.xdata > xlim[1] - (xlim[1]-xlim[0])*0.15:
                offset_x = -60
            # 靠近左邊界
            elif event.xdata < xlim[0] + (xlim[1]-xlim[0])*0.15:
                offset_x = 20
            # 靠近上邊界
            if event.ydata > ylim[1] - (ylim[1]-ylim[0])*0.15:
                offset_y = -40
            # 靠近下邊界
            elif event.ydata < ylim[0] + (ylim[1]-ylim[0])*0.15:
                offset_y = 20
            self.annot.set_position((offset_x, offset_y))
            self.annot.set_visible(True)
            self.f.canvas.draw_idle()
        else:
            self.annot.set_visible(False)
            self.f.canvas.draw_idle()

    def onselect(self, xmin, xmax):
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        self.h2.set_clim(xmin, xmax)
        self.f0.show()
        self.h1.set_clim(xmin, xmax)
        self.f.show()


    def onmove_callback(self, xmin, xmax):
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        self.h2.set_clim(xmin, xmax)
        self.f0.show()
        self.h1.set_clim(xmin, xmax)
        self.f.show()

    def cut_move(self, event):
        self.f.canvas.get_tk_widget().config(cursor="")
        if event.inaxes:
            cxdata = event.xdata
            cydata = event.ydata
            xf = (cxdata >= self.a.get_xlim()[0] and cxdata <= self.a.get_xlim()[1])
            if self.emf=='KE':
                yf = (cydata >= self.a.get_ylim()[0] and cydata <= self.a.get_ylim()[1])
            else:
                yf = (cydata <= self.a.get_ylim()[0] and cydata >= self.a.get_ylim()[1])
            if xf and yf:
                self.f.canvas.get_tk_widget().config(cursor="crosshair")
                try:
                    self.xx.remove()
                    self.yy.remove()
                except:
                    pass
                self.xx=self.a.axvline(cxdata,color='r')
                self.yy=self.a.axhline(cydata,color='r')
                if self.cf:
                    if self.emf=='KE':
                        dx = self.data.sel(
                            eV=cydata, method='nearest').to_numpy().reshape(len(self.phi))
                    else:
                        dx = self.data.sel(eV=self.vfe-cydata, method='nearest').to_numpy().reshape(len(self.phi))
                    dy = self.data.sel(
                        phi=cxdata, method='nearest').to_numpy().reshape(len(self.ev))
                    self.acx.clear()
                    self.acy.clear()
                    self.acx.set_title('                Raw Data', font='Arial', fontsize=self.size(18))
                    self.acx.plot(self.phi, dx, c='black')
                    if self.emf=='KE':
                        self.acy.plot(dy, self.ev, c='black')
                    else:
                        self.acy.plot(dy, self.vfe-self.ev, c='black')
                    self.acx.set_xticks([])
                    self.acy.set_yticks([])
                    self.acx.set_xlim(self.a.get_xlim())
                    self.acy.set_ylim(self.a.get_ylim())
                    # self.f.canvas.draw_idle()
        else:
            try:
                if self.cf:
                    self.acx.clear()
                    self.acy.clear()
                    self.acx.set_title('                Raw Data', font='Arial', fontsize=self.size(18))
                    self.acx.set_xticks([])
                    self.acx.set_yticks([])
                    self.acy.set_xticks([])
                    self.acy.set_yticks([])
                self.xx.remove()
                self.yy.remove()
            except:
                pass
        self.f.show()

    def cut_select(self, event):
        if event.button == 1 and self.cf:
            self.cf = False
            self.x = self.a.axvline(event.xdata, color='red')
            self.y = self.a.axhline(event.ydata, color='red')
        elif event.button == 1 and not self.cf:
            if self.x:
                self.x.remove()
                self.y.remove()
            self.x = self.a.axvline(event.xdata, color='red')
            self.y = self.a.axhline(event.ydata, color='red')
            if self.emf=='KE':
                dx = self.data.sel(eV=event.ydata,
                            method='nearest').to_numpy().reshape(len(self.phi))
            else:
                dx = self.data.sel(eV=self.vfe-event.ydata,
                            method='nearest').to_numpy().reshape(len(self.phi))
            dy = self.data.sel(phi=event.xdata,
                        method='nearest').to_numpy().reshape(len(self.ev))
            self.acx.clear()
            self.acy.clear()
            self.acx.set_title('                Raw Data', font='Arial', fontsize=self.size(18))
            self.acx.plot(self.phi, dx, c='black')
            if self.emf=='KE':
                self.acy.plot(dy, self.ev, c='black')
            else:
                self.acy.plot(dy, self.vfe-self.ev, c='black')
            self.acx.set_xticks([])
            self.acy.set_yticks([])
            self.acx.set_xlim(self.a.get_xlim())
            self.acy.set_ylim(self.a.get_ylim())

        elif event.button == 3:
            self.cf = True
            if self.x:
                self.x.remove()
                self.y.remove()
        # self.f.canvas.draw_idle()
        copy_to_clipboard(self.f)
        self.f.show()

class plots_util(ABC):
    def __init__(self, scale: float, value: tk.StringVar, value1: tk.StringVar, value2: tk.StringVar, value3: tk.StringVar,
                 be: np.ndarray, k: np.ndarray, k_offset: tk.StringVar, bb_offset: tk.StringVar, bbk_offset: tk.StringVar, b_sw: tk.Button,
                 limg: tk.Label, img: list[tk.PhotoImage], st: queue.Queue, im_kernel: int,
                 optionList: list[str], optionList1: list[str], optionList2: list[str],
                 emf: Literal['KE', 'BE'], data: xr.DataArray, vfe: float, ev: np.ndarray, phi: np.ndarray,
                 pos: np.ndarray, fwhm: np.ndarray, rpos: np.ndarray, ophi: np.ndarray, fev: np.ndarray,
                 epos: np.ndarray, efwhm: np.ndarray, fk: np.ndarray, ffphi: np.ndarray, fphi: np.ndarray,
                 mp: int, ep: int, mf: int, ef: int, npzf: bool, fig: Figure, out: FigureCanvasTkAgg,
                 d: int, l: int, p: int, dl: int,
                 rx: np.ndarray, ry: np.ndarray, ix: np.ndarray, iy: np.ndarray
                 ):
        self.scale, self.value, self.value1, self.value2, self.value3 = scale, value, value1, value2, value3
        self.be, self.k, self.k_offset, self.bb_offset, self.bbk_offset, self.b_sw = be, k, k_offset, bb_offset, bbk_offset, b_sw
        self.limg, self.img, self.st, self.im_kernel = limg, img, st, im_kernel
        self.optionList, self.optionList1, self.optionList2 = optionList, optionList1, optionList2
        self.emf, self.data, self.vfe, self.ev, self.phi = emf, data, vfe, ev, phi
        self.pos, self.fwhm, self.rpos, self.ophi, self.fev = pos, fwhm, rpos, ophi, fev
        self.epos, self.efwhm, self.fk, self.ffphi, self.fphi = epos, efwhm, fk, ffphi, fphi
        self.mp, self.ep, self.mf, self.ef, self.npzf = mp, ep, mf, ef, npzf
        self.fig, self.out = fig, out
        self.d, self.l, self.p, self.dl = d, l, p, dl
        self.rx, self.ry, self.ix, self.iy = rx, ry, ix, iy
        self.warn_str = ''

    @abstractmethod
    def pars_warn(self):
        pass

    @abstractmethod
    def pars1(self):
        pass
    
    @abstractmethod
    def pars2(self):
        pass
    
    @abstractmethod
    def pars3(self):
        pass

    @abstractmethod
    def show_version(self):
        pass
    
    @abstractmethod
    def main_plot_bind(self):
        pass
    
    @abstractmethod
    def climon(self):
        pass
    
    @abstractmethod
    def climoff(self):
        pass
    
    def size(self, s: int) -> int:
        return int(self.scale*s)
    
    def rplot(self, f, canvas):
        """
        Plot the raw data on a given canvas.

        Parameters
        -----
            f (Figure object): The figure object on which the plot will be created.
            canvas (Canvas object): The canvas object on which the plot will be drawn.

        Returns
        -----
            None
        """
        data, ev, phi, value3, ev, phi, vfe = self.data, self.ev, self.phi, self.value3, self.ev, self.phi, self.vfe
        emf, npzf = self.emf, self.npzf
        
        ao = f.add_axes([0.13, 0.1, 0.6, 0.65])
        rcx = f.add_axes([0.13, 0.78, 0.6, 0.15])
        rcy = f.add_axes([0.75, 0.1, 0.12, 0.65])
        acb = f.add_axes([0.9, 0.1, 0.02, 0.65])
        rcx.set_xticks([])
        rcx.set_yticks([])
        rcy.set_xticks([])
        rcy.set_yticks([])
        if emf=='KE':
            tx, ty = np.meshgrid(phi, ev)
        else:
            tx, ty = np.meshgrid(phi, vfe-ev)
        tz = data.to_numpy()
        # h1 = a.scatter(tx,ty,c=tz,marker='o',s=self.scale*self.scale*0.9,cmap=value3.get());
        h0 = ao.pcolormesh(tx, ty, tz, cmap=value3.get())
        f.colorbar(h0, cax=acb, orientation='vertical')
        # a.set_title('Raw Data',font='Arial',fontsize=self.size(16))
        rcx.set_title('            Raw Data', font='Arial', fontsize=self.size(16))
        if npzf:ao.set_xlabel(r'k ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=self.size(12))
        else:ao.set_xlabel('Angle (deg)', font='Arial', fontsize=self.size(12))
        if emf=='KE':
            ao.set_ylabel('Kinetic Energy (eV)', font='Arial', fontsize=self.size(12))
        else:
            ao.set_ylabel('Binding Energy (eV)', font='Arial', fontsize=self.size(12))
            ao.invert_yaxis()
        xl = ao.get_xlim()
        yl = ao.get_ylim()
        canvas.draw()
        
        self.h0, self.ao, self.xl, self.yl, self.rcx, self.rcy, self.acb, self.pflag = h0, ao, xl, yl, rcx, rcy, acb, 1
        self.pars1()
    
    def o_plot1(self):
        self.rcx, self.rcy, self.acb = None, None, None
        h0, pflag = '', 1
        ao, xl, yl = None, None, None
        value, value1, value2, value3 = self.value, self.value1, self.value2, self.value3
        data, ev, phi, vfe, fig, out = self.data, self.ev, self.phi, self.vfe, self.fig, self.out
        k_offset = self.k_offset
        limg, img, optionList, st, b_sw = self.limg, self.img, self.optionList, self.st, self.b_sw
        emf, npzf, im_kernel = self.emf, self.npzf, self.im_kernel
        d, l, p = self.d, self.l, self.p
        m, h = 9.1093837015e-31, 6.62607015e-34
        
        if value.get() in optionList:
            try:
                b_sw.grid_remove()
            except:
                pass
            limg.config(image=img[np.random.randint(len(img))])
            print('Plotting...')
            st.put('Plotting...')
            pflag = 1
            value1.set('---Plot2---')
            value2.set('---Plot3---')
            fig.clear()
            if data is None:
                messagebox.showwarning("Warning","Please load Raw Data")
                self.warn_str = "Please load Raw Data"
                self.pars_warn()
                print('Please load Raw Data')
                st.put('Please load Raw Data')
                self.show_version()
                return
            if value.get() == 'Raw Data':
                self.rplot(fig, out)
            else:
                if value.get() == 'First Derivative':   #axis: phi
                    ao = fig.subplots()
                    pz = np.diff(smooth(data.to_numpy()))/np.diff(phi)
                    if emf=='KE':
                        px, py = np.meshgrid(phi[0:-1]+np.diff(phi)/2, ev)
                        tev = py.copy()
                    else:
                        px, py = np.meshgrid(phi[0:-1]+np.diff(phi)/2, vfe-ev)
                        tev = vfe-py.copy()
                    if npzf:
                        px = phi[0:-1]+np.diff(phi)/2
                    else:
                        px = (2*m*tev*1.602176634*10**-19)**0.5*np.sin((np.float64(k_offset.get())+px+np.diff(phi)/2)/180*np.pi)*10**-10/(h/2/np.pi)
                    h0 = ao.pcolormesh(px, py, pz, cmap=value3.get())
                    cb = fig.colorbar(h0)
                    cb.set_ticklabels(cb.get_ticks(), font='Arial')
                    
                # if value.get() == 'First Derivative':    #axis: eV
                #     ao = fig.subplots()
                #     pz = np.diff(smooth(data.to_numpy().transpose()))/np.diff(ev)
                #     pz = pz.transpose()
                #     if emf=='KE':
                #         px, py = np.meshgrid(phi, ev[0:-1]+np.diff(ev)/2)
                #         tev = py.copy()
                #     else:
                #         px, py = np.meshgrid(phi, vfe-ev[0:-1]-np.diff(ev)/2)
                #         tev = vfe-py.copy()
                #     px = (2*m*tev*1.602176634*10**-19)**0.5*np.sin((np.float64(k_offset.get())+px)/180*np.pi)*10**-10/(h/2/np.pi)
                #     h0 = ao.pcolormesh(px, py, pz, cmap=value3.get())
                #     cb = fig.colorbar(h0)
                #     cb.set_ticklabels(cb.get_ticks(), font='Arial')
                
                elif value.get() == 'Second Derivative':    #axis: phi, eV
                    ao = fig.subplots()                
                    pz = laplacian_filter(data.to_numpy(), im_kernel)
                    if emf=='KE':
                        px, py = np.meshgrid(phi, ev)
                        tev = py.copy()
                    else:
                        px, py = np.meshgrid(phi, vfe-ev)
                        tev = vfe-py.copy()
                    if npzf:
                        px = phi
                    else:
                        px = (2*m*tev*1.602176634*10**-19)**0.5*np.sin((np.float64(k_offset.get())+px)/180*np.pi)*10**-10/(h/2/np.pi)
                    
                    h0 = ao.pcolormesh(px, py, pz, cmap=value3.get())
                    cb = fig.colorbar(h0)
                    cb.set_ticklabels(cb.get_ticks(), font='Arial')
                else:
                    if 'MDC Curves' not in value.get():
                        fig.clear()
                        ao = fig.subplots()
                    elif value.get() == 'MDC Curves':
                        fig.clear()
                        ao = fig.add_axes([0.2, 0.13, 0.5, 0.8])
                    else:
                        fig.clear()
                        at = fig.add_axes([0.25, 0.13, 0.5, 0.8])
                        at.set_xticks([])
                        at.set_yticks([])
                        ao = fig.add_axes([0.1, 0.13, 0.4, 0.8])
                        ao1 = fig.add_axes([0.5, 0.13, 0.4, 0.8])
                    if value.get() == 'E-k Diagram':
                        # h1=a.scatter(mx,my,c=mz,marker='o',s=self.scale*self.scale*0.9,cmap=value3.get());
                        if emf=='KE':
                            px, py = np.meshgrid(phi, ev)
                            tev = py.copy()
                        else:
                            px, py = np.meshgrid(phi, vfe-ev)
                            tev = vfe-py.copy()
                        if npzf:
                            px = phi
                        else:
                            px = (2*m*tev*1.602176634*10**-19)**0.5*np.sin((np.float64(k_offset.get())+px)/180*np.pi)*10**-10/(h/2/np.pi)
                        pz = data.to_numpy()
                        h0 = ao.pcolormesh(px, py, pz, cmap=value3.get())
                        cb = fig.colorbar(h0)
                        cb.set_ticklabels(cb.get_ticks(), font='Arial')
                        
                    elif value.get() == 'MDC Normalized':
                        if emf=='KE':
                            px, py = np.meshgrid(phi, ev)
                            tev = py.copy()
                        else:
                            px, py = np.meshgrid(phi, vfe-ev)
                            tev = vfe-py.copy()
                        if npzf:
                            px = phi
                        else:
                            px = (2*m*tev*1.602176634*10**-19)**0.5*np.sin((np.float64(k_offset.get())+px)/180*np.pi)*10**-10/(h/2/np.pi)
                        pz = data.to_numpy().copy().astype(float)
                        pz /= np.max(pz, axis=1)[:, np.newaxis]
                        pz = np.nan_to_num(pz)
                        ao.pcolormesh(px, py, pz, cmap=value3.get())
                    elif value.get() == 'MDC Curves':
                        pbar = tqdm.tqdm(
                            total=len(ev)//d, desc='MDC', colour='red')
                        y = np.zeros([len(ev),len(phi)],dtype=float)
                        for n in range(len(ev)):
                            ecut = data.sel(eV=ev[n], method='nearest')
                            if npzf:
                                x = phi
                            else:
                                x = (2*m*ev[n]*1.602176634*10**-19)**0.5*np.sin(
                                (np.float64(k_offset.get())+phi)/180*np.pi)*10**-10/(h/2/np.pi)
                            y[n][:] = ecut.to_numpy().reshape(len(ecut))
                        for n in range(len(ev)//d):
                            yy=y[n*d][:]+n*np.max(y)/d
                            yy=smooth(yy,l,p)
                            ao.plot(x, yy, c='black')
                            pbar.update(1)
                            # print(str(round((n+1)/(len(ev))*100))+'%'+' ('+str(len(ev))+')')
                            st.put(str(round((n+1)/(len(ev)//d)*100)) +
                                '%'+' ('+str(len(ev)//d)+')')
                        pbar.close()
                    elif value.get() == 'E-k with MDC Curves':
                        pbar = tqdm.tqdm(
                            total=len(ev)//d, desc='MDC', colour='red')
                        y = np.zeros([len(ev),len(phi)],dtype=float)
                        for n in range(len(ev)):
                            ecut = data.sel(eV=ev[n], method='nearest')
                            if npzf:
                                x = phi
                            else:
                                x = (2*m*ev[n]*1.602176634*10**-19)**0.5*np.sin(
                                (np.float64(k_offset.get())+phi)/180*np.pi)*10**-10/(h/2/np.pi)
                            y[n][:] = ecut.to_numpy().reshape(len(ecut))
                        for n in range(len(ev)//d):
                            yy=y[n*d][:]+n*np.max(y)/d
                            yy=smooth(yy,l,p)
                            ao1.plot(x, yy, c='black')
                            pbar.update(1)
                            # print(str(round((n+1)/(len(ev))*100))+'%'+' ('+str(len(ev))+')')
                            st.put(str(round((n+1)/(len(ev)//d)*100)) +
                                '%'+' ('+str(len(ev)//d)+')')
                        pbar.close()
                        if emf=='KE':
                            px, py = np.meshgrid(phi, ev)
                            tev = py.copy()
                        else:
                            px, py = np.meshgrid(phi, vfe-ev)
                            tev = vfe-py.copy()
                        if npzf:
                            px = phi
                        else:
                            px = (2*m*tev*1.602176634*10**-19)**0.5*np.sin((np.float64(k_offset.get())+px)/180*np.pi)*10**-10/(h/2/np.pi)
                        pz = data.to_numpy()
                        h0 = ao.pcolormesh(px, py, pz, cmap=value3.get())
                        ylb=ao1.twinx()
                        ylb.set_ylabel('Intensity (a.u.)', font='Arial', fontsize=self.size(14))
                        ylb.set_yticklabels([])
                        # cb = fig.colorbar(h0, ax=ao1)
                        # cb.set_ticklabels(cb.get_ticks(), font='Arial')
                if 'E-k with' not in value.get():
                    ao.set_title(value.get(), font='Arial', fontsize=self.size(16))
                else:
                    at.set_title(value.get(), font='Arial', fontsize=self.size(18))
                ao.set_xlabel(r'k ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=self.size(14))
                if 'MDC Curves' not in value.get():
                    if emf=='KE':
                        ao.set_ylabel('Kinetic Energy (eV)', font='Arial', fontsize=self.size(14))
                    else:
                        ao.set_ylabel('Binding Energy (eV)', font='Arial', fontsize=self.size(14))
                        ao.invert_yaxis()
                else:
                    if 'E-k with' in value.get():
                        if emf=='KE':
                            ao.set_ylabel('Kinetic Energy (eV)', font='Arial', fontsize=self.size(14))
                            ao.set_ylim([ev[0], ev[n*d]])
                        else:
                            ao.set_ylabel('Binding Energy (eV)', font='Arial', fontsize=self.size(14))
                            ao.invert_yaxis()
                            ao.set_ylim([vfe-ev[0], vfe-ev[n*d]])
                        ao1.set_xlabel(r'k ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=self.size(14))
                        ao1.set_yticklabels([])
                        ao1.set_xlim([min(x), max(x)])
                        ao1.set_ylim([0, np.max(n*np.max(y)/d)])
                    else:
                        ylr=ao.twinx()
                        ao.set_yticklabels([])
                        ao.set_ylabel('Intensity (a.u.)', font='Arial', fontsize=self.size(14))
                        ylr.set_ylabel(r'$\longleftarrow$ Binding Energy', font='Arial', fontsize=self.size(14))
                        ylr.set_yticklabels([])
                        ao.set_xlim([min(x), max(x)])
                        ao.set_ylim([0, np.max(n*np.max(y)/d)])
                    
                xl = ao.get_xlim()
                yl = ao.get_ylim()
                self.pflag, self.h0, self.ao, self.xl, self.yl = pflag, h0, ao, xl, yl
                self.pars1()
            try:
                if value.get() != 'MDC Normalized' and value.get() != 'MDC Curves':
                    self.climon()
                    out.draw()
                else:
                    self.climoff()
                    out.draw()
            except:
                pass
            print('Done')
            st.put('Done')
            self.main_plot_bind()
            gc.collect()

    def o_plot2(self):
        rx, ry, ix, iy, pflag = None, None, None, None, 2
        value, value1, value2 = self.value, self.value1, self.value2
        vfe, fig, out = self.vfe, self.fig, self.out
        limg, img, st, b_sw = self.limg, self.img, self.st, self.b_sw
        pos, fwhm, fev = self.pos, self.fwhm, self.fev
        epos, efwhm, fk = self.epos, self.efwhm, self.fk
        k, be = self.k, self.be
        bb_offset, bbk_offset = self.bb_offset, self.bbk_offset
        optionList1 = self.optionList1
        if value1.get() in optionList1:
            try:
                b_sw.grid_remove()
            except:
                pass
            limg.config(image=img[np.random.randint(len(img))])
            print('Plotting...')
            st.put('Plotting...')
            pflag = 2
            value.set('---Plot1---')
            value2.set('---Plot3---')
            fig.clear()
            self.climoff()
            if value1.get() == 'MDC fitted Data':
                try:
                    x = (vfe-fev)*1000
                    # y = (fwhm*6.626*10**-34/2/3.1415926/(10**-10))**2/2/(9.11*10**-31)/(1.602176634*10**-19)*1000
                except:
                    messagebox.showwarning("Warning", "Please load MDC fitted file")
                    self.warn_str = "Please load MDC fitted file"
                    self.pars_warn()
                    print(r'Please Load MDC fitted file')
                    st.put(r'Please Load MDC fitted file')
                    self.show_version()
                    return
                a = fig.subplots(2, 1)
                a[0].set_title('MDC Fitting Result', font='Arial', fontsize=self.size(18))
                a[0].set_xlabel('Binding Energy (meV)',
                                font='Arial', fontsize=self.size(14))
                a[0].set_ylabel(
                    r'Position ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=self.size(14))
                a[0].tick_params(direction='in')
                a[0].scatter(x, pos, c='black', s=self.scale*self.scale*5)

                a[1].set_xlabel('Binding Energy (meV)',
                                font='Arial', fontsize=self.size(14))
                a[1].set_ylabel(
                    r'FWHM ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=self.size(14))
                a[1].tick_params(direction='in')
                a[1].scatter(x, fwhm, c='black', s=self.scale*self.scale*5)
                
                a[0].invert_xaxis()
                a[1].invert_xaxis()
            elif value1.get() == 'EDC fitted Data':
                try:
                    x = fk + 0
                except:
                    messagebox.showwarning("Warning", "Please load EDC fitted file")
                    self.warn_str = "Please load EDC fitted file"
                    self.pars_warn()
                    print(r'Please Load EDC fitted file')
                    st.put(r'Please Load EDC fitted file')
                    self.show_version()
                    return
                a = fig.subplots(2, 1)
                a[0].set_title('EDC Fitting Result', font='Arial', fontsize=self.size(18))
                a[0].set_xlabel(
                    r'Position ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=self.size(14))
                a[0].set_ylabel('Binding Energy (meV)',
                                font='Arial', fontsize=self.size(14))
                a[0].tick_params(direction='in')
                a[0].scatter(x, (vfe-epos)*1000, c='black', s=self.scale*self.scale*5)

                a[1].set_xlabel(
                    r'Position ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=self.size(14))
                a[1].set_ylabel('FWHM (meV)', font='Arial', fontsize=self.size(14))
                a[1].tick_params(direction='in')
                a[1].scatter(x, efwhm*1000, c='black', s=self.scale*self.scale*5)
                
                a[0].invert_yaxis()
            elif value1.get() == 'Real Part':
                try:
                    x = (vfe-fev)*1000
                    y = pos
                except:
                    messagebox.showwarning("Warning", "Please load MDC fitted file")
                    self.warn_str = "Please load MDC fitted file"
                    self.pars_warn()
                    print('Please load MDC fitted file')
                    st.put('Please load MDC fitted file')
                    self.show_version()
                    return
                try:
                    yy = interp(y, k*np.float64(bbk_offset.get()), be -
                                # interp x into be,k set
                                np.float64(bb_offset.get()))
                except:
                    messagebox.showwarning("Warning", "Please load Bare Band file")
                    self.warn_str = "Please load Bare Band file"
                    self.pars_warn()
                    print('Please load Bare Band file')
                    st.put('Please load Bare Band file')
                    self.show_version()
                    return
                a = fig.subplots(2, 1)
                a[0].set_title('Real Part', font='Arial', fontsize=self.size(18))
                a[0].plot(x, -(x+yy), c='black', linestyle='-', marker='.')

                rx = x
                ry = -(x+yy)
                a[0].tick_params(direction='in')
                a[0].set_xlabel('Binding Energy (meV)', font='Arial', fontsize=self.size(14))
                a[0].set_ylabel(r'Re $\Sigma$ (meV)', font='Arial', fontsize=self.size(14))

                h1 = a[1].scatter(y, x, c='black', s=self.scale*self.scale*5)
                h2 = a[1].scatter(k*np.float64(bbk_offset.get()),
                                -be+np.float64(bb_offset.get()), c='red', s=self.scale*self.scale*5)

                a[1].legend([h1, h2], ['fitted data', 'bare band'])
                a[1].tick_params(direction='in')
                a[1].set_ylabel('Binding Energy (meV)', font='Arial', fontsize=self.size(14))
                a[1].set_xlabel(
                    r'Pos ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=self.size(14))
                
                a[0].invert_xaxis()
                a[1].invert_yaxis()

                # a[0].set_xlim([-1000,50])
                # a[0].set_ylim([-100,500])
                # a[1].set_ylim([-600,200])
                # a[1].set_xlim([-0.05,0.05])
            elif value1.get() == 'Imaginary Part':
                try:
                    tbe = (vfe-fev)*1000
                except:
                    messagebox.showwarning("Warning", "Please load MDC fitted file")
                    self.warn_str = "Please load MDC fitted file"
                    self.pars_warn()
                    print(r'Please Load MDC fitted file')
                    st.put(r'Please Load MDC fitted file')
                    self.show_version()
                    return
                try:
                    x = interp(tbe, -be+np.float64(bb_offset.get()),
                            k*np.float64(bbk_offset.get()))
                    y = interp(x, k*np.float64(bbk_offset.get()),
                            -be+np.float64(bb_offset.get()))
                except:
                    messagebox.showwarning("Warning", "Please load Bare Band file")
                    self.warn_str = "Please load Bare Band file"
                    self.pars_warn()
                    print('Please load Bare Band file')
                    st.put('Please load Bare Band file')
                    self.show_version()
                    return
                xx = np.diff(x)
                yy = np.diff(y)

                # eliminate vf in gap
                for i in range(len(yy)):
                    if yy[i]/xx[i] > 20000:
                        yy[i] = 0
                v = yy/xx
                # v = np.append(v, v[-1])  # fermi velocity
                v=interp(pos,x[0:-1]+xx/2,v)
                yy = np.abs(v*fwhm/2)
                xx = tbe
                ax = fig.subplots(2, 1)
                a = ax[0]
                b = ax[1]
                a.set_title('Imaginary Part', font='Arial', fontsize=self.size(18))
                a.plot(xx, yy, c='black', linestyle='-', marker='.')

                ix = xx
                iy = yy
                a.tick_params(direction='in')
                a.set_xlabel('Binding Energy (eV)', font='Arial', fontsize=self.size(14))
                a.set_ylabel(r'Im $\Sigma$ (meV)', font='Arial', fontsize=self.size(14))

                x = (vfe-fev)*1000
                y = fwhm
                b.plot(x, y, c='black', linestyle='-', marker='.')
                b.tick_params(direction='in')
                b.set_xlabel('Binding Energy (eV)', font='Arial', fontsize=self.size(14))
                b.set_ylabel(r'FWHM ($\frac{2\pi}{\AA}$)',
                            font='Arial', fontsize=self.size(14))
                
                a.invert_xaxis()
                b.invert_xaxis()
            out.draw()
            print('Done')
            st.put('Done')
            self.rx, self.ry, self.ix, self.iy, self.pflag = rx, ry, ix, iy, pflag
            self.pars2()
            self.main_plot_bind()
            gc.collect()

    def o_plot3(self):
        ao, h0, xl, yl, pflag = None, '', None, None, 3
        tb0, tb0_, tb1, tb1_, tb2 = None, None, None, None, None
        value, value1, value2, value3 = self.value, self.value1, self.value2, self.value3
        vfe, fig, out = self.vfe, self.fig, self.out
        limg, img, st, b_sw = self.limg, self.img, self.st, self.b_sw
        data, ev, phi = self.data, self.ev, self.phi
        emf, npzf, dl = self.emf, self.npzf, self.dl
        mf, ef, mp, ep = self.mf, self.ef, self.mp, self.ep
        rx, ry, ix, iy = self.rx, self.ry, self.ix, self.iy
        pos, fwhm, fev = self.pos, self.fwhm, self.fev
        epos, efwhm, fk = self.epos, self.efwhm, self.fk
        rpos, ophi = self.rpos, self.ophi
        k, be = self.k, self.be
        k_offset, bb_offset, bbk_offset = self.k_offset, self.bb_offset, self.bbk_offset
        optionList2 = self.optionList2
        m, h = 9.1093837015e-31, 6.62607015e-34
        if value2.get() in optionList2:
            limg.config(image=img[np.random.randint(len(img))])
            print('Plotting...')
            st.put('Plotting...')
            pflag = 3
            value.set('---Plot1---')
            value1.set('---Plot2---')
            fig.clear()
            ophi = np.arcsin(rpos/(2*m*fev*1.602176634*10**-19)**0.5 /
                            10**-10*(h/2/np.pi))*180/np.pi
            pos = (2*m*fev*1.602176634*10**-19)**0.5 * \
                np.sin((np.float64(k_offset.get())+ophi)/180*np.pi)*10**-10/(h/2/np.pi)
            try:
                x = (vfe-fev)*1000
                y = pos
            except:
                print('Please load MDC fitted file')
                st.put('Please load MDC fitted file')
            if 'Data Plot with Pos' in value2.get():
                try:
                    b_sw.grid_remove()
                except:
                    pass
            else:
                try:
                    b_sw.grid(row=0, column=4)
                except:
                    pass    
            if value2.get() != 'Data Plot with Pos':
                try:
                    yy = interp(y, k*np.float64(bbk_offset.get()), be -
                                # interp x into be,k set
                                np.float64(bb_offset.get()))
                    rx = x
                    ry = -(x+yy)
                    tbe = (vfe-fev)*1000
                    x = interp(tbe, -be+np.float64(bb_offset.get()),
                            k*np.float64(bbk_offset.get()))
                    y = interp(x, k*np.float64(bbk_offset.get()),
                            -be+np.float64(bb_offset.get()))
                    xx = np.diff(x)
                    yy = np.diff(y)

                    # eliminate vf in gap
                    for i in range(len(yy)):
                        if yy[i]/xx[i] > 20000:
                            yy[i] = 0
                    v = yy/xx
                    # v = np.append(v, v[-1])  # fermi velocity
                    v=interp(pos,x[0:-1]+xx/2,v)
                    yy = np.abs(v*fwhm/2)
                    xx = tbe
                    ix = xx
                    iy = yy
                    self.rx, self.ry, self.ix, self.iy = rx, ry, ix, iy
                except:
                    messagebox.showwarning("Warning", "Please load Bare Band file")
                    self.warn_str = "Please load Bare Band file"
                    self.pars_warn()
                    print('Please load Bare Band file')
                    st.put('Please load Bare Band file')
                    self.show_version()
                    return
            if value2.get() == 'Real & Imaginary':
                a = fig.subplots(2, 1)
                a[0].set_title(r'Self Energy $\Sigma$', font='Arial', fontsize=self.size(18))
                if dl==0:
                    a[0].scatter(rx, ry, edgecolors='black', c='w')
                elif dl==1:
                    a[0].plot(rx, ry, c='black')
                elif dl==2:
                    a[0].plot(rx, ry, c='black', linestyle='-', marker='.')
                a[0].tick_params(direction='in')
                a[0].set_xlabel('Binding Energy (meV)', font='Arial', fontsize=self.size(14))
                a[0].set_ylabel(r'Re $\Sigma$ (meV)', font='Arial', fontsize=self.size(14))
                if dl==0:
                    a[1].scatter(ix, iy, edgecolors='black', c='w')
                elif dl==1:
                    a[1].plot(ix, iy, c='black')
                elif dl==2:
                    a[1].plot(ix, iy, c='black', linestyle='-', marker='.')
                a[1].tick_params(direction='in')
                a[1].set_xlabel('Binding Energy (meV)', font='Arial', fontsize=self.size(14))
                a[1].set_ylabel(r'Im $\Sigma$ (meV)', font='Arial', fontsize=self.size(14))
                a[0].invert_xaxis()
                a[1].invert_xaxis()
            elif 'KK Transform' in value2.get():
                ################################################################################## Hilbert Transform
                ##################################################################################
                tbe = (vfe-fev)*1000
                
                ix=(tbe-tbe[-1])*-1
                cix=np.append(ix+ix[0],ix)
                tix=cix[0:len(cix)-1]*-1
                # kx=ix
                kx = np.append(cix,tix[::-1])
                ky = np.linspace(0, 1, len(kx))
                ciy=np.append(iy*0+np.mean(iy),iy)
                tiy=ciy[0:len(ciy)-1]
                ciy = np.append(ciy,tiy[::-1])

                #for imaginary part
                ix=(tbe-tbe[-1])*-1
                cix=np.append(ix+ix[0],ix)
                tix=cix[0:len(cix)-1]*-1
                kx = np.append(cix,tix[::-1])
                ky = np.linspace(0, 1, len(kx))
                cry=np.append(ry*0,ry)
                tcry=cry[0:len(cry)-1]*-1
                cry = np.append(cry,tcry[::-1])

                # Hilbert transform
                analytic_signal_r = hilbert(cry)
                amplitude_envelope_r = np.abs(analytic_signal_r)
                instantaneous_phase_r = np.unwrap(np.angle(analytic_signal_r))
                instantaneous_frequency_r = np.diff(instantaneous_phase_r) / (2.0 * np.pi)

                analytic_signal_i = hilbert(ciy)
                amplitude_envelope_i = np.abs(analytic_signal_i)
                instantaneous_phase_i = np.unwrap(np.angle(analytic_signal_i))
                instantaneous_frequency_i = np.diff(instantaneous_phase_i) / (2.0 * np.pi)

                # Reconstructed real and imaginary parts
                reconstructed_real = np.imag(analytic_signal_i)
                reconstructed_imag = -np.imag(analytic_signal_r)
                ################################################################################## # Export data points as txt files
                ##################################################################################
                
                # np.savetxt('re_sigma.txt', np.column_stack((tbe, ry)), delimiter='\t', header='Binding Energy (meV)\tRe Sigma (meV)', comments='')
                # np.savetxt('kk_re_sigma.txt', np.column_stack((tbe, reconstructed_real[len(ix):2*len(ix)])), delimiter='\t', header='Binding Energy (meV)\tRe Sigma KK (meV)', comments='')
                # np.savetxt('im_sigma.txt', np.column_stack((tbe, iy)), delimiter='\t', header='Binding Energy (meV)\tIm Sigma (meV)', comments='')
                # np.savetxt('kk_im_sigma.txt', np.column_stack((tbe, reconstructed_imag[len(ix):2*len(ix)])), delimiter='\t', header='Binding Energy (meV)\tIm Sigma KK (meV)', comments='')
                
                ##################################################################################
                ################################################################################## # Export data points as txt files
                    # Plot
                if 'Real Part' not in value2.get() and 'Imaginary Part' not in value2.get():
                    ax = fig.subplots(2, 1)
                    a = ax[0]
                    b = ax[1]
                    # Plot imaginary data and its Hilbert transformation
                    a.set_title(r'Self Energy $\Sigma$', font='Arial', fontsize=self.size(18))
                    if dl==0:
                        a.scatter(tbe, ry, edgecolors='black', c='w', label=r'Re $\Sigma$')
                        a.scatter(tbe, reconstructed_real[len(ix):2*len(ix)]+(ry-np.mean(reconstructed_real[len(ix):2*len(ix)])), edgecolors='red', c='w', label=r'Re $\Sigma_{KK}$=KK(Im $\Sigma$)')
                    elif dl==1:
                        a.plot(tbe, ry, c='black', label=r'Re $\Sigma$')
                        a.plot(tbe, reconstructed_real[len(ix):2*len(ix)]+(ry-np.mean(reconstructed_real[len(ix):2*len(ix)])), c='red', label=r'Re $\Sigma_{KK}$=KK(Im $\Sigma$)')
                    elif dl==2:
                        a.plot(tbe, ry, c='black', linestyle='-', marker='.', label=r'Re $\Sigma$')
                        a.plot(tbe, reconstructed_real[len(ix):2*len(ix)]+(ry-np.mean(reconstructed_real[len(ix):2*len(ix)])), c='red', linestyle='-', marker='.', label=r'Re $\Sigma_{KK}$=KK(Im $\Sigma$)')
                    a.set_xlabel('Binding Energy (eV)', font='Arial', fontsize=self.size(14))
                    a.set_ylabel(r'Re $\Sigma$ (meV)', font='Arial', fontsize=self.size(14))
                    a.legend()
                    if dl==0:
                        b.scatter(tbe, iy, edgecolors='black', c='w', label=r'Im $\Sigma$')
                        b.scatter(tbe, reconstructed_imag[len(ix):2*len(ix)]+(iy-np.mean(reconstructed_imag[len(ix):2*len(ix)])), edgecolors='red', c='w', label=r'Im $\Sigma_{KK}$=KK(Re $\Sigma$)')
                    elif dl==1:
                        b.plot(tbe, iy, c='black', label=r'Im $\Sigma$')
                        b.plot(tbe, reconstructed_imag[len(ix):2*len(ix)]+(iy-np.mean(reconstructed_imag[len(ix):2*len(ix)])), c='red', label=r'Im $\Sigma_{KK}$=KK(Re $\Sigma$)')
                    elif dl==2:
                        b.plot(tbe, iy, c='black', linestyle='-', marker='.', label=r'Im $\Sigma$')
                        b.plot(tbe, reconstructed_imag[len(ix):2*len(ix)]+(iy-np.mean(reconstructed_imag[len(ix):2*len(ix)])), c='red', linestyle='-', marker='.', label=r'Im $\Sigma_{KK}$=KK(Re $\Sigma$)')
                    b.set_xlabel('Binding Energy (meV)', font='Arial', fontsize=self.size(14))
                    b.set_ylabel(r'Im $\Sigma$ (meV)', font='Arial', fontsize=self.size(14))
                    b.legend()
                    a.invert_xaxis()
                    b.invert_xaxis()
                elif 'Real Part' in value2.get():
                    ax = fig.subplots()
                    ttbe=tbe/1000
                    if 'nd' in value2.get():
                        ax.set_title(r'Self Energy $\Sigma$ Real Part', font='Arial', fontsize=self.size(20))
                        ty=np.diff(smooth(ry,20,3))/np.diff(ttbe)
                        if dl==0:
                            ax.scatter(ttbe[0:-1], ty, edgecolors='black', c='w', label=r'Re $\Sigma$')
                        elif dl==1:
                            ax.plot(ttbe[0:-1], ty, c='black', label=r'Re $\Sigma$')
                        elif dl==2:
                            ax.plot(ttbe[0:-1], ty, c='black', linestyle='-', marker='.', label=r'Re $\Sigma$')
                        ax.set_xlabel('Binding Energy (eV)', font='Arial', fontsize=self.size(18))
                        ax.set_ylabel(r'$2^{nd} der. Re \Sigma$', font='Arial', fontsize=self.size(18))
                        ax.set_xticklabels(ax.get_xticklabels(),fontsize=self.size(16))
                        ax.set_yticks([0])
                        ax.set_yticklabels(ax.get_yticklabels(),fontsize=self.size(16))
                    else:
                        ax.set_title(r'Self Energy $\Sigma$ Real Part', font='Arial', fontsize=self.size(20))
                        if dl==0:
                            ax.scatter(ttbe, ry, edgecolors='black', c='w', label=r'Re $\Sigma$')
                            ax.scatter(ttbe, reconstructed_real[len(ix):2*len(ix)]+(ry-np.mean(reconstructed_real[len(ix):2*len(ix)])), edgecolors='red', c='w', label=r'Re $\Sigma_{KK}$=KK(Im $\Sigma$)')
                        elif dl==1:
                            ax.plot(ttbe, ry, c='black', label=r'Re $\Sigma$')
                            ax.plot(ttbe, reconstructed_real[len(ix):2*len(ix)]+(ry-np.mean(reconstructed_real[len(ix):2*len(ix)])), c='red', label=r'Re $\Sigma_{KK}$=KK(Im $\Sigma$)')
                        elif dl==2:
                            ax.plot(ttbe, ry, c='black', linestyle='-', marker='.', label=r'Re $\Sigma$')
                            ax.plot(ttbe, reconstructed_real[len(ix):2*len(ix)]+(ry-np.mean(reconstructed_real[len(ix):2*len(ix)])), c='red', linestyle='-', marker='.', label=r'Re $\Sigma_{KK}$=KK(Im $\Sigma$)')
                        ax.set_xlabel('Binding Energy (eV)', font='Arial', fontsize=self.size(18))
                        ax.set_ylabel(r'Re $\Sigma$ (meV)', font='Arial', fontsize=self.size(18))
                        ax.set_xticklabels(ax.get_xticklabels(),fontsize=self.size(16))
                        ax.set_yticklabels(ax.get_yticklabels(),fontsize=self.size(16))
                        l=ax.legend(fontsize=self.size(16))
                        l.draw_frame(False)
                    ax.invert_xaxis()
                elif 'Imaginary Part' in value2.get():
                    ax = fig.subplots()
                    ttbe=tbe/1000
                    if 'st' in value2.get():
                        ax.set_title(r'Self Energy $\Sigma$ Imaginary Part', font='Arial', fontsize=self.size(20))
                        ty=np.diff(smooth(iy,20,3))/np.diff(ttbe)
                        if dl==0:
                            ax.scatter(ttbe[0:-1], ty, edgecolors='black', c='w', label=r'Im $\Sigma$')
                        elif dl==1:
                            ax.plot(ttbe[0:-1], ty, c='black', label=r'Im $\Sigma$')
                        elif dl==2:
                            ax.plot(ttbe[0:-1], ty, c='black', linestyle='-', marker='.', label=r'Im $\Sigma$')
                        ax.set_xlabel('Binding Energy (eV)', font='Arial', fontsize=self.size(18))
                        ax.set_ylabel(r'$1^{st} der. Im \Sigma$', font='Arial', fontsize=self.size(18))
                        ax.set_xticklabels(ax.get_xticklabels(),fontsize=self.size(16))
                        ax.set_yticks([0])
                        ax.set_yticklabels(ax.get_yticklabels(),fontsize=self.size(16))
                    else:
                        ax.set_title(r'Self Energy $\Sigma$ Imaginary Part', font='Arial', fontsize=self.size(20))
                        if dl==0:
                            ax.scatter(ttbe, iy, edgecolors='black', c='w', label=r'Im $\Sigma$')
                            ax.scatter(ttbe, reconstructed_imag[len(ix):2*len(ix)]+(iy-np.mean(reconstructed_imag[len(ix):2*len(ix)])), edgecolors='red', c='w', label=r'Im $\Sigma_{KK}$=KK(Re $\Sigma$)')
                        elif dl==1:
                            ax.plot(ttbe, iy, c='black', label=r'Im $\Sigma$')
                            ax.plot(ttbe, reconstructed_imag[len(ix):2*len(ix)]+(iy-np.mean(reconstructed_imag[len(ix):2*len(ix)])), c='red', label=r'Im $\Sigma_{KK}$=KK(Re $\Sigma$)')
                        elif dl==2:
                            ax.plot(ttbe, iy, c='black', linestyle='-', marker='.', label=r'Im $\Sigma$')
                            ax.plot(ttbe, reconstructed_imag[len(ix):2*len(ix)]+(iy-np.mean(reconstructed_imag[len(ix):2*len(ix)])), c='red', linestyle='-', marker='.', label=r'Im $\Sigma_{KK}$=KK(Re $\Sigma$)')
                        ax.set_xlabel('Binding Energy (eV)', font='Arial', fontsize=self.size(18))
                        ax.set_ylabel(r'Im $\Sigma$ (meV)', font='Arial', fontsize=self.size(18))
                        ax.set_xticklabels(ax.get_xticklabels(),fontsize=self.size(16))
                        ax.set_yticklabels(ax.get_yticklabels(),fontsize=self.size(16))
                        l=ax.legend(fontsize=self.size(16))
                        l.draw_frame(False)
                    ax.invert_xaxis()
                ##################################################################################
                ################################################################################## Hilbert Transform

            elif value2.get() == 'Data Plot with Pos' or value2.get() == 'Data Plot with Pos and Bare Band':
                ao = fig.subplots()
                if emf=='KE':
                    px, py = np.meshgrid(phi, ev)
                    tev = py.copy()
                else:
                    px, py =np.meshgrid(phi, vfe-ev)
                    tev = vfe-py.copy()
                if npzf:
                    px = phi
                else:
                    px = (2*m*tev*1.602176634*10**-19)**0.5*np.sin((np.float64(k_offset.get())+px)/180*np.pi)*10**-10/(h/2/np.pi)
                pz = data.to_numpy()
                h0 = ao.pcolormesh(px, py, pz, cmap=value3.get())
                txl = ao.get_xlim()
                tyl = ao.get_ylim()
                cb = fig.colorbar(h0)
                # cb.set_ticklabels(cb.get_ticks(), font='Arial', fontsize=self.size(14), minor=False)
                cb.set_ticklabels(cb.get_ticks(), font='Arial')
                
                #   MDC Norm
                # for i in range(len(ev)):
                #     b.scatter(mx[len(phi)*i:len(phi)*(i+1)],my[len(phi)*i:len(phi)*(i+1)],c=mz[len(phi)*i:len(phi)*(i+1)],marker='o',s=self.scale*self.scale*0.9,cmap='viridis',alpha=0.3)
                # a.set_title('MDC Normalized')
                ao.set_title(value2.get(), font='Arial', fontsize=self.size(18))
                # a.set_xlabel(r'k ($\frac{2\pi}{\AA}$)',fontsize=self.size(14))
                # a.set_ylabel('Kinetic Energy (eV)',fontsize=self.size(14))
                ao.set_xlabel(r'k ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=self.size(16))
                if emf=='KE':
                    ao.set_ylabel('Kinetic Energy (eV)', font='Arial', fontsize=self.size(16))
                else:
                    ao.set_ylabel('Binding Energy (eV)', font='Arial', fontsize=self.size(16))
                # b.set_xticklabels(labels=b.get_xticklabels(),fontsize=self.size(14))
                # b.set_yticklabels(labels=b.get_yticklabels(),fontsize=self.size(14))
                try:
                    if mp == 1:
                        if emf=='KE':
                            tb0 = ao.scatter(pos, fev, marker='.', s=self.scale*self.scale*0.3, c='black')
                        else:
                            tb0 = ao.scatter(pos, vfe-fev, marker='.', s=self.scale*self.scale*0.3, c='black')
                    if mf == 1:
                        ophimin = np.arcsin(
                            (rpos-fwhm/2)/np.sqrt(2*m*fev*1.602176634*10**-19)/10**-10*(h/2/np.pi))*180/np.pi
                        ophimax = np.arcsin(
                            (rpos+fwhm/2)/np.sqrt(2*m*fev*1.602176634*10**-19)/10**-10*(h/2/np.pi))*180/np.pi
                        posmin = np.sqrt(2*m*fev*1.602176634*10**-19)*np.sin(
                            (np.float64(k_offset.get())+ophimin)/180*np.pi)*10**-10/(h/2/np.pi)
                        posmax = np.sqrt(2*m*fev*1.602176634*10**-19)*np.sin(
                            (np.float64(k_offset.get())+ophimax)/180*np.pi)*10**-10/(h/2/np.pi)
                        if emf=='KE':
                            tb0_ = ao.scatter([posmin, posmax], [
                                            fev, fev], marker='|', c='grey', s=self.scale*self.scale*10, alpha=0.8)
                        else:
                            tb0_ = ao.scatter([posmin, posmax], [vfe-fev, vfe-fev], marker='|', c='grey', s=self.scale*self.scale*10, alpha=0.8)    
                except:
                    pass
                try:
                    if ep == 1:
                        if emf=='KE':
                            tb1 = ao.scatter(fk, epos, marker='.', s=self.scale*self.scale*0.3, c='black')
                        else:
                            tb1 = ao.scatter(fk, vfe-epos, marker='.', s=self.scale*self.scale*0.3, c='black')
                    if ef == 1:
                        eposmin = epos-efwhm/2
                        eposmax = epos+efwhm/2
                        if emf=='KE':
                            tb1_ = ao.scatter(
                                [fk, fk], [eposmin, eposmax], marker='_', c='grey', s=self.scale*self.scale*10, alpha=0.8)
                        else:
                            tb1_ = ao.scatter(
                                [fk, fk], [vfe-eposmin, vfe-eposmax], marker='_', c='grey', s=self.scale*self.scale*10, alpha=0.8)
                        
                except:
                    pass
                if value2.get() == 'Data Plot with Pos and Bare Band':
                    if emf=='KE':
                        tb2, = ao.plot(k*np.float64(bbk_offset.get()), (be -
                                    np.float64(bb_offset.get()))/1000+vfe, linewidth=self.scale*0.3, c='red', linestyle='--')
                    else:
                        tb2, = ao.plot(k*np.float64(bbk_offset.get()), (-be +
                                np.float64(bb_offset.get()))/1000, linewidth=self.scale*0.3, c='red', linestyle='--')
                    ao.set_xlim(txl)
                    ao.set_ylim(tyl)
                if emf=='BE':
                    ao.invert_yaxis()
            try:
                self.tb0, self.tb0_, self.tb1, self.tb1_, self.tb2 = tb0, tb0_, tb1, tb1_, tb2
                self.ao, self.h0, self.xl, self.yl, self.pflag = ao, h0, xl, yl, pflag
                self.pars3()
                if value2.get() != 'Real & Imaginary' and 'KK Transform' not in value2.get():
                    xl = ao.get_xlim()
                    yl = ao.get_ylim()
                    self.xl, self.yl = xl, yl
                    self.pars3()
                    self.climon()
                    out.draw()
                else:
                    self.climoff()
                    out.draw()
            except:
                pass
            print('Done')
            st.put('Done')
            self.main_plot_bind()
            gc.collect()

class exp_util(exp_motion):
    def __init__(self, scale: float, value: tk.StringVar, value1: tk.StringVar, value2: tk.StringVar, value3: tk.StringVar, k_offset: tk.StringVar,
                 be: np.ndarray, k: np.ndarray, bb_offset: tk.StringVar, bbk_offset: tk.StringVar,
                 emf: Literal['KE', 'BE'], data: xr.DataArray, vfe: float, ev: np.ndarray, phi: np.ndarray,
                 pos: np.ndarray, fwhm: np.ndarray, rpos: np.ndarray, ophi: np.ndarray, fev: np.ndarray,
                 epos: np.ndarray, efwhm: np.ndarray, fk: np.ndarray, ffphi: np.ndarray, fphi: np.ndarray,
                 mp: int, ep: int, mf: int, ef: int, xl: tuple[float], yl: tuple[float],
                 cm: tk.DoubleVar, cM: tk.DoubleVar, vcmin: tk.DoubleVar, vcmax: tk.DoubleVar, dl: int,
                 st: queue.Queue, pflag: int, limg: tk.Label, img: list[tk.PhotoImage],
                 d: int, l: int, p: int, npzf: bool, im_kernel: int,
                 rx: np.ndarray, ry: np.ndarray, ix: np.ndarray, iy: np.ndarray
                 ) -> None:
        self.cf = True
        self.scale = scale
        self.value, self.value1, self.value2, self.value3, self.k_offset = value, value1, value2, value3, k_offset
        self.be, self.k, self.bb_offset, self.bbk_offset = be, k, bb_offset, bbk_offset
        self.a, self.a0, self.f, self.f0, self.h1, self.h2 = None, None, None, None, None, None
        self.acx, self.acy, self.annot = None, None, None
        self.emf, self.data, self.vfe, self.ev, self.phi = emf, data, vfe, ev, phi
        self.pos, self.fwhm, self.rpos, self.ophi, self.fev = pos, fwhm, rpos, ophi, fev
        self.epos, self.efwhm, self.fk, self.ffphi, self.fphi = epos, efwhm, fk, ffphi, fphi
        self.mp, self.ep, self.mf, self.ef, self.xl, self.yl = mp, ep, mf, ef, xl, yl
        self.cm, self.cM, self.vcmin, self.vcmax, self.dl = cm, cM, vcmin, vcmax, dl
        self.pflag, self.st, self.limg, self.img = pflag, st, limg, img
        self.d, self.l, self.p, self.npzf, self.im_kernel = d, l, p, npzf, im_kernel
        self.rx, self.ry, self.ix, self.iy = rx, ry, ix, iy
        self.ta0, self.ta0_, self.ta1, self.ta1_, self.ta2 = None, None, None, None, None
    
    @abstractmethod
    def pars(self):
        pass
    
    @abstractmethod
    def show_info(self):
        pass
    
    @abstractmethod
    def show_version(self):
        pass
    
    def exp(self):
        def size(s):
            return int(s*self.scale)
        m, h = 9.1093837015e-31, 6.62607015e-34
        ta0, ta0_, ta1, ta1_, ta2 = self.ta0, self.ta0_, self.ta1, self.ta1_, self.ta2
        value, value1, value2, value3 = self.value, self.value1, self.value2, self.value3
        data, ev, phi = self.data, self.ev, self.phi
        fev, fwhm, pos = self.fev, self.fwhm, self.pos
        k, be = self.k, self.be
        k_offset, bb_offset, bbk_offset = self.k_offset, self.bb_offset, self.bbk_offset
        mp, ep, mf, ef, xl, yl = self.mp, self.ep, self.mf, self.ef, self.xl, self.yl
        vfe, emf = self.vfe, self.emf
        limg, img, st, pflag = self.limg, self.img, self.st, self.pflag
        pos, fwhm = self.pos, self.fwhm
        rpos, ophi = self.rpos, self.ophi
        epos, efwhm, fk = self.epos, self.efwhm, self.fk
        ffphi, fphi = self.ffphi, self.fphi
        cm, cM, vcmin, vcmax, dl = self.cm, self.cM, self.vcmin, self.vcmax, self.dl
        d, l, p, npzf, im_kernel = self.d, self.l, self.p, self.npzf, self.im_kernel
        scale = self.scale
        posmin, posmax, eposmin, eposmax = None, None, None, None
        rx, ry, ix, iy = self.rx, self.ry, self.ix, self.iy
        a, a0 = None, None
        acx, acy, annot = None, None, None
        ta0, ta0_, ta1, ta1_, ta2 = None, None, None, None, None
        
        props = dict(facecolor='green', alpha=0.3)
        
        limg.config(image=img[np.random.randint(len(img))])
        selectors = []
        cursor = []
        h1 = ''
        h2 = ''
        f = []
        f0 = []
        if pflag == 0:
            print('Choose a plot type first')
            st.put('Choose a plot type first')
            return
        if pflag == 1:
            if 'MDC Curves' not in value.get():
                mz = data.to_numpy()
                f0 = plt.figure(figsize=(8*scale, 7*scale), layout='constrained')
                a0 = plt.axes([0.13, 0.45, 0.8, 0.5])
                a1 = plt.axes([0.13, 0.08, 0.8, 0.2])
                a0.set_title('Drag to select specific region', font='Arial', fontsize=size(18))
                # f0.canvas.mpl_connect('key_press_event',toggle_selector)
            if value.get() != 'Raw Data' and 'MDC Curves' not in value.get():
                f, a = plt.subplots(dpi=150)
            elif value.get() == 'MDC Curves':
                f=plt.figure(figsize=(4*scale, 6*scale),dpi=150)
                a = f.subplots()
            elif value.get() == 'E-k with MDC Curves':
                f = plt.figure(figsize=(9*scale, 7*scale), layout='constrained')
                at_ = plt.axes([0.28, 0.15, 0.5, 0.75])
                at_.set_xticks([])
                at_.set_yticks([])
                a = plt.axes([0.13, 0.15, 0.4, 0.75])
                a1_ = plt.axes([0.53, 0.15, 0.4, 0.75])
            if value.get() == 'Raw Data':
                f = plt.figure(figsize=(9*scale, 7*scale), layout='constrained')
                a = plt.axes([0.13, 0.1, 0.55, 0.6])
                acx = plt.axes([0.13, 0.73, 0.55, 0.18])
                acy = plt.axes([0.7, 0.1, 0.15, 0.6])
                eacb = plt.axes([0.87, 0.1, 0.02, 0.6])
                if emf=='KE':
                    mx, my = np.meshgrid(phi, ev)
                else:
                    mx, my = np.meshgrid(phi, vfe-ev)
                # h1 = a.scatter(mx,my,c=mz,marker='o',s=scale*scale*0.9,cmap=value3.get());
                h1 = a.pcolormesh(mx, my, mz, cmap=value3.get())
                annot = a.annotate(
                    "", xy=(0,0), xytext=(20,20), textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w", alpha=0.6),
                    fontsize=size(14)
                    # arrowprops=dict(arrowstyle="->")
                )
                annot.set_visible(False)
                if emf=='KE':
                    yl = a.get_ylim()
                else:
                    yl = sorted(a.get_ylim(), reverse=True)
                cb = f.colorbar(h1, cax=eacb, orientation='vertical')
                # cb.set_ticklabels(cb.get_ticks(), font='Arial', fontsize=size(14))
                
                h2 = a0.pcolormesh(mx, my, mz, cmap=value3.get())
                cb1 = f0.colorbar(h2)
                # cb1.set_ticklabels(cb.get_ticks(), font='Arial', fontsize=size(14))

                acx.set_xticks([])
                acx.set_yticks([])
                acy.set_xticks([])
                acy.set_yticks([])
                
                n = a1.hist(mz.flatten(), bins=np.linspace(
                    min(mz.flatten()), max(mz.flatten()), 50), color='green')
                a1.set_xlabel('Intensity')
                a1.set_ylabel('Counts')
                a1.set_title('Drag to Select the range of Intensity ')
                self.a, self.a0, self.f, self.f0, self.h1, self.h2 = a, a0, f, f0, h1, h2
                self.acx, self.acy, self.annot = acx, acy, annot
                self.posmin, self.posmax, self.eposmin, self.eposmax = posmin, posmax, eposmin, eposmax
                self.ta0, self.ta0_, self.ta1, self.ta1_, self.ta2 = ta0, ta0_, ta1, ta1_, ta2
                self.rx, self.ry, self.ix, self.iy = rx, ry, ix, iy
                self.pars()
                # Exp_Motion = ExpMotion()
                f.canvas.mpl_connect('motion_notify_event', self.cut_move)
                f.canvas.mpl_connect('button_press_event', self.cut_select)
                f.canvas.mpl_connect('motion_notify_event', self.cur_on_move)
                selectors.append(SpanSelector(
                    a1,
                    self.onselect,
                    "horizontal",
                    useblit=True,
                    props=dict(alpha=0.3, facecolor="tab:blue"),
                    onmove_callback=self.onmove_callback,
                    interactive=True,
                    drag_from_anywhere=True,
                    snap_values=n[1]
                ))
            elif value.get() == 'First Derivative':
                pz = np.diff(smooth(data.to_numpy()))/np.diff(phi)
                if emf=='KE':
                    px, py = np.meshgrid(phi[0:-1], ev)
                    tev = py.copy()
                else:
                    px, py = np.meshgrid(phi[0:-1], vfe-ev)
                    tev = vfe-py.copy()
                if npzf:
                    px = phi[0:-1]+np.diff(phi)/2
                else:
                    px = (2*m*tev*1.602176634*10**-19)**0.5*np.sin((np.float64(k_offset.get())+px+np.diff(phi)/2)/180*np.pi)*10**-10/(h/2/np.pi)
                h1 = a.pcolormesh(px, py, pz, cmap=value3.get())
                # cursor = Cursor(a, useblit=True, color='red', linewidth=scale*1)
                annot = a.annotate(
                    "", xy=(0,0), xytext=(20,20), textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w", alpha=0.6),
                    fontsize=size(12)
                    # arrowprops=dict(arrowstyle="->")
                )
                annot.set_visible(False)
                if emf=='KE':
                    yl = a.get_ylim()
                else:
                    yl = sorted(a.get_ylim(), reverse=True)
                h2 = a0.pcolormesh(px, py, pz, cmap=value3.get())
                cb = f.colorbar(h1)
                # cb.set_ticklabels(cb.get_ticks(), font='Arial', fontsize=size(14))
                
                cb1 = f0.colorbar(h2)
                # cb1.set_ticklabels(cb1.get_ticks(), font='Arial', fontsize=size(14))

                n = a1.hist(pz.flatten(), bins=np.linspace(
                    min(pz.flatten()), max(pz.flatten()), 50), color='green')
                a1.set_xlabel('Intensity')
                a1.set_ylabel('Counts')
                a1.set_title('Drag to Select the range of Intensity ')
                self.a, self.a0, self.f, self.f0, self.h1, self.h2 = a, a0, f, f0, h1, h2
                self.acx, self.acy, self.annot = acx, acy, annot
                self.posmin, self.posmax, self.eposmin, self.eposmax = posmin, posmax, eposmin, eposmax
                self.ta0, self.ta0_, self.ta1, self.ta1_, self.ta2 = ta0, ta0_, ta1, ta1_, ta2
                self.rx, self.ry, self.ix, self.iy = rx, ry, ix, iy
                self.pars()
                # Exp_Motion = ExpMotion()
                f.canvas.mpl_connect('motion_notify_event', self.cur_move)
                f.canvas.mpl_connect('motion_notify_event', self.cur_on_move)
                selectors.append(SpanSelector(
                    a1,
                    self.onselect,
                    "horizontal",
                    useblit=True,
                    props=dict(alpha=0.3, facecolor="tab:blue"),
                    onmove_callback=self.onmove_callback,
                    interactive=True,
                    drag_from_anywhere=True,
                    snap_values=n[1]
                ))
            elif value.get() == 'Second Derivative':            
                pz = laplacian_filter(data.to_numpy(), im_kernel)
                if emf=='KE':
                    px, py = np.meshgrid(phi, ev)
                    tev = py.copy()
                else:
                    px, py = np.meshgrid(phi, vfe-ev)
                    tev = vfe-py.copy()
                if npzf:
                    px = phi
                else:
                    px = (2*m*tev*1.602176634*10**-19)**0.5*np.sin((np.float64(k_offset.get())+px)/180*np.pi)*10**-10/(h/2/np.pi)
                
                h1 = a.pcolormesh(px, py, pz, cmap=value3.get())
                # cursor = Cursor(a, useblit=True, color='red', linewidth=scale*1)
                annot = a.annotate(
                    "", xy=(0,0), xytext=(20,20), textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w", alpha=0.6),
                    fontsize=size(12)
                    # arrowprops=dict(arrowstyle="->")
                )
                annot.set_visible(False)
                if emf=='KE':
                    yl = a.get_ylim()
                else:
                    yl = sorted(a.get_ylim(), reverse=True)
                h2 = a0.pcolormesh(px, py, pz, cmap=value3.get())
                cb = f.colorbar(h1)
                # cb.set_ticklabels(cb.get_ticks(), font='Arial', fontsize=size(14))
                
                cb1 = f0.colorbar(h2)
                # cb1.set_ticklabels(cb1.get_ticks(), font='Arial', fontsize=size(14))

                n = a1.hist(pz.flatten(), bins=np.linspace(
                    min(pz.flatten()), max(pz.flatten()), 50), color='green')
                a1.set_xlabel('Intensity')
                a1.set_ylabel('Counts')
                a1.set_title('Drag to Select the range of Intensity ')
                self.a, self.a0, self.f, self.f0, self.h1, self.h2 = a, a0, f, f0, h1, h2
                self.acx, self.acy, self.annot = acx, acy, annot
                self.posmin, self.posmax, self.eposmin, self.eposmax = posmin, posmax, eposmin, eposmax
                self.ta0, self.ta0_, self.ta1, self.ta1_, self.ta2 = ta0, ta0_, ta1, ta1_, ta2
                self.rx, self.ry, self.ix, self.iy = rx, ry, ix, iy
                self.pars()
                # Exp_Motion = ExpMotion()
                f.canvas.mpl_connect('motion_notify_event', self.cur_move)
                f.canvas.mpl_connect('motion_notify_event', self.cur_on_move)
                selectors.append(SpanSelector(
                    a1,
                    self.onselect,
                    "horizontal",
                    useblit=True,
                    props=dict(alpha=0.3, facecolor="tab:blue"),
                    onmove_callback=self.onmove_callback,
                    interactive=True,
                    drag_from_anywhere=True,
                    snap_values=n[1]
                ))
            else:
                if value.get() == 'E-k Diagram':
                    # h1=a.scatter(mx,my,c=mz,marker='o',s=scale*scale*0.9,cmap=value3.get());
                    if emf=='KE':
                        px, py = np.meshgrid(phi, ev)
                        tev = py.copy()
                    else:
                        px, py = np.meshgrid(phi, vfe-ev)
                        tev = vfe-py.copy()
                    if npzf:
                        px = phi
                    else:
                        px = (2*m*tev*1.602176634*10**-19)**0.5*np.sin((np.float64(k_offset.get())+px)/180*np.pi)*10**-10/(h/2/np.pi)
                    pz = data.to_numpy()
                    h1 = a.pcolormesh(px, py, pz, cmap=value3.get())
                    # cursor = Cursor(a, useblit=True, color='red', linewidth=scale*1)
                    annot = a.annotate(
                        "", xy=(0,0), xytext=(20,20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w", alpha=0.6),
                        fontsize=size(12)
                        # arrowprops=dict(arrowstyle="->")
                    )
                    annot.set_visible(False)
                    if emf=='KE':
                        yl = a.get_ylim()
                    else:
                        yl = sorted(a.get_ylim(), reverse=True)
                    h2 = a0.pcolormesh(px, py, pz, cmap=value3.get())
                    cb = f.colorbar(h1)
                    # cb.set_ticklabels(cb.get_ticks(), font='Arial', fontsize=size(14))
                    
                    cb1 = f0.colorbar(h2)
                    # cb1.set_ticklabels(cb1.get_ticks(), font='Arial', fontsize=size(14))
                    

                    n = a1.hist(pz.flatten(), bins=np.linspace(
                        min(pz.flatten()), max(pz.flatten()), 50), color='green')
                    a1.set_xlabel('Intensity')
                    a1.set_ylabel('Counts')
                    a1.set_title('Drag to Select the range of Intensity ')
                    self.a, self.a0, self.f, self.f0, self.h1, self.h2 = a, a0, f, f0, h1, h2
                    self.acx, self.acy, self.annot = acx, acy, annot
                    self.posmin, self.posmax, self.eposmin, self.eposmax = posmin, posmax, eposmin, eposmax
                    self.ta0, self.ta0_, self.ta1, self.ta1_, self.ta2 = ta0, ta0_, ta1, ta1_, ta2
                    self.rx, self.ry, self.ix, self.iy = rx, ry, ix, iy
                    self.pars()
                    # Exp_Motion = ExpMotion()
                    f.canvas.mpl_connect('motion_notify_event', self.cur_move)
                    f.canvas.mpl_connect('motion_notify_event', self.cur_on_move)
                    selectors.append(SpanSelector(
                        a1,
                        self.onselect,
                        "horizontal",
                        useblit=True,
                        props=dict(alpha=0.3, facecolor="tab:blue"),
                        onmove_callback=self.onmove_callback,
                        interactive=True,
                        drag_from_anywhere=True,
                        snap_values=n[1]
                    ))
                elif value.get() == 'MDC Normalized':
                    if emf=='KE':
                        px, py = np.meshgrid(phi, ev)
                        tev = py.copy()
                    else:
                        px, py = np.meshgrid(phi, vfe-ev)
                        tev = vfe-py.copy()
                    if npzf:
                        px = phi
                    else:
                        px = (2*m*tev*1.602176634*10**-19)**0.5*np.sin((np.float64(k_offset.get())+px)/180*np.pi)*10**-10/(h/2/np.pi)
                    pz = data.to_numpy().copy().astype(float)
                    pz = np.nan_to_num(pz)
                    pz /= np.max(pz, axis=1)[:, np.newaxis]
                    a.pcolormesh(px, py, pz, cmap=value3.get())
                    a0.pcolormesh(px, py, pz, cmap=value3.get())
                    # cursor = Cursor(a, useblit=True, color='red', linewidth=scale*1)
                    annot = a.annotate(
                        "", xy=(0,0), xytext=(20,20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w", alpha=0.6),
                        fontsize=size(12)
                        # arrowprops=dict(arrowstyle="->")
                    )
                    annot.set_visible(False)
                    self.a, self.a0, self.f, self.f0, self.h1, self.h2 = a, a0, f, f0, h1, h2
                    self.acx, self.acy, self.annot = acx, acy, annot
                    self.posmin, self.posmax, self.eposmin, self.eposmax = posmin, posmax, eposmin, eposmax
                    self.ta0, self.ta0_, self.ta1, self.ta1_, self.ta2 = ta0, ta0_, ta1, ta1_, ta2
                    self.rx, self.ry, self.ix, self.iy = rx, ry, ix, iy
                    self.pars()
                    # Exp_Motion = ExpMotion()
                    f.canvas.mpl_connect('motion_notify_event', self.cur_move)
                    f.canvas.mpl_connect('motion_notify_event', self.cur_on_move)
                elif value.get() == 'MDC Curves':
                    y = np.zeros([len(ev),len(phi)],dtype=float)
                    for n in range(len(ev)):
                        ecut = data.sel(eV=ev[n], method='nearest')
                        if npzf:
                            x = phi
                        else:
                            x = (2*m*ev[n]*1.602176634*10**-19)**0.5*np.sin(
                            (np.float64(k_offset.get())+phi)/180*np.pi)*10**-10/(h/2/np.pi)
                        y[n][:] = ecut.to_numpy().reshape(len(ecut))
                    for n in range(len(ev)//d):
                        yy=y[n*d][:]+n*np.max(y)/d
                        yy=smooth(yy,l,p)
                        a.plot(x, yy, c='black')
                elif value.get() == 'E-k with MDC Curves':
                        y = np.zeros([len(ev),len(phi)],dtype=float)
                        for n in range(len(ev)):
                            ecut = data.sel(eV=ev[n], method='nearest')
                            if npzf:
                                x = phi
                            else:
                                x = (2*m*ev[n]*1.602176634*10**-19)**0.5*np.sin(
                                (np.float64(k_offset.get())+phi)/180*np.pi)*10**-10/(h/2/np.pi)
                            y[n][:] = ecut.to_numpy().reshape(len(ecut))
                        for n in range(len(ev)//d):
                            yy=y[n*d][:]+n*np.max(y)/d
                            yy=smooth(yy,l,p)
                            a1_.plot(x, yy, c='black')
                        if emf=='KE':
                            px, py = np.meshgrid(phi, ev)
                            tev = py.copy()
                        else:
                            px, py = np.meshgrid(phi, vfe-ev)
                            tev = vfe-py.copy()
                        if npzf:
                            px = phi
                        else:
                            px = (2*m*tev*1.602176634*10**-19)**0.5*np.sin((np.float64(k_offset.get())+px)/180*np.pi)*10**-10/(h/2/np.pi)
                        pz = data.to_numpy()
                        h1 = a.pcolormesh(px, py, pz, cmap=value3.get())
                        ylb=a1_.twinx()
                        ylb.set_ylabel('Intensity (a.u.)', font='Arial', fontsize=size(22))
                        ylb.set_yticklabels([])
                        # cb = fig.colorbar(h1, ax=a1_)
                        # cb.set_ticklabels(cb.get_ticks(), font='Arial', fontsize=size(20))
            if 'E-k with' not in value.get():
                if  value.get() != 'Raw Data':
                    a.set_title(value.get(), font='Arial', fontsize=size(18))
            else:
                at_.set_title(value.get(), font='Arial', fontsize=size(24))
            a.set_xlabel(r'k ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=size(16))
            # a.set_xticklabels(labels=a.get_xticklabels(), fontsize=size(20))
            if 'MDC Curves' not in value.get():
                a0.set_xlabel(r'k ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=size(16))
                # a0.set_xticklabels(labels=a0.get_xticklabels(), fontsize=size(14))
                if emf=='KE':
                    a.set_ylabel('Kinetic Energy (eV)', font='Arial', fontsize=size(16))
                    # a.set_yticklabels(labels=a.get_yticklabels(), fontsize=size(20))
                    a0.set_ylabel('Kinetic Energy (eV)', font='Arial', fontsize=size(16))
                    # a0.set_yticklabels(labels=a0.get_yticklabels(), fontsize=size(14))
                    if value.get() == 'Raw Data':
                        a.set_ylabel('Kinetic Energy (eV)', font='Arial', fontsize=size(16))
                        # a.set_yticklabels(labels=a.get_yticklabels(), fontsize=size(14))
                        a0.set_ylabel('Kinetic Energy (eV)', font='Arial', fontsize=size(16))
                        # a0.set_yticklabels(labels=a0.get_yticklabels(), fontsize=size(14))
                else:
                    a.set_ylabel('Binding Energy (eV)', font='Arial', fontsize=size(16))
                    # a.set_yticklabels(labels=a.get_yticklabels(), fontsize=size(20))
                    a0.set_ylabel('Binding Energy (eV)', font='Arial', fontsize=size(16))
                    # a0.set_yticklabels(labels=a0.get_yticklabels(), fontsize=size(14))
                    if value.get() == 'Raw Data':
                        a.set_ylabel('Binding Energy (eV)', font='Arial', fontsize=size(16))
                        # a.set_yticklabels(labels=a.get_yticklabels(), fontsize=size(14))
                        a0.set_ylabel('Binding Energy (eV)', font='Arial', fontsize=size(16))
                        # a0.set_yticklabels(labels=a0.get_yticklabels(), fontsize=size(14))
                    a.invert_yaxis()
                    a0.invert_yaxis()
            else:
                if 'E-k with' in value.get():
                    if emf=='KE':
                        a.set_ylabel('Kinetic Energy (eV)', font='Arial', fontsize=size(22))
                        a.set_yticklabels(labels=a.get_yticklabels(), fontsize=size(20))
                        a.set_ylim([ev[0], ev[n*d]])
                    else:
                        a.set_ylabel('Binding Energy (eV)', font='Arial', fontsize=size(22))
                        a.set_yticklabels(labels=a.get_yticklabels(), fontsize=size(20))
                        a.invert_yaxis()
                        a.set_ylim([vfe-ev[0], vfe-ev[n*d]])
                    a.set_xlabel(r'k ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=size(22))
                    a.set_xticklabels(labels=a.get_xticklabels(), fontsize=size(20))
                    a1_.set_xlabel(r'k ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=size(22))
                    a1_.set_xticklabels(labels=a1_.get_xticklabels(), fontsize=size(20))
                    a1_.set_yticklabels([])
                    a1_.set_xlim([min(x), max(x)])
                    a1_.set_ylim([0, np.max(n*np.max(y)/d)])
                else:
                    ylr=a.twinx()
                    a.set_ylabel('Intensity (a.u.)', font='Arial', fontsize=size(22))
                    a.set_yticklabels([])
                    ylr.set_ylabel(r'$\longleftarrow$ Binding Energy', font='Arial', fontsize=size(22))
                    ylr.set_yticklabels([])
                    a.set_xlim([min(x), max(x)])
                    a.set_ylim([0, np.max(n*np.max(y)/d)])
            if value.get() == 'Raw Data':
                acx.set_title('                Raw Data', font='Arial', fontsize=size(18))
                # cb.set_ticklabels(cb.get_ticks(), font='Arial', fontsize=size(14))
                if npzf:
                    a.set_xlabel(r'k ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=size(16))
                    a0.set_xlabel(r'k ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=size(16))
                else:
                    a.set_xlabel('Angle (deg)', font='Arial', fontsize=size(16))
                    a0.set_xlabel('Angle (deg)', font='Arial', fontsize=size(16))
                # a.set_xticklabels(labels=a.get_xticklabels(), fontsize=size(14))
                # a0.set_xticklabels(labels=a0.get_xticklabels(), fontsize=size(14))
            # a.set_xticklabels(labels=a.get_xticklabels(),fontsize=size(10))
            # a.set_yticklabels(labels=a.get_yticklabels(),fontsize=size(10))
            if 'MDC Curves' not in value.get():
                selectors.append(RectangleSelector(
                    a0, self.select_callback,
                    useblit=True,
                    button=[1, 3],  # disable middle button
                    minspanx=5, minspany=5,
                    spancoords='pixels',
                    interactive=True,
                    props=props))
        if pflag == 2:
            f, a = plt.subplots(2, 1, dpi=150)
            if value1.get() == 'MDC fitted Data':
                x = (vfe-fev)*1000

                a[0].set_title('MDC Fitting Result', font='Arial', fontsize=size(24))
                a[0].set_xlabel('Binding Energy (meV)', font='Arial', fontsize=size(22))
                a[0].set_xticklabels(labels=a[0].get_xticklabels(), fontsize=size(20))
                a[0].set_ylabel(
                    r'Position ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=size(22))
                a[0].set_yticklabels(labels=a[0].get_yticklabels(), fontsize=size(20))
                a[0].tick_params(direction='in')
                a[0].scatter(x, pos, c='black', s=scale*scale*5)

                a[1].set_xlabel('Binding Energy (meV)', font='Arial', fontsize=size(22))
                a[1].set_xticklabels(labels=a[1].get_xticklabels(), fontsize=size(20))
                a[1].set_ylabel(
                    r'FWHM ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=size(22))
                a[1].set_yticklabels(labels=a[1].get_yticklabels(), fontsize=size(20))
                a[1].tick_params(direction='in')
                a[1].scatter(x, fwhm, c='black', s=scale*scale*5)
                
                a[0].invert_xaxis()
                a[1].invert_xaxis()
            elif value1.get() == 'EDC fitted Data':
                x = fk

                a[0].set_title('EDC Fitting Result', font='Arial', fontsize=size(24))
                a[0].set_xlabel(
                    r'Position ($\frac{2\pi}{\AA}$', font='Arial', fontsize=size(22))
                a[0].set_xticklabels(labels=a[0].get_xticklabels(), fontsize=size(20))
                a[0].set_ylabel('Binding Energy (meV))', font='Arial', fontsize=size(22))
                a[0].set_yticklabels(labels=a[0].get_yticklabels(), fontsize=size(20))
                a[0].tick_params(direction='in')
                a[0].scatter(x, (vfe-epos)*1000, c='black', s=scale*scale*5)

                a[1].set_xlabel(
                    r'Position ($\frac{2\pi}{\AA}$', font='Arial', fontsize=size(22))
                a[1].set_xticklabels(labels=a[1].get_xticklabels(), fontsize=size(20))
                a[1].set_ylabel('FWHM (meV)', font='Arial', fontsize=size(22))
                a[1].set_yticklabels(labels=a[1].get_yticklabels(), fontsize=size(20))
                a[1].tick_params(direction='in')
                a[1].scatter(x, efwhm*1000, c='black', s=scale*scale*5)
                
                a[0].invert_yaxis()
                
            elif value1.get() == 'Real Part':
                x = (vfe-fev)*1000
                y = pos
                a[0].set_title('Real Part', font='Arial', fontsize=size(24))
                a[0].plot(rx, ry, c='black', linestyle='-', marker='.')

                a[0].tick_params(direction='in')
                a[0].set_xlabel('Binding Energy (meV)', font='Arial', fontsize=size(22))
                a[0].set_xticklabels(labels=a[0].get_xticklabels(), fontsize=size(20))
                a[0].set_ylabel(r'Re $\Sigma$ (meV)', font='Arial', fontsize=size(22))
                a[0].set_yticklabels(labels=a[0].get_yticklabels(), fontsize=size(20))

                h1 = a[1].scatter(y, x, c='black', s=scale*scale*5)
                h2 = a[1].scatter(k*np.float64(bbk_offset.get()),
                                -be+np.float64(bb_offset.get()), c='red', s=scale*scale*5)

                a[1].legend([h1, h2], ['fitted data', 'bare band'],fontsize=size(20))
                a[1].tick_params(direction='in')
                a[1].set_ylabel('Binding Energy (meV)', font='Arial', fontsize=size(22))
                a[1].set_yticklabels(labels=a[1].get_yticklabels(), fontsize=size(20))
                a[1].set_xlabel(
                    r'Pos ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=size(22))
                a[1].set_xticklabels(labels=a[1].get_xticklabels(), fontsize=size(20))
                
                a[0].invert_xaxis()
                a[1].invert_yaxis()

                # a[0].set_xlim([-1000,50])
                # a[0].set_ylim([-100,500])
                # a[1].set_ylim([-600,200])
                # a[1].set_xlim([-0.05,0.05])
            elif value1.get() == 'Imaginary Part':

                tbe = (vfe-fev)*1000

                x = interp(tbe, -be+np.float64(bb_offset.get()),
                        k*np.float64(bbk_offset.get()))
                y = interp(x, k*np.float64(bbk_offset.get()),
                        -be+np.float64(bb_offset.get()))
                xx = np.diff(x)
                yy = np.diff(y)

                # eliminate vf in gap
                for i in range(len(yy)):
                    if yy[i]/xx[i] > 20000:
                        yy[i] = 0
                v = yy/xx
                # v = np.append(v, v[-1])  # fermi velocity
                v=interp(pos,x[0:-1]+xx/2,v)
                yy = np.abs(v*fwhm/2)
                xx = tbe
                ax = a
                a = ax[0]
                b = ax[1]
                a.set_title('Imaginary Part', font='Arial', fontsize=size(24))
                a.plot(xx, yy, c='black', linestyle='-', marker='.')

                ix = xx
                iy = yy
                a.tick_params(direction='in')
                a.set_xlabel('Binding Energy (eV)', font='Arial', fontsize=size(22))
                a.set_xticklabels(labels=a.get_xticklabels(), fontsize=size(20))
                a.set_ylabel(r'Im $\Sigma$ (meV)', font='Arial', fontsize=size(22))
                a.set_yticklabels(labels=a.get_yticklabels(), fontsize=size(20))

                x = (vfe-fev)*1000
                y = fwhm
                b.plot(x, y, c='black', linestyle='-', marker='.')
                b.tick_params(direction='in')
                b.set_xlabel('Binding Energy (eV)', font='Arial', fontsize=size(22))
                b.set_xticklabels(labels=b.get_xticklabels(), fontsize=size(20))
                b.set_ylabel(r'FWHM ($\frac{2\pi}{\AA}$)',
                            font='Arial', fontsize=size(22))
                b.set_yticklabels(labels=b.get_yticklabels(), fontsize=size(20))

                x = (vfe-fev)*1000
                y = pos
                yy = interp(y, k*np.float64(bbk_offset.get()), be -
                            np.float64(bb_offset.get()))  # interp x into be,k set
                
                a.invert_xaxis()
                b.invert_xaxis()
        if pflag == 3:
            if value2.get() == 'Real & Imaginary':
                f, a = plt.subplots(2, 1, dpi=150)
                a[0].set_title(r'Self Energy $\Sigma$', font='Arial', fontsize=size(24))
                if dl==0:
                    a[0].scatter(rx, ry, edgecolors='black', c='w')
                elif dl==1:
                    a[0].plot(rx, ry, c='black')
                elif dl==2:
                    a[0].plot(rx, ry, c='black', linestyle='-', marker='.')
                a[0].tick_params(direction='in')
                a[0].set_xlabel('Binding Energy (meV)', font='Arial', fontsize=size(22))
                a[0].set_xticklabels(labels=a[0].get_xticklabels(), fontsize=size(20))
                a[0].set_ylabel(r'Re $\Sigma$ (meV)', font='Arial', fontsize=size(22))
                a[0].set_yticklabels(labels=a[0].get_yticklabels(), fontsize=size(20))
                if dl==0:
                    a[1].scatter(ix, iy, edgecolors='black', c='w')
                elif dl==1:
                    a[1].plot(ix, iy, c='black')
                elif dl==2:
                    a[1].plot(ix, iy, c='black', linestyle='-', marker='.')
                a[1].tick_params(direction='in')
                a[1].set_xlabel('Binding Energy (meV)', font='Arial', fontsize=size(22))
                a[1].set_xticklabels(labels=a[1].get_xticklabels(), fontsize=size(20))
                a[1].set_ylabel(r'Im $\Sigma$ (meV)', font='Arial', fontsize=size(22))
                a[1].set_yticklabels(labels=a[1].get_yticklabels(), fontsize=size(20))
                
                a[0].invert_xaxis()
                a[1].invert_xaxis()
            elif 'KK Transform' in value2.get():
                
                tbe = (vfe-fev)*1000
                ix=(tbe-tbe[-1])*-1
                cix=np.append(ix+ix[0],ix)
                tix=cix[0:len(cix)-1]*-1
                # kx=ix
                kx = np.append(cix,tix[::-1])
                ky = np.linspace(0, 1, len(kx))
                ciy=np.append(iy*0+np.mean(iy),iy)
                tiy=ciy[0:len(ciy)-1]
                ciy = np.append(ciy,tiy[::-1])

                #for imaginary part
                ix=(tbe-tbe[-1])*-1
                cix=np.append(ix+ix[0],ix)
                tix=cix[0:len(cix)-1]*-1
                kx = np.append(cix,tix[::-1])
                ky = np.linspace(0, 1, len(kx))
                cry=np.append(ry*0,ry)
                tcry=cry[0:len(cry)-1]*-1
                cry = np.append(cry,tcry[::-1])

                # Hilbert transform
                analytic_signal_r = hilbert(cry)
                amplitude_envelope_r = np.abs(analytic_signal_r)
                instantaneous_phase_r = np.unwrap(np.angle(analytic_signal_r))
                instantaneous_frequency_r = np.diff(instantaneous_phase_r) / (2.0 * np.pi)

                analytic_signal_i = hilbert(ciy)
                amplitude_envelope_i = np.abs(analytic_signal_i)
                instantaneous_phase_i = np.unwrap(np.angle(analytic_signal_i))
                instantaneous_frequency_i = np.diff(instantaneous_phase_i) / (2.0 * np.pi)

                # Reconstructed real and imaginary parts
                reconstructed_real = np.imag(analytic_signal_i)
                reconstructed_imag = -np.imag(analytic_signal_r)

                    # Plot
                if 'Real Part' not in value2.get() and 'Imaginary Part' not in value2.get():
                    f, a = plt.subplots(2, 1, dpi=150)
                    # Plot imaginary data and its Hilbert transformation
                    a[0].set_title(r'Self Energy $\Sigma$', font='Arial', fontsize=size(24))
                    if dl==0:
                        a[0].scatter(tbe, ry, edgecolors='black', c='w', label=r'Re $\Sigma$')
                        a[0].scatter(tbe, reconstructed_real[len(ix):2*len(ix)]+(ry-np.mean(reconstructed_real[len(ix):2*len(ix)])), edgecolors='red', c='w', label=r'Re $\Sigma_{KK}$=KK(Im $\Sigma$)')
                    elif dl==1:
                        a[0].plot(tbe, ry, c='black', label=r'Re $\Sigma$')
                        a[0].plot(tbe, reconstructed_real[len(ix):2*len(ix)]+(ry-np.mean(reconstructed_real[len(ix):2*len(ix)])), c='red', label=r'Re $\Sigma_{KK}$=KK(Im $\Sigma$)')
                    elif dl==2:
                        a[0].plot(tbe, ry, c='black', linestyle='-', marker='.', label=r'Re $\Sigma$')
                        a[0].plot(tbe, reconstructed_real[len(ix):2*len(ix)]+(ry-np.mean(reconstructed_real[len(ix):2*len(ix)])), c='red', linestyle='-', marker='.', label=r'Re $\Sigma_{KK}$=KK(Im $\Sigma$)')
                    a[0].set_xlabel('Binding Energy (meV)', font='Arial', fontsize=size(22))
                    a[0].set_xticklabels(a[0].get_xticklabels(), fontsize=size(20))
                    a[0].set_ylabel(r'Re $\Sigma$ (meV)', font='Arial', fontsize=size(22))
                    a[0].set_yticklabels(a[0].get_yticklabels(), fontsize=size(20))
                    a[0].legend(fontsize=size(20))
                    if dl==0:
                        a[1].scatter(tbe, iy, edgecolors='black', c='w', label=r'Im $\Sigma$')
                        a[1].scatter(tbe, reconstructed_imag[len(ix):2*len(ix)]+(iy-np.mean(reconstructed_imag[len(ix):2*len(ix)])), edgecolors='red', c='w', label=r'Im $\Sigma_{KK}$=KK(Re $\Sigma$)')
                    elif dl==1:
                        a[1].plot(tbe, iy, c='black', label=r'Im $\Sigma$')
                        a[1].plot(tbe, reconstructed_imag[len(ix):2*len(ix)]+(iy-np.mean(reconstructed_imag[len(ix):2*len(ix)])), c='red', label=r'Im $\Sigma_{KK}$=KK(Re $\Sigma$)')
                    elif dl==2:
                        a[1].plot(tbe, iy, c='black', linestyle='-', marker='.', label=r'Im $\Sigma$')
                        a[1].plot(tbe, reconstructed_imag[len(ix):2*len(ix)]+(iy-np.mean(reconstructed_imag[len(ix):2*len(ix)])), c='red', linestyle='-', marker='.', label=r'Im $\Sigma_{KK}$=KK(Re $\Sigma$)')
                    a[1].set_xlabel('Binding Energy (meV)', font='Arial', fontsize=size(22))
                    a[1].set_xticklabels(a[1].get_xticklabels(), fontsize=size(20))
                    a[1].set_ylabel(r'Im $\Sigma$ (meV)', font='Arial', fontsize=size(22))
                    a[1].set_yticklabels(a[1].get_yticklabels(), fontsize=size(20))
                    a[1].legend(fontsize=size(20))
                    a[0].invert_xaxis()
                    a[1].invert_xaxis()
                elif 'Real Part' in value2.get():
                    f = plt.figure(figsize=(8*scale, 7*scale),layout='constrained')
                    a=plt.axes([0.2,0.12,0.7,0.8])
                    ttbe=tbe/1000
                    if 'nd' in value2.get():
                        a.set_title(r'Self Energy $\Sigma$ Real Part', font='Arial', fontsize=size(24))
                        ty=np.diff(smooth(ry,20,3))/np.diff(ttbe)
                        if dl==0:
                            a.scatter(ttbe[0:-1], ty, edgecolors='black', c='w', label=r'Re $\Sigma$')
                        elif dl==1:
                            a.plot(ttbe[0:-1], ty, c='black', label=r'Re $\Sigma$')
                        elif dl==2:
                            a.plot(ttbe[0:-1], ty, c='black', linestyle='-', marker='.', label=r'Re $\Sigma$')
                        a.set_xlabel('Binding Energy (eV)', font='Arial', fontsize=size(22))
                        a.set_ylabel(r'$2^{nd} der. Re \Sigma$', font='Arial', fontsize=size(22))
                        a.set_xticklabels(a.get_xticklabels(),fontsize=size(20))
                        a.set_yticks([0])
                        a.set_yticklabels(a.get_yticklabels(),fontsize=size(20))
                    else:
                        a.set_title(r'Self Energy $\Sigma$ Real Part', font='Arial', fontsize=size(24))
                        if dl==0:
                            a.scatter(ttbe, ry, edgecolors='black', c='w', label=r'Re $\Sigma$')
                            a.scatter(ttbe, reconstructed_real[len(ix):2*len(ix)]+(ry-np.mean(reconstructed_real[len(ix):2*len(ix)])), edgecolors='red', c='w', label=r'Re $\Sigma_{KK}$=KK(Im $\Sigma$)')
                        elif dl==1:
                            a.plot(ttbe, ry, c='black', label=r'Re $\Sigma$')
                            a.plot(ttbe, reconstructed_real[len(ix):2*len(ix)]+(ry-np.mean(reconstructed_real[len(ix):2*len(ix)])), c='red', label=r'Re $\Sigma_{KK}$=KK(Im $\Sigma$)')
                        elif dl==2:
                            a.plot(ttbe, ry, c='black', linestyle='-', marker='.', label=r'Re $\Sigma$')
                            a.plot(ttbe, reconstructed_real[len(ix):2*len(ix)]+(ry-np.mean(reconstructed_real[len(ix):2*len(ix)])), c='red', linestyle='-', marker='.', label=r'Re $\Sigma_{KK}$=KK(Im $\Sigma$)')
                        a.set_xlabel('Binding Energy (eV)', font='Arial', fontsize=size(22))
                        a.set_ylabel(r'Re $\Sigma$ (meV)', font='Arial', fontsize=size(22))
                        a.set_xticklabels(a.get_xticklabels(),fontsize=size(20))
                        a.set_yticklabels(a.get_yticklabels(),fontsize=size(20))
                        ll=a.legend(fontsize=size(20))
                        ll.draw_frame(False)
                    a.invert_xaxis()
                elif 'Imaginary Part' in value2.get():
                    f = plt.figure(figsize=(8*scale, 7*scale),layout='constrained')
                    a=plt.axes([0.2,0.12,0.7,0.8])
                    ttbe=tbe/1000
                    if 'st' in value2.get():
                        a.set_title(r'Self Energy $\Sigma$ Imaginary Part', font='Arial', fontsize=size(24))
                        ty=np.diff(smooth(iy,20,3))/np.diff(ttbe)
                        if dl==0:
                            a.scatter(ttbe[0:-1], ty, edgecolors='black', c='w', label=r'Im $\Sigma$')
                        elif dl==1:
                            a.plot(ttbe[0:-1], ty, c='black', label=r'Im $\Sigma$')
                        elif dl==2:
                            a.plot(ttbe[0:-1], ty, c='black', linestyle='-', marker='.', label=r'Im $\Sigma$')
                        a.set_xlabel('Binding Energy (eV)', font='Arial', fontsize=size(22))
                        a.set_ylabel(r'$1^{st} der. Im \Sigma$', font='Arial', fontsize=size(22))
                        a.set_xticklabels(a.get_xticklabels(),fontsize=size(20))
                        a.set_yticks([0])
                        a.set_yticklabels(a.get_yticklabels(),fontsize=size(20))
                    else:
                        a.set_title(r'Self Energy $\Sigma$ Imaginary Part', font='Arial', fontsize=size(24))
                        if dl==0:
                            a.scatter(ttbe, iy, edgecolors='black', c='w', label=r'Im $\Sigma$')
                            a.scatter(ttbe, reconstructed_imag[len(ix):2*len(ix)]+(iy-np.mean(reconstructed_imag[len(ix):2*len(ix)])), edgecolors='red', c='w', label=r'Im $\Sigma_{KK}$=KK(Re $\Sigma$)')
                        elif dl==1:
                            a.plot(ttbe, iy, c='black', label=r'Im $\Sigma$')
                            a.plot(ttbe, reconstructed_imag[len(ix):2*len(ix)]+(iy-np.mean(reconstructed_imag[len(ix):2*len(ix)])), c='red', label=r'Im $\Sigma_{KK}$=KK(Re $\Sigma$)')
                        elif dl==2:
                            a.plot(ttbe, iy, c='black', linestyle='-', marker='.', label=r'Im $\Sigma$')
                            a.plot(ttbe, reconstructed_imag[len(ix):2*len(ix)]+(iy-np.mean(reconstructed_imag[len(ix):2*len(ix)])), c='red', linestyle='-', marker='.', label=r'Im $\Sigma_{KK}$=KK(Re $\Sigma$)')
                        a.set_xlabel('Binding Energy (eV)', font='Arial', fontsize=size(22))
                        a.set_ylabel(r'Im $\Sigma$ (meV)', font='Arial', fontsize=size(22))
                        a.set_xticklabels(a.get_xticklabels(),fontsize=size(20))
                        a.set_yticklabels(a.get_yticklabels(),fontsize=size(20))
                        ll=a.legend(fontsize=size(20))
                        ll.draw_frame(False)
                    a.invert_xaxis()
                
            elif value2.get() == 'Data Plot with Pos' or value2.get() == 'Data Plot with Pos and Bare Band':
                f0 = plt.figure(figsize=(8*scale, 7*scale), layout='constrained')
                a0 = plt.axes([0.13, 0.45, 0.8, 0.5])
                a1 = plt.axes([0.13, 0.08, 0.8, 0.2])
                a0.set_title('Drag to select specific region', font='Arial', fontsize=size(18))
                # f0.canvas.mpl_connect('key_press_event',toggle_selector)
                f, a = plt.subplots(dpi=150)
                if emf=='KE':
                    px, py = np.meshgrid(phi, ev)
                    tev = py.copy()
                else:
                    px, py = np.meshgrid(phi, vfe-ev)
                    tev = vfe - py.copy()
                if npzf:
                    px = phi
                else:
                    px = (2*m*tev*1.602176634*10**-19)**0.5*np.sin((np.float64(k_offset.get())+px)/180*np.pi)*10**-10/(h/2/np.pi)
                pz = data.to_numpy()
                h1 = a.pcolormesh(px, py, pz, cmap=value3.get())
                if emf=='KE':
                    yl = a.get_ylim()
                else:
                    yl = sorted(a.get_ylim(), reverse=True)
                cb = f.colorbar(h1)
                # cb.set_ticklabels(cb.get_ticks(), font='Arial', fontsize=size(14))
                
                a.set_title(value2.get(), font='Arial', fontsize=size(18))
                a.set_xlabel(r'k ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=size(16))
                # a.set_xticklabels(labels=a.get_xticklabels(), fontsize=size(20))
                if emf=='KE':
                    a.set_ylabel('Kinetic Energy (eV)', font='Arial', fontsize=size(16))
                else:
                    a.set_ylabel('Binding Energy (eV)', font='Arial', fontsize=size(16))
                # a.set_yticklabels(labels=a.get_yticklabels(), fontsize=size(20))
                try:
                    if mp == 1:
                        if emf=='KE':
                            a.scatter(pos, fev, marker='.', s=scale*scale*0.3, c='black')
                        else:
                            a.scatter(pos, vfe-fev, marker='.', s=scale*scale*0.3, c='black')
                            
                    if mf == 1:
                        ophimin = np.arcsin(
                            (rpos-fwhm/2)/(2*m*fev*1.602176634*10**-19)**0.5/10**-10*(h/2/np.pi))*180/np.pi
                        ophimax = np.arcsin(
                            (rpos+fwhm/2)/(2*m*fev*1.602176634*10**-19)**0.5/10**-10*(h/2/np.pi))*180/np.pi
                        posmin = (2*m*fev*1.602176634*10**-19)**0.5*np.sin(
                            (np.float64(k_offset.get())+ophimin)/180*np.pi)*10**-10/(h/2/np.pi)
                        posmax = (2*m*fev*1.602176634*10**-19)**0.5*np.sin(
                            (np.float64(k_offset.get())+ophimax)/180*np.pi)*10**-10/(h/2/np.pi)
                        if emf=='KE':
                            a.scatter([posmin, posmax], [fev, fev],
                                    marker='|', c='grey', s=scale*scale*10, alpha=0.8)
                        else:
                            a.scatter([posmin, posmax], [vfe-fev, vfe-fev],
                                    marker='|', c='grey', s=scale*scale*10, alpha=0.8)
                except:
                    pass
                try:
                    if ep == 1:
                        if emf=='KE':
                            a.scatter(fk, epos, marker='.', s=scale*scale*0.3, c='black')
                        else:
                            a.scatter(fk, vfe-epos, marker='.', s=scale*scale*0.3, c='black')
                                
                    if ef == 1:
                        eposmin = epos-efwhm/2
                        eposmax = epos+efwhm/2
                        if emf=='KE':
                            a.scatter([fk, fk], [eposmin, eposmax],
                                    marker='_', c='grey', s=scale*scale*10, alpha=0.8)
                        else:
                            a.scatter([fk, fk], [vfe-eposmin, vfe-eposmax],
                                    marker='_', c='grey', s=scale*scale*10, alpha=0.8)
                            
                except:
                    pass
                h2 = a0.pcolormesh(px, py, pz, cmap=value3.get())
                cb1 = f0.colorbar(h2)
                cb1.set_ticks(cb1.get_ticks())
                cb1.set_ticklabels(cb1.get_ticks(), font='Arial',
                                fontsize=size(14), minor=False)
                a0.set_xlabel(r'k ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=size(16))
                if emf=='KE':
                    a0.set_ylabel('Kinetic Energy (eV)', font='Arial', fontsize=size(16))
                else:
                    a0.set_ylabel('Binding Energy (eV)', font='Arial', fontsize=size(16))
                    
                try:
                    if mp == 1:
                        if emf=='KE':
                            a0.scatter(pos, fev, marker='.', s=scale*scale*0.3, c='black')
                        else:
                            a0.scatter(pos, vfe-fev, marker='.', s=scale*scale*0.3, c='black')
                            
                    if mf == 1:
                        ophimin = np.arcsin(
                            (rpos-fwhm/2)/(2*m*fev*1.602176634*10**-19)**0.5/10**-10*(h/2/np.pi))*180/np.pi
                        ophimax = np.arcsin(
                            (rpos+fwhm/2)/(2*m*fev*1.602176634*10**-19)**0.5/10**-10*(h/2/np.pi))*180/np.pi
                        posmin = (2*m*fev*1.602176634*10**-19)**0.5*np.sin(
                            (np.float64(k_offset.get())+ophimin)/180*np.pi)*10**-10/(h/2/np.pi)
                        posmax = (2*m*fev*1.602176634*10**-19)**0.5*np.sin(
                            (np.float64(k_offset.get())+ophimax)/180*np.pi)*10**-10/(h/2/np.pi)
                        if emf=='KE':
                            a0.scatter([posmin, posmax], [fev, fev],
                                    marker='|', c='grey', s=scale*scale*10, alpha=0.8)
                        else:
                            a0.scatter([posmin, posmax], [vfe-fev, vfe-fev],
                                    marker='|', c='grey', s=scale*scale*10, alpha=0.8)
                except:
                    pass
                try:
                    if ep == 1:
                        if emf=='KE':
                            a0.scatter(fk, epos, marker='.', s=scale*scale*0.3, c='black')
                        else:
                            a0.scatter(fk, vfe-epos, marker='.', s=scale*scale*0.3, c='black')
                            
                    if ef == 1:
                        eposmin = epos-efwhm/2
                        eposmax = epos+efwhm/2
                        if emf=='KE':
                            a0.scatter([fk, fk], [eposmin, eposmax],
                                    marker='_', c='grey', s=scale*scale*10, alpha=0.8)
                        else:
                            a0.scatter([fk, fk], [vfe-eposmin, vfe-eposmax],
                                    marker='_', c='grey', s=scale*scale*10, alpha=0.8)
                except:
                    pass
                # b.set_xticklabels(labels=b.get_xticklabels(),font='Arial',fontsize=size(20))
                # b.set_yticklabels(labels=b.get_yticklabels(),font='Arial',fontsize=size(20))

                n = a1.hist(pz.flatten(), bins=np.linspace(
                    min(pz.flatten()), max(pz.flatten()), 50), color='green')
                a1.set_xlabel('Intensity')
                a1.set_ylabel('Counts')
                a1.set_title('Drag to Select the range of Intensity ')
                try:
                    if value2.get() == 'Data Plot with Pos and Bare Band':
                        if emf=='KE':
                            a.plot(k*np.float64(bbk_offset.get()), (be -
                                np.float64(bb_offset.get()))/1000+vfe, linewidth=scale*0.3, c='red', linestyle='--')
                            a0.plot(k*np.float64(bbk_offset.get()), (be -
                                    np.float64(bb_offset.get()))/1000+vfe, linewidth=scale*0.3, c='red', linestyle='--')
                        else:
                            a.plot(k*np.float64(bbk_offset.get()), (-be +
                                np.float64(bb_offset.get()))/1000, linewidth=scale*0.3, c='red', linestyle='--')
                            a0.plot(k*np.float64(bbk_offset.get()), (-be +
                                    np.float64(bb_offset.get()))/1000, linewidth=scale*0.3, c='red', linestyle='--')
                except:
                    messagebox.showwarning("Warning", "Please load bare band file")
                    print('Please load Bare Band file')
                    st.put('Please load Bare Band file')
                    self.show_version()
                    return
                if emf=='BE':
                    a.invert_yaxis()
                    a0.invert_yaxis()
                # cursor = Cursor(a, useblit=True, color='red', linewidth=scale*1)
                annot = a.annotate(
                    "", xy=(0,0), xytext=(20,20), textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w", alpha=0.6),
                    fontsize=size(12)
                    # arrowprops=dict(arrowstyle="->")
                )
                annot.set_visible(False)
                self.a, self.a0, self.f, self.f0, self.h1, self.h2 = a, a0, f, f0, h1, h2
                self.acx, self.acy, self.annot = acx, acy, annot
                self.posmin, self.posmax, self.eposmin, self.eposmax = posmin, posmax, eposmin, eposmax
                self.ta0, self.ta0_, self.ta1, self.ta1_, self.ta2 = ta0, ta0_, ta1, ta1_, ta2
                self.rx, self.ry, self.ix, self.iy = rx, ry, ix, iy
                self.pars()
                # Exp_Motion = ExpMotion()
                f.canvas.mpl_connect('motion_notify_event', self.cur_move)
                f.canvas.mpl_connect('motion_notify_event', self.cur_on_move)
                selectors.append(RectangleSelector(
                    a0, self.select_callback,
                    useblit=True,
                    button=[1, 3],  # disable middle button
                    minspanx=5, minspany=5,
                    spancoords='pixels',
                    interactive=True,
                    props=props))
                selectors.append(SpanSelector(
                    a1,
                    self.onselect,
                    "horizontal",
                    useblit=True,
                    props=dict(alpha=0.3, facecolor="tab:blue"),
                    onmove_callback=self.onmove_callback,
                    interactive=True,
                    drag_from_anywhere=True,
                    snap_values=n[1]
                ))
        try:
            if value1.get() == '---Plot2---' and value2.get() != 'Real & Imaginary' and 'KK Transform' not in value2.get() and 'MDC Curves' != value.get():
                try:
                    h1.set_clim([vcmin.get(), vcmax.get()])
                    h2.set_clim([vcmin.get(), vcmax.get()])
                except:
                    pass
                try:    # ignore the problem occurred in E-k with MDC curves
                    a0.set_xlim(xl)
                    a0.set_ylim(yl)
                    a.set_xlim(xl)
                    a.set_ylim(yl)
                except:
                    pass
                if value.get() != 'Raw Data':
                    plt.tight_layout()
                # if value.get()=='Raw Data':
                #     f0.canvas.mpl_connect('motion_notify_event', cut_move)
                copy_to_clipboard(f)
                st.put('graph copied to clipboard')
                if value.get() != 'Raw Data':
                    threading.Thread(target=self.show_info,daemon=True).start()
                plt.show()
                try:
                    h1.set_clim([cm.get(), cM.get()])
                    h2.set_clim([cm.get(), cM.get()])
                except:
                    pass
            else:
                plt.tight_layout()
                copy_to_clipboard(f)
                st.put('graph copied to clipboard')
                threading.Thread(target=self.show_info,daemon=True).start()
                plt.show()
            # f.ion()
            # f0.ion()
        except:
            print('fail to export graph')
            pass
