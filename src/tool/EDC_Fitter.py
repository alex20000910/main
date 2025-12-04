from MDC_cut_utility import *
from tkinter import filedialog as fd
from tkinter import messagebox
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from lmfit import Parameters, Minimizer
from lmfit.printfuncs import alphanumeric_sort, gformat, report_fit
from scipy.optimize import curve_fit
import copy
import tqdm
import queue
import warnings

class EDC_Fitter(ABC):
    @abstractmethod
    def __init__(self):
        pass
    @abstractmethod
    def mfit(self):
        pass

# class fite:
#     def __init__(self, a):
#         print("Fitting started", a)

def init_pars(app_pars=None):
    if app_pars is not None:
        global ScaleFactor, sc_y, g, scale, npzf, vfe, emf, st, dpath, name, k_offset, value3, ev, phi, data, base, fpr, semin, semax, sefp, sefi, seaa1, seaa2, seresult, secst, m, h, seresult_original
        ScaleFactor = app_pars.ScaleFactor
        sc_y = app_pars.sc_y
        g = app_pars.g
        scale = app_pars.scale
        npzf = app_pars.npzf
        vfe = app_pars.vfe
        emf = app_pars.emf
        st = app_pars.st
        dpath = app_pars.dpath
        name = app_pars.name
        k_offset = app_pars.k_offset
        value3 = app_pars.value3
        ev = app_pars.ev
        phi = app_pars.phi
        data = app_pars.data
        base = app_pars.base
        fpr = app_pars.fpr
        semin = app_pars.semin
        semax = app_pars.semax
        sefp = app_pars.sefp
        sefi = app_pars.sefi
        seaa1 = app_pars.seaa1
        seaa2 = app_pars.seaa2
        # seresult = app_pars.seresult
        # secst = app_pars.secst
        m = 9.10938356e-31  # electron mass kg
        h = 6.62607015e-34  # Planck constant J·s
        # seresult_original = copy.deepcopy(seresult)
        clear(app_pars)
        app_pars = None

def _size(s: int) -> int:
    return int(s * scale)

def toa2(xx):
    global fswa1a2
    a2 = []
    a2.append(result.params['x1'].value)
    a2.append(result.params['h1'].value)
    a2.append(result.params['w1'].value)
    a2.append(result.params['y1'].value)
    a2.append(result.params['x2'].value)
    a2.append(result.params['h2'].value)
    a2.append(result.params['w2'].value)
    a2.append(result.params['y2'].value)
    
    fswa1a2 = 0
    i = efiti.get()
    
    # fmxx[i, :len(xx)] = xx
    # x = fmxx[i, np.argwhere(fmxx[i, :] >= -20)].flatten()
    x = xx
    ty = gl1(x, *a2[:4])
    s1 = np.sum(np.array([((ty[i]+ty[i+1])/2)for i in range(len(x)-1)])
            # Area 1
            * np.array(([(x[i+1]-x[i])for i in range(len(x)-1)])))
    ty = gl1(x, *a2[-4:])
    s2 = np.sum(np.array([((ty[i]+ty[i+1])/2)for i in range(len(x)-1)])
            # Area 2
            * np.array(([(x[i+1]-x[i])for i in range(len(x)-1)])))
    if s1 < s2:
        t1, t2 = a2[:4], a2[-4:]
        a2 = np.array([t2, t1]).flatten()
        fswa1a2 = 1
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

bg_warn = 0
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
        # ebase[i] = int(base.get())  # 待調整
        ebase[i] = 0
        # fexx[i, :] = fexx[i, :]/fexx[i, :]*-50
        # feyy[i, :] = feyy[i, :]/feyy[i, :]*-50
        x = ev
        angcut = data.sel(phi=phi[i], method='nearest')
        y = angcut.to_numpy().reshape(len(x))
        xx, x_arg = filter(x, emin[i], emax[i])
        # tx = x[np.argwhere(x >= emin[i])].flatten()
        # xx = tx[np.argwhere(tx <= emax[i])].flatten()
        # ty = y[np.argwhere(x >= emin[i])].flatten()
        # yy = ty[np.argwhere(tx <= emax[i])].flatten()
        # yy = np.where(yy > ebase[i], yy, ebase[i])
        yy = y[x_arg]
        yy = np.where(yy > ebase[i], yy, ebase[i])
        try:
            # if (emin[i],emax[i])==(np.min(ev),np.max(ev)) and i not in efi:
            # if i not in efi:
            #     if i not in efi_x:
            #         efi_x.append(i)
            #     if i in efi:
            #         efi.remove(i)
            #     if i in efi_err:
            #         efi_err.remove(i)
            #     a1 = [(emin[i]+emax[i])/2, (np.max(y)-ebase[i]), 5, ebase[i]]
            #     a2 = [(emin[i]+emax[i])/2, (np.max(y)-ebase[i]), 5, ebase[i],
            #           (emin[i]+emax[i])/2, (np.max(y)-ebase[i]), 5, ebase[i]]
            # # elif (emin[i],emax[i])!=(np.min(ev),np.max(ev)):
            # else:
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
                    a2 = toa2(xx)
                    checkfit()
                    if fit_warn == 1:
                        t = 5
                        while t > 0 and fit_warn == 1:
                            result = fitter.minimize()
                            a2 = toa2(xx)
                            checkfit()
                            t -= 1

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

        # fexx[i, :len(xx)] = xx
        # feyy[i, :len(yy)] = yy
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
    # ebase[i] = int(base.get())  # 待調整
    ebase[i] = 0
    # fexx[i, :] = fexx[i, :]/fexx[i, :]*-50
    # feyy[i, :] = feyy[i, :]/feyy[i, :]*-50
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
            a2 = toa2(xx)
            checkfit()
            if fit_warn == 1:
                t = 5
                while t > 0 and fit_warn == 1:
                    result = fitter.minimize()
                    a2 = toa2(xx)
                    checkfit()
                    t -= 1
            report_fit(result)

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

    # fexx[i, :len(xx)] = xx
    # feyy[i, :len(yy)] = yy
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
        eedxdata.config(text='xdata:'+str(' %.3f' % event.xdata))
        eedydata.config(text='ydata:'+str(' %.3f' % event.ydata))
    else:
        eedfitout.get_tk_widget().config(cursor="")
        try:
            eedxdata.config(text='xdata:')
            eedydata.config(text='ydata:')
        except NameError:
            pass


def saveefit():
    global epos, efwhm, fphi, efwhm, epos, semin, semax, seaa1, seaa2, sefp, sefi
    path = fd.asksaveasfilename(title="Save EDC Fitted Data", initialdir=dpath,
                                initialfile=name+"_efit", filetype=[("NPZ files", ".npz"),], defaultextension=".npz")
    try:
        egg.focus_force()
    except:
        pass
    if len(path) > 2:
        eendg.destroy()
        efwhm = res(sefi, efwhm)
        epos = res(sefi, epos)
        # semin = res(sefi, semin)
        # semax = res(sefi, semax)
        # sefp = res(sefi, sefp)
        fphi = res(sefi, fphi)
        sefi = res(sefi, sefi)
        np.savez(path, path=dpath, fphi=fphi, efwhm=efwhm, epos=epos, semin=semin,
                 semax=semax, seaa1=seaa1, seaa2=seaa2, sefp=sefp, sefi=sefi)
    else:
        eendg.focus_force()

scei = []


def feend():
    global epos, efwhm, fphi, eedxdata, eedydata, eedfitout, semin, semax, seaa1, seaa2, sefp, sefi, fk, fpr, scei, eendg
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
            
    efwhm = res(sefi, efwhm)
    epos = res(sefi, epos)
    # semin = res(sefi, semin)
    # semax = res(sefi, semax)
    # sefp = res(sefi, sefp)
    fphi = res(sefi, fphi)
    sefi = res(sefi, sefi)
            
    fphi, epos, efwhm = np.float64(fphi), np.float64(epos), np.float64(efwhm)
    ffphi = np.float64(k_offset.get())+fphi
    fk = (2*m*epos*1.602176634*10**-19)**0.5 * \
        np.sin(ffphi/180*np.pi)*10**-10/(h/2/np.pi)
    scei = cei
    fpr = 1
    if 'eendg' in globals():
        eendg.destroy()
    eendg = tk.Toplevel(g)
    eendg.title('EDC Lorentz Fit Result')
    fr = tk.Frame(master=eendg, bd=5)
    fr.grid(row=0, column=0)
    efitfig = Figure(figsize=(8*scale, 6*scale), layout='constrained')
    eedfitout = FigureCanvasTkAgg(efitfig, master=fr)
    eedfitout.get_tk_widget().grid(row=0, column=0)
    eedfitout.mpl_connect('motion_notify_event', feedmove)

    a = efitfig.subplots()
    a.scatter(fphi, epos+efwhm/2, c='r', s=scale*scale*10)
    a.scatter(fphi, epos-efwhm/2, c='r', s=scale*scale*10)
    a.scatter(fphi, epos, c='k', s=scale*scale*10)
    if npzf:a.set_xlabel(r'k ($\frac{2\pi}{\AA}$)')
    else:a.set_xlabel('Angle (deg)')
    a.set_ylabel('Kinetic Energy (eV)', fontsize=_size(14))
    eedfitout.draw()

    xydata = tk.Frame(master=fr, bd=2)
    xydata.grid(row=1, column=0)

    eedxdata = tk.Label(xydata, text='xdata:', font=(
        "Arial", _size(12), "bold"), width='15', height='1', bd=10, bg='white')
    eedxdata.grid(row=0, column=0)
    eedydata = tk.Label(xydata, text='ydata:', font=(
        "Arial", _size(12), "bold"), width='15', height='1', bd=10, bg='white')
    eedydata.grid(row=0, column=1)

    bsave = tk.Button(master=eendg, text='Save Fitted Data', command=saveefit,
                      width=30, height=1, font=('Arial', _size(14), "bold"), bg='white', bd=10)
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
    if npzf:
        efitax.set_title('k:'+str(round(evv[i], 3))+r' ($\frac{2\pi}{\AA}$)'+', '+str(efp[i])+' Peak')
    else:
        efitax.set_title('Deg:'+str(round(evv[i], 3))+r' $^{\circ}$'+', '+str(efp[i])+' Peak')
    efitax.scatter(fex[i, :], fey[i, :], c='k', s=scale*scale*4)
    x, x_arg = filter(fex[i, :], emin[i], emax[i])
    y = fey[i, x_arg]
    sbg = shirley_bg(y)
    # sbg = shirley_bg(feyy[i, np.argwhere(feyy[i, :] >= -20)])
    if efp[i] == 1:
        if eaa1[i, 0] == (emin[i]+emax[i])/2 and eaa1[i, 2] == 5:
            fl, = efitax.plot(x, gl1(
                x, *eaa1[i, :])+sbg, 'r-', lw=2)
        else:
            gl1_1 = gl1(x, *eaa1[i, :])+sbg
            fl, = efitax.plot(x, gl1(x, *eaa1[i, :])+sbg, 'b-', lw=2)
            efitax.fill_between(x, sbg, gl1_1, facecolor='blue', alpha=0.5)

    elif efp[i] == 2:
        if eaa2[i, 0] == (emin[i]+emax[i])/2 and eaa2[i, 2] == 5:
            fl, = efitax.plot(x, gl2(
                x, *eaa2[i, :])+sbg, 'r-', lw=2)
        else:
            gl2_1 = gl1(x, *eaa2[i, :4])+sbg
            gl2_2 = gl1(x, *eaa2[i, -4:])+sbg
            fl, = efitax.plot(x, gl2(
                x, *eaa2[i, :])+sbg, 'b-', lw=2)
            efitax.fill_between(x, sbg, gl2_1, facecolor='green', alpha=0.5)
            efitax.fill_between(x, sbg, gl2_2, facecolor='purple', alpha=0.5)

    if bg_warn == 1:  # shirley base line warn
        efitax.plot(x, sbg, 'r--')
    else:
        efitax.plot(x, sbg, 'g--')

    efitax.scatter(x,
                   y, c='g', s=scale*scale*4)
    if (emin[i], emax[i]) != (np.min(ev), np.max(ev)):
        elmin = efitax.axvline(emin[i], c='r')
        elmax = efitax.axvline(emax[i], c='r')
    else:
        elmin = efitax.axvline(emin[i], c='grey')
        elmax = efitax.axvline(emax[i], c='grey')
        fl.set_alpha(0.3)

    efitax.set_xlabel('Kinetic Energy (eV)', fontsize=_size(14))
    efitax.set_ylabel('Intensity (Counts)', fontsize=_size(14))
    efitax.set_xticklabels(np.round(efitax.get_xticks(),2), fontsize=_size(12))
    efitax.set_yticklabels(np.round(efitax.get_yticks(),2), fontsize=_size(12))
    exl = efitax.get_xlim()
    eyl = efitax.get_ylim()
    texl = np.copy(exl)
    efitout.draw()
    eplfi()


def emove(event):
    global exdata, eydata, edxdata, edydata, x2, y2, efitax, efitout, elmin, elmax, emin, emax, tpx1, tpx2, tpy1, tpy2, tx2, ty2
    if event.xdata != None:
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
                edxdata.config(text='dx:'+str(' %.3f' % abs(x2-x1)))
                edydata.config(text='dy:'+str(' %.3f' % abs(y2-y1)))
        exdata.config(text='xdata:'+str(' %.3f' % event.xdata))
        eydata.config(text='ydata:'+str(' %.3f' % event.ydata))
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


def egg_close():
    global egg
    try:
        flag = True
        # count = 0
        # seresult = pack_fitpar(eresult)
        # for i, j in zip(seresult, seresult_original):
        #     if not np.array_equal(i, j):
        #         flag = False
        #         break
        #     count += 1
        if flag:
            egg.destroy()
            clear(egg)
            egg=True
        else:
            if messagebox.askyesno("MDC Fitter", "Unsaved changes detected. Do you want to exit without saving?", default='no', icon='warning'):
                try:
                    egg.destroy()
                    clear(egg)
                    egg=True
                except:
                    pass
            else:
                feend()
                saveefit()
    except Exception as ex:
        print(f"Error:({__file__}, line:{ex.__traceback__.tb_lineno})", ex)
        egg.destroy()
        clear(egg)
        egg=True

def ejob():     # EDC Fitting GUI
    global g, efiti, efitfig, efitout, egg, exdata, eydata, edxdata, edydata, eiout, eifig, efi, efi_err, efi_x, ebrmv, flermv, ebcgl2, efp, flecgl2, fpr, est, estate, ewf1, ewf2, eaf1, eaf2, elind, erind, ein_w1, ein_w2, ein_a1, ein_a2
    egg = tk.Toplevel(g, bg='white')
    egg.protocol("WM_DELETE_WINDOW", egg_close)
    edpi = egg.winfo_fpixels('1i')
    t_sc_w = windll.user32.GetSystemMetrics(0)
    tx = t_sc_w if g.winfo_x()+g.winfo_width()/2 > t_sc_w else 0
    egg.geometry(f"1900x1000+{tx}+{sc_y}")
    egg.title('EDC Lorentz Fit')
    est = queue.Queue(maxsize=0)
    estate = tk.Label(egg, text='', font=(
        "Arial", _size(14), "bold"), bg="white", fg="black")
    estate.grid(row=0, column=0)

    fr = tk.Frame(master=egg, bg='white')
    fr.grid(row=1, column=0)
    frind = tk.Frame(master=fr, bg='white')
    frind.grid(row=0, column=0)
    elind = tk.Button(frind, text='<<', command=eflind, width=10,
                      height=5, font=('Arial', _size(12), "bold"), bg='white')
    elind.grid(row=0, column=0)
    erind = tk.Button(frind, text='>>', command=efrind, width=10,
                      height=5, font=('Arial', _size(12), "bold"), bg='white')
    erind.grid(row=0, column=2)

    efiti = tk.IntVar()
    efiti.set(0)
    efiti.trace_add('write', fchei)
    if ScaleFactor <= 100:
        tlength = int(1/0.975*6*edpi)  # 100
        twidth = int(1/0.975*0.2*edpi)
    elif ScaleFactor <= 125:
        tlength = int(1/0.985*6*edpi)  # 125
        twidth = int(1/0.985*0.2*edpi)
    elif ScaleFactor <= 150:
        tlength = int(1*6*edpi)  # 150
        twidth = int(1*0.2*edpi)
    elif ScaleFactor <= 175:
        tlength = int(0.99*6*edpi)  # 175
        twidth = int(0.99*0.2*edpi)
    elif ScaleFactor <= 200:
        tlength = int(0.985*6*edpi)  # 200
        twidth = int(0.985*0.2*edpi)
    elif ScaleFactor <= 225:
        tlength = int(0.98*6*edpi)  # 225
        twidth = int(0.98*0.2*edpi)
    elif ScaleFactor <= 250:
        tlength = int(0.977*6*edpi)  # 250
        twidth = int(0.977*0.2*edpi)
    elif ScaleFactor <= 275:
        tlength = int(0.975*6*edpi)  # 275
        twidth = int(0.975*0.2*edpi)
    elif ScaleFactor <= 300:
        tlength = int(0.97*6*edpi)  # 300
        twidth = int(0.97*0.2*edpi)
    tlength = int(tlength*scale)
    twidth = int(twidth*scale)
    chi = tk.Scale(frind, label='Index', from_=0, to=len(phi)-1, orient='horizontal',
                   variable=efiti, state='active', bg='white', fg='black', length=tlength, width=twidth, resolution=1)
    chi.grid(row=0, column=1)

    efi, efi_err, efi_x = [], [], [i for i in range(len(phi))]
    eifig = Figure(figsize=(6*scale, 0.2*scale), layout='tight')
    eiout = FigureCanvasTkAgg(eifig, master=frind)
    eiout.get_tk_widget().grid(row=1, column=1)

    efitfig = Figure(figsize=(8*scale, 6*scale), layout='constrained')
    efitout = FigureCanvasTkAgg(efitfig, master=fr)
    efitout.get_tk_widget().grid(row=1, column=0)
    efitout.mpl_connect('motion_notify_event', emove)
    efitout.mpl_connect('button_press_event', epress)
    efitout.mpl_connect('button_release_event', erelease)

    xydata = tk.Frame(master=fr, bd=5, bg='white')
    xydata.grid(row=2, column=0)

    exdata = tk.Label(xydata, text='xdata:', font=(
        "Arial", _size(12), "bold"), width='15', height='1', bd=5, bg='white')
    exdata.grid(row=0, column=0)
    eydata = tk.Label(xydata, text='ydata:', font=(
        "Arial", _size(12), "bold"), width='15', height='1', bd=5, bg='white')
    eydata.grid(row=0, column=1)
    edxdata = tk.Label(xydata, text='dx:', font=(
        "Arial", _size(12), "bold"), width='15', height='1', bd=5, bg='white')
    edxdata.grid(row=0, column=2)
    edydata = tk.Label(xydata, text='dy:', font=(
        "Arial", _size(12), "bold"), width='15', height='1', bd=5, bg='white')
    edydata.grid(row=0, column=3)

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
        "Arial", _size(12), "bold"), width='15', height='1', bd=5, bg='white')
    l1.grid(row=0, column=0)
    froperind = tk.Frame(master=frpara00, bd=5, bg='white')
    froperind.grid(row=1, column=0)
    ebcgl2 = tk.Button(froperind, text='Start Add 2 Peaks', command=fecgl2,
                       width=30, height=1, font=('Arial', _size(16), "bold"), bg='white')
    ebcgl2.grid(row=0, column=0)
    ebrmv = tk.Button(froperind, text='Start Remove', command=fermv,
                      width=30, height=1, font=('Arial', _size(16), "bold"), bg='white')
    ebrmv.grid(row=0, column=1)

    frwr = tk.Frame(master=froperind, bd=5, bg='white')
    frwr.grid(row=1, column=0)
    l2 = tk.Label(frwr, text='FWHM Ratio', font=(
        "Arial", _size(12), "bold"), width='15', height='1', bd=5, bg='white')
    l2.grid(row=0, column=1)
    l3 = tk.Label(frwr, text=':', font=("Arial", _size(12), "bold"),
                  width='15', height='1', bd=5, bg='white')
    l3.grid(row=1, column=1)
    ewf1 = tk.StringVar()
    ewf1.set('0')
    ewf1.trace_add('write', fewf1)
    ein_w1 = tk.Entry(frwr, font=("Arial", _size(12), "bold"),
                      width=7, textvariable=ewf1, bd=5)
    ein_w1.grid(row=1, column=0)
    ewf2 = tk.StringVar()
    ewf2.set('0')
    ewf2.trace_add('write', fewf2)
    ein_w2 = tk.Entry(frwr, font=("Arial", _size(12), "bold"),
                      width=7, textvariable=ewf2, bd=5)
    ein_w2.grid(row=1, column=2)

    frar = tk.Frame(master=froperind, bd=5, bg='white')
    frar.grid(row=2, column=0)
    l2 = tk.Label(frar, text='Area Ratio', font=(
        "Arial", _size(12), "bold"), width='15', height='1', bd=5, bg='white')
    l2.grid(row=0, column=1)
    l3 = tk.Label(frar, text=':', font=("Arial", _size(12), "bold"),
                  width='15', height='1', bd=5, bg='white')
    l3.grid(row=1, column=1)
    eaf1 = tk.StringVar()
    eaf1.set('0')
    eaf1.trace_add('write', feaf1)
    ein_a1 = tk.Entry(frar, font=("Arial", _size(12), "bold"),
                      width=7, textvariable=eaf1, bd=5)
    ein_a1.grid(row=1, column=0)
    eaf2 = tk.StringVar()
    eaf2.set('0')
    eaf2.trace_add('write', feaf2)
    ein_a2 = tk.Entry(frar, font=("Arial", _size(12), "bold"),
                      width=7, textvariable=eaf2, bd=5)
    ein_a2.grid(row=1, column=2)

    frout = tk.Frame(master=egg, bd=5, bg='white')
    frout.grid(row=2, column=0)
    bfall = tk.Button(frout, text='Fit All', command=fefall,
                      width=30, height=1, font=('Arial', _size(14), "bold"), bg='white')
    bfall.grid(row=0, column=0)
    flermv = -1
    bend = tk.Button(frout, text='Finish', command=feend, width=30,
                     height=1, font=('Arial', _size(16), "bold"), bg='white')
    bend.grid(row=1, column=0)

    if eprfit == 1:
        fefall()
    else:
        efitplot()
    tt = threading.Thread(target=testate)
    tt.daemon = True
    tt.start()
    egg.update()
    screen_width = egg.winfo_reqwidth()
    screen_height = egg.winfo_reqheight()
    tx = int(t_sc_w*windll.shcore.GetScaleFactorForDevice(0)/100) if g.winfo_x()+g.winfo_width()/2 > t_sc_w else 0
    egg.geometry(f"{screen_width}x{screen_height}+{tx}+{sc_y}")
    egg.update()

eprfit = 0


class oelim():
    def __init__(self, npzf, ev, phi):
        if npzf:
            avg = np.mean(ev)
            l = max(ev) - min(ev)
            self.min = np.float64([avg - l/40 for i in phi])
            self.max = np.float64([avg + l/40 for i in phi])
        else:
            self.min = [np.min(ev) for i in phi]
            self.max = [np.max(ev) for i in phi]

egg=None
def fite(edc_pars):
    if egg is not None:
        egg.lift()
        return
    init_pars(edc_pars)
    global ev, phi, data, evv, eaa1, eaa2, fexx, feyy, fex, fey, emin, emax, cei, ebase, eprfit
    cei = []
    ebase = [0 for i in range(len(phi))]
    elim = oelim(npzf, ev, phi)
    if fpr == 1:
        try:
            emin, emax = semin, semax
        except NameError:
            emin, emax = elim.min.copy(), elim.max.copy()
        if len(scei) >= 2:
            cei = scei
    else:
        emin, emax = elim.min.copy(), elim.max.copy()
    # fexx = np.float64((np.ones(len(ev)*len(phi))).reshape(len(phi), len(ev)))
    # feyy = np.float64((np.ones(len(ev)*len(phi))).reshape(len(phi), len(ev)))
    # fexx *= -50
    # feyy *= -50
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
                # a1 = [(emin[i]+emax[i])/2, (np.max(y) -
                #                             int(base.get())), 5, int(base.get())]
                # a2 = [(emin[i]+emax[i])/2, (np.max(y)-int(base.get())), 5, int(base.get()),
                #       (emin[i]+emax[i])/2, (np.max(y)-int(base.get())), 5, int(base.get())]
                a1 = [(emin[i]+emax[i])/2, (np.max(y)-0), 5, 0]
                a2 = [(emin[i]+emax[i])/2, (np.max(y)-0), 5, 0,
                    (emin[i]+emax[i])/2, (np.max(y)-0), 5, 0]
        except:
            # a1 = [(emin[i]+emax[i])/2, (np.max(y) -
            #                             int(base.get())), 5, int(base.get())]
            # a2 = [(emin[i]+emax[i])/2, (np.max(y)-int(base.get())), 5, int(base.get()),
            #       (emin[i]+emax[i])/2, (np.max(y)-int(base.get())), 5, int(base.get())]
            a1 = [(emin[i]+emax[i])/2, (np.max(y)-0), 5, 0]
            a2 = [(emin[i]+emax[i])/2, (np.max(y)-0), 5, 0,
                    (emin[i]+emax[i])/2, (np.max(y)-0), 5, 0]

        # fexx[i, :len(xx)] = xx
        # feyy[i, :len(yy)] = yy
        fex[i, :] = x
        fey[i, :] = y
        evv[i] = v
        eaa1[i, :] = a1
        eaa2[i, :] = a2
        pbar.update(1)
    pbar.close()
    # global egg
    # try:
    #     egg.destroy()
    # except:
    #     pass
    tt2 = threading.Thread(target=ejob)
    tt2.daemon = True
    tt2.start()
