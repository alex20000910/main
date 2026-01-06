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
import copy
import tqdm
import queue

class MDC_Fitter(ABC):
    @abstractmethod
    def __init__(self):
        pass
    @abstractmethod
    def mfit(self):
        pass

# class fitm:
#     def __init__(self, a):
#         print("Fitting started", a)

def init_pars(app_pars=None):
    if app_pars is not None:
        global ScaleFactor, sc_y, g, scale, npzf, vfe, emf, st, dpath, name, k_offset, value3, ev, phi, data, base, fpr, skmin, skmax, smfp, smfi, smaa1, smaa2, smresult, smcst, mdet, m, h, smresult_original
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
        skmin = app_pars.skmin
        skmax = app_pars.skmax
        smfp = app_pars.smfp
        smfi = app_pars.smfi
        smaa1 = app_pars.smaa1
        smaa2 = app_pars.smaa2
        smresult = app_pars.smresult
        smcst = app_pars.smcst
        mdet = app_pars.mdet
        m = 9.10938356e-31  # electron mass kg
        h = 6.62607015e-34  # Planck constant J·s
        smresult_original = copy.deepcopy(smresult)
        clear(app_pars)
        app_pars = None

def _size(s: int) -> int:
    return int(s * scale)

#############################

wr1 = 0
wr2 = 0
fa1 = 0
fa2 = 0

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

def toa1():
    a1 = []
    a1.append(result.params['x'].value)
    a1.append(result.params['h'].value)
    a1.append(result.params['w'].value)
    a1.append(result.params['y'].value)
    return a1

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
    i = mfiti.get()
    
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

def swapc1c2():
    i = mfiti.get()
    if mfp[i] == 2 and fswa1a2 == 1:
        o_result = copy.deepcopy(result)
        a1=['x1', 'h1', 'w1', 'y1']
        a2=['x2', 'h2', 'w2', 'y2']
        for i in range(4):
            if o_result.params[a2[i]].expr is not None:
                if a1[i] in o_result.params[a2[i]].expr:
                    o_result.params[a2[i]].set(expr=o_result.params[a2[i]].expr.replace(a1[i], a2[i]))
            result.params[a1[i]].set(value=o_result.params[a2[i]].value, min=o_result.params[a2[i]].min, max=o_result.params[a2[i]].max, expr=o_result.params[a2[i]].expr, brute_step=o_result.params[a2[i]].brute_step, vary=o_result.params[a2[i]].vary)
            result.params[a2[i]].set(value=o_result.params[a1[i]].value, min=o_result.params[a1[i]].min, max=o_result.params[a1[i]].max, expr=o_result.params[a1[i]].expr, brute_step=o_result.params[a1[i]].brute_step, vary=o_result.params[a1[i]].vary)
    return result


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
                inval = f'(init = {par.init_value:.7g})'
            if modelpars is not None and name in modelpars:
                inval = f'{inval}, model_value = {modelpars[name].value:.7g}'
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
                add(f"    {nout} {par.value: .7g} (fixed)")
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
        if 'x2*xr1+xr2' in i:
            if xr2>=0:
                i = i.replace(' == \'x2*xr1+xr2\'', '='+str(xr1)+'*x2+'+str(xr2))
            else:
                i = i.replace(' == \'x2*xr1+xr2\'', '='+str(xr1)+'*x2-'+str(-xr2))
        if "(x2-xr2) / xr1" in i:
            if xr2>=0:
                i = i.replace(' == \'(x2-xr2) / xr1\'','=(x2-'+str(xr2) + ')/'+str(xr1))
            else:
                i = i.replace(' == \'(x2-xr2) / xr1\'','=(x2+'+str(-xr2) + ')/'+str(xr1))
        if "(x1-xr2) / xr1" in i:
            if xr2>=0:
                i = i.replace(' == \'(x1-xr2) / xr1\'','=(x1-'+str(xr2) + ')/'+str(xr1))
            else:
                i = i.replace(' == \'(x1-xr2) / xr1\'','=(x1+'+str(-xr2) + ')/'+str(xr1))
        if 'w1/wr1*wr2' in i:
            i = i.replace(' == \'w1/wr1*wr2\'', '=w1/'+str(wr1)+'*'+str(wr2))
        if 'w2/wr1*wr2' in i:
            i = i.replace(' == \'w2/wr1*wr2\'', '=w2/'+str(wr1)+'*'+str(wr2))
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


def lnr_bg(x: np.ndarray, n_samples=5) -> np.ndarray:
    while len(x) < 2*n_samples:
        if len(x) < 2:
            o = np.array([])
        n_samples -= 1
    left, right = np.mean(x[:n_samples]), np.mean(x[-n_samples:])
    o = np.ones(len(x))*np.mean([left, right])
    return o+mbgv

############################### separator ###############################




def fmcgl2():
    global mbcgl2, kmin, kmax, flmcgl2, micgl2, mfp, mbcomp1, mbcomp2, flmcomp1, flmcomp2
    msave_state()
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
        mbcgl2.config(text='Add 2 Peaks', bg='white')
        mfitplot()


def pack_fitpar(mresult):
    if len(smresult) > 1:
        o=smresult
        for ii, result in enumerate(mresult):
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
                    if 'x2*xr1+xr2' in i:
                        if xr2>=0:
                            i = i.replace(' == \'x2*xr1+xr2\'', '='+str(xr1)+'*x2+'+str(xr2))
                        else:
                            i = i.replace(' == \'x2*xr1+xr2\'', '='+str(xr1)+'*x2-'+str(-xr2))
                    if "(x2-xr2) / xr1" in i:
                        if xr2>=0:
                            i = i.replace(' == \'(x2-xr2) / xr1\'','=(x2-'+str(xr2) + ')/'+str(xr1))
                        else:
                            i = i.replace(' == \'(x2-xr2) / xr1\'','=(x2+'+str(-xr2) + ')/'+str(xr1))
                    if "(x1-xr2) / xr1" in i:
                        if xr2>=0:
                            i = i.replace(' == \'(x1-xr2) / xr1\'','=(x1-'+str(xr2) + ')/'+str(xr1))
                        else:
                            i = i.replace(' == \'(x1-xr2) / xr1\'','=(x1+'+str(-xr2) + ')/'+str(xr1))
                    if 'w1/wr1*wr2' in i:
                        i = i.replace(' == \'w1/wr1*wr2\'', '=w1/'+str(wr1)+'*'+str(wr2))
                    if 'w2/wr1*wr2' in i:
                        i = i.replace(' == \'w2/wr1*wr2\'', '=w2/'+str(wr1)+'*'+str(wr2))
                        
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
                        o[ii][1]=i
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
                    for i in ['x: nofit','h: nofit','w: nofit','n1: nofit','n2: nofit','n3: nofit']:
                        s.append(i)
            for i in s:
                if 'nofit' in i:
                    o[ii].append(i)
                else:
                    '''preprocess the string to put values in the labels'''
                    if 'x1*xr1+xr2' in i:
                        if xr2>=0:
                            i = i.replace(' == \'x1*xr1+xr2\'', '='+str(xr1)+'*x1+'+str(xr2))
                        else:
                            i = i.replace(' == \'x1*xr1+xr2\'', '='+str(xr1)+'*x1-'+str(-xr2))
                    if 'x2*xr1+xr2' in i:
                        if xr2>=0:
                            i = i.replace(' == \'x2*xr1+xr2\'', '='+str(xr1)+'*x2+'+str(xr2))
                        else:
                            i = i.replace(' == \'x2*xr1+xr2\'', '='+str(xr1)+'*x2-'+str(-xr2))
                    if "(x2-xr2) / xr1" in i:
                        if xr2>=0:
                            i = i.replace(' == \'(x2-xr2) / xr1\'','=(x2-'+str(xr2) + ')/'+str(xr1))
                        else:
                            i = i.replace(' == \'(x2-xr2) / xr1\'','=(x2+'+str(-xr2) + ')/'+str(xr1))
                    if "(x1-xr2) / xr1" in i:
                        if xr2>=0:
                            i = i.replace(' == \'(x1-xr2) / xr1\'','=(x1-'+str(xr2) + ')/'+str(xr1))
                        else:
                            i = i.replace(' == \'(x1-xr2) / xr1\'','=(x1+'+str(-xr2) + ')/'+str(xr1))
                    if 'w1/wr1*wr2' in i:
                        i = i.replace(' == \'w1/wr1*wr2\'', '=w1/'+str(wr1)+'*'+str(wr2))
                    if 'w2/wr1*wr2' in i:
                        i = i.replace(' == \'w2/wr1*wr2\'', '=w2/'+str(wr1)+'*'+str(wr2))
                        
                    '''assign the values to the labels'''
                    if 'x:' in i:
                        o[ii].append(i)
                    if 'h:' in i:
                        o[ii].append(i)
                    if 'w:' in i:
                        o[ii].append(i)
                        o[ii].append('')
                        o[ii].append('')
                        o[ii].append('')
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
    global fmxx, fmyy, fmx, fmy, mvv, maa1, maa2, kmin, kmax, mfi, mfi_err, mfi_x, st, mst, result, fa1, fa2, fit_warn, wr1, wr2, mresult, xr1, xr2, smcst
    if len(mfi) < 1:
        mfi, mfi_err, mfi_x = [], [], []
    else:
        mfi, mfi_err, mfi_x = list(mfi), list(mfi_err), list(mfi_x)
    msave_state()
    pbar = tqdm.tqdm(total=len(ev), desc='Fitting MDC', colour='green')
    for i in range(len(ev)):
        # mbase[i] = int(base.get())  # 待調整
        mbase[i] = 0  # 待調整
        # fmxx[i, :] = fmxx[i, :]/fmxx[i, :]*-50
        # fmyy[i, :] = fmyy[i, :]/fmyy[i, :]*-50
        ecut = data.sel(eV=ev[i], method='nearest')
        if npzf:x = phi
        else:x = (2*m*ev[i]*1.602176634*10**-19)**0.5*np.sin(phi/180*np.pi)*10**-10/(h/2/np.pi)
        y = ecut.to_numpy().reshape(len(x))
        xx, x_arg = filter(x, kmin[i], kmax[i])
        # tx = x[np.argwhere(x >= kmin[i])].flatten()
        # xx = tx[np.argwhere(tx <= kmax[i])].flatten()
        # ty = y[np.argwhere(x >= kmin[i])].flatten()
        # yy = ty[np.argwhere(tx <= kmax[i])].flatten()
        yy = y[x_arg]
        yy = np.where(yy > mbase[i], yy, mbase[i])
        try:
            # if (kmin[i],kmax[i])==(klim.min[i], klim.max[i]) and i not in mfi:
            # if i not in mfi:
            #     if i not in mfi_x:
            #         mfi_x.append(i)
            #     # if i in mfi:
            #     #     mfi.remove(i)
            #     if i in mfi_err:
            #         mfi_err.remove(i)
            #     a1=[(kmin[i]+kmax[i])/2,(np.max(y)-mbase[i]),5,mbase[i]]
            #     a2=[(kmin[i]+kmax[i])/2,(np.max(y)-mbase[i]),5,mbase[i],(kmin[i]+kmax[i])/2,(np.max(y)-mbase[i]),5,mbase[i]]
            # elif (kmin[i],kmax[i])!=(klim.min[i], klim.max[i]):
            if mfp[i] == 1:
                smcst[i] = [0, 0, 0, 0, 0, 0]
                if i in mfi_err and (kmin[i], kmax[i]) != (klim.min[i], klim.max[i]):
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
                    if (kmin[i], kmax[i]) == (klim.min[i], klim.max[i]):
                        fit_warn = 2
                    elif i not in mfi:
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
                        fit_warn = 0
            elif mfp[i] == 2:
                if i in mfi_err and (kmin[i], kmax[i]) != (klim.min[i], klim.max[i]):
                    pars = Parameters()
                    xr1, xr2 = float(mxf1.get()), float(mxf2.get())
                    wr1, wr2 = float(mwf1.get()), float(mwf2.get())
                    fa1, fa2 = float(maf1.get()), float(maf2.get())
                    smcst[i]=[xr1,xr2,wr1,wr2,fa1,fa2]
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
                    a2 = toa2(xx)
                    checkfit()
                    if fit_warn == 1:
                        t = 5
                        while t > 0 and fit_warn == 1:
                            result = fitter.minimize()
                            a2 = toa2(xx)
                            checkfit()
                            t -= 1
                else:
                    if i in mfi:
                        result = mresult[i]
                    a2 = maa2[i, :]
                    if (kmin[i], kmax[i]) == (klim.min[i], klim.max[i]):
                        fit_warn = 2
                    elif i not in mfi:
                        pars = Parameters()
                        xr1, xr2 = float(mxf1.get()), float(mxf2.get())
                        wr1, wr2 = float(mwf1.get()), float(mwf2.get())
                        fa1, fa2 = float(maf1.get()), float(maf2.get())
                        smcst[i]=[xr1,xr2,wr1,wr2,fa1,fa2]
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
                        a2 = toa2(xx)
                        checkfit()
                        if fit_warn == 1:
                            t = 5
                            while t > 0 and fit_warn == 1:
                                result = fitter.minimize()
                                a2 = toa2(xx)
                                checkfit()
                                t -= 1
                    else:
                        fit_warn = 0
            try:
                '''using lmfit'''
                result=swapc1c2()
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
        except IndexError:
            if i not in mfi_err:
                mfi_err.append(i)
            if i in mfi_x:
                mfi_x.remove(i)
            if i in mfi:
                mfi.remove(i)
            a1 = [(kmin[i]+kmax[i])/2, (np.max(y)-mbase[i]), 5, mbase[i]]
            a2 = [(kmin[i]+kmax[i])/2, (np.max(y)-mbase[i]), 5, mbase[i],
                (kmin[i]+kmax[i])/2, (np.max(y)-mbase[i]), 5, mbase[i]]
            smcst=np.zeros(len(ev)*6).reshape(len(ev),6)
        # fmxx[i, :len(xx)] = xx
        # fmyy[i, :len(yy)] = yy
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
    global fmxx, fmyy, fmx, fmy, mvv, maa1, maa2, kmin, kmax, mfi, mfi_err, mfi_x, result, fa1, fa2, fit_warn, wr1, wr2, flmcomp1, flmcomp2, mbcomp1, mbcomp2, mresult, xr1, xr2, smcst
    mbcomp1.config(bg='white')
    mbcomp2.config(bg='white')
    mfi, mfi_err, mfi_x = list(mfi), list(mfi_err), list(mfi_x)
    msave_state()
    i = mfiti.get()
    # mbase[i] = int(base.get())  # 待調整
    mbase[i] = 0  # 待調整
    # fmxx[i, :] = fmxx[i, :]/fmxx[i, :]*-50
    # fmyy[i, :] = fmyy[i, :]/fmyy[i, :]*-50
    ecut = data.sel(eV=ev[i], method='nearest')
    if npzf:x = phi
    else:x = (2*m*ev[i]*1.602176634*10**-19)**0.5*np.sin(phi/180*np.pi)*10**-10/(h/2/np.pi)
    y = ecut.to_numpy().reshape(len(x))
    xx, x_arg = filter(x, kmin[i], kmax[i])
    # tx = x[np.argwhere(x >= kmin[i])].flatten()
    # xx = tx[np.argwhere(tx <= kmax[i])].flatten()
    # ty = y[np.argwhere(x >= kmin[i])].flatten()
    # yy = ty[np.argwhere(tx <= kmax[i])].flatten()
    yy = y[x_arg]
    yy = np.where(yy > mbase[i], yy, mbase[i])
    try:
        if mfp[i] == 1:
            smcst[i] = [0, 0, 0, 0, 0, 0]
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
            smcst[i] = [xr1, xr2, wr1, wr2, fa1, fa2]
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
        result=swapc1c2()
        mresult[i] = result

        if (kmin[i], kmax[i]) == (klim.min[i], klim.max[i]):
            if i not in mfi_x:
                mfi_x.append(i)
            if i in mfi:
                mfi.remove(i)
            if i in mfi_err:
                mfi_err.remove(i)
        elif (kmin[i], kmax[i]) != (klim.min[i], klim.max[i]):
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
    except IndexError:
        if i not in mfi_err:
            mfi_err.append(i)
        if i in mfi_x:
            mfi_x.remove(i)
        if i in mfi:
            mfi.remove(i)
        a1 = [(kmin[i]+kmax[i])/2, (np.max(y)-mbase[i]), 0.5, mbase[i]]
        a2 = [(kmin[i]+kmax[i])/2, (np.max(y)-mbase[i]), 0.5, mbase[i],
              (kmin[i]+kmax[i])/2, (np.max(y)-mbase[i]), 0.5, mbase[i]]
        smcst=np.zeros(len(ev)*6).reshape(len(ev),6)

    # fmxx[i, :len(xx)] = xx
    # fmyy[i, :len(yy)] = yy
    fmx[i, :] = x
    fmy[i, :] = y
    mvv[i] = ev[i]
    if mfp[i] == 1:
        maa1[i, :] = a1
    elif mfp[i] == 2:
        maa2[i, :] = a2

# 初始化撤銷和重做堆疊
mundo_stack = []
mredo_stack = []


def msave_state():
    # 保存當前狀態到撤銷堆疊，並清空重做堆疊
    smresult = pack_fitpar(mresult)
    state = {
        'mfi': mfi.copy(),
        'mfp': mfp.copy(),
        'kmin': kmin.copy(),
        'kmax': kmax.copy(),
        'maa1': maa1.copy(),
        'maa2': maa2.copy(),
        'smresult': smresult.copy(),
        'smcst': smcst.copy(),
        'mfi_err': mfi_err.copy()
    }
    mundo_stack.append(state)
    mredo_stack.clear()


def mundo():
    if mundo_stack:
        global mfi, mfp, kmin, kmax, maa1, maa2, smresult, mresult, smcst, mfi_err, fdo
        # 從撤銷堆疊中彈出上一個狀態並恢復，並將當前狀態推入重做堆疊
        state = mundo_stack.pop()
        smresult = pack_fitpar(mresult)
        mredo_stack.append({
            'mfi': mfi.copy(),
            'mfp': mfp.copy(),
            'kmin': kmin.copy(),
            'kmax': kmax.copy(),
            'maa1': maa1.copy(),
            'maa2': maa2.copy(),
            'smresult': smresult.copy(),
            'smcst': smcst.copy(),
            'mfi_err': mfi_err.copy()
        })
        mfi = state['mfi']
        mfp = state['mfp']
        kmin = state['kmin']
        kmax = state['kmax']
        maa1 = state['maa1']
        maa2 = state['maa2']
        smresult = state['smresult']
        mresult = state['smresult']
        smcst = state['smcst']
        mfi_err = state['mfi_err']
        mst.put("Undo")
        print("Undo")
        fdo=1
        mfitplot()
    else:
        mst.put("No more actions to undo.")
        print("No more actions to undo.")


def mredo():
    if mredo_stack:
        global mfi, mfp, kmin, kmax, maa1, maa2, smresult, mresult, smcst, mfi_err, fdo
        # 從重做堆疊中彈出上一個狀態並恢復，並將當前狀態推入撤銷堆疊
        state = mredo_stack.pop()
        smresult = pack_fitpar(mresult)
        mundo_stack.append({
            'mfi': mfi.copy(),
            'mfp': mfp.copy(),
            'kmin': kmin.copy(),
            'kmax': kmax.copy(),
            'maa1': maa1.copy(),
            'maa2': maa2.copy(),
            'smresult': smresult.copy(),
            'smcst': smcst.copy(),
            'mfi_err': mfi_err.copy()
        })
        mfi = state['mfi']
        mfp = state['mfp']
        kmin = state['kmin']
        kmax = state['kmax']
        maa1 = state['maa1']
        maa2 = state['maa2']
        smresult = state['smresult']
        mresult = state['smresult']
        smcst = state['smcst']
        mfi_err = state['mfi_err']
        mst.put("Redo")
        print("Redo")
        fdo=1
        mfitplot()
    else:
        mst.put("No more actions to redo.")
        print("No more actions to redo.")



def fmrmv():
    global mbrmv, flmrmv, mirmv, kmin, kmax, mfi, mfi_err, mfi_x, cki, mfp, mresult, smresult, smcst
    msave_state()
    i = mfiti.get()
    flmrmv *= -1
    if flmrmv == 1:
        mirmv = i
        mbrmv.config(text='End Remove', bg='red')
    else:
        ti = sorted([i, mirmv])
        for i in np.linspace(ti[0], ti[1], ti[1]-ti[0]+1, dtype=int):
            mfp[i] = 1
            kmin[i], kmax[i] = klim.min[i], klim.max[i]
            if i not in mfi_x:
                mfi_x.append(i)
            if i in mfi:
                mfi.remove(i)
            if i in mfi_err:
                mfi_err.remove(i)
            if i in cki:
                cki.remove(i)
            mresult[i] = []
            try:
                for j in range(6):
                    smresult[i][j] = 'nofit'
                    smcst[i][j] = 0
            except:
                pass
        mplfi()
        mbrmv.config(text='Remove', bg='white')
        mfitplot()



def fmedmove(event):
    global medxdata, medydata, medfitout
    if event.xdata != None:
        medfitout.get_tk_widget().config(cursor="crosshair")
        medxdata.config(text='xdata:'+str(' %.3f' % event.xdata))
        medydata.config(text='ydata:'+str(' %.3f' % event.ydata))
    else:
        medfitout.get_tk_widget().config(cursor="")
        try:
            medxdata.config(text='xdata:')
            medydata.config(text='ydata:')
        except NameError:
            pass



def savemfit():
    global smresult, smcst, fev, fwhm, pos, skmin, skmax, smaa1, smaa2, smfp, smfi, mdet
    smresult = pack_fitpar(mresult)
    path = fd.asksaveasfilename(title="Save MDC Fitted Data", initialdir=dpath,
                                initialfile=name+"_mfit", filetype=[("NPZ files", ".npz"),], defaultextension=".npz")
    try:
        mgg.focus_force()
    except:
        pass
    if len(path) > 2:
        mendg.destroy()
        shape=data.shape
        mdet=data.data[shape[0]//2, shape[1]//2]
        np.savez(path, path=dpath, fev=fev, fwhm=fwhm, pos=pos, skmin=skmin,
                 skmax=skmax, smaa1=smaa1, smaa2=smaa2, smfp=smfp, smfi=smfi, smresult=smresult, smcst=smcst, mdet=mdet)
    else:
        mendg.focus_force()



def fmresidual():
    plt.figure()
    s3,s4=[],[]
    for i in range(len(ev)):
        if i in mfi_err or i in mfi:
            # x = fmxx[i, np.argwhere(fmxx[i, :] >= -20)].flatten()
            # y = fmyy[i, np.argwhere(fmxx[i, :] >= -20)].flatten()
            # lbg=lnr_bg(fmyy[i, :len(x)])
            x, x_arg = filter(fmx[i, :], kmin[i], kmax[i])
            y = fmy[i, x_arg]
            lbg = lnr_bg(y)
            s3.append(np.std(gl2(x, *maa2[i, :])+lbg-y))  # STD
            s4.append(np.sqrt(np.mean((gl2(x, *maa2[i, :])+lbg-y)**2)))  # RMS
        else:
            s3.append(0)
            s4.append(0)
    plt.plot(ev,s3,label='STD',c='r')
    plt.plot(ev,s4,label='RMS',c='b')
    plt.title('Residual')
    plt.xlabel('Kinetic Energy (eV)')
    plt.ylabel('Intensity (Counts)')
    plt.legend()
    plt.show()
    

def fmarea():
    plt.figure()
    s1,s2=[],[]
    for i in range(len(ev)):
        if i in mfi_err or i in mfi:
            # x = fmxx[i, np.argwhere(fmxx[i, :] >= -20)].flatten()
            # y = fmyy[i, np.argwhere(fmxx[i, :] >= -20)].flatten()
            x, x_arg = filter(fmx[i, :], kmin[i], kmax[i])
            ty = gl1(x, *maa2[i, :4])
            s1.append(np.sum(np.array([((ty[i]+ty[i+1])/2)for i in range(len(x)-1)])
                        # Area 1
                        * np.array(([(x[i+1]-x[i])for i in range(len(x)-1)]))))
            ty = gl1(x, *maa2[i, -4:])
            s2.append(np.sum(np.array([((ty[i]+ty[i+1])/2)for i in range(len(x)-1)])
                        # Area 2
                        * np.array(([(x[i+1]-x[i])for i in range(len(x)-1)]))))
        else:
            s1.append(0)
            s2.append(0)
    plt.plot(ev,s1,label='Area 1',c='r')
    plt.plot(ev,s2,label='Area 2',c='b')
    plt.title('Area')
    plt.xlabel('Kinetic Energy (eV)')
    plt.ylabel('Intensity (Counts)')
    plt.legend()
    plt.show()
    

def fmfwhm():
    global pos, fwhm, fev, rpos, ophi
    fev, pos, fwhm = [], [], []
    f=plt.figure()
    a1=f.add_subplot(311)
    a2=f.add_subplot(312)
    a3=f.add_subplot(313)
    x1=[]
    x2=[]
    y1=[]
    y2=[]
    for i, v in enumerate(mfi):
        if mfp[v] == 1:
            fev.append(ev[v])
            pos.append(maa1[v, 0])
            fwhm.append(maa1[v, 2])
            x1.append(ev[v])
            y1.append(maa1[v, 2])
        elif mfp[v] == 2:
            x1.append(ev[v])
            x2.append(ev[v])
            y1.append(maa2[v, 2])
            y2.append(maa2[v, 6])
            
            fev.append(ev[v])
            fev.append(ev[v])
            pos.append(maa2[v, 0])
            pos.append(maa2[v, 4])
            fwhm.append(maa2[v, 2])
            fwhm.append(maa2[v, 6])
    fev = np.float64(fev)
    rpos = np.float64(pos)
    
    ophi = np.arcsin(rpos/(2*m*fev*1.602176634*10**-19)**0.5/10**-10*(h/2/np.pi))*180/np.pi
    pos = (2*m*fev*1.602176634*10**-19)**0.5 * np.sin((np.float64(k_offset.get())+ophi)/180*np.pi)*10**-10/(h/2/np.pi)

    rpos = res(fev, rpos)
    ophi = res(fev, ophi)
    fwhm = res(fev, fwhm)
    pos = res(fev, pos)
    fev = res(fev, fev)
    
    ha=a1.scatter(x1,y1,c='r')
    a1.set_title('FWHM')
    a1.set_ylabel(r'FWHM ($\frac{2\pi}{\AA}$)')
    a1.legend([ha],['Comp 1'])
    hb=a2.scatter(x2,y2,c='b')
    a2.set_ylabel(r'FWHM ($\frac{2\pi}{\AA}$)')
    a2.legend([hb],['Comp 2'])
    h2=a3.scatter(x2,y2,c='b')
    h1=a3.scatter(x1,y1,c='r')
    a3.set_xlabel('Kinetic Energy (eV)')
    a3.set_ylabel(r'FWHM ($\frac{2\pi}{\AA}$)')
    a3.legend([h1,h2],['Comp 1','Comp 2'])
    plt.tight_layout()
    plt.show()


def fmimse():
    global pos, fwhm, fev, rpos, ophi
    fev, pos, fwhm = [], [], []
    f=plt.figure()
    a1=f.add_subplot(221)
    a2=f.add_subplot(222)
    a3=f.add_subplot(223)
    a4=f.add_subplot(224)
    y=[]
    pos1=[]
    pos2=[]
    fwhm1=[]
    fwhm2=[]
    for i, v in enumerate(mfi):
        if mfp[v] == 1:
            fev.append(ev[v])
            pos.append(maa1[v, 0])
            fwhm.append(maa1[v, 2])
        elif mfp[v] == 2:
            y.append(ev[v])
            pos1.append(maa2[v, 0])
            pos2.append(maa2[v, 4])
            fwhm1.append(maa2[v, 2])
            fwhm2.append(maa2[v, 6])
            
            fev.append(ev[v])
            fev.append(ev[v])
            pos.append(maa2[v, 0])
            pos.append(maa2[v, 4])
            fwhm.append(maa2[v, 2])
            fwhm.append(maa2[v, 6])
    y = np.float64(y)
    fev = np.float64(fev)
    rpos = np.float64(pos)
    
    ophi = np.arcsin(rpos/np.sqrt(2*m*fev*1.602176634*10**-19)/10**-10*(h/2/np.pi))*180/np.pi
    pos = (2*m*fev*1.602176634*10**-19)**0.5 * np.sin((np.float64(k_offset.get())+ophi)/180*np.pi)*10**-10/(h/2/np.pi)
    
    rpos = res(fev, rpos)
    ophi = res(fev, ophi)
    fwhm = res(fev, fwhm)
    pos = res(fev, pos)
    fev = res(fev, fev)
    
    pos1 = res(y, pos1)
    pos2 = res(y, pos2)
    fwhm1 = res(y, fwhm1)
    fwhm2 = res(y, fwhm2)
    y = res(y, y)
    
    xx = np.diff(y)
    yy1 = np.diff(pos1)
    yy2 = np.diff(pos2)
    
    # eliminate infinite vf
    for i in range(len(yy1)):
        if xx[i]/yy1[i] > 20000:
            yy1[i] = 0
    for i in range(len(yy2)):
        if xx[i]/yy2[i] > 20000:
            yy2[i] = 0
    
    v1 = xx/yy1
    v2 = xx/yy2
    yy1 = v1*fwhm1[1::]/2
    yy2 = v2*fwhm2[1::]/2
    xx/=2
    print(len(y))
    print(len(xx))
    x = ((y[-1:0:-1]+xx[::-1])-vfe)*1000
    print(len(x))
    ha=a1.scatter(x,v1,c='r')
    hb=a2.scatter(x,v2,c='b')
    h1=a3.scatter(x,yy1*1000,c='r')
    h2=a4.scatter(x,yy2*1000,c='b')
    a1.set_title('Group Velocity')
    a1.set_xlabel('Binding Energy (meV)', font='Arial', fontsize=_size(14))
    a1.set_ylabel(r'v ($eV\AA$)', font='Arial', fontsize=_size(14))
    a1.legend([ha],['Comp 1'])
    a2.set_title('Group Velocity')
    a2.set_xlabel('Binding Energy (meV)', font='Arial', fontsize=_size(14))
    a2.set_ylabel(r'v ($eV\AA$)', font='Arial', fontsize=_size(14))
    a2.legend([hb],['Comp 2'])
    a3.set_title('Imaginary Part')
    a3.set_xlabel('Binding Energy (meV)', font='Arial', fontsize=_size(14))
    a3.set_ylabel(r'Im $\Sigma$ (meV)', font='Arial', fontsize=_size(14))
    a3.legend([h1],['Comp 1'])
    a4.set_title('Imaginary Part')
    a4.set_xlabel('Binding Energy (meV)', font='Arial', fontsize=_size(14))
    a4.set_ylabel(r'Im $\Sigma$ (meV)', font='Arial', fontsize=_size(14))
    a4.legend([h2],['Comp 2'])
    plt.tight_layout()
    plt.show()
    

def fmpreview():
    mprvg = tk.Toplevel(g)
    mprvg.geometry('300x320')
    mprvg.title(' Preview MDC Result')
    bmresidual = tk.Button(mprvg, text='Residual', command=fmresidual, width=30, height=2, font=('Arial', _size(16), "bold"), bg='white', bd=10)
    bmresidual.pack()
    bmarea = tk.Button(mprvg, text='Area', command=fmarea, width=30, height=2, font=('Arial', _size(16), "bold"), bg='white', bd=10)
    bmarea.pack()
    bmfwhm = tk.Button(mprvg, text='FWHM', command=fmfwhm, width=30, height=2, font=('Arial', _size(16), "bold"), bg='white', bd=10)
    bmfwhm.pack()
    bmimse = tk.Button(mprvg, text='Imaginary Part', command=fmimse, width=30, height=2, font=('Arial', _size(16), "bold"), bg='white', bd=10)
    bmimse.pack()
    mprvg.update()
    w=mprvg.winfo_reqwidth()
    h=mprvg.winfo_reqheight()
    mprvg.geometry(f'{w}x{h}')
    mprvg.update()
    
scki = []


def mprend(p=0):
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
            if p == 1:
                fev.append(ev[v])
                pos.append(maa2[v, 0])
                fwhm.append(maa2[v, 2])
            elif p == 2:
                fev.append(ev[v])
                pos.append(maa2[v, 4])
                fwhm.append(maa2[v, 6])
            else:
                fev.append(ev[v])
                fev.append(ev[v])
                pos.append(maa2[v, 0])
                pos.append(maa2[v, 4])
                fwhm.append(maa2[v, 2])
                fwhm.append(maa2[v, 6])
            
    fwhm = res(fev, fwhm)
    pos = res(fev, pos)
    # skmin = res(smfi, skmin)
    # skmax = res(smfi, skmax)
    # smfp = res(smfi, smfp)
    fev = res(fev, fev)
    smfi = res(smfi, smfi)
            
    rpos, fev, pos, fwhm = np.float64(pos), np.float64(
        fev), np.float64(pos), np.float64(fwhm)


def fmend():
    global rpos, pos, fwhm, fev, medxdata, medydata, medfitout, skmin, skmax, smaa1, smaa2, smfp, smfi, fpr, scki, mendg
    mprend()
    scki = cki
    fpr = 1
    if 'mendg' in globals():
        mendg.destroy()
    mendg = tk.Toplevel(g)
    mendg.title('MDC Lorentz Fit Result')
    fr = tk.Frame(master=mendg, bd=5)
    fr.grid(row=0, column=0)
    mfitfig = Figure(figsize=(8*scale, 6*scale), layout='constrained')
    medfitout = FigureCanvasTkAgg(mfitfig, master=fr)
    medfitout.get_tk_widget().grid(row=0, column=0)
    medfitout.mpl_connect('motion_notify_event', fmedmove)

    a = mfitfig.subplots()
    a.scatter(pos+fwhm/2, fev, c='r', s=scale*scale*10)
    a.scatter(pos-fwhm/2, fev, c='r', s=scale*scale*10)
    a.scatter(pos, fev, c='k', s=scale*scale*10)
    a.set_xlabel(r'k ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=_size(14))
    a.set_ylabel('Kinetic Energy (eV)', font='Arial', fontsize=_size(14))
    medfitout.draw()
    xydata = tk.Frame(master=fr, bd=5)
    xydata.grid(row=1, column=0)

    medxdata = tk.Label(xydata, text='xdata:', font=(
        "Arial", _size(12), "bold"), width='15', height='1', bd=10, bg='white')
    medxdata.grid(row=0, column=0)
    medydata = tk.Label(xydata, text='ydata:', font=(
        "Arial", _size(12), "bold"), width='15', height='1', bd=10, bg='white')
    medydata.grid(row=0, column=1)

    bsave = tk.Button(master=mendg, text='Save Fitted Data', command=savemfit,
                      width=30, height=2, font=('Arial', _size(14), "bold"), bg='white', bd=10)
    bsave.grid(row=1, column=0)
    
    mendg.update()



def fmend1():
    global rpos, pos, fwhm, fev, medxdata, medydata, medfitout, skmin, skmax, smaa1, smaa2, smfp, smfi, fpr, scki, mendg
    mprend(p=1)
    scki = cki
    fpr = 1
    if 'mendg' in globals():
        mendg.destroy()
    mendg = tk.Toplevel(g)
    mendg.title('MDC Lorentz Fit Result')
    fr = tk.Frame(master=mendg, bd=5)
    fr.grid(row=0, column=0)
    mfitfig = Figure(figsize=(8*scale, 6*scale), layout='constrained')
    medfitout = FigureCanvasTkAgg(mfitfig, master=fr)
    medfitout.get_tk_widget().grid(row=0, column=0)
    medfitout.mpl_connect('motion_notify_event', fmedmove)

    a = mfitfig.subplots()
    a.scatter(pos+fwhm/2, fev, c='r', s=scale*scale*10)
    a.scatter(pos-fwhm/2, fev, c='r', s=scale*scale*10)
    a.scatter(pos, fev, c='k', s=scale*scale*10)
    a.set_xlabel(r'k ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=_size(14))
    a.set_ylabel('Kinetic Energy (eV)', font='Arial', fontsize=_size(14))
    medfitout.draw()
    xydata = tk.Frame(master=fr, bd=5)
    xydata.grid(row=1, column=0)

    medxdata = tk.Label(xydata, text='xdata:', font=(
        "Arial", _size(12), "bold"), width='15', height='1', bd=10, bg='white')
    medxdata.grid(row=0, column=0)
    medydata = tk.Label(xydata, text='ydata:', font=(
        "Arial", _size(12), "bold"), width='15', height='1', bd=10, bg='white')
    medydata.grid(row=0, column=1)

    bsave = tk.Button(master=mendg, text='Save Fitted Data', command=savemfit,
                      width=30, height=2, font=('Arial', _size(14), "bold"), bg='white', bd=10)
    bsave.grid(row=1, column=0)
    
    mendg.update()


def fmend2():
    global rpos, pos, fwhm, fev, medxdata, medydata, medfitout, skmin, skmax, smaa1, smaa2, smfp, smfi, fpr, scki, mendg
    mprend(p=2)
    scki = cki
    fpr = 1
    if 'mendg' in globals():
        mendg.destroy()
    mendg = tk.Toplevel(g)
    mendg.title('MDC Lorentz Fit Result')
    fr = tk.Frame(master=mendg, bd=5)
    fr.grid(row=0, column=0)
    mfitfig = Figure(figsize=(8*scale, 6*scale), layout='constrained')
    medfitout = FigureCanvasTkAgg(mfitfig, master=fr)
    medfitout.get_tk_widget().grid(row=0, column=0)
    medfitout.mpl_connect('motion_notify_event', fmedmove)

    a = mfitfig.subplots()
    a.scatter(pos+fwhm/2, fev, c='r', s=scale*scale*10)
    a.scatter(pos-fwhm/2, fev, c='r', s=scale*scale*10)
    a.scatter(pos, fev, c='k', s=scale*scale*10)
    a.set_xlabel(r'k ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=_size(14))
    a.set_ylabel('Kinetic Energy (eV)', font='Arial', fontsize=_size(14))
    medfitout.draw()
    xydata = tk.Frame(master=fr, bd=5)
    xydata.grid(row=1, column=0)

    medxdata = tk.Label(xydata, text='xdata:', font=(
        "Arial", _size(12), "bold"), width='15', height='1', bd=10, bg='white')
    medxdata.grid(row=0, column=0)
    medydata = tk.Label(xydata, text='ydata:', font=(
        "Arial", _size(12), "bold"), width='15', height='1', bd=10, bg='white')
    medydata.grid(row=0, column=1)

    bsave = tk.Button(master=mendg, text='Save Fitted Data', command=savemfit,
                      width=30, height=2, font=('Arial', _size(14), "bold"), bg='white', bd=10)
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
    global mfitout, mdxdata, mdydata, mbcomp1, mbcomp2, mbgv, flmcomp1, flmcomp2
    i = mfiti.get()
    mbgv = 0
    try:
        flmcomp1,flmcomp2 = -1, -1
        mfitout.get_tk_widget().delete('rec')
        mdxdata.config(text='dx:')
        mdydata.config(text='dy:')
        if mfp[i] == 2:
            mbcomp1.config(state='active', bg='white')
            mbcomp2.config(state='active', bg='white')
        else:
            mbcomp1.config(state='disabled', bg='white')
            mbcomp2.config(state='disabled', bg='white')
    except:
        pass
    mfitplot()
    mprplot(mxl)



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
    mprplot(mxl)
    miout.draw()


def mfbgu(event):
    global mbgv
    i=mfiti.get()
    # mbase[i] = int(base.get())  # 待調整
    mbase[i] = 0  # 待調整
    # fmxx[i, :] = fmxx[i, :]/fmxx[i, :]*-50
    # fmyy[i, :] = fmyy[i, :]/fmyy[i, :]*-50
    ecut = data.sel(eV=ev[i], method='nearest')
    if npzf:x = phi
    else:x = (2*m*ev[i]*1.602176634*10**-19)**0.5*np.sin(phi/180*np.pi)*10**-10/(h/2/np.pi)
    y = ecut.to_numpy().reshape(len(x))
    xx, x_arg = filter(x, kmin[i], kmax[i])
    # tx = x[np.argwhere(x >= kmin[i])].flatten()
    # xx = tx[np.argwhere(tx <= kmax[i])].flatten()
    # ty = y[np.argwhere(x >= kmin[i])].flatten()
    # yy = ty[np.argwhere(tx <= kmax[i])].flatten()
    yy = y[x_arg]
    yy = np.where(yy > mbase[i], yy, mbase[i])
    d = sorted(abs(np.diff(np.append(yy[0:5],yy[-6:-1]))))
    t=0
    ti=0
    while t==0:
        t=d[ti]
        ti+=1
        if ti==len(d):
            break
    print(t)
    try:
        mbgv+=t/2
        mfit()
        mfitplot()
    except:
        pass


def mfbgd(event):
    global mbgv
    i=mfiti.get()
    # mbase[i] = int(base.get())  # 待調整
    mbase[i] = 0  # 待調整
    # fmxx[i, :] = fmxx[i, :]/fmxx[i, :]*-50
    # fmyy[i, :] = fmyy[i, :]/fmyy[i, :]*-50
    ecut = data.sel(eV=ev[i], method='nearest')
    if npzf:x = phi
    else:x = (2*m*ev[i]*1.602176634*10**-19)**0.5*np.sin(phi/180*np.pi)*10**-10/(h/2/np.pi)
    y = ecut.to_numpy().reshape(len(x))
    xx, x_arg = filter(x, kmin[i], kmax[i])
    # tx = x[np.argwhere(x >= kmin[i])].flatten()
    # xx = tx[np.argwhere(tx <= kmax[i])].flatten()
    # ty = y[np.argwhere(x >= kmin[i])].flatten()
    # yy = ty[np.argwhere(tx <= kmax[i])].flatten()
    yy = y[x_arg]
    yy = np.where(yy > mbase[i], yy, mbase[i])
    d = sorted(abs(np.diff(np.append(yy[0:5],yy[-6:-1]))))
    t=0
    ti=0
    while t==0:
        t=d[ti]
        ti+=1
        if ti==len(d):
            break
    print(t)
    try:
        mbgv-=t/2
        mfit()
        mfitplot()
    except:
        pass


def _mpr2draw():
    global mfitprfig2, mfitprout2, mfprb
    i = mfiti.get()
    try:
        mfitprfig2.clear()
        mfprb = mfitprfig2.subplots()
        mfprb.set_xlabel(r'k ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=_size(12))
        if emf=='KE':
            mfprb.set_ylabel('Kinetic Energy (eV)', font='Arial', fontsize=_size(12))
        else:
            mfprb.set_ylabel('Binding Energy (eV)', font='Arial', fontsize=_size(12))
            mfprb.invert_yaxis()
        mprend()
        if emf == 'KE':
            mfprb.scatter(pos + fwhm / 2, fev, c='r', s=scale*scale*0.5)
            mfprb.scatter(pos - fwhm / 2, fev, c='r', s=scale*scale*0.5)
            mfprb.scatter(pos, fev, c='k', s=scale*scale*0.5)
        else:
            mfprb.scatter(pos + fwhm / 2, vfe - fev, c='r', s=scale*scale*0.5)
            mfprb.scatter(pos - fwhm / 2, vfe - fev, c='r', s=scale*scale*0.5)
            mfprb.scatter(pos, vfe - fev, c='k', s=scale*scale*0.5)

        if emf == 'KE':
            mfprb.plot(mfprb.get_xlim(), [ev[i], ev[i]], 'b-', alpha=0.5)
        else:
            mfprb.plot(mfprb.get_xlim(), [vfe - ev[i], vfe - ev[i]], 'b-', alpha=0.5)
        
        mfitprout2.draw()
    except:
        pass
    

def mpr2draw():
    t = threading.Thread(target=_mpr2draw)
    t.daemon = True
    t.start()
    

def _mpr3draw():
    global mfitprfig3, mfitprout3, mfprc
    i = mfiti.get()
    try:
        mfitprfig3.clear()
        mfprc = mfitprfig3.subplots(2, 1)
        mfprc[1].set_xlabel('Binding Energy (eV)')
        mfprc[0].set_ylabel(r'FWHM ($\frac{2\pi}{\AA}$)')
        mfprc[1].set_ylabel(r'FWHM ($\frac{2\pi}{\AA}$)')
        mfprc[0].set_xticks([])
        mfprc[0].invert_xaxis()
        mfprc[1].invert_xaxis()
        x1=[]
        x2=[]
        y1=[]
        y2=[]
        for j, v in enumerate(mfi):
            if mfp[v] == 1:
                x1.append(vfe-ev[v])
                y1.append(maa1[v, 2])
            elif mfp[v] == 2:
                x1.append(vfe-ev[v])
                x2.append(vfe-ev[v])
                y1.append(maa2[v, 2])
                y2.append(maa2[v, 6])
        y1 = res(x1, y1)
        y2 = res(x2, y2)
        x1 = res(x1, x1)
        x2 = res(x2, x2)
        mfprc[0].plot(x1, y1, c='r', marker='o', markersize=scale*0.5, label='Comp 1')    #plot
        mfprc[1].plot(x2, y2, c='b', marker='o', markersize=scale*0.5, label='Comp 2')    #plot
        # mfprc[0].scatter(x1, y1, c='r', s=scale*scale*0.5, label='Comp 1')    #scatter
        # mfprc[1].scatter(x2, y2, c='b', s=scale*scale*0.5, label='Comp 2')    #scatter
        l1 = mfprc[0].legend()
        l2 = mfprc[1].legend()
        l1.draw_frame(False)
        l2.draw_frame(False)
        mfprc[0].plot([vfe - ev[i], vfe - ev[i]], mfprc[0].get_ylim(), 'b-', alpha=0.5)
        mfprc[1].plot([vfe - ev[i], vfe - ev[i]], mfprc[1].get_ylim(), 'r-', alpha=0.5)
        mfitprout3.draw()
    except:
        pass
    

def mpr3draw():
    t = threading.Thread(target=_mpr3draw)
    t.daemon = True
    t.start()


def _mprplot_job1():
    global mfitprfig1, mfitprout1, mfpra, mfprl1, mfprl2, mfprl3, mfpr
    i = mfiti.get()
    try:
        xl=mprxl
        if mfpr == 0:
            mfpr = 1
            mfitprfig1.clear()
            mfpra = mfitprfig1.subplots()
            mprend()
            if emf == 'KE':
                px, py = np.meshgrid(phi, ev)
                tev = py.copy()
                mfpra.set_xlabel(r'k ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=_size(8))
                mfpra.set_ylabel('Kinetic Energy (eV)', font='Arial', fontsize=_size(12))
            else:
                px, py = np.meshgrid(phi, vfe - ev)
                tev = vfe - py.copy()
                mfpra.set_xlabel(r'k ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=_size(8))
                mfpra.set_ylabel('Binding Energy (eV)', font='Arial', fontsize=_size(12))
                mfpra.invert_yaxis()
            if npzf:
                px = phi
            else:
                px = (2 * m * tev * 1.6 * 10 ** -19) ** 0.5 * np.sin(px / 180 * np.pi) * 10 ** -10 / (h / 2 / np.pi)
            pz = data.to_numpy()
            mfpra.pcolormesh(px, py, pz, cmap=value3.get())
            pz=None
            oyl = mfpra.get_ylim()

            if emf == 'KE':
                mfprl1,=mfpra.plot([xl[0], xl[1]], [ev[i], ev[i]], 'r-')
            else:
                mfprl1,=mfpra.plot([xl[0], xl[1]], [vfe - ev[i], vfe - ev[i]], 'r-')

            de = (ev[1] - ev[0]) * 8
            mfprl2,=mfpra.plot([xl[0], xl[0]], [ev[i] - de, ev[i] + de], 'r-')
            mfprl3,=mfpra.plot([xl[1], xl[1]], [ev[i] - de, ev[i] + de], 'r-')
            mfpra.set_ylim(oyl)
        else:
            mprend()
            de = (ev[1] - ev[0]) * 8
            if emf == 'KE':
                mfpra.set_xlabel(r'k ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=_size(8))
                mfpra.set_ylabel('Kinetic Energy (eV)', font='Arial', fontsize=_size(12))
                mfprl1.set_xdata([xl[0], xl[1]])
                mfprl1.set_ydata([ev[i], ev[i]])
                mfprl2.set_ydata([ev[i] - de, ev[i] + de])
                mfprl3.set_ydata([ev[i] - de, ev[i] + de])
                # mfprl1,=mfpra.plot([xl[0], xl[1]], [ev[i], ev[i]], 'r-')
            else:
                mfpra.set_xlabel(r'k ($\frac{2\pi}{\AA}$)', font='Arial', fontsize=_size(8))
                mfpra.set_ylabel('Binding Energy (eV)', font='Arial', fontsize=_size(12))
                mfprl1.set_xdata([xl[0], xl[1]])
                mfprl1.set_ydata([vfe - ev[i], vfe - ev[i]])
                mfprl2.set_ydata([vfe - ev[i] - de, vfe - ev[i] + de])
                mfprl3.set_ydata([vfe - ev[i] - de, vfe - ev[i] + de])
                # mfprl1,=mfpra.plot([xl[0], xl[1]], [vfe - ev[i], vfe - ev[i]], 'r-')
            mfprl2.set_xdata([xl[0], xl[0]])
            mfprl3.set_xdata([xl[1], xl[1]])
            # mfprl2,=mfpra.plot([xl[0], xl[0]], [ev[i] - de, ev[i] + de], 'r-')
            # mfprl3,=mfpra.plot([xl[1], xl[1]], [ev[i] - de, ev[i] + de], 'r-')
            
        mfitprout1.draw()
    except:
        pass



def mprplot_job1():
    t = threading.Thread(target=_mprplot_job1)
    t.daemon = True
    t.start()


def mprplot(xl):
    global mprxl
    if mpr==1:
        mprxl = xl
        mpr2draw()
        mpr3draw()
        mprplot_job1()


def mprbgjob1():
    while True:
        if mpr==1:
            try:
                mfitprout1.draw()
            except:
                pass

def mprbg1():
    t = threading.Thread(target=mprbgjob1)
    t.daemon = True
    t.start()
    

def mprbgjob2():
    while True:
        if mpr==1:
            try:
                mfitprout2.draw()
            except:
                pass

def mprbg2():
    t = threading.Thread(target=mprbgjob2)
    t.daemon = True
    t.start()
    

def mprbgjob3():
    while True:
        if mpr==1:
            try:
                mfitprout3.draw()
            except:
                pass
            

def mprbg3():
    t = threading.Thread(target=mprbgjob3)
    t.daemon = True
    t.start() 


def f_pr():
    global mfpr, mpr, mfitprfig1, mfitprfig2, mfitprfig3, mfitprout1, mfitprout2, mfitprout3
    mfpr=0
    if mpr==1:
        mpr=0
        b_pr.config(text='Real Time Preview OFF', fg='red')
        mfitprfig1.clear()
        mfitprfig2.clear()
        mfitprfig3.clear()
        mfitprout1.draw()
        mfitprout2.draw()
        mfitprout3.draw()
    else:
        mpr=1
        mprplot(mxl)
        b_pr.config(text='Real Time Preview ON', fg='green')

def mfitplot():  # mfiti Scale
    global mfitax, mxl, myl, klmin, klmax, tmxl, kmin, kmax, maa2, flmcomp, lm1, lm2, lm3, lm4, lm5, lm6, mxf1, mxf2, mwf1, mwf2, maf1, maf2, mt1, mt2, mt3, mt4, mt5, fdo, mf_prswap
    i = mfiti.get()
    mfitfig.clear()
    mfitax = mfitfig.subplots()
    # 'Pos:'+str(round(maa1[i,0],3))+r' $(\frac{2\pi}{\AA})$'+', FWHM:'+str(round(maa1[i,2],3))+r' $(\frac{2\pi}{\AA})$'
    if emf=='KE':
        mfitax.set_title('Kinetic Energy:' + str(round(mvv[i], 3))+' eV, '+str(mfp[i])+' Peak')
    else:
        mfitax.set_title('Binding Energy:' + str(round(vfe-mvv[i], 3))+' eV, '+str(mfp[i])+' Peak')
    mfitax.scatter(fmx[i, :], fmy[i, :], c='k', s=scale*scale*4)
    tyl = mfitax.get_ylim()
    txl = mfitax.get_xlim()
    dy = (tyl[1]-tyl[0])/20
    dx = (txl[1]-txl[0])/50
    tymin = tyl[0]
    tymax = tyl[1]
    txmin = txl[0]
    txmax = txl[1]
    mfitax.axhline(tymax+dy, c='grey')
    # x = fmxx[i, np.argwhere(fmxx[i, :] >= -20)].flatten()
    # y = fmyy[i, np.argwhere(fmxx[i, :] >= -20)].flatten()
    x, x_arg = filter(fmx[i, :], kmin[i], kmax[i])
    y = fmy[i, x_arg]
    lbg = lnr_bg(y)
    if i in mfi_x:
        for l, v in zip([lm1, lm2, lm3, lm4, lm5, lm6], ['', '', '', '', '', '']):
            l.config(text=v)
        try:
            mxf1.set(str(smcst[i][0]))
            mxf2.set(str(smcst[i][1]))
            mwf1.set(str(smcst[i][2]))
            mwf2.set(str(smcst[i][3]))
            maf1.set(str(smcst[i][4]))
            maf2.set(str(smcst[i][5]))
        except:
            pass
    if mfp[i] == 1:
        try:
            mxf1.set(str(smcst[i][0]))
            mxf2.set(str(smcst[i][1]))
            mwf1.set(str(smcst[i][2]))
            mwf2.set(str(smcst[i][3]))
            maf1.set(str(smcst[i][4]))
            maf2.set(str(smcst[i][5]))
        except:
            pass
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
            mt1=mfitax.text(txmin+dx, tymax-dy, 'Residual STD: '+str(round(s, 2)), fontsize=_size(12))
            s = np.sqrt(np.mean((gl1(x, *maa1[i, :])+lbg-y)**2))  # RMS
            mt2=mfitax.text(txmin+dx, tymax-2*dy,
                        'Residual RMS: '+str(round(s, 2)), fontsize=_size(12))
            ty = gl1(x, *maa1[i, :])
            s = np.sum(np.array([((ty[i]+ty[i+1])/2)for i in range(len(x)-1)])
                    # Area
                    * np.array(([(x[i+1]-x[i])for i in range(len(x)-1)])))
            mt3=mfitax.text(txmin+dx, tymax-3*dy, 'Area: '+str(round(s, 2)), fontsize=_size(12))
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
                vv = smresult[i]
                for l, v in zip([lm1, lm2, lm3, lm4, lm5, lm6], vv):
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
            mfitax.fill_between(x, lbg, gl2_2, facecolor='purple', alpha=0.5)
        if i in mfi_err or i in mfi:
            if i in mfi:
                mfitax.plot(x, gl2(x, *maa2[i, :]) +
                            lbg-y+tymax+dy, color='gray', lw=1)
            else:
                mfitax.plot(x, gl2(x, *maa2[i, :]) +
                            lbg-y+tymax+dy, color='red', lw=1)
            # s=(np.sum((gl2(x,*maa2[i,:])+lbg-y)**2)/(max(x)-min(x)))**0.5
            s = np.std(gl2(x, *maa2[i, :])+lbg-y)  # STD
            mt1=mfitax.text(txmin+dx, tymax-dy, 'Residual STD: '+str(round(s, 2)), fontsize=_size(12))
            s = np.sqrt(np.mean((gl2(x, *maa2[i, :])+lbg-y)**2))  # RMS
            mt2=mfitax.text(txmin+dx, tymax-2*dy,
                        'Residual RMS: '+str(round(s, 2)), fontsize=_size(12))
            ty = gl1(x, *maa2[i, :4])
            s = np.sum(np.array([((ty[i]+ty[i+1])/2)for i in range(len(x)-1)])
                    # Area 1
                    * np.array(([(x[i+1]-x[i])for i in range(len(x)-1)])))
            mt3=mfitax.text(txmin+dx, tymax-3*dy, 'Area 1: '+str(round(s, 2)), fontsize=_size(12))
            ty = gl1(x, *maa2[i, -4:])
            s = np.sum(np.array([((ty[i]+ty[i+1])/2)for i in range(len(x)-1)])
                    # Area 2
                    * np.array(([(x[i+1]-x[i])for i in range(len(x)-1)])))
            mt4=mfitax.text(txmin+dx, tymax-4*dy, 'Area 2: '+str(round(s, 2)), fontsize=_size(12))
            try:
                if smcst[i][4] != 0 and smcst[i][5] != 0:
                    mt5=mfitax.text(txmin+dx, tymax-5*dy, 'A1:A2='+str(smcst[i][4])+':'+str(smcst[i][5]), fontsize=_size(12))
                mxf1.set(str(smcst[i][0]))
                mxf2.set(str(smcst[i][1]))
                mwf1.set(str(smcst[i][2]))
                mwf2.set(str(smcst[i][3]))
                maf1.set(str(smcst[i][4]))
                maf2.set(str(smcst[i][5]))
            except:
                pass
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
                vv = smresult[i]
                for l, v in zip([lm1, lm2, lm3, lm4, lm5, lm6], vv):
                    if 'nofit' not in v:
                        l.config(text=v)
                        l.config(anchor='w')
            except:
                pass
            try:
                if fdo==0 or i not in mf_prswap:
                    fitpar2(mresult[i], lm1, lm2, lm3, lm4, lm5, lm6)
                else:
                    mresult[i]=smresult[i]
                    fdo=0
                    try:
                        if mf_prswap:
                            mf_prswap.remove(i)
                    except:
                        pass
            except:
                pass
    # mfitax.plot(fmxx[i, np.argwhere(fmxx[i, :] >= -20)], lbg, 'g--')
    mfitax.plot(x, lbg, 'g--')
    # if bg_warn==1:  #shirley base line warn
    #     mfitax.plot(fmxx[i,np.argwhere(fmxx[i,:]>=-20)],lbg,'r--')
    # else:
    #     mfitax.plot(fmxx[i,np.argwhere(fmxx[i,:]>=-20)],lbg,'g--')

    # mfitax.scatter(fmxx[i, np.argwhere(fmxx[i, :] >= -20)], y, c='g', s=scale*scale*4)
    mfitax.scatter(x, y, c='g', s=scale*scale*4)
    if (kmin[i], kmax[i]) != (klim.min[i], klim.max[i]):
        klmin = mfitax.axvline(kmin[i], c='r')
        klmax = mfitax.axvline(kmax[i], c='r')
    else:
        klmin = mfitax.axvline(kmin[i], c='grey')
        klmax = mfitax.axvline(kmax[i], c='grey')
        fl.set_alpha(0.3)
    mfitax.set_xlabel(r'k ($\frac{2\pi}{\AA}$)', fontsize=_size(14))
    mfitax.set_ylabel('Intensity (Counts)', fontsize=_size(14))
    mfitax.set_xticklabels(np.round(mfitax.get_xticks(),2), fontsize=_size(12))
    mfitax.set_yticklabels(np.round(mfitax.get_yticks(),2), fontsize=_size(12))
    mxl = mfitax.get_xlim()
    myl = mfitax.get_ylim()
    tmxl = np.copy(mxl)
    mfitout.draw()
    x, x_arg, y, lbg, vv, ty, fl, txl, tyl, dx, dy = None, None, None, None, None, None, None, None, None, None, None
    mplfi()



def mmove(event):
    global mxdata, mydata, mdxdata, mdydata, x2, y2, mfitax, mfitout, klmin, klmax, kmin, kmax, tpx1, tpx2, tpy1, tpy2, tx2, ty2, mcpx1, mcpy1, mcpx2, mcpy2
    if event.xdata != None:
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
                mdxdata.config(text='dx:'+str(' %.3f' % abs(x2-x1)))
                mdydata.config(text='dy:'+str(' %.3f' % abs(y2-y1)))
        mxdata.config(text='xdata:'+str(' %.3f' % event.xdata))
        mydata.config(text='ydata:'+str(' %.3f' % event.ydata))
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
            mt1.set_visible(True)
            mt2.set_visible(True)
            mt3.set_visible(True)
            mt4.set_visible(True)
            mt5.set_visible(True)
        except:
            pass
        mfitax.set_xlim(mxl)
        mfitax.set_ylim(myl)
        mprplot(mxl)
        mfitout.draw()
        mmof = 1



def mrelease(event):
    global x1, y1, x2, y2, mmof, mfitout, mfitax, fklmax, fklmin, klmin, klmax, kmin, kmax, fkregion, tmxl, mbgv
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
                mt1.set_visible(False)
                mt2.set_visible(False)
                mt3.set_visible(False)
                mt4.set_visible(False)
                mt5.set_visible(False)
            except:
                pass
            mfitax.set_xlim(sorted([tx1, tx2]))
            mfitax.set_ylim(sorted([ty1, ty2]))
            mprplot(sorted([tx1, tx2]))
            tmxl = sorted([x1, x2])
            mfitout.draw()
        elif fklmin == 1 or fklmax == 1 or fkregion == 1:
            func_cki()
            x1, x2, y1, y2 = [], [], [], []
            mbgv=0
            mfit()
            mfitplot()
        mmof = 1


def tmstate():
    try:
        while True:
            mstate.config(text=str(mst.get()))
    except KeyboardInterrupt:
        pass



def mfli(event):
    global mfiti
    i=mfiti.get()
    if i>0:
        mfiti.set(i-1)
    else:
        mfiti.set(len(ev)-1)
        

def mfri(event):
    global mfiti
    i=mfiti.get()
    if i<len(ev)-1:
        mfiti.set(i+1)
    else:
        mfiti.set(0)


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
            mbcomp2.config(text='Comp 2', bg='purple')
        else:
            mbcomp2.config(text='Comp 2', bg='white')
    else:
        flmcomp1 *= -1
        flmcomp2 *= -1
        mbcomp2.config(text='Comp 2', bg='purple')
        mbcomp1.config(text='Comp 1', bg='white')



def ffitcp():
    mfit()
    mfitplot()



def fmaccept():
    global mfi, mfi_x, mfi_err
    msave_state()
    i = mfiti.get()
    if i not in mfi:
        mfi.append(i)
    if i in mfi_x:
        mfi_x.remove(i)
    if i in mfi_err:
        mfi_err.remove(i)
    mfitplot()



def fmreject():
    global mfi, mfi_x, mfi_err, mbreject, flmreject, mirej
    msave_state()
    i = mfiti.get()
    flmreject *= -1
    if flmreject == 1:
        mirej = i
        mbreject.config(text='End Reject', bg='red')
    else:
        ti = sorted([i, mirej])
        for i in np.linspace(ti[0], ti[1], ti[1]-ti[0]+1, dtype=int):
            if i not in mfi_x:
                mfi_x.append(i)
            if i in mfi:
                mfi.remove(i)
            if i in mfi_err:
                mfi_err.remove(i)
        mbreject.config(text='Reject', bg='white')
        mfitplot()
    



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
        mbposcst.config(bg='purple')
    else:
        min_x1.config(state='disabled')
        min_x2.config(state='disabled')
        mbposcst.config(bg='white')

def mgg_close():
    global mgg
    try:
        flag = True
        count = 0
        smresult = pack_fitpar(mresult)
        for i, j in zip(smresult, smresult_original):
            if not np.array_equal(i, j):
                flag = False
                break
            count += 1
        if flag:
            mgg.destroy()
            clear(mgg)
            mgg=True
        else:
            if messagebox.askyesno("MDC Fitter", "Unsaved changes detected. Do you want to exit without saving?", default='no', icon='warning'):
                try:
                    mgg.destroy()
                    clear(mgg)
                    mgg=True
                except:
                    pass
            else:
                fmend()
                savemfit()
    except Exception as ex:
        print(f"Error:({__file__}, line:{ex.__traceback__.tb_lineno})", ex)
        mgg.destroy()
        clear(mgg)
        mgg=True
    
def mjob():     # MDC Fitting GUI
    global g, mfiti, mfitfig, mfitout, mgg, mxdata, mydata, mdxdata, mdydata, miout, mifig, mfi, mfi_err, mfi_x, mbrmv, flmrmv, mbcgl2, mfp, flmcgl2, fpr, mst, mstate, mwf1, mwf2, maf1, maf2, mxf1, mxf2, mlind, mrind, mbcomp1, flmcomp1, mbcomp2, flmcomp2, min_w1, min_w2, min_a1, min_a2, min_x1, min_x2, lm1, lm2, lm3, lm4, lm5, lm6, mresult, smresult, mbposcst, flmposcst, smcst, mbreject, flmreject, mfitprfig1, mfitprout1, mfitprfig2, mfitprout2, mfitprfig3, mfitprout3, mfpr, mpr, b_pr, mbgv, fdo
    mgg = tk.Toplevel(g, bg='white')
    mgg.protocol("WM_DELETE_WINDOW", mgg_close)
    mdpi = mgg.winfo_fpixels('1i')
    t_sc_w = windll.user32.GetSystemMetrics(0)
    tx = int(t_sc_w*windll.shcore.GetScaleFactorForDevice(0)/100) if g.winfo_x()+g.winfo_width()/2 > t_sc_w else 0
    mgg.geometry(f"1900x1000+{tx}+{sc_y}")
    mgg.title('MDC Lorentz Fit')
    fdo=0
    mpr=0   #button flag 1:ON 0:OFF initial 0:OFF
    # b_pr = tk.Button(mgg, text='Real Time Preview ON', command=f_pr, width=20, height=2, font=('Arial', _size(12), "bold"), bg='white')
    # b_pr.grid(row=0, column=0)
    
    mfpr = 0    #preview plot pcolormesh flag 1:setdata 0:pcolormesh
    fr_pr1 = tk.Frame(master=mgg, bg='white')
    fr_pr1.grid(row=1, column=0)
    b_pr = tk.Button(fr_pr1, text='Real Time Preview OFF', command=f_pr, width=20, height=2, font=('Arial', _size(12), "bold"), bg='white',fg='red')
    b_pr.grid(row=0, column=0)
    mfitprfig2 = Figure(figsize=(3*scale, 3*scale), layout='constrained')
    mfitprout2 = FigureCanvasTkAgg(mfitprfig2, master=fr_pr1)
    mfitprout2.get_tk_widget().grid(row=1, column=0)
    mfitprfig3 = Figure(figsize=(3*scale, 3*scale), layout='constrained')
    mfitprout3 = FigureCanvasTkAgg(mfitprfig3, master=fr_pr1)
    mfitprout3.get_tk_widget().grid(row=2, column=0)
    
    fr_pr2 = tk.Frame(master=mgg, bg='white')
    fr_pr2.grid(row=2, column=0)
    mfitprfig1 = Figure(figsize=(3*scale, 2*scale), layout='constrained')
    mfitprout1 = FigureCanvasTkAgg(mfitprfig1, master=fr_pr2)
    mfitprout1.get_tk_widget().grid(row=0, column=0)
    mst = queue.Queue(maxsize=0)
    mstate = tk.Label(mgg, text='', font=(
        "Arial", _size(14), "bold"), bg="white", fg="black")
    mstate.grid(row=0, column=1)
    fr = tk.Frame(master=mgg, bg='white')
    fr.grid(row=1, column=1)
    frind = tk.Frame(master=fr, bg='white')
    frind.grid(row=0, column=0)
    mlind = tk.Button(frind, text='<<', command=mflind, width=10,
                      height=2, font=('Arial', _size(14), "bold"), bg='white')
    mlind.grid(row=0, column=0)
    mrind = tk.Button(frind, text='>>', command=mfrind, width=10,
                      height=2, font=('Arial', _size(14), "bold"), bg='white')
    mrind.grid(row=0, column=2)

    mfiti = tk.IntVar()
    mfiti.set(0)
    mfiti.trace_add('write', fchki)
    if ScaleFactor <= 100:
        tlength = int(1/0.975*6*mdpi)  # 100
        twidth = int(1/0.975*0.2*mdpi)
    elif ScaleFactor <= 125:
        tlength = int(1/0.985*6*mdpi)  # 125
        twidth = int(1/0.985*0.2*mdpi)
    elif ScaleFactor <= 150:
        tlength = int(1*6*mdpi)  # 150
        twidth = int(1*0.2*mdpi)
    elif ScaleFactor <= 175:
        tlength = int(0.99*6*mdpi)  # 175
        twidth = int(0.99*0.2*mdpi)
    elif ScaleFactor <= 200:
        tlength = int(0.985*6*mdpi)  # 200
        twidth = int(0.985*0.2*mdpi)
    elif ScaleFactor <= 225:
        tlength = int(0.98*6*mdpi)  # 225
        twidth = int(0.98*0.2*mdpi)
    elif ScaleFactor <= 250:
        tlength = int(0.977*6*mdpi)  # 250
        twidth = int(0.977*0.2*mdpi)
    elif ScaleFactor <= 275:
        tlength = int(0.975*6*mdpi)  # 275
        twidth = int(0.975*0.2*mdpi)
    elif ScaleFactor <= 300:
        tlength = int(0.97*6*mdpi)  # 300
        twidth = int(0.97*0.2*mdpi)
    tlength = int(tlength*scale)
    twidth = int(twidth*scale)
    chi = tk.Scale(frind, label='Index', from_=0, to=len(ev)-1, orient='horizontal',
                   variable=mfiti, state='active', bg='white', fg='black', length=tlength, width=twidth, resolution=1)
    chi.grid(row=0, column=1)

    mfi, mfi_err, mfi_x = [], [], [i for i in range(len(ev))]
    mifig = Figure(figsize=(6*scale, 0.2*scale), layout='tight')
    miout = FigureCanvasTkAgg(mifig, master=frind)
    miout.get_tk_widget().grid(row=1, column=1)

    mfitfig = Figure(figsize=(8*scale, 6*scale), layout='constrained')
    mfitout = FigureCanvasTkAgg(mfitfig, master=fr)
    mfitout.get_tk_widget().grid(row=1, column=0)
    mfitout.mpl_connect('motion_notify_event', mmove)
    mfitout.mpl_connect('button_press_event', mpress)
    mfitout.mpl_connect('button_release_event', mrelease)

    xydata = tk.Frame(master=fr, bd=5, bg='white')
    xydata.grid(row=2, column=0)

    mxdata = tk.Label(xydata, text='xdata:', font=(
        "Arial", _size(12), "bold"), width='15', height='1', bd=5, bg='white')
    mxdata.grid(row=0, column=0)
    mydata = tk.Label(xydata, text='ydata:', font=(
        "Arial", _size(12), "bold"), width='15', height='1', bd=5, bg='white')
    mydata.grid(row=0, column=1)
    mdxdata = tk.Label(xydata, text='dx:', font=(
        "Arial", _size(12), "bold"), width='15', height='1', bd=5, bg='white')
    mdxdata.grid(row=0, column=2)
    mdydata = tk.Label(xydata, text='dy:', font=(
        "Arial", _size(12), "bold"), width='15', height='1', bd=5, bg='white')
    mdydata.grid(row=0, column=3)

    # bstop=tk.Button(gg,command=stop,text='Stop',font=('Arial',20),bd=5)
    # bstop.grid(row=1,column=0)

    frpara = tk.Frame(master=mgg, bd=5, bg='white')
    frpara.grid(row=1, column=2)
    
    mfp = [1 for i in range(len(ev))]
    try:
        if fpr == 1:
            mfp = list(smfp)
            mfi = list(smfi)
    except:
        pass
    flmcgl2 = -1
    frre = tk.Frame(master=frpara, bd=5, bg='white')
    frre.grid(row=0, column=0)
    b_mundo = tk.Button(frre, text='Undo', command=mundo,width=25, height=1, font=('Arial', _size(14), "bold"), bg='white')
    b_mundo.grid(row=0, column=0)
    b_mredo = tk.Button(frre, text='Redo', command=mredo,width=25, height=1, font=('Arial', _size(14), "bold"), bg='white')
    b_mredo.grid(row=0, column=1)
    frpara00 = tk.Frame(master=frpara, bd=5, bg='white')
    frpara00.grid(row=1, column=0)

    frfitpar = tk.Frame(master=frpara00, bd=5, bg='white')
    frfitpar.grid(row=0, column=0)
    lm1 = tk.Label(frfitpar, anchor='w', text='', font=(
        "Arial", _size(16), "bold"), width='50', height='1', bd=5, bg='white')
    lm1.grid(row=0, column=0)
    lm2 = tk.Label(frfitpar, anchor='w', text='', font=(
        "Arial", _size(16), "bold"), width='50', height='1', bd=5, bg='white')
    lm2.grid(row=1, column=0)
    lm3 = tk.Label(frfitpar, anchor='w', text='', font=(
        "Arial", _size(16), "bold"), width='50', height='1', bd=5, bg='white')
    lm3.grid(row=2, column=0)
    lm4 = tk.Label(frfitpar, anchor='w', text='', font=(
        "Arial", _size(16), "bold"), width='50', height='1', bd=5, bg='white')
    lm4.grid(row=3, column=0)
    lm5 = tk.Label(frfitpar, anchor='w', text='', font=(
        "Arial", _size(16), "bold"), width='50', height='1', bd=5, bg='white')
    lm5.grid(row=4, column=0)
    lm6 = tk.Label(frfitpar, anchor='w', text='', font=(
        "Arial", _size(16), "bold"), width='50', height='1', bd=5, bg='white')
    lm6.grid(row=5, column=0)

    frYN = tk.Frame(master=frfitpar, bd=5, bg='white')
    frYN.grid(row=6, column=0)
    mbaccept = tk.Button(frYN, text='Accept', command=fmaccept,
                         width=25, height=1, font=('Arial', _size(14), "bold"), bg='white')
    mbaccept.grid(row=0, column=0)
    mbreject = tk.Button(frYN, text='Reject', command=fmreject,
                         width=25, height=1, font=('Arial', _size(14), "bold"), bg='white')
    mbreject.grid(row=0, column=1)

    l1 = tk.Label(frpara00, text='Index Operation', font=(
        "Arial", _size(12), "bold"), width='15', height='1', bd=5, bg='white')
    l1.grid(row=1, column=0)
    froperind = tk.Frame(master=frpara00, bd=5, bg='white')
    froperind.grid(row=2, column=0)
    mbcgl2 = tk.Button(froperind, text='Add 2 Peaks', command=fmcgl2,
                       width=25, height=1, font=('Arial', _size(14), "bold"), bg='white')
    mbcgl2.grid(row=0, column=0)
    mbrmv = tk.Button(froperind, text='Remove', command=fmrmv,
                      width=25, height=1, font=('Arial', _size(14), "bold"), bg='white')
    mbrmv.grid(row=0, column=1)
    mbcomp1 = tk.Button(froperind, text='Comp 1', command=mfcomp1,
                        width=14, height=1, font=('Arial', _size(14), "bold"), bg='white')
    mbcomp1.grid(row=1, column=0)
    mbcomp2 = tk.Button(froperind, text='Comp 2', command=mfcomp2,
                        width=14, height=1, font=('Arial', _size(14), "bold"), bg='white')
    mbcomp2.grid(row=1, column=1)

    mbfitcp = tk.Button(master=frpara00, text='Fit Components', command=ffitcp,
                        width=40, height=1, font=('Arial', _size(14), "bold"), bg='white')
    mbfitcp.grid(row=3, column=0)

    frwr = tk.Frame(master=frpara00, bd=5, bg='white')
    frwr.grid(row=4, column=0)
    l2 = tk.Label(frwr, text='FWHM Ratio', font=(
        "Arial", _size(12), "bold"), width='15', height='1', bd=5, bg='white')
    l2.grid(row=0, column=1)
    l3 = tk.Label(frwr, text=':', font=("Arial", _size(12), "bold"),
                  width='15', height='1', bd=5, bg='white')
    l3.grid(row=1, column=1)
    mwf1 = tk.StringVar()
    mwf1.set('0')
    mwf1.trace_add('write', fmwf1)
    min_w1 = tk.Entry(frwr, font=("Arial", _size(12), "bold"),
                      width=7, textvariable=mwf1, bd=5)
    min_w1.grid(row=1, column=0)
    mwf2 = tk.StringVar()
    mwf2.set('0')
    mwf2.trace_add('write', fmwf2)
    min_w2 = tk.Entry(frwr, font=("Arial", _size(12), "bold"),
                      width=7, textvariable=mwf2, bd=5)
    min_w2.grid(row=1, column=2)

    frar = tk.Frame(master=frpara00, bd=5, bg='white')
    frar.grid(row=5, column=0)
    l2 = tk.Label(frar, text='Area Ratio', font=(
        "Arial", _size(12), "bold"), width='15', height='1', bd=5, bg='white')
    l2.grid(row=0, column=1)
    l3 = tk.Label(frar, text=':', font=("Arial", _size(12), "bold"),
                  width='15', height='1', bd=5, bg='white')
    l3.grid(row=1, column=1)
    maf1 = tk.StringVar()
    maf1.set('0')
    maf1.trace_add('write', fmaf1)
    min_a1 = tk.Entry(frar, font=("Arial", _size(12), "bold"),
                      width=7, textvariable=maf1, bd=5)
    min_a1.grid(row=1, column=0)
    maf2 = tk.StringVar()
    maf2.set('0')
    maf2.trace_add('write', fmaf2)
    min_a2 = tk.Entry(frar, font=("Arial", _size(12), "bold"),
                      width=7, textvariable=maf2, bd=5)
    min_a2.grid(row=1, column=2)

    mbposcst = tk.Button(frpara00, text='Position constraint', command=fmposcst,
                         width=25, height=1, font=('Arial', _size(14), "bold"), bg='white')
    mbposcst.grid(row=6, column=0)

    frxr = tk.Frame(master=frpara00, bd=5, bg='white', padx=30)
    frxr.grid(row=7, column=0)
    l3 = tk.Label(frxr, text='x2 =', font=("Arial", _size(12), "bold"),
                  width='5', height='1', bd=5, bg='white')
    l3.grid(row=0, column=0)
    mxf1 = tk.StringVar()
    mxf1.set('1')
    mxf1.trace_add('write', fmxf1)
    min_x1 = tk.Entry(frxr, font=("Arial", _size(12), "bold"), width=7,
                      textvariable=mxf1, bd=5, state='disabled')
    min_x1.grid(row=0, column=1)
    l3 = tk.Label(frxr, text='* x1 +', font=("Arial", _size(12), "bold"),
                  width='5', height='1', bd=5, bg='white')
    l3.grid(row=0, column=2)
    mxf2 = tk.StringVar()
    mxf2.set('0')
    mxf2.trace_add('write', fmxf2)
    min_x2 = tk.Entry(frxr, font=("Arial", _size(12), "bold"), width=7,
                      textvariable=mxf2, bd=5, state='disabled')
    min_x2.grid(row=0, column=3)

    frout = tk.Frame(master=mgg, bd=5, bg='white')
    frout.grid(row=2, column=1)
    bfall = tk.Button(frout, text='Fit All', command=fmfall,
                      width=25, height=1, font=('Arial', _size(14), "bold"), bg='white')
    bfall.grid(row=0, column=0)

    flmreject = -1
    flmposcst = -1
    flmrmv = -1
    flmcomp1 = -1
    flmcomp2 = -1

    bprv = tk.Button(frout, text='Preview', command=fmpreview, width=25,
                     height=1, font=('Arial', _size(14), "bold"), bg='white')
    bprv.grid(row=1, column=0)
    
    bend = tk.Button(frout, text='Export All', command=fmend, width=25,
                     height=1, font=('Arial', _size(14), "bold"), bg='white')
    bend.grid(row=2, column=0)

    frexp = tk.Frame(frout, bd=5, bg='white')
    frexp.grid(row=3, column=0)

    bend1 = tk.Button(frexp, text='Export Comp 1', command=fmend1, width=25,
                      height=1, font=('Arial', _size(14), "bold"), bg='white')
    bend1.grid(row=0, column=0)
    
    bend2 = tk.Button(frexp, text='Export Comp 2', command=fmend2, width=25,
                      height=1, font=('Arial', _size(14), "bold"), bg='white')
    bend2 .grid(row=0, column=1)
    
    mbgv=0
    mgg.bind("<Up>",mfbgu)
    mgg.bind("<Down>",mfbgd)
    mgg.bind("<Left>",mfli)
    mgg.bind("<Right>",mfri)
    
    ##### test ##### 
    # mprbg1()
    # mprbg2()
    # mprbg3()
    
    mresult = [[]for i in range(len(ev))]
    try:
        flsmresult = smresult
        flsmcst = smcst
        flsmresult = None
        flsmcst = None
    except:
        smcst=np.zeros(len(ev)*6).reshape(len(ev),6)
        smresult = [1]
    if mprfit == 1:
        fmfall()
    else:
        mfitplot()
    tt = threading.Thread(target=tmstate)
    tt.daemon = True
    tt.start()
    mgg.update()
    screen_width = mgg.winfo_reqwidth()
    screen_height = mgg.winfo_reqheight()
    tx = int(t_sc_w*windll.shcore.GetScaleFactorForDevice(0)/100) if g.winfo_x()+g.winfo_width()/2 > t_sc_w else 0
    mgg.geometry(f"{screen_width}x{screen_height}+{tx}+{sc_y}")
    mgg.update()


#################################### prefit ######################################################
mprfit = 0


class oklim():
    def __init__(self, npzf, ev, phi):
        if npzf:
            avg = np.mean(phi)
            l = max(phi) - min(phi)
            self.min = np.float64([avg - l/40 for i in ev])
            self.max = np.float64([avg + l/40 for i in ev])
        else:
            self.min = np.float64((2*m*ev*1.602176634*10**-19)**0.5*np.sin(-0.5/180*np.pi)*10**-10/(h/2/np.pi))
            self.max = np.float64((2*m*ev*1.602176634*10**-19)**0.5*np.sin(0.5/180*np.pi)*10**-10/(h/2/np.pi))
        
mgg=None
def fitm(mdc_pars):
    if mgg is not None:
        mgg.lift()
        return
    init_pars(mdc_pars)
    global ev, phi, data, mvv, maa1, maa2, fmxx, fmyy, fmx, fmy, kmin, kmax, cki, mbase, mprfit, mf_prswap, smresult, klim, fpr
    mprfit = 0
    cki = []
    mbase = [0 for i in range(len(ev))]
    mf_prswap = []
    klim = oklim(npzf, ev, phi)
    shape=data.shape
    det=data.data[shape[0]//2, shape[1]//2]
    if mdet != det:
        fpr = 0
    if fpr == 1:
        try:
            kmin, kmax = skmin, skmax
        except NameError:
            kmin, kmax = klim.min.copy(), klim.max.copy()
        if len(scki) >= 2:
            cki = scki
    else:
        kmin, kmax = klim.min.copy(), klim.max.copy()
    # fmxx = np.float64((np.ones(len(phi)*len(ev))).reshape(len(ev), len(phi)))
    # fmyy = np.float64((np.ones(len(phi)*len(ev))).reshape(len(ev), len(phi)))
    # fmxx *= -50
    # fmyy *= -50
    fmx = np.float64(np.arange(len(phi)*len(ev)).reshape(len(ev), len(phi)))
    fmy = np.float64(np.arange(len(phi)*len(ev)).reshape(len(ev), len(phi)))
    mvv = np.float64(np.arange(len(ev)))
    maa1 = np.float64(np.arange(4*len(ev)).reshape(len(ev), 4))
    maa2 = np.float64(np.arange(8*len(ev)).reshape(len(ev), 8))
    pbar = tqdm.tqdm(total=len(ev), desc='MDC', colour='green')
    for i, v in enumerate(ev):
        ecut = data.sel(eV=v, method='nearest')
        if npzf:x = phi
        else:x = np.float64((2*m*v*1.602176634*10**-19)**0.5*np.sin(phi/180*np.pi)*10**-10/(h/2/np.pi))
        y = ecut.to_numpy().reshape(len(x))
        try:
            xx, x_arg = filter(x, kmin[i], kmax[i])
        except IndexError:
            print("\033[31m\nCheck the Raw Data compatible with the current MDC Fitted File\n\033[0m")
            return
        except Exception as e:
            print("\nError occurred while filtering:", e, '\n')
            return
        # tx = x[np.argwhere(x >= kmin[i])].flatten()
        # xx = tx[np.argwhere(tx <= kmax[i])].flatten()
        # ty = y[np.argwhere(x >= kmin[i])].flatten()
        # yy = ty[np.argwhere(tx <= kmax[i])].flatten()
        # yy = y[x_arg]
        # yy = np.where(yy > int(base.get()), yy, int(base.get()))
        try:
            if i in smfi and fpr == 1:
                a1 = smaa1[i, :]
                a2 = smaa2[i, :]
                smrx1 = smresult[i, 0]
                smrx2 = smresult[i, 1]
                smrh1 = smresult[i, 2]
                smrh2 = smresult[i, 3]
                smrw1 = smresult[i, 4]
                smrw2 = smresult[i, 5]
                if smaa1[i, 1] == 10 or smaa2[i, 1] == 10:
                    mprfit = 1
                else:
                    # fmxx[i, :len(xx)] = xx
                    # tx = fmxx[i, np.argwhere(fmxx[i, :] >= -20)].flatten()
                    tx = xx
                    ty = gl1(tx, *a2[:4])
                    s1 = np.sum(np.array([((ty[i]+ty[i+1])/2)for i in range(len(tx)-1)])
                            # Area 1
                            * np.array(([(tx[i+1]-tx[i])for i in range(len(tx)-1)])))
                    ty = gl1(tx, *a2[-4:])
                    s2 = np.sum(np.array([((ty[i]+ty[i+1])/2)for i in range(len(tx)-1)])
                            # Area 2
                            * np.array(([(tx[i+1]-tx[i])for i in range(len(tx)-1)])))
                    if s1 < s2:
                        t1, t2 = a2[:4], a2[-4:]
                        a2 = np.array([t2, t1]).flatten()
                        mf_prswap.append(i)
                        smrx1 = smrx1.replace('x2', 'x1').replace('x1:', 'x2:')
                        smrx2 = smrx2.replace('x1', 'x2').replace('x2:', 'x1:')
                        smrh1 = smrh1.replace('h1:', 'h2:')
                        smrh2 = smrh2.replace('h2:', 'h1:')
                        smrw1 = smrw1.replace('w1:', 'w2:').replace('w2', 'w1')
                        smrw2 = smrw2.replace('w2:', 'w1:').replace('w1', 'w2')
                        smr = np.array([smrx2,smrx1,smrh2,smrh1,smrw2,smrw1]).flatten()
                    else:
                        smr = np.array([smrx1,smrx2,smrh1,smrh2,smrw1,smrw2]).flatten()
            else:
                # a1 = [(kmin[i]+kmax[i])/2, (np.max(y) -
                #                             int(base.get())), 0.5, int(base.get())]
                # a2 = [(kmin[i]+kmax[i])/2, (np.max(y)-int(base.get())), 0.5, int(base.get()),
                #       (kmin[i]+kmax[i])/2, (np.max(y)-int(base.get())), 5, int(base.get())]
                a1 = [(kmin[i]+kmax[i])/2, (np.max(y)-0), 0.5, 0]
                a2 = [(kmin[i]+kmax[i])/2, (np.max(y)-0), 0.5, 0,
                      (kmin[i]+kmax[i])/2, (np.max(y)-0), 5, 0]
                smr = ['' for i in range(6)]
        except:
            # a1 = [(kmin[i]+kmax[i])/2, (np.max(y) -
            #                             int(base.get())), 0.5, int(base.get())]
            # a2 = [(kmin[i]+kmax[i])/2, (np.max(y)-int(base.get())), 0.5, int(base.get()),
            #       (kmin[i]+kmax[i])/2, (np.max(y)-int(base.get())), 0.5, int(base.get())]
            a1 = [(kmin[i]+kmax[i])/2, (np.max(y)-0), 0.5, 0]
            a2 = [(kmin[i]+kmax[i])/2, (np.max(y)-0), 0.5, 0,
                  (kmin[i]+kmax[i])/2, (np.max(y)-0), 0.5, 0]
            smr = ['' for i in range(6)]

        # fmxx[i, :len(xx)] = xx
        # fmyy[i, :len(yy)] = yy
        fmxx, fmyy = 1, 1 # 未使用 暫時保留
        
        fmx[i, :] = x
        fmy[i, :] = y
        mvv[i] = v
        maa1[i, :] = a1
        maa2[i, :] = a2
        try:
            smresult[i, :]=smr
        except:
            pass
        pbar.update(1)
    pbar.close()
    tt1 = threading.Thread(target=mjob)
    tt1.daemon = True
    tt1.start()
