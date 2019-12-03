import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import h5py
import os
import qutip as qt
from ipywidgets import interactive, fixed

"""Fluxonium stuff"""

#gives a qutip object for the fluxonium hamiltonian
def H_fluxonium(n,ec,el,ej,phiext): #n dimension of hilbert space
    a = qt.destroy(n)
    num = 1j*(a-a.dag())/(np.sqrt(2*np.sqrt(8*ec/el)))
    phi = (a+a.dag())*(np.sqrt(np.sqrt(8*ec/el)/2))
    cosphi = (phi+phiext*qt.qeye(n)).cosm()
    ham = 4*ec*num**2-ej*cosphi+0.5*el*phi**2
    return ham.copy()

def H_fluxonium_new(n,ec,el,ej,phiext,operators = False): #n dimension of hilbert space
    a = qt.destroy(n)
    num = 1j*(a.dag()-a)/(np.sqrt(2*np.sqrt(8*ec/el)))
    phi = (a+a.dag())*(np.sqrt(np.sqrt(8*ec/el)/2))
    cosphi = (phi+phiext*qt.qeye(n)).cosm()
    ham = 4*ec*num**2-ej*cosphi+0.5*el*phi**2
    if operators:
        return ham.copy(), num.copy(), phi.copy()
    else:
        return ham.copy()

#returns the spectrum of the fluxonium
def fluxonium_spectrum(flux,Ec=5,El=0.5,Ej=20,a=[0,1],N=1): #transitions from a, up to levels N higher than a
    H = H_fluxonium(50,Ec,El,Ej,flux)
    eigenvalues = H.eigenstates()[0].round(5)
    transitions2D = []
    for start in a:
        atoaplusN = eigenvalues[start+1:]-eigenvalues[start]
        #transitions2D.append(np.array(atoaplusN[start:start+N]))
        transitions2D.append(np.array(atoaplusN[:N]))
    return np.array(transitions2D)

def fluxonium_spectrum_new(flux,Ec=5,El=0.5,Ej=20,a=[0,1],Nmax = 2, full_info = False): #transitions from a, up to levels N higher than a
    if full_info:
        H, n, phi = H_fluxonium_new(50,Ec,El,Ej,flux,operators=True)
    else:
        H = H_fluxonium_new(50,Ec,El,Ej,flux)
     
    eigenvalues, eigenstates = H.eigenstates()
    eigenvalues = eigenvalues.round(5)
    
    transitions2D = np.zeros((len(a),Nmax-a[0]))
    for i in range(len(a)):
        start = a[i]
        transitions2D[i,start:] = eigenvalues[start:Nmax]-eigenvalues[start]
        
    if full_info:
        return {'transitions': np.array(transitions2D), 'hamiltonian': H.copy(), 'charge_op': n.copy(), 'phase_op': phi.copy(),
                'eigenvals': eigenvalues, 'eigenstates': eigenstates}
    else:
        return {'transitions': np.array(transitions2D)}

def plot_fluxonium_spectrum(**topkwargs):  
    #finding plot limits
    try:
        xlims = topkwargs['xlims']
        xlimited = True
    except KeyError:
        xlimited = False
     
    try:
        ylims = topkwargs['ylims']
        ylimited = True
    except KeyError:
        ylimited = False
                
    #helper function to plot fluxonium spectrum above an image
    flux = np.linspace(-2,2,50)
    def plot_FS(Ec=5,El=0.5,Ej=20,a=[0,1],N=1,alpha=1,beta=0,**kwargs):
               
        #clear figure
        plt.clf()
                   
        #calculate transitions
        transitions3D = []
        for flx in flux:
            transitions3D.append(fluxonium_spectrum(flx*2*np.pi,Ec,El,Ej,a,N))
        transitions = np.array(transitions3D)
        
        #plot transitions
        for i in range(np.shape(transitions)[1]):
            plt.plot(alpha*(flux-np.median(flux))+beta,transitions[:,i,:],c='C'+str(i),label=str(i))
            
        #show plot
        plt.xlabel(r'Voltage [V]')
        plt.ylabel('Energy [GHz]')
        if xlimited:
            plt.xlim(xlims)
        if ylimited:
            plt.ylim(ylims)
        #plt.legend()
        plt.show()

    # actual generation of the interactive plot
    plt.figure('Spectrum')
    interactive_plot = interactive(plot_FS, Ec=(0, 1,0.01), El=(0, 1,0.01),Ej=(0,3,0.1),a=fixed([0,1]),N = fixed(2),alpha = (0,3,0.01),beta=(0,2,0.01))
    display(interactive_plot)
    
#plug and play fluxonium fitting
def fit_fluxonium_spectrum(spectrum_images,xs,ys,**topkwargs):
    
    #number of images to show
    image_num = len(xs) 
    
    #finding plot limits
    try:
        xlims = topkwargs['xlims']
    except KeyError:
        xx = [item for x in xs for item in x]
        xlims = (np.amin(xx),np.amax(xx))
     
    try:
        ylims = topkwargs['ylims']
    except KeyError:
        yy = [item for y in ys for item in y]
        ylims = (np.amin(yy),np.amax(yy))   
    
    #creating meshgrids for color plots
    def meshgrids(xs,ys):
        Xs = []
        Ys = []
        for i in range(len(xs)):
            X,Y = np.meshgrid(xs[i],ys[i])
            Xs.append(X)
            Ys.append(Y)
        return Xs, Ys
            
    Xs, Ys = meshgrids(xs,ys)
    
    #helper function to plot fluxonium spectrum above an image
    flux = np.linspace(0,1,50)
    def plot_FS(Ec=5,El=0.5,Ej=20,a=[0,1],N=1,alpha=1,beta=0,**kwargs):
        
        #clear figure
        plt.clf()
        
        #plot images
        for i in range(image_num):
            plt.pcolormesh(Xs[i],Ys[i],spectrum_images[i])
            
        #calculate transitions
        transitions3D = []
        for flx in flux:
            transitions3D.append(fluxonium_spectrum(flx*2*np.pi,Ec,El,Ej,a,N))
        transitions = np.array(transitions3D)
        
        #plot overlay
        for i in range(np.shape(transitions)[1]):
            plt.plot(alpha*(flux-np.median(flux))+beta,transitions[:,i,:],c='C'+str(i),label=str(i))
            
        #show plot
        plt.xlabel(r'Voltage [V]')
        plt.ylabel('Energy [GHz]')
        plt.xlim(xlims)
        plt.ylim(ylims)
        #plt.legend()
        plt.show()

    # actual generation of the interactive plot
    plt.figure('Spectrum')
    interactive_plot = interactive(plot_FS, Ec=(0, 1,0.01), El=(0, 1,0.01),Ej=(0,3,0.1),a=fixed([0,1]),N = fixed(2),alpha = (0,3,0.01),beta=(0,2,0.01))
    display(interactive_plot)

#only works in a jupyter notebook
# fs must be defined using the def command with arguments with default values
def plot_animated (fs, xmin, xmax, N = 1000): #fs is list of functions to be plotted
    import inspect
    args_names,args_arg = np.unique(np.concatenate([inspect.getfullargspec(f).args[1:] for f in fs]), return_index = True)
    args_default = np.concatenate([inspect.getfullargspec(f).defaults for f in fs])[args_arg]
    args = dict(zip(args_names, args_default))
    
    from IPython.display import display
    
    
    plt.figure('test')
    def update_plot(xmin, xmax, **args):
        xs = np.linspace(xmin, xmax, N)
        plt.clf()
        for f in fs:
            plt.plot(xs, f(xs, *[args[name] for name in inspect.getfullargspec(f).args[1:]]))
            plt.show()
    
    args['xmin'] = xmin
    args['xmax'] = xmax
    
    w = widgets.interactive(update_plot, **args)
    
    display(w)

"""Other stuff"""
def make_polar(real,imag):
    cplx = real+1j*imag
    #return abs(cplx), np.unwrap(np.angle(cplx))
    return abs(cplx), np.angle(cplx)

def limit_data_x(x,y,xmin,xmax):
    minindex = np.argmax(x>xmin)-1
    maxindex = np.argmax(x>=xmax)
    return x[minindex:maxindex+1], y[minindex:maxindex+1]

def import_file(path,**kwargs): #laziness 10/10, practicality 10/10
    extension = os.path.splitext(path)[1][1:]
    if extension == 'csv':
        return import_csv(path,**kwargs)
    if extension == 'h5':
        return import_h5(path)
   
def import_h5(path):
    openfile = h5py.File(path,'r')
    data_dict = {}
    for key in list(openfile.keys()):
        data_dict[key] = np.array(openfile[key])
    openfile.close()
    return data_dict, data_dict.keys()
    
def import_csv(path,param_sweep=False,title_line=8):
    title_pos = title_line
    with open(path,'r') as openfile:

        if param_sweep == False:
            for i in range(title_pos):
                line = openfile.readline()
            titles = line[:-1].split(',') #get column titles
            data = np.loadtxt(path,delimiter=',',skiprows=title_pos)  #import all the data
            data_dict = {titles[i]: data[:,i] for i in range(len(titles))} #create results dictionary
            return data_dict, list(data_dict.keys())

        if param_sweep == True:
            data_dict = {} #overall dictionary
            params_dict = {} #dictionary with data from individual parameter values
            values = [] #parameter values
        
            for i in range(title_pos):
                line = openfile.readline()
            
            titles = line[:-1].split(',') #get column titles
            line = openfile.readline()
            param_name, val = line[:-1].split(' = ') #get the name of the parameter
            values.append(float(val)) #register first parameter value
            params_dict[param_name + ' = ' + val] = [] #start data dictionary
            line = openfile.readline() #go to first line of data
            while line != "": #keep going until end of file
                try:
                    params_dict[param_name + ' = ' + val].append(list(map(float,line[:-1].split(',')))) #attempt to add line to data array
                except ValueError: #float will throw value error
                    if line[:len(param_name)] == param_name: #check if the line with the error is a line with the parameter value
                        val = line[:-1].split(' = ')[1] #update parameter value
                        values.append(float(val)) #store it
                        params_dict[param_name + ' = ' + val] = [] #start data dictionary
                line = openfile.readline() #next line
            
            keys, full_values = (list(params_dict.keys()),list(params_dict.values())) #sale mais efficace
            params_dict = {keys[i]: np.array(full_values[i]) for i in range(len(keys))} #sale mais efficace
            data_dict = {'titles':titles,'param data':params_dict,'param values': values}
            return data_dict, list(data_dict.keys())

def lorentzian_function(x,A,sig,xres,offset):
    return offset+A/(1+(x-xres)**2/sig**2)

def abs_hanger_function(x,a,b,sig,xres,offset):
    return np.where(abs(offset)>abs(a),abs(offset-(a-2j*b)/(1+2j*(x-xres)/sig)),0) #we must ensure that offset > a which is equivalent to saying that Qc,Qi>Q0

def abs_sym_hanger_function(x,a,sig,xres,offset):
    return np.where(abs(offset)>abs(a),abs(offset-a/(1+2j*(x-xres)/sig)),0)

def sym_hanger_function(x,Qi,Qc,f0):
    return (Qc+2j*Qc*Qi*(x-f0)/f0)/(Qi+Qc+2j*Qc*Qi*(x-f0)/f0)

def hanger_function(x,Qi,Qc,f0,deltaf,f0_sym):
    #only two out of f0, f0_smy and deltaf are independent given that f0 = f0_sym + deltaf
    return (Qc+1j*Qc*Qi*(2*(x-f0)/f0+2*deltaf/f0_sym))/(Qi+Qc+2j*Qc*Qi*(x-f0)/f0)

def reflection_function(x,kc,kl,f0):
    return (kc-kl-2j*(x-f0))/(kc+kl+2j*(x-f0))

def transmission_function(x,kcscaled,kt,f0):
    return (2*np.sqrt(kcscaled))/(kt+2j*(x-f0)) #we put a plus on the denominator in this formula to deal with quadrant problems in the arctan function

def abs_reflection_function(x,kc,kl,f0,scaling):
    return scaling*abs((kc-kl-2j*(x-f0))/(kc+kl+2j*(x-f0)))

def abs_transmission_function(x,kcscaled,kt,f0):
    return abs((2*np.sqrt(kcscaled))/(kt-2j*(x-f0)))

def phase_adjust(x,func,delay,offset):
    return np.unwrap(np.angle(func(x)))-2*np.pi*x*delay+offset

def phase_adjust_fit(freqs,S21arg,model='hanger',**kwargs):
    base_function = lambda x: 1 #we have to define the function so that we don't get a referenced before assignment error
    try:
        p0 = kwargs['p0_arg'] #try to find user defined starting parameters
    except KeyError:
        delay0 = -(S21arg[-1]-S21arg[0])/(2*np.pi*(freqs[-1]-freqs[0])) #determine slope of phase = electrical delay
        offset0 = S21arg[0]+2*np.pi*delay0*freqs[0] #determine offset (this is just solving an affine equation)
        p0 = [delay0, offset0]
    if model == 'sym_hanger':
        base_function = lambda x: sym_hanger_function(x,*kwargs['sym_hanger_params']) #the item 'hanger_params' must be a list of length 3
    if model == 'hanger':
        base_function = lambda x: hanger_function(x,*kwargs['hanger_params']) #the item 'hanger_params' must be a list of length 5
    if model == 'reflection':
        base_function = lambda x: reflection_function(x,*kwargs['reflection_params']) #the item 'reflection_params' must be a list of length 3
    if model == 'transmission':
        base_function = lambda x: transmission_function(x,*kwargs['transmission_params']) #the item 'transmission_params' must be a list of length 3
        
    def model_function(x,delay,offset):
        return phase_adjust(x,base_function,delay,offset)
        
    popt, pcov = opt.curve_fit(model_function,freqs,S21arg,p0)
    fit = model_function(freqs,*popt)
    initial = model_function(freqs,*p0)
    
    return popt, p0, fit, pcov, initial

def abs_reflection_fit(x,y,**kwargs):
    try:
        p0 = kwargs['p0_abs']
        x0 = p0[2]
        p0[2] = 0
    except KeyError:
        scale0 = np.average([y[0],y[-1]])
        res_index = np.argmin(y)
        x0 = x[res_index]
        dip = y[res_index]/scale0
        HM_index = np.argmax(abs(y/scale0)>(1/2+1/2*dip)) #we find the half maximum of the inverse bump aka dip
        sig = 2*abs(x0-x[HM_index])
        #we assume that the two port resonator is critically coupled with respect to each port, and that the FWHM is the sum of the kappas
        kc0 = sig/3
        kl0 = 2*sig/3
        p0 = [kc0,kl0,0,scale0] #ACHTUNG kc and kl have units of frequency and not 2pi frequency
        
    popt, pcov = opt.curve_fit(abs_reflection_function,x-x0,y,p0)
    popt[2] += x0
    p0[2] += x0
    fit = abs_reflection_function(x,*popt)
    initial = abs_reflection_function(x,*p0)
    
    return popt, p0, fit, pcov, initial

def abs_sym_hanger_fit(x,y,**kwargs):
    try:
        p0 = kwargs['p0_abs']
        x0 = p0[2]
        p0[2] = 0
    except KeyError:
        offset0 = np.average([y[0],y[-1]])
        res_index = np.argmin(y)
        x0 = x[res_index]
        a0 = abs(y[res_index]-offset0)
        HM_index = np.argmax(abs(y-offset0)>abs(a0)/2)
        sig0 = 2*abs(x0-x[HM_index])
        p0 = [a0,sig0,0,offset0]
        
    popt, pcov = opt.curve_fit(abs_sym_hanger_function,x-x0,y,p0)
    popt[2] += x0
    p0[2] += x0
    fit = abs_sym_hanger_function(x,*popt)
    initial = abs_sym_hanger_function(x,*p0)
    
    return popt, p0, fit, pcov, initial
    
def abs_hanger_fit(x,y,**kwargs):
    try:
        p0 = kwargs['p0_abs']
        x0 = p0[3]
        p0[3] = 0
    except KeyError:
        offset0 = np.average([y[0],y[-1]])
        res_index = np.argmin(y)
        x0 = x[res_index]
        a0 = abs(y[res_index]-offset0)
        b0 = 0 #we assume the asymmetry to be small (may be a bad guess though)
        HM_index = np.argmax(abs(y-offset0)>abs(a0)/2)
        sig0 = 2*abs(x0-x[HM_index])
        p0 = [a0,b0,sig0,0,offset0]
    
    popt, pcov = opt.curve_fit(abs_hanger_function,x-x0,y,p0)
    popt[3] += x0
    p0[3] += x0
    fit = abs_hanger_function(x,*popt)
    initial = abs_hanger_function(x,*p0)
    
    return popt, p0, fit, pcov, initial

def abs_transmission_fit(x,y,**kwargs):
    try:
        p0 = kwargs['p0_abs']
        x0 = p0[2]
        p0[2] = 0
    except KeyError:
        res_index = np.argmax(y) #find resonance frequency
        x0 = x[res_index]
        y0 = y[res_index]
        HM_index = np.argmax(abs(y)>y0/2)
        kt0 = 2*abs(x0-x[HM_index])/np.sqrt(3) #determine FWHM (not sure about the sqrt(3) though)
        kcscaled0 = (y0*kt0/2)**2 # determine scaling and kc multiplied
        p0 = [kcscaled0,kt0,0]
    
    popt, pcov = opt.curve_fit(abs_transmission_function,x-x0,y,p0)
    popt[2] += x0
    p0[2] += x0
    fit = abs_transmission_function(x,*popt)
    initial = abs_transmission_function(x,*p0)

    return popt, p0, fit, pcov, initial

def lorentzian_fit(x,y,**kwargs):        
    try: 
        p0 = kwargs['p0']
        x0 = p0[2]
        p0[2] = 0 #we force the fit to be centered around the guessed resonance frequency
    except KeyError:
        offset0 = np.average([y[0],y[-1]]) #find function offset
        res_index = np.argmax(abs(y-offset0)) #index of resonance in array
        x0 = x[res_index] #frequency of resonance
        A0 = y[res_index]-offset0 #amplitude at resonance
        HM_index = np.argmax(abs(y-offset0)>abs(A0)/2) #index of half maximum
        sig0 = abs(x0-x[HM_index]) #value of half maximum frequency difference
        p0 = [A0,sig0,0,offset0] #define start parameters of fit (we fit with the resonance centered around 0)

    popt, pcov = opt.curve_fit(lorentzian_function,x-x0,y,p0)
    popt[2] += x0 #we correct the parameters by what we originally substracted
    p0[2] = x0
    fit = lorentzian_function(x,*popt) #find the y of the fitted model
    initial = lorentzian_function(x,*p0) #find the y of the original guess
    return popt, p0, fit, pcov, initial

def curve_fit(freqs,data,model='lorentzian',plots = False,verbose = False,**kwargs):
    if model == 'lorentzian': #for this method mke sure data is an array with only 1 dimension
        popt, p0, fit, pcov, initial = lorentzian_fit(freqs,data,**kwargs) #do the fit
        results = {'popt':popt} #return the optimal parameters
        if plots: 
            fig, ax = plt.subplots()
            ax.plot(freqs,data,label='data') #plot the original data
            ax.plot(freqs,fit,label='fit') #plot the fit
            ax.plot(freqs,initial,'--',label='initial') #plot the initial guess
            plt.legend()
            plt.show()
        if verbose:
            results.update({'p0':p0,'pcov':pcov,'fit':fit,'initial':initial}) #return all results and parameters for analysis
        return results
    
    else:
        results = {}
        phys_params = {}
        abs_results = {}
        arg_results = {}
       
        if model == 'sym_abs_hanger': #for this method make sure data is an array with 2 dimensions (abs and arg)
            dataabs, dataarg = data
            popt_abs, p0_abs, fit_abs, pcov_abs, initial_abs = abs_sym_hanger_fit(freqs,dataabs,**kwargs)
            
            #we have to do some calculations so that the parameters we return are the ones which are physically interesting
            a,sig,xres,offset = popt_abs
            Q0 = xres/sig #see Geerlings' thesis for details on these formulas
            Qc = offset*xres/(a*sig)
            Qi = 1/(1/Q0-1/Qc)
            phys_params.update({'Q0':Q0,'Qi':Qi,'Qc':Qc,'f0':xres})
            abs_results.update({'popt':popt_abs,'p0':p0_abs,'pcov':pcov_abs,'fit':fit_abs,'initial':initial_abs})
            
            #we now need to fit the phase
            popt_arg, p0_arg, fit_arg, pcov_arg, initial_arg = phase_adjust_fit(freqs,dataarg,model='sym_hanger',sym_hanger_params=list(phys_params.values())[1:],**kwargs) #use the above defined fitting algorithm
            delay = popt_arg[0]
            phys_params.update({'delay':delay})
            arg_results.update({'popt':popt_arg,'p0':p0_arg,'pcov':pcov_arg,'fit':fit_arg,'initial':initial_arg})
            
            results.update({'phys_params':phys_params})
            
        if model == 'abs_hanger': #for this method make sure data is an array with 2 dimensions (abs and arg)
            dataabs, dataarg = data
            popt_abs, p0_abs, fit_abs, pcov_abs, initial_abs = abs_hanger_fit(freqs,dataabs,**kwargs)
            
            #we have to do some calculations so that the parameters we return are the ones which are physically interesting
            a,b,sig,xres,offset = popt_abs
            Q0 = xres/sig #see Geerlings' thesis for details on these formulas
            Qc = offset*xres/(a*sig)
            Qi = 1/(1/Q0-1/Qc)
            deltaf = xres*(1-1/(1+b/(Q0*offset)))
            xres_sym = xres/(1+b/(Q0*offset))
            phys_params.update({'Q0':Q0,'Qi':Qi,'Qc':Qc,'f0':xres,'deltaf':deltaf,'f0_sym':xres_sym})
            abs_results.update({'popt':popt_abs,'p0':p0_abs,'pcov':pcov_abs,'fit':fit_abs,'initial':initial_abs})
            
            #we now need to fit the phase
            popt_arg, p0_arg, fit_arg, pcov_arg, initial_arg = phase_adjust_fit(freqs,dataarg,hanger_params=list(phys_params.values())[1:],**kwargs) #use the above defined fitting algorithm
            delay = popt_arg[0]
            phys_params.update({'delay':delay})
            arg_results.update({'popt':popt_arg,'p0':p0_arg,'pcov':pcov_arg,'fit':fit_arg,'initial':initial_arg})
            
            results.update({'phys_params':phys_params})
        
        if model == 'abs_reflection': #for this method make sure data is an array with 2 dimensions (abs and arg)
            dataabs, dataarg = data
            popt_abs, p0_abs, fit_abs, pcov_abs, initial_abs = abs_reflection_fit(freqs,dataabs,**kwargs)
            kc,kl,f0,scale = popt_abs
            phys_params.update({'kc':kc,'kl':kl,'f0':f0})
            abs_results.update({'popt':popt_abs,'p0':p0_abs,'pcov':pcov_abs,'fit':fit_abs,'initial':initial_abs})
            
            #we fit the phase
            popt_arg, p0_arg, fit_arg, pcov_arg, initial_arg = phase_adjust_fit(freqs,dataarg,model='reflection',reflection_params=list(phys_params.values()),**kwargs) #use the above defined fitting algorithm
            delay = popt_arg[0]
            phys_params.update({'delay':delay})
            arg_results.update({'popt':popt_arg,'p0':p0_arg,'pcov':pcov_arg,'fit':fit_arg,'initial':initial_arg})
            
            results.update({'phys_params':phys_params})
        
        if model == 'transmission': #for this method make sure data is an array with 2 dimensions (abs and arg)
            dataabs, dataarg = data
            popt_abs, p0_abs, fit_abs, pcov_abs, initial_abs = abs_transmission_fit(freqs,dataabs,**kwargs)
            kcscaled, kt, f0 = popt_abs
            phys_params.update({'kcscaled':kcscaled,'kt':kt,'f0':f0})
            abs_results.update({'popt':popt_abs,'p0':p0_abs,'pcov':pcov_abs,'fit':fit_abs,'initial':initial_abs})

            #we fit the phase
            popt_arg, p0_arg, fit_arg, pcov_arg, initial_arg = phase_adjust_fit(freqs,dataarg,model='transmission',transmission_params=list(phys_params.values()),**kwargs) #use the above defined fitting algorithm
            delay = popt_arg[0]
            phys_params.update({'delay':delay})
            arg_results.update({'popt':popt_arg,'p0':p0_arg,'pcov':pcov_arg,'fit':fit_arg,'initial':initial_arg})

            results.update({'phys_params':phys_params})
        
        if plots:
            fig, (ax1, ax2) = plt.subplots(nrows=2,figsize=(7, 9.6))
            ax1.plot(freqs,dataabs,c='C0',label='data') #plot the original data
            ax1.plot(freqs,fit_abs,c='C1',label='fit') #plot the fit
            ax1.plot(freqs,initial_abs,'--',c='C2',label='initial') #plot the initial guess
            ax1.set_ylabel('Linear Amplitude')
            ax1.set_xlabel('Frequency (GHz)')
            ax1.legend(loc='best')
            ax2.plot(freqs,dataarg,c='C0',label='data') #plot the original data
            ax2.plot(freqs,fit_arg,c='C1',label='fit') #plot the fit
            ax2.plot(freqs,initial_arg,'--',c='C2',label='initial') #plot the initial guess
            ax2.set_ylabel('Phase (rad)')
            ax2.set_xlabel('Frequency (GHz)')
            ax2.legend(loc='best')
            plt.show()

        if verbose:
            results.update({'abs_results':abs_results,'arg_results':arg_results})
        
        return results