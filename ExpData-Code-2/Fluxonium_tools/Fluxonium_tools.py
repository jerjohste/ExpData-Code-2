#dependencies
import qutip as qt
from ipywidgets import interactive, fixed
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite
from math import factorial
from scipy.interpolate import splrep
from matplotlib.colors import Normalize, LogNorm
from IPython.display import display
import itertools

#helper class to define the fancy color scales

class MidpointNormalize(Normalize):
        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            self.midpoint = midpoint
            Normalize.__init__(self, vmin, vmax, clip)

        def __call__(self, value, clip=None):
            x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
            return np.ma.masked_array(np.interp(value, x, y))

class Fluxonium:
    def __init__(self, ec, el, ej, dim):
        self.Ec = ec
        self.El = el
        self.Ej = ej
        self.Hilbert_dim = dim
        
    #returns a qubit object with the different fluxonium operators
    def H_fluxonium(self, phiext, operators = False):
        a = qt.destroy(self.Hilbert_dim)
        num = 1j*(a.dag()-a)/(np.sqrt(2*np.sqrt(8*self.Ec/self.El)))
        phi = (a+a.dag())*(np.sqrt(np.sqrt(8*self.Ec/self.El)/2))
        cosphi = (phi+phiext*qt.qeye(self.Hilbert_dim)).cosm()
        ham = 4*self.Ec*num**2-self.Ej*cosphi+0.5*self.El*phi**2
        if operators:
            return ham.copy(), num.copy(), phi.copy()
        else:
            return ham.copy()
    
    #calculates the spectrum of the fluxonium qubit for a given flux
    #It calculates the transisitions from all levels in lvls to all levels in lvls, negative frequencies are given as 0
    #Can return various Hamiltonian parameters as well if needed
    #Transitions has dimensions (Nmax+1,Nmax+1) so that you can look up the transition from i to j by looking at the element (i,j). ! (j,i) for j>i will not be filled
    
    def spectrum(self,phiext,lvls=[0,1],full_info = False): 
        N = len(lvls) #this means Nmax is included
        if full_info:
            H, n, phi = self.H_fluxonium(phiext,operators=True)
        else:
            H = self.H_fluxonium(phiext)

        eigenvalues, eigenstates = H.eigenstates()
        eigenvalues = eigenvalues.round(5)
        
        M = np.amax(lvls)+1
        transition_mat = np.zeros((M,M))
        for i in range(M):
            for j in range(M):
                if i in lvls and j in lvls:
                    transition_mat[i,j] = eigenvalues[i]-eigenvalues[j]
        transitions2D = transition_mat.T #np.where(transition_mat<0,0,transition_mat)

        if full_info:
            return {'transitions': np.array(transitions2D), 'hamiltonian': H.copy(),
                    'charge_op': n.copy(), 'phase_op': phi.copy(),
                    'eigenvals': eigenvalues, 'eigenstates': eigenstates}
        else:
            return {'transitions': np.array(transitions2D)}
        
    #defines the fluxonium potential
    def potential(self,phi,phiext):
        return -self.Ej*np.cos(phi+phiext)+0.5*self.El*phi**2
    
    #plots the fluxonium potential
    def potential_plot(self,phiext):
        phi = np.linspace(-1,1,100)
        plt.figure()
        plt.plot(phi,self.potential(2*np.pi*phi,phiext*np.pi*2))
        plt.ylabel('Frequency [GHz]')
        plt.xlabel(r'Flux [/$\Phi_0$]')
        plt.show()
        
    #defines the nth fock states in phi space
    def fock_state(self,phi,n):
        Kinv = 1/np.sqrt(8*self.Ec/self.El)
        return 1/np.sqrt(2**n*factorial(n))*(Kinv/np.pi)**0.25*np.exp(-Kinv*phi**2/2)*hermite(int(n))(np.sqrt(Kinv)*phi)
    
    #defines the wavefunction given by a series of coefficients multiplying fock states
    def wave_function(self,coefs,phi):
        funcvals = [self.fock_state(phi,float(n)) for n in range(len(coefs))]
        return np.dot(coefs,funcvals)
    
    #Return the N first wavefunctions for a given flux bias, and on a range phi_max
    def return_wave_functions(self, N, phiext, phi_max):
        fluxonium_data = self.spectrum(phiext,lvls=np.arange(0,N+1,1),full_info = True)
        eigenstates = fluxonium_data['eigenstates']
        eigenenergies = fluxonium_data['eigenvals']
        Phi = np.linspace(-phi_max, phi_max, 101)
        wavefunctions = np.array([self.wave_function(eigenstates[i].full().flatten(),phi = 2*np.pi*Phi) + eigenenergies[i] for i in range(N)])
        potential_points = self.potential(Phi*2*np.pi,phiext)
        
        return (Phi, wavefunctions, eigenenergies, potential_points)
        

    #plots the fluxonium potential with its N first wavefunctions for a given flux bias
    def show_wave_functions(self,N,phiext):
        fluxonium_data = self.spectrum(phiext,lvls=np.arange(0,N+1,1),full_info = True)
        eigenstates = fluxonium_data['eigenstates']
        eigenenergies = fluxonium_data['eigenvals']
        Phi = np.linspace(-1,1,100)
        wavefunctions = np.array([self.wave_function(eigenstates[i].full().flatten(),phi = 2*np.pi*Phi) + eigenenergies[i] for i in range(N)])
        
        potential_points = self.potential(Phi*2*np.pi,phiext)
        plt.figure()
        plt.plot(Phi,potential_points,c='C0')
        for i in range(N):
            #*(eigenenergies[-1]-eigenenergies[0])*0.1
            plt.plot(Phi,wavefunctions[i].real,c='C'+str(i+1))
            plt.plot(Phi,np.ones(len(Phi))*eigenenergies[i],'--',c='C'+str(i+1))
        plt.ylabel('Frequency [GHz]')
        plt.xlabel(r'Flux [/$\Phi_0$]')
        ymax = np.amax(wavefunctions.flatten())
        ymin = np.amin(potential_points)
        ybounds = (ymin*1.1,ymax*1.1)
        plt.ylim(ybounds)
        plt.show()
        
    #calculates the transitions over a wide flux span
    def transition_spectrum(self,phiext = 2*np.pi*np.linspace(0,1,101),lvls=[0,1,2]):
        transitions = np.array([self.spectrum(flx,lvls)['transitions'] for flx in phiext])
        return transitions
        
    #plots the spectrum of the fluxonium object (with it's own parameters)
    def plot_spectrum(self,phiext = 2*np.pi*np.linspace(0,1,101),lvls = [0,1,2]):
        #calculate transitions
        transitions = self.transition_spectrum(phiext=phiext,lvls=lvls)

        plt.figure('Spectrum')
        for i in lvls[:-1]:
            for j in lvls[1:]:
                if j > i:
                    plt.plot(phiext/(2*np.pi),transitions[:,i,j],c='C'+str(i+j),label=str(i)+'->'+str(j))

        plt.xlabel(r'Flux [/$\Phi_0$]')
        plt.ylabel('Frequency [GHz]')
        plt.legend(loc='lower right')
        plt.show()
        
    #gives the thermal occupation of the first N states of the fluxonium at temperature T and flux phiext
    def thermal_occupation(self,phiext,T,N=5):
        kB = 1.38e-23
        h = 6.626e-34*1e9 #so that we are in GHz
        fluxonium_data = self.spectrum(phiext,lvls=[0,1],full_info = True)
        H = fluxonium_data['hamiltonian']
        rho = (-H/(kB*T/h)).expm()/((-H/(kB*T/h)).expm()).tr()
        eigenstates = fluxonium_data['eigenstates']
        occupations = []
        for i in range(N):
            occupations.append((eigenstates[i]*eigenstates[i].dag()*rho).tr())
        return occupations

    #calculates the dispersive shift of all the transitions in the list lvls and optionnally returns the matrix of the two by two differences
    
    def dispersive_shift(self, lvls, fcav, phiext, g=1, inductive_coupling = False):
        fluxonium_data = self.spectrum(phiext,np.arange(0,self.Hilbert_dim,1),full_info = True)
        
        if inductive_coupling:
            n = fluxonium_data['phase_op']
        else:
            n = fluxonium_data['charge_op']

        eigenstates = fluxonium_data['eigenstates']
        transitions = fluxonium_data['transitions']

        #copy the upper triangle of the matrix to the lower triangle to make sure the order
        #doesn't matter in the next step with a minus sign!! super important
        transitions = -np.transpose(transitions)+transitions

        def sum_element(lvla,lvlb):
            return -abs(n.matrix_element(eigenstates[lvla].dag(),eigenstates[lvlb]))**2*2*transitions[lvla,lvlb] \
                        /(transitions[lvla,lvlb]**2-fcav**2)
                        
        M = np.amax(lvls)+1
        chis = np.zeros(M,dtype=float)
        shifts = np.zeros((M,M),dtype=float)
        for lvl in range(M):
            if lvl in lvls:
                for i in range(self.Hilbert_dim-1):
                    chis[lvl] += sum_element(lvl,i)
        for i in range(M):
            for j in range(M):
                if i in lvls and j in lvls:
                    shifts[i,j] = chis[j]-chis[i]
        #shifts = np.array([[chis[j]-chis[i] for i in range(M) if i in lvls and j in lvls] for j in range(M)])
        return chis*g**2, shifts.T*g**2
    
    #plots the dispersive shift calculated for the flux poiints given by phiext
    def plot_dispersive_shift(self,lvl0,lvl1,fcav,phiext=2*np.pi*np.linspace(0,1,101)):
        from matplotlib.collections import LineCollection
        
        transitions = self.transition_spectrum(phiext,[lvl0,lvl1])[:,lvl0,lvl1]
        shifts = [self.dispersive_shift([lvl0,lvl1],fcav,flx)[1][lvl0,lvl1] for flx in phiext] 
        
        points = np.array([phiext/(2*np.pi),transitions]).T.reshape(-1,1,2) #creates list of point defining the curve
        segments = np.concatenate([points[:-1], points[1:]], axis=1) #creates the segments defining the curve
        
        fig, axs = plt.subplots(2,1,sharex=True)
        ax1, ax2 = axs
        norm = MidpointNormalize(midpoint=0,vmin=np.min(shifts),vmax=-np.min(shifts))
        lc = LineCollection(segments, cmap='RdBu_r', norm=norm)
        lc.set_array(np.array(shifts))
        lc.set_linewidth(2)
        line = ax1.add_collection(lc)
        cbaxes = fig.add_axes([0.79, 0.55, 0.01, 0.25]) 
        cb = plt.colorbar(line, cax = cbaxes)  
        ax1.set_xlim(phiext.min()/(2*np.pi), phiext.max()/(2*np.pi))
        ax1.set_ylim(np.min(transitions)-0.1, np.max(transitions)+0.1)
        ax1.set_ylabel('Frequency [GHz]')

        ax2.plot(phiext/(2*np.pi),shifts)
        ax2.plot(phiext/(2*np.pi),np.zeros(len(phiext)),'r--')
        ax2.set_ylabel('Frequency [GHz]')
        ax2.set_xlabel(r'Flux [/$\Phi_0$]')
        plt.show()
    
    #fancy piece of code to match the measured flux sweep with the fluxonium spectrum
    def find_flux_scale(self,flux_voltage,signal,span_freq,phiext = np.linspace(-1.5,1.5,101)):        
        N = len(phiext)
        resonances = self.transition_spectrum(phiext) #find the transition spectrum
        i,j,k = resonances.shape
        res_with_flux = {}
        x_crossings = {}
        
        #calculate the xs at which the spectrum crosses the given frequency
        for jj in range(j):
            for kk in range(k):
                if kk > jj:
                    key = str(jj)+'->'+str(kk)
                    res_with_flux[key] = resonances[:,jj,kk]
                    x_crossings[key] = find_all_x_for_y(phiext,res_with_flux[key],span_freq)

        fig, axs  = plt.subplots(2,1,sharex = True)
        ax1, ax2 = axs 

        def update_plot(phizero,period):
            def flux_rescale(x):
                return(x*period+phizero)

            ax1.cla()
            ax2.cla()

            ax1.plot(flux_voltage,signal,c='C0') #plot signal
            for i in range(len(res_with_flux.keys())):
                key = list(res_with_flux.keys())[i]
                for j in range(len(x_crossings[key])):
                    ax1.axvline(x = flux_rescale(x_crossings[key][j]),c='C'+str(i+1)) #plot vertical lines for the crossings
                ax2.plot(flux_rescale(phiext),res_with_flux[key],c='C'+str(i+1),label=key) #plot spectrum

            ax1.set_title('Measurement signal')
            ax1.set_xlim((flux_voltage[0],flux_voltage[-1]))
            ax1.set_ylabel('Phase')

            ax2.plot(flux_rescale(phiext),np.ones(N)*span_freq,'r--')
            ax2.legend(loc='best')
            ax2.set_title('Qubit Spectrum')
            ax2.set_ylabel('Frequency [GHz]')
            ax2.set_xlabel('Voltage [V]')

            plt.show()

        interactive_plot = interactive(update_plot, phizero=(-2, 2,0.01), period = (0, 3,0.01)) #make plot interactive to be able to fit
        display(interactive_plot)

    #find the matrix element of given transition mediated by a given operator
    def matrix_element(self,lvls,phiext,operator='charge'): #possible operators: 'charge', 'phase'
        fluxonium_data = self.spectrum(phiext,[0,1],full_info = True)
        eigenstates = fluxonium_data['eigenstates']
        op = fluxonium_data[operator+'_op']
        
        M = np.amax(lvls)+1
        mat_els = np.zeros((M,M),dtype=np.complex)
        for i in range(M):
            for j in range(M):
                if i in lvls and j in lvls:
                    mat_els[i,j] = op.matrix_element(eigenstates[i].dag(),eigenstates[j])
        
        #mat_els = np.array([[op.matrix_element(eigenstates[lvl1].dag(),eigenstates[lvl2]) for lvl2 in lvls] for lvl1 in lvls])
        return mat_els

    #plot the matrix elements for the flux points given by phiext
    def plot_matrix_element(self,lvl0,lvl1,phiext = 2*np.pi*np.linspace(0,1,101),operator='charge'):
        from matplotlib.collections import LineCollection

        transitions = self.transition_spectrum(phiext,[lvl0,lvl1])[:,lvl0,lvl1]
        matrix_elements = [abs(self.matrix_element([lvl0,lvl1],flx,operator)[lvl0,lvl1])**2 for flx in phiext]

        points = np.array([phiext/(2*np.pi),transitions]).T.reshape(-1,1,2) #creates list of point defining the curve
        segments = np.concatenate([points[:-1], points[1:]], axis=1) #creates the segments defining the curve

        fig, axs = plt.subplots(2,1,sharex=True)
        ax1, ax2 = axs
        norm = Normalize()
        lc = LineCollection(segments, cmap='viridis', norm=norm)
        lc.set_array(np.array(matrix_elements))
        lc.set_linewidth(2)
        line = ax1.add_collection(lc)
        cbaxes = fig.add_axes([0.79, 0.55, 0.01, 0.25]) 
        cb = plt.colorbar(line, cax = cbaxes)  
        ax1.set_xlim(phiext.min()/(2*np.pi), phiext.max()/(2*np.pi))
        ax1.set_ylim(np.min(transitions)-0.1, np.max(transitions)+0.1)
        ax1.set_ylabel('Frequency [GHz]')

        ax2.plot(phiext/(2*np.pi),matrix_elements)
        ax2.set_ylabel('Matrix Element Magnitude [a.u.]')
        ax2.set_xlabel(r'Flux [/$\Phi_0$]')        
        plt.show()

    # plots the chose properties for all levels in levels at a given flux point
    # ktot is only necessary if Rabi_charge or Rabi_phase are chosen
    def plot_level_properties(self,phiext,fcav,levels,prop_list,ktot=0):
        """
        prop_list is a list of any of the following: 
        'Dispersive_shift'
        'Rabi_charge'
        'Rabi_phase'
        'Charge_matrix_els'
        'Phase_matrix_els'
        """
        #define dictionary of properties
        keylist = ['Dispersive_shift','Rabi_charge','Rabi_phase','Charge_matrix_els','Phase_matrix_els'] 
        axlabels = [r'$\frac{\chi}{g^2}$',r'$\Omega^n$',r'$\Omega^\varphi$',r'$|<i|n|j>|$',r'$|<i|\varphi|j>|$']
        axdict = dict(zip(keylist,axlabels))

        transitions = list(itertools.combinations(levels,2)) # make pairs of levels
        freqs = self.spectrum(phiext,levels)['transitions'] # obtain transition frequencies
        tups = {} # define empty dict which we will use for the data later in tuple form
        
        if 'Dispersive_shift' in prop_list:
            shifts = self.dispersive_shift(levels,fcav,phiext) # calculate dispersive shifts
            freq_shift_tups = [(trans,freqs[trans],shifts[1][trans]) for trans in transitions] #create tuple
            tups['Dispersive_shift'] = freq_shift_tups #add to dict
            
        if 'Rabi_charge' in prop_list:
            if 'charge' not in locals():
                charge = self.matrix_element(levels,phiext,'charge')
            freq_chargerabi_tups = [(trans,freqs[trans],
                                    np.sqrt(freqs[trans]/(ktot**2+4*(fcav-freqs[trans])**2))*abs(charge[trans])) \
                                for trans in transitions]
            tups['Rabi_charge'] = freq_chargerabi_tups
            
        if 'Charge_matrix_els' in prop_list:
            if 'charge' not in locals():
                charge = self.matrix_element(levels,phiext,'charge')
            freq_charge_tups = [(trans,freqs[trans],abs(charge[trans])) for trans in transitions]
            tups['Charge_matrix_els'] = freq_charge_tups
            
        if 'Rabi_phase' in prop_list:
            if 'phase' not in locals():
                phase = self.matrix_element(levels,phiext,'phase')
            freq_phaserabi_tups = [(trans,freqs[trans],
                                    np.sqrt(freqs[trans]**1.5/(ktot**2+4*(fcav-freqs[trans])**2))*abs(phase[trans])) \
                                for trans in transitions]
            tups['Rabi_phase'] = freq_phaserabi_tups
            
        if 'Phase_matrix_els' in prop_list:
            if 'phase' not in locals():
                phase = self.matrix_element(levels,phiext,'phase')
            freq_phase_tups = [(trans,freqs[trans],abs(phase[trans])) for trans in transitions]
            tups['Phase_matrix_els'] = freq_phase_tups
            
        fignum = len(prop_list)
        fig, axs = plt.subplots(1,fignum,figsize=(8,6),sharey=True) #create figure with as many plots as necessary
        
        for k, prop in enumerate(prop_list):
            if len(prop_list) == 1: # make sure the figure axes are treated correctly 
                ax = axs 
            else:
                ax = axs[k]
            toplot = tups[prop]
            xmin = np.min([ch[2] for ch in toplot])
            xmax = np.max([ch[2] for ch in toplot])
            step = (xmax-xmin)/2
            if prop == 'Dispersive_shift': # the dispersive shift plot is slightly difference than the others because of negative values
                
                for i, el in enumerate(toplot):
                    if el[2] != 0:
                        #ax.set_xscale('symlog')
                        ax.scatter(el[2],el[1],c='C'+str(el[0][0]))
                        ax.hlines(el[1],0,el[2],color='C'+str(el[0][0]))
                        if el[2] < 0:
                            ax.annotate(str(el[0]),(el[2],el[1]),(-35,-3),color='C'+str(el[0][0]),textcoords='offset points')
                        else:
                            ax.annotate(str(el[0]),(el[2],el[1]),(10,-3),color='C'+str(el[0][0]),textcoords='offset points')
                    else:
                        ax.annotate(str(el[0]),(xmin,el[1]),(20,-3),color='C'+str(el[0][0]),textcoords='offset points',arrowprops={'arrowstyle':'->','color':'C'+str(el[0][0])})
                        ax.annotate('<---', (0,el[1]-0.1),color='C'+str(el[0][0]))
                ax.axvline(x=0,linestyle='-',color='black',lw=0.5)
                #ax.xaxis.set_ticks(np.arange(xmin,xmax,step))       
            
            else:
                
                xmin = np.min([ch[2] for ch in toplot if ch[2] > 0])
                for i, el in enumerate(toplot):
                    if el[2] != 0:
                        ax.semilogx(el[2],el[1],'.',c='C'+str(el[0][0]),marker='o')
                        ax.hlines(el[1],0,el[2],color='C'+str(el[0][0]))
                        ax.annotate(str(el[0]),(el[2],el[1]),(10,-3),color='C'+str(el[0][0]),textcoords='offset points')
                    else:
                        ax.semilogx(xmin,el[1],'.',c='C'+str(el[0][0]),marker='')
                        ax.annotate(str(el[0]),(xmin,el[1]),(20,-3),color='C'+str(el[0][0]),textcoords='offset points',arrowprops={'arrowstyle':'->','color':'C'+str(el[0][0])})
                
            ax.axhline(fcav,linestyle='--',color='black')    
            ax.set_xlabel(axdict[prop])
            ax.set_title(prop)
            if k == 0:
                ax.set_ylabel('Transition Frequency [GHz]')
        plt.tight_layout()
        plt.show()

#much simpler way of finding the relevant flux information based on a single flux sweep based on periodicity        
def find_scaling(flux_voltage,signal,plot=True):
    N = len(signal)
    #determine periodicity
    shifted_segment = signal[:N//3]
    reference = signal[N//3:]
    similarity = [np.linalg.norm(shifted_segment-reference[i:N//3+i]) for i in range(N//3)]
    index_period = np.argmin(similarity)+N//3
    period = flux_voltage[index_period]-flux_voltage[0]
    print(index_period)

    #determine symmetry axis
    similarity = [np.linalg.norm(np.flipud(signal[i:i+index_period])-signal[i+index_period:i+2*index_period]) \
                  for i in range(N-2*index_period)]
    index_symmetry = np.argmin(similarity)+index_period
    axis = flux_voltage[index_symmetry]
    
    if plot:
        plt.figure()
        plt.plot(flux_voltage,signal,c='C0')
        plt.axvline(x = axis,c='C1')
        plt.axvline(x = axis+period,c='C1')
        plt.axvline(x = axis-period,c='C1')
        plt.show()
    
    return period, axis

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
    phiext = 2*np.pi*np.linspace(-1,1,101)
    def update_plot(Ec,El,Ej,alpha,beta,**kwargs):
        
        model = Fluxonium(Ec,El,Ej,50)
        #clear figure
        plt.cla()
        
        #plot images
        for i in range(image_num):
            vmin, vmax = np.percentile(spectrum_images[i].flatten(),[1,99])
            plt.pcolormesh(Xs[i],Ys[i],spectrum_images[i],vmin=vmin,vmax=vmax)
            
        #calculate transitions
        transitions = model.transition_spectrum(phiext,lvls=[0,1,2,3])
        
        #plot overlay
        i,j,k = transitions.shape
        count = 0
        colours = ['red','orange','pink']
        for jj in range(j):
            for kk in range(k):
                if kk>jj:
                    plt.plot(alpha*(phiext-np.median(phiext))+beta,transitions[:,jj,kk],#c=colours[count],#
                             label=str(jj)+'->'+str(kk))
                    count += 1
            
        #show plot
        plt.xlabel(r'Voltage [V]')
        plt.ylabel('Energy [GHz]')
        plt.xlim(xlims)
        plt.ylim(ylims)
        plt.legend(loc='lower left')
        plt.show()

    # actual generation of the interactive plot
    plt.figure('Spectrum')
    interactive_plot = interactive(update_plot, Ec=(0.1, 2,0.01), El=(0.1, 2,0.01),Ej=(0.1,5,0.1),alpha = (0,64,0.01),beta=(-32,32,0.01))
    display(interactive_plot)

#finds all the inverse images for a value of y given some data
def find_all_x_for_y(datax,datay,y0,tol=1e-3):
    
    x0s = []
    
    #finds all the points where datay-y changes sign,
    #or equivalently where the data crosses the line y = y0
    lowerindexes = np.where(np.diff(np.sign(datay-y0)))[0] 
    
    if not list(lowerindexes): #returns True if list is empty
        print("There is no corresponding x")
        return x0s
    
    def data_spline(xs):
        return splrep(datax,datay,xs)
    
    #lowerindexes = upperindexes-1
    upperindexes = lowerindexes+1
    error = 1
    
    for i in range(len(upperindexes)):
        error = 1
        xpoints = datax
        lowerindex = lowerindexes[i]
        upperindex = upperindexes[i]
        counts = 0
        while error > tol: #keep going until tol is reached
            if counts > 10:
                return
            xs = np.linspace(xpoints[lowerindex],xpoints[upperindex],11) #x points to calculate through spline
            ys = data_spline(xs)
            diffs = ys-y0
            errors = abs(diffs) #absolute difference between target y and calculated ys
            index_min = np.argmin(errors) #index at which this distance is minimised
            error = errors[index_min]
            
            #now we have to go through a bunch of cases to find the right interval to zoom on
            if index_min == 0:
                lowerindex = 0
                upperindex = 1
            elif index_min == 10:
                lowerindex = 9
                upperindex = 10
            elif errors[index_min-1]<errors[index_min+1]:
                lowerindex = index_min-1
                upperindex = index_min
            else:
                lowerindex = index_min+1
                upperindex = index_min
                
            xpoints = xs
            counts += 1
            
        x0s.append(xs[index_min])
    return x0s