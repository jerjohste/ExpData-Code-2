#dependencies
import qutip as qt
from ipywidgets import interactive, fixed
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite
from math import factorial
from scipy.interpolate import spline
from matplotlib.colors import Normalize, LogNorm
from IPython.display import display

#helper class to define the fancy color scales
class MidpointNormalize(Normalize):
        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            self.midpoint = midpoint
            Normalize.__init__(self, vmin, vmax, clip)

        def __call__(self, value, clip=None):
            # I'm ignoring masked values and all kinds of edge cases to make a
            # simple example...
            x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
            return np.ma.masked_array(np.interp(value, x, y))

class Fluxonium:
    def __init__(self, ec, el, ej, dim):
        self.Ec = ec
        self.El = el
        self.Ej = ej
        self.Hilbert_dim = dim
        
    #returns a qubit object with the different fluxonium operators
    def H_fluxonium(self,phiext,operators = False):
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
    #It calculates the transisitions from the levels in a to all other levels below and including level Nmax
    #Can return various Hamiltonian parameters as well if needed
    #Transitions has dimensions (Nmax+1,Nmax+1) so that you can look up the transition from i to j by looking at the element (i,j). ! (j,i) for j>i will not be filled
    def spectrum(self,phiext,a=[0,1],Nmax = 2,full_info = False): 
        Nmax += 1 #this means Nmax is included
        if full_info:
            H, n, phi = self.H_fluxonium(phiext,operators=True)
        else:
            H = self.H_fluxonium(phiext)

        eigenvalues, eigenstates = H.eigenstates()
        eigenvalues = eigenvalues.round(5)

        transitions2D = np.zeros((Nmax,Nmax))
        for i in range(len(a)):
            start = a[i]
            transitions2D[start,start:] = eigenvalues[start:Nmax]-eigenvalues[start]

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
    
    #defines the wavefunction given by a series of coeficients multiplying fock states
    def wave_function(self,coefs,phi):
        funcvals = [self.fock_state(phi,float(n)) for n in range(len(coefs))]
        return np.dot(coefs,funcvals)
    
    #plots the fluxonium potential with its N first wavefunctions for a given flux bias
    def show_wave_functions(self,N,phiext):
        fluxonium_data = self.spectrum(phiext,a=[0],Nmax = 1,full_info = True)
        eigenstates = fluxonium_data['eigenstates']
        eigenenergies = fluxonium_data['eigenvals']
        Phi = np.linspace(-1,1,100)
        wavefunctions = [self.wave_function(eigenstates[i].full().flatten(),phi = Phi*2*np.pi) + eigenenergies[i] for i in range(N)]
        
        plt.figure()
        plt.plot(Phi,self.potential(Phi*2*np.pi,phiext),c='C0')
        for i in range(N):
            plt.plot(Phi,abs(wavefunctions[i]),c='C'+str(i+1))
            plt.plot(Phi,np.ones(len(Phi))*eigenenergies[i],'--',c='C'+str(i+1))
        plt.ylabel('Frequency [GHz]')
        plt.xlabel(r'Flux [/$\Phi_0$]')
        plt.show()
        
    #calculates the transitions over a wide flux span
    def transition_spectrum(self,phiext = np.linspace(-1,1,100),a = [0,1],N=2):
        transitions = np.array([self.spectrum(flx*2*np.pi,a,N)['transitions'] for flx in phiext])
        return transitions
        
    #plots the spectrum of the fluxonium object (with it's own parameters)
    def plot_spectrum(self,phiext = np.linspace(-1,1,100),a = [0,1],N=2):
        #calculate transitions
        transitions = self.transition_spectrum(phiext=phiext,a=a,N=N)

        plt.figure('Spectrum')
        
        dimension = len(a)+N-a[-1]-1
        for i in range(dimension):
            for j in range(dimension):
                jj = j + 1
                if jj > i:
                    plt.plot(phiext,transitions[:,i,jj],c='C'+str(i+jj),label=str(i)+'->'+str(jj))

        plt.xlabel(r'Flux [/$\Phi_0$]')
        plt.ylabel('Frequency [GHz]')
        plt.legend(loc='best')
        plt.show()
        
    #gives the thermal occupation of the first N states of the fluxonium at temperature T and flux phiext
    def thermal_occupation(self,phiext,T,N=5):
        kB = 1.38e-23
        h = 6.626e-34*1e9 #so that we are in GHz
        fluxonium_data = self.spectrum(phiext*2*np.pi,a=[0],Nmax = 1,full_info = True)
        H = fluxonium_data['hamiltonian']
        rho = (-H/(kB*T/h)).expm()/((-H/(kB*T/h)).expm()).tr()
        eigenstates = fluxonium_data['eigenstates']
        occupations = []
        for i in range(N):
            occupations.append((eigenstates[i]*eigenstates[i].dag()*rho).tr())
        return occupations
            
    #calculates the dispersive shift of the transition between lvl0 and lvl1 depending on the flux point and the cavity resonance
    def dispersive_shift(self,lvl0,lvl1,fcav,phiext,g=1):
        fluxonium_data = self.spectrum(phiext,a=np.arange(0,self.Hilbert_dim,1),Nmax = self.Hilbert_dim-1,full_info = True)
        n = fluxonium_data['charge_op']
        eigenstates = fluxonium_data['eigenstates']
        transitions = fluxonium_data['transitions']

        #copy the upper triangle of the matrix to the lower triangle to make sure the order
        #doesn't matter in the next step with a minus sign!! super important
        transitions = -np.transpose(transitions)+transitions

        def sum_element(lvla,lvlb):
            return -abs(n.matrix_element(eigenstates[lvla].dag(),eigenstates[lvlb]))**2*2*transitions[lvla,lvlb] \
                        /(transitions[lvla,lvlb]**2-fcav**2)

        shift = 0
        chi0 = 0 
        chi1 = 0
        for i in range(self.Hilbert_dim-1):
            chi0 += sum_element(lvl0,i)
            chi1 += sum_element(lvl1,i)
            shift += sum_element(lvl0,i) #we don't need to check if the i==lvl0 because the transition frequency is 0 so it will not contribute to the sum
            shift -= sum_element(lvl1,i)
        return shift*g**2, chi0*g**2, chi1*g**2
    
    #plots the dispersive shift calculated for the flux poiints given by phiext
    def plot_dispersive_shift(self,lvl0,lvl1,fcav,phiext=np.linspace(-1,1,100)):
        from matplotlib.collections import LineCollection
        
        transitions = self.transition_spectrum(phiext,[lvl0],lvl1)[:,lvl0,lvl1]
        shifts = [self.dispersive_shift(lvl0,lvl1,fcav,flx*2*np.pi) for flx in phiext] 
        
        points = np.array([phiext,transitions]).T.reshape(-1,1,2) #creates list of point defining the curve
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
        ax1.set_xlim(phiext.min(), phiext.max())
        ax1.set_ylim(np.min(transitions)-0.1, np.max(transitions)+0.1)
        ax1.set_ylabel('Frequency [GHz]')

        ax2.plot(phiext,shifts)
        ax2.plot(phiext,np.zeros(len(phiext)),'r--')
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
    def matrix_element(self,lvl0,lvl1,phiext,operator='charge'): #possible operators: 'charge', 'phase'
        fluxonium_data = self.spectrum(phiext,a=np.arange(0,self.Hilbert_dim,1),Nmax = self.Hilbert_dim-1,full_info = True)
        eigenstates = fluxonium_data['eigenstates']
        op = fluxonium_data[operator+'_op']
        return op.matrix_element(eigenstates[lvl0].trans(),eigenstates[lvl1])
    
    #plot the matrix elements for the flux points given by phiext
    def plot_matrix_element(self,lvl0,lvl1,phiext = np.linspace(-1,1,100),operator='charge'):
        from matplotlib.collections import LineCollection

        transitions = self.transition_spectrum(phiext,[lvl0],lvl1)[:,lvl0,lvl1]
        matrix_elements = [abs(self.matrix_element(lvl0,lvl1,flx*2*np.pi,operator))**2 for flx in phiext]

        points = np.array([phiext,transitions]).T.reshape(-1,1,2) #creates list of point defining the curve
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
        ax1.set_xlim(phiext.min(), phiext.max())
        ax1.set_ylim(np.min(transitions)-0.1, np.max(transitions)+0.1)
        ax1.set_ylabel('Frequency [GHz]')

        ax2.plot(phiext,matrix_elements)
        ax2.set_ylabel('Matrix Element Magnitude [a.u.]')
        ax2.set_xlabel(r'Flux [/$\Phi_0$]')        
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
    phiext = np.linspace(0,1,51)
    def update_plot(Ec,El,Ej,alpha,beta,**kwargs):
        
        model = Fluxonium(Ec,El,Ej,50)
        #clear figure
        plt.clf()
        
        #plot images
        for i in range(image_num):
            plt.pcolormesh(Xs[i],Ys[i],spectrum_images[i])
            
        #calculate transitions
        transitions = model.transition_spectrum(phiext,a=[0,1],N=2)
        
        #plot overlay
        i,j,k = transitions.shape
        for jj in range(j):
            for kk in range(k):
                if kk>jj:
                    plt.plot(alpha*(phiext-np.median(phiext))+beta,transitions[:,jj,kk],c='C'+str(jj*k+kk),
                             label=str(jj)+'->'+str(kk))
            
        #show plot
        plt.xlabel(r'Voltage [V]')
        plt.ylabel('Energy [GHz]')
        plt.xlim(xlims)
        plt.ylim(ylims)
        plt.legend(loc='lower left')
        plt.show()

    # actual generation of the interactive plot
    plt.figure('Spectrum')
    interactive_plot = interactive(update_plot, Ec=(0, 1,0.01), El=(0, 1,0.01),Ej=(0,3,0.1),alpha = (0,3,0.01),beta=(0,2,0.01))
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
        return spline(datax,datay,xs)
    
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