import shtns
import numpy as np
from scipy.integrate import simpson
from scipy.special import spherical_jn as jn
from scipy.optimize import root
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def make_coords(simpars): #put initial conditions here

    sh = picklable_sht(simpars.lmax,simpars.mmax)  # create sht object with given lmax and mmax (orthonormalized)
    cost = sh.cos_theta
    phi = sh.phi
    r = np.linspace(0,1,simpars.nr)*simpars.rmax
    PH,CO,AR = np.meshgrid(phi,cost,r)

    return PH,CO,AR,sh

def initialize_arrays(simpars,sh):
    nlat = simpars.nlat
    nphi = simpars.nphi
    nr = simpars.nr
    wr = np.zeros((nlat,nphi,nr))
    wth = np.zeros((nlat,nphi,nr))
    wphi = np.zeros((nlat,nphi,nr))
    mr = np.zeros((nlat,nphi,nr))
    mth = np.zeros((nlat,nphi,nr))
    mphi = np.zeros((nlat,nphi,nr))
    
    return wr,wth,wphi,mr,mth,mphi

class SimPars():
    def __init__(self,lmax,mmax,nmax,nr,rmax,dt,nt):
        self.lmax = lmax
        self.mmax = mmax
        self.nmax = nmax
        self.nr = nr
        self.rmax = rmax
        self.dt = dt
        self.nt = nt

class PhysPars():
    def __init__(self,eps,lam,ls,ld):
        self.eps = eps
        self.lam = lam
        self.ls = ls
        self.ld = ld
        

def meq(wr,wth,wph):
    wsq = (wr**2 + wth**2 + wph**2)
    mr_eq = wr*(1-3*wsq/5)
    mth_eq = wth*(1-3*wsq/5)
    mph_eq = wph*(1-3*wsq/5)

    return mr_eq,mth_eq,mph_eq

def meq_1d(w):
    return w*(1-3*w**2/5)

class PickalableSWIG:
    def __setstate__(self, state):
        self.__init__(*state['args'])

    def __getstate__(self):
        return {'args': self.args}

class picklable_sht(shtns.sht,PickalableSWIG):
    def __init__(self, *args):
        self.args = args
        shtns.sht.__init__(self,*args)


def shaperight(zers,sh): #converts the array of zeros to the correct shape, so it can be used to evolve the coefficients straightforwardly
    if zers.shape != (sh.lmax,sh.nmax):
        raise ValueError('Array of zeros does not match expected size.')

    zers_out = np.zeros((sh.nlm,sh.nmax))
    el = sh.l
    for i in range(len(el)):
        if el[i]==0:
            zers_out[i,:] = 0
        else:
            zers_out[i,:] = zers[el[i]-1,:]

    return zers_out

def my_grad(rho,sh,besselzer=None):
    r = sh.r
    nmax = sh.nvals[-1]
    nr = len(r)
    el = sh.l

    oynlm = np.tile(sh.spec_array(),[nmax,1])
    opsinlm = np.tile(sh.spec_array(),[nmax,1])
    olm_r = np.tile(sh.spec_array(),[nr,1])
    olm_r_grad = np.tile(sh.spec_array(),[nr,1])

    o_grad = np.zeros(rho.shape)
    lm_num = olm_r.shape[1]
    
    for i in range(nr): #transform from spatial to spherical coords
        olm_r[i,:] = sh.analys(rho[:,:,i])

    for i in range(lm_num):
        olm_r_grad[:,i] = np.gradient(olm_r[:,i],r,edge_order = 2)

        oynlm[:,i] = func2bessel(r,olm_r[:,i],nmax,el[i],besselzer)
        opsinlm[:,i] = func2bessel(r,olm_r[:,i],nmax,el[i],besselzer)

    for i in range(nr):
        olm_r[i,:] = sh.synth(

    return oynlm,opsinlm

def my_analys(rho,sh,besselzer=None):
    r = sh.r
    nmax = sh.nvals[-1]
    nr = len(r)
    el = sh.l

    onlm = np.tile(sh.spec_array(),[nmax,1])
    olm_r = np.tile(sh.spec_array(),[nr,1])

    lm_num = olm_r.shape[1]
    
    for i in range(nr): #transform from spatial to spherical coords
        olm_r[i,:] = sh.analys(rho[:,:,i])

    for i in range(lm_num):
        onlm[:,i] = func2bessel(r,olm_r[:,i],nmax,el[i],besselzer)

    return onlm
    
def my_spat_to_sh(v_th,v_ph,sh,besselzer = None): #this routine converts from spatial to harmonic representation, including the radial decomposition into bessel functions. Only keeps the curly part.

    #shape of v is (nr,nlat,nphi)
    r = sh.r
    nmax = sh.nvals[-1]
    nr = len(r)
    el = sh.l

    anlm = np.tile(sh.spec_array(),[nmax,1])
    bnlm = np.tile(sh.spec_array(),[nmax,1])
    alm_r = np.tile(sh.spec_array(),[nr,1])
    blm_r = np.tile(sh.spec_array(),[nr,1])
    slm = sh.spec_array()
    tlm = sh.spec_array()
    lm_num = alm_r.shape[1]

    one_spat = sh.spat_array()
    two_spat = sh.spat_array()
    
    for i in range(nr): #transform from spatial to spherical coords
        one_spat[:,:] = v_th[:,:,i]
        two_spat[:,:] = v_ph[:,:,i]
        sh.spat_to_SHsphtor(one_spat,two_spat,slm,tlm)
        alm_r[i,:] = tlm
        blm_r[i,:] = slm

    for i in range(lm_num):
        anlm[:,i] = func2bessel(r,alm_r[:,i],nmax,el[i],besselzer)
        bnlm[:,i] = func2bessel(r,r*blm_r[:,i],nmax,el[i],besselzer)

    return anlm,bnlm

def my_sh_to_spat(anlm,bnlm,sh,besselzer = None):
    
    r = sh.r
    nmax = sh.nvals[-1]
    nr = len(r)
    el = sh.l
    nlat,nphi = sh.set_grid()

    alm_r = np.tile(sh.spec_array(),[nr,1])
    blm_r = np.tile(sh.spec_array(),[nr,1])

    vth = np.zeros((nlat,nphi,nr))
    vph = np.zeros((nlat,nphi,nr))
    vr = np.zeros((nlat,nphi,nr))

    one_spat = sh.spat_array()
    two_spat = sh.spat_array()
    lm_num = alm_r.shape[1]

    for i in range(lm_num):
        alm_r[:,i] = bessel2func(r,anlm[:,i],el[i],besselzer)
        blm_r[:,i] = bessel2func(r,bnlm[:,i],el[i],besselzer)

    scal_blm_r = np.gradient(blm_r,r,edge_order = 2,axis = 0)
    for i in range(nr):
        if i>0:
            sh.SHsphtor_to_spat(blm_r[i,:]/r[i],alm_r[i,:],one_spat,two_spat)
        else:
            sh.SHsphtor_to_spat(blm_r[i,:],alm_r[i,:],one_spat,two_spat) #first point is always 0 anyways

        vth[:,:,i] = one_spat
        vph[:,:,i] = two_spat
        vr[:,:,i] = sh.synth(scal_blm_r[i,:])
    
    return vth,vph,vr

def func2bessel(x,y,nmax,l=1,besselzer = None):
    ncoeffs = np.zeros(nmax,dtype = complex)
    ncoeffs2 = np.zeros(nmax,dtype = complex)
    if l == 0:
        return ncoeffs
    else:
        R = np.max(x)
        x_sc = x/R
        if besselzer is None:
            besselzer = np.loadtxt('zerovals.txt')

        lbesselzer = besselzer[l-1,:nmax]
        xalpha = np.outer(lbesselzer,x)
        xbig = np.tile(x_sc,[nmax,1])
        ybig = np.tile(y,[nmax,1])
        alphabig = np.tile(lbesselzer,[len(x),1])
        integ = simpson(xbig**2*jn(l,xalpha)*ybig,xbig)
        ncoeffs = -2*integ/(jn(l-1,alphabig)*jn(l+1,alphabig))[0,:]

        return ncoeffs

def bessel2func(x,ncoeffs,l=1,besselzer = None):
    nmax = len(ncoeffs)
    if besselzer is None:
        besselzer = np.loadtxt('zerovals.txt')
    lbesselzer = besselzer[l-1,:nmax]

#    y = np.zeros(x.shape,dtype=complex)

    xalpha = np.outer(x,lbesselzer)
     
#    for i in range(nmax):
#        alpha = lbesselzer[i]
#        y += ncoeffs[i]*jn(l,x*alpha)

    y = np.sum(np.tile(ncoeffs,[len(x),1])*jn(l,xalpha),1)

    return y

def genzeros(lmax,mmax):
    zervals = np.zeros((lmax,mmax))
    for i in range(1,lmax+1):
        zervals[i-1,:] = np.array(spherical_jn_zeros(i,mmax,int(1e6)))

    np.savetxt('zerovals.txt',zervals)

def spherical_jn_sensible_grid(n, m, ngrid=100):
    """Returns a grid of x values that should contain the first m zeros, but not too many.
    """
    return np.linspace(n, n + 2*m*(np.pi * (np.log(n)+1)), ngrid)
    

def spherical_jn_zeros(n, m, ngrid=100):
    """Returns first m zeros of spherical bessel function of order n
    """
    # calculate on a sensible grid
    x = spherical_jn_sensible_grid(n, m, ngrid=ngrid)
    y = jn(n, x)
    
    # Find m good initial guesses from where y switches sign
    diffs = np.sign(y)[1:] - np.sign(y)[:-1]
    ind0s = np.where(diffs)[0][:m]  # first m times sign of y changes
    x0s = x[ind0s]
    
    def fn(x):
        return jn(n, x)
    
    return [root(fn, x0).x[0] for x0 in x0s]

def animate_soln(cnhist,dnhist,x,undersamp = 1,fname = 'bla',minmax=[-1,1]):
    nx = len(r) 
    nt = cnhist.shape[1]
    plt.style.use('seaborn-pastel')
    
    fig,(ax1,ax2) = plt.subplots(2,1)
    ax1.set_xlim(np.min(x), np.max(x))
    ax2.set_xlim(np.min(x), np.max(x))

    #if minmax is None: 
    #    ax1.set_ylim(np.min(h), np.max(h))
    #    ax2.set_ylim(np.min(m), np.max(m))
    #else:
    #    ax1.set_ylim(minmax[0], minmax[1])
    #    ax2.set_ylim(minmax[0], minmax[1])

    ax1.set_ylim(minmax[0], minmax[1])
    ax2.set_ylim(minmax[0], minmax[1])

    ax1.set_xlabel('x')
    ax1.set_ylabel(r'$w\tau/3a$')
    ax2.set_xlabel('r')
    ax2.set_ylabel('m')
    
    line, = ax1.plot([], [], lw=3)
    line2, = ax2.plot([], [], lw=3)

    def init():
        line.set_data([],[])
        line2.set_data([],[])
        if eps != 0:
            line3.set_data([],[])
            line4.set_data([],[])
            line5.set_data([],[])
            line6.set_data([],[])
        return line,

    def animate(i):
        y = ex.bessel2func(r,cnhist[:,i*undersamp])
        y2 = ex.bessel2func(r,dnhist[:,i*undersamp])
        line.set_data(x, y)
        line2.set_data(x, y2)
        
        return line,
    
    anim = FuncAnimation(fig, animate,frames=int(nt/undersamp), blit=True)
    
    anim.save('movies/'+fname+'.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
